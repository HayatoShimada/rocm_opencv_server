"""PiCar-X（85pi4 の snoo_trucar）から UDP で届くカメラ画像から猫を検出して返す。

やりとりの仕様は snoo_trucar の docs/PROTOCOL.md。1 データグラム = 1 フレームで、
20 バイトのヘッダー（struct "<4sIQHH"）の後に JPEG が続く。返事は JSON 1 つで、
猫が見つからなくても dets=[] で必ず返す（Pi は返事の有無で通信の状態を見ている）。

検出は RT-DETR v2（COCO の cat。Apache-2.0）を GPU で fp16 で動かす。640x480 で 6 ms ほど
（--compile で 3.7 ms）。入力を 640 より小さくしても速くならず（GPU の呼び出しの手間が大半）、
320 で 7〜10% 見落とすので小さくしない。暗い 60 fps の画像（ゲイン 16 相当のノイズ）でも 640 なら
見つかる割合は変わらず、明るさを補正するとかえって下がったので補正もしない。
snoo らしさは、猫の箱を CLIP で埋め込み、snoo の写真から作った見本（scripts/snoo_gallery.py）との
近さを 0..1 にしたもの。家の猫は snoo だけなので、実際には猫でない物の誤検出を落とす役目が大きい。
見本が無ければ snoo は null で返す（Pi 側は require_snoo=false で猫を追う）。
torch・transformers は重いので、使うときだけ読み込む（uv sync --group clip --group clef-local）。

    uv run python -m app.snoo --host 100.108.168.101
"""

import argparse
import errno
import json
import logging
import os
import socket
import statistics
import struct
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

HEADER = struct.Struct("<4sIQHH")
MAGIC = b"SNO1"
MODEL = os.getenv("SNOO_DET_MODEL", "PekingU/rtdetr_v2_r18vd")
# 返す猫の確からしさの下限（Pi は 0.35 以上を使う。それより少し低いものまで返す）
CAT_MIN = float(os.getenv("SNOO_CAT_MIN", "0.25"))
# 返す箱の数の上限
MAX_DETS = 5
# snoo の見本（scripts/snoo_gallery.py で作る）
GALLERY = os.getenv("SNOO_GALLERY", "data/snoo/gallery.npz")
CLIP_MODEL = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"
# 見本のうち近い順に何枚の類似度を平均するか
TOP_K = 5
# 同じ猫（前のフレームの箱と IoU がこれ以上）の snoo らしさは、
# このフレーム数まで CLIP を回さずに使い回す
REUSE_IOU = 0.5
REUSE_FRAMES = int(os.getenv("SNOO_REUSE_FRAMES", "6"))
# 検出を torch.compile（HIP グラフ）で動かす。6 ms → 3.7 ms。
# 初回の準備に数分かかる（キャッシュがあれば数秒）
COMPILE = os.getenv("SNOO_COMPILE", "0") == "1"

log = logging.getLogger("snoo")


class FrameError(ValueError):
    pass


@dataclass
class Frame:
    seq: int
    t_capture_ns: int
    width: int
    height: int
    jpeg: bytes


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2（幅・高さに対する比）
    cat: float
    snoo: float | None = None


def parse_frame(data: bytes) -> Frame:
    if len(data) < HEADER.size:
        raise FrameError(f"短すぎる（{len(data)} バイト）")
    magic, seq, t_capture_ns, width, height = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise FrameError(f"先頭が {MAGIC!r} ではない（{magic!r}）")
    return Frame(seq, t_capture_ns, width, height, data[HEADER.size :])


def encode_reply(frame: Frame, dets: list[Detection], infer_ms: float) -> bytes:
    reply = {
        "v": 1,
        "seq": frame.seq,
        "t_capture_ns": frame.t_capture_ns,
        "infer_ms": round(infer_ms, 2),
        "dets": [
            {
                "box": [round(v, 4) for v in d.box],
                "cat": round(d.cat, 4),
                "snoo": None if d.snoo is None else round(d.snoo, 4),
            }
            for d in dets
        ],
    }
    return json.dumps(reply, separators=(",", ":")).encode()


def decode_jpeg(jpeg: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FrameError("JPEG を読めない")
    return image


def to_tensor(image: np.ndarray, device: str, dtype):
    """BGR の画像を GPU に載せ、RGB・0..1 の (1, 3, H, W) にする。"""
    import torch

    x = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(device)
    return x.permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255)


class SnooScorer:
    """猫の箱を CLIP で埋め込み、snoo の見本との近さを 0..1 にする。"""

    def __init__(self, device: str, dtype, gallery: str | None = None):
        import open_clip
        import torch

        self.torch = torch
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device
        )
        self.model = model.to(dtype).eval()
        self.size = model.visual.image_size[0]
        self.mean = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=device, dtype=dtype)
        self.std = torch.tensor(open_clip.OPENAI_DATASET_STD, device=device, dtype=dtype)
        self.gallery = None
        if gallery:
            g = np.load(gallery)
            if str(g["model"]) != f"{CLIP_MODEL}/{CLIP_PRETRAINED}":
                raise ValueError(f"{gallery} は別のモデル（{g['model']}）で作られている")
            self.gallery = torch.from_numpy(g["embeddings"]).to(device, dtype)
            self.mid, self.gain = float(g["mid"]), float(g["gain"])

    def embed(self, x, boxes: list[tuple[float, float, float, float]]):
        """画像（to_tensor の形）から箱を切り出して埋め込む（長さ 1 に揃える）。"""
        torch = self.torch
        _, _, h, w = x.shape
        crops = []
        for x1, y1, x2, y2 in boxes:
            # 箱の縁で毛が切れないよう少し広げる。縦横比は崩して正方形にする（見本も同じ）
            pw, ph = (x2 - x1) * 0.05, (y2 - y1) * 0.05
            left, right = int(max(0, x1 - pw) * w), int(min(1, x2 + pw) * w)
            top, bottom = int(max(0, y1 - ph) * h), int(min(1, y2 + ph) * h)
            right, bottom = max(right, left + 1), max(bottom, top + 1)
            crop = x[:, :, top:bottom, left:right]
            crop = torch.nn.functional.interpolate(
                crop, size=(self.size, self.size), mode="bilinear", antialias=True
            )
            crops.append(crop[0])
        batch = (torch.stack(crops).permute(0, 2, 3, 1) - self.mean) / self.std
        with torch.inference_mode():
            e = self.model.encode_image(batch.permute(0, 3, 1, 2)).float()
        return e / e.norm(dim=-1, keepdim=True)

    def similarity(self, embeddings, gallery=None):
        """見本のうち近い TOP_K 枚との類似度の平均。"""
        gallery = self.gallery if gallery is None else gallery
        sims = embeddings @ gallery.float().T
        return sims.topk(min(TOP_K, sims.shape[1]), dim=1).values.mean(dim=1)

    def score(self, x, boxes) -> list[float]:
        if not boxes:
            return []
        s = self.similarity(self.embed(x, boxes))
        return self.torch.sigmoid((s - self.mid) * self.gain).cpu().tolist()


class Stages:
    """1 フレームの区間（受信・デコード・検出・CLIP・送信）ごとの時間を集め、p50・p95 を出す。

    検出と CLIP は結果を CPU に移す所で GPU を待つので、その直後に区切れば GPU の時間も入る。
    """

    ORDER = ("受信", "デコード", "検出", "CLIP", "送信")

    def __init__(self):
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._t: float | None = None

    def start(self) -> None:
        self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        if self._t is None:
            return
        now = time.perf_counter()
        self.samples[name].append((now - self._t) * 1000)
        self._t = now

    def summary(self) -> str:
        """「区間 p50/p95 ms」を ORDER の順に並べた 1 行。集めた値は消す。"""
        parts = []
        for name in self.ORDER:
            v = sorted(self.samples.get(name, []))
            if v:
                p95 = v[max(0, -(-95 * len(v) // 100) - 1)]  # 小さい順に 95% の所（切り上げ）
                parts.append(f"{name} {statistics.median(v):.2f}/{p95:.2f}")
        self.samples.clear()
        return "  ".join(parts)


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class SnooMemory:
    """送り元ごとに、前のフレームの箱と snoo らしさを覚えておく。

    60 fps では隣のフレームの猫はほぼ同じ所にいるので、重なる箱は前の値を使い回し、
    CLIP（1 箱 2 ms）は REUSE_FRAMES フレームに 1 回だけ回す。
    """

    def __init__(self, frames: int = REUSE_FRAMES):
        self.frames = frames
        self.items: list[tuple[tuple, float, int]] = []  # (箱, snoo, 使い回した回数)

    def reuse(self, dets: list[Detection]) -> list[int]:
        """使い回せる箱に snoo を入れ、CLIP にかける必要がある箱の番号を返す。"""
        todo, items = [], []
        for i, d in enumerate(dets):
            best = max(self.items, key=lambda m: _iou(m[0], d.box), default=None)
            if best is not None and best[2] < self.frames and _iou(best[0], d.box) >= REUSE_IOU:
                d.snoo = best[1]
                items.append((d.box, best[1], best[2] + 1))
            else:
                todo.append(i)
        self.items = items
        return todo

    def remember(self, dets: list[Detection], scored: list[int]) -> None:
        self.items += [(dets[i].box, dets[i].snoo, 0) for i in scored]


class CatDetector:
    def __init__(self, model: str = MODEL, cat_min: float = CAT_MIN, gallery: str | None = None,
                 compile: bool = False):  # fmt: skip
        import torch
        from transformers import RTDetrV2ForObjectDetection

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model)
        self.model = self.model.to(self.device, self.dtype).eval()
        labels = self.model.config.id2label
        self.cat_class = next(int(k) for k, v in labels.items() if v == "cat")
        self.size = 640  # RT-DETR は 640x640 で学習している（正規化はしない）
        self.cat_min = cat_min
        self.scorer = SnooScorer(self.device, self.dtype, gallery) if gallery else None
        self.forward = self.model
        if compile and self.device == "cuda":
            # 入力はいつも 640x640 なので dynamic=False。出力はグラフの中の同じ領域に
            # 上書きされるので、次に呼ぶ前に CPU に移す（detect_tensor の中で済ませている）
            self.forward = torch.compile(self.model, mode="reduce-overhead", dynamic=False)
        # 最初の数回は遅い（compile なら準備と HIP グラフの記録）ので先に済ませる
        for _ in range(3 if compile else 1):
            self.detect(np.zeros((480, 640, 3), np.uint8))

    def detect(
        self, image: np.ndarray, memory: SnooMemory | None = None, stages: Stages | None = None
    ) -> list[Detection]:
        """BGR の画像から猫を探す。確からしさの高い順に返す。

        memory を渡すと、前のフレームと重なる箱は snoo らしさを使い回す。
        stages を渡すと、検出（GPU に載せるところから）と CLIP の時間を記録する。
        """
        x = to_tensor(image, self.device, self.dtype)
        dets = self.detect_tensor(x)
        if stages is not None:
            stages.mark("検出")
        if self.scorer:
            todo = memory.reuse(dets) if memory is not None else list(range(len(dets)))
            for i, s in zip(todo, self.scorer.score(x, [dets[i].box for i in todo]), strict=True):
                dets[i].snoo = s
            if memory is not None:
                memory.remember(dets, todo)
            if stages is not None:
                stages.mark("CLIP")
        return dets

    def detect_tensor(self, x) -> list[Detection]:
        torch = self.torch
        from torchvision.ops import nms

        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear")
        with torch.inference_mode():
            out = self.forward(pixel_values=x)
            scores = out.logits[0, :, self.cat_class].sigmoid().float()
            keep = scores >= self.cat_min
            scores = scores[keep]
            cx, cy, w, h = out.pred_boxes[0][keep].float().unbind(-1)
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).clamp(0, 1)
            # RT-DETR は重なった箱をあまり出さないが、念のため同じ猫の箱を 1 つにまとめる
            keep = nms(boxes, scores, 0.7)[:MAX_DETS]
            boxes, scores = boxes[keep].cpu().tolist(), scores[keep].cpu().tolist()
        return [Detection(tuple(b), s) for b, s in zip(boxes, scores, strict=True)]


def _latest_per_sender(sock: socket.socket, first: tuple[bytes, tuple]) -> dict[tuple, bytes]:
    """溜まっているデータグラムを読み切り、送り元ごとに最新の 1 つだけ残す（古いものは捨てる）。"""
    latest = {first[1]: first[0]}
    sock.setblocking(False)
    try:
        while True:
            data, addr = sock.recvfrom(65535)
            latest[addr] = data
    except BlockingIOError:
        pass
    finally:
        sock.setblocking(True)
    return latest


def _bind(host: str, port: int) -> socket.socket:
    """起動直後は tailscale の IP がまだ無いことがあるので、付くまで待ってから受ける。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    while True:
        try:
            sock.bind((host, port))
            return sock
        except OSError as e:
            if e.errno != errno.EADDRNOTAVAIL:
                raise
            log.info("%s がまだ無いので待つ", host)
            time.sleep(5)


def serve(host: str, port: int, detector: CatDetector, stages: Stages | None = None) -> None:
    sock = _bind(host, port)
    log.info("%s:%d/udp で待ち受け（%s）", host, port, detector.device)
    times: list[float] = []
    memories: dict[tuple, SnooMemory] = {}
    stages = Stages() if stages is None else stages
    last_report = time.monotonic()
    while True:
        first = sock.recvfrom(65535)
        t0 = time.perf_counter()
        stages.start()
        latest = _latest_per_sender(sock, first)
        stages.mark("受信")
        for addr, data in latest.items():
            stages.start()  # フレームごとに測り始める（前のフレームの失敗の分を持ち込まない）
            try:
                frame = parse_frame(data)
            except FrameError as e:
                log.warning("%s から: %s", addr, e)
                continue
            try:
                memory = memories.setdefault(addr, SnooMemory())
                image = decode_jpeg(frame.jpeg)
                stages.mark("デコード")
                dets = detector.detect(image, memory, stages)
            except FrameError as e:
                log.warning("%s の seq=%d: %s", addr, frame.seq, e)
                dets = []
                stages.start()  # 読めなかった分は「送信」に入れない
            infer_ms = (time.perf_counter() - t0) * 1000
            sock.sendto(encode_reply(frame, dets, infer_ms), addr)
            stages.mark("送信")
            times.append(infer_ms)
        now = time.monotonic()
        if now - last_report >= 10:
            # 返事が無かった間も区間は消す（次の報告に古い値を混ぜない）
            summary = stages.summary()
            if times:
                log.info(
                    "%.1f fps、受けてから返すまで 中央値 %.1f ms・最大 %.1f ms｜p50/p95 ms: %s",
                    len(times) / (now - last_report),
                    statistics.median(times),
                    max(times),
                    summary,
                )
            times.clear()
            last_report = now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=os.getenv("SNOO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SNOO_PORT", "8710")))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cat-min", type=float, default=CAT_MIN)
    parser.add_argument("--gallery", default=GALLERY, help="snoo の見本。無ければ snoo は null")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=COMPILE,
        help="検出を torch.compile で速くする（環境変数 SNOO_COMPILE=1）",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gallery = args.gallery if os.path.exists(args.gallery) else None
    if gallery is None:
        log.warning("snoo の見本（%s）が無いので、snoo は null で返す", args.gallery)
    serve(args.host, args.port, CatDetector(args.model, args.cat_min, gallery, args.compile))


if __name__ == "__main__":
    main()
