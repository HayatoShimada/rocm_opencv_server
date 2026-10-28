"""PiCar-X（85pi4 の snoo_trucar）から UDP で届くカメラ画像から猫を検出して返す。

やりとりの仕様は snoo_trucar の docs/PROTOCOL.md。1 データグラム = 1 フレームで、
20 バイトのヘッダー（struct "<4sIQHH"）の後に JPEG が続く。返事は JSON 1 つで、
猫が見つからなくても dets=[] で必ず返す（Pi は返事の有無で通信の状態を見ている）。

検出は RT-DETR v2（COCO の cat。Apache-2.0）を GPU で fp16 で動かす。640x480 で 6 ms ほど。
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


class CatDetector:
    def __init__(self, model: str = MODEL, cat_min: float = CAT_MIN, gallery: str | None = None):
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
        self.detect(np.zeros((480, 640, 3), np.uint8))  # 最初の 1 回は遅いので先に済ませる

    def detect(self, image: np.ndarray) -> list[Detection]:
        """BGR の画像から猫を探す。確からしさの高い順に返す。"""
        x = to_tensor(image, self.device, self.dtype)
        dets = self.detect_tensor(x)
        if self.scorer:
            for d, s in zip(dets, self.scorer.score(x, [d.box for d in dets]), strict=True):
                d.snoo = s
        return dets

    def detect_tensor(self, x) -> list[Detection]:
        torch = self.torch
        from torchvision.ops import nms

        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear")
        with torch.inference_mode():
            out = self.model(pixel_values=x)
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


def serve(host: str, port: int, detector: CatDetector) -> None:
    sock = _bind(host, port)
    log.info("%s:%d/udp で待ち受け（%s）", host, port, detector.device)
    times: list[float] = []
    last_report = time.monotonic()
    while True:
        first = sock.recvfrom(65535)
        t0 = time.perf_counter()
        latest = _latest_per_sender(sock, first)
        for addr, data in latest.items():
            try:
                frame = parse_frame(data)
            except FrameError as e:
                log.warning("%s から: %s", addr, e)
                continue
            try:
                dets = detector.detect(decode_jpeg(frame.jpeg))
            except FrameError as e:
                log.warning("%s の seq=%d: %s", addr, frame.seq, e)
                dets = []
            infer_ms = (time.perf_counter() - t0) * 1000
            sock.sendto(encode_reply(frame, dets, infer_ms), addr)
            times.append(infer_ms)
        now = time.monotonic()
        if now - last_report >= 10:
            if times:
                log.info(
                    "%.1f fps、受けてから返すまで 中央値 %.1f ms・最大 %.1f ms",
                    len(times) / (now - last_report),
                    statistics.median(times),
                    max(times),
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gallery = args.gallery if os.path.exists(args.gallery) else None
    if gallery is None:
        log.warning("snoo の見本（%s）が無いので、snoo は null で返す", args.gallery)
    serve(args.host, args.port, CatDetector(args.model, args.cat_min, gallery))


if __name__ == "__main__":
    main()
