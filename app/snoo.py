"""PiCar-X（85pi4 の snoo_trucar）から UDP で届くカメラ画像から猫を検出して返す。

やりとりの仕様は snoo_trucar の docs/PROTOCOL.md。1 データグラム = 1 フレームで、
20 バイトのヘッダー（struct "<4sIQHH"）の後に JPEG が続く。版 2（"SNO2"、37 バイト）は
撮った時刻の指令（ハンドル・速さ・パン・チルト・走っているか）も載せる。返事は JSON 1 つで、
猫が見つからなくても dets=[] で必ず返す（Pi は返事の有無で通信の状態を見ている）。

検出は RT-DETR v2（COCO の cat。Apache-2.0）を GPU で fp16 で動かす。640x480 で 6 ms ほど
（--compile で 3.7 ms）。入力を 640 より小さくしても速くならず（GPU の呼び出しの手間が大半）、
320 で 7〜10% 見落とすので小さくしない。暗い 60 fps の画像（ゲイン 16 相当のノイズ）でも 640 なら
見つかる割合は変わらず、明るさを補正するとかえって下がったので補正もしない。
snoo らしさは、猫の箱を CLIP で埋め込み、snoo の写真から作った見本（scripts/snoo_gallery.py）との
近さを 0..1 にしたもの。家の猫は snoo だけなので、実際には猫でない物の誤検出を落とす役目が大きい。
見本が無ければ snoo は null で返す（Pi 側は require_snoo=false で猫を追う）。
torch・transformers は重いので、使うときだけ読み込む（uv sync --group clip --group clef-local）。

家の中では、Pi は tailscale を通さず LAN の IP に送る（tailscale 経由の ping は平均 6.7 ms、
LAN 宛ては 3.7 ms）。そのときは 0.0.0.0 で受け、認証の無い UDP なので --allow に書いた送り元
（Pi の LAN と tailscale の IP）以外は捨てる。

--record-dir を渡すと、猫が写っている間のフレームを学習用に保存する（app/snoo_record.py）。
--live-dir を渡すと、送り元ごとに最新のフレームと返事を 1 秒に 1 回ほど書く（app/snoo_live.py。
地図のサーバーの /twin で見る）。
--slam-tap-dir を渡すと、走っている間のフレームを 4 枚/秒ほど SLAM に渡す（app/snoo_slam_tap.py。
ホストの app/snoo_slam.py が読む）。

走っている間（SNO2 で flags に MOVING）は、前があと何 cm 空いているか（clear_cm、
app/clearance.py）も返事の一番上の段に載せる。超音波のセンサーが壊れていて、走っている間は
前が見えないため。
検出を遅くしないよう、返事を送った後に別のスレッドで最新のフレームだけを見積もり、次の返事に載せる
（ClearanceWorker）。走っていないフレームと SNO1 には clear の欄を載せない。

    uv run python -m app.snoo --host 0.0.0.0 --allow 192.168.0.188,100.73.14.8
"""

import argparse
import errno
import ipaddress
import json
import logging
import math
import os
import socket
import statistics
import struct
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from app.snoo_live import LIVE_DIR, LiveWriter
from app.snoo_record import RECORD_DIR, Recorder
from app.snoo_slam_tap import SLAM_TAP_DIR, SlamTap

HEADER = struct.Struct("<4sIQHH")
MAGIC = b"SNO1"
# 版 2：撮った時刻に Pi が書いていた指令（ハンドル・速さ・パン・チルト）と、走っているかを足す。
# 走っている間の前の空き（clear_cm）を、これから通る弧の上で測るために要る。SNO1 もそのまま読む
HEADER_V2 = struct.Struct("<4sIQHHffffB")
MAGIC_V2 = b"SNO2"
MOVING, REVERSING = 1, 2  # flags のビット
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
# 受ける送り元（IP か CIDR をカンマで区切る）。空ならどこからでも受ける
ALLOW = os.getenv("SNOO_ALLOW", "")
# フレームを処理している間、1 秒ごとに触る（更新時刻を新しくする）ファイル。ホストの
# deploy/snoo-gpu-perf.sh がこれを見て、検出が動いている間だけ GPU の時計を上げる
ACTIVE_FILE = os.getenv("SNOO_ACTIVE_FILE", "")
# 走っている間の前の空き（clear_cm）を見積もるか
CLEARANCE = os.getenv("SNOO_CLEARANCE", "1") == "1"
# clear_cm の元にしたフレームが、返事のフレームよりこれだけ前に撮ったものなら古いとして null にする
CLEAR_MAX_AGE_MS = 150.0
# 確からしさがこれより低い clear_cm は null にする（地図の画像では、床が見えていれば 0.95 以上）
CLEAR_MIN_CONF = float(os.getenv("SNOO_CLEAR_MIN_CONF", "0.5"))
# 見積もりを始める間の最短（ms）。15 cm/s なら 45 ms（60 fps の 3 フレームに 1 回、20 回/秒）で
# 0.7 cm しか進まないので、それで足りる。毎フレーム回しても検出の p50 は 0.3 ms ほどしか
# 遅れなかったが、GPU は検出のために空けておく
CLEAR_INTERVAL_MS = float(os.getenv("SNOO_CLEAR_INTERVAL_MS", "45"))
# 首（パン・チルト）を動かしてから・走り出してから、この間は clear_cm を見積もらない。
# サーボが止まるまで（0.4 s ほど）は、撮った時刻の指令と実際の向きが違い、床の位置を見誤る。
# 1 回目の実走の 2 回目の探索では、撮り終えて走り出した直後に 18 cm（超音波は 67 cm）と出て止まった
CLEAR_SETTLE_MS = float(os.getenv("SNOO_CLEAR_SETTLE_MS", "400"))

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
    # 版 2 だけ（撮った時刻の指令。サーボやモーターの実際の値ではない）
    steer_deg: float | None = None
    speed: float | None = None
    pan_deg: float | None = None
    tilt_deg: float | None = None
    flags: int = 0

    @property
    def moving(self) -> bool:
        return bool(self.flags & MOVING)


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2（幅・高さに対する比）
    cat: float
    snoo: float | None = None


def parse_frame(data: bytes) -> Frame:
    magic = data[:4]
    if magic == MAGIC_V2:
        if len(data) < HEADER_V2.size:
            raise FrameError(f"短すぎる（{len(data)} バイト）")
        _, seq, t_ns, width, height, steer, speed, pan, tilt, flags = HEADER_V2.unpack_from(data)
        return Frame(
            seq, t_ns, width, height, data[HEADER_V2.size :], steer, speed, pan, tilt, flags
        )
    if magic != MAGIC:
        raise FrameError(f"先頭が {MAGIC!r} でも {MAGIC_V2!r} でもない（{magic!r}）")
    if len(data) < HEADER.size:
        raise FrameError(f"短すぎる（{len(data)} バイト）")
    _, seq, t_capture_ns, width, height = HEADER.unpack_from(data)
    return Frame(seq, t_capture_ns, width, height, data[HEADER.size :])


def encode_reply(
    frame: Frame, dets: list[Detection], infer_ms: float, clear: dict | None = None
) -> bytes:
    """clear は ClearanceWorker.fields の値（None なら clear の欄は載せない）。"""
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
    if clear is not None:
        reply.update(clear)
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

    ORDER = ("受信", "デコード", "検出", "CLIP", "送信", "clear")

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

    def add(self, name: str, ms: float) -> None:
        """別のスレッドで測った時間を足す（clear は返事の後に別のスレッドで回す）。"""
        self.samples[name].append(ms)

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


# 走っているが、使える clear_cm が無いときの欄（Pi は null が 0.3 秒続くと止まる）
NO_CLEAR = {"clear_cm": None, "clear_conf": None, "clear_seq": None, "clear_blind_cm": None}


@dataclass
class _ClearResult:
    seq: int
    t_capture_ns: int
    clearance: object  # app.clearance.Clearance


# 首の向きがこれだけ変わったら、サーボが止まるまで clear_cm を待つ
SETTLE_TOL_DEG = 1.5


def same_pose(a: tuple, b: tuple) -> bool:
    """(走っているか, パン, チルト) が同じとみなせるか。"""
    return a[0] == b[0] and abs(a[1] - b[1]) < SETTLE_TOL_DEG and abs(a[2] - b[2]) < SETTLE_TOL_DEG


class ClearanceWorker:
    """走っている間の前の空き（clear_cm）を、検出とは別のスレッドで見積もる。

    検出は 60 fps・1 フレーム 5 ms 足らずで回っているので、その間に入れると遅くなる。
    そこで返事を送った後に submit でフレームを渡し、別のスレッドで見積もって、次の返事に載せる。
    GPU は 1 つを分け合うので、同時には 1 つしか回さない。まだ回っている間に来たフレームは
    溜めずに捨てる（終わった後に来た最新のフレームを使う）。前に始めてから interval_ms 経って
    いないフレームも見積もらない。

    返事の欄（fields）の決まり（Pi と決めたもの）
    - 欄が無い = サーバーは clear を見ていない（Pi は止まらない）。走っていないフレーム・SNO1・
      モデルがまだ無い（読み込み中か失敗）とき。それと、通り道がそもそも画面に入らないとき
      （clear_skip に理由を書く）。舵を約 10° 以上切ると、帯が画角 54° からはみ出す。下がって
      いるときは、カメラが前を向いているので進む先が見えない。どちらも null にすると Pi は
      0.3 秒で止まり、曲がるたび・切り返すたびに止まってしまう。その間は超音波・画像が
      動かないときの判定・床が見えた所だけを通る地図の道で守る。
    - null = 見ているはずなのに、このフレームには使える値が無い（Pi は走っている間に null が
      0.3 秒続くと止まる）。元のフレームが CLEAR_MAX_AGE_MS より古い・まだ無い・床に合わない・
      確からしさが CLEAR_MIN_CONF より低いとき。

    build は見積もりの道具（app.clearance.Estimator か、テストでは同じ形の物）を作る関数。
    compile した HIP グラフは作ったスレッドでしか使い回せないので、見積もるスレッドの中で作る。
    """

    def __init__(
        self,
        build,
        max_age_ms: float = CLEAR_MAX_AGE_MS,
        min_conf: float = CLEAR_MIN_CONF,
        interval_ms: float = CLEAR_INTERVAL_MS,
        settle_ms: float = CLEAR_SETTLE_MS,
    ):
        self.max_age_ns = int(max_age_ms * 1e6)
        self.min_conf = min_conf
        self.interval_ns = int(interval_ms * 1e6)
        self._last_start_ns = -(1 << 62)
        self.estimator = None
        self.computed = self.skipped = 0
        self._ready = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._busy = False
        self._job = None
        self._results: dict[tuple, _ClearResult] = {}
        self.settle_ns = int(settle_ms * 1e6)
        # 送り元ごとの、前のフレームの (走っているか, パン, チルト) と、それが変わった時刻
        self._pose: dict[tuple, tuple[tuple, int]] = {}
        threading.Thread(target=self._run, args=(build,), daemon=True, name="snoo-clear").start()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """モデルの読み込み（と compile の準備）が終わるまで待つ。失敗したら False。"""
        self._ready.wait(timeout)
        return self.estimator is not None

    def settling(self, addr: tuple, frame: Frame) -> bool:
        """首を動かした・走り出した直後で、サーボがまだ止まっていないか。フレームごとに呼ぶ。"""
        key = (frame.moving, frame.pan_deg or 0.0, frame.tilt_deg or 0.0)
        last = self._pose.get(addr)
        # 首の指令は滑らかに動くので、少しの変化（SETTLE_TOL_DEG 未満）では待ち直さない。
        # 0.1° で比べていたら、3 回目の実走では走っている間の 1454 コマのうち 574 コマが
        # settling になり、clear_cm がほとんど出なかった
        if last is None or frame.t_capture_ns < last[1] or not same_pose(last[0], key):
            self._pose[addr] = (key, frame.t_capture_ns)
            return self.settle_ns > 0
        return frame.t_capture_ns - last[1] < self.settle_ns

    def fields(self, addr: tuple, frame: Frame) -> dict | None:
        """このフレームの返事に載せる clear の欄。None なら載せない。"""
        settling = self.settling(addr, frame)
        if not frame.moving or self.estimator is None:
            return None
        if frame.flags & REVERSING:
            return {"clear_skip": "reversing"}
        if settling:
            return {"clear_skip": "settling"}
        with self._lock:
            r = self._results.get(addr)
        # サーボが動いていた間のフレームから見積もった値は使わない
        changed = self._pose[addr][1]
        if r is not None and r.t_capture_ns - changed < self.settle_ns:
            r = None
        # 古さは Pi の撮った時刻どうしで比べる（届くまでの揺れが入らない）。
        # 負なら Pi が起動し直して時刻が戻った
        if r is None or not 0 <= frame.t_capture_ns - r.t_capture_ns <= self.max_age_ns:
            return dict(NO_CLEAR)
        c = r.clearance
        if getattr(c, "out_of_view", False):
            return {"clear_skip": "out_of_view"}
        ok = c.clear_cm is not None and c.confidence >= self.min_conf
        return {
            "clear_cm": round(c.clear_cm, 1) if ok else None,
            "clear_conf": round(c.confidence, 2),
            "clear_seq": r.seq,
            "clear_blind_cm": round(c.blind_cm, 1) if math.isfinite(c.blind_cm) else None,
        }

    def submit(self, addr: tuple, frame: Frame, image: np.ndarray, stages=None) -> bool:
        """返事を送った後に呼ぶ。前のがまだ回っている・間が短ければ捨てて False。"""
        if not frame.moving or frame.flags & REVERSING or self.estimator is None:
            return False
        if self.settling(addr, frame):
            return False  # サーボが止まるまでは見積もらない
        now = time.monotonic_ns()
        with self._lock:
            if now - self._last_start_ns < self.interval_ns:
                return False
            if self._busy:
                self.skipped += 1
                return False
            self._busy = True
            self._last_start_ns = now
            self._job = (addr, frame, image, stages)
        self._wake.set()
        return True

    def report(self) -> str:
        """前に呼んでからの回数（ログ用）。数えた値は消す。"""
        with self._lock:
            text = f"clear {self.computed} 回（回っていて飛ばした {self.skipped}）"
            self.computed = self.skipped = 0
        return text

    def _run(self, build) -> None:
        try:
            self.estimator = build()
            log.info("clear_cm を見積もる準備ができた")
        except Exception:
            log.exception("clear_cm を見積もる道具を作れないので、clear の欄は載せない")
        finally:
            self._ready.set()
        if self.estimator is None:
            return
        failed = False
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                addr, frame, image, stages = self._job
                self._job = None
            t0 = time.perf_counter()
            try:
                c = self.estimator.estimate(image, frame.tilt_deg, frame.pan_deg, frame.steer_deg)
                failed = False
            except Exception:
                if not failed:  # 続けて失敗しても 1 回だけ書く
                    log.exception("clear_cm を見積もれない（null で返す）")
                failed = True
                c = None
            if stages is not None:
                stages.add("clear", (time.perf_counter() - t0) * 1000)
            with self._lock:
                if c is None:  # 前の値も使わない（null になる）
                    self._results.pop(addr, None)
                else:
                    self._results[addr] = _ClearResult(frame.seq, frame.t_capture_ns, c)
                self.computed += 1
                self._busy = False


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


def parse_allow(text: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """「192.168.0.188,100.73.14.8」のような並びを、送り元を調べるための範囲にする。"""
    return [ipaddress.ip_network(s.strip(), strict=False) for s in text.split(",") if s.strip()]


def _allowed(addr: tuple, allow: list) -> bool:
    return not allow or any(ipaddress.ip_address(addr[0]) in net for net in allow)


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


def _touch(path: str) -> None:
    """path の更新時刻を今にする（書けなくても検出は止めない）。"""
    try:
        with open(path, "a"):
            os.utime(path)
    except OSError as e:
        log.warning("%s を更新できない: %s", path, e)


def serve(
    host: str,
    port: int,
    detector: CatDetector,
    stages: Stages | None = None,
    allow: list | None = None,
    active_file: str = "",
    recorder: Recorder | None = None,
    clearance: ClearanceWorker | None = None,
    live: LiveWriter | None = None,
    slam_tap: SlamTap | None = None,
) -> None:
    sock = _bind(host, port)
    allow = allow or []
    log.info(
        "%s:%d/udp で待ち受け（%s、受ける送り元: %s）",
        host,
        port,
        detector.device,
        ", ".join(map(str, allow)) or "すべて",
    )
    refused: set[str] = set()
    last_touch = 0.0
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
            if not _allowed(addr, allow):
                if addr[0] not in refused:  # 同じ送り元は 1 回だけ書く
                    refused.add(addr[0])
                    log.warning("%s は受ける送り元に無いので捨てる", addr[0])
                continue
            stages.start()  # フレームごとに測り始める（前のフレームの失敗の分を持ち込まない）
            try:
                frame = parse_frame(data)
            except FrameError as e:
                log.warning("%s から: %s", addr, e)
                continue
            image = None
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
            # clear の欄は前に見積もった値を読むだけ（見積もりは返事の後）
            clear = clearance.fields(addr, frame) if clearance is not None else None
            sock.sendto(encode_reply(frame, dets, infer_ms, clear), addr)
            stages.mark("送信")
            times.append(infer_ms)
            if clearance is not None and image is not None:
                clearance.submit(addr, frame, image, stages)
            if recorder is not None:  # 返事の後に渡す（書き込みは別のスレッド）
                recorder.offer(addr, frame, dets)
            if live is not None:  # これも返事の後。書くのは 1 秒に 1 回ほど、別のスレッドで
                live.offer(addr, frame, dets, infer_ms, clear)
            if slam_tap is not None:  # これも返事の後。走っている間だけ、別のスレッドで書く
                slam_tap.offer(addr, frame, dets)
        now = time.monotonic()
        if active_file and times and now - last_touch >= 1.0:
            _touch(active_file)
            last_touch = now
        if now - last_report >= 10:
            # 返事が無かった間も区間は消す（次の報告に古い値を混ぜない）
            summary = stages.summary()
            clear = f"｜{clearance.report()}" if clearance is not None else ""
            if times:
                log.info(
                    "%.1f fps、受けてから返すまで 中央値 %.1f ms・最大 %.1f ms｜p50/p95 ms: %s%s",
                    len(times) / (now - last_report),
                    statistics.median(times),
                    max(times),
                    summary,
                    clear,
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
    parser.add_argument(
        "--allow",
        default=ALLOW,
        help="受ける送り元の IP か CIDR（カンマ区切り。環境変数 SNOO_ALLOW）。空ならすべて",
    )
    parser.add_argument(
        "--active-file",
        default=ACTIVE_FILE,
        help="フレームを処理している間、1 秒ごとに触るファイル（環境変数 SNOO_ACTIVE_FILE）",
    )
    parser.add_argument(
        "--record-dir",
        default=RECORD_DIR,
        help="猫が写っている間のフレームを保存する所（環境変数 SNOO_RECORD_DIR）。空なら保存しない",
    )
    parser.add_argument(
        "--live-dir",
        default=LIVE_DIR,
        help="最新のフレームと返事を 1 秒に 1 回ほど書く所（環境変数 SNOO_LIVE_DIR）。空は書かない",
    )
    parser.add_argument(
        "--slam-tap-dir",
        default=SLAM_TAP_DIR,
        help="走っている間のフレームを SLAM に渡す所（環境変数 SNOO_SLAM_TAP_DIR）。空は書かない",
    )
    parser.add_argument(
        "--clearance",
        action=argparse.BooleanOptionalAction,
        default=CLEARANCE,
        help="走っている間の前の空き（clear_cm）を見積もる（環境変数 SNOO_CLEARANCE、既定は 1）",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gallery = args.gallery if os.path.exists(args.gallery) else None
    if gallery is None:
        log.warning("snoo の見本（%s）が無いので、snoo は null で返す", args.gallery)
    detector = CatDetector(args.model, args.cat_min, gallery, args.compile)
    clearance = None
    if args.clearance:
        from app.clearance import CAM_HEIGHT_CM, TILT_GAIN, TILT_OFFSET_DEG, Estimator

        log.info(
            "clear_cm を見積もる（カメラの高さ %.1f cm、tilt = 指令 × %.2f %+.1f°）",
            CAM_HEIGHT_CM,
            TILT_GAIN,
            TILT_OFFSET_DEG,
        )
        clearance = ClearanceWorker(lambda: Estimator(compile=args.compile))
        # 検出の compile と同時に準備しないよう、終わるまで待ってから受け始める
        clearance.wait_ready()
    serve(
        args.host,
        args.port,
        detector,
        allow=parse_allow(args.allow),
        active_file=args.active_file,
        recorder=Recorder(args.record_dir) if args.record_dir else None,
        clearance=clearance,
        live=LiveWriter(args.live_dir) if args.live_dir else None,
        slam_tap=SlamTap(args.slam_tap_dir) if args.slam_tap_dir else None,
    )


if __name__ == "__main__":
    main()
