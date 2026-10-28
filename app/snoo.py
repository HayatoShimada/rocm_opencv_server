"""PiCar-X（85pi4 の snoo_trucar）から UDP で届くカメラ画像から猫を検出して返す。

やりとりの仕様は snoo_trucar の docs/PROTOCOL.md。1 データグラム = 1 フレームで、
20 バイトのヘッダー（struct "<4sIQHH"）の後に JPEG が続く。返事は JSON 1 つで、
猫が見つからなくても dets=[] で必ず返す（Pi は返事の有無で通信の状態を見ている）。

検出は RT-DETR v2（COCO の cat。Apache-2.0）を GPU で fp16 で動かす。640x480 で 6 ms ほど。
snoo かどうかの判定はまだ無いので、snoo は null で返す（Pi 側は require_snoo=false で猫を追う）。
torch・transformers は重いので、使うときだけ読み込む（uv sync --group clip --group clef-local）。

    uv run python -m app.snoo --host 100.108.168.101
"""

import argparse
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


class CatDetector:
    def __init__(self, model: str = MODEL, cat_min: float = CAT_MIN):
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
        self.detect(np.zeros((480, 640, 3), np.uint8))  # 最初の 1 回は遅いので先に済ませる

    def detect(self, image: np.ndarray) -> list[Detection]:
        """BGR の画像から猫を探す。確からしさの高い順に返す。"""
        torch = self.torch
        from torchvision.ops import nms

        x = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255)
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


def serve(host: str, port: int, detector: CatDetector) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((host, port))
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(args.host, args.port, CatDetector(args.model, args.cat_min))


if __name__ == "__main__":
    main()
