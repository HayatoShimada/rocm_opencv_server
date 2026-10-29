"""snoo の写真から、snoo らしさの判定に使う見本を作る（app/snoo.py の SnooScorer）。

  uv run --group clip --group clef-local python -m scripts.snoo_gallery
写真（data/snoo_raw/*.jpg。snoo だけが写っている）ごとに、いちばん確かな猫の箱を見本にする。
猫でない物と比べるため、同じ写真の猫の箱にかからない所をいくつか切り出し、
見本との近さを 0..1 にする境目（mid）と傾き（gain）を決める。
猫の箱から外れた snoo の顔や耳が入らないよう、猫の箱に少しでもかかる所は使わない。
追いかけるときは後ろ姿や体の一部しか見えないことが多いので、境目は猫でない所のすぐ上に置き、
体の一部（猫の箱の上下左右の半分）がどれだけ snoo とみなされるかも出す。
見本の近さは、自分の写真を除いて測る（自分自身とは必ず近いので）。
結果は data/snoo/gallery.npz に保存する。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.snoo import CLIP_MODEL, CLIP_PRETRAINED, CatDetector, SnooScorer, to_tensor

SOURCE = Path("data/snoo_raw")
OUTPUT = Path("data/snoo/gallery.npz")
# 見本にする猫の箱の確からしさの下限
CAT_MIN = 0.6
# 写真ごとに切り出す、猫でない所の数
NEGATIVES_PER_IMAGE = 6


def _overlaps(box, cat) -> bool:
    """box が猫の箱（毛がはみ出すので 15% 広げる）に少しでもかかるか。"""
    pw, ph = (cat[2] - cat[0]) * 0.15, (cat[3] - cat[1]) * 0.15
    w = min(box[2], cat[2] + pw) - max(box[0], cat[0] - pw)
    h = min(box[3], cat[3] + ph) - max(box[1], cat[1] - ph)
    return w > 0 and h > 0


def _background_boxes(cats, rng, n: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    for _ in range(n * 20):
        w, h = rng.uniform(0.15, 0.6, 2)
        x1, y1 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
        box = (x1, y1, x1 + w, y1 + h)
        if not any(_overlaps(box, c) for c in cats):
            boxes.append(box)
            if len(boxes) == n:
                break
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    detector = CatDetector(cat_min=0.25)
    scorer = SnooScorer(detector.device, detector.dtype)
    rng = np.random.default_rng(0)
    positives, parts, part_of, negatives, files = [], [], [], [], []
    for path in sorted(args.source.glob("*.jpg")):
        image = cv2.imread(str(path))
        x = to_tensor(image, detector.device, detector.dtype)
        dets = detector.detect_tensor(x)
        if not dets or dets[0].cat < CAT_MIN:
            print(f"{path.name}: 確かな猫が無いので使わない")
            continue
        positives.append(scorer.embed(x, [dets[0].box])[0])
        x1, y1, x2, y2 = dets[0].box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        halves = [(x1, y1, cx, y2), (cx, y1, x2, y2), (x1, y1, x2, cy), (x1, cy, x2, y2)]
        parts.extend(scorer.embed(x, halves))
        part_of.extend([len(files)] * len(halves))
        files.append(path.name)
        background = _background_boxes([d.box for d in dets], rng, NEGATIVES_PER_IMAGE)
        if background:
            negatives.extend(scorer.embed(x, background))

    torch = scorer.torch
    gallery = torch.stack(positives)
    # 見本どうしの近さは自分の写真を除いて測る（同じ写真との類似度を -1 にして topk から外す）
    sims = gallery @ gallery.T
    sims.fill_diagonal_(-1)
    pos = sims.topk(5, dim=1).values.mean(dim=1).cpu().numpy()
    sims = torch.stack(parts) @ gallery.T
    sims[torch.arange(len(parts)), torch.tensor(part_of)] = -1
    part = sims.topk(5, dim=1).values.mean(dim=1).cpu().numpy()
    neg = scorer.similarity(torch.stack(negatives), gallery).cpu().numpy()

    pos_low, neg_high = np.percentile(pos, 1), np.percentile(neg, 99.5)
    print(f"見本 {len(pos)} 枚・猫でない所 {len(neg)} 個")
    groups = (("snoo（全身）", pos), ("snoo（体の一部）", part), ("猫でない所", neg))
    for name, v in groups:
        low, high = np.percentile(v, 1), np.percentile(v, 99)
        print(f"{name}の近さ: 1% {low:.3f}  中央 {np.median(v):.3f}  99% {high:.3f}")
    if pos_low <= neg_high:
        raise SystemExit("snoo と猫でない所の近さが重なっているので、境目を決められない")
    # 猫でない所の 99.5% が 0.5 未満、snoo（全身）の 99% が 0.95 以上になるようにする
    mid = neg_high
    gain = np.log(19) / (pos_low - mid)
    for name, v in groups:
        scores = 1 / (1 + np.exp(-(v - mid) * gain))
        print(f"{name}: snoo ≥ 0.6 になる割合 {np.mean(scores >= 0.6):.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        embeddings=gallery.cpu().numpy().astype(np.float16),
        model=f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
        mid=mid,
        gain=gain,
        files=np.array(files),
    )
    print(f"{args.output} に保存した（mid {mid:.3f}・gain {gain:.1f}）")


if __name__ == "__main__":
    main()
