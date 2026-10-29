"""猫の画像に、Clef で状態（何をしているか・どこが見えているか・どちらを向いているか）の印を付ける。

  uv run --group clip --group clef-local python -m scripts.snoo_teacher [--source data/snoo_raw]
Clef は 1 枚 0.5 秒ほどかかり、検出（app/snoo.py）と同じ GPU で動かすと検出が遅くなる。
そのため速い経路では使わず、ここで付けた印を CLIP の小さな分類器に覚えさせる
（scripts/snoo_state_head.py。蒸留）。車が止まっているときや夜に回す。
印は猫の箱ごとに data/snoo/teacher.json に保存し、質問が変わっていなければ付け直さない。
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from app import clef
from app.config import settings
from app.snoo import CatDetector

OUTPUT = Path("data/snoo/teacher.json")
# 印を付ける猫の箱の確からしさの下限
CAT_MIN = 0.5
STATE = "床の近くを走るロボットのカメラで、家の猫を見ている。画像は猫の箱を切り出したもの"
QUESTIONS = {
    "activity": {
        "type": "choice",
        "instructions": "猫は何をしているか",
        "criteria": {
            "sleeping": "寝ている・丸まって目を閉じている",
            "lying": "寝そべっているが起きている",
            "sitting": "座っている・香箱座り",
            "standing": "立ち止まっている",
            "walking": "歩いている・走っている",
            "eating": "食べている・飲んでいる",
            "grooming": "毛づくろいしている",
        },
    },
    "visible": {
        "type": "choice",
        "instructions": "猫の体のどこが見えているか",
        "criteria": {
            "face": "顔が見えている",
            "side": "横から体が見えている（顔は見えない・横顔）",
            "back": "背中・後ろ姿",
            "part": "しっぽや体の一部だけ",
        },
    },
    "facing": {
        "type": "choice",
        "instructions": "猫の頭は画像のどちらを向いているか",
        "criteria": {
            "toward": "カメラのほう（手前）",
            "away": "カメラと反対（奥）",
            "left": "画像の左",
            "right": "画像の右",
        },
    },
}
# 質問を変えたら印を付け直す
VERSION = hashlib.sha1(json.dumps([STATE, QUESTIONS], ensure_ascii=False).encode()).hexdigest()[:8]


def crop(image, box, pad: float = 0.1):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    left, right = int(max(0, x1 - pw) * w), int(min(1, x2 + pw) * w)
    top, bottom = int(max(0, y1 - ph) * h), int(min(1, y2 + ph) * h)
    return image[top:bottom, left:right]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=Path("data/snoo_raw"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    labels = json.loads(args.output.read_text()) if args.output.exists() else {}
    detector = CatDetector(cat_min=CAT_MIN)
    teacher = clef.from_settings(settings)
    paths = sorted(args.source.glob("*.jpg"))
    for i, path in enumerate(paths):
        image = cv2.imread(str(path))
        for j, det in enumerate(detector.detect(image)):
            key = f"{path}#{j}"
            if labels.get(key, {}).get("version") == VERSION:
                continue
            jpeg = cv2.imencode(".jpg", crop(image, det.box), [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
            answers = teacher.ask(STATE, QUESTIONS, [(jpeg.tobytes(), "image/jpeg")])
            labels[key] = {
                "version": VERSION,
                "file": str(path),
                "box": [round(v, 4) for v in det.box],
                "cat": round(det.cat, 4),
                **{q: answers[q]["probabilities"] for q in QUESTIONS},
            }
        if i % 20 == 19 or i == len(paths) - 1:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(labels, ensure_ascii=False, indent=1))
            print(f"{i + 1}/{len(paths)}")


if __name__ == "__main__":
    main()
