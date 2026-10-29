"""snoo の動画から、フレームごとの猫の箱と関節（app/cat_pose.py）を取り出す。

  uv run --group clip --group clef-local python -m scripts.snoo_pose data/snoo_video/IMG_5903.mov
結果は動画の隣に <名前>_pose.json で保存する（names: 関節の名前、fps、frames: フレームごとの
{box, cat, keypoints} か、猫がいなければ null）。
snoo のシミュレーター用のモデル（動き・姿勢・見え方）の素材にする。
"""

import argparse
import json
from pathlib import Path

import cv2

from app.cat_pose import NAMES, CatPose
from app.snoo import CatDetector

# 関節を取る猫の箱の確からしさの下限
CAT_MIN = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args()

    detector = CatDetector(cat_min=CAT_MIN)
    pose = CatPose(detector.device)
    for path in args.videos:
        capture = cv2.VideoCapture(str(path))  # 縦に撮った動画は向きを直して読む
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames, size = [], [0, 0]
        while True:
            ok, image = capture.read()
            if not ok:
                break
            size = [image.shape[1], image.shape[0]]
            dets = detector.detect(image)
            if not dets:
                frames.append(None)
                continue
            keypoints = pose.estimate(image, dets[0].box)
            frames.append(
                {
                    "box": [round(v, 4) for v in dets[0].box],
                    "cat": round(dets[0].cat, 3),
                    "keypoints": [[round(float(v), 1) for v in kp[:2]] + [round(float(kp[2]), 3)]
                                  for kp in keypoints],
                }
            )  # fmt: skip
        out = path.with_name(f"{path.stem}_pose.json")
        out.write_text(json.dumps({"names": NAMES, "fps": fps, "size": size, "frames": frames}))
        seen = sum(f is not None for f in frames)
        print(f"{path.name}: {len(frames)} フレーム中 {seen} に猫 → {out}")


if __name__ == "__main__":
    main()
