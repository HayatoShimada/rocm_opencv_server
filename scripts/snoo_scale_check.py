"""床に置いた A4 の紙を 1 枚撮って、カメラの高さ（地図の縮尺のもと）を確かめる。

  python -m scripts.snoo_scale_check <画像> --tilt -15 [--pan 0] [--height-cm 8]
  make scale-check IMAGE=... TILT=-15

カメラの高さは SD カード 2.5 枚分と目で測った 8 cm で、地図の大きさ・clear_cm の距離はみなこれに
比例する。白い A4（29.7 × 21 cm）を、車の 30〜60 cm 前の床に、縁が全部写るように置いて撮る。
紙の 4 つの角を画像から探し（いちばん大きい明るい四角）、床の上の位置に戻して（app/clearance の
floor_point）、辺の長さが 29.7 × 21 cm になる高さを求める。
- 長い辺と短い辺から求めた高さが大きく違えば（縦横の比が合わなければ）、チルトがずれている
- 紙は床に平らに置く。暗いと角が取れないので明かりをつける
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.clearance import CAM_HEIGHT_CM, Camera, floor_point

A4_CM = (29.7, 21.0)
# 長い辺と短い辺から求めた高さの差がこの割合を超えたら、チルトのずれを疑う
ASPECT_TOL = 0.08


def find_sheet(image: np.ndarray) -> np.ndarray | None:
    """いちばん大きい明るい四角の 4 つの角（画素、(4, 2)、ぐるりと回る順）。

    見つからなければ None。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < 0.005 * gray.size:
            break
        quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(quad) == 4 and cv2.isContourConvex(quad):
            best = quad.reshape(4, 2).astype(np.float64)
            break
    return best


def sheet_sides(cam: Camera, corners: np.ndarray) -> tuple[float, float] | None:
    """角を床に戻した四角の、向かい合う辺の平均の長さ（長いほう, 短いほう）。

    単位は cam.height_cm と同じ。
    """
    x, y = floor_point(cam, corners[:, 0], corners[:, 1])
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    p = np.stack([x, y], 1)
    d = [float(np.linalg.norm(p[(k + 1) % 4] - p[k])) for k in range(4)]
    a, b = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
    return max(a, b), min(a, b)


def estimate_height(image: np.ndarray, tilt_deg: float, pan_deg: float = 0.0) -> dict | None:
    """A4 の紙から求めたカメラの高さ（cm）。紙が見つからなければ None。"""
    corners = find_sheet(image)
    if corners is None:
        return None
    h, w = image.shape[:2]
    cam = Camera(height_cm=1.0, tilt_deg=tilt_deg, pan_deg=pan_deg, width=w, height=h)
    sides = sheet_sides(cam, corners)
    if sides is None:
        return None
    long_h, short_h = A4_CM[0] / sides[0], A4_CM[1] / sides[1]
    height = (long_h + short_h) / 2
    spread = abs(long_h - short_h) / height
    return {
        "height_cm": round(height, 2),
        "from_long_cm": round(long_h, 2),
        "from_short_cm": round(short_h, 2),
        "aspect_ok": spread <= ASPECT_TOL,
        "corners": corners.round(1).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--tilt", type=float, required=True, help="撮った時のチルト（度、上が +）")
    parser.add_argument("--pan", type=float, default=0.0)
    parser.add_argument("--height-cm", type=float, default=CAM_HEIGHT_CM, help="今の設定の高さ")
    args = parser.parse_args()
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"{args.image} を読めない")
    r = estimate_height(img, args.tilt, args.pan)
    if r is None:
        raise SystemExit("A4 の紙が見つからない（明かりをつけ、縁が全部写るように置く）")
    ratio = r["height_cm"] / args.height_cm
    print(
        f"カメラの高さ: {r['height_cm']:.2f} cm（長い辺から {r['from_long_cm']:.2f}・"
        f"短い辺から {r['from_short_cm']:.2f}）。今の設定 {args.height_cm:.1f} cm の {ratio:.2f} 倍"
    )
    if not r["aspect_ok"]:
        print("長い辺と短い辺で高さが合わない：チルトがずれているか、紙が床に平らでない")
    elif abs(ratio - 1) > 0.05:
        print(f"SNOO_CAMERA_HEIGHT_CM と Pi の camera_height_cm を {r['height_cm']:.1f} にする")
    else:
        print("今の設定のままでよい")


if __name__ == "__main__":
    main()
