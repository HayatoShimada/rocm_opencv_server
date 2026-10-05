"""単品の商品写真のホワイトバランスと明るさを、基準の画像の壁の色に揃える。

基準は amazon-hooded-sweat-parkaused（[GILDAN] "Amazon Blueprint" Hooded Sweatshirt）の1枚目
（0E2A5894.jpg）の、右上の壁の範囲の平均。それぞれの写真の同じ範囲を測り、リニア RGB で
チャンネルごとのゲイン（目標 ÷ 測った値）を画像全体に掛ける。色かぶりと明るさが同時に揃う。
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

# 基準の壁の色（sRGB、0〜255）
TARGET_RGB = (205.1, 206.7, 206.9)
# 測る範囲（幅・高さに対する割合 x0, x1, y0, y1）。最初が基準の範囲。
# 服やハンガーが入っていたら次を試す
REGIONS = [
    (0.751, 0.839, 0.047, 0.101),  # 右上（基準）
    (0.161, 0.249, 0.047, 0.101),  # 左上（左右対称）
    (0.751, 0.839, 0.110, 0.164),  # 右上の少し下
    (0.161, 0.249, 0.110, 0.164),  # 左上の少し下
    (0.870, 0.960, 0.047, 0.101),  # 右端
    (0.040, 0.130, 0.047, 0.101),  # 左端
]
# 壁とみなす範囲の条件: 画素のばらつき（標準偏差。基準は約 6）と、
# 平均の色の彩度（Lab の a*b* の大きさ）
MAX_STD = 12.0
MAX_CHROMA = 12.0
# 壁は明るい。暗く均一な服（紺のニットなど）を壁とみなさないため
MIN_WALL_LEVEL = 140.0
# これを超えるゲインは自動では当てない（撮り方が違う・壁でない可能性が高い）
GAIN_RANGE = (0.7, 1.4)
# 目標との色差（CIE76）がこれ未満なら補正しない（もう一度流しても変わらない）
MIN_DELTA_E = 2.0
# 壁の平均（sRGB）がこれ以上なら白い壁とみなし、補正しない
# （灰色の壁は明るく写っても 215 くらいまで）
WHITE_WALL_LEVEL = 220
# リニア RGB の輝度の重み（Rec.709）。色かぶりだけ直すときに明るさを保つ
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])

Status = Literal["apply", "ok", "review"]


@dataclass
class Correction:
    status: Status  # apply: 補正する / ok: 補正不要 / review: 要確認
    reason: str
    region: tuple[float, float, float, float] | None = None
    measured_rgb: tuple[float, float, float] | None = None
    gains: tuple[float, float, float] | None = None
    delta_e: float | None = None


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _lab(rgb: tuple[float, float, float]) -> np.ndarray:
    pixel = np.array([[rgb[::-1]]], dtype=np.float32) / 255  # BGR、0〜1
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]


def _delta_e(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(_lab(a) - _lab(b)))


def measure(image: np.ndarray):
    """壁とみなせる最初の範囲と、その平均の色（RGB）を返す。どれも壁でなければ None。"""
    height, width = image.shape[:2]
    for region in REGIONS:
        x0, x1, y0, y1 = region
        crop = image[int(height * y0) : int(height * y1), int(width * x0) : int(width * x1)]
        pixels = crop.reshape(-1, 3).astype(np.float32)
        if pixels.std(axis=0).mean() > MAX_STD:
            continue
        b, g, r = pixels.mean(axis=0)
        rgb = (float(r), float(g), float(b))
        if min(rgb) < MIN_WALL_LEVEL:
            continue
        _, a_star, b_star = _lab(rgb)
        if np.hypot(a_star, b_star) > MAX_CHROMA:
            continue
        return region, rgb
    return None


def plan(image: np.ndarray) -> Correction:
    """補正するかどうかと、ゲインを決める（画像は変えない）。"""
    found = measure(image)
    if not found:
        return Correction("review", "壁とみなせる範囲がない")
    region, rgb = found
    if sum(rgb) / 3 >= WHITE_WALL_LEVEL:
        return Correction("ok", "白い壁（触らない）", region, rgb, (1.0, 1.0, 1.0), 0.0)
    delta = _delta_e(rgb, TARGET_RGB)
    if delta < MIN_DELTA_E:
        return Correction("ok", "目標に近い", region, rgb, (1.0, 1.0, 1.0), delta)
    target = srgb_to_linear(np.array(TARGET_RGB) / 255)
    measured = srgb_to_linear(np.array(rgb) / 255)
    gains = tuple(float(v) for v in target / measured)
    if all(GAIN_RANGE[0] <= g <= GAIN_RANGE[1] for g in gains):
        return Correction("apply", "補正する", region, rgb, gains, delta)
    # 明るさまで合わせると大きすぎる（壁が真っ白・暗い）ときは、明るさはそのままで色かぶりだけ直す
    scale = float(LUMINANCE @ measured) / float(LUMINANCE @ target)
    cast_target = target * scale
    cast_gains = tuple(float(v) for v in cast_target / measured)
    cast_rgb = tuple(float(v) * 255 for v in linear_to_srgb(cast_target))
    cast_delta = _delta_e(rgb, cast_rgb)
    if cast_delta < MIN_DELTA_E:
        return Correction(
            "ok", "色かぶりなし（明るさは基準と違う）", region, rgb, (1.0, 1.0, 1.0), cast_delta
        )
    if all(GAIN_RANGE[0] <= g <= GAIN_RANGE[1] for g in cast_gains):
        return Correction(
            "apply", "色かぶりだけ補正（明るさはそのまま）", region, rgb, cast_gains, cast_delta
        )
    return Correction("review", "補正が大きすぎる", region, rgb, gains, delta)


# 補正後に白に近い画素（いちばん暗いチャンネルがこれ以上）は、無彩色に寄せる。
# 白い服は1つのチャンネルだけ白飛びしていることが多く（青い壁の写真の青など）、
# そのままゲインを掛けると黄色・緑に色づくため。色の服はいちばん暗いチャンネルが低いので変わらない
NEUTRAL_FROM = 225
NEUTRAL_TO = 245


def apply_gains(image: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    """リニア RGB でゲインを掛ける。

    1 を超える画素は、色の比率を保ったまま縮める。白に近い画素は無彩色に寄せる。
    """
    linear = srgb_to_linear(image.astype(np.float32) / 255)
    linear *= np.array(gains[::-1], dtype=np.float32)  # BGR の順
    peak = linear.max(axis=2, keepdims=True)
    linear = np.where(peak > 1, linear / np.maximum(peak, 1e-6), linear)
    out = linear_to_srgb(linear) * 255
    floor = out.min(axis=2, keepdims=True)
    weight = np.clip((floor - NEUTRAL_FROM) / (NEUTRAL_TO - NEUTRAL_FROM), 0, 1)
    out = out * (1 - weight) + out.max(axis=2, keepdims=True) * weight
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
