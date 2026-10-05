"""画像処理の本体。

入力は numpy 配列で受け取り、OpenCL が有効なら UMat に載せて GPU で処理する。
OpenCL が無効な環境では同じコードがそのまま CPU で動く。
"""

import cv2
import numpy as np

ENCODINGS = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
QUALITY_FLAGS = {"jpeg": cv2.IMWRITE_JPEG_QUALITY, "webp": cv2.IMWRITE_WEBP_QUALITY}


class ImageError(ValueError):
    pass


def decode(data: bytes, keep_alpha: bool = False) -> np.ndarray:
    # IMREAD_COLOR は EXIF の向きを反映する。透過を残すときは UNCHANGED で読む（向きは反映されない）
    flag = cv2.IMREAD_UNCHANGED if keep_alpha else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
    if image is None:
        raise ImageError("画像としてデコードできませんでした")
    return image


def has_alpha(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] == 4


def encode(image: np.ndarray, fmt: str, quality: int | None = None) -> bytes:
    params = [QUALITY_FLAGS[fmt], quality] if quality is not None and fmt in QUALITY_FLAGS else []
    ok, buf = cv2.imencode(ENCODINGS[fmt], image, params)
    if not ok:
        raise ImageError(f"{fmt} へのエンコードに失敗しました")
    return buf.tobytes()


def _to_device(image: np.ndarray) -> cv2.UMat | np.ndarray:
    return cv2.UMat(image) if cv2.ocl.useOpenCL() else image


def _to_host(image: cv2.UMat | np.ndarray) -> np.ndarray:
    return image.get() if isinstance(image, cv2.UMat) else image


def grayscale(image: np.ndarray) -> np.ndarray:
    return _to_host(cv2.cvtColor(_to_device(image), cv2.COLOR_BGR2GRAY))


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return _to_host(cv2.resize(_to_device(image), (width, height), interpolation=cv2.INTER_AREA))


def fit_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """縦横比を保ち、長辺が max_side に収まるように縮める（拡大はしない）。"""
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return _to_host(cv2.resize(_to_device(image), size, interpolation=cv2.INTER_AREA))


def gaussian_blur(image: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0:
        raise ImageError("ksize は奇数で指定してください")
    return _to_host(cv2.GaussianBlur(_to_device(image), (ksize, ksize), sigma))


def canny(image: np.ndarray, threshold1: float, threshold2: float) -> np.ndarray:
    gray = cv2.cvtColor(_to_device(image), cv2.COLOR_BGR2GRAY)
    return _to_host(cv2.Canny(gray, threshold1, threshold2))
