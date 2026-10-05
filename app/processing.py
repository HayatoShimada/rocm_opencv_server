"""画像処理の本体。

入力は numpy 配列で受け取り、OpenCL が有効なら UMat に載せて GPU で処理する。
OpenCL が無効な環境では同じコードがそのまま CPU で動く。
"""

import cv2
import numpy as np

ENCODINGS = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}


class ImageError(ValueError):
    pass


def decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageError("画像としてデコードできませんでした")
    return image


def encode(image: np.ndarray, fmt: str) -> bytes:
    ok, buf = cv2.imencode(ENCODINGS[fmt], image)
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


def gaussian_blur(image: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0:
        raise ImageError("ksize は奇数で指定してください")
    return _to_host(cv2.GaussianBlur(_to_device(image), (ksize, ksize), sigma))


def canny(image: np.ndarray, threshold1: float, threshold2: float) -> np.ndarray:
    gray = cv2.cvtColor(_to_device(image), cv2.COLOR_BGR2GRAY)
    return _to_host(cv2.Canny(gray, threshold1, threshold2))
