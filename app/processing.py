"""画像処理の本体。

入力は numpy 配列で受け取り、OpenCL が有効なら UMat に載せて GPU で処理する。
OpenCL が無効な環境では同じコードがそのまま CPU で動く。
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageCms

ENCODINGS = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
QUALITY_FLAGS = {"jpeg": cv2.IMWRITE_JPEG_QUALITY, "webp": cv2.IMWRITE_WEBP_QUALITY}
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


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
    if fmt == "jpeg":
        # ハフマン表を最適化する（画質は同じで数％小さくなる）
        params += [cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    ok, buf = cv2.imencode(ENCODINGS[fmt], image, params)
    if not ok:
        raise ImageError(f"{fmt} へのエンコードに失敗しました")
    return buf.tobytes()


def icc_profile(data: bytes) -> bytes | None:
    """埋め込みの ICC プロファイルを返す（OpenCV は読み書きできないので Pillow で読む）。"""
    try:
        with Image.open(BytesIO(data)) as image:
            return image.info.get("icc_profile")
    except OSError:
        return None


def _primaries(profile: ImageCms.ImageCmsProfile) -> list[tuple[float, float]]:
    p = profile.profile
    return [c[1][:2] for c in (p.red_colorant, p.green_colorant, p.blue_colorant)]


def _is_srgb(profile: ImageCms.ImageCmsProfile) -> bool:
    """原色の色度が sRGB と同じか（名前は "c2ci" のような小さい sRGB もあるので当てにしない）。"""
    try:
        primaries = _primaries(profile)
    except (TypeError, ValueError):
        return False
    return all(
        abs(a - b) < 0.005
        for p, q in zip(primaries, _primaries(SRGB), strict=True)
        for a, b in zip(p, q, strict=True)
    )


def to_srgb(image: np.ndarray, icc: bytes | None) -> np.ndarray:
    """ICC が sRGB 以外の RGB（Display P3 など）なら sRGB に変換する。

    書き出す画像には ICC を埋め込まない（ブラウザは sRGB とみなす）ので、変換しないと色がずれる。
    """
    if not icc or image.dtype != np.uint8 or image.ndim != 3:
        return image
    try:
        profile = ImageCms.ImageCmsProfile(BytesIO(icc))
    except (OSError, ImageCms.PyCMSError):
        return image
    if profile.profile.xcolor_space.strip() != "RGB" or _is_srgb(profile):
        return image
    alpha = has_alpha(image)
    mode = "RGBA" if alpha else "RGB"
    rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if alpha else cv2.COLOR_BGR2RGB)
    converted = ImageCms.profileToProfile(
        Image.fromarray(rgb, mode), profile, SRGB, outputMode=mode
    )
    return cv2.cvtColor(np.asarray(converted), cv2.COLOR_RGBA2BGRA if alpha else cv2.COLOR_RGB2BGR)


def _to_device(image: np.ndarray) -> cv2.UMat | np.ndarray:
    return cv2.UMat(image) if cv2.ocl.useOpenCL() else image


def _to_host(image: cv2.UMat | np.ndarray) -> np.ndarray:
    return image.get() if isinstance(image, cv2.UMat) else image


def grayscale(image: np.ndarray) -> np.ndarray:
    return _to_host(cv2.cvtColor(_to_device(image), cv2.COLOR_BGR2GRAY))


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return _to_host(cv2.resize(_to_device(image), (width, height), interpolation=cv2.INTER_AREA))


def fit_long_side(image: np.ndarray, max_side: int, device: bool = True) -> np.ndarray:
    """縦横比を保ち、長辺が max_side に収まるように縮める（拡大はしない）。

    1枚を縮めるだけなら GPU との転送のほうが重いので、device=False で CPU で処理できる。
    """
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    src = _to_device(image) if device else image
    return _to_host(cv2.resize(src, size, interpolation=cv2.INTER_AREA))


def gaussian_blur(image: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0:
        raise ImageError("ksize は奇数で指定してください")
    return _to_host(cv2.GaussianBlur(_to_device(image), (ksize, ksize), sigma))


def canny(image: np.ndarray, threshold1: float, threshold2: float) -> np.ndarray:
    gray = cv2.cvtColor(_to_device(image), cv2.COLOR_BGR2GRAY)
    return _to_host(cv2.Canny(gray, threshold1, threshold2))
