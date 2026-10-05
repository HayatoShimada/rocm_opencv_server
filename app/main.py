import secrets
from contextlib import asynccontextmanager
from functools import cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import processing, shopify
from app.config import settings
from app.gpu import device_info, init_opencl

ImageFormat = Literal["png", "jpeg", "webp"]
ImageFile = Annotated[UploadFile, File()]
MEDIA_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_opencl(settings.use_opencl)
    yield


# 画像処理はブロッキングなので、ハンドラは同期関数にしてスレッドプールで実行させる
app = FastAPI(title="ROCm OpenCV Server", version="0.1.0", lifespan=lifespan)


def _read_image(file: UploadFile, keep_alpha: bool = False):
    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "画像サイズが上限を超えています")
    try:
        return processing.decode(data, keep_alpha)
    except processing.ImageError as e:
        raise HTTPException(400, str(e)) from e


def _respond(image, fmt: ImageFormat, quality: int | None = None) -> Response:
    return Response(processing.encode(image, fmt, quality), media_type=MEDIA_TYPES[fmt])


def _require_token(x_internal_token: Annotated[str, Header()] = "") -> None:
    if not settings.api_token:
        raise HTTPException(503, "API_TOKEN が設定されていません")
    if not secrets.compare_digest(x_internal_token, settings.api_token):
        raise HTTPException(401, "X-Internal-Token が違います")


@cache
def _shopify() -> shopify.Shopify:
    return shopify.Shopify(
        settings.shopify_store,
        settings.shopify_client_id,
        settings.shopify_client_secret,
        settings.shopify_admin_token,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/device")
def device():
    return device_info()


@app.post("/v1/grayscale")
def grayscale(file: ImageFile, format: ImageFormat = "png"):
    image = _read_image(file)
    return _respond(processing.grayscale(image), format)


@app.post("/v1/resize")
def resize(
    file: ImageFile,
    width: Annotated[int, Query(gt=0, le=16384)],
    height: Annotated[int, Query(gt=0, le=16384)],
    format: ImageFormat = "png",
):
    image = _read_image(file)
    return _respond(processing.resize(image, width, height), format)


@app.post("/v1/fit")
def fit(
    file: ImageFile,
    max_side: Annotated[int, Query(gt=0, le=16384)],
    format: ImageFormat = "jpeg",
    quality: Annotated[int, Query(ge=1, le=100)] = 90,
):
    # PNG・WebP は透過を残す。JPEG は透過を捨てる
    image = _read_image(file, keep_alpha=format != "jpeg")
    return _respond(processing.fit_long_side(image, max_side), format, quality)


class FitImagesRequest(BaseModel):
    productId: str
    maxSide: int | None = Field(default=None, gt=0, le=16384)
    fillAlt: bool = False


@app.post("/v1/shopify/products/fit-images", dependencies=[Depends(_require_token)])
def fit_shopify_product_images(body: FitImagesRequest):
    """商品の画像のうち、長辺が上限を超えるものを縮めて Shopify で差し替える（CMS から呼ぶ）。

    fillAlt なら、空の alt も商品名で埋める。
    """
    try:
        result = shopify.fit_product_images(
            _shopify(),
            shopify.product_search(body.productId),
            body.maxSide or settings.fit_max_side,
            settings.fit_quality,
            apply=True,
            fill_alt=body.fillAlt,
        )
    except shopify.ShopifyError as e:
        raise HTTPException(502, str(e)) from e
    return result.to_dict()


@app.post("/v1/blur")
def blur(
    file: ImageFile,
    ksize: Annotated[int, Query(gt=0, le=99)] = 5,
    sigma: Annotated[float, Query(ge=0)] = 0,
    format: ImageFormat = "png",
):
    image = _read_image(file)
    try:
        result = processing.gaussian_blur(image, ksize, sigma)
    except processing.ImageError as e:
        raise HTTPException(400, str(e)) from e
    return _respond(result, format)


@app.post("/v1/canny")
def canny(
    file: ImageFile,
    threshold1: Annotated[float, Query(ge=0)] = 100,
    threshold2: Annotated[float, Query(ge=0)] = 200,
    format: ImageFormat = "png",
):
    image = _read_image(file)
    return _respond(processing.canny(image, threshold1, threshold2), format)
