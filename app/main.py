from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app import processing
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


def _read_image(file: UploadFile):
    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "画像サイズが上限を超えています")
    try:
        return processing.decode(data)
    except processing.ImageError as e:
        raise HTTPException(400, str(e)) from e


def _respond(image, fmt: ImageFormat) -> Response:
    return Response(processing.encode(image, fmt), media_type=MEDIA_TYPES[fmt])


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
