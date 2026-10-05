import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def png_bytes() -> bytes:
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    cv2.rectangle(image, (16, 16), (80, 48), (255, 255, 255), -1)
    return cv2.imencode(".png", image)[1].tobytes()


def _decode(content: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)


def _upload(png: bytes):
    return {"file": ("test.png", png, "image/png")}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_device(client):
    body = client.get("/device").json()
    assert "opencl_available" in body
    assert "opencv_version" in body


def test_grayscale(client, png_bytes):
    res = client.post("/v1/grayscale", files=_upload(png_bytes))
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert _decode(res.content).shape == (64, 96)


def test_resize(client, png_bytes):
    res = client.post("/v1/resize?width=48&height=32", files=_upload(png_bytes))
    assert res.status_code == 200
    assert _decode(res.content).shape == (32, 48, 3)


def test_blur_rejects_even_ksize(client, png_bytes):
    res = client.post("/v1/blur?ksize=4", files=_upload(png_bytes))
    assert res.status_code == 400


def test_canny_detects_edges(client, png_bytes):
    res = client.post("/v1/canny", files=_upload(png_bytes))
    assert res.status_code == 200
    assert _decode(res.content).max() == 255


def test_jpeg_output(client, png_bytes):
    res = client.post("/v1/grayscale?format=jpeg", files=_upload(png_bytes))
    assert res.headers["content-type"] == "image/jpeg"


def test_invalid_image(client):
    res = client.post("/v1/grayscale", files={"file": ("x.png", b"not an image", "image/png")})
    assert res.status_code == 400
