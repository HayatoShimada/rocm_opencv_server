import cv2
import numpy as np
import pytest

from app.clearance import Camera, project
from scripts.snoo_scale_check import estimate_height


def _render(height_cm, tilt_deg, x0=35.0, y0=0.0):
    """床の (x0, y0) を中心に置いた A4（長い辺を前後に）を、高さ height_cm のカメラで描く。"""
    cam = Camera(height_cm=height_cm, tilt_deg=tilt_deg)
    corners = np.array([
        [x0 - 14.85, y0 - 10.5, 0], [x0 + 14.85, y0 - 10.5, 0],
        [x0 + 14.85, y0 + 10.5, 0], [x0 - 14.85, y0 + 10.5, 0],
    ])  # fmt: skip
    u, v = project(cam, corners)
    img = np.full((480, 640, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.stack([u, v], 1).round().astype(np.int32)], (235, 235, 235))
    return img


@pytest.mark.parametrize("height", [7.0, 8.0, 9.5])
def test_estimate_height_recovers_the_camera_height(height):
    r = estimate_height(_render(height, -15.0), tilt_deg=-15.0)
    assert r is not None and r["aspect_ok"]
    assert r["height_cm"] == pytest.approx(height, rel=0.05)


def test_wrong_tilt_breaks_the_aspect_ratio():
    r = estimate_height(_render(8.0, -15.0), tilt_deg=-10.0)  # 5° ずれた設定で計算する
    assert r is not None and not r["aspect_ok"]
    # 10° ずれると、奥の角が水平より上に戻って答えが出ない
    assert estimate_height(_render(8.0, -15.0), tilt_deg=-5.0) is None


def test_no_sheet():
    assert estimate_height(np.full((480, 640, 3), 60, np.uint8), tilt_deg=-15.0) is None
