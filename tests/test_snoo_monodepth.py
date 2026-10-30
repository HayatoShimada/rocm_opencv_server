"""単眼の metric の深度の答え合わせの計算（GPU もモデルも使わない。深度は作り物）。"""

import math

import numpy as np
import pytest

from app.clearance import MAX_CM, Camera, Path, pixel_rays
from app.snoo_monodepth import (
    SCALE_TOL,
    clearance_from_metric,
    depth_stats,
    floor_depth,
    floor_scale_check,
    median_ratio,
)


def test_floor_depth_level_camera_matches_pinhole():
    # 水平のカメラでは、中心より dv 画素下の床は z = 高さ × f / dv
    cam = Camera(height_cm=8, tilt_deg=0, width=160, height=120)
    z = floor_depth(cam)
    cx, cy = cam.center
    for row in (80, 100, 119):
        assert z[row, 80] == pytest.approx(8 * cam.f / (row - cy), rel=1e-6)
    # 水平より上は床が映らない
    assert np.isnan(z[: int(cy)]).all()


def test_floor_depth_center_pixel_when_tilted_down():
    # 15° 下を向くと、真ん中の光線は 8 / sin 15° 先で床に当たる。z はその光線の長さそのもの
    cam = Camera(height_cm=8, tilt_deg=-15, width=161, height=121)
    assert floor_depth(cam)[60, 80] == pytest.approx(8 / math.sin(math.radians(15)), rel=1e-6)


def test_floor_depth_does_not_depend_on_pan():
    a = floor_depth(Camera(tilt_deg=-15, pan_deg=0, width=64, height=48))
    b = floor_depth(Camera(tilt_deg=-15, pan_deg=30, width=64, height=48))
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_depth_stats_and_median_ratio_on_a_halved_depth():
    rng = np.random.default_rng(0)
    gt = rng.uniform(20, 200, (40, 50))
    pred = gt / 2  # 尺度だけ半分（形は合っている）
    s = depth_stats(pred, gt)
    assert s.ratio == pytest.approx(2.0)
    assert s.abs_rel == pytest.approx(0.5)
    assert s.delta1 == 0.0
    assert s.abs_rel_scaled == pytest.approx(0.0, abs=1e-12)
    assert median_ratio(pred, gt) == pytest.approx(2.0)


def test_depth_stats_ignores_invalid_pixels_and_needs_enough():
    gt = np.full((10, 10), 100.0)
    pred = gt.copy()
    pred[0] = np.nan
    pred[1] = 0
    gt[2] = np.inf
    mask = np.ones_like(gt, bool)
    mask[3] = False
    s = depth_stats(pred, gt, mask)
    assert s.n == 60 and s.abs_rel == 0 and s.delta1 == 1.0 and s.rmse_cm == 0
    assert depth_stats(pred, gt, np.zeros_like(mask)) is None
    assert median_ratio(pred, gt, np.zeros_like(mask)) is None


def test_floor_scale_check_finds_the_scale_and_ignores_a_box():
    cam = Camera(tilt_deg=-15, width=160, height=120)
    floor = floor_depth(cam)
    pred = floor * 3.0  # 床を 3 倍遠く見る（低い視点でよく起きた）
    pred[90:, 60:90] = 10.0  # 手前の箱（床より近い）
    c = floor_scale_check(pred, cam)
    assert c.ratio == pytest.approx(1 / 3, rel=1e-3)
    assert not c.ok and 0.5 < c.inliers < 1.0
    good = floor_scale_check(floor * 1.1, cam)
    assert good.ok and good.ratio == pytest.approx(1 / 1.1, rel=1e-3)
    assert 1 / SCALE_TOL < good.ratio < SCALE_TOL


def test_floor_scale_check_without_floor_in_view():
    cam = Camera(tilt_deg=20, width=160, height=120)
    c = floor_scale_check(np.full((120, 160), 100.0), cam)
    assert c.ratio is None and not c.ok


def _metric_depth(cam, wall_cm=300.0, box=None):
    """床と奥の壁（と箱）の奥行き（cm）。box=(前の面の x, 横の中心, 幅, 高さ)。"""
    v, u = np.mgrid[0 : cam.height, 0 : cam.width]
    ray = pixel_rays(cam, u, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(ray[..., 2] < -1e-9, cam.height_cm / -ray[..., 2], np.inf)
        z = np.minimum(z, wall_cm / ray[..., 0])
        if box is not None:
            x0, y0, width, height = box
            zb = x0 / ray[..., 0]
            hit = np.abs(zb * ray[..., 1] - y0) < width / 2
            hit &= cam.height_cm + zb * ray[..., 2] < height
            z = np.where(hit & (zb > 0) & (zb < z), zb, z)
    return z.astype(np.float32)


def test_clearance_from_metric_free_floor_and_a_box():
    full = Camera(tilt_deg=-20)
    cam = full.scaled(217, 161)
    path = Path()
    assert clearance_from_metric(_metric_depth(cam), full, path).clear_cm == MAX_CM
    z = _metric_depth(cam, box=(45.0, 0.0, 8.0, 10.0))
    r = clearance_from_metric(z, full, path)
    assert r.method == "metric"
    assert r.clear_cm == pytest.approx(45 - path.car.cam_to_front_cm, abs=2)


def test_clearance_from_metric_scale_undoes_a_wrong_scale():
    # 3 倍遠く見たままだと、箱は遠くへ行き床は沈んで見える。床で決めた尺度をかければ戻る
    full = Camera(tilt_deg=-20)
    cam = full.scaled(217, 161)
    path = Path()
    z = _metric_depth(cam, box=(45.0, 0.0, 8.0, 10.0)) * 3
    ratio = floor_scale_check(z, cam).ratio
    assert ratio == pytest.approx(1 / 3, rel=0.05)
    r = clearance_from_metric(z, full, path, scale=ratio)
    assert r.clear_cm == pytest.approx(45 - path.car.cam_to_front_cm, abs=2)


def test_floor_affine_undoes_a_wrong_inverse_offset():
    # 見積もりの逆数が本当の逆数の一次式（1/pred = 0.4/z + 0.002）なら、床で合わせ直すと戻る
    from app.snoo_monodepth import floor_affine

    full = Camera(tilt_deg=-20)
    cam = full.scaled(217, 161)
    z = _metric_depth(cam, box=(45.0, 0.0, 8.0, 10.0))
    pred = 1 / (0.4 / z + 0.002)
    fixed, inliers = floor_affine(pred, cam)
    ok = np.isfinite(z) & (z < 250)
    np.testing.assert_allclose(fixed[ok], z[ok], rtol=1e-3)
    assert inliers > 0.8
    path = Path()
    r = clearance_from_metric(fixed, full, path)
    assert r.clear_cm == pytest.approx(45 - path.car.cam_to_front_cm, abs=2)
