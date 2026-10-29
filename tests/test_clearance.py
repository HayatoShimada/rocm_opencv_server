"""clear_cm の幾何と計算（GPU は使わない。画像と深度は作り物）。"""

import math

import cv2
import numpy as np
import pytest

from app.clearance import (
    MAX_CM,
    TILT_GAIN,
    TILT_OFFSET_DEG,
    Camera,
    Car,
    Clearance,
    FloorClearance,
    Grid,
    Path,
    clearance_from_disparity,
    combine,
    first_blocked,
    floor_point,
    floor_row,
    pixel_rays,
    project,
)


def test_floor_row_matches_pinhole_when_level():
    cam = Camera(height_cm=10, tilt_deg=0)
    cx, cy = cam.center
    for d in [20, 50, 100]:
        assert floor_row(cam, d) == pytest.approx(cy + cam.f * 10 / d)
    # 水平を向くと、画面の真ん中の行が水平線（床には当たらない）
    x, _ = floor_point(cam, cx, cy - 1)
    assert np.isnan(x)


@pytest.mark.parametrize("tilt,pan", [(0, 0), (-20, 0), (-20, 30), (10, -25)])
def test_floor_point_and_project_round_trip(tilt, pan):
    cam = Camera(height_cm=9, tilt_deg=tilt, pan_deg=pan)
    u = np.array([10.0, 320, 600, 200])
    v = np.array([470.0, 440, 420, 479])  # どの tilt でも水平線より下
    x, y = floor_point(cam, u, v)
    pu, pv = project(cam, np.stack([x, y, np.zeros_like(x)], axis=-1))
    np.testing.assert_allclose(pu, u, atol=1e-6)
    np.testing.assert_allclose(pv, v, atol=1e-6)


def test_tilt_down_sees_the_center_row_at_h_over_tan():
    cam = Camera(height_cm=10, tilt_deg=-20)
    cx, cy = cam.center
    x, y = floor_point(cam, cx, cy)
    assert x == pytest.approx(10 / math.tan(math.radians(20)))
    assert y == pytest.approx(0, abs=1e-9)


def test_pan_right_looks_to_the_right():
    cam = Camera(height_cm=10, tilt_deg=-20, pan_deg=30)
    x, y = floor_point(cam, *cam.center)
    assert y < 0  # 右は y が負
    assert math.degrees(math.atan2(-y, x)) == pytest.approx(30)


def test_rays_have_unit_forward_component():
    cam = Camera(tilt_deg=-15, pan_deg=10)
    d = pixel_rays(cam, 100.0, 300.0)
    # カメラの前への成分が 1（奥行き z をかければ点になる）
    assert (d @ cam.rotation())[2] == pytest.approx(1.0)


def test_from_pi_applies_tilt_calibration(monkeypatch):
    cam = Camera.from_pi(-15, 5)
    assert cam.tilt_deg == pytest.approx(-15 * TILT_GAIN + TILT_OFFSET_DEG)
    assert cam.pan_deg == 5
    # Pi が校正を直す前に撮った画像は、環境変数で前の直し方に戻して見る
    monkeypatch.setattr("app.clearance.TILT_GAIN", 1.08)
    monkeypatch.setattr("app.clearance.TILT_OFFSET_DEG", -4.5)
    assert Camera.from_pi(-15, 5).tilt_deg == pytest.approx(-15 * 1.08 - 4.5)


def test_tilt_is_not_corrected_by_default(monkeypatch):
    import importlib

    import app.clearance

    monkeypatch.delenv("SNOO_TILT_GAIN", raising=False)
    monkeypatch.delenv("SNOO_TILT_OFFSET_DEG", raising=False)
    module = importlib.reload(app.clearance)
    try:  # Pi が 10/29 20:20 ごろに校正を直したので、今のフレームは直さない
        assert (module.TILT_GAIN, module.TILT_OFFSET_DEG) == (1.0, 0.0)
    finally:
        importlib.reload(app.clearance)


def test_straight_path_is_x_minus_front():
    car = Car(cam_to_front_cm=5)
    path = Path(car, steer_deg=0)
    s, lat = path.to_path(np.array([25.0, 5.0]), np.array([3.0, -2.0]))
    np.testing.assert_allclose(s, [20, 0])
    np.testing.assert_allclose(lat, [3, -2])
    half = car.width_cm / 2 + car.margin_cm
    assert path.bounds() == (-half, half)
    assert path.inside(10, half - 0.1) and not path.inside(10, half + 0.1)
    assert not path.inside(-1, 0) and not path.inside(MAX_CM + 1, 0)


@pytest.mark.parametrize("steer", [-30, -10, 15, 30])
def test_curved_path_round_trip(steer):
    path = Path(Car(), steer_deg=steer)
    s = np.array([0.0, 10, 40, 70])
    s = s[s <= path.max_s]  # 半周まで
    lat = np.array([0.0, 5, -8, 3])[: len(s)]
    x, y = path.from_path(s, lat)
    s2, lat2 = path.to_path(x, y)
    np.testing.assert_allclose(s2, s, atol=1e-9)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)


def test_right_steer_turns_right_and_widens_the_outer_side():
    car = Car()
    path = Path(car, steer_deg=20)
    assert path.radius < 0
    x, y = path.from_path(40.0, 0.0)
    assert y < -5  # 右（y が負）へ曲がっていく
    lo, hi = path.bounds()
    half = car.width_cm / 2 + car.margin_cm
    assert lo == -half and hi > half  # 右に曲がると、前の左の角が外へ振れる
    # 半周より先は扱わない
    assert not path.inside(path.max_s + 1, 0) and path.inside(path.max_s - 1, 0)
    assert Grid(Camera(tilt_deg=-20), path).s[-1] <= path.max_s
    # まっすぐと、ほぼまっすぐはつながる
    a = Path(car, 0).from_path(30.0, 0.0)
    b = Path(car, 0.6).from_path(30.0, 0.0)
    assert a[0] == pytest.approx(float(b[0]), abs=0.5)


def test_grid_blind_zone_grows_when_looking_up():
    path = Path()
    low = Grid(Camera(tilt_deg=-20), path)
    level = Grid(Camera(tilt_deg=0), path)
    up = Grid(Camera(tilt_deg=12), path)
    assert low.blind_cm < level.blind_cm < up.blind_cm
    # 画面の下の端に映る床までが見えない（tilt 0° なら 10 cm × f / 下の端までの行数）
    cam = Camera(tilt_deg=0)
    bottom = cam.height_cm * cam.f / (cam.height - 1 - cam.center[1])
    assert level.blind_cm == pytest.approx(bottom - Car().cam_to_front_cm, abs=1.5)
    assert low.seen_cm == MAX_CM


def test_first_blocked_ignores_single_cells_but_keeps_a_lying_cable():
    s = np.arange(0.0, 50)
    w = np.zeros((50, 20), bool)
    w[10, 5] = True  # ノイズ 1 マス
    assert first_blocked(w, s) is None
    w[30, 2:18] = True  # 横に寝たケーブル（1 マスの厚さ）
    assert first_blocked(w, s) == 30
    counts = np.zeros((50, 20))
    counts[25, 7] = 6  # 立った脚の点が 1 マスに積み重なる
    assert first_blocked(counts, s, min_weight=4) == 25
    assert first_blocked(counts, s, min_weight=7) is None


def _floor_disparity(cam, a=150.0, b=0.2, wall_cm=300.0, box=None):
    """床（と奥の壁）の相対の視差 disp = a / z + b を作る。

    box=(x0, y0, 幅, 高さ) で、前の面が x = x0、横の中心が y0 の箱を置く。
    """
    v, u = np.mgrid[0 : cam.height, 0 : cam.width]
    ray = pixel_rays(cam, u, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(ray[..., 2] < -1e-9, cam.height_cm / -ray[..., 2], np.inf)
        z = np.minimum(z, wall_cm / ray[..., 0])  # 奥の壁
        if box is not None:
            x0, y0, width, height = box
            zb = x0 / ray[..., 0]  # 箱の前の面（x = x0）
            hit = np.abs(zb * ray[..., 1] - y0) < width / 2
            hit &= cam.height_cm + zb * ray[..., 2] < height
            z = np.where(hit & (zb > 0) & (zb < z), zb, z)
    return (a / z + b).astype(np.float32)


def test_depth_sees_free_floor_as_clear():
    cam = Camera(tilt_deg=-20).scaled(217, 161)
    r = clearance_from_disparity(_floor_disparity(cam), Camera(tilt_deg=-20), Path())
    assert r.clear_cm == MAX_CM and r.confidence > 0.9


@pytest.mark.parametrize("x0", [25.0, 45.0, 70.0])
def test_depth_finds_a_box_in_the_path(x0):
    full = Camera(tilt_deg=-20)
    cam = full.scaled(217, 161)
    path = Path()
    disp = _floor_disparity(cam, box=(x0, 0.0, 8.0, 10.0))
    r = clearance_from_disparity(disp, full, path)
    assert r.clear_cm == pytest.approx(x0 - path.car.cam_to_front_cm, abs=2)


def test_depth_ignores_a_box_beside_the_path_and_a_low_bump():
    full = Camera(tilt_deg=-20)
    cam = full.scaled(217, 161)
    path = Path()
    _, hi = path.bounds()
    # 道の左の外（左の端から 3 cm 先から）に置いた箱は数えない
    side = _floor_disparity(cam, box=(40.0, hi + 3 + 4, 8.0, 10.0))
    assert clearance_from_disparity(side, full, path).clear_cm == MAX_CM
    # 同じ箱を道の端にかかるように置くと数える
    edge = _floor_disparity(cam, box=(40.0, hi + 2, 8.0, 10.0))
    assert clearance_from_disparity(edge, full, path).clear_cm == pytest.approx(35, abs=2)
    # 高さ 1 cm の段差（MIN_HEIGHT_CM 未満）は障害物にしない
    low = _floor_disparity(cam, box=(40.0, 0.0, 30.0, 1.0))
    assert clearance_from_disparity(low, full, path).clear_cm == MAX_CM


def test_depth_returns_null_when_the_camera_looks_up():
    full = Camera(tilt_deg=12)
    r = clearance_from_disparity(_floor_disparity(full.scaled(217, 161)), full, Path())
    assert r.clear_cm is None and r.to_json() is None


def _floor_image(cam, rng, gradient=0.0):
    """木の床の色（とノイズ）の画像。gradient で奥ほど暗くする（影）。"""
    img = np.empty((cam.height, cam.width, 3), np.float32)
    img[:] = (60, 120, 170)  # BGR のオレンジ
    rows = np.linspace(1 - gradient, 1, cam.height)[:, None, None]
    img = img * rows + rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_floor_method_clear_floor_and_slow_shadow():
    cam = Camera(tilt_deg=-20)
    rng = np.random.default_rng(0)
    floor = FloorClearance()
    assert floor.estimate(_floor_image(cam, rng), cam).clear_cm == MAX_CM
    # 奥へ行くほどゆっくり暗くなる床（影）では止まらない
    assert floor.estimate(_floor_image(cam, rng, gradient=0.5), cam).clear_cm == MAX_CM


@pytest.mark.parametrize("x0", [20.0, 40.0, 60.0])
def test_floor_method_finds_a_lying_cable(x0):
    cam = Camera(tilt_deg=-20)
    img = _floor_image(cam, np.random.default_rng(1))
    (u0, u1), (v0, v1) = project(cam, [[x0, 30.0, 0.0], [x0, -30.0, 0.0]])
    thick = max(2, round(cam.f * 0.8 / x0))
    cv2.line(img, (int(u0), int(v0)), (int(u1), int(v1)), (25, 25, 25), thick)
    r = FloorClearance().estimate(img, cam)
    assert r.clear_cm == pytest.approx(x0 - Car().cam_to_front_cm, abs=3)


def test_floor_method_null_when_blind():
    cam = Camera(tilt_deg=12)
    img = _floor_image(cam, np.random.default_rng(2))
    assert FloorClearance().estimate(img, cam).clear_cm is None


def test_combine_takes_the_nearer_and_keeps_null():
    a = Clearance(60.0, 0.9, 7.0, "depth")
    b = Clearance(30.0, 0.8, 7.0, "floor")
    c = combine(a, b)
    assert (c.clear_cm, c.confidence, c.method) == (30.0, 0.8, "both")
    assert combine(Clearance(None, 0.0, 50.0, "depth"), b).clear_cm is None
    assert c.to_json() == {"cm": 30.0, "conf": 0.8, "blind_cm": 7.0}


def test_estimator_uses_tilt_pan_and_steer_from_the_header():
    from app.clearance import Estimator

    seen = []

    class _Depth:
        def estimate(self, image, cam, steer_deg):
            seen.append((cam.tilt_deg, cam.pan_deg, steer_deg, cam.height_cm, cam.width))
            return Clearance(50.0, 0.9, 13.0, "depth")

    est = Estimator.__new__(Estimator)  # 深度のモデルは読まない
    est.height_cm, est.depth, est.floor = 8.0, _Depth(), FloorClearance()
    cam = Camera(tilt_deg=-15, pan_deg=20)
    r = est.estimate(_floor_image(cam, np.random.default_rng(3)), -15.0, 20.0, 20.0)
    # 探索中の Pi は曲がる向きに pan を振る。通り道はヘッダーの pan で見る
    assert seen == [(-15.0 * TILT_GAIN + TILT_OFFSET_DEG, 20.0, 20.0, 8.0, 640)]
    assert r.method == "both"  # (a) と (b) の近い方
