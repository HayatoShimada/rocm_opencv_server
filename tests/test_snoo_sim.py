import math

import numpy as np
import pytest

pytest.importorskip("mujoco")  # uv sync --group sim

from app.roommap import FREE, WHEELBASE_M, Pose, arc, curvature  # noqa: E402
from app.snoo_mapper import MapParams  # noqa: E402
from app.snoo_sim import (  # noqa: E402
    CAM_H_M,
    CAR_FRONT_M,
    CAR_REAR_M,
    CAR_WIDTH_M,
    CarState,
    Sim,
    build_mjcf,
    footprint,
    merge_cells,
    rect_hits,
    synthetic_room,
)


def test_merge_cells_covers_exactly_the_mask_with_fewer_boxes():
    rng = np.random.default_rng(0)
    mask = np.zeros((40, 40), bool)
    mask[5:20, 3:30] = True  # 大きな長方形
    mask[25:30, 10:12] = True
    mask |= rng.random((40, 40)) > 0.97  # ばらばらの点
    rects = merge_cells(mask)
    covered = np.zeros_like(mask, int)
    for i0, i1, j0, j1 in rects:
        covered[i0:i1, j0:j1] += 1
    assert (covered == mask).all()  # 重ならず、はみ出さず、取りこぼさない
    assert len(rects) < mask.sum() / 5  # セルごとの箱よりずっと少ない


def test_bicycle_model_goes_straight_and_turns_on_the_right_radius():
    x, y, yaw = arc(0.0, 0.0, 0.0, 0.0, 0.40)
    assert (x, y, yaw) == pytest.approx((0.40, 0.0, 0.0))
    # 右に 30° 切ると、半径 ホイールベース / tan 30° の円を右回り（yaw が減る）
    radius = WHEELBASE_M / math.tan(math.radians(30))
    assert curvature(30.0) == pytest.approx(-1 / radius)
    x, y, yaw = arc(0.0, 0.0, 0.0, 30.0, math.pi * radius)  # 半周
    assert (x, y) == pytest.approx((0.0, -2 * radius), abs=1e-9)
    assert math.cos(yaw) == pytest.approx(-1)
    x, y, _ = arc(0.0, 0.0, 0.0, -30.0, math.pi / 2 * radius)  # 左に 4 分の 1 周
    assert (x, y) == pytest.approx((radius, radius), abs=1e-9)


def test_camera_pose_round_trips_and_stays_in_minus_pi_to_pi():
    pose = Pose(0.3, -0.2, math.radians(170))
    car = CarState.from_camera(pose)
    back = car.camera_pose()
    assert (back.x, back.y, back.yaw) == pytest.approx((pose.x, pose.y, pose.yaw))
    car.yaw += 2 * math.pi  # 何周しても同じ向き
    assert -math.pi <= car.camera_pose().yaw <= math.pi


def test_footprint_and_rect_hits():
    corners = footprint(0.0, 0.0, 0.0)
    assert corners[:, 0].min() == pytest.approx(CAR_REAR_M)
    assert corners[:, 0].max() == pytest.approx(CAR_FRONT_M)
    assert corners[:, 1].max() - corners[:, 1].min() == pytest.approx(CAR_WIDTH_M)
    boxes = np.array([[0.30, -0.1, 0.40, 0.1]])  # 車の前の端の 9 cm 先
    assert not rect_hits(corners, boxes)
    assert rect_hits(footprint(0.15, 0.0, 0.0), boxes)
    # 斜めの車は、外接する長方形どうしが重なっても、本当に重ならなければ当たらない
    tilted = footprint(0.0, 0.0, math.radians(45))
    assert not rect_hits(tilted, np.array([[0.17, 0.17, 0.30, 0.30]]))


def test_synthetic_room_builds_a_mujoco_model():
    import mujoco

    model = mujoco.MjModel.from_xml_string(build_mjcf(synthetic_room()))
    assert model.camera("car_cam").id >= 0
    for name in ("car", "snoo", "pan", "tilt", "steer_fl", "steer_fr"):
        assert model.joint(name).id >= 0


@pytest.fixture(scope="module")
def sim():
    return Sim(synthetic_room(), rays=(64, 48), seed=0)


def test_scan_marks_the_floor_around_the_car(sim):
    sim.scan()
    state = sim.grid.state()
    assert len(sim.stations) == 1
    assert (state == FREE).sum() * sim.grid.res**2 > 0.5  # 少なくとも 0.5 m² の床が分かる
    pose = sim.car.camera_pose()
    ahead = sim.grid.cell(pose.x + 0.5 * math.cos(pose.yaw), pose.y + 0.5 * math.sin(pose.yaw))
    assert state[ahead] == FREE  # 50 cm 先の床は見えている


def test_explore_runs_a_few_cycles_without_bumping():
    s = Sim(synthetic_room(), rays=(64, 48), seed=1)
    summary = s.explore(3, plan_frames=1)
    assert 1 <= summary.stations <= 3
    assert summary.bumps == 0
    assert 0 < summary.coverage <= 1


def test_explore_with_sonar_segments_and_retrace():
    """地図のサーバーと同じ決め方（超音波・曲がる → まっすぐ・来た道を戻る）でも回る。"""
    params = MapParams(camera_height_m=CAM_H_M, segments=True)
    s = Sim(synthetic_room(), rays=(64, 48), params=params, seed=1, sonar_map=True)
    summary = s.explore(3, plan_frames=1, backtrack=2)
    assert summary.bumps == 0
    assert s.driven  # 前へ走った道を覚えている
    assert all(m > 0 for _, m in s.driven)
    assert s.action.get("action") in ("move", "done")
    if s.action.get("action") == "move":
        assert "segments" in s.action


def test_explore_clock_follows_the_real_pace():
    """1 周の時間は、首振り（10 コマ × 1.44 秒）+ 地図のサーバー（11 秒）+ 走る時間。"""
    from app.snoo_sim import REAL_PACE

    s = Sim(synthetic_room(), rays=(64, 48), seed=1, pace=REAL_PACE)
    summary = s.explore(1)
    scan_and_server = 10 * 1.44 + 11.0
    assert summary.stations == 1
    drive_s = s.distance_m / (12 * 0.017)
    assert summary.t_s == pytest.approx(scan_and_server + drive_s, abs=0.2)
    assert summary.covered_m2 > 0.5
    assert summary.m2_per_min == pytest.approx(summary.covered_m2 / (summary.t_s / 60), rel=0.01)
    # 時間で止める
    s = Sim(synthetic_room(), rays=(64, 48), seed=1, pace=REAL_PACE)
    assert s.explore(10, time_budget_s=30).reason == "time"
    assert len(s.stations) == 2


def test_window_keeps_only_the_stations_near_the_car():
    """地図のサーバーの窓をまねると、窓の外のステーションで見た床は地図から消える。"""

    def two_stations(window):
        s = Sim(synthetic_room(), rays=(64, 48), seed=0, window=window)
        s.scan()
        s.car = CarState.from_camera(Pose(1.2, 0.3, math.pi / 2))
        s.sync()
        s.scan()
        return int((s.grid.state() == FREE).sum())

    assert two_stations(1) < two_stations(None)
    assert two_stations(2) == two_stations(None)
