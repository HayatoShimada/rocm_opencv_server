"""作った家（app/snoo_homes.py）・速いシミュレーター（app/snoo_gridsim.py）・学んだ選び方
の試験。どれも GPU も torch も使わない。"""

import math

import cv2
import numpy as np
import pytest

from app.roommap import FREE, OCCUPIED, UNKNOWN, Pose, next_best_view, next_best_views
from app.snoo_gridsim import GridSim
from app.snoo_homes import (
    CAR_CELLS,
    CLOSED,
    DOOR_MAX_M,
    DOOR_MIN_M,
    FLOOR,
    HIGH,
    HOME_M2,
    LOW,
    Home,
    make_home,
    render_home,
)
from app.snoo_mapper import MapParams, decide, plan_action

SEEDS = (0, 1, 2, 3, 7, 11)


@pytest.fixture(scope="module")
def homes():
    return {s: make_home(s, agent_t_s=300) for s in SEEDS}


def test_home_is_about_30_tsubo(homes):
    for h in homes.values():
        assert abs(h.footprint_m2 - HOME_M2) <= 0.1 * HOME_M2


def test_home_is_the_same_for_the_same_seed(homes):
    again = make_home(SEEDS[0], agent_t_s=300)
    h = homes[SEEDS[0]]
    assert (again.kind == h.kind).all() and again.start == h.start
    assert np.allclose(again.agents[0].traj, h.agents[0].traj)
    assert not (make_home(SEEDS[1], agent_t_s=300).kind.shape == h.kind.shape
                and (make_home(SEEDS[1], agent_t_s=300).kind == h.kind).all())  # fmt: skip


def test_start_is_free_and_every_door_is_reachable_by_the_car(homes):
    for h in homes.values():
        ix, iy = h.cell(h.start.x, h.start.y)
        assert h.kind[ix, iy] == FLOOR
        blocked = (h.kind == LOW) | (h.kind == CLOSED)
        dist = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 5)
        free = dist > CAR_CELLS
        assert free[ix, iy]
        _, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
        for d in h.doors:
            cx, cy = h.cell((d[0] + d[2]) / 2, (d[1] + d[3]) / 2)
            assert labels[cx, cy] == labels[ix, iy]


def test_doors_are_70_to_90_cm(homes):
    for h in homes.values():
        assert h.doors
        for x0, y0, x1, y1 in h.doors:
            width = max(x1 - x0, y1 - y0)
            assert DOOR_MIN_M - 1e-6 <= width <= DOOR_MAX_M + 1e-6


def test_reachable_floor_is_connected_and_agents_stay_on_it(homes):
    for h in homes.values():
        reach = h.reachable_floor()
        n, _ = cv2.connectedComponents(reach.astype(np.uint8), connectivity=4)
        assert n == 2  # 背景と 1 つの塊
        assert 30 < reach.sum() * h.res**2 < HOME_M2
        assert {a.name for a in h.agents} >= {"person", "snoo"}
        for a in h.agents:
            ix, iy = h.cell(a.traj[:, 0], a.traj[:, 1])
            assert reach[ix, iy].all()


def test_render_home_draws_a_picture(homes):
    img = render_home(homes[SEEDS[0]], px=2, t_s=10)
    assert img.ndim == 3 and img.shape[0] == homes[SEEDS[0]].n * 2


# --- 手で作った部屋（4 x 4 m） ---


def _room(items=()) -> Home:
    n = 120  # 6 m
    origin = -1.0
    kind = np.full((n, n), FLOOR, np.uint8)
    height = np.zeros((n, n), np.float32)
    kind[:2], kind[-2:], kind[:, :2], kind[:, -2:] = LOW, LOW, LOW, LOW
    height[kind == LOW] = 2.4

    def cell(x, y):
        return int(round((x - origin) / 0.05)), int(round((y - origin) / 0.05))

    for (x0, y0, x1, y1), k, h in items:
        a0, b0 = cell(x0, y0)
        a1, b1 = cell(x1, y1)
        kind[a0:a1, b0:b1] = k
        height[a0:a1, b0:b1] = h
    return Home(0, origin, kind, height, np.zeros((n, n), bool), [], [], [], [],
                Pose(0.0, 0.0, 0.0), "rect", 36.0)  # fmt: skip


def test_scan_sees_floor_in_front_and_not_behind_a_box():
    box = ((1.0, -0.2, 1.3, 0.2), LOW, 0.4)
    sim = GridSim(_room([box]), agents=False)
    sim.scan()
    s = sim.grid.state()
    g = sim.grid
    assert s[g.cell(0.6, 0.0)] == FREE
    assert s[g.cell(1.05, 0.0)] == OCCUPIED
    assert s[g.cell(1.8, 0.0)] == UNKNOWN  # 箱の陰
    assert s[g.cell(1.8, 0.8)] == FREE  # 陰の外
    assert s[g.cell(-0.8, 0.0)] == UNKNOWN  # 後ろは写らない
    # 超音波は前の端から箱まで（mark_sonar で箱の手前は床）
    d = sim.sonar(sim.pose, 0.0)
    assert d == pytest.approx(1.0 - 0.05, abs=0.03)


def test_low_cable_hides_only_a_little_and_a_table_top_hides_nothing():
    cable = ((0.6, -0.5, 0.65, 0.5), LOW, 0.03)
    table = ((1.2, 0.6, 1.8, 1.2), HIGH, 0.0)
    sim = GridSim(_room([cable, table]), agents=False)
    sim.scan()
    s, g = sim.grid.state(), sim.grid
    assert s[g.cell(0.62, 0.0)] == OCCUPIED
    # 高さ 3 cm のコードは、8 cm のカメラから 0.6 / (1 - 3/8) = 0.96 m までの床を隠す
    assert s[g.cell(0.85, 0.0)] == UNKNOWN
    assert s[g.cell(1.3, 0.0)] == FREE
    assert s[g.cell(1.5, 0.9)] == FREE  # 天板の下の床は見える


def test_drive_stops_before_a_wall_and_bumps_when_reversing_into_a_box():
    box = ((-0.6, -0.3, -0.4, 0.3), LOW, 0.3)
    sim = GridSim(_room([box]), agents=False)
    # 前の壁は x = 4.9 から。車の前の端はカメラの 5 cm 先、Pi は 5 cm 手前で止まる
    sim.pose = Pose(3.5, 0.0, 0.0)
    result, moved = sim.drive(0.0, 12, 2.0, reverse=False)
    assert result == "stop" and sim.bumps == 0
    assert moved == pytest.approx(4.9 - 3.55 - 0.05, abs=0.04)
    sim.pose = Pose(0.0, 0.0, 0.0)
    result, moved = sim.drive(0.0, 8, 0.5, reverse=True)
    assert result == "bump" and sim.bumps == 1
    # 後ろの端（カメラの 21 cm 後ろ）が箱（x = -0.4）に当たる手前
    assert moved == pytest.approx(0.4 - 0.21, abs=0.03)


def test_grid_sim_is_deterministic():
    a = GridSim(make_home(5, agent_t_s=400)).run(240)
    b = GridSim(make_home(5, agent_t_s=400)).run(240)
    assert a.timeline == b.timeline and a.kinds == b.kinds
    assert a.covered_m2 > 1.0


# --- 候補と点数 ---


def _scanned(seed=3):
    sim = GridSim(make_home(seed, agent_t_s=200))
    sim.scan()
    return sim


def test_top1_of_next_best_views_is_next_best_view():
    sim = _scanned()
    one = next_best_view(sim.grid, sim.pose, segments=True, max_step_m=0.6)
    many = next_best_views(sim.grid, sim.pose, 4, segments=True, max_step_m=0.6)
    assert one is not None and many[0].target == one.target and many[0].segments == one.segments
    for p in many[1:]:
        assert math.hypot(p.target.x - one.target.x, p.target.y - one.target.y) >= 0.4


def test_decide_follows_next_best_view_without_randomness():
    sim = _scanned()
    params = MapParams(camera_height_m=0.08, segments=True, reverse=True, random_p=0.0,
                       plan_turns=False, max_step_m=0.6)  # fmt: skip
    action, plan = decide(sim.grid, sim.stations, params, [0.08])
    ref = next_best_view(sim.grid, sim.pose, segments=True, max_step_m=0.6)
    assert plan is not None and plan.target == ref.target
    assert action == plan_action(ref, sim.pose, params)
