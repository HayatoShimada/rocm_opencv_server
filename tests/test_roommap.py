import math

import numpy as np
import pytest

from app.roommap import (
    CAR_CENTER_M,
    FREE,
    MAX_STEP_M,
    OCCUPIED,
    SONAR_AHEAD_M,
    STRAIGHT_STEER_DEG,
    UNKNOWN,
    WHEELBASE_M,
    Grid,
    Pose,
    camera_of,
    drive_poses,
    dubins,
    frontiers,
    inflate,
    mark_sonar,
    next_best_view,
    sample_path,
    turn_then_straight,
    view_gain,
)


def _room(seen_until_x: float = 0.8) -> Grid:
    """4 m 四方の部屋（壁あり）。車の周りの x < seen_until_x だけ床が見えている。"""
    g = Grid(size_m=6.0)
    xs = np.arange(-2.0, 2.0, 0.025)
    gx, gy = np.meshgrid(xs, xs)
    seen = gx < seen_until_x
    g.mark_free(gx[seen], gy[seen])
    wall = np.concatenate([np.full_like(xs, -2.0), np.full_like(xs, 2.0)])
    g.mark_occupied(np.concatenate([wall, np.tile(xs, 2)]), np.concatenate([np.tile(xs, 2), wall]))
    return g


def test_grid_prefers_obstacle_when_floor_and_obstacle_share_a_cell():
    g = Grid(size_m=2.0)
    g.mark_free(np.array([0.01]), np.array([0.01]))
    g.mark_occupied(np.array([0.02]), np.array([0.02]))
    g.mark_free(np.array([0.51]), np.array([0.01]))
    s = g.state()
    assert s[g.cell(0.01, 0.01)] == OCCUPIED
    assert s[g.cell(0.51, 0.01)] == FREE
    assert s[g.cell(-0.5, -0.5)] == UNKNOWN


def test_frontier_is_the_edge_of_the_seen_floor():
    g = _room()
    f = frontiers(g.state())
    ix, iy = np.nonzero(f)
    x, _ = g.center(ix, iy)
    assert len(ix) and np.all(np.abs(x - 0.775) < 0.06)  # x = 0.8 の手前の列だけ


def test_view_gain_counts_unknown_only_in_the_viewing_direction():
    g = _room()
    s = g.state()
    assert view_gain(g, s, Pose(0.5, 0, 0)) > 100  # 見ていない +x 側を向く
    assert view_gain(g, s, Pose(0.5, 0, math.pi)) == 0  # 見た側を向く


@pytest.mark.parametrize(
    ("goal", "length"),
    [(Pose(1.0, 0, 0), 1.0), (Pose(0, 0.44, math.pi), math.pi * 0.22)],  # まっすぐ・半円で U ターン
)
def test_dubins_lengths(goal, length):
    route = dubins(Pose(0, 0, 0), goal, r=0.22)
    assert route[0] == pytest.approx(length, abs=1e-6)
    end = sample_path(Pose(0, 0, 0), route[1], route[2], r=0.22)[-1]
    assert (end.x, end.y) == pytest.approx((goal.x, goal.y), abs=1e-6)
    assert math.cos(end.yaw - goal.yaw) == pytest.approx(1, abs=1e-6)


def test_next_best_view_heads_toward_the_unseen_side_without_leaving_the_floor():
    g = _room()
    plan = next_best_view(g, Pose(0, 0, 0))
    assert plan is not None
    assert plan.target.x > 0.2  # 見ていない +x 側へ
    assert 0 < plan.step_m <= 0.4
    s = g.state()
    for p in plan.path:  # 見えている床の上だけを通る
        assert s[g.cell(p.x, p.y)] == FREE


def test_next_best_view_is_none_when_everything_is_seen():
    g = _room(seen_until_x=3.0)
    assert next_best_view(g, Pose(0, 0, 0)) is None


def test_first_step_runs_a_useful_distance_toward_the_path():
    g = _room()
    plan = next_best_view(g, Pose(0, 0, 0))
    # 数 cm で止まらず、経路の先へまとめて進む
    assert plan.step_m >= min(0.2, plan.length_m)
    # 左（+y）に曲がる経路ならハンドルは左（−）、右なら右（+）
    if plan.target.y > 0.1:
        assert plan.steer_deg < 0
    elif plan.target.y < -0.1:
        assert plan.steer_deg > 0


def test_first_step_goes_straight_for_a_straight_path():
    g = _room()
    path = sample_path(Pose(0, 0, 0), "S", (1.0,), r=1.0)
    from app.roommap import first_step

    blocked = g.state() != FREE
    steer, step = first_step(g, blocked, Pose(0, 0, 0), path, 1.0)
    assert steer == pytest.approx(0, abs=0.5)
    assert step == pytest.approx(0.4)


def test_sonar_marks_the_arc_as_obstacle_and_the_cone_before_it_as_floor():
    g = Grid(size_m=8.0)
    mark_sonar(g, Pose(0.0, 0.0, 0.0), 0.52)
    s = g.state()
    front = SONAR_AHEAD_M + 0.52
    assert s[g.cell(front, 0.0)] == OCCUPIED
    # 弧は円すいの幅いっぱい（±15°）
    assert s[g.cell(SONAR_AHEAD_M + 0.52 * math.cos(math.radians(14)), 0.12)] == OCCUPIED
    assert s[g.cell(0.30, 0.0)] == FREE  # 手前は床
    assert s[g.cell(front - 0.06, 0.0)] != FREE  # 読みの手前の余裕は床にしない
    assert s[g.cell(0.30, 0.25)] == UNKNOWN  # 円すいの外
    assert s[g.cell(-0.30, 0.0)] == UNKNOWN  # 後ろ
    assert s[g.cell(front + 0.15, 0.0)] == UNKNOWN  # 障害物の向こう


def test_sonar_follows_the_station_pose():
    g = Grid(size_m=8.0)
    mark_sonar(g, Pose(1.0, -1.0, math.pi / 2), 0.42)  # +y を向いている
    s = g.state()
    assert s[g.cell(1.0, -1.0 + SONAR_AHEAD_M + 0.42)] == OCCUPIED
    assert s[g.cell(1.0, -0.7)] == FREE
    assert s[g.cell(1.3, -1.0)] == UNKNOWN


def test_sonar_beyond_its_range_only_marks_floor_and_none_does_nothing():
    g = Grid(size_m=8.0)
    mark_sonar(g, Pose(0.0, 0.0, 0.0), None)
    assert not g.logodds.any()
    mark_sonar(g, Pose(0.0, 0.0, 0.0), 3.5)
    s = g.state()
    assert not (s == OCCUPIED).any()
    assert s[g.cell(2.5, 0.0)] == FREE
    assert s[g.cell(3.2, 0.0)] == UNKNOWN


def test_sonar_floor_does_not_erase_an_obstacle_the_camera_saw():
    """超音波は 4.5 cm の高さを見るので、それより低い物を床と言って消さない。"""
    g = Grid(size_m=8.0)
    g.mark_occupied(np.array([0.30]), np.array([0.0]))
    for _ in range(5):
        mark_sonar(g, Pose(0.0, 0.0, 0.0), 1.0)
    assert g.state()[g.cell(0.30, 0.0)] == OCCUPIED


def _open_floor(size_m: float = 6.0) -> tuple[Grid, np.ndarray]:
    g = Grid(size_m=size_m)
    xs = np.arange(-2.0, 2.0, 0.025)
    gx, gy = np.meshgrid(xs, xs)
    g.mark_free(gx.ravel(), gy.ravel())
    return g, g.state() != FREE


def test_segments_turn_only_as_much_as_needed_then_go_straight():
    g, blocked = _open_floor()
    # 右前（-y）の、右を向いた位置へ：右に最小半径で曲がり、まっすぐ
    pieces = turn_then_straight(g, blocked, Pose(0, 0, 0), Pose(0.9, -0.6, -math.pi / 2))
    assert len(pieces) == 2
    (turn, turn_m), (straight, straight_m) = pieces
    assert turn == pytest.approx(math.degrees(math.atan(WHEELBASE_M / 0.26)))  # 右が +
    assert abs(straight) < STRAIGHT_STEER_DEG
    assert sum(m for _, m in pieces) == pytest.approx(MAX_STEP_M)
    # 曲がった後は、次の位置のほうへ向いている（そのまま直線で近づく）
    end = camera_of(drive_poses(Pose(0, 0, 0), pieces[:1])[-1][1])
    to_goal = math.atan2(-0.6 - end.y, 0.9 - end.x)
    assert 0.05 < turn_m < 0.30
    assert abs(math.remainder(end.yaw - to_goal, 2 * math.pi)) < math.radians(25)


def test_segments_go_straight_when_the_target_is_ahead():
    g, blocked = _open_floor()
    pieces = turn_then_straight(g, blocked, Pose(0, 0, 0), Pose(1.0, 0, 0))
    assert pieces == [(0.0, pytest.approx(MAX_STEP_M))]


def test_segments_stop_before_an_obstacle():
    g, _ = _open_floor()
    wall = np.arange(-1.0, 1.0, 0.02)
    g.mark_occupied(np.full_like(wall, 0.55), wall)
    for _ in range(3):  # 床と重なったセルも障害物にする
        g.mark_occupied(np.full_like(wall, 0.55), wall)
    s = g.state()
    blocked = inflate(s == OCCUPIED, 4) | (s != FREE)
    pieces = turn_then_straight(g, blocked, Pose(0, 0, 0), Pose(1.2, 0, 0))
    total = sum(m for _, m in pieces)
    assert 0.05 <= total < MAX_STEP_M
    for _, r in drive_poses(Pose(0, 0, 0), pieces):  # 車の真ん中は広げた障害物に入らない
        c = g.cell(r.x + CAR_CENTER_M * math.cos(r.yaw), r.y + CAR_CENTER_M * math.sin(r.yaw))
        assert not blocked[c]


def test_next_best_view_with_segments_keeps_the_first_segment_on_top():
    g = _room()
    plan = next_best_view(g, Pose(0, 0, 0), segments=True)
    assert plan is not None and plan.segments
    assert (plan.steer_deg, plan.step_m) == plan.segments[0]
    assert 0.05 <= sum(m for _, m in plan.segments) <= MAX_STEP_M + 1e-9
    assert len(plan.segments) <= 2
    if len(plan.segments) == 2:
        assert abs(plan.segments[1][0]) < STRAIGHT_STEER_DEG
    # segments を頼まなければ今までどおり（区間は無い）
    old = next_best_view(g, Pose(0, 0, 0))
    assert old.segments == [] and old.target == plan.target
