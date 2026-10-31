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


def _view_gain_by_steps(grid, state, pose, rays=61):
    """前の view_gain（1 マスずつ辿る）。numpy の版と数が同じかを見る。"""
    from app.roommap import HALF_VIEW_DEG, VIEW_RANGE_M

    seen = set()
    half = math.radians(HALF_VIEW_DEG)
    for a in np.linspace(pose.yaw - half, pose.yaw + half, rays):
        for k in range(1, int(VIEW_RANGE_M / (grid.res * 0.5)) + 1):
            d = k * grid.res * 0.5
            ix, iy = grid.cell(pose.x + d * math.cos(a), pose.y + d * math.sin(a))
            if not grid.inside(ix, iy) or state[ix, iy] == OCCUPIED:
                break
            if state[ix, iy] == UNKNOWN:
                seen.add((int(ix), int(iy)))
    return len(seen)


def test_view_gain_matches_walking_the_rays_one_cell_at_a_time():
    rng = np.random.default_rng(0)
    g = Grid(size_m=4.0)
    for _ in range(20):
        s = rng.choice([UNKNOWN, FREE, OCCUPIED], g.logodds.shape, p=[0.5, 0.45, 0.05])
        s = s.astype(np.uint8)
        p = Pose(*rng.uniform(-2.2, 2.2, 2), float(rng.uniform(-4, 4)))  # 格子の外へも向く
        assert view_gain(g, s, p) == _view_gain_by_steps(g, s, p)


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


def test_sonar_is_weak_evidence_of_an_obstacle_on_the_arc_only():
    """超音波は地図の弱い証拠だけ：1 回では障害物にならず、床にもしない。"""
    from app.roommap import SONAR_HIT

    g = Grid(size_m=8.0)
    mark_sonar(g, Pose(0.0, 0.0, 0.0), 0.52)
    front = SONAR_AHEAD_M + 0.52
    assert g.logodds[g.cell(front, 0.0)] == pytest.approx(SONAR_HIT)
    assert g.state()[g.cell(front, 0.0)] == UNKNOWN  # 1 回では障害物にしない
    assert g.logodds[g.cell(SONAR_AHEAD_M + 0.52 * math.cos(math.radians(14)), 0.12)] > 0
    assert g.logodds[g.cell(0.30, 0.0)] == 0  # 手前を床にしない
    for _ in range(2):
        mark_sonar(g, Pose(0.0, 0.0, 0.0), 0.52)
    assert g.state()[g.cell(front, 0.0)] == OCCUPIED  # 何度も同じ所なら障害物
    # 向きはステーションに合わせる
    g = Grid(size_m=8.0)
    mark_sonar(g, Pose(1.0, -1.0, math.pi / 2), 0.42)
    assert g.logodds[g.cell(1.0, -1.0 + SONAR_AHEAD_M + 0.42)] > 0


def test_sonar_drops_none_minus_one_far_and_tiny_readings():
    g = Grid(size_m=8.0)
    for v in (None, -0.01, 0.0, 0.01, 2.5, 3.5, float("nan")):
        mark_sonar(g, Pose(0.0, 0.0, 0.0), v)
    assert not g.logodds.any()


def test_sonar_does_not_override_the_camera_floor():
    g = Grid(size_m=8.0)
    g.mark_free(np.array([0.60]), np.array([0.0]))
    mark_sonar(g, Pose(0.0, 0.0, 0.0), 0.60 - SONAR_AHEAD_M)
    assert g.state()[g.cell(0.60, 0.0)] == FREE


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


def test_car_next_to_furniture_can_still_leave_away_from_it():
    """左前 14 cm に障害物があっても、居る所はふさがらず、離れる向きへは行ける
    （3 回目の実走の最後は、ここで周りが全部ふさがって unreachable だった）。
    """
    from app.roommap import blocked_cells

    g = _room(seen_until_x=1.5)
    g.mark_occupied(np.array([0.10, 0.10]), np.array([0.10, 0.15]))
    pose = Pose(0, 0, 0)
    blocked = blocked_cells(g, g.state(), pose)
    assert not blocked[g.cell(0.0, 0.0)]
    assert blocked[g.cell(0.10, 0.05)]  # 障害物のすぐ横には入らない
    plan = next_best_view(g, pose, segments=True)
    assert plan is not None and plan.segments


def test_mark_driven_makes_the_floor_under_the_car_free_but_keeps_obstacles():
    from app.roommap import CAR_REAR_M, mark_driven

    g = Grid(size_m=4.0)
    g.mark_occupied(np.array([0.30]), np.array([0.0]))
    mark_driven(g, Pose(0, 0, 0), [(0.0, 0.40)])
    s = g.state()
    assert s[g.cell(0.0, 0.0)] == FREE and s[g.cell(0.41, 0.0)] == FREE  # 前の端 +5 cm まで
    assert s[g.cell(0.30, 0.0)] == OCCUPIED  # 見た障害物は消さない
    assert s[g.cell(0.0, 0.12)] == UNKNOWN  # 車の幅の外
    # 後ろへ下がったら、車の後ろが床になる（カメラの真下から CAM_X_M + 後ろの端まで）
    g = Grid(size_m=4.0)
    mark_driven(g, Pose(0, 0, 0), [(0.0, 0.20)], reverse=True)
    assert g.state()[g.cell(-0.16 + CAR_REAR_M - 0.15, 0.0)] == FREE


def test_long_step_only_on_a_straight_segment_over_seen_floor():
    from app.roommap import LONG_STEP_M

    g, blocked = _open_floor()
    pieces = turn_then_straight(g, blocked, Pose(0, 0, 0), Pose(1.5, 0, 0), LONG_STEP_M)
    assert pieces == [(0.0, pytest.approx(LONG_STEP_M))]
    # 曲がるだけで終わる 1 歩は 40 cm まで
    turn = turn_then_straight(g, blocked, Pose(0, 0, 0), Pose(0.1, -0.5, math.pi), LONG_STEP_M)
    assert sum(m for _, m in turn) <= MAX_STEP_M + 1e-9
    # 床が見えている所までしか延ばさない（45 cm 先から見ていない。車の真ん中はカメラの 8 cm 後ろ）
    g, blocked = _open_floor()
    g.logodds[g.cell(0.45, 0.0)[0] :, :] = 0
    blocked = g.state() != FREE
    short = turn_then_straight(g, blocked, Pose(0, 0, 0), Pose(1.5, 0, 0), LONG_STEP_M)
    assert MAX_STEP_M <= sum(m for _, m in short) < LONG_STEP_M


def test_first_step_keeps_the_real_car_off_obstacles():
    """Pi は後ろの車軸を中心に曲がる。カメラの真下から引いた弧では車体が障害物に入っていた。"""
    from app.roommap import blocked_cells, first_step

    g, _ = _open_floor()
    for _ in range(3):
        g.mark_occupied(np.array([0.0]), np.array([0.45]))  # 左前の 1 セル
    pose = Pose(0.0, 0.0, 0.0)
    blocked = blocked_cells(g, g.state(), pose)

    def hits(steer, run):
        for _, r in drive_poses(pose, [(steer, run)])[1:]:
            c = g.cell(r.x + CAR_CENTER_M * math.cos(r.yaw), r.y + CAR_CENTER_M * math.sin(r.yaw))
            if blocked[c]:
                return True
        return False

    path = sample_path(pose, "L", (math.pi / 2,), r=0.26)  # 左へいっぱいに曲がる道
    steer, run = first_step(g, blocked, pose, path, 0.4)
    assert run == 0 or not hits(steer, run)


def test_big_frontiers_drop_tiny_holes_in_the_floor():
    from app.roommap import big_frontiers

    g = _room()  # x < 0.8 が床の大きな境目
    for _ in range(3):  # 床の中に 1 セルの見ていない穴（VGGT の点の抜け）を作る
        g.logodds[g.cell(-1.0, -1.0)] = 0.0
    s = g.state()
    assert frontiers(s)[g.cell(-1.05, -1.0)] or frontiers(s)[g.cell(-0.95, -1.0)]
    big = big_frontiers(s)
    ix, iy = np.nonzero(big)
    x, _ = g.center(ix, iy)
    assert len(ix) and np.all(np.abs(x - 0.775) < 0.06)  # 大きな境目だけが残る


def test_clip_pieces_keeps_the_front_edge_off_a_wall_near_the_start():
    """出発点のまわりは障害物を 10 cm しか広げないので、車の真ん中だけ見ると前の端が入った。"""
    from app.roommap import CAR_FRONT_M, blocked_cells, clip_pieces, rear_axle

    g, _ = _open_floor()
    wall = np.linspace(-0.3, 0.3, 40)
    for _ in range(4):
        g.mark_occupied(np.full_like(wall, 0.27), wall)  # カメラの 27 cm 前に壁
    pose = Pose(0.0, 0.0, 0.0)
    pieces = clip_pieces(g, blocked_cells(g, g.state(), pose), pose, [(0.0, 0.40)])
    run = sum(m for _, m in pieces)
    front = rear_axle(pose).x + run + CAR_FRONT_M  # 走った後の前の端
    assert front < 0.27 - 0.025  # 壁のセルの手前の縁より前に出ない


def test_blocked_ahead_is_weak_and_keeps_the_seen_floor():
    """Pi が止まった所の先は弱い証拠だけ：1 回では障害物にならず、2 回で障害物。床と見たセルは
    変えない（前を横切った猫で障害物を作らない）。"""
    from app.roommap import FREE_HIT, mark_blocked_ahead

    g = Grid(size_m=4.0)
    cam = Pose(0.0, 0.0, 0.0)  # 前の端は x = 0.05。その先 10 cm
    seen = g.cell(0.125, 0.075)
    g.logodds[seen] = FREE_HIT
    ahead = g.cell(0.075, 0.025)
    mark_blocked_ahead(g, cam, 0.0)
    assert g.state()[ahead] == UNKNOWN and g.logodds[ahead] > 0
    assert g.logodds[seen] == pytest.approx(FREE_HIT)
    mark_blocked_ahead(g, cam, 0.0)
    assert g.state()[ahead] == OCCUPIED and g.state()[seen] == FREE
