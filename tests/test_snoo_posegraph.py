"""app/snoo_posegraph.py（位置のグラフ）を、作り物の道筋で試す。GPU は使わない。"""

import json
import math

import numpy as np
import pytest

from app import snoo_posegraph as pg
from app.roommap import Pose
from app.snoo_mapper import StationResult, to_frame


def _close(a: Pose, b: Pose, m=1e-6, rad=1e-6):
    assert math.hypot(a.x - b.x, a.y - b.y) < m
    assert abs(pg.wrap(a.yaw - b.yaw)) < rad


def test_relative_and_inverse_round_trip():
    a, b = Pose(0.3, -0.2, 0.7), Pose(-0.5, 1.1, -2.9)
    _close(to_frame(a, pg.relative(a, b)), b)
    _close(to_frame(a, pg.inverse(a)), Pose(0, 0, 0))


def test_odom_pose_straight_and_arc():
    _close(pg.odom_pose([(0.0, 0.3)]), Pose(0.3, 0.0, 0.0))
    # 右に切って後ろの車軸が 1/4 周すると、向きは -90°（右回り）
    k = math.tan(math.radians(20)) / 0.14
    p = pg.odom_pose([(20.0, (math.pi / 2) / k)])
    assert p.yaw == pytest.approx(-math.pi / 2, abs=1e-3)
    assert p.y < 0
    back = pg.odom_pose([(0.0, 0.2)], reverse=True)
    _close(back, Pose(-0.2, 0.0, 0.0))


def _truth(n=8):
    """四角を描いて戻ってくる道筋（ステーション n 個）。"""
    out = [Pose(0.0, 0.0, 0.0)]
    for _ in range(1, n):
        out.append(to_frame(out[-1], Pose(0.3, 0.0, math.radians(360 / n))))
    return out


def _edge(t, a, b, kind, sig_m=0.02, sig_deg=2.0, noise=None):
    z = pg.relative(t[a], t[b])
    if noise is not None:
        z = Pose(z.x + noise[0], z.y + noise[1], z.yaw + noise[2])
    return pg.Edge(a, b, z, sig_m, math.radians(sig_deg), kind, f"{kind}:{a}-{b}")


def test_optimize_recovers_poses_and_rejects_a_broken_window():
    t = _truth()
    rng = np.random.default_rng(0)
    edges = []
    for i in range(len(t) - 1):  # 走った辺：長さ 10 % 短く、向きもずれる（Pi の見積もり）
        z = pg.relative(t[i], t[i + 1])
        z = Pose(0.9 * z.x, 0.9 * z.y + 0.02, z.yaw + math.radians(5))
        edges.append(pg.Edge(i, i + 1, z, 0.08, math.radians(10), "odom"))
    for i in range(len(t)):  # 窓の辺：2 つ先まで、少しの雑音
        for j in (i + 1, i + 2):
            if j < len(t):
                edges.append(_edge(t, i, j, "window", noise=rng.normal(0, 0.005, 3)))
    edges.append(_edge(t, 7, 0, "loop"))
    # 崩れた窓：ステーション 4 を 40 cm・60° 外す
    edges.append(_edge(t, 3, 4, "window", noise=(0.4, -0.3, 1.0)))
    sol = pg.optimize(edges, fixed=0)
    for i, p in enumerate(t):
        _close(sol.poses[i], p, m=0.03, rad=math.radians(3))
    bad = [e for e in sol.edges if e.outlier]
    assert [(e.a, e.b) for e in bad] == [(3, 4)]
    assert sol.seen == set(range(len(t)))


def test_odometry_alone_keeps_every_station_and_gaps_get_the_origin():
    t = _truth(4)
    edges = [_edge(t, i, i + 1, "odom", 0.05, 8.0) for i in range(3)]
    sol = pg.optimize(edges, fixed=0)
    for i, p in enumerate(t):
        _close(sol.poses[i], p, m=1e-4, rad=1e-4)
    assert sol.seen == set()


def _result(name, pose, spread=0.01, height=0.08):
    return StationResult(name, pose, height, spread, 10)


def test_window_edges_skip_bad_stations_and_weigh_spread():
    w = pg.Window(
        "live:003",
        [1, 2, 3],
        [
            _result("001", Pose(0, 0, 0)),
            _result("002", Pose(0.3, 0, 0.1), spread=0.15),  # 首振りで 15 cm 散った
            _result("003", Pose(0.6, 0.1, 0.2), spread=0.04),
        ],
        [0.08, 0.08, 0.08],
    )
    es = pg.window_edges(w)
    assert [(e.a, e.b) for e in es] == [(1, 3)]
    e = es[0]
    _close(e.z, pg.relative(Pose(0, 0, 0), Pose(0.6, 0.1, 0.2)))
    assert e.sigma_m > pg.VISION_SIGMA_M + 0.04  # 離れている分とばらつきの分
    # 高さが見込みと大きく違う（大きさが崩れた）ステーションも使わない
    w.results[2] = _result("003", Pose(0.6, 0.1, 0.2), height=0.02)
    assert pg.window_edges(w) == []


def test_observation_places_the_local_cloud_at_the_pose():
    xy = np.array([[1.0, 0.0]] * 5, np.float32)
    cloud = pg.LocalCloud(xy, xy[:0], xy[:0], 0.01, "x")
    o = pg.observation(cloud, Pose(1.0, 2.0, math.pi / 2))
    assert o.free.shape == (1, 2)
    assert o.free[0] == pytest.approx([1.0, 3.0], abs=0.05)
    assert len(o.occupied) == 0


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    pairs = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 1)]
    edges = [pg.Edge(a, b, Pose(*rng.normal(0, 0.3, 3)), 0.05, 0.1, "window") for a, b in pairs]
    prob = pg._Problem([0, 1, 2, 3], 0, edges)
    x = rng.normal(0, 0.5, 9)
    J = prob.jacobian(x).toarray()
    N = np.zeros_like(J)
    for k in range(9):
        d = np.zeros(9)
        d[k] = 1e-6
        N[:, k] = (prob.residuals(x + d) - prob.residuals(x - d)) / 2e-6
    assert np.abs(J - N).max() < 1e-6


def test_window_edges_ignore_non_stations():
    """走りながらのコマ（番号 -1）は、ばらつきが小さくても辺にしない。"""
    w = pg.Window(
        "pair",
        [3, -1, 4],
        [_result("003", Pose(0, 0, 0)), _result("b", Pose(0.1, 0, 0), height=None),
         _result("004", Pose(0.2, 0, 0))],
        [0.08, None, 0.08],
    )  # fmt: skip
    assert [(e.a, e.b) for e in pg.window_edges(w)] == [(3, 4)]


def test_window_edges_bridge_a_gap_in_the_odometry():
    """prev_step が無い所（4 → 5）を窓の辺でつなぎ、その先も走った辺でたどる。"""
    t = _truth(8)
    edges = [_edge(t, i, i + 1, "odom", 0.05, 8.0) for i in range(7) if i != 4]
    edges.append(_edge(t, 3, 5, "window"))
    sol = pg.optimize(edges, fixed=0)
    for i, p in enumerate(t):
        _close(sol.poses[i], p, m=1e-3, rad=1e-3)


def test_edge_round_trip_connected_and_warm_start():
    e = pg.odom_edge(0, 1, [(10.0, 0.3)], False)
    back = pg.edge_from_dict(json.loads(json.dumps(pg.edge_to_dict(e, 1))))
    _close(back.z, e.z)
    assert (back.a, back.b, back.kind, back.sigma_m) == (0, 1, "odom", e.sigma_m)
    far = pg.odom_edge(5, 6, [(0.0, 0.2)], False)
    assert pg.connected([e, far], 0) == {0, 1}
    # 前の答え（start）から始めると、そこにあるものは動かさず、無いものだけを辺でたどる
    start = {0: Pose(0, 0, 0), 1: Pose(0.31, 0.01, 0.2)}
    two = pg.odom_edge(1, 2, [(0.0, 0.2)], False)
    init = pg.chain_init([0, 1, 2], [e, two], 0, start=start)
    _close(init[1], start[1])
    _close(init[2], to_frame(start[1], two.z))
