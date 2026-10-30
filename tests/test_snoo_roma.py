"""RoMa の辺（app/snoo_roma.py）を、作った対応で確かめる。GPU も RoMa も使わない。"""

import math

import numpy as np
import pytest

from app import snoo_posegraph as pg
from app import snoo_roma as sr
from app.roommap import Pose


def _scene(rng: np.random.Generator) -> np.ndarray:
    """車 a の座標（原点はカメラの真下の床）の点：前の床と、前と左右の壁。"""
    floor = np.c_[rng.uniform(0.2, 2.0, 1500), rng.uniform(-1.0, 1.0, 1500), np.zeros(1500)]
    front = np.c_[np.full(800, 2.2), rng.uniform(-1.5, 1.5, 800), rng.uniform(0, 0.8, 800)]
    left = np.c_[rng.uniform(0.0, 2.2, 600), np.full(600, 1.2), rng.uniform(0, 0.8, 600)]
    right = np.c_[rng.uniform(0.0, 2.2, 600), np.full(600, -1.3), rng.uniform(0, 0.8, 600)]
    return np.vstack([floor, front, left, right])


def _project(P_car: np.ndarray, car: Pose, pan: float, tilt: float, K: np.ndarray):
    """車 a の座標の点を、car（a から見た位置）の車のカメラ（パン・チルト）に写す。"""
    c, s = math.cos(car.yaw), math.sin(car.yaw)
    Rw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    local = (P_car - [car.x, car.y, 0.0]) @ Rw  # 車 b の座標
    cam = (local - [0, 0, sr.CAMERA_HEIGHT_M]) @ sr.cam_to_car(pan, tilt)  # カメラの座標
    ok = cam[:, 2] > 0.05
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < sr.IMAGE_W) & (uv[:, 1] >= 0) & (uv[:, 1] < sr.IMAGE_H)
    return uv, ok


def _matches(b: Pose, pan_a, tilt_a, pan_b, tilt_b, noise_px=0.3, outliers=0.2, seed=0):
    rng = np.random.default_rng(seed)
    K = sr.intrinsics()
    P = _scene(rng)
    ua, oka = _project(P, Pose(0, 0, 0), pan_a, tilt_a, K)
    ub, okb = _project(P, b, pan_b, tilt_b, K)
    ok = oka & okb
    ka = ua[ok] + rng.normal(0, noise_px, (ok.sum(), 2))
    kb = ub[ok] + rng.normal(0, noise_px, (ok.sum(), 2))
    n_out = int(outliers * len(ka))
    kb[:n_out] = rng.uniform([0, 0], [sr.IMAGE_W, sr.IMAGE_H], (n_out, 2))
    return ka, kb


def test_cam_to_car_axes():
    R = sr.cam_to_car(0, 0)
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])  # 光軸は車の前
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0])  # 画像の右は車の右
    # パンが右に 30° なら光軸は右前、チルトが上に 20° なら光軸は上を向く
    f = sr.cam_to_car(30, 0) @ [0, 0, 1]
    assert f[1] < 0 and math.isclose(math.degrees(math.atan2(-f[1], f[0])), 30, abs_tol=1e-6)
    u = sr.cam_to_car(0, 20) @ [0, 0, 1]
    assert math.isclose(math.degrees(math.asin(u[2])), 20, abs_tol=1e-6)


@pytest.mark.parametrize(
    "b,pan_a,tilt_a,pan_b,tilt_b",
    [
        (Pose(0.30, 0.05, math.radians(10)), 0, -15, 0, -15),
        (Pose(0.25, -0.10, math.radians(-20)), 0, 0, -15, 0),
        (Pose(-0.20, 0.0, 0.0), 15, -15, 15, -15),  # 後ろへ下がった
    ],
)
def test_pair_pose_recovers_se2_and_floor_scale(b, pan_a, tilt_a, pan_b, tilt_b):
    ka, kb = _matches(b, pan_a, tilt_a, pan_b, tilt_b)
    e = sr.pair_pose(ka, kb, pan_a, tilt_a, pan_b, tilt_b)
    assert e is not None and not e.rotation_only
    assert e.floor_points >= sr.MIN_FLOOR_POINTS
    assert abs(e.pose.x - b.x) < 0.02 and abs(e.pose.y - b.y) < 0.02
    assert abs(math.degrees(pg.wrap(e.pose.yaw - b.yaw))) < 1.0


def test_scale_follows_camera_height():
    """同じ画素でも、カメラが 2 倍の高さなら動きも 2 倍（大きさは床までの距離だけで決まる）。"""
    b = Pose(0.3, 0.0, 0.0)
    ka, kb = _matches(b, 0, -15, 0, -15, outliers=0.0)
    e1 = sr.pair_pose(ka, kb, 0, -15, 0, -15, height_m=0.08)
    e2 = sr.pair_pose(ka, kb, 0, -15, 0, -15, height_m=0.16)
    assert e1 and e2
    assert math.isclose(e2.scale, 2 * e1.scale, rel_tol=0.02)


def test_rotation_only_when_no_parallax():
    """首を振っただけ（その場で回った）なら、回転だけを使い、動きは 0 にする。"""
    b = Pose(0.0, 0.0, math.radians(15))
    ka, kb = _matches(b, 0, 0, 0, 0, outliers=0.1)
    e = sr.pair_pose(ka, kb, 0, 0, 0, 0)
    assert e is not None and e.rotation_only
    assert abs(math.degrees(e.pose.yaw) - 15) < 0.5
    assert math.hypot(e.pose.x, e.pose.y) < 0.01


def test_too_few_matches_is_none():
    ka, kb = _matches(Pose(0.3, 0, 0), 0, -15, 0, -15)
    assert sr.pair_pose(ka[:20], kb[:20], 0, -15, 0, -15) is None


def test_combine_weighted_and_sigma():
    good = [sr.PairEstimate(Pose(0.30, 0.0, 0.10), 800, 100, 3.0, False, 0.1, 1.0)] * 3
    bad = sr.PairEstimate(Pose(1.0, 0.5, 1.0), 70, 30, 3.0, False, 0.1, 1.0)
    st = sr.combine([*good, bad, None])
    # 向きがそろわない組（bad）は外す
    assert st is not None and st.pairs == 3 and st.inliers == 2400
    assert abs(st.pose.x - 0.30) < 1e-9 and abs(st.pose.yaw - 0.10) < 1e-9
    assert sr.combine([]) is None
    # 向きのそろう組が足りない・対応が少ないときは辺にしない
    assert sr.combine(good[:1]) is None
    assert sr.combine(good[:2], min_inliers=5000) is None
    single = sr.combine(good[:1], min_pairs=1, min_inliers=0)
    assert single.sigma_m > sr.ROMA_SIGMA_M  # 1 組だけなら確からしさを下げる
    # 回転だけの組は向きにだけ使う
    rot = sr.PairEstimate(Pose(0.0, 0.0, 0.10), 1500, 0, 0.2, True, 0.0, 1.0)
    st = sr.combine([good[0], rot])
    assert abs(st.pose.x - 0.30) < 1e-9 and not st.rotation_only


def test_pick_frame_pairs_follows_heading_prior():
    fr = [{"file": f"{p}_{t}", "pan_deg": p, "tilt_deg": t}
          for t in (-15, 0) for p in (-30, -15, 0, 15, 30)]  # fmt: skip
    pairs = sr.pick_frame_pairs(fr, fr, 30.0, k=4)
    assert len(pairs) == 4
    assert {round(a["tilt_deg"]) for a, _ in pairs} == {-15, 0}
    for a, b in pairs:  # 車が左へ 30° 回ったら、b は 30° 右へ首を振ると同じ向きを見る
        assert b["pan_deg"] - a["pan_deg"] == 30


def test_roma_edge_kind_in_optimize_and_outlier():
    """roma の辺は窓の辺と同じく解に使い、合わなければ外れ値として外す。"""
    truth = {0: Pose(0, 0, 0), 1: Pose(0.3, 0, 0.2), 2: Pose(0.55, 0.1, 0.4)}
    odom = [pg.Edge(a, b, pg.relative(truth[a], truth[b]), 0.05, math.radians(15), "odom")
            for a, b in ((0, 1), (1, 2))]  # fmt: skip
    # 走った辺の向きは 10° ずれている。roma の辺は正しい
    for e in odom:
        e.z = Pose(e.z.x, e.z.y, e.z.yaw + math.radians(10))
    roma = [pg.Edge(a, b, pg.relative(truth[a], truth[b]), 0.03, math.radians(2), "roma")
            for a, b in ((0, 1), (1, 2), (0, 2))]  # fmt: skip
    wrong = pg.Edge(0, 2, Pose(-1.0, 1.0, 2.0), 0.03, math.radians(2), "roma")
    sol = pg.optimize(odom + roma + [wrong], 0)
    assert wrong.outlier and not any(e.outlier for e in roma)
    assert abs(math.degrees(pg.wrap(sol.poses[2].yaw - truth[2].yaw))) < 2.0
    assert 2 in sol.seen  # roma の辺は「見た」辺に数える


def test_roma_edge_from_estimate_roundtrips():
    st = sr.StationEstimate(Pose(0.3, 0.1, 0.2), 0.04, 0.05, 3, 900, False)
    e = sr.to_edge(4, 5, st, "roma:004_005")
    assert e.kind == "roma" and (e.a, e.b) == (4, 5)
    d = pg.edge_to_dict(e, 5)
    back = pg.edge_from_dict(d)
    assert back.kind == "roma" and back.z == e.z
