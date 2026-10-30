import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("mujoco")  # uv sync --group sim

import mujoco  # noqa: E402

from app import snoo_body as sb  # noqa: E402
from app.snoo_sim import GROUP_SNOO, Sim, SnooPose, build_mjcf, synthetic_room  # noqa: E402


@dataclass
class Posed:
    """姿勢の名前を持つ SnooPose（振る舞いの側が posture を足す前でも試せるように）。"""

    x: float
    y: float
    yaw: float
    moving: bool = False
    posture: str = "walk"


def _snoo_subtree_geoms(model) -> np.ndarray:
    root = model.body("snoo").id
    inside = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        while b not in (0, root):
            b = model.body_parentid[b]
        if b == root:
            inside.append(g)
    return np.array(inside)


def test_mjcf_builds_and_all_cat_geoms_are_in_the_snoo_group():
    model = mujoco.MjModel.from_xml_string(build_mjcf(synthetic_room()))
    geoms = _snoo_subtree_geoms(model)
    assert 20 <= len(geoms) <= 40  # 車のカメラの描く数（max_geom 2000）に効かない数
    assert (model.geom_group[geoms] == GROUP_SNOO).all()
    assert set(np.flatnonzero(model.geom_group == GROUP_SNOO)) == set(geoms)
    for name in sb.JOINT_NAMES:
        assert model.joint(name).id >= 0
    standalone = mujoco.MjModel.from_xml_string(sb.standalone_mjcf())
    assert standalone.nsite >= 30  # DeepLabCut の点に合わせる site


def _lowest_z(model, data, geoms) -> float:
    """geom の下の端のうち、いちばん低い高さ（楕円・カプセル・球の形から）。"""
    low = math.inf
    for g in geoms:
        R = data.geom_xmat[g].reshape(3, 3)
        size, z = model.geom_size[g], data.geom_xpos[g][2]
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            z -= math.sqrt(float(((R[2, :] * size) ** 2).sum()))
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            z -= abs(R[2, 2]) * size[1] + size[0]
        else:
            z -= size[0]
        low = min(low, z)
    return low


def test_every_posture_loads_within_the_joint_limits_and_stays_on_the_floor():
    data = json.loads(sb.POSTURES_PATH.read_text())["postures"]
    assert set(data) == set(sb.POSTURES)
    lo, hi = sb.limits()
    postures = sb.load_postures()
    model = mujoco.MjModel.from_xml_string(sb.standalone_mjcf(floor=False))
    d = mujoco.MjData(model)
    d.qpos[3] = 1.0
    adr = [model.joint(n).qposadr[0] for n in sb.JOINT_NAMES]
    geoms = np.flatnonzero(model.geom_group == 1)
    tops = {}
    for name, q in postures.items():
        assert data[name]["source"] in ("fit", "hand")
        assert set(data[name]["joints"]) <= set(sb.JOINT_NAMES)
        assert ((q >= lo - 1e-6) & (q <= hi + 1e-6)).all(), name
        d.qpos[adr] = q
        mujoco.mj_kinematics(model, d)
        # 床に 1.5 cm より深く沈まない（長い毛がつぶれる分くらいは許す）
        assert _lowest_z(model, d, geoms) > -0.015, name
        tops[name] = float(max(d.geom_xpos[g][2] for g in geoms))
    # 寝ている猫は、立っている猫より低い
    assert tops["sleep"] < tops["lie"] < tops["alert"]


def test_walk_cycle_is_periodic_and_moves_diagonal_legs_together():
    for phase in (0.0, 0.13, 0.5, 0.77):
        assert sb.gait_offsets(phase) == pytest.approx(sb.gait_offsets(phase + 1.0))
        d = sb.gait_offsets(phase)
        i = sb.JOINT_INDEX
        assert d[i["snoo_fl_shoulder"]] == pytest.approx(d[i["snoo_br_hip"]])
        assert d[i["snoo_fr_shoulder"]] == pytest.approx(d[i["snoo_bl_hip"]])
        # 首と頭は動かさない
        for n in ("snoo_neck_pitch", "snoo_neck_yaw", "snoo_head_pitch", "snoo_head_yaw"):
            assert d[i[n]] == 0.0
    # 1 歩の周期ぶん歩くと、同じ形に戻る
    anim = sb.SnooAnimator()
    dt, speed = 0.05, 0.25
    x = 0.0
    for _ in range(20):
        x += speed * dt
        anim.update(dt, Posed(x, 0.0, 0.0, True))
    start = anim.qpos.copy()
    n = round(sb.STRIDE_M / (speed * dt))
    seen = []
    for _ in range(n):
        x += speed * dt
        seen.append(anim.update(dt, Posed(x, 0.0, 0.0, True)).copy())
    assert seen[-1] == pytest.approx(start, abs=1e-9)
    assert max(np.abs(s - start).max() for s in seen) > math.radians(10)  # 途中は脚が動く


def test_posture_changes_blend_smoothly():
    anim = sb.SnooAnimator()
    walk, sleep = anim.postures["walk"], anim.postures["sleep"]
    anim.update(0.0, Posed(0, 0, 0, False, "walk"))
    assert anim.qpos == pytest.approx(walk)
    q = anim.update(sb.BLEND_S / 2, Posed(0, 0, 0, False, "sleep"))
    mid = np.abs(q - walk).max()
    assert 0 < mid < np.abs(sleep - walk).max()
    anim.update(sb.BLEND_S, Posed(0, 0, 0, False, "sleep"))
    assert anim.qpos == pytest.approx(sleep)
    # 名前の無い姿勢（SnooPose に posture が無い）は walk、別名も通じる
    assert sb.posture_name(getattr(SnooPose(0, 0, 0), "posture", "walk")) == "walk"
    assert sb.posture_name("sphinx") == "lie"
    assert sb.posture_name("unknown") == "walk"


def test_rays_hit_the_cat_and_the_detection_box_follows_the_posture():
    sim = Sim(synthetic_room(), rays=(64, 48), seed=0)
    sim.car.pan_deg = sim.car.tilt_deg = 0.0
    cam = sim.car.camera_pose()

    class Still:
        def __init__(self, pose):
            self.pose = pose

        def step(self, dt, robot):
            return self.pose

    heights = {}
    for name in ("alert", "sleep"):
        pose = Posed(cam.x + 1.0, cam.y, cam.yaw + math.pi / 2, False, name)
        sim.snoo = Still(pose)
        sim.advance(1.0)
        sim.sync()
        assert sim.detection is not None, name
        u0, v0, u1, v1, seen = sim.detection
        assert seen > 0.5
        heights[name] = v1 - v0
        # カメラから猫の胴の真ん中へ引いた光線は、snoo の geom に当たる
        pos, _ = sim.camera()
        target = sim.data.geom_xpos[sim.model.geom("snoo_body").id]
        d = target - pos
        dist, gid = sim.cast(pos, (d / np.linalg.norm(d))[None], 3.0)
        assert sim.model.geom_group[gid[0]] == GROUP_SNOO
    assert heights["sleep"] < 0.7 * heights["alert"]
