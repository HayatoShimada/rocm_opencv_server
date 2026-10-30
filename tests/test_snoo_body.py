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


def test_mjcf_builds_with_textures_and_all_cat_geoms_are_in_the_snoo_group():
    model = mujoco.MjModel.from_xml_string(build_mjcf(synthetic_room()))
    geoms = _snoo_subtree_geoms(model)
    assert 30 <= len(geoms) <= 70  # 車のカメラの描く数（max_geom 2000）に効かない数
    assert (model.geom_group[geoms] == GROUP_SNOO).all()
    assert set(np.flatnonzero(model.geom_group == GROUP_SNOO)) == set(geoms)
    for name in sb.JOINT_NAMES:
        assert model.joint(name).id >= 0
    # 毛の模様が読めている（殻は毛の筋の形にすける 4 チャンネル）
    assert sb.has_textures()
    for name, shell in sb.TEXTURES.items():
        tex = model.texture(f"snoo_{name}")
        assert model.tex_nchannel[tex.id] == (4 if shell else 3)
    # 猫の geom はみな材質を持つ（模様か色）
    assert (model.geom_matid[geoms] >= 0).all()
    fur = [g for g in geoms if model.geom(g).name.startswith("snoo_fur_")]
    assert len(fur) >= 10
    standalone = mujoco.MjModel.from_xml_string(sb.standalone_mjcf())
    assert standalone.nsite >= 30  # DeepLabCut の点に合わせる site
    # 模様は小さく（全部で 2 MB 未満）
    assert sum(p.stat().st_size for p in sb.ASSETS.glob("*")) < 2_000_000


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
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            continue  # 耳（頭の上）
        else:
            z -= size[0]
        low = min(low, z)
    return low


def _core_geoms(model) -> np.ndarray:
    """毛の殻（snoo_fur_）と顔の小さな物を除いた、体の芯の geom。"""
    return np.array(
        [
            g
            for g in np.flatnonzero(model.geom_group == 1)
            if not model.geom(g).name.startswith("snoo_fur_")
        ]
    )


def test_every_posture_loads_within_the_joint_limits_and_stays_on_the_floor():
    data = json.loads(sb.POSTURES_PATH.read_text())["postures"]
    assert set(data) == set(sb.POSTURES)
    lo, hi = sb.limits()
    postures = sb.load_postures()
    model = mujoco.MjModel.from_xml_string(sb.standalone_mjcf(floor=False))
    d = mujoco.MjData(model)
    d.qpos[3] = 1.0
    adr = [model.joint(n).qposadr[0] for n in sb.JOINT_NAMES]
    core = _core_geoms(model)
    tops = {}
    for name, q in postures.items():
        assert data[name]["source"] in ("fit", "hand")
        assert set(data[name]["joints"]) <= set(sb.JOINT_NAMES)
        assert ((q >= lo - 1e-6) & (q <= hi + 1e-6)).all(), name
        d.qpos[adr] = q
        mujoco.mj_kinematics(model, d)
        # 床に 1.5 cm より深く沈まない（長い毛がつぶれる分くらいは許す）
        assert _lowest_z(model, d, core) > -0.015, name
        tops[name] = float(max(d.geom_xpos[g][2] for g in core))
    # 寝ている猫は、立っている猫より低い
    assert tops["sleep"] < tops["lie"] < tops["alert"]
    assert data["groom"]["source"] == data["sniff"]["source"] == "hand"


def _walk(anim, speed, seconds, dt=1 / 60, x=0.0, posture="walk"):
    out = []
    for _ in range(round(seconds / dt)):
        x += speed * dt
        out.append(anim.update(dt, Posed(x, 0.0, 0.0, True, posture)).copy())
    return out, x


def test_walk_cycle_is_periodic_and_lateral_sequence():
    for phase in (0.0, 0.13, 0.5, 0.77):
        assert sb.gait_offsets(phase) == pytest.approx(sb.gait_offsets(phase + 1.0), abs=1e-9)
    # 足を着く順は 左後ろ → 左前 → 右後ろ → 右前、1/4 周ずつ（lateral-sequence walk）
    order = sorted(sb.WALK_LEGS, key=sb.WALK_LEGS.get)
    assert order == ["bl", "fl", "br", "fr"]
    gaps = np.diff([sb.WALK_LEGS[k] for k in order] + [1.0])
    assert gaps == pytest.approx([0.25] * 4)
    # 振り出しの時間は動画の前足（約 0.23 秒）で、速いほど地面に着く割合が減る
    assert sb.duty(0.2, False) > sb.duty(0.4, False)
    period = sb.stride(0.25, False) / 0.25
    assert (1 - sb.duty(0.25, False)) * period == pytest.approx(sb.SWING_S, abs=0.01)


def test_walking_feet_stay_planted_and_the_cycle_repeats():
    anim = sb.SnooAnimator(seed=0)
    speed, dt = 0.25, 1 / 120
    _, x = _walk(anim, speed, 2.0, dt)
    feet = []
    for _ in range(round(1.5 / dt)):
        x += speed * dt
        q = anim.update(dt, Posed(x, 0.0, 0.0, True))
        feet.append([(x + sb.foot(q, leg)[0], sb.foot(q, leg)[1]) for leg in ("fl", "bl")])
    feet = np.array(feet)  # (T, 2 本, (x, z))
    for k in range(2):
        xw, z = feet[:, k, 0], feet[:, k, 1]
        stance = z < z.min() + 0.002
        assert 0.55 < stance.mean() < 0.85  # 地面に着いている割合（duty）
        vx = np.abs(np.gradient(xw) / dt)[stance]
        assert np.median(vx) < 0.03 * speed / 0.25 + 0.01  # 着いた足は床で滑らない
        assert z.max() - z.min() > 0.015  # 振り出しで足が上がる
    # 1 周進むと同じ形に戻る（体の揺れも足の運びも）
    st = sb.stride(anim.speed, False)
    start_phase = anim.phase
    start = anim.qpos.copy()
    n = 0
    while True:
        x += speed * dt
        q = anim.update(dt, Posed(x, 0.0, 0.0, True)).copy()
        n += 1
        if n * speed * dt >= st - 1e-9:
            break
    assert anim.phase == pytest.approx(start_phase, abs=0.02)
    legs = [sb.JOINT_INDEX[n] for n in sb.GROUPS["front"] + sb.GROUPS["hind"]]
    assert q[legs] == pytest.approx(start[legs], abs=math.radians(4))


def test_transitions_follow_multi_stage_plans_within_the_joint_limits():
    lo, hi = sb.limits()
    names = sb.POSTURES
    for a in names:
        for b in names:
            stages = sb.plan(a, b)
            assert stages[-1].target == b
            anim = sb.SnooAnimator(posture=a, seed=1)
            anim.update(0.0, Posed(0, 0, 0, False, a))
            prev = anim.qpos.copy()
            total = sum(s.seconds for s in stages)
            dt = 1 / 30
            for _ in range(round((total + 0.5) / dt)):
                q = anim.update(dt, Posed(0, 0, 0, False, b)).copy()
                assert ((q >= lo - 1e-6) & (q <= hi + 1e-6)).all(), (a, b)
                # 飛ばない（1/30 秒で 25° より大きく動く関節は無い。耳とまぶたは除く）
                body = np.setdiff1d(np.arange(len(q)), sb.FACE_IDX)
                assert np.abs(q - prev)[body].max() < math.radians(25), (a, b)
                prev = q
            assert anim.posture_settled()
            assert anim.base() == pytest.approx(anim.postures[b])
    # 立った形から伏せる・寝るには、座る・伏せるを通る
    assert [s.target for s in sb.plan("walk", "sleep")][-2:] == ["sleep", "sleep"]
    assert len(sb.plan("walk", "sleep")) >= 3
    # 座るときは後ろ足（腰）が先に動き、前足はあとから
    sit = sb.plan("alert", "sit")[0]
    w = sb.stage_weights(sit, 0.3)
    i = sb.JOINT_INDEX
    assert w[i["snoo_bl_knee"]] > w[i["snoo_fl_shoulder"]]
    # 伏せた形から起きるときは、伸びの段を通る
    up = sb.plan("lie", "walk")
    assert any(s.bump for s in up)


def test_posture_change_midway_starts_from_the_current_shape():
    anim = sb.SnooAnimator(seed=0)
    walk = anim.postures["walk"]
    anim.update(0.0, Posed(0, 0, 0, False, "walk"))
    assert anim.base() == pytest.approx(walk)
    q = anim.update(0.3, Posed(0, 0, 0, False, "sleep")).copy()
    assert 0 < np.abs(anim.base() - walk).max()
    # 途中で行き先を変えても、いまの形から始まる（飛ばない）
    q2 = anim.update(1 / 30, Posed(0, 0, 0, False, "alert"))
    body = np.setdiff1d(np.arange(len(q)), sb.FACE_IDX)
    assert np.abs(q2 - q)[body].max() < math.radians(15)
    # 名前の無い姿勢（SnooPose に posture が無い）は walk、別名も通じる
    assert sb.posture_name(getattr(SnooPose(0, 0, 0), "posture", "walk")) == "walk"
    assert sb.posture_name("sphinx") == "lie"
    assert sb.posture_name("run") == "walk"
    assert sb.posture_name("groom") == "groom"
    assert sb.posture_name("unknown") == "walk"


def test_idle_motion_is_deterministic_per_seed_and_alive():
    def run(seed):
        anim = sb.SnooAnimator(posture="sit", seed=seed)
        out = []
        for _ in range(300):
            out.append(anim.update(1 / 30, Posed(0, 0, 0, False, "sit"), (1.0, 0.5, 0.08)).copy())
        return np.array(out)

    a, b, c = run(3), run(3), run(4)
    assert a == pytest.approx(b)
    assert np.abs(a - c).max() > 1e-3
    i = sb.JOINT_INDEX
    # 息としっぽの先：座っていても止まらない
    assert np.ptp(a[:, i["snoo_spine2_pitch"]]) > math.radians(0.5)
    assert np.ptp(a[:, i["snoo_tail4_yaw"]]) > math.radians(5)


def test_sleeping_closes_the_eyes_and_alert_looks_at_the_robot():
    i = sb.JOINT_INDEX
    anim = sb.SnooAnimator(posture="sleep", seed=0)
    for _ in range(30):
        q = anim.update(1 / 30, Posed(0, 0, 0, False, "sleep"), (1.0, 0.0, 0.08))
    assert q[i["snoo_lid"]] > math.radians(70)
    # 立って見る猫は、左にいるロボットの方へ首と頭を回す
    anim = sb.SnooAnimator(posture="alert", seed=0)
    for _ in range(60):
        q = anim.update(1 / 30, Posed(0, 0, 0, False, "alert"), (0.2, 1.0, 0.08))
    if anim.attend > 0.5:
        assert q[i["snoo_neck_yaw"]] + q[i["snoo_head_yaw"]] > math.radians(45)
    assert q[i["snoo_lid"]] < math.radians(60) or anim.events  # 起きていれば目は開いている


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
        for _ in range(40):
            sim.advance(0.25)  # 姿勢の移り（寝るには数段）を終える
        sim.sync()
        assert sim.detection is not None, name
        u0, v0, u1, v1, seen = sim.detection
        assert seen > 0.5
        heights[name] = v1 - v0
        # カメラから猫の胴の真ん中へ引いた光線は、snoo の geom（毛の殻か芯）に当たる
        pos, _ = sim.camera()
        target = sim.data.geom_xpos[sim.model.geom("snoo_body").id]
        d = target - pos
        dist, gid = sim.cast(pos, (d / np.linalg.norm(d))[None], 3.0)
        assert sim.model.geom_group[gid[0]] == GROUP_SNOO
    assert heights["sleep"] < 0.7 * heights["alert"]
