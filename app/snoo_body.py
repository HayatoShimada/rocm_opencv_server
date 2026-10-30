"""シミュレーターの snoo の体：関節のある猫の MJCF・姿勢の本・歩く動き。

体は腰・胴・胸の 3 つと首・頭・4 本の脚（上と下）・4 つに分けたしっぽ。物理は解かず、
関節の角度（qpos）を書いて形を作るだけ（app/snoo_sim.py と同じく接触も重力も切る）。
姿勢（歩く・座る・伏せる・丸まって寝る・立って見る）の角度は app/snoo_postures.json に置く。
本物の snoo の写真と動画に DeepLabCut（app/cat_pose.py）をかけた関節の点に、この体の
点（kp_ の site）を合わせて決めた（scripts/snoo_posture_fit.py）。合わなかったものは手で決めて
"source": "hand" と書く。

大きさ：鼻からお尻まで 約 44 cm、肩の高さ 約 25 cm（長毛なので毛も入れて太めにする）。
原点は体の真ん中の床、x が頭の向き、y が猫の左、z が上。関節がみな 0 のとき、脚をまっすぐ
立てて、しっぽを後ろへ伸ばした形になる。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSTURES_PATH = Path(__file__).with_name("snoo_postures.json")
POSTURES = ("walk", "sit", "lie", "sleep", "alert")
# 振る舞いの側が別の名前で呼んでも描けるように
ALIASES = {
    "stand": "alert",
    "stand_alert": "alert",
    "sphinx": "lie",
    "loaf": "lie",
    "curl": "sleep",
    "curled": "sleep",
    "rest": "sit",
}
# 毛の色。写真の体の真ん中の色（灰色に少し茶色が混ざる。部屋の明かりで暗く見える）から決めた
FUR_RGBA = (0.46, 0.43, 0.41, 1.0)
DARK_RGBA = (0.08, 0.07, 0.07, 1.0)
EYE_RGBA = (0.72, 0.58, 0.22, 1.0)


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str  # hinge か slide
    axis: tuple[float, float, float]
    lo: float  # hinge は度、slide は m
    hi: float


def _hinge(name, axis, lo, hi) -> Joint:
    return Joint(name, "hinge", axis, lo, hi)


Y, Z, X = (0, 1, 0), (0, 0, 1), (1, 0, 0)
# 体の根元（腰）の 3 つ：高さ・前後の傾き・横の傾き（横に倒れて寝るため）
ROOT_JOINTS = [
    Joint("snoo_z", "slide", Z, -0.16, 0.06),
    _hinge("snoo_pitch", Y, -70, 70),
    _hinge("snoo_roll", X, -100, 100),
]
SPINE_JOINTS = [
    _hinge("snoo_spine1_pitch", Y, -40, 40),
    _hinge("snoo_spine1_yaw", Z, -55, 55),
    _hinge("snoo_spine2_pitch", Y, -40, 40),
    _hinge("snoo_spine2_yaw", Z, -55, 55),
]
HEAD_JOINTS = [
    _hinge("snoo_neck_pitch", Y, -70, 60),
    _hinge("snoo_neck_yaw", Z, -80, 80),
    _hinge("snoo_head_pitch", Y, -60, 60),
    _hinge("snoo_head_yaw", Z, -60, 60),
    _hinge("snoo_head_roll", X, -45, 45),
]
# 脚は前後に振る 2 つ。+ は脚の先を後ろへ（y 軸まわりの右ねじ）
LEG_JOINTS = [
    j
    for leg in ("fl", "fr")
    for j in (_hinge(f"snoo_{leg}_shoulder", Y, -110, 80), _hinge(f"snoo_{leg}_elbow", Y, -30, 150))
] + [
    j
    for leg in ("bl", "br")
    for j in (_hinge(f"snoo_{leg}_hip", Y, -110, 80), _hinge(f"snoo_{leg}_knee", Y, -150, 40))
]
# しっぽは + の pitch で上がる
TAIL_JOINTS = [
    j
    for k in range(1, 5)
    for j in (_hinge(f"snoo_tail{k}_pitch", Y, -90, 90), _hinge(f"snoo_tail{k}_yaw", Z, -90, 90))
]
JOINTS = ROOT_JOINTS + SPINE_JOINTS + HEAD_JOINTS + LEG_JOINTS + TAIL_JOINTS
JOINT_NAMES = [j.name for j in JOINTS]
JOINT_INDEX = {n: i for i, n in enumerate(JOINT_NAMES)}

# しっぽの 4 つの長さと太さ（毛でふさふさなので太い）
TAIL_SEGS = ((0.06, 0.028), (0.065, 0.032), (0.065, 0.031), (0.06, 0.025))


def _f(*v) -> str:
    return " ".join(f"{x:.4g}" for x in v)


def _joint_xml(j: Joint) -> str:
    rng = f"{j.lo:.4g} {j.hi:.4g}"
    if j.kind == "slide":
        return f'<joint name="{j.name}" type="slide" axis="{_f(*j.axis)}" range="{rng}"/>'
    return f'<joint name="{j.name}" axis="{_f(*j.axis)}" range="{rng}"/>'


def _joints(names: list[str]) -> str:
    return "".join(_joint_xml(JOINTS[JOINT_INDEX[n]]) for n in names)


def _site(name: str, *pos: float) -> str:
    # 点は描かない組（5）に入れる。光線にも当たらない（site は光線を受けない）
    return f'<site name="kp_{name}" pos="{_f(*pos)}" size="0.006" group="5"/>'


def _leg_xml(leg: str, pos, front: bool) -> str:
    """脚 1 本（上・下・足先）。足先の中心は、関節がみな 0 のとき床から 1.6 cm。"""
    side = "left" if leg[1] == "l" else "right"
    sy = 1 if leg[1] == "l" else -1
    if front:
        a, b = "shoulder", "elbow"
        upper = '<geom type="capsule" fromto="0 0 0 0 0 -0.075" size="0.025"/>'
        lower = '<geom type="capsule" fromto="0 0 0 0 0 -0.056" size="0.018"/>'
        lower_pos, paw = (0, 0, -0.075), (0.008, 0, -0.059)
        dlc = f"front_{side}"
    else:
        a, b = "hip", "knee"
        # 後ろ脚の太ももは毛が多く太い。下はかかとから少し後ろへ下りる
        upper = '<geom type="capsule" fromto="0 0 0 0.02 0 -0.07" size="0.032"/>'
        lower = '<geom type="capsule" fromto="0 0 0 -0.012 0 -0.058" size="0.018"/>'
        lower_pos, paw = (0.02, 0, -0.07), (-0.004, 0, -0.064)
        dlc = f"back_{side}"
    return f"""
<body name="snoo_{leg}_upper" pos="{_f(*pos)}">{_joints([f"snoo_{leg}_{a}"])}
  {upper}{_site(f"{dlc}_thai", 0, 0.02 * sy, 0)}
  <body name="snoo_{leg}_lower" pos="{_f(*lower_pos)}">{_joints([f"snoo_{leg}_{b}"])}
    {lower}<geom type="ellipsoid" pos="{_f(*paw)}" size="0.024 0.019 0.016"/>
    {_site(f"{dlc}_knee", 0, 0.01 * sy, 0)}{_site(f"{dlc}_paw", *paw)}
  </body>
</body>"""


def _tail_xml() -> str:
    xml = ""
    for k, (length, r) in reversed(list(enumerate(TAIL_SEGS, 1))):
        pos = (-0.085, 0, 0.03) if k == 1 else (-TAIL_SEGS[k - 2][0], 0, 0)
        end = _site("tail_end", -length, 0, 0) if k == len(TAIL_SEGS) else ""
        xml = (
            f'<body name="snoo_tail{k}" pos="{_f(*pos)}">'
            f"{_joints([f'snoo_tail{k}_pitch', f'snoo_tail{k}_yaw'])}"
            f'<geom type="capsule" fromto="0 0 0 {-length:.4g} 0 0" size="{r}"/>{end}{xml}</body>'
        )
    return xml


def _head_xml() -> str:
    eye = f'size="0.011" rgba="{_f(*EYE_RGBA)}"'
    eyes = "".join(
        f'<geom type="sphere" pos="{_f(0.044, 0.022 * s, 0.01)}" {eye}/>' for s in (1, -1)
    )
    # 耳は三角だが楕円で。少し外へ倒す
    ears = "".join(
        f'<geom type="ellipsoid" pos="{_f(0.004, 0.032 * s, 0.042)}" size="0.01 0.02 0.03" '
        f'euler="{-22 * s} 0 0"/>'
        for s in (1, -1)
    )
    sites = "".join(
        _site(n, *p)
        for n, p in (
            ("nose", (0.074, 0, -0.008)),
            ("upper_jaw", (0.066, 0, -0.024)),
            ("lower_jaw", (0.052, 0, -0.04)),
            ("mouth_end_left", (0.055, 0.018, -0.03)),
            ("mouth_end_right", (0.055, -0.018, -0.03)),
            ("left_eye", (0.048, 0.022, 0.01)),
            ("right_eye", (0.048, -0.022, 0.01)),
            ("left_earbase", (0.01, 0.034, 0.034)),
            ("right_earbase", (0.01, -0.034, 0.034)),
            ("left_earend", (0.004, 0.046, 0.07)),
            ("right_earend", (0.004, -0.046, 0.07)),
            ("neck_end", (-0.035, 0, 0.03)),
            ("throat_end", (0.02, 0, -0.045)),
        )
    )
    return f"""
<body name="snoo_head" pos="0.045 0 0.045">
  {_joints(["snoo_head_pitch", "snoo_head_yaw", "snoo_head_roll"])}
  <geom name="snoo_head" type="sphere" size="0.05"/>
  <geom type="ellipsoid" pos="0.045 0 -0.018" size="0.026 0.029 0.021"/>
  <geom type="sphere" pos="0.069 0 -0.007" size="0.007" rgba="{_f(*DARK_RGBA)}"/>
  {eyes}{ears}{sites}
</body>"""


def body_xml(pos: tuple[float, float, float] = (1.0, -0.5, 0.0)) -> str:
    """関節のある猫の body。geom の色と組は、呼ぶ側の default の snoo の class で付ける。"""
    return f"""
<body name="snoo" pos="{_f(*pos)}" childclass="snoo">
  <freejoint name="snoo"/>
  <body name="snoo_hips" pos="-0.11 0 0.17">
    {_joints([j.name for j in ROOT_JOINTS])}
    <geom name="snoo_hips" type="ellipsoid" size="0.095 0.074 0.08"/>
    {_site("back_end", -0.045, 0, 0.07)}{_site("tail_base", -0.085, 0, 0.03)}
    {_leg_xml("bl", (-0.01, 0.045, -0.02), False)}
    {_leg_xml("br", (-0.01, -0.045, -0.02), False)}
    {_tail_xml()}
    <body name="snoo_waist" pos="0.07 0 0.005">
      {_joints(["snoo_spine1_pitch", "snoo_spine1_yaw"])}
      <geom name="snoo_body" type="ellipsoid" size="0.085 0.072 0.079"/>
      {_site("back_middle", 0, 0, 0.077)}{_site("belly_bottom", 0, 0, -0.075)}
      {_site("body_middle_left", 0, 0.069, 0)}{_site("body_middle_right", 0, -0.069, 0)}
      <body name="snoo_chest" pos="0.075 0 0.01">
        {_joints(["snoo_spine2_pitch", "snoo_spine2_yaw"])}
        <geom name="snoo_chest" type="ellipsoid" pos="0.005 0 0" size="0.08 0.07 0.08"/>
        {_site("back_base", -0.01, 0, 0.078)}{_site("neck_base", 0.045, 0, 0.066)}
        {_site("throat_base", 0.07, 0, -0.01)}
        {_leg_xml("fl", (0.03, 0.04, -0.035), True)}
        {_leg_xml("fr", (0.03, -0.04, -0.035), True)}
        <body name="snoo_neck" pos="0.07 0 0.035">
          {_joints(["snoo_neck_pitch", "snoo_neck_yaw"])}
          <geom type="capsule" fromto="0 0 0 0.04 0 0.035" size="0.048"/>
          {_head_xml()}
        </body>
      </body>
    </body>
  </body>
</body>"""


def standalone_mjcf(floor: bool = True) -> str:
    """猫だけの MJCF（姿勢を合わせる・絵を確かめるため）。組と色は snoo_sim と同じ。"""
    plane = '<geom name="floor" type="plane" size="2 2 0.05" rgba="0.75 0.7 0.64 1"/>'
    return f"""<mujoco model="snoo_body">
  <compiler angle="degree" boundmass="0.001" boundinertia="1e-7"/>
  <option><flag contact="disable" gravity="disable"/></option>
  <visual>
    <global offwidth="1280" offheight="1280"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0"/>
    <default class="snoo"><geom group="1" rgba="{_f(*FUR_RGBA)}"/></default>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <light directional="true" dir="0.3 0.4 -1" diffuse="0.3 0.3 0.3"/>
    {plane if floor else ""}
    {body_xml((0.0, 0.0, 0.0))}
    <camera name="fit" pos="1 0 0.2" xyaxes="0 1 0 0 0 1"/>
  </worldbody>
</mujoco>"""


# --- 姿勢の本 ---


def limits() -> tuple[np.ndarray, np.ndarray]:
    """関節ごとの下限・上限（qpos の単位：hinge はラジアン、slide は m）。"""
    lo = np.array([math.radians(j.lo) if j.kind == "hinge" else j.lo for j in JOINTS])
    hi = np.array([math.radians(j.hi) if j.kind == "hinge" else j.hi for j in JOINTS])
    return lo, hi


def to_vector(angles: dict[str, float]) -> np.ndarray:
    """{関節の名前: 度（slide は m）} を qpos の並び（ラジアン）に。書いていない関節は 0。"""
    v = np.zeros(len(JOINTS))
    for name, value in angles.items():
        j = JOINTS[JOINT_INDEX[name]]
        v[JOINT_INDEX[name]] = math.radians(value) if j.kind == "hinge" else value
    return v


def to_angles(v: np.ndarray) -> dict[str, float]:
    """qpos の並びを、保存する {関節の名前: 度（slide は m）} に。"""
    return {
        j.name: round(
            math.degrees(float(x)) if j.kind == "hinge" else float(x), 2 if j.kind == "hinge" else 4
        )
        for j, x in zip(JOINTS, v, strict=True)
    }


def load_postures(path: Path = POSTURES_PATH) -> dict[str, np.ndarray]:
    """姿勢の本を {名前: qpos の並び} で。"""
    data = json.loads(path.read_text())
    return {name: to_vector(p["joints"]) for name, p in data["postures"].items()}


def posture_name(name: str | None) -> str:
    name = ALIASES.get(name or "walk", name or "walk")
    return name if name in POSTURES else "walk"


# --- 歩く動き ---

# 1 歩の周期（左右の前脚が 1 回ずつ出る）で進む長さ。猫の歩幅 約 25 cm から
STRIDE_M = 0.30
# 姿勢を移すのにかける秒
BLEND_S = 0.4
# 止まってから歩く振りが消えるまで・歩き出して振りが付くまでの秒
GAIT_FADE_S = 0.3
# 対角の脚（左前と右後ろ、右前と左後ろ）を同じ位相で動かす（速歩）
LEG_PHASE = {"fl": 0.0, "br": 0.0, "fr": 0.5, "bl": 0.5}


def gait_offsets(phase: float) -> np.ndarray:
    """歩く周期の位相 phase（1 で 1 周）での、関節の角度の足し分（ラジアン）。

    脚は前後に振り、前へ戻す間（振り出し）だけ下の関節を曲げて足を上げる。
    胴としっぽは少し左右に揺れる。首と頭は動かさない（猫は歩くとき頭がぶれない）。
    """
    d = np.zeros(len(JOINTS))
    for leg, off in LEG_PHASE.items():
        a = 2 * math.pi * (phase + off)
        upper, lower = ("shoulder", "elbow") if leg[0] == "f" else ("hip", "knee")
        # sin が + の間は脚が後ろへ（地面を蹴る）、- の間は前へ戻す
        d[JOINT_INDEX[f"snoo_{leg}_{upper}"]] = math.radians(22) * math.sin(a)
        lift = max(0.0, -math.cos(a))  # 前へ戻す半周（sin が減る間）だけ上げる
        sign = 1 if leg[0] == "f" else -1  # 前脚の肘は + で、後ろ脚の膝は - で曲がる
        d[JOINT_INDEX[f"snoo_{leg}_{lower}"]] = sign * math.radians(35) * lift
    sway = math.sin(2 * math.pi * phase)
    d[JOINT_INDEX["snoo_spine1_yaw"]] = math.radians(4) * sway
    d[JOINT_INDEX["snoo_spine2_yaw"]] = -math.radians(4) * sway
    d[JOINT_INDEX["snoo_tail2_yaw"]] = math.radians(8) * sway
    d[JOINT_INDEX["snoo_tail3_yaw"]] = math.radians(10) * sway
    # 体の上下（1 周に 2 回）
    d[JOINT_INDEX["snoo_z"]] = 0.004 * math.cos(4 * math.pi * phase)
    return d


def _smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class SnooAnimator:
    """snoo の姿勢（SnooPose.posture）と動き（歩いた長さ）から、関節の角度を作る。

    姿勢が変わったら BLEND_S 秒かけて滑らかに移す。歩いている間は、歩いた長さで
    歩く周期を進め、脚を振る（止まれば振りは GAIT_FADE_S 秒で消える）。
    """

    def __init__(self, postures: dict[str, np.ndarray] | None = None, posture: str = "walk"):
        self.postures = postures if postures is not None else load_postures()
        self.target = posture_name(posture)
        self.start = self.postures[self.target].copy()
        self.blend = 1.0
        self.phase = 0.0
        self.gait = 0.0  # 歩く振りの強さ（0..1）
        self.last_xy: tuple[float, float] | None = None
        self.qpos = self.start.copy()

    def update(self, dt: float, pose) -> np.ndarray:
        """dt 秒進めて、関節の角度（JOINTS の順、ラジアン）を返す。pose は SnooPose。"""
        name = posture_name(getattr(pose, "posture", "walk"))
        if name != self.target:
            # いまの形から次の形へ（途中で変わっても、いまの形から始めるので飛ばない）
            self.start, self.target, self.blend = self.base(), name, 0.0
        self.blend = min(1.0, self.blend + dt / BLEND_S)
        xy = (pose.x, pose.y)
        ds = 0.0 if self.last_xy is None else math.dist(xy, self.last_xy)
        self.last_xy = xy
        self.phase = (self.phase + ds / STRIDE_M) % 1.0
        walking = name == "walk" and getattr(pose, "moving", False) and ds > 0
        step = dt / GAIT_FADE_S
        self.gait = min(1.0, self.gait + step) if walking else max(0.0, self.gait - step)
        self.qpos = self.base() + self.gait * gait_offsets(self.phase)
        return self.qpos

    def base(self) -> np.ndarray:
        """歩く振りを足す前の形（姿勢の間を移っている途中なら、その途中の形）。"""
        w = _smooth(self.blend)
        return (1 - w) * self.start + w * self.postures[self.target]
