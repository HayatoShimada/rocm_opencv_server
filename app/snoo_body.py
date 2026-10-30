"""シミュレーターの snoo の体：関節のある猫の MJCF・毛の模様・姿勢の本・動き。

体は腰・胴・胸の 3 つと首・頭・4 本の脚（上と下）・4 つに分けたしっぽ。物理は解かず、
関節の角度（qpos）を書いて形を作るだけ（app/snoo_sim.py と同じく接触も重力も切る）。
姿勢（歩く・座る・伏せる・丸まって寝る・立って見る・毛づくろい・嗅ぐ）の角度は
app/snoo_postures.json に置く。本物の snoo の写真と動画に DeepLabCut（app/cat_pose.py）を
かけた関節の点に、この体の点（kp_ の site）を合わせて決めた（scripts/snoo_posture_fit.py）。
合わなかったものと、写真の無いもの（毛づくろい・嗅ぐ）は手で決めて "source": "hand" と書く。

見た目：snoo は長毛の灰色（少し茶の混ざる）のさば虎で、胸と腹が明るく、目は琥珀色。
毛の色と縞は写真から測り（scripts/snoo_fur_texture.py → app/assets/snoo/）、体の geom に
模様として貼る。長い毛でふくらんだ外形は、芯の geom の外に、毛の筋の形にすける殻
（snoo_fur_ で始まる geom）を重ねて作る。しっぽは太い房にする。殻も光線に当たる
（本物の猫も毛の外形でカメラや超音波に写る）。

大きさ：鼻からお尻まで 約 44 cm、肩の高さ 約 25 cm（毛も入れて）。
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
ASSETS = Path(__file__).resolve().with_name("assets") / "snoo"
POSTURES = ("walk", "sit", "lie", "sleep", "alert", "groom", "sniff")
# 振る舞いの側が別の名前で呼んでも描けるように
ALIASES = {
    "stand": "alert",
    "stand_alert": "alert",
    "sphinx": "lie",
    "loaf": "lie",
    "curl": "sleep",
    "curled": "sleep",
    "rest": "sit",
    # 走るのは歩く形で、脚の運びを駆け足（ギャロップ）にする（SnooAnimator）
    "run": "walk",
}
# 模様が読めないときの色（写真の体の真ん中の色。灰色に少し茶色が混ざる）
FUR_RGBA = (0.46, 0.43, 0.41, 1.0)
DARK_RGBA = (0.08, 0.07, 0.07, 1.0)
EYE_RGBA = (0.72, 0.58, 0.22, 1.0)
NOSE_RGBA = (0.42, 0.33, 0.33, 1.0)
EAR_INNER_RGBA = (0.62, 0.55, 0.52, 1.0)
EAR_OUTER_RGBA = (0.36, 0.33, 0.31, 1.0)
MUZZLE_RGBA = (0.58, 0.55, 0.52, 1.0)
# 耳の形（三角すい）。付け根は前後に薄く左右に広い三角、先は少し後ろ。耳の body の座標（m）
EAR_MESH = {
    "outer": ((0.006, -0.021, -0.006), (0.006, 0.021, -0.006), (-0.012, 0.0, -0.008),
              (-0.003, 0.0, 0.04)),
    "inner": ((0.0085, -0.014, -0.002), (0.0085, 0.014, -0.002), (0.0005, 0.0, -0.003),
              (0.0015, 0.0, 0.03)),
}  # fmt: skip


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
    for j in (_hinge(f"snoo_{leg}_hip", Y, -110, 80), _hinge(f"snoo_{leg}_knee", Y, -150, 70))
]
# しっぽは + の pitch で上がる
TAIL_JOINTS = [
    j
    for k in range(1, 5)
    for j in (_hinge(f"snoo_tail{k}_pitch", Y, -90, 90), _hinge(f"snoo_tail{k}_yaw", Z, -90, 90))
]
# 耳（+ で前へ、- で後ろへ回す）とまぶた（0 で開き、90 で閉じる）。姿勢の本には書かず、
# SnooAnimator の細かい動き（耳を動かす・まばたき・寝ると閉じる）が決める
FACE_JOINTS = [
    _hinge("snoo_ear_l", Z, -70, 30),
    _hinge("snoo_ear_r", Z, -30, 70),
    _hinge("snoo_lid", Y, -80, 95),
]
JOINTS = ROOT_JOINTS + SPINE_JOINTS + HEAD_JOINTS + LEG_JOINTS + TAIL_JOINTS + FACE_JOINTS
JOINT_NAMES = [j.name for j in JOINTS]
JOINT_INDEX = {n: i for i, n in enumerate(JOINT_NAMES)}

# しっぽの 4 つの長さと芯の太さ、毛の房（殻）の半径。房は付け根から真ん中で太く、先で細る
TAIL_SEGS = ((0.06, 0.028), (0.065, 0.032), (0.065, 0.031), (0.06, 0.025))
TAIL_PLUME = (0.042, 0.052, 0.056, 0.05)
LID_OPEN_DEG = -75.0
LID_CLOSED_DEG = 88.0


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


# --- 模様（app/assets/snoo） ---

# 模様の名前と、すける殻か。殻は毛の筋の形のアルファを持つ（4 チャンネル）
TEXTURES = {
    "fur_body": False,
    "fur_chest": False,
    "fur_head": False,
    "fur_leg": False,
    "fur_tail": False,
    "fur_eye": False,
    "fur_shell": True,
    "fur_shell_chest": True,
    "fur_shell_tail": True,
}


def _fur_colors() -> dict:
    """写真から測った色（fur.json）。無ければ空。"""
    path = ASSETS / "fur.json"
    try:
        return json.loads(path.read_text()).get("rgba", {})
    except (OSError, ValueError):
        return {}


def has_textures() -> bool:
    return all((ASSETS / f"{n}.png").exists() for n in TEXTURES)


def assets_xml() -> str:
    """猫の模様と材質（<asset> の中に入れる）。模様の画像が無ければ色だけの材質にする。"""
    colors = _fur_colors()

    def rgba(key, default):
        return _f(*colors.get(key, default))

    plain = {
        "snoo_nose": rgba("nose", NOSE_RGBA),
        "snoo_dark": _f(*DARK_RGBA),
        "snoo_ear_inner": rgba("ear_inner", EAR_INNER_RGBA),
        "snoo_ear_outer": rgba("ear_outer", EAR_OUTER_RGBA),
        "snoo_muzzle": rgba("muzzle", MUZZLE_RGBA),
    }
    xml = "".join(
        f'<material name="{n}" rgba="{c}" specular="0" shininess="0"/>' for n, c in plain.items()
    )
    for n, verts in EAR_MESH.items():
        v = " ".join(_f(*p) for p in verts)
        xml += f'<mesh name="snoo_ear_{n}" vertex="{v}" face="0 2 1 0 1 3 1 2 3 2 0 3"/>'
    if not has_textures():
        fur = _f(*FUR_RGBA)
        for n, shell in TEXTURES.items():
            c = _f(*EYE_RGBA) if n == "fur_eye" else (_f(*FUR_RGBA[:3], 0.35) if shell else fur)
            xml += f'<material name="snoo_{n}" rgba="{c}" specular="0.05"/>'
        return xml
    for n, shell in TEXTURES.items():
        ch = ' nchannel="4"' if shell else ""
        xml += f'<texture name="snoo_{n}" type="2d"{ch} file="{ASSETS / (n + ".png")}"/>'
        # 毛はつやが少ない。目だけ光る
        spec = 'specular="0.6" shininess="0.8"' if n == "fur_eye" else 'specular="0" shininess="0"'
        # 殻は材質のアルファを 1 より少し下げる。すける物として描かれ、影を落とさない
        # （影を落とすと、中の芯が殻の影に入り、影の細かいむらが縞になって見える）
        alpha = ' rgba="1 1 1 0.98"' if shell else ""
        xml += f'<material name="snoo_{n}" texture="snoo_{n}" {spec}{alpha}/>'
    return xml


def default_xml(group: int) -> str:
    """猫の geom の既定（組と材質）。<default> の中に入れる。"""
    return f'<default class="snoo"><geom group="{group}" material="snoo_fur_body"/></default>'


def _shell(name: str, kind: str, size, pos=(0, 0, 0), mat="fur_shell", euler=None) -> str:
    """毛の殻（すける）。名前は snoo_fur_ で始める（床に沈むかの試験から外すため）。"""
    e = f' euler="{_f(*euler)}"' if euler else ""
    return (
        f'<geom name="snoo_fur_{name}" type="{kind}" pos="{_f(*pos)}" size="{_f(*size)}" '
        f'material="snoo_{mat}"{e}/>'
    )


def _leg_xml(leg: str, pos, front: bool) -> str:
    """脚 1 本（上・下・足先）。足先の中心は、関節がみな 0 のとき床から 1.6 cm。"""
    side = "left" if leg[1] == "l" else "right"
    sy = 1 if leg[1] == "l" else -1
    m = 'material="snoo_fur_leg"'
    if front:
        a, b = "shoulder", "elbow"
        upper = f'<geom type="capsule" fromto="0 0 0 0 0 -0.075" size="0.025" {m}/>'
        lower = f'<geom type="capsule" fromto="0 0 0 0 0 -0.056" size="0.018" {m}/>'
        lower_pos, paw = (0, 0, -0.075), (0.008, 0, -0.059)
        dlc = f"front_{side}"
        # 前足の上は毛が長く、胸の毛とつながる
        fluff = _shell(
            f"{leg}", "ellipsoid", (0.032, 0.03, 0.048), (0.0, 0, -0.032), "fur_shell_chest"
        )
    else:
        a, b = "hip", "knee"
        # 後ろ脚の太ももは毛が多く太い（ズボンのような毛）。下はかかとから少し後ろへ下りる
        upper = f'<geom type="capsule" fromto="0 0 0 0.02 0 -0.07" size="0.032" {m}/>'
        lower = f'<geom type="capsule" fromto="0 0 0 -0.012 0 -0.058" size="0.018" {m}/>'
        lower_pos, paw = (0.02, 0, -0.07), (-0.004, 0, -0.064)
        dlc = f"back_{side}"
        fluff = _shell(f"{leg}", "ellipsoid", (0.048, 0.044, 0.058), (-0.004, 0, -0.03))
    return f"""
<body name="snoo_{leg}_upper" pos="{_f(*pos)}">{_joints([f"snoo_{leg}_{a}"])}
  {upper}{fluff}{_site(f"{dlc}_thai", 0, 0.02 * sy, 0)}
  <body name="snoo_{leg}_lower" pos="{_f(*lower_pos)}">{_joints([f"snoo_{leg}_{b}"])}
    {lower}<geom type="ellipsoid" pos="{_f(*paw)}" size="0.024 0.019 0.016" {m}/>
    {_site(f"{dlc}_knee", 0, 0.01 * sy, 0)}{_site(f"{dlc}_paw", *paw)}
  </body>
</body>"""


def _tail_xml() -> str:
    xml = ""
    for k, (length, r) in reversed(list(enumerate(TAIL_SEGS, 1))):
        pos = (-0.085, 0, 0.03) if k == 1 else (-TAIL_SEGS[k - 2][0], 0, 0)
        end = _site("tail_end", -length, 0, 0) if k == len(TAIL_SEGS) else ""
        p = TAIL_PLUME[k - 1]
        # 房は節に沿った太いカプセル。両端の丸みが隣の節の房と重なって、曲げても 1 本の房に見える
        plume = (
            f'<geom name="snoo_fur_tail{k}" type="capsule" size="{p:.4g}" '
            f'fromto="0 0 0 {-length:.4g} 0 0" material="snoo_fur_shell_tail"/>'
        )
        xml = (
            f'<body name="snoo_tail{k}" pos="{_f(*pos)}">'
            f"{_joints([f'snoo_tail{k}_pitch', f'snoo_tail{k}_yaw'])}"
            f'<geom type="capsule" fromto="0 0 0 {-length:.4g} 0 0" size="{r}" '
            f'material="snoo_fur_tail"/>{plume}{end}{xml}</body>'
        )
    return xml


def _head_xml() -> str:
    # 目は前から見て大きく、横から見て薄い（頭の球から少しだけ出る）
    eyes = "".join(
        f'<geom type="ellipsoid" pos="{_f(0.042, 0.021 * s, 0.01)}" size="0.0065 0.0115 0.0098" '
        f'material="snoo_fur_eye" euler="0 0 {16 * s}"/>'
        for s in (1, -1)
    )
    # まぶた：両目の真ん中を通る y 軸で回る薄い楕円。開くと目の上（頭の中）に隠れ、閉じると
    # 目の前をふさぐ
    lids = "".join(
        f'<geom type="ellipsoid" pos="{_f(0, 0.021 * s, 0.013)}" size="0.0122 0.0122 0.006" '
        f'material="snoo_fur_head"/>'
        for s in (1, -1)
    )
    # 耳は三角すい（mesh）。少し外へ倒す。内側は明るい毛（写真の耳の内側の色）
    ears = "".join(
        f'<body name="snoo_ear_{n}" pos="{_f(0.006, 0.03 * s, 0.04)}" euler="{-24 * s} 0 0">'
        f"{_joints([f'snoo_ear_{n}'])}"
        f'<geom type="mesh" mesh="snoo_ear_outer" material="snoo_ear_outer"/>'
        f'<geom type="mesh" mesh="snoo_ear_inner" material="snoo_ear_inner"/></body>'
        for n, s in (("l", 1), ("r", -1))
    )
    # 頬の毛：顔を丸く広く見せる（長毛の猫は頬と首の毛で顔が大きく見える）。目より後ろに置く
    cheeks = "".join(
        _shell(
            f"cheek_{n}",
            "ellipsoid",
            (0.03, 0.026, 0.03),
            (-0.006, 0.03 * s, -0.022),
            "fur_shell_chest",
        )  # fmt: skip
        for n, s in (("l", 1), ("r", -1))
    )
    # ひげの付け根のふくらみ（左右）
    pads = "".join(
        f'<geom type="ellipsoid" pos="{_f(0.045, 0.0095 * s, -0.021)}" '
        f'size="0.0095 0.011 0.0095" material="snoo_muzzle"/>'
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
  <geom name="snoo_head" type="sphere" size="0.05" material="snoo_fur_head"/>
  <geom type="ellipsoid" pos="0.033 0 -0.02" size="0.02 0.026 0.018" material="snoo_muzzle"/>
  {pads}
  <geom type="ellipsoid" pos="0.0525 0 -0.0085" size="0.005 0.0065 0.0045" material="snoo_nose"/>
  {eyes}{ears}{cheeks}{sites}
  <body name="snoo_lids" pos="0.042 0 0.01">{_joints(["snoo_lid"])}{lids}</body>
</body>"""


def body_xml(pos: tuple[float, float, float] = (1.0, -0.5, 0.0)) -> str:
    """関節のある猫の body。geom の組と既定の材質は、呼ぶ側の default の snoo の class で付ける
    （default_xml）。材質と模様は assets_xml を <asset> に入れる。"""
    return f"""
<body name="snoo" pos="{_f(*pos)}" childclass="snoo">
  <freejoint name="snoo"/>
  <body name="snoo_hips" pos="-0.11 0 0.17">
    {_joints([j.name for j in ROOT_JOINTS])}
    <geom name="snoo_hips" type="ellipsoid" size="0.095 0.074 0.08"/>
    {_shell("hips", "ellipsoid", (0.12, 0.088, 0.092), (-0.004, 0, 0.002))}
    {_site("back_end", -0.045, 0, 0.07)}{_site("tail_base", -0.085, 0, 0.03)}
    {_leg_xml("bl", (-0.01, 0.045, -0.02), False)}
    {_leg_xml("br", (-0.01, -0.045, -0.02), False)}
    {_tail_xml()}
    <body name="snoo_waist" pos="0.07 0 0.005">
      {_joints(["snoo_spine1_pitch", "snoo_spine1_yaw"])}
      <geom name="snoo_body" type="ellipsoid" size="0.085 0.072 0.079"/>
      {_shell("waist", "ellipsoid", (0.11, 0.087, 0.09))}
      {_shell("belly", "ellipsoid", (0.085, 0.072, 0.05), (0, 0, -0.05), "fur_shell_chest")}
      {_site("back_middle", 0, 0, 0.077)}{_site("belly_bottom", 0, 0, -0.075)}
      {_site("body_middle_left", 0, 0.069, 0)}{_site("body_middle_right", 0, -0.069, 0)}
      <body name="snoo_chest" pos="0.075 0 0.01">
        {_joints(["snoo_spine2_pitch", "snoo_spine2_yaw"])}
        <geom name="snoo_chest" type="ellipsoid" pos="0.005 0 0" size="0.08 0.07 0.08"
              material="snoo_fur_chest"/>
        {_shell("chest", "ellipsoid", (0.092, 0.086, 0.094), (0.012, 0, -0.004), "fur_shell_chest")}
        {_site("back_base", -0.01, 0, 0.078)}{_site("neck_base", 0.045, 0, 0.066)}
        {_site("throat_base", 0.07, 0, -0.01)}
        {_leg_xml("fl", (0.03, 0.04, -0.035), True)}
        {_leg_xml("fr", (0.03, -0.04, -0.035), True)}
        <body name="snoo_neck" pos="0.07 0 0.035">
          {_joints(["snoo_neck_pitch", "snoo_neck_yaw"])}
          <geom type="capsule" fromto="0 0 0 0.04 0 0.035" size="0.048" material="snoo_fur_chest"/>
          {_shell("ruff", "ellipsoid", (0.058, 0.074, 0.062), (0.014, 0, 0.006), "fur_shell_chest")}
          {_head_xml()}
        </body>
      </body>
    </body>
  </body>
</body>"""


def standalone_mjcf(floor: bool = True) -> str:
    """猫だけの MJCF（姿勢を合わせる・絵を確かめるため）。組と材質は snoo_sim と同じ。"""
    plane = '<geom name="floor" type="plane" size="2 2 0.05" rgba="0.75 0.7 0.64 1"/>'
    return f"""<mujoco model="snoo_body">
  <compiler angle="degree" boundmass="0.001" boundinertia="1e-7"/>
  <option><flag contact="disable" gravity="disable"/></option>
  <visual>
    <global offwidth="1280" offheight="1280"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45"/>
  </visual>
  <asset>{assets_xml()}</asset>
  <default>
    <geom contype="0" conaffinity="0"/>
    {default_xml(1)}
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


# --- 姿勢の移り方 ---
#
# 姿勢の間は、まっすぐ混ぜずに、体の部分ごとに時間をずらして動かす（猫の体のつくりと、
# 猫の動き方の決め事。snoo の録画にはこの移る途中がほとんど写っていないので、データではない）。
# - 座る：後ろ足から（お尻を下ろし、前足はまっすぐのまま）。しっぽは最後に前へ回す
# - 伏せる：座った形から前足を前へ出し、胸を下ろす
# - 寝る：横に倒れてから、背中を丸めて顔をしっぽの方へ埋める
# - 起きる：頭が先に上がり、前足で胸を押し上げ、ときどき伸び（前足を前に伸ばしてお尻を上げる）
#   をしてから立つ
# 段（Stage）ごとに、行き先の姿勢と、体の部分ごとの動く時間（段の中の割合 a..b）と、
# 途中だけ足す形（伸びなど。段の真ん中でいちばん大きく、両端で 0）を持つ。

GROUPS = {
    "root": ("snoo_z", "snoo_pitch", "snoo_roll"),
    "spine": tuple(j.name for j in SPINE_JOINTS),
    "head": tuple(j.name for j in HEAD_JOINTS),
    "front": tuple(j.name for j in LEG_JOINTS[:4]),
    "hind": tuple(j.name for j in LEG_JOINTS[4:]),
    "tail": tuple(j.name for j in TAIL_JOINTS),
}
GROUP_OF = np.zeros(len(JOINTS), int)
for _k, (_g, _names) in enumerate(GROUPS.items()):
    for _n in _names:
        GROUP_OF[JOINT_INDEX[_n]] = _k
GROUP_NAMES = list(GROUPS)
FACE_IDX = np.array([JOINT_INDEX[j.name] for j in FACE_JOINTS])


@dataclass(frozen=True)
class Stage:
    seconds: float
    target: str
    windows: tuple[tuple[str, float, float], ...] = ()  # (部分, 始め, 終わり)。無ければ 0..1
    bump: tuple[tuple[str, float], ...] = ()  # 途中だけ足す (関節, 度)


STAND = ("walk", "alert", "sniff")
# 伸び：前足を前に伸ばし、胸を下げ、お尻を上げる（起きたあとの猫の伸び）
STRETCH = (
    ("snoo_pitch", 18.0), ("snoo_spine2_pitch", 12.0), ("snoo_fl_shoulder", -45.0),
    ("snoo_fr_shoulder", -45.0), ("snoo_fl_elbow", -10.0), ("snoo_fr_elbow", -10.0),
    ("snoo_neck_pitch", -15.0), ("snoo_tail1_pitch", 20.0), ("snoo_z", -0.02),
)  # fmt: skip
EDGES: dict[tuple[str, str], tuple[Stage, ...]] = {
    ("stand", "sit"): (
        Stage(
            0.9,
            "sit",
            (
                ("root", 0.0, 0.75),
                ("hind", 0.0, 0.7),
                ("spine", 0.1, 0.8),
                ("front", 0.25, 0.85),
                ("head", 0.0, 0.6),
                ("tail", 0.45, 1.0),
            ),
        ),
    ),  # fmt: skip
    ("sit", "stand"): (
        Stage(
            0.65,
            "alert",
            (
                ("root", 0.0, 0.8),
                ("hind", 0.0, 0.8),
                ("front", 0.0, 0.5),
                ("spine", 0.0, 0.7),
                ("tail", 0.2, 1.0),
            ),
        ),
    ),  # fmt: skip
    ("sit", "lie"): (
        Stage(
            1.3,
            "lie",
            (
                ("front", 0.0, 0.55),
                ("head", 0.0, 0.5),
                ("spine", 0.15, 0.8),
                ("root", 0.3, 1.0),
                ("hind", 0.35, 1.0),
                ("tail", 0.5, 1.0),
            ),
        ),
    ),  # fmt: skip
    ("lie", "sit"): (
        Stage(
            0.9,
            "sit",
            (
                ("head", 0.0, 0.4),
                ("front", 0.0, 0.6),
                ("spine", 0.0, 0.7),
                ("root", 0.15, 0.9),
                ("hind", 0.2, 0.9),
                ("tail", 0.3, 1.0),
            ),
        ),
    ),  # fmt: skip
    ("stand", "lie"): (
        # 立ったまま前後を一緒にかがめ、後ろが先に着く
        Stage(
            1.4,
            "lie",
            (
                ("hind", 0.0, 0.6),
                ("root", 0.0, 0.85),
                ("front", 0.2, 0.9),
                ("spine", 0.1, 0.9),
                ("head", 0.1, 0.7),
                ("tail", 0.4, 1.0),
            ),
        ),
    ),  # fmt: skip
    ("lie", "stand"): (
        Stage(
            0.55,
            "sit",
            (("head", 0.0, 0.6), ("front", 0.0, 1.0), ("spine", 0.0, 1.0), ("root", 0.2, 1.0)),
        ),
        Stage(
            1.0,
            "alert",
            (
                ("root", 0.0, 0.7),
                ("hind", 0.0, 0.6),
                ("front", 0.1, 0.9),
                ("spine", 0.0, 0.8),
                ("tail", 0.2, 1.0),
            ),
            STRETCH,
        ),
    ),  # fmt: skip
    ("lie", "sleep"): (
        Stage(1.0, "sleep", (("root", 0.0, 0.6), ("hind", 0.0, 0.8), ("front", 0.1, 0.8))),
        Stage(1.2, "sleep", (("spine", 0.0, 0.8), ("head", 0.2, 1.0), ("tail", 0.3, 1.0))),
    ),  # fmt: skip
    ("sleep", "lie"): (
        Stage(
            1.3,
            "lie",
            (
                ("head", 0.0, 0.45),
                ("spine", 0.1, 0.7),
                ("root", 0.25, 1.0),
                ("front", 0.3, 1.0),
                ("hind", 0.4, 1.0),
                ("tail", 0.2, 0.9),
            ),
        ),
    ),  # fmt: skip
    ("sit", "groom"): (Stage(0.6, "groom", (("front", 0.0, 1.0), ("head", 0.2, 1.0))),),
    ("groom", "sit"): (Stage(0.5, "sit", (("front", 0.0, 0.8), ("head", 0.0, 1.0))),),
}
# 名前の付いた姿勢から、移り方の表の「場所」へ（立つ姿勢はみな stand）
NODE = {"walk": "stand", "alert": "stand", "sniff": "stand", "sit": "sit", "lie": "lie",
        "sleep": "sleep", "groom": "groom"}  # fmt: skip
# 立つ姿勢どうし（歩く・立って見る・嗅ぐ）はそのまま混ぜる
DIRECT_S = 0.45


def route(a: str, b: str) -> list[str]:
    """移り方の表の場所 a から b への道（幅優先。a と b を含む）。"""
    if a == b:
        return [a]
    nodes = {n for e in EDGES for n in e}
    prev = {a: None}
    todo = [a]
    while todo:
        n = todo.pop(0)
        for x, y in EDGES:
            if x == n and y not in prev:
                prev[y] = n
                todo.append(y)
    if b not in prev or b not in nodes:
        return [a, b]
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def plan(a_name: str, b_name: str) -> list[Stage]:
    """姿勢 a_name から b_name への段の並び。最後の段の行き先は b_name そのもの。"""
    a, b = NODE[a_name], NODE[b_name]
    if a == b:
        return [Stage(DIRECT_S, b_name)]
    stages: list[Stage] = []
    path = route(a, b)
    for x, y in zip(path, path[1:], strict=False):
        stages += list(EDGES.get((x, y), (Stage(0.8, "alert" if y == "stand" else y),)))
    last = stages[-1]
    if last.target != b_name:
        # 立つ姿勢の行き先（walk・sniff）は、最後の段の向かう先を差し替える
        stages[-1] = Stage(last.seconds, b_name, last.windows, last.bump)
    return stages


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def stage_weights(stage: Stage, u: float) -> np.ndarray:
    """段の中の割合 u での、関節ごとの進み（0..1）。"""
    w = np.full(len(JOINTS), float(_smooth(u)))
    for g, a, b in stage.windows:
        k = GROUP_NAMES.index(g)
        w[GROUP_OF == k] = _smooth((u - a) / max(b - a, 1e-6))
    w[FACE_IDX] = 0.0
    return w


def _bump(stage: Stage, u: float) -> np.ndarray:
    d = np.zeros(len(JOINTS))
    s = math.sin(math.pi * min(max(u, 0.0), 1.0))
    for name, value in stage.bump:
        j = JOINTS[JOINT_INDEX[name]]
        d[JOINT_INDEX[name]] = (math.radians(value) if j.kind == "hinge" else value) * s
    return d


# --- 歩く動き ---
#
# 脚の運び（足を着く順と間合い）は、猫の歩き方の決め事（lateral-sequence walk：左後ろ →
# 左前 → 右後ろ → 右前 を 1/4 周ずつ）を使う。snoo の動画（IMG_5903）は手持ちで正面から
# 近づく場面が多く、足の点（DeepLabCut）の左右の取り違えもあって、4 本の順は決めきれなかった
# （向こうへ歩き去る数コマは、後ろ足の左右を入れ替えるとこの順に合う）。
# 動画から取ったのは、足を振り出す時間（前足 6〜8 コマ ≒ 0.23 秒）。歩く速さから 1 周の時間を
# 決め、地面に着いている割合（duty）は 1 - 振り出しの時間 / 1 周の時間にする（遅いほど長く着く）。
# 地面に着いている間は、足が床の上で止まって見えるよう、脚の 2 つの関節を逆に解いて足先を
# 床に沿って後ろへ送る（IK）。この体の脚（肩から足裏まで 15 cm）は本物より短いので、
# 足の送りの幅には上限を付ける（長すぎると脚がのびきる）。

SWING_S = 0.23  # 足を振り出す時間（動画の前足）
WALK_LEGS = {"bl": 0.0, "fl": 0.25, "br": 0.5, "fr": 0.75}  # 足を着く時（1 周の割合）
# 駆け足（逃げるとき）：後ろ 2 本、前 2 本の順に着く（transverse gallop）
RUN_LEGS = {"bl": 0.0, "br": 0.1, "fl": 0.45, "fr": 0.55}
WALK_CROUCH_M = 0.02  # 歩くときは脚を少し曲げて体を下げる
SWEEP_MAX_M = 0.16  # 着いている間に足を送る幅の上限
LIFT_M = {"f": 0.032, "h": 0.026}  # 振り出しで足を上げる高さ
GAIT_FADE_S = 0.3
# 互いに比べるための、前の版の名前（1 周で進む長さの目安。速さで変わる：stride）
STRIDE_M = 0.26
BLEND_S = DIRECT_S

# 脚の 2 つの棒（ひじ・ひざまでと、足裏まで）。足裏 = 足先の楕円の下の端
LEG_LINKS = {"f": ((0.0, -0.075), (0.008, -0.075)), "h": ((0.02, -0.07), (-0.004, -0.080))}


def stride(speed: float, run: bool) -> float:
    """1 周で進む長さ（m）。猫は速いほど歩幅が伸びる。"""
    if run:
        return float(np.clip(0.35 + 0.25 * speed, 0.4, 0.8))
    return float(np.clip(0.14 + 0.32 * speed, 0.16, 0.3))


def duty(speed: float, run: bool) -> float:
    if run:
        return 0.38
    period = stride(speed, False) / max(speed, 0.05)
    return float(np.clip(1.0 - SWING_S / period, 0.55, 0.8))


def _rot(theta: float, v) -> np.ndarray:
    """y 軸まわりに theta 回した (x, z)（hinge の + と同じ向き）。"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c])


def _hang(v) -> float:
    """(x, z) を、真下から何度回した向きか（_rot と同じ向き）。"""
    return math.atan2(-v[0], -v[1])


def pivots(q: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
    """脚の付け根の (x, z)（猫の向きの座標、床が 0）と、付け根の体の前後の傾き（ラジアン）。"""
    i = JOINT_INDEX
    p0 = q[i["snoo_pitch"]]
    hips = np.array([-0.11, 0.17 + q[i["snoo_z"]]])
    hind = hips + _rot(p0, (-0.01, -0.02))
    waist = hips + _rot(p0, (0.07, 0.005))
    p1 = p0 + q[i["snoo_spine1_pitch"]]
    chest = waist + _rot(p1, (0.075, 0.01))
    p2 = p1 + q[i["snoo_spine2_pitch"]]
    front = chest + _rot(p2, (0.03, -0.035))
    return {"f": (front, p2), "h": (hind, p0)}


def foot(q: np.ndarray, leg: str) -> np.ndarray:
    """足裏の (x, z)（pivots と同じ座標）。"""
    kind = "f" if leg[0] == "f" else "h"
    (piv, pitch), (a, b) = pivots(q)[kind], LEG_LINKS[kind]
    upper, lower = _leg_joint_names(leg)
    t1, t2 = q[JOINT_INDEX[upper]], q[JOINT_INDEX[lower]]
    return piv + _rot(pitch + t1, a) + _rot(pitch + t1 + t2, b)


def _leg_joint_names(leg: str) -> tuple[str, str]:
    return (
        (f"snoo_{leg}_shoulder", f"snoo_{leg}_elbow")
        if leg[0] == "f"
        else (f"snoo_{leg}_hip", f"snoo_{leg}_knee")
    )


def leg_ik(q: np.ndarray, leg: str, target: np.ndarray) -> tuple[float, float]:
    """足裏を target (x, z) に置く、脚の 2 つの角度（ラジアン）。届かなければ伸ばして向ける。

    ひじ・ひざは、姿勢の本の形と同じ向き（+）に曲げる。
    """
    kind = "f" if leg[0] == "f" else "h"
    (piv, pitch), (a, b) = pivots(q)[kind], LEG_LINKS[kind]
    la, lb = math.hypot(*a), math.hypot(*b)
    t = np.asarray(target) - piv
    d = float(np.clip(math.hypot(*t), abs(la - lb) + 1e-6, la + lb))
    cos_d = (d * d - la * la - lb * lb) / (2 * la * lb)
    delta = math.acos(float(np.clip(cos_d, -1.0, 1.0)))
    beta = math.atan2(lb * math.sin(delta), la + lb * math.cos(delta))
    a_ang = _hang(t) - beta
    t1 = a_ang - _hang(a) - pitch
    t2 = delta - _hang(b) + _hang(a)
    return t1, t2


def gait_legs(
    base: np.ndarray,
    phase: float,
    speed: float,
    run: bool = False,
    frame: np.ndarray | None = None,
) -> np.ndarray:
    """歩く周期の位相 phase（1 で 1 周）での、脚の関節の角度（IK）。

    足を着く所は base の形の足の位置を中心にし、脚の付け根は frame（base に体の揺れを
    足した形。無ければ base）から解く。体が上下に揺れても、着いた足は床から離れない。
    返すのは frame に、脚の角度と体を下げた高さを入れたもの。
    """
    if frame is None:
        frame = base
    q = frame.copy()
    crouched = frame.copy()
    crouched[JOINT_INDEX["snoo_z"]] -= WALK_CROUCH_M * (0.6 if run else 1.0)
    du = duty(speed, run)
    sweep = min(stride(speed, run) * du, SWEEP_MAX_M * (1.4 if run else 1.0))
    legs = RUN_LEGS if run else WALK_LEGS
    for leg, touch in legs.items():
        kind = "f" if leg[0] == "f" else "h"
        home = foot(base, leg)
        lp = (phase - touch) % 1.0
        if lp < du:  # 着いている：前から後ろへ一定の速さで送る
            s = lp / du
            dx, lift = sweep / 2 - sweep * s, 0.0
        else:  # 振り出し：後ろから前へ、上げて戻す
            s = (lp - du) / (1 - du)
            dx = -sweep / 2 + sweep * float(_smooth(s))
            lift = LIFT_M[kind] * (1.3 if run else 1.0) * math.sin(math.pi * s)
        t1, t2 = leg_ik(crouched, leg, home + np.array([dx, lift]))
        upper, lower = _leg_joint_names(leg)
        q[JOINT_INDEX[upper]], q[JOINT_INDEX[lower]] = t1, t2
    q[JOINT_INDEX["snoo_z"]] = crouched[JOINT_INDEX["snoo_z"]]
    return q


def gait_body(phase: float, run: bool = False) -> np.ndarray:
    """歩く周期での、体・頭・しっぽの揺れ（足し分、ラジアンと m）。

    歩き：前足が着くたびに体が少し沈み（1 周に 2 回）、胴は左右に少しくねる。頭は体の揺れを
    打ち消して、ほとんど動かない（猫の頭は歩いてもぶれない）。しっぽは立てたまま、根元から
    先へ遅れて伝わる波でゆっくり揺れる。
    駆け足：背中を大きく曲げ伸ばしする（1 周に 1 回）。
    """
    d = np.zeros(len(JOINTS))
    i = JOINT_INDEX
    a = 2 * math.pi * phase
    if run:
        flex = math.sin(a)
        d[i["snoo_spine1_pitch"]] = math.radians(12) * flex
        d[i["snoo_spine2_pitch"]] = math.radians(8) * flex
        d[i["snoo_pitch"]] = -math.radians(6) * math.cos(a)
        d[i["snoo_z"]] = 0.012 * math.sin(a + 0.5)
        d[i["snoo_neck_pitch"]] = -math.radians(6) * flex
        d[i["snoo_tail1_pitch"]] = -math.radians(10) * flex
        return d
    sway = math.sin(a)
    d[i["snoo_spine1_yaw"]] = math.radians(3) * sway
    d[i["snoo_spine2_yaw"]] = -math.radians(3) * sway
    d[i["snoo_roll"]] = math.radians(1.5) * sway
    d[i["snoo_z"]] = 0.004 * math.cos(2 * a)
    d[i["snoo_neck_pitch"]] = -math.radians(2.5) * math.cos(2 * a)  # 体の上下を首で打ち消す
    d[i["snoo_neck_yaw"]] = -math.radians(2) * sway
    for k, lag in ((1, 0.0), (2, 0.6), (3, 1.2), (4, 1.8)):
        d[i[f"snoo_tail{k}_yaw"]] = math.radians(3 + 2 * k) * math.sin(a - lag)
    d[i["snoo_tail4_pitch"]] = math.radians(5) * math.sin(2 * a - 1.0)
    return d


def gait_offsets(phase: float, speed: float = 0.25, base: np.ndarray | None = None) -> np.ndarray:
    """walk の形を中心にした、歩く周期の位相での関節の足し分（試験と前の版のため）。"""
    if base is None:
        base = load_postures()["walk"]
    frame = base + gait_body(phase)
    return gait_legs(base, phase, speed, False, frame) - base


# --- 細かい動き（息・しっぽ・耳・まばたき・見る） ---

BREATH_HZ = {"sleep": 0.33, "lie": 0.4, "sit": 0.45, "groom": 0.45}  # 寝ると 1 分に 20 回
TAIL_FLICK_PER_S = {"alert": 0.25, "sniff": 0.2, "sit": 0.12, "lie": 0.08, "sleep": 0.02}
EAR_TWITCH_PER_S = {"sleep": 0.05}
EAR_TWITCH_AWAKE_PER_S = 0.15
LOOK_POSTURES = ("alert", "sit", "lie", "sniff", "walk")
LOOK_MAX_M = 3.5


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Event:
    start: float
    seconds: float
    kind: str
    value: float


class SnooAnimator:
    """snoo の姿勢（SnooPose.posture）と動き（歩いた長さ）から、関節の角度を作る。

    姿勢が変わったら、移り方の表（plan）の段を順にたどる。歩いている間は、歩いた長さと
    速さで歩く周期を進め、脚を IK で運ぶ（止まれば GAIT_FADE_S 秒で消える）。その上に、
    息・しっぽの先の揺れとはね・耳を動かす・まばたき・ロボットを見る、の細かい動きを足す。
    くじは seed で決まる（同じ seed・同じ時間の刻みなら同じ動き）。
    """

    def __init__(
        self,
        postures: dict[str, np.ndarray] | None = None,
        posture: str = "walk",
        seed: int = 0,
    ):
        self.postures = postures if postures is not None else load_postures()
        self.rng = np.random.default_rng(seed)
        self.target = posture_name(posture)
        self.reached = self.target
        self.stages: list[Stage] = []
        self.stage_t = 0.0
        self.start = self.postures[self.target].copy()
        self.current = self.start.copy()
        self.phase = 0.0
        self.gait = 0.0  # 歩く振りの強さ（0..1）
        self.run = 0.0  # 駆け足の強さ（0..1）
        self.speed = 0.0
        self.last_xy: tuple[float, float] | None = None
        self.t = 0.0
        self.events: list[Event] = []
        self.next_event = {"tail": 1.0, "ear": 2.0, "blink": 3.0, "glance": 2.0}
        self.look = np.zeros(2)  # 首と頭を回している角度（yaw・pitch、ラジアン）
        self.attend = 1.0
        self.lid = math.radians(LID_OPEN_DEG)
        self.qpos = self.start.copy()

    # 前の版と同じ名前（試験・呼ぶ側のため）
    @property
    def blend(self) -> float:
        return 1.0 if not self.stages else 0.0

    def update(self, dt: float, pose, look_at=None) -> np.ndarray:
        """dt 秒進めて、関節の角度（JOINTS の順、ラジアン）を返す。pose は SnooPose。

        look_at は見る先（ロボットのカメラの (x, y, z)、部屋の座標）。無ければ見ない。
        """
        self.t += dt
        raw = getattr(pose, "posture", "walk") or "walk"
        name = posture_name(raw)
        if name != self.target:
            self._go(name)
        base = self._advance_stages(dt)
        # 歩く周期
        xy = (pose.x, pose.y)
        ds = 0.0 if self.last_xy is None else math.dist(xy, self.last_xy)
        self.last_xy = xy
        if dt > 0:
            k = min(1.0, dt / 0.3)
            self.speed += k * (min(ds / dt, 3.0) - self.speed)
        running = raw == "run"
        walking = name == "walk" and getattr(pose, "moving", False) and ds > 0
        st = stride(max(self.speed, 0.05), running)
        self.phase = (self.phase + ds / st) % 1.0
        step = dt / GAIT_FADE_S
        self.gait = min(1.0, self.gait + step) if walking else max(0.0, self.gait - step)
        self.run = min(1.0, self.run + step) if running and walking else max(0.0, self.run - step)
        q = base
        if self.gait > 0:
            # 体の揺れ（歩きと駆け足を混ぜる）を足した形から、脚を IK で解く
            body = (1 - self.run) * gait_body(self.phase) + self.run * gait_body(self.phase, True)
            frame = base + self.gait * body
            run = self.run > 0.5
            legs = gait_legs(base, self.phase, max(self.speed, 0.5 if run else 0.05), run, frame)
            q = frame + self.gait * (legs - frame)
        q = q + self._idle(dt, pose, look_at, name)
        lo, hi = limits()
        self.qpos = np.clip(q, lo, hi)
        return self.qpos

    def _go(self, name: str) -> None:
        # いまの形から次の形へ（途中で変わっても、いまの形から始めるので飛ばない）。
        # 移っている途中なら、向かっていた姿勢から道を引く
        src = self.target if self.stages else self.reached
        self.stages = plan(src, name)
        self.target = name
        self.stage_t = 0.0
        self.start = self.current.copy()

    def _advance_stages(self, dt: float) -> np.ndarray:
        while self.stages:
            stage = self.stages[0]
            self.stage_t += dt
            u = self.stage_t / stage.seconds
            if u < 1.0:
                goal = self.postures[stage.target]
                w = stage_weights(stage, u)
                self.current = self.start + w * (goal - self.start) + _bump(stage, u)
                return self.current
            # 段を終えた：余った時間は次の段へ
            dt = self.stage_t - stage.seconds
            self.stages.pop(0)
            self.start = self.postures[stage.target].copy()
            self.current = self.start.copy()
            self.stage_t = 0.0
        self.reached = self.target
        self.current = self.postures[self.target].copy()
        return self.current

    def posture_settled(self) -> bool:
        return not self.stages

    # --- 細かい動き ---

    def _schedule(self, key: str, rate: float) -> bool:
        """key のできごとが起きる時か（間はくじ。rate は 1 秒あたりの回数）。"""
        if rate <= 0:
            return False
        if self.t >= self.next_event[key]:
            self.next_event[key] = self.t + float(self.rng.exponential(1.0 / rate)) + 0.3
            return True
        return False

    def _idle(self, dt: float, pose, look_at, name: str) -> np.ndarray:
        d = np.zeros(len(JOINTS))
        i = JOINT_INDEX
        settled = not self.stages
        asleep = name == "sleep" and settled
        still = 1.0 - self.gait
        # 息：胸と腹がふくらむ（胸の前後の傾きと体の高さ）。寝ると深く遅い
        hz = BREATH_HZ.get(name, 0.5)
        br = math.sin(2 * math.pi * hz * self.t)
        amp = 1.6 if asleep else 1.0
        d[i["snoo_spine2_pitch"]] += math.radians(0.9 * amp) * br * still
        d[i["snoo_z"]] += 0.0012 * amp * br * still
        # しっぽ：先がゆっくり揺れ、ときどき先をはねる
        slow = math.sin(2 * math.pi * 0.17 * self.t)
        tail_amp = 0.3 if asleep else 1.0
        d[i["snoo_tail3_yaw"]] += math.radians(5) * tail_amp * slow * still
        d[i["snoo_tail4_yaw"]] += (
            math.radians(9) * tail_amp * math.sin(2 * math.pi * 0.17 * self.t - 0.9) * still
        )
        if self._schedule("tail", TAIL_FLICK_PER_S.get(name, 0.05)):
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            self.events.append(Event(self.t, 0.45, "tail", sign * float(self.rng.uniform(15, 30))))
        # 耳：ときどき片方の耳を後ろへはねる
        if self._schedule("ear", EAR_TWITCH_PER_S.get(name, EAR_TWITCH_AWAKE_PER_S)):
            side = "l" if self.rng.random() < 0.5 else "r"
            self.events.append(Event(self.t, 0.28, f"ear_{side}", float(self.rng.uniform(25, 45))))
        # まばたき：起きていれば時々。ロボットを見ている間は、ゆっくりのまばたき（猫のあいさつ）
        if not asleep and self._schedule("blink", 0.18):
            slow_blink = self.attend > 0.5 and name in ("alert", "sit", "lie")
            self.events.append(Event(self.t, 1.1 if slow_blink else 0.25, "blink", 1.0))
        lid = LID_CLOSED_DEG if asleep else LID_OPEN_DEG
        if name == "lie" and settled:
            lid = 25.0  # 伏せているときは少し眠そうに
        keep = []
        for e in self.events:
            u = (self.t - e.start) / e.seconds
            if u >= 1.0:
                continue
            keep.append(e)
            s = math.sin(math.pi * u)
            if e.kind == "tail":
                d[i["snoo_tail4_yaw"]] += math.radians(e.value) * s
                d[i["snoo_tail3_yaw"]] += math.radians(e.value * 0.4) * math.sin(math.pi * u * 0.9)
            elif e.kind.startswith("ear"):
                side = e.kind[-1]
                sign = 1.0 if side == "l" else -1.0
                # 後ろへ（左耳は -、右耳は +）
                d[i[f"snoo_ear_{side}"]] += -sign * math.radians(e.value) * (s**0.5)
            elif e.kind == "blink" and not asleep:
                close = min(1.0, 1.6 * s)
                lid = lid + (LID_CLOSED_DEG - lid) * close
        self.events = keep
        # まぶたは少し遅れて動く
        k = min(1.0, dt / 0.05) if dt > 0 else 1.0
        self.lid += k * (math.radians(lid) - self.lid)
        d[i["snoo_lid"]] = self.lid
        # 見る：ロボット（look_at）の方へ首と頭を回す。ときどきよそ見をする
        d += self._look(dt, pose, look_at, name, settled and not asleep)
        # 嗅ぐ・毛づくろい：決まった形の上で、鼻と前足を細かく動かす
        if name == "sniff" and settled:
            d[i["snoo_head_pitch"]] += math.radians(2.5) * math.sin(2 * math.pi * 5.0 * self.t)
            d[i["snoo_neck_yaw"]] += math.radians(14) * math.sin(2 * math.pi * 0.3 * self.t)
            d[i["snoo_ear_l"]] += math.radians(10)
            d[i["snoo_ear_r"]] -= math.radians(10)
        if name == "groom" and settled:
            # 舌でなめる（頭を小さく上下に 3 回/秒）と、前足で顔をなでる（1 回/1.5 秒）
            lick = math.sin(2 * math.pi * 3.0 * self.t)
            wipe = math.sin(2 * math.pi * 0.65 * self.t)
            d[i["snoo_head_pitch"]] += math.radians(5) * lick
            d[i["snoo_fl_shoulder"]] += math.radians(12) * wipe
            d[i["snoo_fl_elbow"]] += math.radians(10) * wipe
            d[i["snoo_head_roll"]] += math.radians(8) * wipe
            d[i["snoo_lid"]] += math.radians(40)
        return d

    def _look(self, dt, pose, look_at, name, awake) -> np.ndarray:
        d = np.zeros(len(JOINTS))
        i = JOINT_INDEX
        want = np.zeros(2)
        if look_at is not None and awake and name in LOOK_POSTURES:
            dx, dy = look_at[0] - pose.x, look_at[1] - pose.y
            dist = math.hypot(dx, dy)
            err = _wrap(math.atan2(dy, dx) - pose.yaw)
            # 歩いている間は前を見ることが多い。止まっていれば、ロボットをよく見る
            if self._schedule("glance", 0.25 if name != "alert" else 0.12):
                p_look = 0.85 if name in ("alert", "sniff") else (0.35 if name == "walk" else 0.65)
                self.attend = 1.0 if self.rng.random() < p_look else 0.0
            if dist < LOOK_MAX_M and abs(err) < math.radians(150) and self.attend > 0.5:
                head_h = 0.22 if name in ("alert", "walk", "sniff") else 0.18
                pitch = math.atan2(head_h - (look_at[2] if len(look_at) > 2 else 0.08), dist)
                want = np.array([err, pitch])
        # 首を回す速さ：猫は見る先へ素早く向く（0.15 秒ほど）
        k = min(1.0, dt / 0.15) if dt > 0 else 1.0
        self.look += k * (want - self.look)
        yaw, pitch = self.look
        base_yaw = self.current[i["snoo_neck_yaw"]] + self.current[i["snoo_head_yaw"]]
        if abs(yaw) > 1e-4:
            # 姿勢の本の首の向きを打ち消してから、首 55%・頭 45% で回す
            frac = min(1.0, abs(yaw) / 0.2)
            d[i["snoo_neck_yaw"]] += 0.55 * yaw - frac * 0.55 * base_yaw
            d[i["snoo_head_yaw"]] += 0.45 * yaw - frac * 0.45 * base_yaw
            d[i["snoo_head_pitch"]] += pitch * 0.7
            # 見るときは耳を前へ
            d[i["snoo_ear_l"]] += math.radians(12) * frac
            d[i["snoo_ear_r"]] -= math.radians(12) * frac
        return d

    def base(self) -> np.ndarray:
        """細かい動きと歩く振りを足す前の形（姿勢の間を移っている途中なら、その途中の形）。"""
        return self.current.copy()
