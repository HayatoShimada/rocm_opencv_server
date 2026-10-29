"""PiCar-X（85pi4 の snoo_trucar）と部屋と snoo を MuJoCo に置き、間取り作りを机の上で試す。

    MUJOCO_GL=egl .venv/bin/python -m scripts.snoo_sim_view --explore 20 --out out/sim
実機で 1 周（撮る → 地図を直す → 1 歩走る）を回すと、電池と時間と猫の機嫌を使う。
ここでは同じ周りを、本物の部屋の代わりに MuJoCo の部屋で回す。

- 部屋：箱（Block）の集まり。作り物の部屋（synthetic_room）か、app/snoo_mapper.py が書いた
  間取り（room.json・grid.png・high.png）から作る（room_from_map）。
- 車：自転車モデル（物理は解かない。姿勢を直接置く）。Pi と同じ指令（steer_deg は右が +、
  speed、distance_cm）で動く。寸法は cau-se/ARCHES-PiCar-X の URDF（Apache-2.0）から。
  メッシュ（.dae）は MuJoCo で読めないので、箱と円柱で同じ寸法に作る。
- カメラ：パン・チルトの台の上に、実機と同じ画角（横 54°、640x480）の MuJoCo のカメラを置く。
- 地図の更新：VGGT の代わりに、カメラの画素ごとの光線を部屋に当て（mj_multiRay）、
  当たった点の高さで床・障害物・「上」の層に分けて app/roommap.Grid に足す。
  分け方と次の 1 歩の決め方は app/snoo_mapper.py と同じ（decide をそのまま使う）。
- snoo：v1 は箱と球の猫が、止まる・歩くを繰り返す（RandomWalkSnoo）。
  step(dt, robot) の形を守れば、あとで DeepLabCut の関節から動かすものに差し替えられる。

描くのは app/snoo_sim_render.py。ここは描かない（GL が無くても試験できるように）。

座標は m。原点は最初のステーションのカメラの真下の床、x は最初の車の向き、y は左、
yaw は左回りが +（app/roommap.py と同じ）。車の姿勢は後ろの車軸の中央で持ち、
地図に渡すステーションの位置はカメラの真下にする（app/snoo_mapper.py と同じ）。
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.roommap import (
    CAM_X_M,
    CAR_FRONT_M,
    CAR_HEIGHT_M,
    CAR_REAR_M,
    CAR_WIDTH_M,
    FREE,
    FREE_HIT,
    OCCUPIED,
    OCCUPIED_HIT,
    REVERSE_MAX_M,
    SONAR_HALF_DEG,
    SONAR_RANGE_M,
    UNKNOWN,
    VIEW_RANGE_M,
    WHEELBASE_M,
    Grid,
    Plan,
    Pose,
    arc,
    consume,
    mark_driven,
    mark_sonar,
)
from app.snoo_mapper import (
    CLUSTER_STATIONS,
    RECENT_STATIONS,
    History,
    MapParams,
    StationResult,
    choose_window,
    clustered,
    decide,
    or_reverse,
)

# GL の選び方は mujoco を読み込む前に決める（画面の無いマシンでも描けるよう EGL にする）
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

# --- 車の寸法（m）。原点は後ろの車軸の中央の床 ---
# URDF から：ホイールベース 14.0 cm、トレッド 12 cm、車輪の半径 3.26 cm、ハンドル ±40°
TRACK_M = 0.12
WHEEL_R_M = 0.0326
# 外形 約 26 x 19 cm、高さ 約 12 cm。前後の端（CAR_FRONT_M・CAR_REAR_M）と、後ろの車軸から
# カメラまで（CAM_X_M）は、地図のサーバーも使うので app/roommap.py に置く
CAM_H_M = 0.08  # 床からレンズの中心まで（SD カード 2.5 枚分と測った）
HFOV_DEG = 54.0
IMG_W, IMG_H = 640, 480
VFOV_DEG = math.degrees(2 * math.atan(math.tan(math.radians(HFOV_DEG / 2)) * IMG_H / IMG_W))
PAN_LIMIT_DEG = 35.0
STEER_LIMIT_DEG = 30.0  # URDF は ±40° だが、Pi が ±30° で止める
# 前の超音波の高さ（円すいの広さ・届く距離は app/roommap.py）
SONAR_H_M = 0.045
# Pi の speed 1 あたりの速さ（m/s）。1 回目の実走（explore_20261029_210008）の VGGT の位置から
# 測った（speed 12 で約 20 cm/s）
SPEED_M_S = 0.017
# 1 ステーションで撮る向き。実機の scan.json と同じ（パン -35〜35° を 10° おき × チルト 3 段）
SCAN_PANS = tuple(range(-35, 36, 10))
SCAN_TILTS = (-15, 0, 15)
# 1 コマ撮る間に進む時間（サーボを回して止まるまで。測っていない）
SCAN_FRAME_S = 0.4


@dataclass(frozen=True)
class Pace:
    """1 周（撮る → 地図を直す → 走る）の撮り方と時間。Sim.t を実機の時計に合わせるため。

    既定は前のシミュレーター（24 コマ × 0.4 秒、地図 3 秒）。REAL_PACE は 3 回目の実走
    （explore_20261029_223944）：チルト -15・0 × パン -30〜30 を 15° おきの 10 コマを
    1.44 秒おきに撮り（1 か所 14.4 秒）、地図のサーバーは平均 11 秒で返した。走る時間は
    速さから出す（speed 12 で 20 cm/s。実走の prev_step の seconds と合う）。
    """

    pans: tuple[float, ...] = SCAN_PANS
    tilts: tuple[float, ...] = SCAN_TILTS
    hold_s: float = SCAN_FRAME_S  # 1 コマにかかる時間（首を回して止まるまで）
    server_s: float = 3.0  # 撮り終えてから返事が来るまで（送る・VGGT・間取り）


REAL_PACE = Pace(tuple(range(-30, 31, 15)), (-15, 0), 1.44, 11.0)
# 走るときの首の向き。通り道の手前が映るよう少し下を向く（Pi の値は決まっていない）
DRIVE_TILT_DEG = -15.0
# 通り道（clear_cm）の帯：車の幅の両脇に足す余裕と、先を見る長さ（app/clearance.py と同じ）
CORRIDOR_MARGIN_M = 0.02
CLEAR_MAX_M = 1.0
# 行き場が無いときに来た道を戻る長さ（地図のサーバーの reverse と同じ）
BACKTRACK_M = REVERSE_MAX_M
# 走っている間、前がこれより近くまでふさがったら止まる（Pi が clear_cm を見て止まる代わり）
STOP_CLEAR_M = 0.05

# 格子（app/roommap.Grid の既定。間取りの room.json も同じ）の原点と大きさ
GRID_ORIGIN_M = Grid().origin
RES_M = Grid().res

# MuJoCo の geom の組。光線は 0（部屋）と 1（snoo）にだけ当てる
GROUP_ROOM, GROUP_SNOO, GROUP_CAR, GROUP_DIM = 0, 1, 2, 4
RAY_GROUPS = np.array([1, 1, 0, 0, 0, 0], np.uint8)

# snoo の大きさ（m）。体の中心の床が原点、x が頭の向き。長さ 約 48 cm（しっぽまで）、高さ 約 27 cm
SNOO_FRONT_M = 0.22
SNOO_REAR_M = -0.26
SNOO_HALF_W_M = 0.07
SNOO_HEIGHT_M = 0.27
SNOO_RADIUS_M = 0.12  # 歩くときにぶつからないかを見る円（しっぽは含めない）


# --- 部屋 ---


@dataclass(frozen=True)
class Block:
    """部屋の中の、軸にそろえた箱。lo・hi は角の座標（x, y, z）。"""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.55, 0.45, 0.35, 1.0)
    name: str = ""


@dataclass
class Room:
    blocks: list[Block]
    bounds: tuple[float, float, float, float]  # 描く範囲 (x0, y0, x1, y1)
    start: Pose = Pose(0.0, 0.0, 0.0)  # 最初のステーション（カメラの真下）
    # 床と分かっているセル（間取りから作ったとき）。そこ以外の床に当たった光線は使わない
    # （見ていない所を床と言わない）。None は全部床
    floor_ok: np.ndarray | None = None
    # 暗くする所（間取りで見ていなかったセル）。(x0, y0, x1, y1) の長方形
    dim: list[tuple[float, float, float, float]] = field(default_factory=list)
    name: str = "room"
    # 間取りから作ったときの元の地図（描くときに使う）
    source_state: np.ndarray | None = None
    source_high: np.ndarray | None = None
    source_stations: list[Pose] = field(default_factory=list)


def merge_cells(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """True のセルを、少ない数の長方形 (i0, i1, j0, j1)（半開区間）にまとめる。

    5 cm のセルを 1 つずつ箱にすると、部屋 1 つで数千の geom になり、描くのも光線も遅くなる。
    行ごとに右へ伸ばし、同じ幅で下へ伸ばせるだけ伸ばす（いちばん少ないとは限らないが十分）。
    """
    m = np.asarray(mask, bool).copy()
    n0, n1 = m.shape
    rects = []
    for i in range(n0):
        j = 0
        while j < n1:
            rest = np.flatnonzero(m[i, j:])
            if not len(rest):
                break
            j0 = j + int(rest[0])
            j1 = j0
            while j1 < n1 and m[i, j1]:
                j1 += 1
            i1 = i + 1
            while i1 < n0 and m[i1, j0:j1].all():
                i1 += 1
            m[i:i1, j0:j1] = False
            rects.append((i, i1, j0, j1))
            j = j1
    return rects


def _box(x0, y0, x1, y1, z0, z1, rgba, name="") -> Block:
    return Block((x0, y0, z0), (x1, y1, z1), rgba, name)


def synthetic_room() -> Room:
    """4 x 3 m の居間。机（天板の下はくぐれる）・椅子 2 脚・ソファ・テレビ台・箱。

    壁は 50 cm にする（本物より低いが、上から見たときに部屋の中が隠れないように）。
    """
    wall, wood, dark = (0.86, 0.84, 0.80, 1.0), (0.62, 0.45, 0.30, 1.0), (0.30, 0.30, 0.34, 1.0)
    sofa, box = (0.35, 0.45, 0.60, 1.0), (0.78, 0.62, 0.40, 1.0)
    x0, y0, x1, y1, t, h = -0.7, -1.4, 3.3, 1.6, 0.05, 0.5
    b = [
        _box(x0 - t, y0 - t, x1 + t, y0, 0, h, wall, "wall_s"),
        _box(x0 - t, y1, x1 + t, y1 + t, 0, h, wall, "wall_n"),
        _box(x0 - t, y0, x0, y1, 0, h, wall, "wall_w"),
        _box(x1, y0, x1 + t, y1, 0, h, wall, "wall_e"),
        # ソファ（北の壁ぎわ）：座面・背もたれ・ひじ掛け
        _box(0.3, 0.95, 2.1, 1.55, 0, 0.40, sofa, "sofa_seat"),
        _box(0.3, 1.35, 2.1, 1.55, 0.40, 0.80, sofa, "sofa_back"),
        _box(0.3, 0.95, 0.48, 1.35, 0.40, 0.58, sofa, "sofa_arm_w"),
        _box(1.92, 0.95, 2.1, 1.35, 0.40, 0.58, sofa, "sofa_arm_e"),
        # テレビ台（東の壁ぎわ）
        _box(2.9, -0.4, 3.3, 0.9, 0, 0.45, dark, "tv_stand"),
        # 段ボール箱（低い障害物）
        _box(0.45, -1.15, 0.80, -0.85, 0, 0.25, box, "box"),
    ]
    # 机：天板は高さ 70 cm（車も猫も下をくぐれる）。脚は 5 cm 角
    tx0, ty0, tx1, ty1 = 1.4, -0.85, 2.6, -0.1
    b.append(_box(tx0, ty0, tx1, ty1, 0.68, 0.72, wood, "table_top"))
    for k, (lx, ly) in enumerate(
        [
            (tx0 + 0.03, ty0 + 0.03),
            (tx1 - 0.08, ty0 + 0.03),
            (tx0 + 0.03, ty1 - 0.08),
            (tx1 - 0.08, ty1 - 0.08),
        ]
    ):
        b.append(_box(lx, ly, lx + 0.05, ly + 0.05, 0, 0.68, wood, f"table_leg{k}"))
    # 椅子 2 脚：座面は高さ 45 cm、脚は 3 cm 角、背もたれ
    for k, (cx, cy, back) in enumerate([(2.0, -1.12, "s"), (2.88, -0.48, "e")]):
        sx0, sy0, sx1, sy1 = cx - 0.21, cy - 0.21, cx + 0.21, cy + 0.21
        b.append(_box(sx0, sy0, sx1, sy1, 0.43, 0.46, wood, f"chair{k}_seat"))
        for j, (lx, ly) in enumerate(
            [(sx0, sy0), (sx1 - 0.03, sy0), (sx0, sy1 - 0.03), (sx1 - 0.03, sy1 - 0.03)]
        ):
            b.append(_box(lx, ly, lx + 0.03, ly + 0.03, 0, 0.43, wood, f"chair{k}_leg{j}"))
        if back == "s":
            b.append(_box(sx0, sy0, sx1, sy0 + 0.03, 0.46, 0.85, wood, f"chair{k}_back"))
        else:
            b.append(_box(sx1 - 0.03, sy0, sx1, sy1, 0.46, 0.85, wood, f"chair{k}_back"))
    return Room(b, (x0 - t, y0 - t, x1 + t, y1 + t), name="synthetic")


def _png_to_grid(img: np.ndarray) -> np.ndarray:
    """PNG の向き（行 0 が +y の端、列 0 が -x の端）を格子 [ix, iy] に戻す（grid_png の逆）。"""
    return img[::-1].T


def load_map(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray, Grid, dict | None]:
    """room.json（かその置き場）を読み、(room.json, 格子の状態, 上の層, Grid, plan.json)。"""
    p = Path(path)
    if p.is_dir():
        p = p / "room.json"
    room = json.loads(p.read_text())
    layers = room["layers"]
    img = cv2.imread(str(p.parent / layers["static"]["png"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p.parent / layers["static"]["png"])
    raw = _png_to_grid(img)
    state = np.full(raw.shape, UNKNOWN, np.uint8)
    state[raw >= 200] = FREE
    state[raw <= 50] = OCCUPIED
    hi_img = cv2.imread(str(p.parent / layers["high"]["png"]), cv2.IMREAD_GRAYSCALE)
    high = _png_to_grid(hi_img) > 127 if hi_img is not None else np.zeros_like(state, bool)
    res = room["resolution_cm"] / 100
    grid = Grid(size_m=room["size"][0] * res, res=res)
    grid.origin = float(room["origin"][0])
    plan_path = p.parent / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    return room, state, high, grid, plan


def _pose(d: dict) -> Pose:
    return Pose(float(d["x"]), float(d["y"]), math.radians(float(d["yaw_deg"])))


def room_from_map(
    path: str | Path,
    high_board_m: float = 0.40,
    margin_m: float = 0.5,
    unseen_floor: bool = False,
) -> Room:
    """間取り（app/snoo_mapper.py の room.json・grid.png・high.png）から部屋を作る。

    - ふさがったセル：高さ「車の高さ + 余裕」（room.json の min_height_m）の箱
    - 上の層のセル：高さ high_board_m の薄い板（間取りは高さを持たないので決め打ち）
    - 見ていないセル：床を暗くする（光線が当たっても床とは言わない）
    隣り合うセルは長方形にまとめる。

    unseen_floor なら、見ていないセルも床にし、まわり（見た所 + margin_m）を壁で囲む。
    実走の間取りは窓の中だけで狭い（3 回目の実走で床 1.4 m²）ので、探る速さを比べるときは
    「見た障害物はそのまま、見ていない所は床が続く部屋」にする。
    """
    room, state, high, grid, _ = load_map(path)
    top = float(room["layers"]["high"]["legend"].get("min_height_m", 0.15))
    known = np.argwhere((state != UNKNOWN) | high)
    if not len(known):
        known = np.array([[grid.n // 2, grid.n // 2]])
    m = int(margin_m / grid.res)
    i0, j0 = np.maximum(known.min(0) - m, 0)
    i1, j1 = np.minimum(known.max(0) + m + 1, grid.n)

    def rect_xy(r):
        a, b, c, d = r
        return (
            grid.origin + (i0 + a) * grid.res,
            grid.origin + (j0 + c) * grid.res,
            grid.origin + (i0 + b) * grid.res,
            grid.origin + (j0 + d) * grid.res,
        )

    sub = state[i0:i1, j0:j1]
    blocks = []
    for k, r in enumerate(merge_cells(sub == OCCUPIED)):
        x0, y0, x1, y1 = rect_xy(r)
        blocks.append(_box(x0, y0, x1, y1, 0, top, (0.35, 0.33, 0.32, 1.0), f"occ{k}"))
    for k, r in enumerate(merge_cells(high[i0:i1, j0:j1] & (sub != OCCUPIED))):
        x0, y0, x1, y1 = rect_xy(r)
        blocks.append(
            _box(x0, y0, x1, y1, high_board_m, high_board_m + 0.02, (0.4, 0.55, 0.8, 0.9), f"hi{k}")
        )
    stations = [_pose(s["pose"]) for s in room.get("stations", [])]
    x0, y0, x1, y1 = rect_xy((0, i1 - i0, 0, j1 - j0))
    if unseen_floor:
        dim, floor_ok = [], np.zeros_like(state, bool)
        floor_ok[i0:i1, j0:j1] = sub != OCCUPIED
        wall, t, h = (0.86, 0.84, 0.80, 1.0), 0.05, 0.5
        blocks += [
            _box(x0 - t, y0 - t, x1 + t, y0, 0, h, wall, "wall_s"),
            _box(x0 - t, y1, x1 + t, y1 + t, 0, h, wall, "wall_n"),
            _box(x0 - t, y0, x0, y1, 0, h, wall, "wall_w"),
            _box(x1, y0, x1 + t, y1, 0, h, wall, "wall_e"),
        ]
    else:
        dim = [rect_xy(r) for r in merge_cells(sub == UNKNOWN)]
        floor_ok = state == FREE
    return Room(
        blocks,
        (x0, y0, x1, y1),
        start=stations[-1] if stations else _pose(room["start_pose"]),
        floor_ok=floor_ok,
        dim=dim,
        name=Path(path).name + ("_open" if unseen_floor else ""),
        source_state=state,
        source_high=high,
        source_stations=stations,
    )


def low_aabbs(room: Room, below: float) -> np.ndarray:
    """下の端が below より低い箱の、床に落とした長方形 (K, 4) = [x0, y0, x1, y1]。"""
    rows = [(b.lo[0], b.lo[1], b.hi[0], b.hi[1]) for b in room.blocks if b.lo[2] < below]
    return np.array(rows, np.float64).reshape(-1, 4)


# --- MJCF ---


def _f(*v) -> str:
    return " ".join(f"{x:.4g}" for x in v)


def _car_xml() -> str:
    """PiCar-X を箱と円柱で。車輪の位置・ハンドルの軸は URDF と同じ。"""
    wheel = f'type="cylinder" size="{WHEEL_R_M} 0.013" euler="90 0 0" rgba="0.06 0.06 0.06 1"'
    hub = 'type="box" size="0.022 0.0135 0.004" rgba="0.85 0.75 0.2 1"'
    rear = "".join(
        f'<body name="wheel_r{s}" pos="0 {y} {WHEEL_R_M}">'
        f'<joint name="wheel_r{s}" type="hinge" axis="0 1 0"/>'
        f"<geom {wheel}/><geom {hub}/></body>"
        for s, y in (("l", 0.08), ("r", -0.08))
    )
    front = "".join(
        f'<body name="steer_f{s}" pos="{WHEELBASE_M} {y} {WHEEL_R_M}">'
        f'<joint name="steer_f{s}" type="hinge" axis="0 0 1"/>'
        f'<body pos="0 {math.copysign(0.032, y)} 0">'
        f'<joint name="wheel_f{s}" type="hinge" axis="0 1 0"/>'
        f"<geom {wheel}/><geom {hub}/></body></body>"
        for s, y in (("l", 0.048), ("r", -0.048))
    )
    cx = (CAR_FRONT_M + CAR_REAR_M) / 2
    half = (CAR_FRONT_M - CAR_REAR_M) / 2
    return f"""
<body name="car" childclass="car">
  <freejoint name="car"/>
  <geom name="chassis" type="box" pos="{_f(cx, 0, 0.042)}" size="{_f(half, 0.055, 0.008)}"
        rgba="0.16 0.16 0.18 1"/>
  <geom name="battery" type="box" pos="0.03 0 0.075" size="0.075 0.05 0.025"
        rgba="0.12 0.35 0.75 1"/>
  <geom name="hat" type="box" pos="0.03 0 0.108" size="0.06 0.045 0.008" rgba="0.1 0.5 0.3 1"/>
  <geom name="sonar_board" type="box" pos="{_f(CAR_FRONT_M - 0.006, 0, SONAR_H_M)}"
        size="0.004 0.024 0.011" rgba="0.1 0.3 0.6 1"/>
  <geom name="sonar_l" type="cylinder" pos="{_f(CAR_FRONT_M, 0.013, SONAR_H_M)}"
        size="0.008 0.004" euler="0 90 0" rgba="0.85 0.85 0.9 1"/>
  <geom name="sonar_r" type="cylinder" pos="{_f(CAR_FRONT_M, -0.013, SONAR_H_M)}"
        size="0.008 0.004" euler="0 90 0" rgba="0.85 0.85 0.9 1"/>
  {rear}{front}
  <body name="pan" pos="{_f(CAM_X_M, 0, 0.055)}">
    <joint name="pan" type="hinge" axis="0 0 -1" range="-{PAN_LIMIT_DEG} {PAN_LIMIT_DEG}"/>
    <geom type="box" size="0.012 0.018 0.012" rgba="0.15 0.15 0.15 1"/>
    <body name="tilt" pos="{_f(0, 0, CAM_H_M - 0.055)}">
      <joint name="tilt" type="hinge" axis="0 -1 0" range="-35 65"/>
      <geom type="box" pos="-0.006 0 0" size="0.004 0.02 0.02" rgba="0.1 0.45 0.25 1"/>
      <geom type="cylinder" pos="-0.001 0 0" size="0.006 0.002" euler="0 90 0"
            rgba="0.02 0.02 0.02 1"/>
      <camera name="car_cam" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="{VFOV_DEG:.4f}"/>
    </body>
  </body>
</body>"""


def _snoo_xml() -> str:
    """箱と球の猫（体・頭・耳・脚・しっぽ）。毛の色と組は default の snoo の class で付ける。"""
    dark = "0.12 0.1 0.08 1"
    legs = "".join(
        f'<geom type="capsule" fromto="{_f(x, y, 0.12, x, y, 0.018)}" size="0.018"/>'
        for x in (0.07, -0.08)
        for y in (0.04, -0.04)
    )
    return f"""
<body name="snoo" pos="1 -0.5 0" childclass="snoo">
  <freejoint name="snoo"/>
  <geom name="snoo_body" type="capsule" fromto="-0.09 0 0.15 0.08 0 0.155" size="0.07"/>
  <geom name="snoo_head" type="sphere" pos="0.15 0 0.2" size="0.055"/>
  <geom type="sphere" pos="0.195 0 0.185" size="0.026"/>
  <geom type="ellipsoid" pos="0.155 0.03 0.25" size="0.012 0.018 0.026"/>
  <geom type="ellipsoid" pos="0.155 -0.03 0.25" size="0.012 0.018 0.026"/>
  <geom type="sphere" pos="0.198 0.022 0.212" size="0.008" rgba="{dark}"/>
  <geom type="sphere" pos="0.198 -0.022 0.212" size="0.008" rgba="{dark}"/>
  {legs}
  <geom name="snoo_tail" type="capsule" fromto="-0.14 0 0.16 -0.24 0 0.25" size="0.018"/>
</body>"""


def build_mjcf(room: Room) -> str:
    """部屋・車・snoo の MJCF。物理は解かない（接触も重力も切る）。"""
    x0, y0, x1, y1 = room.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    extent = max(x1 - x0, y1 - y0)
    geoms = []
    for k, b in enumerate(room.blocks):
        c = [(a + z) / 2 for a, z in zip(b.lo, b.hi, strict=True)]
        s = [max((z - a) / 2, 1e-4) for a, z in zip(b.lo, b.hi, strict=True)]
        geoms.append(
            f'<geom name="b{k}_{b.name}" type="box" group="{GROUP_ROOM}" pos="{_f(*c)}" '
            f'size="{_f(*s)}" rgba="{_f(*b.rgba)}"/>'
        )
    for k, (a, b, c, d) in enumerate(room.dim):
        geoms.append(
            f'<geom name="dim{k}" type="box" group="{GROUP_DIM}" '
            f'pos="{_f((a + c) / 2, (b + d) / 2, 0.0008)}" '
            f'size="{_f((c - a) / 2, (d - b) / 2, 0.0008)}" rgba="0.08 0.08 0.1 0.75"/>'
        )
    return f"""<mujoco model="snoo_sim">
  <compiler angle="degree" boundmass="0.001" boundinertia="1e-7"/>
  <option><flag contact="disable" gravity="disable"/></option>
  <statistic center="{_f(cx, cy, 0.2)}" extent="{extent:.3f}"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <map znear="{0.012 / extent:.5f}" zfar="60"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.80 0.76 0.70"
             rgb2="0.75 0.71 0.65" width="256" height="256" mark="edge"
             markrgb="0.6 0.57 0.52"/>
    <material name="floor" texture="floor" texrepeat="2 2" texuniform="true"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.32 0.35 0.4"
             rgb2="0.08 0.08 0.1" width="64" height="64"/>
  </asset>
  <default>
    <geom contype="0" conaffinity="0"/>
    <default class="car"><geom group="{GROUP_CAR}"/></default>
    <default class="snoo"><geom group="{GROUP_SNOO}" rgba="0.78 0.56 0.34 1"/></default>
  </default>
  <worldbody>
    <light name="top" pos="{_f(cx, cy, 3.5)}" dir="0 0 -1" diffuse="0.55 0.55 0.55"
           castshadow="true" cutoff="75"/>
    <light directional="true" dir="0.3 0.4 -1" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" group="{GROUP_ROOM}"/>
    {"".join(geoms)}
    {_car_xml()}
    {_snoo_xml()}
  </worldbody>
</mujoco>"""


# --- 車の動き（自転車モデル。curvature・arc は app/roommap.py） ---


@dataclass
class CarState:
    """車の姿勢（後ろの車軸の中央）と、ハンドル・首の向き（度。Pi と同じ向き）。"""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    steer_deg: float = 0.0  # 右が +
    pan_deg: float = 0.0  # 右が +
    tilt_deg: float = 0.0  # 上が +
    wheel_rad: float = 0.0  # 後ろの車輪の回った角度（絵のため）

    @classmethod
    def from_camera(cls, pose: Pose) -> CarState:
        """カメラの真下の姿勢から（地図のステーションの位置はカメラの真下）。"""
        return cls(
            pose.x - CAM_X_M * math.cos(pose.yaw), pose.y - CAM_X_M * math.sin(pose.yaw), pose.yaw
        )

    def camera_pose(self) -> Pose:
        """カメラの真下の姿勢。yaw は -π..π にする（VGGT から出す向きと同じ範囲）。"""
        yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        return Pose(
            self.x + CAM_X_M * math.cos(self.yaw), self.y + CAM_X_M * math.sin(self.yaw), yaw
        )

    def to_world(self, fx: float, fy: float) -> tuple[float, float]:
        """車の向き（前 fx・左 fy）の点を部屋の座標に。"""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * fx - s * fy, self.y + s * fx + c * fy


def footprint(x: float, y: float, yaw: float, margin: float = 0.0) -> np.ndarray:
    """車（後ろの車軸の中央が x, y）の外形の 4 つの角 (4, 2)。左右に margin を足す。"""
    half = CAR_WIDTH_M / 2 + margin
    local = np.array(
        [[CAR_FRONT_M, half], [CAR_REAR_M, half], [CAR_REAR_M, -half], [CAR_FRONT_M, -half]]
    )
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def rect_hits(corners: np.ndarray, aabbs: np.ndarray) -> bool:
    """凸の四角形（4 つの角）が、どれかの長方形 [x0, y0, x1, y1] と重なるか（分離軸で調べる）。"""
    if not len(aabbs):
        return False
    sep = (corners[:, 0].max() < aabbs[:, 0]) | (corners[:, 0].min() > aabbs[:, 2])
    sep |= (corners[:, 1].max() < aabbs[:, 1]) | (corners[:, 1].min() > aabbs[:, 3])
    box_corners = np.stack(
        [aabbs[:, [0, 1]], aabbs[:, [2, 1]], aabbs[:, [2, 3]], aabbs[:, [0, 3]]], 1
    )  # (K, 4, 2)
    for a, b in ((0, 1), (1, 2)):
        axis = corners[b] - corners[a]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        q = corners @ axis
        p = box_corners @ axis
        sep |= (q.max() < p.min(1)) | (q.min() > p.max(1))
    return bool((~sep).any())


# --- snoo ---


@dataclass
class SnooPose:
    x: float
    y: float
    yaw: float
    moving: bool = False


class SnooModel(Protocol):
    """snoo の動き。dt 秒進めて、新しい姿勢を返す。robot は車のカメラの姿勢。

    v1 は RandomWalkSnoo。DeepLabCut で取った関節の動き（feature/snoo-sim の app/cat_pose.py）
    から作るものも、この形にすれば差し替えられる。
    """

    pose: SnooPose

    def step(self, dt: float, robot: Pose | None) -> SnooPose: ...


class RandomWalkSnoo:
    """止まる・歩くを繰り返す猫。車が近づくと離れる。壁・家具の脚・車にはぶつからない。"""

    def __init__(
        self,
        room: Room,
        start: SnooPose,
        rng: np.random.Generator | None = None,
        speed: float = 0.25,
        shy_m: float = 0.5,
    ):
        self.room = room
        self.pose = start
        self.rng = rng or np.random.default_rng(1)
        self.speed = speed
        self.shy_m = shy_m
        # 猫は 27 cm より上の物の下をくぐれる（机・椅子の座面）
        self.aabbs = low_aabbs(room, SNOO_HEIGHT_M)
        self.timer = float(self.rng.uniform(1, 4))
        self.walking = False
        self.heading = start.yaw
        self.robot_xy: tuple[float, float] | None = None

    def free_at(self, x: float, y: float) -> bool:
        """体の円が、箱・部屋の外・見ていない床・車に重ならないか。"""
        r = SNOO_RADIUS_M
        x0, y0, x1, y1 = self.room.bounds
        if not (x0 + r <= x <= x1 - r and y0 + r <= y <= y1 - r):
            return False
        a = self.aabbs
        if len(a):
            dx = np.maximum(np.maximum(a[:, 0] - x, 0), x - a[:, 2])
            dy = np.maximum(np.maximum(a[:, 1] - y, 0), y - a[:, 3])
            if (dx * dx + dy * dy < r * r).any():
                return False
        if self.robot_xy is not None and math.dist((x, y), self.robot_xy) < 0.3:
            return False
        ok = self.room.floor_ok
        if ok is not None:
            # 間取りから作った部屋では、床と分かっている所だけを歩く
            ix = int(math.floor((x - GRID_ORIGIN_M) / RES_M))
            iy = int(math.floor((y - GRID_ORIGIN_M) / RES_M))
            if not (0 <= ix < ok.shape[0] and 0 <= iy < ok.shape[1]) or not ok[ix, iy]:
                return False
        return True

    def step(self, dt: float, robot: Pose | None) -> SnooPose:
        p = self.pose
        self.robot_xy = None
        if robot is not None:
            # 車の真ん中（カメラから少し後ろ）
            self.robot_xy = (
                robot.x - 0.08 * math.cos(robot.yaw),
                robot.y - 0.08 * math.sin(robot.yaw),
            )
        near = self.robot_xy is not None and math.dist((p.x, p.y), self.robot_xy) < self.shy_m
        self.timer -= dt
        if near and not self.walking:
            # 近づかれたら、車から離れる向きに歩き出す
            away = math.atan2(p.y - self.robot_xy[1], p.x - self.robot_xy[0])
            self.walking, self.heading = True, away + float(self.rng.normal(0, 0.4))
            self.timer = float(self.rng.uniform(1.0, 2.5))
        elif self.timer <= 0:
            self.walking = not self.walking
            if self.walking:
                self.heading = p.yaw + float(self.rng.normal(0, 1.2))
                self.timer = float(self.rng.uniform(1.5, 5))
            else:
                self.timer = float(self.rng.uniform(2, 8))
        if not self.walking:
            self.pose = SnooPose(p.x, p.y, p.yaw, False)
            return self.pose
        # 向きは少しずつ変える（猫はその場でくるっと回れるが、絵が飛ばないように）
        turn = (self.heading - p.yaw + math.pi) % (2 * math.pi) - math.pi
        yaw = p.yaw + max(-3.0 * dt, min(3.0 * dt, turn))
        ds = self.speed * dt * (0.3 if abs(turn) > 0.8 else 1.0)
        nx, ny = p.x + ds * math.cos(yaw), p.y + ds * math.sin(yaw)
        if self.free_at(nx, ny):
            self.pose = SnooPose(nx, ny, yaw, True)
        else:
            # ぶつかるなら向きを変える（次の歩で試す）
            self.heading = yaw + float(self.rng.choice([-1, 1])) * float(self.rng.uniform(1, 2.5))
            self.pose = SnooPose(p.x, p.y, yaw, True)
        return self.pose


def snoo_aabb(p: SnooPose) -> np.ndarray:
    """snoo（しっぽまで）を床に落とした向きの付いた長方形を囲む、軸にそろえた長方形 (1, 4)。"""
    local = np.array(
        [
            [SNOO_FRONT_M, SNOO_HALF_W_M],
            [SNOO_REAR_M, SNOO_HALF_W_M],
            [SNOO_REAR_M, -SNOO_HALF_W_M],
            [SNOO_FRONT_M, -SNOO_HALF_W_M],
        ]
    )
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    w = local @ np.array([[c, s], [-s, c]]) + np.array([p.x, p.y])
    return np.array([[w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max()]])


def find_free_spot(room: Room, near: tuple[float, float], rng: np.random.Generator) -> SnooPose:
    """near の近くで、snoo が立てる所を探す（最初の置き場所）。"""
    probe = RandomWalkSnoo(room, SnooPose(*near, 0.0), rng)
    for k in range(2000):
        r = 0.05 * k**0.5
        x, y = near[0] + rng.normal(0, r), near[1] + rng.normal(0, r)
        if probe.free_at(x, y):
            return SnooPose(x, y, float(rng.uniform(-math.pi, math.pi)))
    return SnooPose(*near, 0.0)


# --- 光線（カメラ・超音波） ---


def focal_px() -> float:
    return IMG_W / 2 / math.tan(math.radians(HFOV_DEG / 2))


def pixel_dirs(cam_mat: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """画素 (u, v) を通る光線の向き（部屋の座標、長さ 1）。MuJoCo のカメラは -z を見る。"""
    f = focal_px()
    d = np.stack([(u - (IMG_W - 1) / 2) / f, -(v - (IMG_H - 1) / 2) / f, -np.ones_like(u)], -1)
    d = d @ np.asarray(cam_mat).reshape(3, 3).T
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def project(cam_pos: np.ndarray, cam_mat: np.ndarray, pts: np.ndarray):
    """部屋の点 (N, 3) を画素 (u, v) と、カメラの前にあるか（front）にする。"""
    c = (np.asarray(pts, np.float64) - cam_pos) @ np.asarray(cam_mat).reshape(3, 3)
    depth = -c[:, 2]
    front = depth > 1e-6
    f = focal_px()
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (IMG_W - 1) / 2 + f * c[:, 0] / depth
        v = (IMG_H - 1) / 2 - f * c[:, 1] / depth
    return u, v, front


def border_pixels(per_edge: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """画面の縁を一周する画素（下の縁の左から、反時計回り）。4 つの角は k * per_edge 番目。"""
    t = np.linspace(0, 1, per_edge, endpoint=False)
    w, h = IMG_W - 1, IMG_H - 1
    u = np.concatenate([t * w, np.full_like(t, w), (1 - t) * w, np.zeros_like(t)])
    v = np.concatenate([np.full_like(t, h), (1 - t) * h, np.zeros_like(t), t * h])
    return u, v


def floor_footprint(
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    max_range: float = 2.0,
    per_edge: int = 16,
    hit_dist: np.ndarray | None = None,
) -> np.ndarray:
    """画面の縁を床に落とした多角形 (M, 2)。カメラから水平に max_range より先は切る。

    上を向いた縁は床に当たらないので、その向きの max_range 先の真下を使う
    （見える床の範囲を描くため。上を向くほど手前の床は画面の下に寄る）。
    hit_dist（縁の光線ごとの、最初に何かに当たるまでの距離）を渡すと、壁などで止める。
    """
    d = pixel_dirs(cam_mat, *border_pixels(per_edge))
    out = []
    for k, di in enumerate(d):
        horiz = math.hypot(di[0], di[1])
        if horiz < 1e-6:
            continue
        s = math.inf
        if di[2] < -1e-6:
            s = -cam_pos[2] / di[2]
        if hit_dist is not None:
            s = min(s, float(hit_dist[k]))
        s = min(s, max_range / horiz)
        out.append(cam_pos[:2] + s * di[:2])
    return np.array(out)


@dataclass
class ScanHits:
    """1 コマ分の光線の当たった点。kind は 0 = 床、1 = 障害物、2 = 上の層。"""

    points: np.ndarray  # (N, 3)
    kind: np.ndarray  # (N,)


@dataclass
class StationObs:
    """1 ステーションで見たもの（格子のセル (K, 2)）と、そのときの超音波。
    地図のサーバーの窓（WINDOW か所だけで間取りを作る）をまねるとき、選んだステーションの分だけ
    足し直す。
    """

    free: np.ndarray
    occupied: np.ndarray
    high: np.ndarray
    sonar_m: float | None
    pose: Pose


# --- シミュレーター ---


@dataclass
class Summary:
    cycles: int
    stations: int
    free_m2: float
    known_m2: float
    coverage: float  # 本当の床（低い物の無い所）のうち、床と分かった割合
    bumps: int
    stops: int
    reverses: int
    distance_m: float
    reason: str
    # 時間（実機の時計。Pace で決める）と、調べた床の広さ
    t_s: float = 0.0
    covered_m2: float = 0.0  # 本当の床のうち、床と分かった広さ
    m2_per_min: float = 0.0
    m2_per_m: float = 0.0  # 走った 1 m あたり


class Sim:
    """部屋・車・snoo・作っている地図をまとめて持つ。描く側（snoo_sim_render）はこれを見るだけ。"""

    def __init__(
        self,
        room: Room,
        snoo: SnooModel | None = None,
        rays: tuple[int, int] = (320, 240),
        params: MapParams | None = None,
        seed: int = 0,
        sonar_map: bool = False,
        pace: Pace | None = None,
        window: int | None = None,
    ):
        self.room = room
        self.pace = pace or Pace()
        # 地図のサーバーの窓（int なら、その数のステーションだけで間取りを作る。None は全部）
        self.window = window
        self.xml = build_mjcf(room)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.rays = rays
        self.params = params or MapParams(camera_height_m=CAM_H_M)
        # 前の超音波をステーションごとに格子に足す（地図のサーバーと同じ）
        self.sonar_map = sonar_map
        # 前へ走った道（走った順の区間 (ハンドル°, m)）。行き場が無いときに戻るのに使う
        self.driven: list[tuple[float, float]] = []
        self.backs = 0  # 続けて後ろへ戻った回数
        self.arrived_reverse: list[bool] = []  # ステーションごとの、後ろへ戻って着いたか
        self._reversed = False
        self.obs: list[StationObs] = []
        self.bump_log: list[dict] = []  # ぶつかったときの時刻・後ろへか・1 歩の種類
        # 撮り終えるごとの (Sim.t, 本当の床のうち床と分かった m²)
        self.timeline: list[tuple[float, float]] = []
        self._truth: np.ndarray | None = None
        # 走った 1 歩ごとの (走り出したカメラの真下, 走った区間, 後ろへか)。地図に足す
        self.drives: list[tuple[Pose, list[tuple[float, float]], bool]] = []
        self.kinds: dict[str, int] = {}  # 決めた 1 歩の種類（nbv・random・reverse・turn・done）の数
        self.rng = np.random.default_rng(seed)
        self.car = CarState.from_camera(room.start)
        if snoo is None:
            x0, y0, x1, y1 = room.bounds
            spot = find_free_spot(room, (x0 + 0.65 * (x1 - x0), y0 + 0.3 * (y1 - y0)), self.rng)
            snoo = RandomWalkSnoo(room, spot, np.random.default_rng(seed + 1))
        self.snoo = snoo
        self.car_aabbs = low_aabbs(room, CAR_HEIGHT_M)
        self.grid = Grid()
        if room.floor_ok is not None and room.floor_ok.shape != self.grid.logodds.shape:
            raise ValueError("間取りの格子の大きさが app/roommap.Grid の既定と違う")
        self.high = np.zeros_like(self.grid.logodds, np.int32)
        self.stations: list[StationResult] = []
        self.station_hits: list[ScanHits] = []  # いまのステーションで撮ったコマの点
        self.last_hits: ScanHits | None = None
        self.action: dict = {}
        self.plan: Plan | None = None
        self.step_arc: list[tuple[float, float, float]] = []  # 決めた 1 歩の弧（後ろの車軸）
        self.trail: list[Pose] = [self.car.camera_pose()]
        self.snoo_trail: list[SnooPose] = [self.snoo.pose]
        self.t = 0.0
        self.cycle = 0
        self.phase = "start"
        self.bumps = 0
        self.stops = 0
        self.reverses = 0
        self.distance_m = 0.0
        self.clear_m = 0.0
        self.corridor = np.zeros((0, 2))
        self.sonar_m: float | None = None
        self.detection: tuple[float, float, float, float, float] | None = None
        self._jid = {
            n: self.model.joint(n).qposadr[0]
            for n in ("car", "snoo", "pan", "tilt", "steer_fl", "steer_fr")
            + tuple(f"wheel_{a}{b}" for a in "rf" for b in "lr")
        }
        self.cam_id = self.model.camera("car_cam").id
        self.sync()

    # --- MuJoCo との受け渡し ---

    def sync(self) -> None:
        """車・首・snoo の姿勢を qpos に書いて、位置（カメラも）を計算し直す。センサーも測る。"""
        q, j, c = self.data.qpos, self._jid, self.car
        q[j["car"] : j["car"] + 7] = [c.x, c.y, 0, math.cos(c.yaw / 2), 0, 0, math.sin(c.yaw / 2)]
        s = self.snoo.pose
        q[j["snoo"] : j["snoo"] + 7] = [s.x, s.y, 0, math.cos(s.yaw / 2), 0, 0, math.sin(s.yaw / 2)]
        q[j["pan"]] = math.radians(c.pan_deg)
        q[j["tilt"]] = math.radians(c.tilt_deg)
        q[j["steer_fl"]] = q[j["steer_fr"]] = -math.radians(c.steer_deg)
        for w in ("wheel_rl", "wheel_rr", "wheel_fl", "wheel_fr"):
            q[j[w]] = c.wheel_rad
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.sonar_m = self.sonar()
        self.detection = self.detect_snoo()
        self.clear_m, self.corridor = self.clearance(self.car.steer_deg)

    def camera(self) -> tuple[np.ndarray, np.ndarray]:
        """車のカメラの位置と向きの行列（列が右・上・後ろ）。"""
        return self.data.cam_xpos[self.cam_id].copy(), self.data.cam_xmat[self.cam_id].reshape(3, 3)

    def view_polygon(self, max_range: float = VIEW_RANGE_M, per_edge: int = 16) -> np.ndarray:
        """カメラの見える床の範囲（画面の縁を床に落とした形。壁や家具の所で止める）。"""
        pos, mat = self.camera()
        d = pixel_dirs(mat, *border_pixels(per_edge))
        dist, _ = self.cast(pos, d, max_range * 3)
        return floor_footprint(pos, mat, max_range, per_edge, dist)

    def cast(self, origin: np.ndarray, dirs: np.ndarray, cutoff: float):
        """光線を部屋と snoo に当てる。(距離（当たらなければ inf）, geom の番号)。"""
        n = len(dirs)
        gid = np.zeros(n, np.int32)
        dist = np.zeros(n)
        mujoco.mj_multiRay(
            self.model,
            self.data,
            np.asarray(origin, np.float64),
            np.ascontiguousarray(dirs, np.float64).ravel(),
            RAY_GROUPS,
            True,
            -1,
            gid,
            dist,
            None,
            n,
            cutoff,
        )
        dist = np.where(gid >= 0, dist, np.inf)
        return dist, gid

    # --- センサー ---

    def shoot(self) -> ScanHits:
        """いまの首の向きで 1 コマ撮る代わりに、画素ごとの光線を当てる。"""
        nu, nv = self.rays
        pos, mat = self.camera()
        u, v = np.meshgrid(np.linspace(0, IMG_W - 1, nu), np.linspace(0, IMG_H - 1, nv))
        d = pixel_dirs(mat, u.ravel(), v.ravel())
        dist, _ = self.cast(pos, d, self.params.max_range_m + 0.5)
        ok = np.isfinite(dist)
        p = pos + d[ok] * dist[ok, None]
        near = np.hypot(p[:, 0] - pos[0], p[:, 1] - pos[1]) <= self.params.max_range_m
        p = p[near]
        z = p[:, 2]
        tol = self.params.floor_tol_m
        top = self.params.car_height_m + self.params.obstacle_margin_m
        kind = np.where(np.abs(z) <= tol, 0, np.where(z <= top, 1, 2))
        if self.room.floor_ok is not None:
            # 間取りで見ていなかった床は、当たっても床と言わない
            ix, iy = self.grid.cell(p[:, 0], p[:, 1])
            fi = np.clip(ix, 0, self.room.floor_ok.shape[0] - 1)
            fj = np.clip(iy, 0, self.room.floor_ok.shape[1] - 1)
            keep = (kind != 0) | self.room.floor_ok[fi, fj]
            p, kind = p[keep], kind[keep]
        return ScanHits(p, kind)

    def sonar(self) -> float | None:
        """前の超音波（±15° の円すい）でいちばん近い物までの距離（m）。3 m より先は None。"""
        c = self.car
        fx, fy = c.to_world(CAR_FRONT_M, 0.0)
        origin = np.array([fx, fy, SONAR_H_M])
        a = np.radians(np.linspace(-SONAR_HALF_DEG, SONAR_HALF_DEG, 9))
        e = np.radians(np.array([-6.0, 0.0, 6.0, 12.0]))  # 床に当てないよう下は浅く
        aa, ee = np.meshgrid(a, e)
        yaw = c.yaw + aa.ravel()
        d = np.stack(
            [
                np.cos(yaw) * np.cos(ee.ravel()),
                np.sin(yaw) * np.cos(ee.ravel()),
                np.sin(ee.ravel()),
            ],
            1,
        )
        dist, gid = self.cast(origin, d, SONAR_RANGE_M)
        floor = self.model.geom("floor").id
        dist = np.where(gid == floor, np.inf, dist)
        best = float(dist.min())
        return best if best <= SONAR_RANGE_M else None

    def snoo_points(self) -> np.ndarray:
        """snoo の箱の角と体の中の点（部屋の座標）。見えるかと、画面の箱を決めるのに使う。"""
        p = self.snoo.pose
        xs, ys, zs = (
            (SNOO_REAR_M, SNOO_FRONT_M),
            (-SNOO_HALF_W_M, SNOO_HALF_W_M),
            (0.0, SNOO_HEIGHT_M),
        )
        local = [(x, y, z) for x in xs for y in ys for z in zs]
        local += [
            (0.0, 0, 0.15),
            (0.15, 0, 0.2),
            (-0.05, 0, 0.15),
            (0.07, 0, 0.05),
            (-0.08, 0, 0.05),
            (-0.2, 0, 0.21),
        ]
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        L = np.array(local)
        return np.stack(
            [p.x + c * L[:, 0] - s * L[:, 1], p.y + s * L[:, 0] + c * L[:, 1], L[:, 2]], 1
        )

    def detect_snoo(self) -> tuple[float, float, float, float, float] | None:
        """車のカメラに snoo が映っていれば、検出の箱 (u0, v0, u1, v1, 見えた割合)。

        体の中の点（8 つの角は使わない）へ光線を引き、snoo より手前で何かに当たれば隠れている。
        箱は、見えた点が 1 つでもあれば、外形の角を画面に落として囲み、画面の中に切る。
        """
        pos, mat = self.camera()
        pts = self.snoo_points()
        u, v, front = project(pos, mat, pts)
        inner = np.arange(8, len(pts))
        d = pts[inner] - pos
        r = np.linalg.norm(d, axis=1)
        dist, gid = self.cast(pos, d / r[:, None], float(r.max()) + 0.1)
        is_snoo = np.isin(gid, np.flatnonzero(self.model.geom_group == GROUP_SNOO))
        in_img = front[inner] & (u[inner] >= 0) & (u[inner] < IMG_W) & (v[inner] >= 0)
        in_img &= v[inner] < IMG_H
        seen = in_img & (is_snoo | (dist >= r - 0.01))
        if not seen.any():
            return None
        corners = front[:8]
        if not corners.any():
            return None
        u0, u1 = np.clip([u[:8][corners].min(), u[:8][corners].max()], 0, IMG_W - 1)
        v0, v1 = np.clip([v[:8][corners].min(), v[:8][corners].max()], 0, IMG_H - 1)
        if u1 - u0 < 2 or v1 - v0 < 2:
            return None
        return float(u0), float(v0), float(u1), float(v1), float(seen.mean())

    def clearance(self, steer_deg: float) -> tuple[float, np.ndarray]:
        """ハンドル steer_deg のまま進んだとき、ぶつかるまでの距離（m、最大 CLEAR_MAX_M）と、
        その通り道の帯（前の左右の角が描く線で囲んだ多角形 (M, 2)）。

        本物の部屋で測る「答え」の clear_cm。Pi が受け取る clear_cm（app/clearance.py で
        カメラから見積もる）は、これより短いか見落とすことがある。
        """
        obst = np.concatenate([self.car_aabbs, snoo_aabb(self.snoo.pose)])
        x, y, yaw = self.car.x, self.car.y, self.car.yaw
        step = 0.02
        left, right = [], []
        s = 0.0
        while True:
            fp = footprint(x, y, yaw, CORRIDOR_MARGIN_M)
            if s > 0 and rect_hits(fp, obst):
                s -= step
                break
            left.append(fp[0])
            right.append(fp[3])
            if s >= CLEAR_MAX_M:
                break
            x, y, yaw = arc(x, y, yaw, steer_deg, step)
            s += step
        poly = np.array(left + right[::-1]) if len(left) > 1 else np.zeros((0, 2))
        return max(s, 0.0), poly

    # --- 地図 ---

    def commit_station(self) -> StationResult:
        """撮り終えたステーションの点を格子に足す（app/snoo_mapper.build_map と同じ分け方）。

        snoo_mapper は 1 セルに 3 点以上ある所だけ使う（VGGT の外れ値のため）が、ここの点は
        外れないので 1 点で使う。
        """
        g = self.grid
        empty = np.zeros((0, 2), int)
        cells = {0: empty, 1: empty, 2: empty}
        if self.station_hits:
            p = np.concatenate([h.points for h in self.station_hits])
            k = np.concatenate([h.kind for h in self.station_hits])
            for kind in cells:
                ix, iy = g.cell(p[k == kind, 0], p[k == kind, 1])
                ok = g.inside(ix, iy)
                cells[kind] = np.unique(np.stack([ix[ok], iy[ok]], 1), axis=0).reshape(-1, 2)
        frames = len(self.station_hits)
        self.station_hits = []
        pose = self.car.camera_pose()
        self.obs.append(StationObs(cells[0], cells[1], cells[2], self.sonar_m, pose))
        st = StationResult(f"sim{len(self.stations):03d}", pose, CAM_H_M, 0.0, frames)
        self.stations.append(st)
        self.arrived_reverse.append(self._reversed)
        self._reversed = False
        self.rebuild_map()
        if self._truth is None:
            self._truth = self.truth_floor()
        covered = float(((self.grid.state() == FREE) & self._truth).sum() * self.grid.res**2)
        self.timeline.append((round(self.t, 1), round(covered, 3)))
        return st

    def rebuild_map(self) -> None:
        """ステーションごとに見たものから格子を作り直す（窓があれば窓の中のステーションだけ）。
        足し方は Grid.mark_free・mark_occupied と同じ（1 回の撮影で 1 セル 1 回）。
        """
        ids = list(range(len(self.obs)))
        if self.window and ids:
            poses = {i: o.pose for i, o in enumerate(self.obs)}
            ids = choose_window(ids, ids[-1], poses, self.window)
        g = Grid(size_m=self.grid.n * self.grid.res, res=self.grid.res)
        g.origin = self.grid.origin
        high = np.zeros_like(g.logodds, np.int32)
        for i in ids:
            o = self.obs[i]
            for c, v in ((o.free, FREE_HIT), (o.occupied, OCCUPIED_HIT)):
                if len(c):
                    g.logodds[c[:, 0], c[:, 1]] += v
                    np.clip(g.logodds, -4, 4, out=g.logodds)
            if len(o.high):
                high[o.high[:, 0], o.high[:, 1]] += 1
            if self.sonar_map:
                mark_sonar(g, o.pose, o.sonar_m)
        if self.params.driven_free:
            for start, pieces, reverse in self.drives:
                mark_driven(g, start, pieces, reverse)
        self.grid, self.high = g, high

    def truth_floor(self) -> np.ndarray:
        """本当の床（部屋の中で、車の高さより低い物の無いセル）。調べた割合を出すのに使う。"""
        g = self.grid
        x0, y0, x1, y1 = self.room.bounds
        cx, cy = g.center(*np.meshgrid(np.arange(g.n), np.arange(g.n), indexing="ij"))
        inside = (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
        for a in self.car_aabbs:
            inside &= ~((cx > a[0]) & (cx < a[2]) & (cy > a[1]) & (cy < a[3]))
        if self.room.floor_ok is not None:
            inside &= self.room.floor_ok
        return inside

    def summary(self, reason: str = "") -> Summary:
        s = self.grid.state()
        cell = self.grid.res**2
        truth = self.truth_floor()
        covered = float(((s == FREE) & truth).sum() * cell)
        return Summary(
            self.cycle,
            len(self.stations),
            round(float((s == FREE).sum() * cell), 2),
            round(float((s != UNKNOWN).sum() * cell), 2),
            round(float(((s == FREE) & truth).sum() / max(truth.sum(), 1)), 3),
            self.bumps,
            self.stops,
            self.reverses,
            round(self.distance_m, 2),
            reason or str(self.action.get("reason", self.action.get("action", ""))),
            round(self.t, 1),
            round(covered, 3),
            round(covered / max(self.t / 60, 1e-9), 3),
            round(covered / max(self.distance_m, 1e-9), 3) if self.distance_m > 0 else 0.0,
        )

    # --- 動かす ---

    def advance(self, dt: float) -> None:
        """時間を進める（snoo が動く）。"""
        self.t += dt
        self.snoo.step(dt, self.car.camera_pose())
        self.snoo_trail.append(self.snoo.pose)
        if len(self.snoo_trail) > 400:
            self.snoo_trail = self.snoo_trail[-400:]

    def scan(self, on_frame=None) -> None:
        """止まって首を振って撮る（実機の scan.json と同じ順）。"""
        self.phase = "scan"
        self.car.steer_deg = 0.0
        for tilt in self.pace.tilts:
            for pan in self.pace.pans:
                self.car.pan_deg, self.car.tilt_deg = float(pan), float(tilt)
                self.advance(self.pace.hold_s)
                self.sync()
                self.last_hits = self.shoot()
                self.station_hits.append(self.last_hits)
                if on_frame:
                    on_frame(self)
        self.car.pan_deg, self.car.tilt_deg = 0.0, 0.0
        self.commit_station()

    def history(self) -> History:
        """地図のサーバーが置き場から作るのと同じ、前のステーションまでの走り方。"""
        return History(list(self.driven), self.backs, sum(self.arrived_reverse[-RECENT_STATIONS:]))

    def think(self) -> dict:
        """次の 1 歩を、地図のサーバー（app/snoo_map_server.plan_station）と同じに決める。
        app/snoo_mapper.decide で決め、行き場が無ければ（unreachable）、Pi が reverse を
        受けられるとき（params.reverse）は reverse_step で後ろへ下がる。でたらめな 1 歩の
        くじ（seed）はステーションの番号、固まったか（clustered）は直前のステーションで見る。
        """
        self.phase = "plan"
        heights = [CAM_H_M] * len(self.stations)
        recent = [s.pose for s in self.stations[-CLUSTER_STATIONS - 1 : -1]]
        params = dataclasses.replace(
            self.params, seed=len(self.stations) - 1, force_random=clustered(recent)
        )
        self.action, self.plan = decide(self.grid, self.stations, params, heights)
        pose = self.stations[-1].pose
        self.action, _ = or_reverse(self.action, self.grid, pose, self.history(), params)
        a = self.action
        kind = "reverse" if a.get("reverse") else "random" if a.get("random") else "nbv"
        kind = "turn" if a.get("turn") else kind
        kind = kind if a.get("action") == "move" else "done"
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        self.step_arc = []
        if self.action.get("action") == "move":
            c = self.car
            sign = -1.0 if self.action.get("reverse") else 1.0
            x, y, yaw = c.x, c.y, c.yaw
            self.step_arc = [(x, y, yaw)]
            for steer, cm in self.pieces(self.action):
                n = max(2, int(cm / 2))
                for _ in range(n):
                    x, y, yaw = arc(x, y, yaw, steer, sign * cm / 100 / n)
                    self.step_arc.append((x, y, yaw))
        self.car.steer_deg = float(self.action.get("steer_deg", 0.0))
        self.car.tilt_deg = DRIVE_TILT_DEG
        self.sync()
        return self.action

    @staticmethod
    def pieces(action: dict) -> list[tuple[float, float]]:
        """返事の区間 [(ハンドル°, cm), ...]。segments が無ければ steer_deg・distance_cm の 1 本"""
        segs = action.get("segments")
        if segs:
            return [(float(x["steer_deg"]), float(x["distance_cm"])) for x in segs]
        return [(float(action["steer_deg"]), float(action["distance_cm"]))]

    def drive(
        self,
        steer_deg: float,
        speed: float,
        distance_cm: float,
        on_frame=None,
        frame_every_m: float = 0.025,
        reverse: bool = False,
    ) -> str:
        """Pi と同じ指令で走る。'ok'・'stop'（前がふさがった）・'bump'（ぶつかった）を返す。

        1 cm ずつ進め、外形が箱か snoo に重なればぶつかったとしてその手前で止める。
        Pi は走る間 clear_cm を見て止まるので、ここでも本当の通り道が STOP_CLEAR_M より短く
        なったら止める。reverse は後ろへ走る（clear_cm は前しか見ないので、ぶつかるまで走る）。
        """
        self.phase = "reverse" if reverse else "drive"
        c = self.car
        c.steer_deg = max(-STEER_LIMIT_DEG, min(STEER_LIMIT_DEG, steer_deg))
        c.pan_deg, c.tilt_deg = 0.0, DRIVE_TILT_DEG
        v = max(speed, 1.0) * SPEED_M_S
        sign = -1.0 if reverse else 1.0
        total, done, since = distance_cm / 100, 0.0, 0.0
        result = "ok"
        while done < total - 1e-9:
            ds = min(0.01, total - done)
            nx, ny, nyaw = arc(c.x, c.y, c.yaw, c.steer_deg, sign * ds)
            obst = np.concatenate([self.car_aabbs, snoo_aabb(self.snoo.pose)])
            if rect_hits(footprint(nx, ny, nyaw), obst):
                result = "bump"
                self.bumps += 1
                a = self.action
                kind = (
                    "turn"
                    if a.get("turn")
                    else "reverse"
                    if a.get("reverse")
                    else ("random" if a.get("random") else "nbv")
                )
                cat = rect_hits(footprint(nx, ny, nyaw), snoo_aabb(self.snoo.pose))
                self.bump_log.append(
                    {"t": round(self.t, 1), "reverse": reverse, "kind": kind, "snoo": cat}
                )
                break
            c.x, c.y, c.yaw = nx, ny, nyaw
            c.wheel_rad += sign * ds / WHEEL_R_M
            done += ds
            since += ds
            self.distance_m += ds
            self.advance(ds / v)
            self.sync()
            if not reverse and self.clear_m < STOP_CLEAR_M and done < total - 1e-9:
                result = "stop"
                self.stops += 1
                break
            if on_frame and since >= frame_every_m:
                since = 0.0
                on_frame(self)
        self.trail.append(c.camera_pose())
        c.steer_deg = 0.0
        self.sync()
        return result

    def explore(
        self,
        cycles: int,
        on_frame=None,
        plan_frames: int | None = None,
        backtrack: int = 0,
        time_budget_s: float | None = None,
    ) -> Summary:
        """撮る → 地図を直す → 1 歩走る、を cycles 回（やめる理由が出たら・time_budget_s 秒
        （Sim.t）を過ぎたらそこまで）。

        次の 1 歩の決め方は地図のサーバーと同じ（think）。backtrack > 0 なら Pi が reverse を
        受けられることにする（params.reverse。行き場が無いとき、切り返しか来た道を戻る）。
        params.segments なら、1 歩を「曲がる → まっすぐ」の区間で走る。
        地図を作り直す間（pace.server_s 秒、plan_frames があれば 0.5 秒 × plan_frames）も猫は動く。
        """
        if backtrack:
            self.params = dataclasses.replace(self.params, reverse=True)
        wait_s = self.pace.server_s if plan_frames is None else 0.5 * plan_frames
        reason = "cycles"
        for k in range(cycles):
            if time_budget_s is not None and self.t >= time_budget_s:
                reason = "time"
                break
            self.cycle = k + 1
            self.scan(on_frame)
            action = self.think()
            left = wait_s
            while left > 1e-9:
                self.advance(min(0.5, left))
                left -= 0.5
                self.sync()
                if on_frame:
                    on_frame(self)
            if action.get("action") != "move":
                reason = str(action.get("reason", "done"))
                break
            reverse = bool(action.get("reverse"))
            for steer, cm in self.pieces(action):
                start, before = self.car.camera_pose(), self.distance_m
                result = self.drive(steer, action["speed"], cm, on_frame, reverse=reverse)
                moved = self.distance_m - before
                if moved > 1e-6:
                    self.drives.append((start, [(steer, moved)], reverse))
                if reverse:
                    self.driven = consume(self.driven, moved)
                elif moved > 1e-6:
                    self.driven.append((steer, moved))
                if result != "ok":
                    break
            self.backs = self.backs + 1 if reverse else 0
            self.reverses += reverse
            self._reversed = reverse
        self.phase = "done"
        return self.summary(reason)

    # --- 地図から作った部屋 ---

    def load_knowledge(self, room_json: str | Path) -> None:
        """間取り（room.json）の格子・上の層・ステーションを、作った地図として読む（1 枚の絵）。"""
        _, state, high, grid, plan = load_map(room_json)
        if grid.n != self.grid.n or abs(grid.origin - self.grid.origin) > 1e-9:
            self.grid = grid
            self.high = np.zeros_like(grid.logodds, np.int32)
        self.grid.logodds[:] = 0
        self.grid.logodds[state == FREE] = -2
        self.grid.logodds[state == OCCUPIED] = 2
        self.high = high.astype(np.int32)
        self.stations = [
            StationResult(f"map{k:03d}", p, CAM_H_M, 0.0, 0)
            for k, p in enumerate(self.room.source_stations or [self.room.start])
        ]
        self.trail = [s.pose for s in self.stations]
        self.cycle = len(self.stations)
        self.think()
        self.phase = "map"
