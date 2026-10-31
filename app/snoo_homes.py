"""30 坪（約 99 m²）の家の間取りを、くじ（seed）から作る。探り方を学ぶための部屋の山。

実機の部屋は 1 つしか無く、MuJoCo の部屋も 2 つしか無い。それだけで探り方を決めると、その部屋に
合わせ込んでしまう。ここでは日本の家らしい間取り（LDK・寝室・和室・廊下・洗面所・風呂・トイレ）に
家具・人・snoo を置いた部屋を、いくらでも作る。

- 外形：長方形か L 字（角を 1 つ欠く）。広さは 30 坪 ± 数 %
- 部屋：片方の端を LDK、残りを廊下で分けて、両側に寝室・和室・水回りを並べる。
  部屋の間は 10 cm の壁で、扉の幅は 70〜90 cm。水回り（風呂・トイレ・洗面所）はたいてい
  扉が閉まっていて入れない（closed）。玄関は段があるので低い障害物にする
- 家具：部屋の種類ごとに、壁ぎわか真ん中に置く。扉の前は空ける。置いたら車が全部の扉へ
  行けるかを確かめ、ふさぐ物は置かない
- 動く物：人（1〜2 人、ゆっくり歩いては止まる）と snoo（歩く・寝る）。車とは関係なく、
  前もって 0.5 秒おきの道のりを作る（くじが同じなら同じ動き）

格子は 5 cm（app/roommap.Grid と同じ。添え字も [ix, iy]）。層は
FLOOR（床）・LOW（床から立つ物。車も目もさえぎる。高さ height）・HIGH（天板・座面など下を
くぐれる物。車は通れて、低いカメラは下の床を見通せる）・CLOSED（家の外・閉まった部屋。
見えず、数えない）。ラグ（rug）は高さ 1 cm なので床のまま（絵にだけ描く）。
HIGH の下の端の高さは under に持つ。make_home(low_sofa=True) は、下の端が 10〜18 cm の
低いソファを 1 つ足す（車の高さ＋余白より低ければ、車は下に入れない。ベンチの --low-sofa）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.roommap import CAR_HEIGHT_M, CAR_RADIUS_M, Grid, Pose

FLOOR, LOW, HIGH, CLOSED = 0, 1, 2, 3
RES_M = 0.05
TSUBO_M2 = 3.3058
HOME_M2 = 30 * TSUBO_M2
# 壁の厚さと高さ
WALL_M = 0.10
WALL_H_M = 2.4
# 扉の幅（下限・上限）
DOOR_MIN_M, DOOR_MAX_M = 0.70, 0.90
# 扉の前の、家具を置かない深さ
DOOR_KEEP_M = 0.60
# 格子の家のまわりの余白
MARGIN_M = 0.5
# 人と snoo の道のりの刻みと長さ（秒）
AGENT_DT_S = 0.5
AGENT_T_S = 1800.0
# 人と snoo の大きさ（床に落とした円の半径、高さ）
PERSON_R_M, PERSON_H_M = 0.20, 1.6
SNOO_R_M, SNOO_H_M = 0.15, 0.27
# 低いソファ（make_home(low_sofa=True)）の下の端の高さ（m）の範囲
LOW_SOFA_Z0_M = (0.10, 0.18)
# 車の真ん中が入れないセル（app/roommap.blocked_cells と同じく、障害物を車の半径だけ広げる）
CAR_CELLS = int(math.ceil(CAR_RADIUS_M / RES_M))


def _q(v: float) -> float:
    """5 cm に丸める（セルの境目にそろえ、格子に落としたときに大きさが変わらないように）。"""
    return round(v / RES_M) * RES_M


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    """lo..hi のくじ。幅がちょうど 0 の所で、小数の誤差で hi < lo になっても落ちない。"""
    return float(rng.uniform(lo, max(hi, lo)))


@dataclass
class RoomSpec:
    kind: str  # ldk・bed・washitsu・hall・wash・bath・toilet・closet・genkan
    rect: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    closed: bool = False

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.rect
        return (x1 - x0) * (y1 - y0)


@dataclass
class Item:
    """置いた物（軸にそろえた箱）。layer は LOW か HIGH か rug。"""

    name: str
    rect: tuple[float, float, float, float]
    layer: str = "low"
    height: float = 0.4  # LOW の上の端（m）
    z0: float = 0.0  # HIGH の下の端（m）


@dataclass
class Agent:
    name: str  # person・snoo
    radius: float
    height: float
    traj: np.ndarray  # (T, 2) の位置（AGENT_DT_S おき）

    def at(self, t: float) -> tuple[float, float]:
        k = min(int(t / AGENT_DT_S), len(self.traj) - 1)
        return float(self.traj[k, 0]), float(self.traj[k, 1])


@dataclass
class Home:
    seed: int
    origin: float  # セル (0, 0) の角の座標（x・y とも。app/roommap.Grid と同じ）
    kind: np.ndarray  # (n, n) uint8：FLOOR・LOW・HIGH・CLOSED
    height: np.ndarray  # (n, n) float32：LOW の上の端（m）
    rug: np.ndarray  # (n, n) bool
    rooms: list[RoomSpec]
    items: list[Item]
    doors: list[tuple[float, float, float, float]]
    agents: list[Agent]
    start: Pose
    shape: str  # rect・L
    footprint_m2: float  # 家の外形の広さ（壁を含む）
    res: float = RES_M
    meta: dict = field(default_factory=dict)
    # (n, n) float32：HIGH の下の端（m）。HIGH でないセルは inf
    under: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.kind.shape[0]

    def grid(self) -> Grid:
        """この家と同じ大きさ・同じ原点の、空の地図。"""
        g = Grid(size_m=self.n * self.res, res=self.res)
        g.origin = self.origin
        return g

    def cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin) / self.res).astype(int)
        iy = np.floor((np.asarray(y) - self.origin) / self.res).astype(int)
        return ix, iy

    def passable(self) -> np.ndarray:
        """車が下を通れるセル（床と、下をくぐれる物。下の端が車より低い物はくぐれない）。"""
        high = self.kind == HIGH
        if self.under is not None:
            high &= self.under >= CAR_HEIGHT_M
        return (self.kind == FLOOR) | high

    def reachable_floor(self) -> np.ndarray:
        """出発点から、床（とくぐれる物の下）をたどって行けるセル。調べた割合の分母。"""
        n, labels = cv2.connectedComponents(self.passable().astype(np.uint8), connectivity=4)
        ix, iy = self.cell(self.start.x, self.start.y)
        return labels == labels[ix, iy]


# 手ざわり（写真の勾配の強さ、0〜1。app/snoo_belief.py）。家具・物は高く、白い壁・白い家電は低い
TEX_WALL = 0.45
TEX_WHITE = 0.05
TEX_FURNITURE = 0.8
WHITE_ITEMS = ("kitchen", "fridge", "wardrobe", "chest")


def home_texture(home: Home, white_walls: bool = False) -> np.ndarray:
    """家のセルごとの手ざわり（シミュレーターで、目印が見えるか・位置を見失うかを決める）。
    white_walls なら壁を白い（手ざわりの無い）壁にし、白い家電・たんすも低くする。"""
    tex = np.zeros(home.kind.shape, np.float32)
    tex[home.kind == CLOSED] = TEX_WHITE if white_walls else TEX_WALL
    tex[(home.kind == LOW) | (home.kind == HIGH)] = TEX_FURNITURE
    if white_walls:
        for it in home.items:
            if it.name in WHITE_ITEMS:
                x0, y0, x1, y1 = it.rect
                ix0, iy0 = home.cell(x0, y0)
                ix1, iy1 = home.cell(x1 - 1e-6, y1 - 1e-6)
                sl = (slice(max(ix0, 0), ix1 + 1), slice(max(iy0, 0), iy1 + 1))
                tex[sl] = np.where(home.kind[sl] != FLOOR, TEX_WHITE, tex[sl])
    return tex


# --- 部屋割り ---


def _split_side(
    rng: np.random.Generator, x0: float, x1: float, y0: float, y1: float, wet: bool
) -> list[RoomSpec]:
    """廊下の片側（x0..x1、奥行き y0..y1）を部屋に分ける。wet なら水回りを入れる。"""
    rooms: list[RoomSpec] = []
    x = x0
    if wet:
        # 水回りは廊下の奥（玄関の側）に寄せることが多い。ここでは x1 の側から置く
        wet_rooms = [("toilet", rng.uniform(0.9, 1.1)), ("wash", rng.uniform(1.6, 2.0))]
        wet_rooms.append(("bath", rng.uniform(1.6, 1.9)))
        xe = x1
        for kind, w in wet_rooms:
            w = _q(w)
            rooms.append(RoomSpec(kind, (xe - w, y0, xe, y1)))
            xe -= w
        x1 = xe
    while x1 - x > 1e-6:
        left = x1 - x
        if left < 2.2:
            if rooms and left < 1.2:
                # 狭すぎる残りは物入れ（扉を閉める）
                rooms.append(RoomSpec("closet", (x, y0, x1, y1), closed=True))
            else:
                rooms.append(RoomSpec("bed", (x, y0, x1, y1)))
            break
        w = _q(rng.uniform(2.6, 3.8))
        if left - w < 1.0:
            w = left
        kind = "washitsu" if rng.random() < 0.3 else "bed"
        rooms.append(RoomSpec(kind, (x, y0, min(x + w, x1), y1)))
        x += w
    return rooms


def _plan_rooms(rng: np.random.Generator, W: float, H: float) -> tuple[list[RoomSpec], dict]:
    """幅 W・奥行き H（W ≥ H）の長方形を、LDK・廊下・両側の部屋に分ける。"""
    L = _q(W * rng.uniform(0.36, 0.48))
    hw = _q(rng.uniform(0.95, 1.2))
    lo, hi = 2.4, H - hw - 2.4
    hy = _q(rng.uniform(lo, hi)) if hi > lo else _q((H - hw) / 2)
    gk = _q(rng.uniform(1.0, 1.4))
    rooms = [
        RoomSpec("ldk", (0.0, 0.0, L, H)),
        RoomSpec("hall", (L, hy, W - gk, hy + hw)),
        RoomSpec("genkan", (W - gk, hy, W, hy + hw)),
    ]
    wet_south = rng.random() < 0.5
    rooms += _split_side(rng, L, W, 0.0, hy, wet_south)
    rooms += _split_side(rng, L, W, hy + hw, H, not wet_south)
    # LDK が広すぎれば、角を 1 部屋に（LDK から入る和室・寝室）
    ldk = rooms[0]
    if ldk.area > 34 and rng.random() < 0.6:
        rw, rd = _q(rng.uniform(2.6, 3.4)), _q(rng.uniform(2.6, 3.2))
        rw = min(rw, L - 1.8)
        north = rng.random() < 0.5
        if north:
            rooms.append(RoomSpec("washitsu", (0.0, H - rd, rw, H)))
        else:
            rooms.append(RoomSpec("washitsu", (0.0, 0.0, rw, rd)))
    return rooms, {"ldk_len": L, "hall_y": hy, "hall_w": hw, "genkan": gk}


def _shared(a: RoomSpec, b: RoomSpec) -> tuple[str, float, float, float] | None:
    """2 部屋の共有する壁 ('x' か 'y', 壁の座標, 始め, 終わり)。'x' は x = 一定の壁。"""
    ax0, ay0, ax1, ay1 = a.rect
    bx0, by0, bx1, by1 = b.rect
    if bx0 <= ax0 and by0 <= ay0 and ax1 <= bx1 and ay1 <= by1:
        # a は b の角の部屋（LDK の中の和室）。b の外形に乗らない縁のうち長いほう
        edges = []
        if ax0 > bx0 + 1e-6:
            edges.append(("x", ax0, ay0, ay1))
        if ax1 < bx1 - 1e-6:
            edges.append(("x", ax1, ay0, ay1))
        if ay0 > by0 + 1e-6:
            edges.append(("y", ay0, ax0, ax1))
        if ay1 < by1 - 1e-6:
            edges.append(("y", ay1, ax0, ax1))
        return max(edges, key=lambda e: e[3] - e[2]) if edges else None
    for xa, xb in ((ax1, bx0), (ax0, bx1)):
        if abs(xa - xb) < 1e-6:
            lo, hi = max(ay0, by0), min(ay1, by1)
            if hi - lo > 1e-6:
                return ("x", xa, lo, hi)
    for ya, yb in ((ay1, by0), (ay0, by1)):
        if abs(ya - yb) < 1e-6:
            lo, hi = max(ax0, bx0), min(ax1, bx1)
            if hi - lo > 1e-6:
                return ("y", ya, lo, hi)
    return None


def _door_rect(rng, wall, width: float, mask_ok) -> tuple[float, float, float, float] | None:
    """共有する壁の上に、幅 width の扉を開ける位置。角から 15 cm は空ける。"""
    axis, c, lo, hi = wall
    lo, hi = lo + 0.15, hi - 0.15
    if hi - lo < width:
        return None
    for _ in range(20):
        s = _q(_u(rng, lo, hi - width))
        if axis == "x":
            r = (c - WALL_M / 2, s, c + WALL_M / 2, s + width)
        else:
            r = (s, c - WALL_M / 2, s + width, c + WALL_M / 2)
        if mask_ok(r):
            return r
    return None


# --- 格子に落とす ---


class _Raster:
    def __init__(self, n: int, origin: float):
        self.n, self.origin = n, origin

    def idx(self, rect) -> tuple[slice, slice]:
        x0, y0, x1, y1 = rect
        i0 = int(round((x0 - self.origin) / RES_M))
        i1 = max(int(round((x1 - self.origin) / RES_M)), i0 + 1)
        j0 = int(round((y0 - self.origin) / RES_M))
        j1 = max(int(round((y1 - self.origin) / RES_M)), j0 + 1)
        return slice(max(i0, 0), min(i1, self.n)), slice(max(j0, 0), min(j1, self.n))


# 部屋の種類ごとの家具。(名前, 幅, 奥行き, 置き方, 高さ, 出る確率)。置き方は wall（壁ぎわ）・
# free（真ん中のほう）。高さが負なら「くぐれる」物（天板の下の端 = -高さ、脚は 4 本）
_FURNITURE = {
    "ldk": [
        ("sofa", (1.6, 2.0), (0.8, 0.95), "wall", 0.75, 0.9),
        ("tv_stand", (1.2, 1.8), (0.4, 0.45), "wall", 0.45, 0.9),
        ("kitchen", (2.1, 2.7), (0.65, 0.65), "wall", 0.9, 1.0),
        ("fridge", (0.65, 0.7), (0.65, 0.7), "wall", 1.8, 0.9),
        ("shelf", (0.8, 1.2), (0.3, 0.4), "wall", 1.2, 0.7),
        ("cat_tower", (0.45, 0.6), (0.45, 0.6), "wall", 1.6, 0.8),
        ("box", (0.3, 0.5), (0.3, 0.45), "free", 0.3, 0.6),
    ],
    "bed": [
        ("bed", (0.95, 1.45), (1.95, 2.05), "wall", 0.45, 0.95),
        ("desk", (0.9, 1.2), (0.55, 0.65), "wall", -0.70, 0.5),
        ("wardrobe", (0.8, 1.2), (0.55, 0.6), "wall", 1.8, 0.6),
        ("box", (0.3, 0.5), (0.3, 0.45), "free", 0.3, 0.4),
    ],
    "washitsu": [
        ("chest", (0.8, 1.0), (0.4, 0.45), "wall", 1.0, 0.7),
        ("zataku", (0.8, 1.2), (0.6, 0.8), "free", -0.30, 0.6),
        ("cushion", (0.5, 0.55), (0.5, 0.55), "free", 0.06, 0.5),
        ("box", (0.3, 0.5), (0.3, 0.45), "free", 0.3, 0.3),
    ],
    "hall": [
        ("box", (0.3, 0.4), (0.3, 0.35), "wall", 0.3, 0.4),
    ],
}


def _legs(name: str, rect, z0: float, leg: float) -> list[Item]:
    x0, y0, x1, y1 = rect
    out = [Item(name + "_top", rect, "high", z0=z0)]
    for k, (lx, ly) in enumerate([(x0, y0), (x1 - leg, y0), (x0, y1 - leg), (x1 - leg, y1 - leg)]):
        out.append(Item(f"{name}_leg{k}", (lx, ly, lx + leg, ly + leg), "low", height=z0))
    return out


def make_home(seed: int, agent_t_s: float = AGENT_T_S, low_sofa: bool = False) -> Home:
    """くじ seed の家。作れなければ（部屋がつながらないなど）くじを引き直す。
    同じ seed なら同じ家になる。low_sofa なら、下の端が 10〜18 cm の低いソファを 1 つ足す
    （別のくじで置くので、ほかの家具・出発点は low_sofa でないときと同じ）。"""
    for attempt in range(300):
        home = _try_home(np.random.default_rng([seed, attempt]), seed, agent_t_s, low_sofa)
        if home is not None:
            home.meta["attempt"] = attempt
            return home
    raise RuntimeError(f"家を作れなかった（seed {seed}）")


def _try_home(
    rng: np.random.Generator, seed: int, agent_t_s: float, low_sofa: bool = False
) -> Home | None:
    shape = "L" if rng.random() < 0.4 else "rect"
    a = b = 0.0
    if shape == "L":
        a, b = rng.uniform(0.2, 0.35), rng.uniform(0.2, 0.3)
    # 外形：W × H（W が長い辺）。L 字は角 (aW × bH) を欠く
    W = _q(rng.uniform(10.0, 12.5))
    H = _q(HOME_M2 / (W * (1 - a * b)))
    if H < 7.6 or H > W:
        return None
    rooms, info = _plan_rooms(rng, W, H)
    n = int(math.ceil((max(W, H) + 2 * MARGIN_M) / RES_M))
    origin = -MARGIN_M
    R = _Raster(n, origin)
    house = np.zeros((n, n), bool)
    house[R.idx((0, 0, W, H))] = True
    notch = None
    if shape == "L":
        nw, nh = _q(a * W), _q(b * H)
        # 欠く角：LDK 側か、反対側。4 隅のどれか
        cx = 0.0 if rng.random() < 0.5 else W - nw
        cy = 0.0 if rng.random() < 0.5 else H - nh
        notch = (cx, cy, cx + nw, cy + nh)
        house[R.idx(notch)] = False
        # 欠いた角にかかる部屋を切り詰める（切り詰めると小さすぎる部屋は物入れにする）
        kept = []
        for r in rooms:
            x0, y0, x1, y1 = r.rect
            sl = R.idx(r.rect)
            left = house[sl].sum() * RES_M**2
            if left < 1e-6:
                continue
            if r.kind in ("hall", "genkan") and left < r.area - 1e-6:
                return None  # 廊下が切れる家は作り直す
            if left < 3.0 and r.kind not in ("toilet", "genkan"):
                r = RoomSpec("closet", r.rect, closed=True)
            kept.append(r)
        rooms = kept
    footprint = float(house.sum() * RES_M**2)

    kind = np.full((n, n), CLOSED, np.uint8)
    height = np.zeros((n, n), np.float32)
    under = np.full((n, n), np.inf, np.float32)
    kind[house] = FLOOR
    # 壁：部屋の縁ごとに 10 cm、外は家の外形の内側 10 cm
    wall = np.zeros((n, n), bool)
    for r in rooms:
        x0, y0, x1, y1 = r.rect
        h = WALL_M / 2
        for e in ((x0 - h, y0 - h, x0 + h, y1 + h), (x1 - h, y0 - h, x1 + h, y1 + h),
                  (x0 - h, y0 - h, x1 + h, y0 + h), (x0 - h, y1 - h, x1 + h, y1 + h)):  # fmt: skip
            wall[R.idx(e)] = True
    k = int(round(WALL_M / RES_M))
    inner = cv2.erode(house.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    wall = (wall | (house & ~inner)) & house

    # 扉：部屋どうしの共有する壁に開ける。どの部屋も LDK から扉をたどって行けるようにする
    def wall_ok(rect) -> bool:
        sl = R.idx(rect)
        return bool(house[sl].all())

    doors: list[tuple[float, float, float, float]] = []
    by_kind = {r.kind: r for r in rooms}
    hall, ldk = by_kind.get("hall"), by_kind.get("ldk")
    if hall is None or ldk is None:
        return None
    links: list[tuple[RoomSpec, RoomSpec]] = [(ldk, hall)]
    reached = {id(ldk), id(hall)}
    order = [r for r in rooms if r.kind not in ("ldk", "hall")]
    # 部屋ごとに、つながった部屋（廊下 → LDK → ほか）の中から扉を開ける相手を選ぶ
    for _ in range(3):
        for r in order:
            if id(r) in reached:
                continue
            if r.kind == "bath":
                pref = ["wash"]
            elif r.kind == "genkan":
                pref = ["hall"]
            else:
                pref = ["hall", "ldk", "bed", "washitsu", "wash"]
            for p in pref:
                cands = [o for o in rooms if o.kind == p and id(o) in reached]
                hit = False
                for o in cands:
                    w = _shared(r, o)
                    if w is None or w[3] - w[2] < DOOR_MIN_M + 0.3:
                        continue
                    links.append((r, o))
                    reached.add(id(r))
                    hit = True
                    break
                if hit:
                    break
    if any(id(r) not in reached for r in rooms if not r.closed and r.kind != "genkan"):
        return None
    # 閉まっている部屋：水回りはたいてい、寝室もときどき
    p_closed = {"bath": 1.0, "toilet": 0.85, "wash": 0.6, "bed": 0.1, "washitsu": 0.1}
    for r in rooms:
        if r.kind in p_closed and rng.random() < p_closed[r.kind]:
            r.closed = True
    # 閉まった部屋の先の部屋も閉まっている（風呂は洗面所から入る）
    for r, o in links:
        if o.closed:
            r.closed = True
    for r, o in links:
        if r.closed or o.closed or r.kind == "genkan":
            continue
        w = _shared(r, o)
        width = _q(rng.uniform(DOOR_MIN_M, DOOR_MAX_M))
        if "hall" in (r.kind, o.kind):
            width = _q(rng.uniform(0.8, 0.9))
        d = _door_rect(rng, w, width, wall_ok)
        if d is None:
            return None
        doors.append(d)
    for d in doors:
        wall[R.idx(d)] = False
    # 玄関と廊下の境は段（低い障害物にする。床ではないので調べる割合に数えない）
    kind[wall] = LOW
    height[wall] = WALL_H_M
    for r in rooms:
        sl = R.idx(r.rect)
        if r.closed:
            m = kind[sl] == FLOOR
            kind[sl][m] = CLOSED
        elif r.kind == "genkan":
            m = kind[sl] == FLOOR
            kind[sl][m] = LOW
            height[sl][m] = 0.03  # 段の下に靴（目は先を見通せる）

    # 扉の前は家具を置かない
    keep = np.zeros((n, n), bool)
    for x0, y0, x1, y1 in doors:
        horizontal = (x1 - x0) > (y1 - y0)
        if horizontal:
            e = (x0 - 0.1, y0 - DOOR_KEEP_M, x1 + 0.1, y1 + DOOR_KEEP_M)
        else:
            e = (x0 - DOOR_KEEP_M, y0 - 0.1, x1 + DOOR_KEEP_M, y1 + 0.1)
        keep[R.idx(e)] = True

    def door_cells() -> list[tuple[int, int]]:
        out = []
        for d in doors:
            sx, sy = R.idx(d)
            out.append(((sx.start + sx.stop) // 2, (sy.start + sy.stop) // 2))
        return out

    doors_c = door_cells()
    # セルごとの持ち主の部屋（後の部屋が上書き。LDK の角の和室は和室）
    owner = np.full((n, n), -1, np.int32)
    for i, r in enumerate(rooms):
        owner[R.idx(r.rect)] = i
    owner[~house] = -1

    def car_free(kind_now: np.ndarray) -> np.ndarray:
        """車の真ん中が入れるセル（くぐれる物の下は入れる）。"""
        blocked = (kind_now == LOW) | (kind_now == CLOSED)
        dist = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 5)
        return dist > CAR_CELLS

    def all_doors_connected(kind_now: np.ndarray) -> bool:
        free = car_free(kind_now)
        _, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
        ids = {int(labels[c]) for c in doors_c}
        return 0 not in ids and len(ids) == 1

    if doors_c and not all_doors_connected(kind):
        return None

    items: list[Item] = []
    rug = np.zeros((n, n), bool)

    def try_place(new: list[Item], clearance: float, room: int | None = None) -> bool:
        trial_k, trial_h, trial_u = kind.copy(), height.copy(), under.copy()
        for it in new:
            if room is not None and it.layer != "rug":
                sl = R.idx(it.rect)
                if (owner[sl] != room).any():
                    return False
            x0, y0, x1, y1 = it.rect
            pad = (x0 - clearance, y0 - clearance, x1 + clearance, y1 + clearance)
            sl = R.idx(it.rect)
            if it.layer != "rug":
                if (trial_k[sl] != FLOOR).any() or keep[sl].any():
                    return False
                # まわりの clearance に、ほかの家具が無いこと（壁はよい）
                ps = R.idx(pad)
                near = (trial_k[ps] != FLOOR) & ~wall[ps] & (trial_k[ps] != CLOSED)
                if near.any() and clearance > 0:
                    return False
        for it in new:
            sl = R.idx(it.rect)
            if it.layer == "low":
                trial_k[sl] = LOW
                trial_h[sl] = np.maximum(trial_h[sl], it.height)
            elif it.layer == "high":
                m = trial_k[sl] == FLOOR
                trial_k[sl][m] = HIGH
                trial_u[sl][m] = it.z0
        if not all_doors_connected(trial_k):
            return False
        kind[:], height[:], under[:] = trial_k, trial_h, trial_u
        items.extend(new)
        return True

    def interior(r: RoomSpec) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = r.rect
        h = WALL_M / 2 + 0.0
        return (x0 + h, y0 + h, x1 - h, y1 - h)

    def wall_spot(r: RoomSpec, w: float, d: float, g: np.random.Generator | None = None):
        g = rng if g is None else g
        ix0, iy0, ix1, iy1 = interior(r)
        side = int(g.integers(4))
        if side in (0, 1):  # 南・北の壁ぎわ（幅が x）
            if ix1 - ix0 < w:
                return None
            x = _q(_u(g, ix0, ix1 - w))
            y = iy0 if side == 0 else iy1 - d
            return (x, _q(y), x + w, _q(y) + d)
        if iy1 - iy0 < w:
            return None
        y = _q(_u(g, iy0, iy1 - w))
        x = ix0 if side == 2 else ix1 - d
        return (_q(x), y, _q(x) + d, y + w)

    def free_spot(r: RoomSpec, w: float, d: float, margin: float = 0.4):
        ix0, iy0, ix1, iy1 = interior(r)
        if rng.random() < 0.5:
            w, d = d, w
        if ix1 - ix0 < w + 2 * margin or iy1 - iy0 < d + 2 * margin:
            return None
        x = _q(_u(rng, ix0 + margin, ix1 - margin - w))
        y = _q(_u(rng, iy0 + margin, iy1 - margin - d))
        return (x, y, x + w, y + d)

    open_rooms = [r for r in rooms if not r.closed and r.kind not in ("genkan", "closet")]
    for r in open_rooms:
        ri = rooms.index(r)
        # LDK はダイニング（机と椅子）とラグ・ローテーブルも
        if r.kind == "ldk":
            for _ in range(20):
                tw, td = _q(rng.uniform(1.2, 1.6)), _q(rng.uniform(0.75, 0.9))
                spot = free_spot(r, tw, td, margin=0.7)
                if spot is None:
                    break
                x0, y0, x1, y1 = spot
                group = _legs("dining", spot, 0.70, 0.05)
                long_x = (x1 - x0) >= (y1 - y0)
                chairs = []
                nch = int(rng.integers(2, 5))
                for c in range(nch):
                    # 長い辺の両側に、半分ほど机の下に入れて置く。椅子は 40 cm 角
                    sgn = -1 if c % 2 == 0 else 1
                    k2 = c // 2
                    if long_x:
                        cx = x0 + (x1 - x0) * (0.3 + 0.4 * k2)
                        cy = (y0 - 0.2) if sgn < 0 else (y1 + 0.2)
                    else:
                        cy = y0 + (y1 - y0) * (0.3 + 0.4 * k2)
                        cx = (x0 - 0.2) if sgn < 0 else (x1 + 0.2)
                    cr = (_q(cx - 0.2), _q(cy - 0.2), _q(cx + 0.2), _q(cy + 0.2))
                    chairs += _legs(f"chair{c}", cr, 0.43, 0.05)
                if try_place(group + chairs, 0.0, ri):
                    break
            for _ in range(10):
                spot = free_spot(r, _q(rng.uniform(1.4, 2.0)), _q(rng.uniform(1.2, 1.6)), 0.3)
                if spot is None or not (owner[R.idx(spot)] == ri).all():
                    continue
                if try_place([Item("rug", spot, "rug")], 0.0):
                    rug[R.idx(spot)] = True
                    x0, y0, x1, y1 = spot
                    lt = (_q(x0 + 0.3), _q(y0 + 0.3), _q(x1 - 0.3), _q(y1 - 0.3))
                    if rng.random() < 0.5:
                        try_place(_legs("low_table", lt, 0.30, 0.05), 0.0, ri)
                    else:
                        try_place([Item("low_table", lt, "low", height=0.35)], 0.0, ri)
                    break
        for name, (w0, w1), (d0, d1), how, h, p in _FURNITURE.get(r.kind, []):
            if rng.random() > p:
                continue
            for _ in range(15):
                w, d = _q(rng.uniform(w0, w1)), _q(rng.uniform(d0, d1))
                spot = wall_spot(r, w, d) if how == "wall" else free_spot(r, w, d)
                if spot is None:
                    continue
                new = _legs(name, spot, -h, 0.05) if h < 0 else [Item(name, spot, "low", height=h)]
                if try_place(new, 0.05 if how == "wall" else 0.25, ri):
                    break
    # コード（細い低い物）：LDK と寝室の壁ぎわから 0.5〜1.5 m
    for r in open_rooms:
        ri = rooms.index(r)
        if r.kind not in ("ldk", "bed") or rng.random() < 0.4:
            continue
        for _ in range(10):
            ix0, iy0, ix1, iy1 = interior(r)
            length = _q(rng.uniform(0.5, 1.5))
            if rng.random() < 0.5:
                x = _q(_u(rng, ix0, ix1 - length))
                y = _q(rng.choice([iy0 + 0.1, iy1 - 0.15]))
                rect = (x, y, x + length, y + RES_M)
            else:
                y = _q(_u(rng, iy0, iy1 - length))
                x = _q(rng.choice([ix0 + 0.1, ix1 - 0.15]))
                rect = (x, y, x + RES_M, y + length)
            if try_place([Item("cable", rect, "low", height=0.03)], 0.0, ri):
                break

    # 出発点：LDK か廊下の、障害物から 40 cm 以上離れた床
    free = car_free(kind)
    dist = cv2.distanceTransform((kind != LOW).astype(np.uint8) * (kind != CLOSED), cv2.DIST_L2, 5)
    mine = np.isin(owner, [i for i, r in enumerate(rooms) if r.kind in ("ldk", "hall")])
    spots_a = np.argwhere(mine & free & (dist * RES_M >= 0.4) & (kind == FLOOR))
    if not len(spots_a):
        return None
    ci, cj = spots_a[int(rng.integers(len(spots_a)))]
    sx, sy = origin + (ci + 0.5) * RES_M, origin + (cj + 0.5) * RES_M
    start = Pose(float(sx), float(sy), float(rng.uniform(-math.pi, math.pi)))
    if low_sofa:
        # 下の端が 10〜18 cm の低いソファ（別のくじ。ほかの家具・出発点は変えない）。車がくぐれない
        # 高さなら、車が全部の扉へ行けて、出発点から 0.6 m 離れている所にだけ置く
        g = np.random.default_rng([seed, 78])
        z0 = round(float(g.uniform(*LOW_SOFA_Z0_M)), 3)
        rooms_ok = [r for r in open_rooms if r.kind in ("ldk", "bed", "washitsu")]
        for _ in range(200 if rooms_ok else 0):
            r = rooms_ok[int(g.integers(len(rooms_ok)))]
            spot = wall_spot(r, _q(g.uniform(1.4, 1.8)), _q(g.uniform(0.75, 0.9)), g)
            if spot is None:
                continue
            x0, y0, x1, y1 = spot
            if math.hypot(min(max(sx, x0), x1) - sx, min(max(sy, y0), y1) - sy) < 0.6:
                continue
            before = kind.copy(), height.copy(), under.copy(), len(items)
            if not try_place(_legs("low_sofa", spot, z0, 0.05), 0.05, rooms.index(r)):
                continue
            low = (kind == HIGH) & (under < CAR_HEIGHT_M)
            if all_doors_connected(np.where(low, LOW, kind).astype(np.uint8)):
                break
            kind[:], height[:], under[:] = before[:3]
            del items[before[3] :]

    home = Home(
        seed, origin, kind, height, rug, rooms, items, doors, [], start, shape, footprint,
        meta={**info, "W": W, "H": H, "notch": notch}, under=under,
    )  # fmt: skip
    home.agents = make_agents(home, rng, agent_t_s)
    return home


# --- 人と snoo ---


def agent_ok(home: Home, radius: float, name: str) -> np.ndarray:
    """人・snoo の真ん中が立てるセル。人は机の下をくぐらない。車が行ける床とつながった所だけ。"""
    free = home.passable() & (home.kind != HIGH) if name == "person" else home.passable()
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * RES_M
    return (dist > radius + RES_M) & home.reachable_floor()


def behavior_snoo(home: Home, start: tuple[float, float], seed: int):
    """車に反応する snoo（app/snoo_behavior.BehaviorAgent）。start から始める。"""
    from app.snoo_behavior import BehaviorAgent, BehaviorPose, BehaviorSnoo, grid_free

    rng = np.random.default_rng(seed)
    free = grid_free(agent_ok(home, SNOO_R_M, "snoo"), home.origin, home.res)
    pose = BehaviorPose(start[0], start[1], float(rng.uniform(-math.pi, math.pi)))
    return BehaviorAgent(BehaviorSnoo(pose, free, rng), SNOO_R_M, SNOO_H_M)


def _walk(
    home: Home, rng: np.random.Generator, radius: float, speed: float, walk_s, rest_s,
    t_s: float, away: tuple[float, float], name: str, height: float,
) -> Agent:  # fmt: skip
    """床の上を、歩く（向きを少しずつ変える）・止まるを繰り返す道のり。"""
    ok = agent_ok(home, radius, name)
    cells = np.argwhere(ok)
    cx, cy = home.origin + (cells[:, 0] + 0.5) * RES_M, home.origin + (cells[:, 1] + 0.5) * RES_M
    far = np.hypot(cx - away[0], cy - away[1]) > 1.0
    pick = cells[far] if far.any() else cells
    i, j = pick[int(rng.integers(len(pick)))]
    x, y = home.origin + (i + 0.5) * RES_M, home.origin + (j + 0.5) * RES_M
    heading = float(rng.uniform(-math.pi, math.pi))
    steps = int(t_s / AGENT_DT_S) + 1
    traj = np.zeros((steps, 2), np.float32)
    walking = rng.random() < 0.5
    timer = float(rng.uniform(*(walk_s if walking else rest_s)))
    for k in range(steps):
        traj[k] = (x, y)
        timer -= AGENT_DT_S
        if timer <= 0:
            walking = not walking
            timer = float(rng.uniform(*(walk_s if walking else rest_s)))
            heading += float(rng.normal(0, 1.5))
        if not walking:
            continue
        heading += float(rng.normal(0, 0.3))
        nx = x + speed * AGENT_DT_S * math.cos(heading)
        ny = y + speed * AGENT_DT_S * math.sin(heading)
        a, b = home.cell(nx, ny)
        if 0 <= a < home.n and 0 <= b < home.n and ok[a, b]:
            x, y = nx, ny
        else:
            heading += float(rng.choice([-1, 1])) * float(rng.uniform(1.0, 2.5))
    return Agent(name, radius, height, traj)


def make_agents(home: Home, rng: np.random.Generator, t_s: float = AGENT_T_S) -> list[Agent]:
    """人 1〜2 人と snoo。出発点の 1 m 以内からは始めない。"""
    away = (home.start.x, home.start.y)
    out = []
    for _ in range(int(rng.integers(1, 3))):
        out.append(_walk(home, rng, PERSON_R_M, 0.35, (3, 15), (10, 90), t_s, away, "person",
                         PERSON_H_M))  # fmt: skip
    out.append(_walk(home, rng, SNOO_R_M, 0.25, (2, 10), (10, 180), t_s, away, "snoo", SNOO_H_M))
    return out


# --- 絵 ---

_COLORS = {
    FLOOR: (235, 235, 235),
    LOW: (60, 60, 70),
    HIGH: (150, 190, 220),
    CLOSED: (120, 110, 100),
}


def render_home(home: Home, px: int = 4, t_s: float | None = None, known=None) -> np.ndarray:
    """家を上から見た絵（BGR）。行 0 が +y の端（app/snoo_mapper.grid_png と同じ向き）。

    known（app/roommap.Grid の state）を渡すと、調べた床を緑、見た障害物を赤で重ねる。
    """
    img = np.zeros((home.n, home.n, 3), np.uint8)
    for k, c in _COLORS.items():
        img[home.kind == k] = c
    img[(home.kind == LOW) & (home.height < 0.1)] = (90, 140, 200)  # コード・玄関の段
    img[home.rug & (home.kind == FLOOR)] = (200, 215, 225)
    if known is not None:
        from app.roommap import FREE, OCCUPIED

        seen = known == FREE
        img[seen] = (0.55 * img[seen] + 0.45 * np.array([80, 200, 80])).astype(np.uint8)
        img[known == OCCUPIED] = (40, 40, 200)
    img = img.transpose(1, 0, 2)[::-1].copy()  # [ix, iy] → 行が y（上が +y）
    img = cv2.resize(img, (home.n * px, home.n * px), interpolation=cv2.INTER_NEAREST)

    def to_px(x, y):
        return (
            int((x - home.origin) / home.res * px),
            int((home.n - (y - home.origin) / home.res) * px),
        )

    colors = {"person": (180, 60, 160), "snoo": (0, 140, 255)}
    for a in home.agents:
        pts = np.array([to_px(x, y) for x, y in a.traj[:: max(1, len(a.traj) // 600)]], np.int32)
        cv2.polylines(img, [pts], False, colors[a.name], 1, cv2.LINE_AA)
        if t_s is not None:
            cv2.circle(img, to_px(*a.at(t_s)), int(a.radius / home.res * px), colors[a.name], -1)
    s = home.start
    p0 = to_px(s.x, s.y)
    p1 = to_px(s.x + 0.4 * math.cos(s.yaw), s.y + 0.4 * math.sin(s.yaw))
    cv2.arrowedLine(img, p0, p1, (0, 0, 255), 2, tipLength=0.4)
    for r in home.rooms:
        x0, y0, x1, y1 = r.rect
        cx, cy = to_px((x0 + x1) / 2, (y0 + y1) / 2)
        label = r.kind + ("*" if r.closed else "")
        cv2.putText(img, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img
