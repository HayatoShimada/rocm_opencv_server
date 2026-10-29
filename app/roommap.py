"""PiCar-X の間取り（5 cm の占有格子）と、次に撮りに行く所（Next-Best-View）を決める。

ルンバのように、止まって首を振って撮る → 地図を更新する → 次の位置へ少し走る、を繰り返して
部屋を探る。車は前輪で曲がる（その場で向きを変えられない）ので、道のりは最小の回転半径を
考えた Dubins 経路（前進だけ）で測る。後ろにはセンサーが無いので後退はしない。

次の位置は「新しく見えるはずの未知のセル × exp(-道のり / 距離の重み)」がいちばん大きい所にする。
見える範囲はカメラの画角とパンの振り幅（± HALF_VIEW_DEG）、見通せる距離（VIEW_RANGE_M）で、
障害物の陰は見えないものとして数える。走れるのは、車の大きさ分を広げた障害物に当たらず、
床が見えている（FREE の）セルだけ（見ていない所には入らない。バンパーの代わり）。

座標は m。原点は最初に撮った位置の床、x は最初の車の向き、y は左、yaw は左回りが +。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
RES_M = 0.05
# 車の外形（約 26x19 cm）を囲む円の半径と余裕
CAR_RADIUS_M = 0.17
# 最小の回転半径。ホイールベースは cau-se/ARCHES-PiCar-X の URDF（Apache-2.0）の 14.0 cm
# （後輪の軸から前輪の操舵軸まで）。ハンドル ±30° で 14 / tan 30° = 24.2 cm なので、余裕を見て 26 cm
TURN_RADIUS_M = 0.26
WHEELBASE_M = 0.14
# 車の前後の端と、カメラ（パンの軸）の位置。どれも後ろの車軸の中央から前へ測る（m）。
# 前後の端は写真から見積もった（測っていない）。カメラの位置も測っていない。
# ステーションの位置（Pose）はカメラの真下なので、車の動き（自転車モデル）を当てるときは
# 後ろの車軸に直してから動かす（app/snoo_sim.py もここの値を使う）
CAR_REAR_M = -0.05
CAR_FRONT_M = 0.21
CAM_X_M = 0.16
# 前の超音波（バンパーの真ん中）。±15° の円すいで、3 m まで測れる
SONAR_HALF_DEG = 15.0
SONAR_RANGE_M = 3.0
# カメラの真下から超音波まで（車の前の端）
SONAR_AHEAD_M = CAR_FRONT_M - CAM_X_M
# 超音波の読みのこれだけ手前までを床にする（読みのぶれと、セルの大きさの分）
SONAR_FREE_MARGIN_M = 0.10
# 1 回の撮影で見える範囲：画角 54° の半分 + パン ±35°
HALF_VIEW_DEG = 62.0
VIEW_RANGE_M = 2.0
# 道のりの重み（この距離で利得が 1/e になる）
DISTANCE_SCALE_M = 1.5
# 1 回に走る長さの上限
MAX_STEP_M = 0.40
# 1 歩の向きは、経路のこの距離だけ先の点を目指して決める（pure pursuit）
LOOKAHEAD_M = 0.30
# 1 回の観測で足す log-odds。障害物は強く、床は弱く数える（同じセルに両方あれば障害物になる。
# 見落とした障害物にぶつかるほうが、床を走れないことにするより悪い）
OCCUPIED_HIT = 2.0
FREE_HIT = -0.85


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float  # rad


class Grid:
    """5 cm の占有格子。log-odds で持ち、何回も見た結果を足し合わせる。"""

    def __init__(self, size_m: float = 12.0, res: float = RES_M):
        self.res = res
        self.n = int(round(size_m / res))
        self.origin = -size_m / 2  # セル (0, 0) の角の座標（x・y とも）
        self.logodds = np.zeros((self.n, self.n), np.float32)  # [ix, iy]

    def cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin) / self.res).astype(int)
        iy = np.floor((np.asarray(y) - self.origin) / self.res).astype(int)
        return ix, iy

    def center(self, ix, iy):
        return self.origin + (np.asarray(ix) + 0.5) * self.res, self.origin + (
            np.asarray(iy) + 0.5
        ) * self.res

    def inside(self, ix, iy):
        return (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)

    def mark_free(self, x, y) -> None:
        """床の点のあるセルを、走れる側に寄せる。"""
        self._add(x, y, FREE_HIT)

    def mark_occupied(self, x, y) -> None:
        """障害物の点のあるセルを、ふさがっている側に寄せる。"""
        self._add(x, y, OCCUPIED_HIT)

    def _add(self, x, y, value: float) -> None:
        """点のあるセルに log-odds を足す。1 回の観測で同じセルに何点あっても 1 回だけ。"""
        ix, iy = self.cell(x, y)
        ok = self.inside(ix, iy)
        cells = np.unique(np.stack([ix[ok], iy[ok]], 1), axis=0)
        if len(cells):
            self.logodds[cells[:, 0], cells[:, 1]] += value
        np.clip(self.logodds, -4, 4, out=self.logodds)

    def state(self) -> np.ndarray:
        s = np.full(self.logodds.shape, UNKNOWN, np.uint8)
        s[self.logodds < -0.4] = FREE
        s[self.logodds > 0.4] = OCCUPIED
        return s


def mark_sonar(grid: Grid, camera: Pose, distance_m: float | None) -> None:
    """前の超音波の 1 回の読みを格子に足す。camera はそのステーションの位置（カメラの真下）。

    超音波は ±SONAR_HALF_DEG の円すいでいちばん近い物までの距離を返すので、どの向きに
    あったかは分からない。そこで読みの距離の弧（円すいの幅いっぱい）を障害物にし、
    それより SONAR_FREE_MARGIN_M 手前までを床にする。SONAR_RANGE_M より遠い（届かなかった）
    ときは床だけ。測れなかった（None）ときは何もしない。
    床は、もう障害物と見ているセルには足さない。超音波は床から 4.5 cm の高さで、それより
    低い物（カメラが見た段差など）を見落とすので、カメラの障害物を消さないようにする。
    """
    if distance_m is None or not math.isfinite(distance_m) or distance_m <= 0:
        return
    hit = distance_m <= SONAR_RANGE_M
    c, s = math.cos(camera.yaw), math.sin(camera.yaw)
    ox, oy = camera.x + SONAR_AHEAD_M * c, camera.y + SONAR_AHEAD_M * s
    half = math.radians(SONAR_HALF_DEG)
    step = grid.res / 2

    def cone(r: np.ndarray):
        # 弧の上の点が step より離れないよう、いちばん遠い所に合わせて向きを刻む
        n = int(math.ceil(2 * half * float(r.max()) / step)) + 1
        a = camera.yaw + np.linspace(-half, half, n)
        rr, aa = np.meshgrid(r, a)
        return ox + rr * np.cos(aa), oy + rr * np.sin(aa)

    if hit:
        grid.mark_occupied(*cone(np.array([distance_m])))
    free_to = distance_m - SONAR_FREE_MARGIN_M if hit else SONAR_RANGE_M
    if free_to <= step:
        return
    x, y = cone(np.arange(step, free_to, step))
    ix, iy = grid.cell(x.ravel(), y.ravel())
    ok = grid.inside(ix, iy)
    cells = np.unique(np.stack([ix[ok], iy[ok]], 1), axis=0)
    cells = cells[grid.logodds[cells[:, 0], cells[:, 1]] <= 0]
    grid.logodds[cells[:, 0], cells[:, 1]] += FREE_HIT
    np.clip(grid.logodds, -4, 4, out=grid.logodds)


def inflate(occupied: np.ndarray, radius_cells: int) -> np.ndarray:
    """障害物を radius_cells だけ広げる（車の中心がそこに入るとぶつかる）。"""
    out = occupied.copy()
    r = radius_cells
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1]
    for dx, dy in zip(xs[xs**2 + ys**2 <= r * r], ys[xs**2 + ys**2 <= r * r], strict=True):
        out |= np.roll(np.roll(occupied, dx, 0), dy, 1)
    return out


def frontiers(state: np.ndarray) -> np.ndarray:
    """床が見えていて、隣（上下左右）に見ていないセルがあるセル。"""
    unknown = state == UNKNOWN
    near = np.zeros_like(unknown)
    near[1:, :] |= unknown[:-1, :]
    near[:-1, :] |= unknown[1:, :]
    near[:, 1:] |= unknown[:, :-1]
    near[:, :-1] |= unknown[:, 1:]
    return (state == FREE) & near


def view_gain(grid: Grid, state: np.ndarray, pose: Pose, rays: int = 61) -> int:
    """pose で首を振って撮ったときに見えるはずの未知のセルの数（障害物の陰は数えない）。"""
    seen: set[tuple[int, int]] = set()
    half = math.radians(HALF_VIEW_DEG)
    steps = int(VIEW_RANGE_M / (grid.res * 0.5))
    for a in np.linspace(pose.yaw - half, pose.yaw + half, rays):
        ca, sa = math.cos(a), math.sin(a)
        for k in range(1, steps + 1):
            d = k * grid.res * 0.5
            ix, iy = grid.cell(pose.x + d * ca, pose.y + d * sa)
            if not grid.inside(ix, iy):
                break
            s = state[ix, iy]
            if s == OCCUPIED:
                break
            if s == UNKNOWN:
                seen.add((int(ix), int(iy)))
    return len(seen)


# --- Dubins 経路（前進だけ・最小回転半径 r）。Shkel & Lumelsky の形 ---


def _mod2pi(a: float) -> float:
    return a % (2 * math.pi)


def dubins(start: Pose, goal: Pose, r: float = TURN_RADIUS_M) -> tuple[float, str, tuple] | None:
    """いちばん短い Dubins 経路の (長さ m, 型 'LSL' など, 各区間の長さ（r で割った角度・距離）)。"""
    dx, dy = goal.x - start.x, goal.y - start.y
    d = math.hypot(dx, dy) / r
    th = math.atan2(dy, dx)
    a, b = _mod2pi(start.yaw - th), _mod2pi(goal.yaw - th)
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    best = None

    def consider(kind, t, p, q):
        nonlocal best
        if t is None:
            return
        length = (t + p + q) * r
        if best is None or length < best[0]:
            best = (length, kind, (t, p, q))

    # LSL
    p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if p2 >= 0:
        tmp = math.atan2(cb - ca, d + sa - sb)
        consider("LSL", _mod2pi(-a + tmp), math.sqrt(p2), _mod2pi(b - tmp))
    # RSR
    p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p2 >= 0:
        tmp = math.atan2(ca - cb, d - sa + sb)
        consider("RSR", _mod2pi(a - tmp), math.sqrt(p2), _mod2pi(-b + tmp))
    # LSR
    p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        consider("LSR", _mod2pi(-a + tmp), p, _mod2pi(-_mod2pi(b) + tmp))
    # RSL
    p2 = d * d - 2 + 2 * cab - 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        consider("RSL", _mod2pi(a - tmp), p, _mod2pi(b - tmp))
    # RLR
    tmp = (6 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8
    if abs(tmp) <= 1:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(a - math.atan2(ca - cb, d - sa + sb) + p / 2)
        consider("RLR", t, p, _mod2pi(a - b - t + p))
    # LRL
    tmp = (6 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8
    if abs(tmp) <= 1:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(-a + math.atan2(-ca + cb, d + sa - sb) + p / 2)
        consider("LRL", t, p, _mod2pi(b - a - t + p))
    return best


def sample_path(start: Pose, kind: str, segs: tuple, r: float = TURN_RADIUS_M, step: float = 0.02):
    """Dubins 経路を step m ごとの Pose の列にする。"""
    poses = [start]
    x, y, yaw = start.x, start.y, start.yaw
    for c, seg in zip(kind, segs, strict=True):
        length = seg * r
        n = max(1, int(math.ceil(length / step)))
        ds = length / n
        for _ in range(n):
            if c == "S":
                x += ds * math.cos(yaw)
                y += ds * math.sin(yaw)
            else:
                turn = ds / r if c == "L" else -ds / r
                x += r * (math.sin(yaw + turn) - math.sin(yaw)) * (1 if c == "L" else -1)
                y += r * (math.cos(yaw) - math.cos(yaw + turn)) * (1 if c == "L" else -1)
                yaw += turn
            poses.append(Pose(x, y, yaw))
    return poses


# --- 車の動き（自転車モデル）。Pi と同じ指令（steer_deg は右が +、距離は後ろの車輪の進み） ---


def curvature(steer_deg: float) -> float:
    """後ろの車軸の中心が描く円の曲率（1/m、左が +）。steer_deg は右が +（Pi と同じ）。"""
    return -math.tan(math.radians(steer_deg)) / WHEELBASE_M


def arc(x: float, y: float, yaw: float, steer_deg: float, ds: float) -> tuple[float, float, float]:
    """後ろの車軸の中心を、ハンドル steer_deg のまま ds m 進めた姿勢（ちょうど円弧の上）。
    ds が負なら同じ弧を後ろへ戻る。
    """
    k = curvature(steer_deg)
    if abs(k) < 1e-9:
        return x + ds * math.cos(yaw), y + ds * math.sin(yaw), yaw
    nyaw = yaw + k * ds
    return (
        x + (math.sin(nyaw) - math.sin(yaw)) / k,
        y - (math.cos(nyaw) - math.cos(yaw)) / k,
        nyaw,
    )


def rear_axle(camera: Pose) -> Pose:
    """カメラの真下の姿勢（ステーションの位置）から、後ろの車軸の中央の姿勢へ。"""
    c, s = math.cos(camera.yaw), math.sin(camera.yaw)
    return Pose(camera.x - CAM_X_M * c, camera.y - CAM_X_M * s, camera.yaw)


def camera_of(rear: Pose) -> Pose:
    c, s = math.cos(rear.yaw), math.sin(rear.yaw)
    return Pose(rear.x + CAM_X_M * c, rear.y + CAM_X_M * s, rear.yaw)


def path_is_clear(grid: Grid, blocked: np.ndarray, poses: list[Pose]) -> bool:
    for p in poses:
        ix, iy = grid.cell(p.x, p.y)
        if not grid.inside(ix, iy) or blocked[ix, iy]:
            return False
    return True


@dataclass
class Plan:
    target: Pose
    gain: int
    length_m: float
    path: list[Pose]
    steer_deg: float  # 最初の 1 歩のハンドル（右が +。Pi の steer と同じ向き）
    step_m: float


def next_best_view(grid: Grid, pose: Pose, candidates_max: int = 400, rng=None) -> Plan | None:
    """次に撮りに行く位置と、そこへの最初の 1 歩。行ける所に見るものが無ければ None。"""
    rng = rng or np.random.default_rng(0)
    state = grid.state()
    occupied = state == OCCUPIED
    # 車の中心が入れない所：広げた障害物と、床が見えていない所
    blocked = inflate(occupied, int(math.ceil(CAR_RADIUS_M / grid.res))) | (state != FREE)
    # いま居るセルの周りは撮ったときに足元が写らないので、走れることにする
    ix0, iy0 = grid.cell(pose.x, pose.y)
    r0 = int(math.ceil(CAR_RADIUS_M / grid.res))
    blocked[max(ix0 - r0, 0) : ix0 + r0 + 1, max(iy0 - r0, 0) : iy0 + r0 + 1] &= occupied[
        max(ix0 - r0, 0) : ix0 + r0 + 1, max(iy0 - r0, 0) : iy0 + r0 + 1
    ]
    front = np.argwhere(frontiers(state))
    if not len(front):
        return None
    # 候補：境目の近く（1.2 m 以内）の、車が入れるセル
    free_ok = np.argwhere(~blocked)
    if not len(free_ok):
        return None
    fx, fy = grid.center(front[:, 0], front[:, 1])
    cx, cy = grid.center(free_ok[:, 0], free_ok[:, 1])
    near = np.zeros(len(free_ok), bool)
    for start in range(0, len(fx), 256):  # メモリを食わないよう分けて測る
        d2 = (cx[:, None] - fx[None, start : start + 256]) ** 2 + (
            cy[:, None] - fy[None, start : start + 256]
        ) ** 2
        near |= (d2 < 1.2**2).any(axis=1)
    idx = np.flatnonzero(near)
    if len(idx) > candidates_max:
        idx = rng.choice(idx, candidates_max, replace=False)
    best: Plan | None = None
    best_utility = 0.0
    for i in idx:
        x, y = float(cx[i]), float(cy[i])
        # 向きは、いちばん近い境目のほうを向く
        j = int(np.argmin((fx - x) ** 2 + (fy - y) ** 2))
        yaw = math.atan2(fy[j] - y, fx[j] - x)
        target = Pose(x, y, yaw)
        gain = view_gain(grid, state, target)
        if gain == 0:
            continue
        route = dubins(pose, target)
        if route is None:
            continue
        length, kind, segs = route
        utility = gain * math.exp(-length / DISTANCE_SCALE_M)
        if utility <= best_utility:
            continue
        path = sample_path(pose, kind, segs)
        if not path_is_clear(grid, blocked, path):
            continue
        steer, step = first_step(grid, blocked, pose, path, length)
        if step <= 0:
            continue
        best_utility = utility
        best = Plan(target, gain, length, path, steer, step)
    return best


def first_step(
    grid: Grid, blocked: np.ndarray, pose: Pose, path: list[Pose], length: float
) -> tuple[float, float]:
    """経路の LOOKAHEAD_M 先の点を目指す 1 本の弧を、ハンドル角（右が +、度）と走る長さ（m）にする。

    Pi は 1 歩をハンドル角 1 つで走るので、Dubins 経路の区間をそのまま渡すと、
    最初の曲がりが数 cm しかないときに数 cm 走っては撮り直すことになる。そこで経路の少し先の点へ
    向かう弧（pure pursuit）で MAX_STEP_M までまとめて進む。弧は経路から少し外れるので、
    ぶつからない所までに縮める（5 cm も進めなければ 0 を返し、この候補は使わない）。
    """
    goal_at = min(LOOKAHEAD_M, length)
    travelled, aim = 0.0, path[-1]
    for a, b in zip(path, path[1:], strict=False):
        travelled += math.hypot(b.x - a.x, b.y - a.y)
        if travelled >= goal_at:
            aim = b
            break
    dx, dy = aim.x - pose.x, aim.y - pose.y
    ahead = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy
    left = -math.sin(pose.yaw) * dx + math.cos(pose.yaw) * dy
    curvature = 2 * left / max(ahead * ahead + left * left, 1e-9)  # 左が +
    curvature = max(-1 / TURN_RADIUS_M, min(1 / TURN_RADIUS_M, curvature))
    run = min(MAX_STEP_M, length)
    if abs(curvature) < 1e-3:
        arc = sample_path(pose, "S", (run,), r=1.0)
    else:
        radius = 1 / abs(curvature)
        arc = sample_path(pose, "L" if curvature > 0 else "R", (run / radius,), r=radius)
    clear = 0.0
    for a, b in zip(arc, arc[1:], strict=False):
        ix, iy = grid.cell(b.x, b.y)
        if not grid.inside(ix, iy) or blocked[ix, iy]:
            break
        clear += math.hypot(b.x - a.x, b.y - a.y)
    if clear < 0.05:
        return 0.0, 0.0
    steer = -math.degrees(math.atan(WHEELBASE_M * curvature))
    return steer, min(run, clear)
