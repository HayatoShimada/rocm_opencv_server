"""PiCar-X の間取り（5 cm の占有格子）と、次に撮りに行く所（Next-Best-View）を決める。

ルンバのように、止まって首を振って撮る → 地図を更新する → 次の位置へ少し走る、を繰り返して
部屋を探る。車は前輪で曲がる（その場で向きを変えられない）ので、道のりは最小の回転半径を
考えた Dubins 経路（前進だけ）で測る。後ろにはセンサーが無いので、後ろへ走るのは、行き場が
無くなったときに来た道をそのまま戻るとき（retrace）だけにする。

次の位置は「新しく見えるはずの未知のセル × exp(-道のり / 距離の重み)」がいちばん大きい所にする。
見える範囲はカメラの画角とパンの振り幅（± HALF_VIEW_DEG）、見通せる距離（VIEW_RANGE_M）で、
障害物の陰は見えないものとして数える。走れるのは、車の大きさ分を広げた障害物に当たらず、
床が見えている（FREE の）セルだけ（見ていない所には入らない。バンパーの代わり）。

座標は m。原点は最初に撮った位置の床、x は最初の車の向き、y は左、yaw は左回りが +。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

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
# 車の幅と高さ（見積もり。測っていない）。app/clearance.py・app/snoo_twin.py もここの値を使う
CAR_WIDTH_M = 0.19
CAR_HEIGHT_M = 0.12
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
# Pi が long_step を受けるときの上限。40 cm を超える分は、まっすぐの区間で、地図で床と
# 分かっている所だけ（turn_then_straight）。3 回目の実走は 15 か所が 47 x 65 cm に収まった
LONG_STEP_M = 0.60
# でたらめな 1 歩の長さの下限
MIN_RANDOM_STEP_M = 0.20
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


# 境目の塊（8 近傍でつながったセル）がこれより小さければ、次に撮る位置の手がかりにしない
MIN_FRONTIER_CELLS = int(os.getenv("SNOO_MIN_FRONTIER_CELLS", "3"))
# 床に囲まれた見ていない穴がこれより小さければ（VGGT の点の抜け）、穴の縁を境目にしない
MIN_UNKNOWN_HOLE_CELLS = int(os.getenv("SNOO_MIN_UNKNOWN_HOLE_CELLS", "6"))


def big_frontiers(
    state: np.ndarray,
    min_cells: int = MIN_FRONTIER_CELLS,
    min_hole: int = MIN_UNKNOWN_HOLE_CELLS,
) -> np.ndarray:
    """次に撮る位置の手がかりにする境目。frontiers から、小さな穴の縁と小さな塊を除く。

    VGGT の点の抜けでできた床の中の小さな穴（1〜数セル）も境目になり、そこを見に行こうとして
    小刻みに動くことがある（explore_lite の min_frontier_size と同じ考え）。実走の地図では、
    境目の塊の半分ほどが 3 セル未満だった。走れるかの判定（blocked）は変えない（穴は見ていない
    ままなので、そこへは入らない）。
    """
    import cv2

    work = state
    if min_hole > 1:
        unknown = (state == UNKNOWN).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(unknown, connectivity=4)
        h, w = state.shape
        tiny = np.zeros(n, bool)
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            edge = x == 0 or y == 0 or x + bw == w or y + bh == h
            tiny[k] = area < min_hole and not edge
        if tiny.any():
            work = state.copy()
            work[tiny[labels]] = FREE  # 境目を探すときだけ、穴を床とみなす
    front = frontiers(work) & (state == FREE)
    if min_cells <= 1 or not front.any():
        return front
    n, labels, stats, _ = cv2.connectedComponentsWithStats(front.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_cells
    return keep[labels]


def view_cells(grid: Grid, state: np.ndarray, pose: Pose, rays: int = 61) -> np.ndarray:
    """pose で首を振って撮ったときに見えるはずの未知のセル（格子の通し番号 ix * n + iy）。
    光線は障害物か格子の外で止める（障害物の陰は数えない）。
    """
    half = math.radians(HALF_VIEW_DEG)
    steps = int(VIEW_RANGE_M / (grid.res * 0.5))
    a = np.linspace(pose.yaw - half, pose.yaw + half, rays)[:, None]
    d = np.arange(1, steps + 1)[None, :] * grid.res * 0.5
    ix, iy = grid.cell(pose.x + d * np.cos(a), pose.y + d * np.sin(a))
    inside = grid.inside(ix, iy)
    s = np.where(inside, state[np.clip(ix, 0, grid.n - 1), np.clip(iy, 0, grid.n - 1)], OCCUPIED)
    # 光線ごとに、最初の障害物（か格子の外）より手前だけ
    ok = np.cumsum(s == OCCUPIED, axis=1) == 0
    hit = ok & (s == UNKNOWN)
    return np.unique(ix[hit] * grid.n + iy[hit])


def view_gain(grid: Grid, state: np.ndarray, pose: Pose, rays: int = 61) -> int:
    """pose で首を振って撮ったときに見えるはずの未知のセルの数（障害物の陰は数えない）。

    1 マスずつ Python で辿ると 1 回 5 ms ほどかかり、次の 1 歩を決めるのに 2 秒かかっていた。
    numpy でまとめて辿る（view_cells）。
    """
    return int(len(view_cells(grid, state, pose, rays)))


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
    # segments で決めたときの 1 歩の区間 [(ハンドル°, m), ...]。steer_deg・step_m は最初の区間
    segments: list[tuple[float, float]] = field(default_factory=list)


# 1 歩先までの道で曲がる角度の合計の上限。これより大きいのは、その場で輪を描く道
MAX_TURN_RAD = 1.25 * math.pi


def turning(route: tuple) -> float:
    """Dubins 経路で曲がる角度の合計（rad）。曲がりの区間は r で割った長さ＝角度。"""
    _, kind, segs = route
    return sum(seg for c, seg in zip(kind, segs, strict=True) if c != "S")


# 車の幅の半分と余裕。いま居る所のまわりだけは、障害物をこれだけ広げて見る（blocked_cells）
NEAR_RADIUS_M = 0.10


def blocked_cells(grid: Grid, state: np.ndarray, pose: Pose, unseen_ok: bool = False) -> np.ndarray:
    """車の中心が入れないセル：車の大きさ分を広げた障害物と、床が見えていない所
    （unseen_ok なら見えていない所は入れる）。"""
    occupied = state == OCCUPIED
    inflated = inflate(occupied, int(math.ceil(CAR_RADIUS_M / grid.res)))
    blocked = inflated if unseen_ok else inflated | (state != FREE)
    # いま居るセルの周りは撮ったときに足元が写らないので、見ていなくても走れることにする。
    # 障害物は車の幅の半分（NEAR_RADIUS_M）だけ広げて見る。車の大きさ分（17 cm）で見ると、
    # 家具のそばで止まったとき自分の居る所がふさがり、どこへも行けなくなる（3 回目の実走の
    # 最後は、左前 14 cm の障害物で周りが全部ふさがって unreachable だった）。
    # 少し先からは車の大きさ分で見るので、離れる向きの道だけが通る
    ix0, iy0 = grid.cell(pose.x, pose.y)
    r0 = int(math.ceil(CAR_RADIUS_M / grid.res))
    near = (slice(max(ix0 - r0, 0), ix0 + r0 + 1), slice(max(iy0 - r0, 0), iy0 + r0 + 1))
    blocked[near] = inflate(occupied, int(math.ceil(NEAR_RADIUS_M / grid.res)))[near]
    return blocked


def random_step(grid: Grid, pose: Pose, rng, tries: int = 40) -> tuple[float, float] | None:
    """床が見えている所に収まる、ハンドル角と長さがでたらめな 1 歩（ハンドル°、m）。

    次に撮る位置の選び方は、同じ所で行き詰まったり、同じ所を回り続けたりする（2 回目の実走で
    15 か所が半径 30 cm に固まった）。ルンバのように、ときどきでたらめに動いて抜け出す。
    ぶつからない弧のうち、着いた所から新しく見えそうなセルが多いものを選ぶ。
    """
    state = grid.state()
    blocked = blocked_cells(grid, state, pose)
    max_steer = math.degrees(math.atan(WHEELBASE_M / TURN_RADIUS_M))
    best, best_gain = None, -1
    for _ in range(tries):
        steer = float(rng.uniform(-max_steer, max_steer))
        run = float(rng.uniform(MIN_RANDOM_STEP_M, MAX_STEP_M))
        k = -math.tan(math.radians(steer)) / WHEELBASE_M  # 左が +
        if abs(k) < 1e-3:
            arc = sample_path(pose, "S", (run,), r=1.0)
        else:
            arc = sample_path(pose, "L" if k > 0 else "R", (run * abs(k),), r=1 / abs(k))
        if any(
            not grid.inside(*grid.cell(p.x, p.y)) or blocked[grid.cell(p.x, p.y)] for p in arc[1:]
        ):
            continue
        # Pi が実際に走る道（後ろの車軸を中心に曲がる自転車モデル・車の真ん中）でもぶつからないこと
        # （カメラの真下の弧だけでは、ハンドルいっぱいで終点が 22 cm ずれる。コードレビュー）
        clipped = clip_pieces(grid, blocked, pose, [(steer, run)])
        if not clipped or clipped[0][1] < run - 1e-6:
            continue
        gain = view_gain(grid, state, arc[-1], rays=21)
        if gain > best_gain:
            best, best_gain = (steer, run), gain
    return best


def next_best_view(
    grid: Grid,
    pose: Pose,
    candidates_max: int = 400,
    rng=None,
    segments: bool = False,
    max_step_m: float = MAX_STEP_M,
) -> Plan | None:
    """次に撮りに行く位置と、そこへの最初の 1 歩。行ける所に見るものが無ければ None。

    segments なら、1 歩を「曲がる → まっすぐ」の区間に分ける（turn_then_straight。
    合わせて max_step_m まで）。分けられない候補（Dubins 経路が曲がる・曲がる・曲がるの形など）は、
    今までの 1 本の弧にする（MAX_STEP_M まで）。
    """
    plans = next_best_views(grid, pose, 1, candidates_max, rng, segments, max_step_m)
    return plans[0] if plans else None


def next_best_views(
    grid: Grid,
    pose: Pose,
    k: int = 1,
    candidates_max: int = 400,
    rng=None,
    segments: bool = False,
    max_step_m: float = MAX_STEP_M,
    min_apart_m: float = 0.4,
) -> list[Plan]:
    """next_best_view の上位 k 個（利得 × exp(-道のり) の大きい順）。行き先どうしは
    min_apart_m 以上離す（同じ所の言い換えばかりにしない。学んだ選び方の候補にする）。

    k = 1 なら next_best_view と同じものを返す：候補のセルごとにいちばん良い向きを選び、
    良い順に、道がふさがっていない・1 歩を走れるものを探す（同じ値なら先の候補）。
    """
    rng = rng or np.random.default_rng(0)
    state = grid.state()
    blocked = blocked_cells(grid, state, pose)
    front = np.argwhere(big_frontiers(state))
    if not len(front):
        return []
    # 候補：境目の近く（1.2 m 以内）の、車が入れるセル
    free_ok = np.argwhere(~blocked)
    if not len(free_ok):
        return []
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
    scored = []
    for i in idx:
        x, y = float(cx[i]), float(cy[i])
        # 向きは、いちばん近い境目のほうを基本に、その左右 60° も試す（首を振れば見える）。
        # 決めた向きにぴったり着こうとすると、Dubins 経路がその場で一周する輪になり、
        # 少し進んでは撮り直すのを繰り返す（2 回目の実走の 15 か所）
        j = int(np.argmin((fx - x) ** 2 + (fy - y) ** 2))
        face = math.atan2(fy[j] - y, fx[j] - x)
        options = []
        for yaw in (face, face - math.pi / 3, face + math.pi / 3):
            target = Pose(x, y, yaw)
            gain = view_gain(grid, state, target)
            route = dubins(pose, target) if gain else None
            if route is None or turning(route) > MAX_TURN_RAD:
                continue
            options.append((gain * math.exp(-route[0] / DISTANCE_SCALE_M), target, gain, route))
        if options:
            utility, target, gain, route = max(options, key=lambda o: o[0])
            if utility > 0:
                scored.append((utility, target, gain, route))
    # 良い順（同じ値なら先の候補が先。sorted は安定）
    scored.sort(key=lambda o: -o[0])
    plans: list[Plan] = []
    for _, target, gain, route in scored:
        if any(math.hypot(p.target.x - target.x, p.target.y - target.y) < min_apart_m
               for p in plans):  # fmt: skip
            continue
        length, kind, segs = route
        path = sample_path(pose, kind, segs)
        if not path_is_clear(grid, blocked, path):
            continue
        pieces = turn_then_straight(grid, blocked, pose, target, max_step_m) if segments else []
        if pieces:
            steer, step = pieces[0]
        else:
            steer, step = first_step(grid, blocked, pose, path, length)
            if step <= 0:
                continue
            pieces = [(steer, step)] if segments else []
        plans.append(Plan(target, gain, length, path, steer, step, pieces))
        if len(plans) >= k:
            break
    return plans


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
    # Pi は後ろの車軸を中心に曲がるので、pure pursuit も後ろの車軸から、目指す点の後ろの車軸へ
    # 向けて引く（カメラから引くと、Pi が走る道が経路から外れ、ぶつかる所までに縮められて
    # 1 歩が短くなり、次に撮る位置をあまり選べなくなった：ベンチの実走の部屋で 16.1 → 11.4 m²）
    rear, goal = rear_axle(pose), rear_axle(aim)
    dx, dy = goal.x - rear.x, goal.y - rear.y
    ahead = math.cos(rear.yaw) * dx + math.sin(rear.yaw) * dy
    left = -math.sin(rear.yaw) * dx + math.cos(rear.yaw) * dy
    curvature = 2 * left / max(ahead * ahead + left * left, 1e-9)  # 左が +
    curvature = max(-1 / TURN_RADIUS_M, min(1 / TURN_RADIUS_M, curvature))
    run = min(MAX_STEP_M, length)
    steer = -math.degrees(math.atan(WHEELBASE_M * curvature))
    # ぶつからないかは、Pi が実際に走る道（後ろの車軸を中心に曲がる自転車モデルで、車の真ん中）で
    # 見る。カメラの真下から弧を引くと、ハンドルいっぱいで 40 cm 走ったとき終点が 22 cm ずれ、
    # 通した弧で車体が障害物に入っていた（コードレビュー）
    clipped = clip_pieces(grid, blocked, pose, [(steer, run)])
    if not clipped:
        return 0.0, 0.0
    return steer, clipped[0][1]


# --- 曲がってから、まっすぐ走る（Pi が segments を受けるとき） ---

# まっすぐの区間のハンドルはこれより小さくする（Pi の clear_cm は、ハンドルが小さいときだけ
# 通り道の帯がカメラの画角 54° に収まって測れる。1 回目の実走では 1 歩がみな大きな弧で、
# 241 回とも画角の外だった）
STRAIGHT_STEER_DEG = 8.0
# 曲がる区間がこれより短ければ省く（向きのずれは 0.02 / 0.26 rad = 4° ほど）
MIN_TURN_M = 0.02
# 1 歩がこれより短ければ走らない
MIN_STEP_M = 0.05
# Pi が 1 回に受ける区間の数の上限
MAX_SEGMENTS = 3
# 車の外形の真ん中（後ろの車軸から前へ）。ぶつかるかは、ここが広げた障害物に入るかで見る
CAR_CENTER_M = (CAR_FRONT_M + CAR_REAR_M) / 2


def turn_steer(direction: str) -> float:
    """最小の回転半径で曲がるハンドル角（右が +）。direction は 'L' か 'R'。"""
    deg = math.degrees(math.atan(WHEELBASE_M / TURN_RADIUS_M))
    return -deg if direction == "L" else deg


def drive_poses(
    start: Pose, pieces: list[tuple[float, float]], reverse: bool = False, step: float = 0.01
) -> list[tuple[float, Pose]]:
    """カメラの真下が start の車が、区間 [(ハンドル°, m), ...] を順に走ったときの
    (走った長さ, 後ろの車軸の姿勢) の列（step m ごと）。reverse なら後ろへ走る。
    """
    r = rear_axle(start)
    x, y, yaw = r.x, r.y, r.yaw
    out = [(0.0, Pose(x, y, yaw))]
    done = 0.0
    sign = -1.0 if reverse else 1.0
    for steer, length in pieces:
        n = max(1, int(math.ceil(length / step)))
        for _ in range(n):
            x, y, yaw = arc(x, y, yaw, steer, sign * length / n)
            done += length / n
            out.append((done, Pose(x, y, yaw)))
    return out


# 車の幅の半分（外形 約 26 x 19 cm）
CAR_HALF_WIDTH_M = 0.095
# 走った所を床にするとき、外形からこれだけ内側だけにする（位置のずれの分）
DRIVEN_SHRINK_M = 0.03
# 走った所の床の強さ（log-odds）。見た床（1 回 -0.85）より確か（車がそこに居た）
DRIVEN_FREE = -1.5


def mark_driven(
    grid: Grid, start: Pose, pieces: list[tuple[float, float]], reverse: bool = False
) -> None:
    """カメラの真下が start の車が区間を走った所（外形の下）を、床にする。

    カメラは前しか見ず、足元も写らないので、走ってきた道（とくに車の後ろ）は見ていないことが多い。
    けれど車が通れたのだから、車の高さより低い物は無い。そこを床にすると、後ろへ下がる・
    切り返す道を地図で確かめられる。障害物と見ているセルは消さない（動く猫かもしれない）。
    """
    if not pieces:
        return
    xs = np.arange(CAR_REAR_M + DRIVEN_SHRINK_M, CAR_FRONT_M - DRIVEN_SHRINK_M + 1e-9, grid.res / 2)
    half = CAR_HALF_WIDTH_M - DRIVEN_SHRINK_M
    ys = np.arange(-half, half + 1e-9, grid.res / 2)
    fx, fy = (a.ravel() for a in np.meshgrid(xs, ys))
    px, py = [], []
    for _, r in drive_poses(start, pieces, reverse, step=grid.res / 2):
        c, s = math.cos(r.yaw), math.sin(r.yaw)
        px.append(r.x + c * fx - s * fy)
        py.append(r.y + s * fx + c * fy)
    ix, iy = grid.cell(np.concatenate(px), np.concatenate(py))
    ok = grid.inside(ix, iy)
    cells = np.unique(np.stack([ix[ok], iy[ok]], 1), axis=0)
    lo = grid.logodds[cells[:, 0], cells[:, 1]]
    keep = lo <= 0.4  # 障害物と見ているセルはそのまま
    grid.logodds[cells[keep, 0], cells[keep, 1]] = np.minimum(lo[keep], DRIVEN_FREE)


def take(pieces: list[tuple[float, float]], total: float) -> list[tuple[float, float]]:
    """区間の頭から total m だけ（足りなければ最後の区間を伸ばす）。"""
    out: list[tuple[float, float]] = []
    left = total
    for i, (steer, length) in enumerate(pieces):
        if left <= 1e-9:
            break
        use = left if i == len(pieces) - 1 else min(length, left)
        out.append((steer, use))
        left -= use
    return out


def clip_pieces(
    grid: Grid, blocked: np.ndarray, start: Pose, pieces: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """区間を自転車モデルで走らせ、車の真ん中が blocked に入る手前までに縮める。
    MIN_STEP_M も進めなければ空。
    """
    clear = 0.0
    for done, r in drive_poses(start, pieces)[1:]:
        # 車の真ん中だけでなく、カメラの真下（前の端の 5 cm 手前）も見る。出発点のまわりは
        # 障害物を 10 cm しか広げないので、真ん中だけでは前の端が 2〜3 cm 入った（コードレビュー）
        c, s = math.cos(r.yaw), math.sin(r.yaw)
        ok = True
        for ahead in (CAR_CENTER_M, CAM_X_M):
            ix, iy = grid.cell(r.x + ahead * c, r.y + ahead * s)
            if not grid.inside(ix, iy) or blocked[ix, iy]:
                ok = False
                break
        if not ok:
            break
        clear = done
    if clear < MIN_STEP_M:
        return []
    out = take(pieces, min(clear, sum(m for _, m in pieces)))
    return [(st, m) for st, m in out if m >= 0.005]


def turn_then_straight(
    grid: Grid, blocked: np.ndarray, pose: Pose, target: Pose, max_m: float = MAX_STEP_M
) -> list[tuple[float, float]]:
    """target へ向かう 1 歩を、最小の回転半径の曲がり（要る分だけ）→ まっすぐ、の区間にする。

    後ろの車軸どうしの Dubins 経路（最小半径 TURN_RADIUS_M）の、最初の曲がりとその後の直線を
    そのまま使う。曲がりの長さは、直線が次の位置のほうを向くまでの分だけになる。
    最後の曲がりは使わない（次のステーションで撮って決め直す）。直線の無い形（RLR・LRL）は
    分けられないので空を返す。区間は自転車モデルで走らせて blocked に当て、ぶつかる手前までにする。
    MAX_STEP_M を超えて走るのは、超えた分がまっすぐの区間のときだけ（Pi の clear_cm は、
    ハンドルが小さいときだけ通り道を測れる）。
    """
    route = dubins(rear_axle(pose), rear_axle(target))
    if route is None or route[1][1] != "S":
        return []
    _, kind, segs = route
    pieces = []
    if segs[0] * TURN_RADIUS_M >= MIN_TURN_M:
        pieces.append((turn_steer(kind[0]), segs[0] * TURN_RADIUS_M))
    if segs[1] * TURN_RADIUS_M > 0:
        pieces.append((0.0, segs[1] * TURN_RADIUS_M))
    total = sum(m for _, m in pieces)
    if total <= 0:
        return []
    if max_m > MAX_STEP_M and pieces[-1][0] != 0.0:
        max_m = MAX_STEP_M
    if max_m > MAX_STEP_M and len(pieces) > 1 and pieces[0][1] > MAX_STEP_M:
        max_m = MAX_STEP_M
    pieces = take(pieces, min(max_m, total))
    return clip_pieces(grid, blocked, pose, pieces)


# --- 来た道を戻る（Pi が reverse を受けられるとき） ---

# 1 回に戻る長さの上限（Pi も 30 cm で止める）
REVERSE_MAX_M = 0.30
# VGGT の位置と、自転車モデルで走らせた位置のずれがこれより大きければ、位置からは測らない
FIT_TOL_M = 0.08


def fit_driven(
    start: Pose, end: Pose, pieces: list[tuple[float, float]], reverse: bool = False,
    s_max: float = 1.2,
) -> tuple[float, float]:  # fmt: skip
    """start（カメラの真下）から区間を走って end に着いたとして、走った長さ（m）と、
    そのときの end とのずれ（位置の m + 向きの rad × 0.1）。

    途中で止まったなら区間の途中、頼んだより長く走ったなら最後の区間の先になるよう、
    最後の区間を s_max まで伸ばして探す（1 回目の実走では、Pi の距離の見積もりが本当の
    0.6 倍だったので、頼んだ長さの 1.7 倍ほど走っていた）。
    """
    best = (0.0, math.inf)
    for done, r in drive_poses(start, take(pieces, s_max), reverse):
        c = camera_of(r)
        err = math.hypot(c.x - end.x, c.y - end.y)
        err += 0.1 * abs(math.remainder(c.yaw - end.yaw, 2 * math.pi))
        if err < best[1]:
            best = (done, err)
    return best


def consume(trail: list[tuple[float, float]], m: float) -> list[tuple[float, float]]:
    """来た道（走った順の区間）の終わりから m だけ除く（後ろへ戻った分）。"""
    out = list(trail)
    while out and m > 1e-9:
        steer, length = out.pop()
        if length > m:
            out.append((steer, length - m))
        m -= length
    return out


def turning_reverse(
    grid: Grid, pose: Pose, length: float = 0.30, unseen_ok_m: float = 0.15
) -> list[tuple[float, float]]:
    """切り返し：ハンドルをいっぱいに切って後ろへ下がる 1 本の弧（ハンドル°、m）。走れなければ空。

    来た道をそのまま戻ると、同じ位置・同じ向きに戻って同じ 1 歩を選び直す（3 回目の実走では
    15 か所・4.7 m 走って、半径 30 cm から出られなかった）。曲がりながら下がると向きが
    40〜60° 変わり、次の 1 歩で別のほうへ進める。

    カメラは前（パン ±35°）しか見ないので、車の後ろはいつも見ていない（3 回目の実走の地図でも
    後ろ側が空いていた）。見えている床だけを通ると、後ろへはほとんど下がれない。そこで
    頭の unseen_ok_m だけは、障害物と分かっていない（見ていない）所も通ってよいことにする
    （Pi は遅く下がり、画像が動かなければ止まる）。その先は見えている床だけ。
    左右のうち、下がった先から新しく見えそうなセルが多いほうを選ぶ。
    """
    state = grid.state()
    blocked = blocked_cells(grid, state, pose)
    hard = blocked_cells(grid, state, pose, unseen_ok=True)
    max_steer = math.degrees(math.atan(WHEELBASE_M / TURN_RADIUS_M))
    best, best_gain = [], -1
    for steer in (max_steer, -max_steer):
        pieces = [(steer, length)]
        ok = True
        for done, rear in drive_poses(pose, pieces, reverse=True)[1:]:
            cx = rear.x + CAR_CENTER_M * math.cos(rear.yaw)
            c = grid.cell(cx, rear.y + CAR_CENTER_M * math.sin(rear.yaw))
            if not grid.inside(*c) or (hard if done <= unseen_ok_m else blocked)[c]:
                ok = False
                break
        if not ok:
            continue
        end = camera_of(drive_poses(pose, pieces, reverse=True)[-1][1])
        gain = view_gain(grid, state, end, rays=21)
        if gain > best_gain:
            best, best_gain = pieces, gain
    return best


# --- 向きを変える（切り返し・3 点ターン）を、次の 1 歩の候補にする ---

# 見たい所がこれより後ろ（車の向きから）なら、前へ輪を描くより先に向きを変える
BEHIND_DEG = 100.0
# 向きを変えたら、見たい所の向きがこれだけは前に寄ること
TURN_GAIN_DEG = 40.0


@dataclass
class Turn:
    """向きを変える 1 歩。moves は [(ハンドル°, m, 後ろへか), ...]（走る順）。"""

    moves: list[tuple[float, float, bool]]
    end: Pose  # 着いた所（カメラの真下）
    gain: int


def drive_moves(start: Pose, moves: list[tuple[float, float, bool]], step: float = 0.01):
    """前へ・後ろへの混じった区間を走ったときの (走った長さ, 後ろの車軸の姿勢, 区間の番号,
    その区間で走った長さ) の列。"""
    r = rear_axle(start)
    x, y, yaw = r.x, r.y, r.yaw
    out = [(0.0, Pose(x, y, yaw), 0, 0.0)]
    done = 0.0
    for k, (steer, length, back) in enumerate(moves):
        n = max(1, int(math.ceil(length / step)))
        for j in range(n):
            x, y, yaw = arc(x, y, yaw, steer, (-1 if back else 1) * length / n)
            done += length / n
            out.append((done, Pose(x, y, yaw), k, (j + 1) * length / n))
    return out


def moves_are_clear(
    grid: Grid, state: np.ndarray, pose: Pose, moves, unseen_ok_m: float = 0.15
) -> bool:
    """区間を走って、車の真ん中が blocked に入らないか。後ろへの区間の頭 unseen_ok_m だけは、
    見ていない所も通ってよい（turning_reverse と同じ。Pi は遅く下がり、動かなければ止まる）。"""
    blocked = blocked_cells(grid, state, pose)
    hard = blocked_cells(grid, state, pose, unseen_ok=True)
    for _, rear, k, part in drive_moves(pose, moves)[1:]:
        c = grid.cell(
            rear.x + CAR_CENTER_M * math.cos(rear.yaw), rear.y + CAR_CENTER_M * math.sin(rear.yaw)
        )
        back = moves[k][2]
        if not grid.inside(*c) or (hard if back and part <= unseen_ok_m else blocked)[c]:
            return False
    return True


def plan_turn(
    grid: Grid,
    pose: Pose,
    toward: Pose | None = None,
    unseen_ok_m: float = 0.0,
) -> Turn | None:
    """向きを変える 1 歩：切り返し（いっぱいに切って 30 cm か 15 cm 下がる）。

    前輪で曲がる車は、真後ろの見たい所へ前進だけで行くと、回転半径 26 cm の輪を描き、
    数歩かけて回り込む（輪の途中で家具に当たる道は使えず、3 回目の実走は unreachable と
    戻るを繰り返した）。toward（見たい所）があれば、着いた向きがそこにいちばん近いもの、
    無ければ着いた所から新しく見えそうなセルがいちばん多いもの。見えるセルが無ければ None。
    後ろへの区間は、頭の unseen_ok_m だけ見ていない所に入ってよい（既定は床と分かった所だけ。
    走ってきた所は mark_driven で床になっている。行き場が無いときの turning_reverse は 15 cm）。
    3 点ターン（前へ曲がる → 逆に切って後ろへ、を 1 回の返事にする）も試したが、
    シミュレーターの実走の部屋で 5 分の広さが 17 % 減ったので入れていない。
    """
    state = grid.state()
    max_steer = math.degrees(math.atan(WHEELBASE_M / TURN_RADIUS_M))
    options: list[list[tuple[float, float, bool]]] = []
    for s in (max_steer, -max_steer):
        options += [[(s, 0.30, True)], [(s, 0.15, True)]]
    best, best_key = None, None
    for moves in options:
        if not moves_are_clear(grid, state, pose, moves, unseen_ok_m):
            continue
        end = camera_of(drive_moves(pose, moves)[-1][1])
        gain = view_gain(grid, state, end, rays=21)
        if gain <= 0:
            continue
        if toward is not None:
            off = abs(
                math.remainder(
                    math.atan2(toward.y - end.y, toward.x - end.x) - end.yaw, 2 * math.pi
                )
            )
            key = (-off, gain)
        else:
            key = (gain, 0.0)
        if best_key is None or key > best_key:
            best, best_key = Turn(moves, end, gain), key
    return best


def bearing(pose: Pose, target: Pose) -> float:
    """pose の向きから見た target の向き（rad、-π..π）。"""
    return math.remainder(math.atan2(target.y - pose.y, target.x - pose.x) - pose.yaw, 2 * math.pi)


def retrace(
    trail: list[tuple[float, float]], budget: float = REVERSE_MAX_M, max_pieces: int = MAX_SEGMENTS
) -> list[tuple[float, float]]:
    """来た道の終わりから budget m を、戻る順の区間にする。

    同じハンドルのまま後ろへ走ると、自転車モデルでは同じ弧をそのまま戻る。なので、走った区間を
    逆の順に、同じハンドルで後ろへ走れば、さっき走った（ぶつからなかった）床の上だけを通る。
    """
    out: list[tuple[float, float]] = []
    left = budget
    for steer, length in reversed(trail):
        if left <= 1e-9:
            break
        use = min(length, left)
        if out and abs(out[-1][0] - steer) < 0.05:
            out[-1] = (steer, out[-1][1] + use)
        elif len(out) == max_pieces:
            break
        else:
            out.append((steer, use))
        left -= use
    return [(st, m) for st, m in out if m >= 0.005]
