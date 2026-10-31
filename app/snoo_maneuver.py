"""前へ・後ろへを切り替える 1 歩（切り返し・3 点ターン・隙間へ合わせる寄せ）を決める。

車は前輪で曲がる（ハンドル ±28.3°、最小の回転半径 26 cm）ので、その場で向きを変えられない。
椅子や机の脚の間では、前へ行く 1 歩だけでは行ける所が無く（unreachable）、下がる 1 歩と前へ行く
1 歩を 1 か所（約 25 秒）ずつ繰り返していた（2026-10-31 の実走の 004〜010）。そこで 1 歩を、
向きの違う区切り（最大 5 つ）の列にして、サーバーがまとめて決める。Pi はそのとおり走るだけ
（前へ走る区切りでは、今までどおり超音波・clear_cm で止まる）。

決め方は 2 つ（MapParams.maneuver）
- hybrid：Hybrid A*。後ろの車軸の (x, y, 向き) の格子の上で、ハンドル {-28.3, 0, +28.3}°・
  前と後ろの 10 cm の弧をつなぐ。点数は下のコストと同じ。見込み（h）は行き先までの道のり
- mppi：MPPI 風。区切りの列（ハンドルと ±距離）をたくさんばらまいて走らせ、コストで重みを付けて
  名目の列を寄せ直す（3 回）。名目の列は、今の決め方（next_best_view）の 1 歩・Hybrid A* の答え・
  切り返しと 3 点ターンのひな形。CPU 1 本で 1 か所 1 秒以内

ぶつかるか：車の外形（後ろの車軸から -5〜+21 cm、幅 19 cm）に GAP_MARGIN_M を足した長方形の
縁の点を、占有の格子に当てる（2.5 cm ごと）。障害物まで車幅の半分 + 2 cm（Pi の clear_cm の帯の
余裕と同じ）まで寄ってよい。出発した時に外形の下にあった障害物は「触れている物」で、押し込む向きに
だけぶつかるとする。

コスト（m で数える。小さいほど良い）
- ぶつかる：使わない（無限大）
- 走る長さ（後ろは W_REV 倍重い）・前後の切り替えごと W_SWITCH・区切りごと W_SEG
- 後ろへの区切りの先の縁が、確信度の低い（見ていない・1 回しか見ていない）セルに入る：
  W_UNK_REV（重い）。合わせて UNSEEN_REV_MAX_M を超えたら使わない（後ろにはセンサーが無い）
- 前への区切りの先の縁が見ていないセルに入る：W_UNK_FWD（軽い。Pi の超音波と clear_cm が見る）
- 前への区切りで、まっすぐに近い（CLEAR_STEER_DEG 以下）のに前の端から CLEAR_STOP_M に障害物が
  入る：Pi が clear で止まるので使わない
- 行き先までの道のり（境目と、確信度の低い縁の手前）の、着いた所の値 × W_GOAL
- 狭い隙間（両側の余裕が NARROW_M 未満）を通り抜けて行き先に近づく道は W_GAP だけ得
- 着いた所（カメラ）から首を振って見える、見ていないセルと確信度の低いセル × W_INFO
- 区切りの終わりから見える目印（手ざわりの高いセル）が LANDMARK_MIN 未満なら W_LOST
  （白い壁しか見えない所では VGGT・RoMa が位置を外す）
- 直近のステーションのそばに戻る：W_REVISIT（同じ所に固まらない）
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from app.roommap import (
    CAM_X_M,
    CAR_CENTER_M,
    CAR_FRONT_M,
    CAR_HALF_WIDTH_M,
    CAR_REAR_M,
    FREE,
    HALF_VIEW_DEG,
    MAX_STEP_M,
    OCCUPIED,
    REVERSE_MAX_M,
    TURN_RADIUS_M,
    UNKNOWN,
    VIEW_RANGE_M,
    WHEELBASE_M,
    Grid,
    Pose,
    big_frontiers,
    camera_of,
    curvature,
    drive_moves,
    rear_axle,
)
from app.snoo_belief import CONF_OK, LANDMARK_MIN, confidence_of, unsure_edges, visible_landmarks

MAX_STEER_DEG = math.degrees(math.atan(WHEELBASE_M / TURN_RADIUS_M))  # 28.3°
# 1 歩の区切りの数の上限（Pi の MAX_SEGMENTS と同じ）
GEAR_MAX_SEGMENTS = 5
# 1 つの区切りの長さの上限
SEG_MAX_M = 0.30
# ハンドルを大きく切った前への区切りは、Pi の clear_cm が画角の外（clear_skip）になり、
# 20 cm（CLEAR_SKIP_MAX_CM）で止まるので、それまでにする
SKIP_STEER_DEG = 15.0
SKIP_SEG_MAX_M = 0.20
# 区切りがこれより短ければ走らない
SEG_MIN_M = 0.02
MIN_TOTAL_M = 0.05
# 車の外形に足す余裕（横・前後）。障害物まで車幅の半分 + これだけ寄ってよい
GAP_MARGIN_M = 0.02
# 両側の余裕（車の横から障害物まで）がこれ未満の所を、狭い隙間とする
NARROW_M = 0.06
# Pi が前へ走っていて止まる、画像の通り道の空き（explore_clear_stop_cm）と、見るハンドルの上限
CLEAR_STOP_M = 0.25
CLEAR_STEER_DEG = SKIP_STEER_DEG
# Pi に渡す clear_cm の下限：地図の見込みからの余裕と、上限（cm）
CLEAR_TOL_CM = 15.0
CLEAR_CAP_CM = 40.0
# 後ろへ確信度の低い所を通ってよい合計（セルの大きさの分だけ。後ろにはセンサーが無いので、
# 確かな床だけを通す。10 cm にしていたら、地図から作った部屋で、見ていない後ろの物に当たった）
UNSEEN_REV_MAX_M = 0.03
# 前へ見ていない所を通ってよい合計
UNSEEN_FWD_MAX_M = 0.10
# 後ろへ通ってよい、床の確信度の下限（1 回見ただけでは足りない。走った所は 1）
CONF_REVERSE = 0.6
# 1 区切りを何回に分けて確かめるか
SUB = 12

# コストの重み
W_FWD = 0.3
W_REV = 0.8
W_SWITCH = 0.25
W_SEG = 0.05
W_UNK_REV = 6.0
W_UNK_FWD = 1.0
W_GOAL = 2.0
W_GAP = 0.4
W_MAYBE = 0.5
W_INFO = 0.005
W_LOST = 1.0
W_REVISIT = 0.3
REVISIT_M = 0.25
# 動けていないときの重みの倍率（切り替えを安く・同じ所へ戻るのを重く）
WIDE_SWITCH = 0.3
WIDE_REVISIT = 3.0
# 確信度の低い縁は、境目より少し後回し（道のりにこれだけ足す）
LOWCONF_OFFSET_M = 1.0
# next_best_view の行き先があるとき、ほかの境目に足す道のり
GOAL_FIRST_M = 0.6
# 撮った所からこれより近い境目は、行き先にしない（足もと・車の横は写らない）
SEEN_NEAR_M = 0.40
# 行き先に行けない（つながっていない）ときの道のりの見込み
D_FAR_M = 5.0


@dataclass(frozen=True)
class Limits:
    """Pi が走れる 1 歩の形（meta の supports で決まる）。"""

    gears: bool = True  # 区切りごとに前・後ろを変えてよい（"gear_segments"）
    reverse: bool = True  # 後ろへ走ってよい（"reverse"）
    max_segments: int = GEAR_MAX_SEGMENTS  # "segments" が無ければ 1
    max_fwd_m: float = MAX_STEP_M
    # 動けていない（同じ所に居続けた）：区切りを長く・切り替えを安く・同じ所へ戻るのを重く
    wide: bool = False


@dataclass
class Maneuver:
    moves: list[tuple[float, float, bool]]  # (ハンドル°, m, 後ろへか)
    cost: float
    ends: list[Pose]  # 区切りの終わりのカメラの真下の姿勢
    min_clear_cm: list[
        float | None
    ]  # 前への区切りで Pi が止まる clear_cm の下限（cm。min_clear_cm）
    gap: bool  # 狭い隙間を通る
    kind: str  # mppi・hybrid・nominal
    info: dict = field(default_factory=dict)
    path: list[Pose] = field(default_factory=list)  # カメラの真下が通る道（絵に描く）
    gain: float = 0.0  # 着いた所から見えるもの（Scene.info）

    @property
    def end(self) -> Pose:
        return self.ends[-1]

    @property
    def target(self) -> Pose:
        return self.ends[-1]

    @property
    def length_m(self) -> float:
        return sum(m for _, m, _ in self.moves)


def _footprint(margin: float = GAP_MARGIN_M, step: float = 0.025):
    """外形の縁の点（後ろの車軸の座標）と、前の縁・後ろの縁の点。"""
    x0, x1 = CAR_REAR_M - margin, CAR_FRONT_M + margin
    w = CAR_HALF_WIDTH_M + margin
    nx = int(math.ceil((x1 - x0) / step)) + 1
    ny = int(math.ceil(2 * w / step)) + 1
    xs, ys = np.linspace(x0, x1, nx), np.linspace(-w, w, ny)
    edge = np.concatenate(
        [
            np.stack([xs, np.full(nx, -w)], 1),
            np.stack([xs, np.full(nx, w)], 1),
            np.stack([np.full(ny, x0), ys], 1),
            np.stack([np.full(ny, x1), ys], 1),
            np.stack([xs, np.zeros(nx)], 1),  # 真ん中の線（細い物が縁の点の間に入らないよう）
        ]
    )
    front = np.stack([np.full(ny, x1), ys], 1)
    rear = np.stack([np.full(ny, x0), ys], 1)
    return edge, front, rear


EDGE_PTS, FRONT_PTS, REAR_PTS = _footprint()


def _body(margin: float, step: float = 0.02) -> np.ndarray:
    """外形（と余裕）の中を埋める点（後ろの車軸の座標）。"""
    xs = np.arange(CAR_REAR_M - margin, CAR_FRONT_M + margin + 1e-9, step)
    w = CAR_HALF_WIDTH_M + margin
    ys = np.arange(-w, w + 1e-9, step)
    return np.stack([a.ravel() for a in np.meshgrid(xs, ys)], 1)


# 出発した時に外形（と余裕）の下にあったセルを見るための点
BODY_PTS = _body(GAP_MARGIN_M)
# 前の端の帯（clear_cm の見込み）：前の端から先、左右と真ん中
AHEAD_S = np.array([0.05, 0.10, 0.15, 0.20])
AHEAD_Y = np.array([-CAR_HALF_WIDTH_M, 0.0, CAR_HALF_WIDTH_M])
# 横の余裕を測る、車の真ん中の線の点（後ろの車軸から）
SIDE_X = np.array([CAR_REAR_M + CAR_HALF_WIDTH_M, CAR_CENTER_M, CAR_FRONT_M - CAR_HALF_WIDTH_M])


def _arc_np(x, y, yaw, k, ds):
    """roommap.arc を配列で。k は曲率（左が +）、ds は符号付きの長さ。"""
    small = np.abs(k) < 1e-9
    ks = np.where(small, 1.0, k)
    nyaw = yaw + k * ds
    nx = np.where(small, x + ds * np.cos(yaw), x + (np.sin(nyaw) - np.sin(yaw)) / ks)
    ny = np.where(small, y + ds * np.sin(yaw), y - (np.cos(nyaw) - np.cos(yaw)) / ks)
    return nx, ny, nyaw


class Scene:
    """1 回決めるあいだ使う、格子から作った層。"""

    def __init__(
        self,
        grid: Grid,
        pose: Pose,
        targets_conf: bool = False,
        recent: list[Pose] | None = None,
        wide: bool = False,
        goal: Pose | None = None,
    ):
        import cv2

        t0 = time.perf_counter()
        self.wide = wide
        self.edge_pts = EDGE_PTS
        self.margin = GAP_MARGIN_M
        self.grid = grid
        self.pose = pose
        self.start = rear_axle(pose)
        st = grid.state()
        self.state = st
        self.conf = confidence_of(grid, st)
        occ = st == OCCUPIED
        # 出発した時に外形（と余裕）の下にあったセル。見ていなくてよい。そこの障害物は
        # 「触れている物」で、進む向きの先の縁（前へなら前の縁、後ろへなら後ろの縁）が入るときだけ
        # ぶつかるとする
        # （離れる向きには動ける。押し込む向きには動かない：白い壁に角が触れたまま前へ出し続けた）
        self.start_cells = self._cells_of(BODY_PTS)
        self.occ = occ & ~self.start_cells
        self.touch = occ & self.start_cells
        self.unknown = (st == UNKNOWN) & ~self.start_cells
        # 後ろへ通ってよくない所：床と分かっていない・確信度が低い
        self.not_sure = ((st != FREE) | (self.conf < CONF_REVERSE)) & ~self.start_cells
        self.maybe = (st == FREE) & (self.conf > 0.0) & (self.conf < CONF_OK)
        # 障害物のセルの中心までの距離（m）。障害物の面はセルの半分だけ手前にあるとみる
        src = np.where(self.occ, 0, 255).astype(np.uint8)
        self.edt = cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(np.float32) * grid.res
        self.edt -= grid.res / 2
        tex = getattr(grid, "texture", None)
        self.landmarks = tex.landmarks(grid) if tex is not None else np.zeros((0, 2))
        self.use_landmarks = len(self.landmarks) >= LANDMARK_MIN
        self.recent = [p for p in (recent or [])]
        self.goal = goal
        self.targets_conf = targets_conf
        self._goal_field()
        self.build_s = time.perf_counter() - t0

    def _cells_of(self, pts) -> np.ndarray:
        g = self.grid
        bx, by = self._world(pts, self.start.x, self.start.y, self.start.yaw)
        ix, iy = g.cell(bx, by)
        ok = g.inside(ix, iy)
        out = np.zeros((g.n, g.n), bool)
        out[ix[ok], iy[ok]] = True
        return out

    def _world(self, pts, x, y, yaw):
        """後ろの車軸の座標の点 (M, 2) を、姿勢（配列 (...,)）で世界へ。(..., M) を 2 つ。"""
        x, y, yaw = (np.asarray(v, float)[..., None] for v in (x, y, yaw))
        c, s = np.cos(yaw), np.sin(yaw)
        return x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]

    def _lookup(self, layer, wx, wy, outside=True):
        ix, iy = self.grid.cell(wx, wy)
        ok = self.grid.inside(ix, iy)
        n = self.grid.n
        v = layer[np.clip(ix, 0, n - 1), np.clip(iy, 0, n - 1)]
        return np.where(ok, v, outside)

    # --- 行き先までの道のり ---

    def _goal_field(self) -> None:
        """境目（と確信度の低い縁の手前）までの道のり（m）の場。車の真ん中が通れるセルだけを
        たどる（横の余裕 GAP_MARGIN_M。next_best_view の 17 cm の広げ方より狭い隙間も通る）。"""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        g, st = self.grid, self.state
        front = big_frontiers(st)
        # 撮った所のすぐそばの境目は行き先にしない。カメラの足もと（画面の下の端より手前）と車の横は
        # いつも写らないので、そこに境目が残り、「もう着いている」ことになって小さく回り続けた
        for p in self.recent + [self.pose]:
            ix, iy = g.cell(p.x, p.y)
            r = int(math.ceil(SEEN_NEAR_M / g.res))
            xs, ys = np.mgrid[-r : r + 1, -r : r + 1]
            disk = xs**2 + ys**2 <= r * r
            cx, cy = ix + xs[disk], iy + ys[disk]
            ok = (cx >= 0) & (cx < g.n) & (cy >= 0) & (cy < g.n)
            front[cx[ok], cy[ok]] = False
        targets = front.copy()
        offset = np.zeros(st.shape, np.float32)
        if self.goal is not None:
            # 今の決め方（next_best_view）が選んだ行き先（見えるものがいちばん多い所）を先にし、
            # ほかの境目は GOAL_FIRST_M だけ後回しにする。小さな見えない窪み（低い物の陰）へ
            # 行き続けない
            offset[:] = GOAL_FIRST_M
            gx, gy = g.cell(self.goal.x, self.goal.y)
            if g.inside(gx, gy):
                sl = (slice(max(gx - 1, 0), gx + 2), slice(max(gy - 1, 0), gy + 2))
                targets[sl] |= st[sl] == FREE
                offset[sl] = 0.0
        self.lowconf_targets = 0
        if self.targets_conf:
            low = self.lowconf_edge()
            self.lowconf_targets = int(low.sum())
            extra = low & ~targets
            targets |= extra
            offset[extra] = LOWCONF_OFFSET_M + (GOAL_FIRST_M if self.goal is not None else 0.0)
        self.targets = targets
        ok = (st == FREE) | self.start_cells
        occ_all = st == OCCUPIED
        passable = ok & ~self.occ & (self.edt >= CAR_HALF_WIDTH_M + self.margin)
        passable |= self.start_cells & ~occ_all
        nodes = passable | targets
        known = np.argwhere(nodes)
        self.D = np.full(st.shape, np.inf, np.float32)
        self.connected = False
        if not len(known) or not targets.any():
            self.D_start = np.inf
            return
        x0, y0 = known.min(0)
        x1, y1 = known.max(0) + 1
        sub = nodes[x0:x1, y0:y1]
        pas = passable[x0:x1, y0:y1]
        tgt = targets[x0:x1, y0:y1]
        h, w = sub.shape
        idx = np.full(sub.shape, -1, np.int64)
        cells = np.argwhere(sub)
        idx[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
        rows, cols, wts = [], [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            a = sub[max(0, -dx) : h - max(0, dx), max(0, -dy) : w - max(0, dy)]
            b = sub[max(0, dx) : h - max(0, -dx) + (h if dx >= 0 else 0), :][: a.shape[0]]
            b = sub[max(0, dx) : max(0, dx) + a.shape[0], max(0, dy) : max(0, dy) + a.shape[1]]
            pa = pas[max(0, -dx) : max(0, -dx) + a.shape[0], max(0, -dy) : max(0, -dy) + a.shape[1]]
            pb = pas[max(0, dx) : max(0, dx) + a.shape[0], max(0, dy) : max(0, dy) + a.shape[1]]
            both = a & b & (pa | pb)  # 行き先どうし（通れないもの）はつながない
            ii = np.argwhere(both)
            ia = idx[ii[:, 0] + max(0, -dx), ii[:, 1] + max(0, -dy)]
            ib = idx[ii[:, 0] + max(0, dx), ii[:, 1] + max(0, dy)]
            d = g.res * math.hypot(dx, dy)
            rows += [ia, ib]
            cols += [ib, ia]
            wts += [np.full(len(ia), d), np.full(len(ia), d)]
        n = len(cells)
        # 行き先へは、仮の始点（番号 n）から offset の重みでつなぐ
        tc = np.argwhere(tgt)
        ti = idx[tc[:, 0], tc[:, 1]]
        rows.append(np.full(len(ti), n))
        cols.append(ti)
        wts.append(offset[tc[:, 0] + x0, tc[:, 1] + y0] + 1e-6)
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        wv = np.concatenate(wts)
        m = coo_matrix((wv, (r, c)), shape=(n + 1, n + 1)).tocsr()
        dist = dijkstra(m, directed=True, indices=n)
        # 向きの無い辺は両向きに入れたので、仮の始点からの距離 = 行き先までの道のり
        self.D[cells[:, 0] + x0, cells[:, 1] + y0] = dist[:n]
        self.D_start = float(self.D_at(self.start.x, self.start.y, self.start.yaw))
        self.connected = math.isfinite(self.D_start)
        if not self.connected:
            # 今の所が行き先とつながっていない：まっすぐの距離で寄せる（離れた所へは行けない）
            import cv2

            src = np.where(targets, 0, 255).astype(np.uint8)
            e = cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(np.float32) * g.res
            self.D = np.where(np.isfinite(self.D), self.D, e + 1.0)
            self.D_start = float(self.D_at(self.start.x, self.start.y, self.start.yaw))

    def lowconf_edge(self) -> np.ndarray:
        """確信度の低い縁（snoo_belief.unsure_edges）の 0.35 m 以内の、通れる床のセル
        （近づいて正面から撮って確かめる行き先）。"""
        import cv2

        low = unsure_edges(self.grid, self.state, self.conf)
        if not low.any():
            return low
        r = int(round(0.35 / self.grid.res))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ring = cv2.dilate(low.astype(np.uint8), disk) > 0
        return ring & (self.state == FREE) & (self.edt >= CAR_HALF_WIDTH_M + self.margin)

    def D_at(self, rx, ry, ryaw):
        """後ろの車軸の姿勢での、車の真ん中の道のり。"""
        cx = np.asarray(rx) + CAR_CENTER_M * np.cos(ryaw)
        cy = np.asarray(ry) + CAR_CENTER_M * np.sin(ryaw)
        return self._lookup(self.D, cx, cy, outside=np.inf)

    # --- 姿勢ごとの確かめ ---

    def check(self, x, y, yaw, back, steer_deg):
        """後ろの車軸の姿勢（配列）ごとに (ぶつかる, 先の縁が見ていない/確かでない, clear で止まる,
        横の余裕 m, たぶん空きの上か)。"""
        ex, ey = self._world(self.edge_pts, x, y, yaw)
        hit = self._lookup(self.occ, ex, ey, True).any(-1)
        fx, fy = self._world(FRONT_PTS, x, y, yaw)
        rx, ry = self._world(REAR_PTS, x, y, yaw)
        if self.touch.any():
            touch_f = self._lookup(self.touch, fx, fy, False).any(-1)
            touch_r = self._lookup(self.touch, rx, ry, False).any(-1)
            hit = hit | np.where(back, touch_r, touch_f)
        unk_f = self._lookup(self.unknown, fx, fy, True).any(-1)
        unk_r = self._lookup(self.not_sure, rx, ry, True).any(-1)
        lead = np.where(back, unk_r, unk_f)
        maybe = self._lookup(self.maybe, fx, fy, False).mean(-1) > 0.5
        # 前の端から先の帯（Pi の clear_cm の見込み）
        k = -np.tan(np.radians(steer_deg)) / WHEELBASE_M
        blocked = np.zeros(np.shape(x), bool)
        straightish = (~np.asarray(back)) & (np.abs(steer_deg) <= CLEAR_STEER_DEG)
        if straightish.any():
            for s in AHEAD_S:
                ax, ay, ayaw = _arc_np(np.asarray(x), np.asarray(y), np.asarray(yaw), k, s)
                pts = np.stack([np.full(len(AHEAD_Y), CAR_FRONT_M), AHEAD_Y], 1)
                px, py = self._world(pts, ax, ay, ayaw)
                blocked |= self._lookup(self.occ, px, py, True).any(-1)
            blocked &= straightish
        sx, sy = self._world(np.stack([SIDE_X, np.zeros(3)], 1), x, y, yaw)
        side = self._lookup(self.edt, sx, sy, 0.0).min(-1) - CAR_HALF_WIDTH_M
        return hit, lead, blocked, side, maybe

    # --- 点数 ---

    def lost_penalty(self, cam_x, cam_y, cam_yaw):
        if not self.use_landmarks:
            return np.zeros(np.shape(cam_x))
        n = visible_landmarks(self.landmarks, cam_x, cam_y, cam_yaw, HALF_VIEW_DEG, VIEW_RANGE_M)
        return W_LOST * np.clip(1.0 - n / LANDMARK_MIN, 0.0, 1.0)

    def revisit(self, cam_x, cam_y):
        if not self.recent:
            return np.zeros(np.shape(cam_x))
        px = np.array([p.x for p in self.recent])
        py = np.array([p.y for p in self.recent])
        d = np.hypot(np.asarray(cam_x)[..., None] - px, np.asarray(cam_y)[..., None] - py)
        return W_REVISIT * (d < REVISIT_M).sum(-1)

    def info(self, cam: Pose, rays: int = 41) -> float:
        """着いた所から首を振って見える、見ていないセル（1）と確信度の低いセル（1 - 確信度）の量。
        近いほど重い（20〜40 cm から正面に撮ると確かめやすい）。目印が見えなければ、確信度の低い
        セルは数えない（白い壁だけを近くで撮っても位置が合わない）。"""
        g, st = self.grid, self.state
        half = math.radians(HALF_VIEW_DEG)
        steps = int(VIEW_RANGE_M / (g.res * 0.5))
        a = np.linspace(cam.yaw - half, cam.yaw + half, rays)[:, None]
        d = np.arange(1, steps + 1)[None, :] * g.res * 0.5
        ix, iy = g.cell(cam.x + d * np.cos(a), cam.y + d * np.sin(a))
        inside = g.inside(ix, iy)
        cx, cy = np.clip(ix, 0, g.n - 1), np.clip(iy, 0, g.n - 1)
        s = np.where(inside, st[cx, cy], OCCUPIED)
        hitwall = np.cumsum(s == OCCUPIED, axis=1)
        # 光線ごとに最初の障害物まで（その障害物のセルも見える）
        see = (hitwall - (s == OCCUPIED)) == 0
        near = np.where(d <= 0.4, 1.0, np.exp(-(d - 0.4) / 0.8))
        w = np.where(s == UNKNOWN, 1.0, 0.0)
        ok_lm = (not self.use_landmarks) or (
            visible_landmarks(self.landmarks, cam.x, cam.y, cam.yaw) >= LANDMARK_MIN
        )
        if ok_lm:
            w = w + np.where(s != UNKNOWN, 0.5 * (1.0 - self.conf[cx, cy]), 0.0)
        cells = ix * g.n + iy
        val = (w * near)[see & inside]
        # 同じセルは 1 回だけ
        _, first = np.unique(cells[see & inside], return_index=True)
        return float(val[first].sum())


# --- 区切りの列を走らせて点数を付ける ---


def project(steer: np.ndarray, dist: np.ndarray, lim: Limits) -> tuple[np.ndarray, np.ndarray]:
    """区切りの列 (K, S) を、Pi が走れる形に直す（ハンドル・長さ・前と後ろの合計・数・向き）。"""
    steer = np.clip(steer, -MAX_STEER_DEG, MAX_STEER_DEG)
    S = steer.shape[1]
    dist = np.where(np.arange(S)[None, :] < lim.max_segments, dist, 0.0)
    if not lim.reverse:
        dist = np.maximum(dist, 0.0)
    if not lim.gears:
        # 向きは最初に走る区切りのものだけ（今の Pi は 1 歩の中で向きを変えられない）
        nz = dist != 0
        first = np.where(nz.any(1), np.sign(dist[np.arange(len(dist)), nz.argmax(1)]), 1.0)
        dist = np.where(np.sign(dist) == first[:, None], dist, 0.0)
    back = dist < 0
    mag = np.abs(dist)
    cap = np.where(~back & (np.abs(steer) > SKIP_STEER_DEG), SKIP_SEG_MAX_M, SEG_MAX_M)
    mag = np.minimum(mag, cap)
    mag = np.where(mag < SEG_MIN_M, 0.0, mag)
    fwd_sum = np.where(~back, mag, 0).sum(1, keepdims=True)
    rev_sum = np.where(back, mag, 0).sum(1, keepdims=True)
    mag = np.where(~back, mag * np.minimum(1.0, lim.max_fwd_m / np.maximum(fwd_sum, 1e-9)), mag)
    mag = np.where(back, mag * np.minimum(1.0, REVERSE_MAX_M / np.maximum(rev_sum, 1e-9)), mag)
    mag = np.where(mag < SEG_MIN_M, 0.0, mag)
    return steer, np.where(back, -mag, mag)


@dataclass
class Rollout:
    cost: np.ndarray  # (K,)
    ok: np.ndarray
    end_rear: np.ndarray  # (K, 3)
    seg_end_rear: np.ndarray  # (K, S, 3)
    gap: np.ndarray
    parts: dict


def rollout(scene: Scene, steer: np.ndarray, dist: np.ndarray, wide: bool = False) -> Rollout:
    """区切りの列 (K, S)（ハンドル°、符号付きの m。負は後ろへ）を走らせてコストを出す。"""
    K, S = steer.shape
    r = scene.start
    x = np.full(K, r.x)
    y = np.full(K, r.y)
    yaw = np.full(K, r.yaw)
    k = -np.tan(np.radians(steer)) / WHEELBASE_M
    ds = dist / SUB
    T = S * SUB
    X = np.empty((K, T))
    Y = np.empty((K, T))
    YAW = np.empty((K, T))
    seg_end = np.empty((K, S, 3))
    for s in range(S):
        for j in range(SUB):
            x, y, yaw = _arc_np(x, y, yaw, k[:, s], ds[:, s])
            t = s * SUB + j
            X[:, t], Y[:, t], YAW[:, t] = x, y, yaw
        seg_end[:, s] = np.stack([x, y, yaw], 1)
    back = np.repeat(dist < 0, SUB, axis=1)
    st = np.repeat(steer, SUB, axis=1)
    step = np.repeat(np.abs(ds), SUB, axis=1)
    moving = step > 0
    hit, lead, blocked, side, maybe = scene.check(X, Y, YAW, back, st)
    hit &= moving
    blocked &= moving
    unk_rev = (step * (lead & back)).sum(1)
    unk_fwd = (step * (lead & ~back)).sum(1)
    maybe_len = (step * (maybe & ~back)).sum(1)
    fwd_len = np.where(dist > 0, dist, 0).sum(1)
    rev_len = np.where(dist < 0, -dist, 0).sum(1)
    total = fwd_len + rev_len
    used = np.abs(dist) > 0
    # 前後の切り替え：0 でない区切りどうしで向きが変わった数
    sign = np.sign(dist)
    switches = np.zeros(K)
    last = np.zeros(K)
    for s in range(S):
        cur = sign[:, s]
        switches += (cur != 0) & (last != 0) & (cur != last)
        last = np.where(cur != 0, cur, last)
    ok = ~hit.any(1) & ~blocked.any(1) & (unk_rev <= UNSEEN_REV_MAX_M + 1e-9)
    ok &= (unk_fwd <= UNSEEN_FWD_MAX_M + 1e-9) & (total >= MIN_TOTAL_M)
    D_end = scene.D_at(x, y, yaw)
    D_end = np.where(np.isfinite(D_end), D_end, max(scene.D_start, 0.0) + D_FAR_M)
    D0 = scene.D_start if math.isfinite(scene.D_start) else float(np.nanmin(D_end))
    side_m = np.where(moving, side, np.inf)
    narrow = side_m.min(1) < NARROW_M
    gap = narrow & (D_end < D0 - 0.10)
    # 区切りの終わりのカメラから見える目印（使わない区切りは数えない）
    cx = seg_end[:, :, 0] + CAM_X_M * np.cos(seg_end[:, :, 2])
    cy = seg_end[:, :, 1] + CAM_X_M * np.sin(seg_end[:, :, 2])
    lost = (scene.lost_penalty(cx, cy, seg_end[:, :, 2]) * used).max(1)
    ex, ey = x + CAM_X_M * np.cos(yaw), y + CAM_X_M * np.sin(yaw)
    revisit = scene.revisit(ex, ey) * (WIDE_REVISIT if wide else 1.0)
    cost = (
        W_FWD * fwd_len
        + W_REV * rev_len
        + W_SWITCH * switches * (WIDE_SWITCH if wide else 1.0)
        + W_SEG * used.sum(1)
        + W_UNK_REV * unk_rev
        + W_UNK_FWD * unk_fwd
        + W_GOAL * (D_end - D0)
        - W_GAP * gap
        - W_MAYBE * maybe_len
        + lost
        + revisit
    )
    cost = np.where(ok, cost, np.inf)
    parts = {"D_end": D_end, "switches": switches, "unk_rev": unk_rev, "unk_fwd": unk_fwd,
             "lost": lost, "narrow": narrow, "fwd": fwd_len, "rev": rev_len}  # fmt: skip
    return Rollout(cost, ok, np.stack([x, y, yaw], 1), seg_end, gap, parts)


def _pad(moves: list[tuple[float, float, bool]], S: int = GEAR_MAX_SEGMENTS):
    steer = np.zeros(S)
    dist = np.zeros(S)
    for i, (st, m, back) in enumerate(moves[:S]):
        steer[i], dist[i] = st, -m if back else m
    return steer, dist


def _moves(steer: np.ndarray, dist: np.ndarray) -> list[tuple[float, float, bool]]:
    return [(float(s), float(abs(d)), bool(d < 0)) for s, d in zip(steer, dist, strict=True)
            if abs(d) >= SEG_MIN_M]  # fmt: skip


def templates(lim: Limits) -> list[list[tuple[float, float, bool]]]:
    """名目の列のひな形：まっすぐ・弧・切り返し・3 点ターン・下がって寄せる。"""
    s = MAX_STEER_DEG
    out: list[list[tuple[float, float, bool]]] = [[(0.0, min(0.3, lim.max_fwd_m), False)]]
    for d in (s, -s):
        out += [
            [(d, 0.2, False)],
            [(d, 0.3, True)],
            [(d, 0.15, True)],
            [(d, 0.2, False), (-d, 0.2, True), (d, 0.2, False)],  # 3 点ターン
            [(-d, 0.15, True), (d, 0.2, False)],  # 下がってから曲がって出る
            [(d, 0.15, True), (0.0, 0.25, False)],  # 下がって向きを変え、まっすぐ出る
            [
                (d, 0.15, False),
                (-d, 0.15, True),
                (d, 0.15, False),
                (-d, 0.15, True),
                (d, 0.15, False),
            ],
            [(d, 0.2, False), (0.0, 0.2, False)],
            [(d, 0.3, True), (0.0, 0.2, True)],
        ]
    return out


# --- MPPI ---


def mppi(
    scene: Scene,
    nominals: list[list[tuple[float, float, bool]]],
    lim: Limits,
    rng: np.random.Generator,
    samples: int = 160,
    iters: int = 3,
    lam: float = 0.15,
    budget_s: float = 0.8,
) -> list[list[tuple[float, float, bool]]]:
    """区切りの列をばらまき、コストで重みを付けて名目の列を寄せ直す。良い順に TOP_K 個の列を返す
    （着いた所から見えるものは plan_maneuver で足す）。"""
    t0 = time.perf_counter()
    S = GEAR_MAX_SEGMENTS
    noms = [_pad(m) for m in nominals if m]
    if not noms:
        return []
    ns = np.array([n[0] for n in noms])
    nd = np.array([n[1] for n in noms])
    ns, nd = project(ns, nd, lim)
    ro = rollout(scene, ns, nd, lim.wide)
    best_s, best_d, best_c = None, None, math.inf
    i = int(np.argmin(ro.cost))
    if math.isfinite(ro.cost[i]):
        best_s, best_d, best_c = ns[i].copy(), nd[i].copy(), float(ro.cost[i])
    pool_s, pool_d, pool_c = ns, nd, ro.cost
    for _ in range(iters):
        if time.perf_counter() - t0 > budget_s:
            break
        # 名目：今までの良い順に（無限大でも、形はばらまきの元にする）
        order = np.argsort(pool_c)[: max(4, len(noms))]
        base = rng.choice(order, samples)
        steer = pool_s[base] + rng.normal(0, 8.0, (samples, S))
        dist = pool_d[base] + rng.normal(0, 0.05, (samples, S)) * (pool_d[base] != 0)
        # ときどき区切りを足す・消す・向きを替える
        add = rng.random((samples, S)) < 0.08
        dist = np.where(
            add & (dist == 0),
            rng.choice([-1, 1], (samples, S)) * rng.uniform(0.05, 0.25, (samples, S)),
            dist,
        )
        drop = rng.random((samples, S)) < 0.05
        dist = np.where(drop, 0.0, dist)
        # 4 分の 1 は形からでたらめ（交互の向き・ハンドルは 5 通り）
        m = samples // 4
        n_seg = rng.integers(1, S + 1, m)
        first = rng.choice([-1.0, 1.0], m)
        alt = np.where(rng.random((m, S)) < 0.6, -1.0, 1.0).cumprod(1) * first[:, None]
        rd = alt * rng.uniform(0.05, 0.30, (m, S)) * (np.arange(S)[None, :] < n_seg[:, None])
        rs = rng.choice([-MAX_STEER_DEG, -14.0, 0.0, 14.0, MAX_STEER_DEG], (m, S)) + rng.normal(
            0, 3, (m, S)
        )
        steer[-m:], dist[-m:] = rs, rd
        steer, dist = project(steer, dist, lim)
        ro = rollout(scene, steer, dist, lim.wide)
        c = ro.cost
        fin = np.isfinite(c)
        if fin.any():
            j = int(np.argmin(np.where(fin, c, np.inf)))
            if c[j] < best_c:
                best_s, best_d, best_c = steer[j].copy(), dist[j].copy(), float(c[j])
            # 重みを付けて寄せる：いちばん良い列と同じ向きの並びの列だけ平均する
            pat = np.sign(dist[j])
            same = fin & (np.sign(dist) == pat).all(1)
            w = np.exp(-(c[same] - c[j]) / lam)
            w /= w.sum()
            ms = (w[:, None] * steer[same]).sum(0)
            md = (w[:, None] * dist[same]).sum(0)
            ms, md = project(ms[None], md[None], lim)
            r2 = rollout(scene, ms, md, lim.wide)
            if r2.cost[0] < best_c:
                best_s, best_d, best_c = ms[0].copy(), md[0].copy(), float(r2.cost[0])
            pool_s = np.concatenate([pool_s, steer[fin], ms])
            pool_d = np.concatenate([pool_d, dist[fin], md])
            pool_c = np.concatenate([pool_c, c[fin], r2.cost])
            keep = np.argsort(pool_c)[:64]
            pool_s, pool_d, pool_c = pool_s[keep], pool_d[keep], pool_c[keep]
    out: list[list[tuple[float, float, bool]]] = []
    if best_s is not None:
        out.append(_moves(best_s, best_d))
    for j in np.argsort(pool_c):
        if len(out) >= TOP_K or not math.isfinite(pool_c[j]):
            break
        mv = _moves(pool_s[j], pool_d[j])
        if mv and mv not in out:
            out.append(mv)
    return out


# MPPI から、着いた所から見えるものを足して比べる列の数
TOP_K = 8

# --- Hybrid A* ---

PRIM_M = 0.10
PRIM_SUB = 4
YAW_BINS = 24


def hybrid_astar(
    scene: Scene, lim: Limits, max_nodes: int = 500, budget_s: float = 0.5
) -> list[tuple[float, float, bool]] | None:
    """(x, y, 向き) の格子の上で、10 cm の弧（ハンドル 3 通り × 前・後ろ）をつなぐ。
    いちばん良い終わり（コスト + 行き先までの道のり）の列を返す。"""
    import heapq

    t0 = time.perf_counter()
    r = scene.start
    s = MAX_STEER_DEG
    prims = [(st, back) for back in ((False, True) if lim.reverse else (False,))
             for st in (-s, 0.0, s)]  # fmt: skip
    D0 = scene.D_start if math.isfinite(scene.D_start) else 0.0
    # ノード：(f, 番号)。中身は別に持つ
    nodes = [dict(x=r.x, y=r.y, yaw=r.yaw, g=0.0, fwd=0.0, rev=0.0, unk_r=0.0, unk_f=0.0,
                  moves=[], narrow=False)]  # fmt: skip
    heap = [(0.0, 0)]
    seen: dict[tuple, float] = {}
    best, best_j = None, math.inf
    pops = 0
    while heap and pops < max_nodes and time.perf_counter() - t0 < budget_s:
        _, i = heapq.heappop(heap)
        nd = nodes[i]
        pops += 1
        # 子をまとめて確かめる
        xs, ys, yaws, backs, steers = [], [], [], [], []
        for st, back in prims:
            k = curvature(st)
            x, y, yaw = nd["x"], nd["y"], nd["yaw"]
            for _ in range(PRIM_SUB):
                x, y, yaw = (
                    float(v)
                    for v in _arc_np(
                        np.array(x),
                        np.array(y),
                        np.array(yaw),
                        np.array(k),
                        np.array((-1 if back else 1) * PRIM_M / PRIM_SUB),
                    )
                )
                xs.append(x)
                ys.append(y)
                yaws.append(yaw)
                backs.append(back)
                steers.append(st)
        hit, lead, blocked, side, _ = scene.check(
            np.array(xs), np.array(ys), np.array(yaws), np.array(backs), np.array(steers)
        )
        for p, (st, back) in enumerate(prims):
            sl = slice(p * PRIM_SUB, (p + 1) * PRIM_SUB)
            if hit[sl].any() or blocked[sl].any():
                continue
            moves = list(nd["moves"])
            if not lim.gears and moves and moves[0][2] != back:
                continue
            if moves and moves[-1][0] == st and moves[-1][2] == back:
                cap = SKIP_SEG_MAX_M if (not back and abs(st) > SKIP_STEER_DEG) else SEG_MAX_M
                if moves[-1][1] + PRIM_M > cap + 1e-9:
                    if len(moves) >= lim.max_segments:
                        continue
                    moves.append((st, PRIM_M, back))
                    new_seg = True
                else:
                    moves[-1] = (st, moves[-1][1] + PRIM_M, back)
                    new_seg = False
            else:
                if len(moves) >= lim.max_segments:
                    continue
                moves.append((st, PRIM_M, back))
                new_seg = True
            fwd = nd["fwd"] + (0 if back else PRIM_M)
            rev = nd["rev"] + (PRIM_M if back else 0)
            if fwd > lim.max_fwd_m + 1e-9 or rev > REVERSE_MAX_M + 1e-9:
                continue
            ul = PRIM_M * lead[sl].mean()
            unk_r = nd["unk_r"] + (ul if back else 0)
            unk_f = nd["unk_f"] + (0 if back else ul)
            if unk_r > UNSEEN_REV_MAX_M + 1e-9 or unk_f > UNSEEN_FWD_MAX_M + 1e-9:
                continue
            switch = bool(nd["moves"]) and nd["moves"][-1][2] != back
            g = nd["g"] + (W_REV if back else W_FWD) * PRIM_M
            g += W_SWITCH * (WIDE_SWITCH if lim.wide else 1.0) * switch
            g += W_SEG * new_seg + (W_UNK_REV if back else W_UNK_FWD) * ul
            x, y, yaw = xs[sl.stop - 1], ys[sl.stop - 1], yaws[sl.stop - 1]
            key = (
                round(x / 0.05),
                round(y / 0.05),
                round(yaw / (2 * math.pi / YAW_BINS)) % YAW_BINS,
                back,
            )
            if seen.get(key, math.inf) <= g:
                continue
            seen[key] = g
            D = float(scene.D_at(x, y, yaw))
            D = D if math.isfinite(D) else D0 + D_FAR_M
            narrow = nd["narrow"] or bool((side[sl] < NARROW_M).any())
            child = dict(x=x, y=y, yaw=yaw, g=g, fwd=fwd, rev=rev, unk_r=unk_r, unk_f=unk_f,
                         moves=moves, narrow=narrow)  # fmt: skip
            nodes.append(child)
            j = len(nodes) - 1
            cam = camera_of(Pose(x, y, yaw))
            J = g + W_GOAL * (D - D0) - W_GAP * (narrow and D < D0 - 0.1)
            J += float(scene.lost_penalty(cam.x, cam.y, cam.yaw))
            J += float(scene.revisit(cam.x, cam.y)) * (WIDE_REVISIT if lim.wide else 1.0)
            if fwd + rev >= MIN_TOTAL_M and J < best_j:
                best, best_j = moves, J
            heapq.heappush(heap, (g + W_GOAL * (D - D0), j))
    return best


# --- まとめ ---


def plan_maneuver(
    grid: Grid,
    pose: Pose,
    lim: Limits | None = None,
    nominal: list[tuple[float, float, bool]] | None = None,
    seed: int = 0,
    recent: list[Pose] | None = None,
    budget_s: float = 0.9,
    mode: str = "mppi",
    goal: Pose | None = None,
) -> Maneuver | None:
    """次の 1 歩（区切りの列）を決める。走れる列が無い・行き先が無ければ None。

    名目の列は、nominal（今の決め方 next_best_view の 1 歩）・Hybrid A* の答え・ひな形。MPPI で
    寄せ直した上位の列と合わせて、同じコストに「着いた所から見えるもの」を足して選ぶ。
    mode="hybrid" は MPPI を使わない（ベンチで比べるため）。
    """
    t0 = time.perf_counter()
    lim = lim or Limits()
    scene = Scene(grid, pose, True, recent, lim.wide, goal)
    if not scene.targets.any():
        return None
    rng = np.random.default_rng(seed)
    cands: list[tuple[list, str]] = []
    if nominal:
        cands.append((list(nominal), "nominal"))
    hyb = hybrid_astar(scene, lim, budget_s=0.35 if mode == "mppi" else budget_s)
    if hyb:
        cands.append((hyb, "hybrid"))
    if mode == "mppi":
        noms = [m for m, _ in cands] + templates(lim)
        left = max(0.1, budget_s - (time.perf_counter() - t0) - 0.15)
        cands += [(m, "mppi") for m in mppi(scene, noms, lim, rng, budget_s=left)]
    elif mode != "hybrid":
        raise ValueError(f"mode は hybrid か mppi: {mode}")
    scored = []
    for moves, kind in cands:
        if not moves:
            continue
        s, d = _pad(moves)
        s, d = project(s[None], d[None], lim)
        ro = rollout(scene, s, d, lim.wide)
        if not math.isfinite(ro.cost[0]):
            continue
        mv = _moves(s[0], d[0])
        end = camera_of(Pose(*map(float, ro.end_rear[0])))
        gain = scene.info(end)
        scored.append((float(ro.cost[0]) - W_INFO * gain, kind, mv, ro, gain))
    if not scored:
        return None
    total, kind, mv, ro, gain = min(scored, key=lambda c: c[0])
    ends, clears = [], []
    for k, (_, _, back) in enumerate(mv):
        ends.append(camera_of(Pose(*map(float, ro.seg_end_rear[0, k]))))
        clears.append(None if back else min_clear_cm(_clear_ahead_cm(scene, k, mv)))
    return Maneuver(
        mv, float(total), ends, clears, bool(ro.gap[0]), kind,
        {"D_start": round(scene.D_start, 3) if math.isfinite(scene.D_start) else None,
         "connected": scene.connected,
         "D_end": round(float(ro.parts["D_end"][0]), 3), "lowconf_targets": scene.lowconf_targets,
         "landmarks": len(scene.landmarks), "ms": round(1000 * (time.perf_counter() - t0), 1)},
        [camera_of(r) for _, r, _, _ in drive_moves(pose, mv, step=0.02)], round(gain, 1),
    )  # fmt: skip


def min_clear_cm(pred_cm: float | None) -> float | None:
    """Pi に渡す clear_cm の下限：地図の見込みより CLEAR_TOL_CM 近くに物が見えたら止まる
    （地図に無い物・猫）。CLEAR_CAP_CM まで。Pi の explore_clear_stop_cm（25 cm）より厳しく
    ならなければ None。"""
    if pred_cm is None:
        return None
    v = min(pred_cm - CLEAR_TOL_CM, CLEAR_CAP_CM)
    return round(v, 1) if v > CLEAR_STOP_M * 100 else None


def _clear_ahead_cm(scene: Scene, k: int, moves) -> float | None:
    """前への区切り k の通り道で、地図から見込む前の端の先の空き（cm）のいちばん小さい値。
    ハンドルが大きい（Pi の clear_cm が画角の外）区切りは None。"""
    st = moves[k][0]
    if abs(st) > CLEAR_STEER_DEG:
        return None
    r = scene.start
    x, y, yaw = r.x, r.y, r.yaw
    for s2, m2, b2 in moves[:k]:
        x, y, yaw = (
            float(v)
            for v in _arc_np(
                np.array(x),
                np.array(y),
                np.array(yaw),
                np.array(curvature(s2)),
                np.array(-m2 if b2 else m2),
            )
        )
    kk = curvature(st)
    best = 100.0
    for f in np.linspace(0, moves[k][1], 4):
        px, py, pyaw = (
            float(v)
            for v in _arc_np(np.array(x), np.array(y), np.array(yaw), np.array(kk), np.array(f))
        )
        for s in np.arange(0.05, 1.0, 0.05):
            ax, ay, ayaw = (
                float(v)
                for v in _arc_np(
                    np.array(px), np.array(py), np.array(pyaw), np.array(kk), np.array(s)
                )
            )
            pts = np.stack([np.full(len(AHEAD_Y), CAR_FRONT_M), AHEAD_Y], 1)
            wx, wy = scene._world(pts, ax, ay, ayaw)
            if scene._lookup(scene.occ, wx, wy, True).any():
                best = min(best, 100 * s)
                break
    return round(best, 1)


def moves_ok(grid: Grid, pose: Pose, moves: list[tuple[float, float, bool]]) -> str | None:
    """区切りの列を、計画と同じ決まりで確かめる。走れれば None、走れなければ理由。
    （エージェントが予約した 1 歩を確かめる。app/snoo_map_server.take_override）"""
    scene = Scene(grid, pose, False, None)
    s, d = _pad(moves)
    ro = rollout(scene, s[None], d[None])
    if ro.ok[0]:
        return None
    p = ro.parts
    if p["unk_rev"][0] > UNSEEN_REV_MAX_M:
        return "後ろの確かでない所に入る"
    if p["unk_fwd"][0] > UNSEEN_FWD_MAX_M:
        return "前の見ていない所に入る"
    return "車の外形が障害物に当たる・前が近い"
