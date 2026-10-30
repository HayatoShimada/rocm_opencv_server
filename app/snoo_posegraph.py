"""ステーションの位置を、VGGT の窓の答えと走った 1 歩（自転車モデル）を重ねた位置のグラフで決める。

今の地図のサーバー（app/snoo_map_server.plan_station）は、窓（4 か所）を VGGT に通し、最後の
ステーションの位置だけを、前に決めた窓の原点の位置に足して決める。一度決めた位置は変えない。
この決め方は、次の 3 つに弱い（2026-10-30 の実走 explore_20261030_190244 で 46 か所中 11 か所が
位置合わせの失敗になった）。
- 1 回の VGGT が崩れると（首振りのカメラの中心が 10〜20 cm 散る）、そのステーションの位置が無くなる
- 窓の原点の位置のずれが、そのまま後のステーションに積もる
- 走った 1 歩（指令のハンドルと Pi が見積もった長さ）を位置に使っていない

ここでは、ステーションを点、次のものを辺にしたグラフを作り、全部の辺に合うよう位置をまとめて
決め直す（scipy の least_squares、Huber の損失、SE(2)）。
- 窓の辺：窓に入ったステーションどうしの、窓の座標での相対の位置（窓 1 つで最大 6 本）。
  窓は毎回すでに VGGT に通しているので、足す GPU の時間は無い。確からしさは首振りのばらつき
  （spread）と、離れているほど大きくする
- 走った辺：前のステーションから prev_step の区間を自転車モデル（app/roommap.drive_poses）で
  走らせた相対の位置。長さは Pi の見積もり（est_distance_cm）で、ハンドルはおおよそなので、
  向きの確からしさは小さめにする
- 見直しの辺（任意）：近くに戻ってきたステーションどうしを、別に VGGT に通した相対の位置

崩れたステーション（ほかの辺と合わない窓の辺）は、辺ごと外れ値として外す。ステーションごと
落とさないので、走った辺と別の窓の辺で位置が残る。

numpy・scipy だけ（GPU は使わない）。地図のサーバーにはまだつないでいない。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.roommap import Pose, camera_of, drive_poses
from app.snoo_mapper import (
    MAX_STATION_SPREAD_M,
    MIN_POINTS_PER_CELL,
    FrameRef,
    MapParams,
    Observation,
    Station,
    StationResult,
    build_map,
    load_station,
    registration_is_bad,
    to_frame,
    xy_to_frame,
)

# 窓の辺の確からしさ（1σ）。首振りのばらつきが小さくても、窓ごとの大きさ（カメラの高さで決める）
# のずれと床の傾きの分だけずれる。離れているほど大きくする
VISION_SIGMA_M = 0.03
VISION_SIGMA_PER_M = 0.08
VISION_SIGMA_DEG = 3.0
# 首振りのばらつき 1 m あたりで足す分（ばらつきが大きい窓ほど信じない）
SPREAD_GAIN = 1.0
SPREAD_GAIN_DEG_PER_M = 40.0
# 走った辺の確からしさ。長さは Pi の見積もり（1 回目の実走では 0.6 倍のこともあった）、
# ハンドルは指令どおりには曲がらない。曲がった角度が大きいほど向きのずれも大きい
ODOM_SIGMA_M = 0.04
ODOM_SIGMA_PER_M = 0.25
ODOM_SIGMA_DEG = 6.0
ODOM_SIGMA_PER_TURN = 0.3
# Huber の損失が 2 乗から 1 次に変わる所（σ の何倍か）
HUBER_SIGMA = 1.5
# 最後の残りがこれ（σ の何倍か、3 成分のノルム）を超えた辺を外れ値とする
OUTLIER_SIGMA = 4.0
# 窓の辺に使うステーションの、首振りのばらつきの上限（今の地図のサーバーと同じ 10 cm）
EDGE_MAX_SPREAD_M = MAX_STATION_SPREAD_M


def wrap(a: float | np.ndarray) -> float | np.ndarray:
    return (
        (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
        if isinstance(a, np.ndarray)
        else (math.atan2(math.sin(a), math.cos(a)))
    )


def relative(a: Pose, b: Pose) -> Pose:
    """b を a から見た位置（a の座標）。to_frame(a, relative(a, b)) == b。"""
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    dx, dy = b.x - a.x, b.y - a.y
    return Pose(c * dx + s * dy, -s * dx + c * dy, wrap(b.yaw - a.yaw))


@dataclass
class Edge:
    """a から見た b の位置（z）と、その確からしさ（1σ）。kind は window・odom・loop。"""

    a: int
    b: int
    z: Pose
    sigma_m: float
    sigma_rad: float
    kind: str
    source: str = ""  # どの窓から出したか（窓の最後のステーション）など
    outlier: bool = False
    chi: float = 0.0  # 最後の残り（σ の何倍か）


# --- 窓の辺 ---


def load_recon(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """ステーションに残した VGGT の点（app/snoo_map_server.save_recon の形）。
    返すのは (keys, extrinsic, points, conf)。keys は「ステーションの置き場/画像の名前」。
    """
    d = np.load(path)
    return [str(k) for k in d["keys"]], d["extrinsic"], d["points"], d["conf"]


def window_stations(session: Path, keys: list[str]) -> tuple[list[int], list[Station], list]:
    """keys に出てくるステーション（出てきた順）と、その止まって撮ったコマと、FrameRef の列。"""
    names: list[str] = []
    for k in keys:
        n = k.split("/")[0]
        if n not in names:
            names.append(n)
    stations = []
    for n in names:
        st = load_station(session / n, "meta.json")
        stations.append(Station(st.dir, [f for f in st.frames if not f.get("moving")]))
    by_name = {n: (i, {f["file"]: f for f in st.frames}) for i, (n, st) in
               enumerate(zip(names, stations, strict=True))}  # fmt: skip
    refs = []
    for k in keys:
        n, file = k.split("/", 1)
        si, frames = by_name[n]
        f = frames[file]
        refs.append(FrameRef(stations[si].dir / file, si, f["pan_deg"], f["tilt_deg"]))
    return [int(n) for n in names], stations, refs


@dataclass
class Window:
    """1 回の VGGT（窓）で求めたステーションの位置（窓の座標）。"""

    source: str
    ids: list[int]
    results: list[StationResult]
    heights: list[float | None]
    seconds: float = 0.0
    # build_map の答え（地図に点を足すときに使う。残さなくてもよい）
    map_result: object | None = None
    points: np.ndarray | None = None
    conf: np.ndarray | None = None
    extrinsic: np.ndarray | None = None
    refs: list = field(default_factory=list)

    def good(self, i: int) -> bool:
        r = self.results[i]
        return bool(r.frames) and not registration_is_bad(r, self.heights[i])


def solve_window(
    session: Path,
    keys: list[str],
    extrinsic: np.ndarray,
    points: np.ndarray,
    conf: np.ndarray,
    source: str,
    heights: list[float | None] | None = None,
    keep: bool = False,
) -> Window:
    """VGGT の 1 回分の答えから、窓の中のステーションの位置を build_map と同じに出す。"""
    ids, stations, refs = window_stations(session, keys)
    if heights is None:
        heights = [camera_height(session / f"{i:03d}") for i in ids]
    params = MapParams(camera_height_m=list(heights))
    r = build_map(stations, refs, extrinsic, points, conf, params, plan=False)
    w = Window(source, ids, r.stations, list(heights))
    if keep:
        w.map_result, w.points, w.conf, w.refs = r, points, conf, refs
        w.extrinsic = extrinsic
    return w


def camera_height(d: Path, default_cm: float = 8.0) -> float:
    try:
        h = json.loads((d / "meta.json").read_text()).get("camera_height_cm")
    except (OSError, ValueError):
        h = None
    return (h or default_cm) / 100


def window_edges(w: Window, max_spread_m: float = EDGE_MAX_SPREAD_M) -> list[Edge]:
    """窓の中の、位置合わせの崩れていないステーションどうしの辺（全部の組）。"""
    # 番号が負のもの（走りながらのコマなど、ステーションでないもの）は使わない
    ok = [i for i in range(len(w.ids))
          if w.ids[i] >= 0 and w.good(i) and w.results[i].spread_m <= max_spread_m]  # fmt: skip
    edges = []
    for x in range(len(ok)):
        for y in range(x + 1, len(ok)):
            i, j = ok[x], ok[y]
            ra, rb = w.results[i], w.results[j]
            z = relative(ra.pose, rb.pose)
            spread = ra.spread_m + rb.spread_m
            dist = math.hypot(z.x, z.y)
            edges.append(
                Edge(
                    w.ids[i], w.ids[j], z,
                    VISION_SIGMA_M + VISION_SIGMA_PER_M * dist + SPREAD_GAIN * spread,
                    math.radians(VISION_SIGMA_DEG + SPREAD_GAIN_DEG_PER_M * spread),
                    "window", w.source,
                )
            )  # fmt: skip
    return edges


# --- 走った辺 ---


def odom_pose(pieces: list[tuple[float, float]], reverse: bool = False) -> Pose:
    """カメラの真下が原点の車が、区間 [(ハンドル°, m), ...] を走った後のカメラの真下の位置。"""
    if not pieces:
        return Pose(0.0, 0.0, 0.0)
    return camera_of(drive_poses(Pose(0.0, 0.0, 0.0), pieces, reverse)[-1][1])


def odom_edge(a: int, b: int, pieces: list[tuple[float, float]], reverse: bool) -> Edge:
    """a から b まで、区間を走った辺。区間の長さは実際に走った分（take で切ってから渡す）。"""
    z = odom_pose(pieces, reverse)
    length = sum(m for _, m in pieces)
    turn = abs(z.yaw)
    return Edge(
        a, b, z,
        ODOM_SIGMA_M + ODOM_SIGMA_PER_M * length,
        math.radians(ODOM_SIGMA_DEG) + ODOM_SIGMA_PER_TURN * turn,
        "odom", f"{b:03d}",
    )  # fmt: skip


# --- まとめて決める ---


@dataclass
class Solution:
    poses: dict[int, Pose]
    edges: list[Edge]
    # 窓か見直しの辺（外れ値でない）が 1 本以上あるステーション。無いものは走った辺だけで決めた
    seen: set[int]
    cost: float
    iterations: int


class _Problem:
    """辺を配列にしたもの。未知数は fixed 以外のステーションの (x, y, yaw)。"""

    def __init__(self, ids: list[int], fixed: int, edges: list[Edge]):
        self.ids = ids
        col = {i: k for k, i in enumerate(i for i in ids if i != fixed)}
        self.n = len(col)
        self.ca = np.array([col.get(e.a, -1) for e in edges], int)
        self.cb = np.array([col.get(e.b, -1) for e in edges], int)
        self.z = np.array([[e.z.x, e.z.y, e.z.yaw] for e in edges], float).reshape(-1, 3)
        self.sm = np.array([e.sigma_m for e in edges], float)
        self.sr = np.array([e.sigma_rad for e in edges], float)

    def _poses(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        p = np.zeros((len(c), 3))
        ok = c >= 0
        p[ok] = x.reshape(-1, 3)[c[ok]]
        return p

    def residuals(self, x: np.ndarray) -> np.ndarray:
        pa, pb = self._poses(x, self.ca), self._poses(x, self.cb)
        c, s = np.cos(pa[:, 2]), np.sin(pa[:, 2])
        dx, dy = pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1]
        r = np.empty((len(c), 3))
        r[:, 0] = (c * dx + s * dy - self.z[:, 0]) / self.sm
        r[:, 1] = (-s * dx + c * dy - self.z[:, 1]) / self.sm
        r[:, 2] = wrap(pb[:, 2] - pa[:, 2] - self.z[:, 2]) / self.sr
        return r.ravel()

    def jacobian(self, x: np.ndarray):
        from scipy.sparse import coo_matrix

        pa, pb = self._poses(x, self.ca), self._poses(x, self.cb)
        c, s = np.cos(pa[:, 2]), np.sin(pa[:, 2])
        dx, dy = pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1]
        sm, sr = self.sm, self.sr
        E = np.arange(len(c))
        # (行の成分, 列のステーション, 列の成分, 値)
        parts = [
            (0, self.ca, 0, -c / sm), (0, self.ca, 1, -s / sm),
            (0, self.ca, 2, (-s * dx + c * dy) / sm), (0, self.cb, 0, c / sm),
            (0, self.cb, 1, s / sm),
            (1, self.ca, 0, s / sm), (1, self.ca, 1, -c / sm),
            (1, self.ca, 2, (-c * dx - s * dy) / sm), (1, self.cb, 0, -s / sm),
            (1, self.cb, 1, c / sm),
            (2, self.ca, 2, -1 / sr), (2, self.cb, 2, 1 / sr),
        ]  # fmt: skip
        rows, cols, vals = [], [], []
        for k, node, j, v in parts:
            ok = node >= 0
            rows.append(3 * E[ok] + k)
            cols.append(3 * node[ok] + j)
            vals.append(v[ok])
        return coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(3 * len(c), 3 * self.n),
        ).tocsr()


def chain_init(ids: list[int], edges: list[Edge], fixed: int) -> dict[int, Pose]:
    """辺をたどって最初の位置を置く。走った辺（番号の順）を先に、残りは確からしい辺から。
    走った辺が途切れた先（prev_step の無いステーションの後）は窓の辺でつなぐ。どの辺でも
    つながらないものは、その塊の最初のステーションを原点に置いて、そこからたどる。
    """
    odom = sorted((e for e in edges if e.kind == "odom"), key=lambda e: e.a)
    order = odom + sorted((e for e in edges if e.kind != "odom"), key=lambda e: e.sigma_m)
    poses = {fixed: Pose(0.0, 0.0, 0.0)}
    while True:
        changed = True
        while changed:
            changed = False
            for e in order:
                if e.a in poses and e.b not in poses:
                    poses[e.b] = to_frame(poses[e.a], e.z)
                    changed = True
                elif e.b in poses and e.a not in poses:
                    poses[e.a] = to_frame(poses[e.b], inverse(e.z))
                    changed = True
        rest = [i for i in ids if i not in poses]
        if not rest:
            return poses
        poses[rest[0]] = Pose(0.0, 0.0, 0.0)


def inverse(p: Pose) -> Pose:
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    return Pose(-(c * p.x + s * p.y), -(-s * p.x + c * p.y), wrap(-p.yaw))


def optimize(
    edges: list[Edge],
    fixed: int,
    init: dict[int, Pose] | None = None,
    huber: float = HUBER_SIGMA,
    outlier_sigma: float = OUTLIER_SIGMA,
    rounds: int = 3,
) -> Solution:
    """辺に合うよう位置を決める。fixed のステーションを原点 (0, 0, 0) に止める。

    Huber の損失で解き、残りが outlier_sigma を超えた窓・見直しの辺を外して解き直す
    （rounds 回まで）。走った辺は外さない（つながりが切れないように）。
    """
    from scipy.optimize import least_squares

    ids = sorted({e.a for e in edges} | {e.b for e in edges} | {fixed})
    live = list(edges)
    for e in live:
        e.outlier = False
    poses = init or chain_init(ids, live, fixed)
    cost, it = 0.0, 0
    for _ in range(rounds):
        use = [e for e in live if not e.outlier]
        x0 = np.array([v for i in ids if i != fixed for v in
                       (poses[i].x, poses[i].y, poses[i].yaw)], float)  # fmt: skip
        if len(x0):
            prob = _Problem(ids, fixed, use)
            res = least_squares(
                prob.residuals, x0, jac=prob.jacobian, loss="huber", f_scale=huber,
                method="trf", tr_solver="lsmr", max_nfev=100,
            )  # fmt: skip
            x, cost, it = res.x, float(res.cost), it + int(res.nfev)
        else:
            x = x0
        k = 0
        poses = {fixed: Pose(0.0, 0.0, 0.0)}
        for i in ids:
            if i != fixed:
                poses[i] = Pose(float(x[k]), float(x[k + 1]), float(wrap(x[k + 2])))
                k += 3
        r = _Problem(ids, fixed, live).residuals(x).reshape(-1, 3)
        chi = np.linalg.norm(r, axis=1)
        changed = False
        for e, c in zip(live, chi, strict=True):
            e.chi = float(c)
            bad = bool(e.kind != "odom" and c > outlier_sigma)
            if bad != e.outlier:
                e.outlier, changed = bad, True
        if not changed:
            break
    seen = {i for e in live if e.kind != "odom" and not e.outlier for i in (e.a, e.b)}
    return Solution(poses, live, seen, cost, it)


# --- 地図 ---


@dataclass
class LocalCloud:
    """1 ステーションで見た点（そのステーションの座標：原点はカメラの真下、x は車の前）。"""

    floor: np.ndarray  # (N, 2)
    obstacle: np.ndarray
    high: np.ndarray
    spread_m: float
    source: str


def local_cloud(w: Window, i: int, params: MapParams | None = None) -> LocalCloud | None:
    """窓（solve_window(keep=True)）の i 番目のステーションの点を、そのステーションの座標で。
    分け方（床・障害物・上）と、遠い点・確からしさの低い点を捨てるのは build_map と同じ。
    """
    from app.snoo_mapper import camera_centers, station_inliers

    r = w.map_result
    if r is None or not w.good(i):
        return None
    params = params or MapParams()
    F = len(w.refs)
    pts = w.points.reshape(F, -1, 3)
    cf = w.conf.reshape(F, -1)
    finite = np.isfinite(pts).all(-1) & (cf > 0)
    good = finite & (cf >= np.percentile(cf[finite], params.conf_percentile))
    cams_w = r.world.apply(camera_centers(w.extrinsic))
    idx = [k for k, ref in enumerate(w.refs) if ref.station == i]
    idx = [k for k, ok in zip(idx, station_inliers(cams_w[idx]), strict=True) if ok]
    pose = w.results[i].pose
    p = r.world.apply(pts[idx][good[idx]])
    p = p[np.hypot(p[:, 0] - pose.x, p[:, 1] - pose.y) <= params.max_range_m]
    q = xy_to_frame(inverse(pose), p)
    z = q[:, 2]
    top = params.car_height_m + params.obstacle_margin_m
    return LocalCloud(
        q[np.abs(z) <= params.floor_tol_m, :2].astype(np.float32),
        q[(z > params.floor_tol_m) & (z <= top), :2].astype(np.float32),
        q[z > top, :2].astype(np.float32),
        w.results[i].spread_m,
        w.source,
    )


def observation(cloud: LocalCloud, pose: Pose, sonar_m: float | None = None) -> Observation:
    """ステーションの点を pose に置き、格子のセルにした見たもの（session_grid に渡す形）。"""
    from app.roommap import Grid

    grid = Grid()
    cells = []
    for xy in (cloud.floor, cloud.obstacle, cloud.high):
        q = xy_to_frame(pose, xy) if len(xy) else xy.reshape(-1, 2)
        ix, iy = grid.cell(q[:, 0], q[:, 1])
        ok = grid.inside(ix, iy)
        if not ok.any():
            cells.append(np.zeros((0, 2)))
            continue
        c, n = np.unique(np.stack([ix[ok], iy[ok]], 1), axis=0, return_counts=True)
        c = c[n >= MIN_POINTS_PER_CELL]
        cells.append(np.stack(grid.center(c[:, 0], c[:, 1]), 1).reshape(-1, 2))
    return Observation(*cells, pose, sonar_m)
