"""家の地図（ホームマップ）。探索のセッションごとの間取りを 1 つの座標（家の座標）に重ねて残す。

探索のセッションは、どれも走り始めた所を原点 (0, 0, 0) にする。同じ部屋を走っても、セッションが
違えば座標がばらばらで、前に見た所をまた見に行く。そこで、セッションの間取り（5 cm の占有格子）を
家の地図に合わせ（位置合わせ）、合ったセッションの見たものを家の地図に足していく。何回も走るうちに
家じゅうが埋まる。

- 家の座標は、最初に足したセッションの座標にする（そのセッションの変換は (0, 0, 0)）
- 位置合わせは格子どうしの照合（register）。向きを 5° ごとに回し、床と障害物の重なりを
  cv2.matchTemplate（FFT の相関）で全部のずらし方について数える → いちばん良い向きの近くを
  1° ごとに見直す → 障害物のセルで ICP のように詰める。2 番目に良い（別の）合わせ方との差が
  小さい・重なりが少ない・間取りが小さいときは「分からない」として断る（間違えて重ねると
  家の地図が壊れる）
- 格子で決められない（ambiguous）ときは、近いステーションの画像を ORB で比べて選ぶ（任意）。
  2026-10-30 の実走のセッションどうしでは ORB の対応がほとんど取れず、まだ役に立っていない
- 合わなかったセッションは「島」（islands/）に残し、あとで別のセッションを足したときに
  島ごともう一度家の地図に合わせてみる（間に足したセッションで重なりが増えれば合う）
- 家具を動かしたときに直せるよう、log-odds は ±CAP で抑え、新しいセッションが見たセルは
  前の値を DECAY 倍してから足す（古い見え方より新しい見え方を重く見る）

置き場（<root>/）：
- home.npz：家の格子（logodds・high）と、いま走っているセッションを足す前の格子（base_*）
- home.json：セッションごとの変換（session → home の x, y, yaw）と点数・重なり、島の一覧
- home.png：上から見た家の地図（人が見る用）
- islands/<島>.npz：まだ家に合わないセッション（の集まり）の格子（島の 1 つ目のセッションの座標）

地図のサーバー（app/snoo_map_server.py）は SNOO_HOME_MAP=1 のときだけこれを使う（on_station）。
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app import atomic
from app.roommap import FREE, OCCUPIED, Grid, Pose
from app.snoo_mapper import Observation, grid_png, pose_dict, session_grid, to_frame

log = logging.getLogger("snoo_home")

# 家の格子の大きさ（m）。セッションの格子（12 m）を、どの向き・どこに置いても収まる大きさ
HOME_SIZE_M = 24.0
# 床・障害物とみなす log-odds（roommap.Grid.state と同じ）
FREE_LO = -0.4
OCC_LO = 0.4
# 家の格子の log-odds の上限。セッションは ±4 まで積むが、家ではこれで抑え、家具を動かしたら
# 新しいセッション 1〜2 回で入れ替わるようにする
CAP = 3.0
# 新しいセッションが見たセルは、前の値をこれだけ掛けてから足す
DECAY = 0.5

# --- 位置合わせの決まり ---
COARSE_YAW_DEG = 5.0
FINE_YAW_DEG = 1.0
# 障害物の重なりは、家の障害物からの距離でぼかして数える（セル。VGGT の障害物は 1〜2 セルぶれる）
OCC_SIGMA_CELLS = 1.0
# 点数の重み：障害物が障害物に重なる・床が床に重なる・食い違う（床が障害物に、障害物が床に）。
# 今日の実走の 3 セッション（同じ部屋）で比べて決めた（2026-10-30）。床の重なりを重くすると、
# 広い床どうしをどこに置いても点数が近くなり、2 番目との差が無くなる。食い違いを重くすると、
# VGGT の障害物のぶれ（ベッドの下・椅子の脚）で正しい合わせ方の点数まで下がる（0.1 前後）。
# 「上」の層（机の天板・棚）を足しても、差は広がらなかった
W_OCC = 1.0
W_FREE = 0.1
W_CONFLICT = 0.3
# これより見たセル（床と障害物）が少ない間取りは合わせない（2.5 m²）。2026-10-30 の実走では、
# 1〜2 か所目までの間取り（370〜900 セル）は、点数と差が足りていても、全部のステーションで
# 合わせた答えと 1 m・120° 食い違うことがあった。1500 セルを超えると食い違わなくなった
MIN_SRC_CELLS = 1000
# 合ったとするのは：点数がこれ以上・重なり（セッションの見たセルのうち、家でも見ているセルの割合）
# がこれ以上・2 番目に良い別の合わせ方より、これだけの割合で上回るとき。
# 作り物の部屋では、正しい合わせ方は点数 0.73〜0.79・差 0.23〜0.28。2026-10-30 の実走では、
# 互いに食い違わない（A→B と B→A が逆になる）合わせ方が点数 0.40 前後・差 0.14 だった。
# 差 0.108 で合ったもの（185508 を 2 か所目までで合わせた）は、ほかの合わせ方と 45 cm・
# 11° 食い違ったので、差は 0.12 からにする
MIN_SCORE = 0.30
MIN_OVERLAP = 0.30
MIN_MARGIN = 0.12
# 2 番目に良い合わせ方は、いちばん良いものとこれだけ違うもの（セッションの真ん中の行き先か向き）
APART_M = 0.30
APART_DEG = 20.0
# ICP で詰めるのは、障害物のセルの対応がこの距離（m）以内のものだけ
ICP_MATCH_M = 0.10
ICP_ITERS = 15

# 見た目で確かめるとき（visual_check）、近くのステーションの画像どうしの ORB の対応がこれ以上で、
# もう一方の合わせ方の 2 倍以上なら、そちらを選ぶ
MIN_ORB_INLIERS = 25
# 比の検定を通った対応がこれより少なければ、基礎行列を求めない
MIN_ORB_MATCHES = 15


def _masks(lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return lo < FREE_LO, lo > OCC_LO


def _bbox(known: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    idx = np.argwhere(known)
    if not len(idx):
        return None
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, known.shape)
    return slice(lo[0], hi[0]), slice(lo[1], hi[1])


def _odd(b: slice, n: int) -> slice:
    """辺の長さを奇数にする（1 つ広げる。端なら前へ広げる）。"""
    if (b.stop - b.start) % 2:
        return b
    return slice(b.start, b.stop + 1) if b.stop < n else slice(b.start - 1, b.stop)


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class Match:
    """位置合わせの答え。pose はセッションの座標 → 家の座標（to_frame の anchor と同じ形）。"""

    ok: bool
    pose: Pose
    score: float = 0.0
    second: float = 0.0
    margin: float = 0.0
    overlap: float = 0.0
    reason: str = ""
    seconds: float = 0.0
    visual: dict | None = None
    reversed: bool = False  # 家のほうをセッションに合わせて逆にした

    def to_dict(self) -> dict:
        out = {
            "ok": self.ok,
            "pose": pose_dict(self.pose),
            "score": round(self.score, 3),
            "second": round(self.second, 3),
            "margin": round(self.margin, 3),
            "overlap": round(self.overlap, 3),
            "reason": self.reason,
            "seconds": round(self.seconds, 3),
        }
        if self.visual is not None:
            out["visual"] = self.visual
        if self.reversed:
            out["reversed"] = True
        return out


@dataclass
class _Peak:
    score: float
    overlap: float
    yaw: float  # rad
    pose: Pose
    center: tuple[float, float]  # セッションの見た所の真ん中が、家のどこに来るか


class _Matcher:
    """セッションの格子（src）を家の格子（dst）に合わせる。向きごとに matchTemplate を 3 回
    （障害物の列・床の列・見たかどうか）回し、全部のずらし方の点数を一度に出す。"""

    def __init__(self, src: Grid, dst: Grid):
        if abs(src.res - dst.res) > 1e-9:
            raise ValueError("格子の大きさが違う")
        self.res = src.res
        self.src, self.dst = src, dst
        sf, so = _masks(src.logodds)
        known = sf | so
        self.n_src = int(known.sum())
        box = _bbox(known, 2)
        self.ok = box is not None
        if not self.ok:
            return
        # 切り出しの辺を奇数にする（真ん中がセルの中心に来て、回さないとき補間でぼけない）
        box = tuple(_odd(b, n) for b, n in zip(box, sf.shape, strict=True))
        self.sf = sf[box].astype(np.float32)
        self.so = so[box].astype(np.float32)
        self.sk = known[box].astype(np.float32)
        self.n_free, self.n_occ = float(self.sf.sum()), float(self.so.sum())
        hs, ws = self.sf.shape
        self.s_center_idx = (box[0].start + (hs - 1) / 2, box[1].start + (ws - 1) / 2)
        self.L = int(math.ceil(math.hypot(hs, ws))) | 1  # 奇数
        # 点で確かめる（ICP・細かい点数）ための、src の床・障害物のセルの中心（セッションの座標）
        fx, fy = np.nonzero(sf)
        ox, oy = np.nonzero(so)
        self.free_xy = np.stack(src.center(fx, fy), 1)
        self.occ_xy = np.stack(src.center(ox, oy), 1)

        df, do = _masks(dst.logodds)
        dbox = _bbox(df | do, 2)
        self.ok = dbox is not None
        if not self.ok:
            return
        self.dbox = dbox
        df, do = df[dbox], do[dbox]
        dist = cv2.distanceTransform((~do).astype(np.uint8), cv2.DIST_L2, 5)
        occ_field = np.exp(-(dist**2) / (2 * OCC_SIGMA_CELLS**2)).astype(np.float32)
        free = df.astype(np.float32)
        occ = do.astype(np.float32)
        self.d_occ_field, self.d_free, self.d_occ = occ_field, free, occ
        P = self.L
        self.P = P

        def pad(a):
            return cv2.copyMakeBorder(a, P, P, P, P, cv2.BORDER_CONSTANT, value=0)

        # src の障害物と組む列・src の床と組む列・見たかどうか
        self.img_occ = pad(W_OCC * occ_field - W_CONFLICT * free)
        self.img_free = pad(W_FREE * free - W_CONFLICT * occ)
        self.img_known = pad((df | do).astype(np.float32))
        self.norm = max(W_OCC * self.n_occ + W_FREE * self.n_free, 1.0)
        # ICP 用：家の障害物のいちばん近いセル
        _, labels = cv2.distanceTransformWithLabels(
            (~do).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        zx, zy = np.nonzero(do)
        # DIST_LABEL_PIXEL は障害物のセルごとに番号を付ける。番号 → そのセルの [ix, iy]
        lut = np.zeros((labels.max() + 1, 2), np.float64)
        lut[labels[zx, zy]] = np.stack([zx, zy], 1)
        self.nearest_lut, self.nearest_labels = lut, labels
        self.has_dst_occ = len(zx) > 0

    # --- 向きごとの全部のずらし方 ---

    def _rotate(self, a: np.ndarray, yaw: float) -> np.ndarray:
        """[ix, iy] の配列を、真ん中の周りに yaw（左回りが +）回して L×L に置く。"""
        hs, ws = a.shape
        c, s = math.cos(yaw), math.sin(yaw)
        cu, cv = (ws - 1) / 2, (hs - 1) / 2  # cv2 の (列, 行) = (iy, ix)
        cL = (self.L - 1) / 2
        M = np.array([[c, s, cL - c * cu - s * cv], [-s, c, cL + s * cu - c * cv]], np.float64)
        return cv2.warpAffine(a, M, (self.L, self.L), flags=cv2.INTER_LINEAR, borderValue=0)

    def scores(self, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        """(点数, 重なり) をずらし方ごとに。点数は -1〜1 ほど（1 が全部合う）。"""
        so, sf, sk = (self._rotate(a, yaw) for a in (self.so, self.sf, self.sk))
        score = cv2.matchTemplate(self.img_occ, so, cv2.TM_CCORR)
        score += cv2.matchTemplate(self.img_free, sf, cv2.TM_CCORR)
        overlap = cv2.matchTemplate(self.img_known, sk, cv2.TM_CCORR)
        return score / self.norm, overlap / max(float(sk.sum()), 1.0)

    def peak_pose(self, yaw: float, i: int, j: int) -> tuple[Pose, tuple[float, float]]:
        """matchTemplate の (i, j) を、セッション → 家の変換にする。"""
        cL = (self.L - 1) / 2
        hx = self.dbox[0].start + i + cL - self.P
        hy = self.dbox[1].start + j + cL - self.P
        home_c = (self.dst.origin + (hx + 0.5) * self.res, self.dst.origin + (hy + 0.5) * self.res)
        sx = self.src.origin + (self.s_center_idx[0] + 0.5) * self.res
        sy = self.src.origin + (self.s_center_idx[1] + 0.5) * self.res
        c, s = math.cos(yaw), math.sin(yaw)
        t = (home_c[0] - (c * sx - s * sy), home_c[1] - (s * sx + c * sy))
        return Pose(t[0], t[1], math.atan2(s, c)), home_c

    def peaks(self, yaw: float, k: int = 2) -> list[_Peak]:
        """この向きでいちばん良いずらし方と、それから APART_M 以上離れた次に良いもの。"""
        score, overlap = self.scores(yaw)
        # 重なりがごく少ないずらし方は、見ていない所に置いただけなので数えない
        score = np.where(overlap >= MIN_OVERLAP / 2, score, -np.inf).astype(np.float32)
        out = []
        r = int(math.ceil(APART_M / self.res))
        for _ in range(k):
            i, j = np.unravel_index(int(np.argmax(score)), score.shape)
            v = float(score[i, j])
            if not math.isfinite(v):
                break
            pose, c = self.peak_pose(yaw, int(i), int(j))
            out.append(_Peak(v, float(overlap[i, j]), yaw, pose, c))
            score[max(i - r, 0) : i + r + 1, max(j - r, 0) : j + r + 1] = -np.inf
        return out

    # --- 点で確かめる ---

    def point_score(self, pose: Pose) -> float:
        """pose で src の床・障害物のセルの中心を家に置いたときの点数（scores と同じ数え方）。"""

        def sample(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
            if not len(xy):
                return np.zeros(0, np.float32)
            p = _apply(pose, xy)
            fx = (p[:, 0] - self.dst.origin) / self.res - 0.5 - self.dbox[0].start
            fy = (p[:, 1] - self.dst.origin) / self.res - 0.5 - self.dbox[1].start
            m = cv2.remap(
                field,
                fy.astype(np.float32)[None, :],
                fx.astype(np.float32)[None, :],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return m.ravel()

        occ_term = W_OCC * sample(self.d_occ_field, self.occ_xy) - W_CONFLICT * sample(
            self.d_free, self.occ_xy
        )
        free_term = W_FREE * sample(self.d_free, self.free_xy) - W_CONFLICT * sample(
            self.d_occ, self.free_xy
        )
        return float((occ_term.sum() + free_term.sum()) / self.norm)

    def polish(self, pose: Pose) -> Pose:
        """点で数えた点数（point_score）が上がる向きへ、半セル・0.5° から刻みを半分ずつにして
        少しずつ動かす（セルより細かく合わせる）。"""
        best = self.point_score(pose)
        step, turn = self.res / 2, math.radians(0.5)
        while step >= 0.005:
            moved = False
            for dx, dy, dyaw in (
                (step, 0, 0),
                (-step, 0, 0),
                (0, step, 0),
                (0, -step, 0),
                (0, 0, turn),
                (0, 0, -turn),
            ):
                q = Pose(pose.x + dx, pose.y + dy, pose.yaw + dyaw)
                v = self.point_score(q)
                if v > best + 1e-6:
                    pose, best, moved = q, v, True
            if not moved:
                step, turn = step / 2, turn / 2
        return Pose(
            float(pose.x), float(pose.y), math.atan2(math.sin(pose.yaw), math.cos(pose.yaw))
        )

    def icp(self, pose: Pose) -> Pose:
        """障害物のセルを、家のいちばん近い障害物のセルに寄せて詰める（ICP。2 次元の剛体）。"""
        if not self.has_dst_occ or len(self.occ_xy) < 10:
            return pose
        for _ in range(ICP_ITERS):
            p = _apply(pose, self.occ_xy)
            ix = np.floor((p[:, 0] - self.dst.origin) / self.res).astype(int) - self.dbox[0].start
            iy = np.floor((p[:, 1] - self.dst.origin) / self.res).astype(int) - self.dbox[1].start
            h, w = self.nearest_labels.shape
            inside = (ix >= 0) & (ix < h) & (iy >= 0) & (iy < w)
            if inside.sum() < 10:
                return pose
            near = self.nearest_lut[self.nearest_labels[ix[inside], iy[inside]]]
            qx = self.dst.origin + (near[:, 0] + self.dbox[0].start + 0.5) * self.res
            qy = self.dst.origin + (near[:, 1] + self.dbox[1].start + 0.5) * self.res
            q = np.stack([qx, qy], 1)
            a = self.occ_xy[inside]
            d = np.hypot(*(q - p[inside]).T)
            keep = d <= ICP_MATCH_M
            if keep.sum() < 10:
                return pose
            # 近い 80% だけ（外れた障害物に引っ張られない）
            keep &= d <= np.quantile(d[keep], 0.8) + 1e-9
            new = _rigid(a[keep], q[keep])
            moved = (
                math.hypot(new.x - pose.x, new.y - pose.y)
                + abs(_wrap_deg(math.degrees(new.yaw - pose.yaw))) * 0.01
            )
            pose = new
            if moved < 1e-4:
                break
        return pose


def _apply(pose: Pose, xy: np.ndarray) -> np.ndarray:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return np.stack([pose.x + c * xy[:, 0] - s * xy[:, 1], pose.y + s * xy[:, 0] + c * xy[:, 1]], 1)


def _rigid(a: np.ndarray, b: np.ndarray) -> Pose:
    """a を b に重ねる 2 次元の回転と平行移動（最小二乗。Kabsch）。"""
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    c, s = math.cos(yaw), math.sin(yaw)
    t = cb - np.array([c * ca[0] - s * ca[1], s * ca[0] + c * ca[1]])
    return Pose(float(t[0]), float(t[1]), yaw)


def _apart(a: _Peak, b: _Peak) -> bool:
    d = math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
    return d > APART_M or abs(_wrap_deg(math.degrees(a.yaw - b.yaw))) > APART_DEG


def register(src: Grid, dst: Grid) -> tuple[Match, list[_Peak]]:
    """セッションの格子 src を家の格子 dst に合わせる。(答え, 良い順の候補) を返す。

    点数は src の見たセルで割るので、小さいほうを大きいほうに合わせる。src のほうが大きければ
    （家がまだ小さいとき）、dst を src に合わせて逆にする。2026-10-30 の実走では、
    大きいセッションを小さい家に合わせると点数 0.16 で合わず、逆向きなら 0.40 で合った。
    """
    ns = int(((src.logodds < FREE_LO) | (src.logodds > OCC_LO)).sum())
    nd = int(((dst.logodds < FREE_LO) | (dst.logodds > OCC_LO)).sum())
    if ns <= nd:
        return _register(src, dst)
    match, cands = _register(dst, src)
    match.pose = inverse(match.pose)
    match.reversed = True
    return match, [_with_pose(c, inverse(c.pose)) for c in cands]


def _register(src: Grid, dst: Grid) -> tuple[Match, list[_Peak]]:
    """src を dst に合わせる（register の中身）。

    向きを COARSE_YAW_DEG ごとに回して全部のずらし方の点数を出し、いちばん良いものと、
    それと違う（APART_M・APART_DEG 以上離れた）2 番目を選ぶ。どちらもその向きの前後を
    FINE_YAW_DEG ごとに見直し、いちばん良いものを ICP で詰める。
    """
    t0 = time.perf_counter()
    m = _Matcher(src, dst)
    if not m.ok:
        return Match(False, Pose(0, 0, 0), reason="empty"), []
    if m.n_src < MIN_SRC_CELLS:
        return Match(False, Pose(0, 0, 0), reason=f"too_small:{m.n_src}"), []
    coarse = []
    for deg in np.arange(0.0, 360.0, COARSE_YAW_DEG):
        coarse += m.peaks(math.radians(deg))
    if not coarse:
        return Match(
            False, Pose(0, 0, 0), reason="no_overlap", seconds=time.perf_counter() - t0
        ), []
    coarse.sort(key=lambda p: -p.score)
    best = coarse[0]
    second = next((p for p in coarse[1:] if _apart(best, p)), None)

    def refine(p: _Peak, avoid: _Peak | None = None) -> _Peak:
        """p の向きの前後を細かく見直す。avoid（いちばん良いもの）の近くへは寄せない。"""
        out = p
        span = COARSE_YAW_DEG - FINE_YAW_DEG
        for d in np.arange(-span, span + 1e-9, FINE_YAW_DEG):
            for q in m.peaks(p.yaw + math.radians(d), k=2):
                if q.score > out.score and not _apart(p, q):
                    if avoid is None or _apart(avoid, q):
                        out = q
        return out

    best = refine(best)
    if second is not None:
        second = refine(second, best)
    # 細かく詰める（ICP）。点数が下がるなら使わない
    pose = best.pose
    fine = m.icp(pose)
    base_score = m.point_score(pose)
    fine_score = m.point_score(fine)
    shift = math.hypot(fine.x - pose.x, fine.y - pose.y)
    turn = abs(_wrap_deg(math.degrees(fine.yaw - pose.yaw)))
    if fine_score >= base_score and shift <= 2 * m.res and turn <= 2 * FINE_YAW_DEG:
        pose = fine
    pose = m.polish(pose)
    s2 = second.score if second is not None else 0.0
    margin = (best.score - s2) / max(best.score, 1e-6)
    match = Match(True, pose, best.score, s2, margin, best.overlap, "", time.perf_counter() - t0)
    if best.score < MIN_SCORE:
        match.ok, match.reason = False, "low_score"
    elif best.overlap < MIN_OVERLAP:
        match.ok, match.reason = False, "small_overlap"
    elif margin < MIN_MARGIN:
        match.ok, match.reason = False, "ambiguous"
    cands = [_with_pose(best, pose)] + ([second] if second is not None else [])
    return match, cands


def _with_pose(p: _Peak, pose: Pose) -> _Peak:
    return _Peak(p.score, p.overlap, p.yaw, pose, p.center)


# --- 格子を別の座標に移す ---


def warp(
    src: np.ndarray, src_origin: float, pose: Pose, dst_shape, dst_origin: float, res: float
) -> np.ndarray:
    """src（座標 A の格子）を、pose（A → B）で座標 B の格子（dst_shape・dst_origin）に移す。
    B のセルの中心ごとに、A のどのセルに入るかを引く（いちばん近いセル）。外は 0。
    cv2.remap の INTER_NEAREST は、ちょうどセルの中心に当たると誤差で隣のセルを引くので使わない。"""
    n0, n1 = dst_shape
    ix, iy = np.meshgrid(np.arange(n0), np.arange(n1), indexing="ij")
    bx = dst_origin + (ix + 0.5) * res
    by = dst_origin + (iy + 0.5) * res
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    dx, dy = bx - pose.x, by - pose.y
    sx = np.floor((c * dx + s * dy - src_origin) / res + 1e-6).astype(np.int64)
    sy = np.floor((-s * dx + c * dy - src_origin) / res + 1e-6).astype(np.int64)
    ok = (sx >= 0) & (sx < src.shape[0]) & (sy >= 0) & (sy < src.shape[1])
    out = np.zeros(dst_shape, src.dtype)
    out[ok] = src[sx[ok], sy[ok]]
    return out


def inverse(p: Pose) -> Pose:
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    return Pose(-(c * p.x + s * p.y), -(-s * p.x + c * p.y), -p.yaw)


# --- 家の地図 ---


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


class HomeMap:
    """家の地図の置き場（<root>/home.npz・home.json・home.png・islands/）。

    セッションを足すときは、足す前の格子（base）を残しておく。同じセッションを（ステーションごとに）
    足し直すときは base から足し直すので、何回足しても 1 回分になる。
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        g = Grid(HOME_SIZE_M)
        self.grid = g
        self.high = np.zeros(g.logodds.shape, np.int16)
        self.base_lo = g.logodds.copy()
        self.base_high = self.high.copy()
        self.open_session: str | None = None
        self.sessions: dict[str, dict] = {}
        self.islands: dict[str, dict] = {}

    # --- 読み書き ---

    @classmethod
    def load(cls, root: Path | str) -> HomeMap:
        h = cls(root)
        npz, js = h.root / "home.npz", h.root / "home.json"
        if js.is_file():
            info = json.loads(js.read_text())
            h.sessions = {s["id"]: s for s in info.get("sessions", [])}
            h.islands = {s["id"]: s for s in info.get("islands", [])}
            h.open_session = info.get("open_session")
        if npz.is_file():
            d = np.load(npz)
            h.grid.logodds[:] = d["logodds"]
            h.high[:] = d["high"]
            h.base_lo[:] = d["base_logodds"]
            h.base_high[:] = d["base_high"]
        return h

    def info(self) -> dict:
        g = self.grid
        st = g.state()
        return {
            "schema_version": 1,
            "resolution_cm": round(g.res * 100, 3),
            "origin": [g.origin, g.origin],
            "size": [g.n, g.n],
            "frame": "家の座標。最初に足したセッションの座標（そのセッションの始めが原点）",
            "free_m2": round(float((st == FREE).sum()) * g.res**2, 2),
            "occupied_cells": int((st == OCCUPIED).sum()),
            "open_session": self.open_session,
            "sessions": sorted(self.sessions.values(), key=lambda s: s["id"]),
            "islands": sorted(self.islands.values(), key=lambda s: s["id"]),
        }

    def save(self, grid: bool = True) -> None:
        """home.json・home.npz・home.png を書く（書きかけを読ませない。app/atomic.py）。"""
        self.root.mkdir(parents=True, exist_ok=True)
        if grid:
            atomic.write_bytes(
                self.root / "home.npz",
                _npz_bytes(
                    logodds=self.grid.logodds.astype(np.float16),
                    high=self.high,
                    base_logodds=self.base_lo.astype(np.float16),
                    base_high=self.base_high,
                ),
            )
            ok, png = cv2.imencode(".png", self.render())
            if ok:
                atomic.write_bytes(self.root / "home.png", png.tobytes())
        atomic.write_text(
            self.root / "home.json", json.dumps(self.info(), ensure_ascii=False, indent=1)
        )

    def empty(self) -> bool:
        return not self.sessions

    # --- 合わせる・足す ---

    def register(self, grid: Grid) -> tuple[Match, list[_Peak]]:
        """セッションの格子を家に合わせる。家が空なら、そのセッションを家の座標にする。"""
        if self.empty():
            return Match(True, Pose(0.0, 0.0, 0.0), 1.0, 0.0, 1.0, 1.0, "anchor"), []
        return register(grid, self.grid)

    def _session_in_home(self, grid: Grid, high: np.ndarray | None, pose: Pose):
        g = self.grid
        lo = warp(grid.logodds, grid.origin, pose, g.logodds.shape, g.origin, g.res)
        hi = None
        if high is not None:
            hi = warp(high.astype(np.float32), grid.origin, pose, g.logodds.shape, g.origin, g.res)
        return lo, hi

    @staticmethod
    def _fold(lo: np.ndarray, high: np.ndarray, s_lo: np.ndarray, s_hi: np.ndarray | None):
        seen = s_lo != 0
        out = np.where(seen, lo * DECAY, lo) + s_lo
        np.clip(out, -CAP, CAP, out=out)
        h = high
        if s_hi is not None:
            h = np.where(seen | (s_hi > 0), high // 2, high) + s_hi.astype(np.int32)
            h = np.clip(h, 0, 255).astype(np.int16)
        return out.astype(np.float32), h

    def merge(
        self,
        sid: str,
        grid: Grid,
        high: np.ndarray | None,
        pose: Pose,
        match: Match | None = None,
        session_dir: Path | None = None,
    ) -> None:
        """セッションの格子を家に足す。いま足している（open の）セッションなら、足す前の格子
        （base）から足し直す。別のセッションなら、今の格子を base にしてから足す。"""
        s_lo, s_hi = self._session_in_home(grid, high, pose)
        if sid != self.open_session:
            self.base_lo = self.grid.logodds.copy()
            self.base_high = self.high.copy()
            self.open_session = sid
        self.grid.logodds[:], self.high = self._fold(self.base_lo, self.base_high, s_lo, s_hi)
        self._remember(sid, grid, pose, match, session_dir)

    def merge_closed(
        self,
        sid: str,
        grid: Grid,
        high: np.ndarray | None,
        pose: Pose,
        match: Match | None = None,
        session_dir: Path | None = None,
    ) -> None:
        """終わったセッション（島）を足す。いま足しているセッションの足し直しで消えないよう、
        base にも足す。"""
        s_lo, s_hi = self._session_in_home(grid, high, pose)
        self.base_lo, self.base_high = self._fold(self.base_lo, self.base_high, s_lo, s_hi)
        self.grid.logodds[:], self.high = self._fold(self.grid.logodds, self.high, s_lo, s_hi)
        self._remember(sid, grid, pose, match, session_dir)

    def _remember(self, sid, grid: Grid, pose: Pose, match: Match | None, session_dir) -> None:
        st = grid.state()
        entry = self.sessions.get(sid, {"id": sid, "added": time.strftime("%Y-%m-%dT%H:%M:%S")})
        entry.update(
            pose=pose_dict(pose),
            free_m2=round(float((st == FREE).sum()) * grid.res**2, 2),
            updated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        if match is not None:
            entry["match"] = match.to_dict()
        if session_dir is not None:
            entry["dir"] = str(Path(session_dir).resolve())
        self.sessions[sid] = entry
        if sid in self.islands and set(self.islands[sid]["members"]) == {sid}:
            self._drop_island(sid)

    # --- 島 ---
    # 島は、家に合わないセッションの集まり。1 つ目のセッションの座標で格子を持ち
    # （islands/<島>.npz）、
    # members に、島に入ったセッションの島の座標での変換を持つ。島が家に合ったら、島ごと家に足す

    def _drop_island(self, iid: str) -> None:
        self.islands.pop(iid, None)
        (self.root / "islands" / f"{iid}.npz").unlink(missing_ok=True)

    def _write_island(self, iid: str, grid: Grid, high: np.ndarray) -> None:
        d = self.root / "islands"
        d.mkdir(parents=True, exist_ok=True)
        atomic.write_bytes(
            d / f"{iid}.npz",
            _npz_bytes(
                logodds=grid.logodds.astype(np.float16),
                high=high.astype(np.int16),
                origin=np.array(grid.origin),
                res=np.array(grid.res),
            ),
        )

    def island_grid(self, iid: str) -> tuple[Grid, np.ndarray]:
        d = np.load(self.root / "islands" / f"{iid}.npz")
        size = float(d["logodds"].shape[0] * float(d["res"]))
        g = Grid(size, float(d["res"]))
        g.origin = float(d["origin"])
        g.logodds[:] = d["logodds"].astype(np.float32)
        return g, d["high"].astype(np.int16)

    def island_of(self, sid: str) -> str | None:
        return next((k for k, v in self.islands.items() if sid in v["members"]), None)

    def add_island(
        self,
        sid: str,
        grid: Grid,
        high: np.ndarray | None,
        match: Match | None,
        session_dir: Path | None = None,
        join: bool = False,
        visual: bool = False,
    ) -> str:
        """合わなかったセッションを島に入れる（あとで connect_islands で家に合わせ直す）。
        島の名前を返す。

        地図のサーバーは、走っているセッションを自分だけの島にして、ステーションごとに置き換える。
        join なら、先に今ある島に合わせてみて、合えばその島に足す（終わったセッションをまとめて
        足すとき。家と違う部屋を何回か走ったら、その部屋の島が育つ）。
        """
        hi = np.zeros(grid.logodds.shape, np.int16) if high is None else high.astype(np.int16)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        dirs = {sid: str(Path(session_dir).resolve())} if session_dir is not None else {}
        info = match.to_dict() if match is not None else None
        iid = self.island_of(sid)
        if iid is not None and set(self.islands[iid]["members"]) == {sid}:
            self._write_island(iid, *self._to_island_frame(grid, hi, Pose(0.0, 0.0, 0.0)))
            self.islands[iid].update(updated=now, match=info)
            return iid
        if join and iid is None:
            for k in sorted(self.islands, key=lambda k: -len(self.islands[k]["members"])):
                ig, ih = self.island_grid(k)
                m, cands = register(grid, ig)
                if visual:
                    m = verify(m, cands, session_dir, island_stations(self.islands[k]))
                if m.ok:
                    lo, h2 = self._to_island_frame(grid, hi, m.pose)
                    ig.logodds[:], ih = self._fold(ig.logodds, ih, lo.logodds, h2)
                    self._write_island(k, ig, ih)
                    self.islands[k]["members"][sid] = pose_dict(m.pose)
                    self.islands[k]["dirs"].update(dirs)
                    self.islands[k]["updated"] = now
                    self.islands[k].setdefault("joins", {})[sid] = m.to_dict()
                    return k
        iid = iid or sid
        self._write_island(iid, *self._to_island_frame(grid, hi, Pose(0.0, 0.0, 0.0)))
        self.islands[iid] = {
            "id": iid,
            "members": {sid: pose_dict(Pose(0.0, 0.0, 0.0))},
            "dirs": dirs,
            "updated": now,
            "match": info,
        }
        return iid

    @staticmethod
    def _to_island_frame(grid: Grid, high: np.ndarray, pose: Pose) -> tuple[Grid, np.ndarray]:
        """島の格子は家と同じ大きさ（セッションをいくつか足しても収まる）。"""
        g = Grid(HOME_SIZE_M, grid.res)
        g.logodds[:] = warp(grid.logodds, grid.origin, pose, g.logodds.shape, g.origin, g.res)
        h = warp(high.astype(np.int16), grid.origin, pose, g.logodds.shape, g.origin, g.res)
        return g, h

    def connect_islands(self, skip: str | None = None, visual: bool = False) -> list[str]:
        """島をもう一度家に合わせ、合ったものを島ごと足す。家に入ったセッションの名前を返す。

        1 つ合うと家が広がって、ほかの島も合うことがあるので、合うものが無くなるまで繰り返す。
        skip のセッションを含む島（走っている途中のセッション）は合わせない。
        """
        joined: list[str] = []
        changed = True
        while changed:
            changed = False
            for iid in sorted(self.islands):
                isl = self.islands[iid]
                if skip in isl["members"] or not (self.root / "islands" / f"{iid}.npz").is_file():
                    continue
                grid, high = self.island_grid(iid)
                match, cands = register(grid, self.grid)
                if visual:
                    match = self._verify_island(isl, match, cands)
                isl["match"] = match.to_dict()
                if not match.ok:
                    continue
                s_lo, s_hi = self._session_in_home(grid, high, match.pose)
                self.base_lo, self.base_high = self._fold(self.base_lo, self.base_high, s_lo, s_hi)
                self.grid.logodds[:], self.high = self._fold(
                    self.grid.logodds, self.high, s_lo, s_hi
                )
                for sid, p in isl["members"].items():
                    inner = Pose(p["x"], p["y"], math.radians(p["yaw_deg"]))
                    d = isl["dirs"].get(sid)
                    entry = self.sessions.setdefault(sid, {"id": sid})
                    entry.update(
                        pose=pose_dict(to_frame(match.pose, inner)),
                        match=match.to_dict(),
                        via_island=iid,
                        updated=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                    if d:
                        entry["dir"] = d
                    joined.append(sid)
                self._drop_island(iid)
                changed = True
                break  # 島の一覧が変わったので数え直す
        return joined

    def _verify_island(self, isl: dict, match: Match, cands: list[_Peak]) -> Match:
        """島の答えを、島の 1 つ目のセッションの画像で確かめる（verify）。島の変換を、
        そのセッションの家での変換に直してから比べる。"""
        sid = next((k for k in isl["members"] if isl.get("dirs", {}).get(k)), None)
        if sid is None:
            return match
        inner = _pose_of(isl["members"][sid])
        m = Match(
            match.ok,
            to_frame(match.pose, inner),
            match.score,
            match.second,
            match.margin,
            match.overlap,
            match.reason,
            match.seconds,
        )
        cs = [_with_pose(c, to_frame(c.pose, inner)) for c in cands]
        m = verify(m, cs, Path(isl["dirs"][sid]), home_stations(self))
        # 島の変換に戻す
        match.ok, match.reason, match.visual = m.ok, m.reason, m.visual
        match.pose = to_frame(m.pose, inverse(inner))
        match.score, match.overlap = m.score, m.overlap
        return match

    # --- セッションの座標で使う ---

    def in_session(self, pose: Pose, like: Grid) -> tuple[np.ndarray, np.ndarray]:
        """家の格子（logodds・high）を、セッションの座標の格子（like と同じ形）に移す。
        pose はセッション → 家の変換。"""
        inv = inverse(pose)
        g = self.grid
        lo = warp(g.logodds, g.origin, inv, like.logodds.shape, like.origin, like.res)
        hi = warp(self.high, g.origin, inv, like.logodds.shape, like.origin, like.res)
        return lo, hi

    def union(self, session: Grid, pose: Pose) -> Grid:
        """次の 1 歩を決めるための格子：セッションで見た所はセッションの値、見ていない所は家の値。
        障害物はどちらかで障害物なら障害物（ぶつからない側に倒す）。"""
        home_lo, _ = self.in_session(pose, session)
        out = Grid(session.n * session.res, session.res)
        out.origin = session.origin
        lo = session.logodds
        known = (lo < FREE_LO) | (lo > OCC_LO)
        merged = np.where(known, lo, home_lo)
        occ = home_lo > OCC_LO
        merged = np.where(occ, np.maximum(merged, home_lo), merged)
        out.logodds[:] = merged.astype(np.float32)
        return out

    # --- 絵 ---

    def render(self, px: int = 4, margin_cells: int = 10) -> np.ndarray:
        """上から見た家の地図（白 = 床、黒 = 障害物、灰 = 見ていない）に、セッションの始まりの
        位置（丸と向き）を色を変えて描く。見た所のまわりだけ切り出す。"""
        g = self.grid
        st = g.state()
        known = st != 0
        box = _bbox(known, margin_cells) or (slice(g.n // 2 - 20, g.n // 2 + 20),) * 2
        img = grid_png(st)  # 行 0 が +y の端、列 0 が -x の端
        # grid_png は [ix, iy] を転置して上下を返すので、切り出しも合わせる
        r0, r1 = g.n - box[1].stop, g.n - box[1].start
        c0, c1 = box[0].start, box[0].stop
        crop = cv2.cvtColor(img[r0:r1, c0:c1], cv2.COLOR_GRAY2BGR)
        crop = cv2.resize(crop, None, fx=px, fy=px, interpolation=cv2.INTER_NEAREST)
        colors = [
            (0, 0, 255),
            (0, 160, 0),
            (255, 0, 0),
            (0, 140, 255),
            (200, 0, 200),
            (160, 160, 0),
            (0, 200, 200),
        ]

        def to_px(x, y):
            ix = (x - g.origin) / g.res - c0
            iy = (y - g.origin) / g.res
            row = g.n - iy - r0
            return int(round(ix * px)), int(round(row * px))

        for k, s in enumerate(sorted(self.sessions.values(), key=lambda s: s["id"])):
            p = s["pose"]
            col = colors[k % len(colors)]
            x, y, yaw = p["x"], p["y"], math.radians(p["yaw_deg"])
            a = to_px(x, y)
            b = to_px(x + 0.25 * math.cos(yaw), y + 0.25 * math.sin(yaw))
            cv2.circle(crop, a, 2 * px, col, -1)
            cv2.arrowedLine(crop, a, b, col, 2, tipLength=0.3)
            cv2.putText(
                crop,
                s["id"][-6:],
                (a[0] + 6, a[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                1,
                cv2.LINE_AA,
            )
        return crop


# --- 置き場のセッションを読む ---


def load_session_grid(session: Path, upto: int | None = None) -> tuple[Grid, np.ndarray, str]:
    """セッションの間取り（セッションの座標）と「上」の層と、何から作ったか。

    ふつうはステーションごとの obs.npz を足す（地図のサーバーの session_grid と同じ）。
    obs.npz が無い古いセッション（2026-10-29 まで）は、最後の窓の grid.png を room.json の
    window.origin_pose でセッションの座標に移す（窓の外で見た所は入らない）。
    """
    session = Path(session)
    obs = []
    for d in sorted(session.iterdir()):
        if d.is_dir() and d.name.isdigit() and len(d.name) == 3 and (d / "obs.npz").is_file():
            if upto is None or int(d.name) <= upto:
                obs.append(Observation.load(d / "obs.npz"))
    if obs:
        grid, high = session_grid(obs)
        return grid, high, f"obs.npz x{len(obs)}"
    png, room = session / "grid.png", session / "room.json"
    if upto is None and png.is_file() and room.is_file():
        info = json.loads(room.read_text())
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        state = img[::-1].T  # grid_png の逆
        g = Grid()
        if state.shape != g.logodds.shape or abs(info["resolution_cm"] / 100 - g.res) > 1e-9:
            raise ValueError(f"{png} の大きさが違う")
        g.origin = float(info["origin"][0])
        lo = np.zeros(state.shape, np.float32)
        lo[state == 255] = -1.5
        lo[state == 0] = 2.0
        o = (info.get("window") or {}).get("origin_pose") or {"x": 0, "y": 0, "yaw_deg": 0}
        pose = Pose(o["x"], o["y"], math.radians(o["yaw_deg"]))
        g.logodds[:] = warp(lo, g.origin, pose, lo.shape, g.origin, g.res)
        return g, np.zeros(lo.shape, np.int32), "grid.png（最後の窓だけ）"
    raise FileNotFoundError(f"{session} に obs.npz も grid.png も無い")


# --- 見た目で確かめる（任意） ---


def station_images(session_dir: Path, station: str, limit: int = 5) -> list[np.ndarray]:
    """ステーションの止まって撮ったコマ（チルト 0 の段を先に）を、小さい灰色の画像で。"""
    d = Path(session_dir) / station
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        return []
    frames = [f for f in meta.get("frames", []) if not f.get("moving")]
    frames.sort(key=lambda f: abs(float(f.get("tilt_deg", 0))))
    out = []
    for f in frames[:limit]:
        img = cv2.imread(str(d / f["file"]), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img is not None:
            out.append(img)
    return out


def orb_inliers(a: list[np.ndarray], b: list[np.ndarray]) -> int:
    """2 組の画像の、いちばん良く合う組の ORB の対応の数（比の検定と基礎行列の RANSAC の後）。"""
    orb = cv2.ORB_create(500)
    fa = [orb.detectAndCompute(x, None) for x in a]
    fb = [orb.detectAndCompute(x, None) for x in b]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    best = 0
    for ka, da in fa:
        for kb, db in fb:
            if da is None or db is None or len(da) < 8 or len(db) < 8:
                continue
            pairs = bf.knnMatch(da, db, k=2)
            good = [m for m, *n in pairs if n and m.distance < 0.75 * n[0].distance]
            # 8 点ちょうどだと findFundamentalMat が RANSAC をせず、mask が当てにならない
            # （別の部屋の画像どうしで 382 と数えた）。少ない対応はたまたま合ったものとみなす
            if len(good) < MIN_ORB_MATCHES:
                continue
            pa = np.float32([ka[m.queryIdx].pt for m in good])
            pb = np.float32([kb[m.trainIdx].pt for m in good])
            _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.99)
            if mask is not None and mask.shape[0] == len(good):
                best = max(best, int(np.count_nonzero(mask)))
    return best


Station = tuple[Path, str, Pose]  # (セッションの置き場, ステーションの名前, 位置)


def session_stations(session_dir: Path, pose: Pose) -> list[Station]:
    """セッションのステーション（poses.json）を、pose（セッション → 家か島）で移した位置で。"""
    p = Path(session_dir) / "poses.json"
    if not p.is_file():
        return []
    out = []
    for k, v in json.loads(p.read_text()).items():
        q = Pose(v["x"], v["y"], math.radians(v["yaw_deg"]))
        out.append((Path(session_dir), f"{int(k):03d}", to_frame(pose, q)))
    return out


def _pose_of(d: dict) -> Pose:
    return Pose(d["x"], d["y"], math.radians(d["yaw_deg"]))


def home_stations(home: HomeMap) -> list[Station]:
    """家に足したセッションのステーション（家の座標）。"""
    out = []
    for s in home.sessions.values():
        if s.get("dir"):
            out += session_stations(Path(s["dir"]), _pose_of(s["pose"]))
    return out


def island_stations(isl: dict) -> list[Station]:
    """島に入ったセッションのステーション（島の座標）。"""
    out = []
    for sid, p in isl["members"].items():
        d = isl.get("dirs", {}).get(sid)
        if d:
            out += session_stations(Path(d), _pose_of(p))
    return out


# 見た目で比べる相手は、位置の差（m）+ 向きの差（°）× 0.01 がこれより小さいステーションだけ
NEAR_COST = 1.0
# 見た目で比べるステーションの組の数（近い順）
VISUAL_STATIONS = 3


def _cost(a: Pose, b: Pose) -> float:
    dyaw = abs(_wrap_deg(math.degrees(a.yaw - b.yaw)))
    return math.hypot(a.x - b.x, a.y - b.y) + 0.01 * dyaw


def visual_support(session_dir: Path, pose: Pose, theirs: list[Station]) -> dict | None:
    """pose（新しいセッション → 家）で置いたとき、新しいセッションのステーションの画像と、
    家のステーションの画像が ORB でどれだけ合うか。位置と向きの近い組（NEAR_COST 未満）を
    近い順に VISUAL_STATIONS 組まで比べ、いちばん合った組を返す。
    近い組が無ければ None（見た目では確かめられない）。"""
    mine = session_stations(session_dir, pose)
    if not mine or not theirs:
        return None
    pairs = []
    for m in mine:
        near = min(theirs, key=lambda t, p=m[2]: _cost(t[2], p))
        c = _cost(near[2], m[2])
        if c < NEAR_COST:
            pairs.append((c, m, near))
    pairs.sort(key=lambda x: x[0])
    best: dict | None = None
    for c, m, near in pairs[:VISUAL_STATIONS]:
        n = orb_inliers(station_images(m[0], m[1]), station_images(near[0], near[1]))
        if best is None or n > best["inliers"]:
            best = {
                "station": m[1],
                "near": f"{near[0].name}/{near[1]}",
                "inliers": n,
                "cost": round(c, 2),
            }
    return best


def verify(
    match: Match, cands: list[_Peak], session_dir: Path | None, theirs: list[Station]
) -> Match:
    """格子で決められない（ambiguous）とき、上位 2 つの候補のうち、近いステーションの画像が
    ORB でよく合うほうを選ぶ（任意。画像や近いステーションが無ければ格子の答えのまま）。

    格子で合ったものを画像で断ることはしない。2026-10-30 の実走のセッションどうしでは、
    同じ部屋でも ORB の対応がほとんど 0 だった（車のカメラが低く暗い・向きが違う）。同じ
    セッションの隣のステーションどうしなら 37〜159 ある
    """
    if session_dir is None or not theirs or match.ok or match.reason != "ambiguous":
        return match
    if len(cands) < 2:
        return match
    try:
        vs = [visual_support(session_dir, c.pose, theirs) for c in cands[:2]]
        n = [v["inliers"] if v else 0 for v in vs]
        match.visual = {"candidates": vs}
        for i, j in ((0, 1), (1, 0)):
            if n[i] >= MIN_ORB_INLIERS and n[i] >= 2 * n[j]:
                c = cands[i]
                match.ok, match.reason = True, "visual"
                match.pose, match.score, match.overlap = c.pose, c.score, c.overlap
        return match
    except (OSError, cv2.error, ValueError, KeyError) as e:
        match.visual = {"error": type(e).__name__}
        return match


def register_session(
    home: HomeMap, grid: Grid, session_dir: Path | None = None, visual: bool = False
) -> Match:
    """家に合わせる（格子）。visual なら画像で確かめる（verify）。"""
    match, cands = home.register(grid)
    if match.reason == "anchor" or not visual:
        return match
    return verify(match, cands, session_dir, home_stations(home))


# --- 地図のサーバーから（app/snoo_map_server.py。SNOO_HOME_MAP=1 のとき） ---

# 1 か所目で合わせ、合わなければこのステーションでやり直す（床が増えるほど合いやすい）。
# その後も RETRY_EVERY か所ごとにやり直す（実走では 1〜2 か所目の間取りは小さすぎて合わせない
# ことが多い。合わせるのは 0.2 秒ほど）
RETRY_STATIONS = (0, 1, 2, 4)
RETRY_EVERY = 4


def retry_at(station_id: int) -> bool:
    return station_id in RETRY_STATIONS or (
        station_id > max(RETRY_STATIONS) and station_id % RETRY_EVERY == 0
    )


def read_session_state(session: Path) -> dict:
    p = session / "home.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def on_station(
    home_root: Path, session: Path, station_id: int, grid: Grid, high: np.ndarray | None
) -> tuple[Grid | None, dict]:
    """1 ステーションごとに：まだ合っていなければ（決めたステーションで）家に合わせ、合っていれば
    足し直して、次の 1 歩に使う格子（セッションと家を合わせたもの）を返す。合っていなければ
    (None, 様子)。様子は plan.json に残す。

    セッションの置き場に home.json（変換・点数・やってみた記録）を書く。
    """
    t0 = time.perf_counter()
    sid = session.name
    state = read_session_state(session)
    home = HomeMap.load(home_root)
    newly = False
    if not state.get("registered") and retry_at(station_id):
        match = register_session(home, grid, session, visual=not home.empty())
        state.setdefault("attempts", []).append({"station": station_id, **match.to_dict()})
        if match.ok:
            newly = True
            state.update(
                registered=True,
                pose=pose_dict(match.pose),
                station=station_id,
                score=round(match.score, 3),
                overlap=round(match.overlap, 3),
                reason=match.reason,
            )
        atomic.write_text(session / "home.json", json.dumps(state, ensure_ascii=False, indent=1))
    union = None
    if state.get("registered"):
        p = state["pose"]
        pose = Pose(p["x"], p["y"], math.radians(p["yaw_deg"]))
        last = state["attempts"][-1] if state.get("attempts") else None
        match = None
        if last is not None:
            match = Match(
                True,
                pose,
                last["score"],
                last["second"],
                last["margin"],
                last["overlap"],
                last["reason"],
            )
        home.merge(sid, grid, high, pose, match, session)
        joined = home.connect_islands(skip=sid, visual=True) if newly else []
        home.save()
        union = home.union(grid, pose)
        out = {"registered": True, "pose": state["pose"], "joined_islands": joined}
    else:
        out = {"registered": False}
        if station_id >= max(RETRY_STATIONS):
            # 決めたステーションで合わなかった。島として残し（足し直す）、あとで合わせ直す
            last = state.get("attempts", [{}])[-1]
            match = Match(False, Pose(0, 0, 0), reason=str(last.get("reason", "")))
            home.add_island(sid, grid, high, match, session)
            home.save(grid=False)
            out["island"] = True
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return union, out
