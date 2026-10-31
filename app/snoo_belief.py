"""間取りの格子に重ねる 2 つの層：確信度（どれだけ確かに見たか）と、手ざわり（テクスチャ・目印）。

確信度（Confidence）
- セルごとに 0〜1。床・障害物のどちらと見たかは今までどおり log-odds（app/roommap.Grid）で決め、
  確信度は「その見え方をどれだけ確かめたか」だけを持つ
- 1 回見るごとに、見えた距離 d が近いほど強く足す：p = P_NEAR × exp(-d / D_FALL)。
  確信度 = 1 - Π(1 - p)（何回も見るほど 1 に近づく）。床と障害物の両方に見えたセルは、
  少ないほうの分だけ下げる（見え方が食い違っている）
- VGGT の確からしさ（conf）があれば p に掛ける（無ければ 1）
- 車が走った所（mark_driven）は 1（車がそこに居た）
- 「調べ終えた」（done no_frontier）は、境目が無いだけでなく、確信度の低い縁が少ないときだけにする
  （app/snoo_mapper.stop_reason。gear_segments の決め方のとき）

手ざわり（Texture）
- 障害物と「上」の層の点に、写真のその所の勾配の強さ（0〜1）を付け、セルごとの平均を持つ。
  白い壁・白いドア・床だけの所は低い。家具・本棚・物の多い所は高い
- 手ざわりの高いセル（LANDMARK_TEX 以上）を目印にする。VGGT と RoMa は手ざわりの無い所で
  位置を外しやすい（白い壁ばかりの部屋で、ステーションの位置合わせが崩れた）。次の 1 歩は、
  着いた所から目印が見える姿勢を選ぶ（app/snoo_maneuver.py）
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# 1 回見たときの強さ（すぐ近くで見たとき）と、距離で弱まる長さ（m）
P_NEAR = 0.6
D_FALL = 1.5
# 走った所の確信度
DRIVEN_CONF = 1.0
# 確信度がこれより低い縁（障害物と床の境）を「まだ確かめていない」とする
CONF_OK = 0.35
# 境目が無くても、確かめていない縁がこれだけあれば調べ終えたとしない
CONF_DONE_MAX_CELLS = 10
# 目印がこれより少なければ、目印で縁を絞らない
LANDMARK_MIN = 8
# 超音波がカメラの障害物と合ったときに足す強さと、カメラの床と食い違ったときの床の確信度の上限
SONAR_AGREE = 0.2
SONAR_CONFLICT_CAP = 0.4
# 目印とみなす手ざわり（0〜1）
LANDMARK_TEX = 0.35
# 勾配の強さをこの値で割って 0〜1 にする（Sobel の大きさの 7x7 の平均。白い壁は 5 前後、
# 本棚・物の多い所は 30 を超える）
TEX_GRAD_SCALE = 30.0


class Confidence:
    """格子と同じ大きさの確信度の層。床・障害物ごとに「まだ確かめていない割合」の積を持つ。"""

    def __init__(self, n: int):
        self.miss_free = np.ones((n, n), np.float32)
        self.miss_occ = np.ones((n, n), np.float32)
        self.driven = np.zeros((n, n), bool)

    def add(self, grid, free_xy: np.ndarray, occ_xy: np.ndarray, pose, conf=None) -> None:
        """1 回の撮影で床・障害物と見えたセル（中心の座標 (K, 2)）を足す。pose は撮った所。
        conf は障害物・床それぞれの VGGT の確からしさ（0〜1、(K,)）か None。"""
        for xy, miss in ((free_xy, self.miss_free), (occ_xy, self.miss_occ)):
            if xy is None or not len(xy):
                continue
            ix, iy = grid.cell(xy[:, 0], xy[:, 1])
            ok = grid.inside(ix, iy)
            d = np.hypot(xy[:, 0] - pose.x, xy[:, 1] - pose.y)
            p = P_NEAR * np.exp(-d / D_FALL)
            if conf is not None:
                p = p * np.clip(conf, 0.0, 1.0)
            cells = np.stack([ix[ok], iy[ok]], 1)
            if not len(cells):
                continue
            # 同じセルが 1 回の撮影で何点あっても 1 回だけ（いちばん近いもの）
            order = np.argsort(-p[ok])
            cells, pv = cells[order], p[ok][order]
            _, first = np.unique(cells, axis=0, return_index=True)
            cells, pv = cells[first], pv[first]
            miss[cells[:, 0], cells[:, 1]] *= (1.0 - pv).astype(np.float32)

    def add_cells(self, grid, free_cells: np.ndarray, occ_cells: np.ndarray, pose) -> None:
        """セルの番号 (K, 2)（[ix, iy]）で足す（シミュレーター）。"""
        out = []
        for c in (free_cells, occ_cells):
            if c is None or not len(c):
                out.append(np.zeros((0, 2)))
                continue
            x, y = grid.center(c[:, 0], c[:, 1])
            out.append(np.stack([x, y], 1))
        self.add(grid, out[0], out[1], pose)

    def add_sonar(self, grid, camera, distance_m) -> None:
        """超音波の読み（地図の弱い証拠）。カメラが障害物と見たセルと合えば確信度を少し上げ、
        カメラが床と見たセルに当たれば食い違い（近づいて確かめる行き先になる）として床の確信度を
        下げる。測れなかった・外れた値は捨てる（roommap.sonar_cells）。"""
        from app.roommap import FREE, OCCUPIED, sonar_cells

        cells = sonar_cells(grid, camera, distance_m)
        if not len(cells):
            return
        st = grid.state()[cells[:, 0], cells[:, 1]]
        occ = cells[st == OCCUPIED]
        self.miss_occ[occ[:, 0], occ[:, 1]] *= 1.0 - SONAR_AGREE
        free = cells[st == FREE]
        self.miss_free[free[:, 0], free[:, 1]] = np.maximum(
            self.miss_free[free[:, 0], free[:, 1]], 1.0 - SONAR_CONFLICT_CAP
        )

    def mark_driven(self, cells: np.ndarray) -> None:
        if len(cells):
            self.driven[cells[:, 0], cells[:, 1]] = True

    def value(self, state: np.ndarray) -> np.ndarray:
        """いまの床・障害物の見え方の確信度（0〜1）。見ていないセルは 0。"""
        from app.roommap import FREE, OCCUPIED

        c_free = 1.0 - self.miss_free
        c_occ = 1.0 - self.miss_occ
        out = np.zeros(state.shape, np.float32)
        f, o = state == FREE, state == OCCUPIED
        out[f] = c_free[f] * (1.0 - 0.5 * c_occ[f])
        out[o] = c_occ[o] * (1.0 - 0.5 * c_free[o])
        out[self.driven & f] = DRIVEN_CONF
        return out


def fallback_confidence(grid, state: np.ndarray) -> np.ndarray:
    """確信度の層が無い格子では、log-odds の大きさから出す（1 回見た床 0.57、走った所 1）。"""
    from app.roommap import UNKNOWN

    out = np.clip(np.abs(grid.logodds) / 1.5, 0.0, 1.0).astype(np.float32)
    out[state == UNKNOWN] = 0.0
    return out


def confidence_of(grid, state: np.ndarray) -> np.ndarray:
    conf = getattr(grid, "confidence", None)
    return conf.value(state) if conf is not None else fallback_confidence(grid, state)


class Texture:
    """セルごとの手ざわりの平均（0〜1）。障害物と「上」の層の点から足す。"""

    def __init__(self, n: int):
        self.sum = np.zeros((n, n), np.float32)
        self.count = np.zeros((n, n), np.int32)

    def add(self, grid, xyt: np.ndarray) -> None:
        """(K, 3) の [x, y, 手ざわり] を足す。"""
        if xyt is None or not len(xyt):
            return
        ix, iy = grid.cell(xyt[:, 0], xyt[:, 1])
        ok = grid.inside(ix, iy)
        np.add.at(self.sum, (ix[ok], iy[ok]), xyt[ok, 2].astype(np.float32))
        np.add.at(self.count, (ix[ok], iy[ok]), 1)

    def value(self) -> np.ndarray:
        return np.where(self.count > 0, self.sum / np.maximum(self.count, 1), 0.0)

    def landmarks(self, grid) -> np.ndarray:
        """目印のセルの中心 (L, 2)。"""
        cells = np.argwhere(self.value() >= LANDMARK_TEX)
        if not len(cells):
            return np.zeros((0, 2))
        x, y = grid.center(cells[:, 0], cells[:, 1])
        return np.stack([x, y], 1)


def texture_map(bgr: np.ndarray) -> np.ndarray:
    """画像（BGR か灰色）の画素ごとの手ざわり（0〜1）。勾配の大きさを 7x7 で平均して割る。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.blur(np.sqrt(gx * gx + gy * gy), (7, 7))
    return np.clip(mag / TEX_GRAD_SCALE, 0.0, 1.0)


def image_texture_score(path) -> dict:
    """1 枚の写真の手ざわり：手ざわりの平均・手ざわりの高い画素の割合・FAST の角点の数。"""
    img = cv2.imread(str(path))
    if img is None:
        return {"mean": 0.0, "rich": 0.0, "fast": 0}
    img = cv2.resize(img, (320, int(round(320 * img.shape[0] / img.shape[1]))))
    t = texture_map(img)
    fast = cv2.FastFeatureDetector_create(threshold=25).detect(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None
    )
    return {
        "mean": round(float(t.mean()), 3),
        "rich": round(float((t >= LANDMARK_TEX).mean()), 3),
        "fast": len(fast),
    }


def visible_landmarks(
    landmarks: np.ndarray, x, y, yaw, half_deg: float = 62.0, range_m: float = 2.0
) -> np.ndarray:
    """姿勢（配列でよい）から、首を振って見える範囲（±half_deg、range_m まで）にある目印の数。
    陰は数えない（速さのため）。"""
    x, y, yaw = (np.asarray(v, float)[..., None] for v in (x, y, yaw))
    if not len(landmarks):
        return np.zeros(x.shape[:-1], int)
    dx = landmarks[:, 0] - x
    dy = landmarks[:, 1] - y
    d = np.hypot(dx, dy)
    off = np.abs(np.remainder(np.arctan2(dy, dx) - yaw + math.pi, 2 * math.pi) - math.pi)
    return ((d <= range_m) & (d > 0.05) & (off <= math.radians(half_deg))).sum(-1)


def confidence_png(conf: np.ndarray, state: np.ndarray) -> np.ndarray:
    """確信度の絵（上が +y）。見ていない所は 0、見た所は 1〜255。"""
    from app.roommap import UNKNOWN

    img = np.where(state == UNKNOWN, 0, 1 + np.round(conf * 254)).astype(np.uint8)
    return img.T[::-1]


def texture_png(tex: np.ndarray) -> np.ndarray:
    return np.round(np.clip(tex, 0, 1) * 255).astype(np.uint8).T[::-1]


def unsure_edges(grid, state: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """確信度の低い障害物・床の縁のセル（確信度の層 grid.confidence が無ければ無し）。
    目印の層に十分な目印があれば、1 m 以内に目印のある縁
    だけ（目印が同時に見える所からだけ確かめに行く。白い壁だけを近くで撮っても位置が合わない）。"""
    from app.roommap import FREE, OCCUPIED

    if getattr(grid, "confidence", None) is None:  # 確信度を持たない格子では使わない
        return np.zeros(state.shape, bool)
    occ = state == OCCUPIED
    free = state == FREE
    k = np.ones((3, 3), np.uint8)
    near_free = cv2.dilate(free.astype(np.uint8), k) > 0
    near_occ = cv2.dilate(occ.astype(np.uint8), k) > 0
    low = ((occ & near_free) | (free & near_occ)) & (conf < CONF_OK)
    tex = getattr(grid, "texture", None)
    lms = tex.landmarks(grid) if tex is not None else np.zeros((0, 2))
    if len(lms) >= LANDMARK_MIN and low.any():
        lm = np.zeros(state.shape, np.uint8)
        ix, iy = grid.cell(lms[:, 0], lms[:, 1])
        ok = grid.inside(ix, iy)
        lm[ix[ok], iy[ok]] = 1
        r = int(round(1.0 / grid.res))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        low &= cv2.dilate(lm, disk) > 0
    return low
