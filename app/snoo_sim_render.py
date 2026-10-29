"""app/snoo_sim.py のシミュレーターを絵にする（3D の鳥瞰・車のカメラ・上から見た地図）。

1 コマは 1920x1080 で、左上が 3D の鳥瞰、左下が車のカメラ、右上が上から見た地図、
右下が数字と凡例。どれにも同じ要素を描く。
- 部屋の格子（作っている地図）：床・障害物・上の層・見ていない所、境目（frontier）
- ステーションの位置と向き・走った跡、車の位置と向き
- カメラの見える範囲（画面の縁を床に落とした形。パン・チルトに付いて動く）
- 超音波の円すい（±15°、当たった距離まで）
- next_best_view の行き先と Dubins 経路、最初の 1 歩の弧
- clear_cm の通り道（ハンドルの弧 × 車の幅、ぶつかるまで）
- snoo の位置と、車のカメラの中の検出の箱
3D は MuJoCo の scene に geom を足して描く（mjv_initGeom）。部屋の物ではないので光線には当たらない。

描くには GL が要る（MUJOCO_GL=egl。AMD の GPU でも radeonsi で動く）。
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.roommap import FREE, OCCUPIED, UNKNOWN, frontiers
from app.snoo_sim import (
    CAM_X_M,
    CAR_FRONT_M,
    CAR_HEIGHT_M,
    GROUP_CAR,
    GROUP_DIM,
    IMG_H,
    IMG_W,
    SNOO_FRONT_M,
    SNOO_HALF_W_M,
    SNOO_REAR_M,
    SONAR_H_M,
    SONAR_HALF_DEG,
    SONAR_RANGE_M,
    Sim,
    footprint,
    merge_cells,
    mujoco,
    project,
)

W, H = 1920, 1080
# 色（3D は RGBA 0..1、2D は BGR 0..255）。同じ要素は同じ色にする
C3 = {
    "free": (0.25, 0.85, 0.35, 0.22),
    "occ": (0.9, 0.12, 0.08, 0.6),
    "high": (0.25, 0.45, 1.0, 0.35),
    "unknown": (0.04, 0.04, 0.07, 0.5),
    "frontier": (1.0, 0.82, 0.05, 0.85),
    "station": (0.15, 0.45, 0.95, 1.0),
    "trail": (0.15, 0.45, 0.95, 0.9),
    "heading": (0.1, 0.95, 1.0, 1.0),
    "frustum": (0.1, 0.95, 1.0, 0.9),
    "sonar": (1.0, 0.2, 1.0, 0.9),
    "path": (0.1, 0.75, 0.15, 1.0),
    "target": (0.95, 0.1, 0.1, 1.0),
    "step": (1.0, 0.55, 0.0, 1.0),
    "corridor": (1.0, 0.55, 0.0, 0.25),
    "snoo": (0.75, 0.25, 0.85, 0.35),
    "seen": (0.1, 1.0, 0.2, 0.5),
}
C2 = {
    "bg": (34, 30, 30),
    "free": (236, 236, 236),
    "occ": (40, 40, 40),
    "high": (215, 165, 110),
    "unknown": (120, 116, 112),
    "frontier": (0, 210, 255),
    "grid": (130, 175, 130),
    "truth_low": (60, 60, 210),
    "truth_high": (210, 140, 60),
    "station": (230, 120, 30),
    "car": (200, 95, 30),
    "heading": (255, 240, 20),
    "frustum": (255, 230, 40),
    "sonar": (255, 40, 255),
    "path": (0, 170, 0),
    "target": (0, 0, 230),
    "step": (0, 140, 255),
    "corridor": (0, 140, 255),
    "snoo": (200, 60, 170),
    "hit_floor": (60, 200, 60),
    "hit_occ": (40, 40, 230),
    "hit_high": (230, 150, 40),
    "text": (235, 235, 235),
    "detect": (40, 255, 40),
}


# --- 3D に足す geom ---


class Overlay:
    """MjvScene の後ろに geom を足す（update_scene の後、render の前に呼ぶ）。"""

    def __init__(self, scene):
        self.scn = scene
        self.eye = np.eye(3).ravel()

    def _next(self):
        if self.scn.ngeom >= self.scn.maxgeom:
            return None
        g = self.scn.geoms[self.scn.ngeom]
        self.scn.ngeom += 1
        return g

    def add(self, kind, size, pos, rgba, mat=None) -> None:
        g = self._next()
        if g is None:
            return
        mujoco.mjv_initGeom(
            g,
            kind,
            np.asarray(size, np.float64),
            np.asarray(pos, np.float64),
            self.eye if mat is None else np.asarray(mat, np.float64).ravel(),
            np.asarray(rgba, np.float32),
        )

    def connect(self, kind, width, a, b, rgba) -> None:
        g = self._next()
        if g is None:
            return
        mujoco.mjv_initGeom(
            g, kind, np.zeros(3), np.zeros(3), self.eye, np.asarray(rgba, np.float32)
        )
        mujoco.mjv_connector(g, kind, width, np.asarray(a, np.float64), np.asarray(b, np.float64))

    def box(self, x0, y0, x1, y1, z, half_h, rgba, yaw=0.0) -> None:
        c, s = math.cos(yaw), math.sin(yaw)
        mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.add(
            mujoco.mjtGeom.mjGEOM_BOX,
            [(x1 - x0) / 2, (y1 - y0) / 2, half_h],
            [(x0 + x1) / 2, (y0 + y1) / 2, z],
            rgba,
            mat,
        )

    def line(self, a, b, rgba, px=2.0) -> None:
        self.connect(mujoco.mjtGeom.mjGEOM_LINE, px, a, b, rgba)

    def rod(self, a, b, r, rgba) -> None:
        if np.linalg.norm(np.subtract(a, b)) > 1e-6:
            self.connect(mujoco.mjtGeom.mjGEOM_CAPSULE, r, a, b, rgba)

    def arrow(self, a, b, r, rgba) -> None:
        self.connect(mujoco.mjtGeom.mjGEOM_ARROW, r, a, b, rgba)

    def sphere(self, p, r, rgba) -> None:
        self.add(mujoco.mjtGeom.mjGEOM_SPHERE, [r, r, r], p, rgba)

    def disk(self, x, y, r, z, rgba) -> None:
        self.add(mujoco.mjtGeom.mjGEOM_CYLINDER, [r, 0.002, 0], [x, y, z], rgba)

    def polyline(self, pts, z, rgba, r=None, px=2.0) -> None:
        for a, b in zip(pts, pts[1:], strict=False):
            a3, b3 = (a[0], a[1], z), (b[0], b[1], z)
            if r is None:
                self.line(a3, b3, rgba, px)
            else:
                self.rod(a3, b3, r, rgba)


def _window(sim: Sim, margin_m: float = 0.25) -> tuple[int, int, int, int]:
    """描く範囲のセル (i0, i1, j0, j1)。部屋の範囲に少し足す。"""
    g = sim.grid
    x0, y0, x1, y1 = sim.room.bounds
    i0, j0 = g.cell(x0 - margin_m, y0 - margin_m)
    i1, j1 = g.cell(x1 + margin_m, y1 + margin_m)
    return max(int(i0), 0), min(int(i1) + 1, g.n), max(int(j0), 0), min(int(j1) + 1, g.n)


def _camera_path(arc_poses) -> list[tuple[float, float]]:
    """後ろの車軸の弧を、カメラの真下が通る線にする（地図の位置はカメラの真下なので）。"""
    return [(x + CAM_X_M * math.cos(t), y + CAM_X_M * math.sin(t)) for x, y, t in arc_poses]


def add_scene_overlays(scn, sim: Sim) -> None:
    """3D の鳥瞰に、地図と計画とセンサーを足す。"""
    ov = Overlay(scn)
    g = sim.grid
    state = g.state()
    i0, i1, j0, j1 = _window(sim)
    sub = state[i0:i1, j0:j1]
    high = (sim.high[i0:i1, j0:j1] > 0) & (sub != OCCUPIED)
    front = frontiers(state)[i0:i1, j0:j1]
    layers = [
        (sub == UNKNOWN, "unknown", 0.0015, 0.0008),
        (sub == FREE, "free", 0.003, 0.0008),
        (front, "frontier", 0.005, 0.0015),
        (sub == OCCUPIED, "occ", 0.012, 0.01),
        (high, "high", CAR_HEIGHT_M + 0.03, 0.002),
    ]
    for mask, name, z, hh in layers:
        for a, b, c, d in merge_cells(mask):
            x0, y0 = g.origin + (i0 + a) * g.res, g.origin + (j0 + c) * g.res
            x1, y1 = g.origin + (i0 + b) * g.res, g.origin + (j0 + d) * g.res
            ov.box(x0, y0, x1, y1, z, hh, C3[name])

    # ステーションと走った跡
    ov.polyline([(p.x, p.y) for p in sim.trail], 0.008, C3["trail"], r=0.006)
    for s in sim.stations:
        p = s.pose
        ov.sphere((p.x, p.y, 0.02), 0.025, C3["station"])
        ov.arrow(
            (p.x, p.y, 0.02),
            (p.x + 0.15 * math.cos(p.yaw), p.y + 0.15 * math.sin(p.yaw), 0.02),
            0.012,
            C3["station"],
        )

    # 車の向き
    cam = sim.car.camera_pose()
    ov.arrow(
        (cam.x, cam.y, CAR_HEIGHT_M + 0.03),
        (cam.x + 0.3 * math.cos(cam.yaw), cam.y + 0.3 * math.sin(cam.yaw), CAR_HEIGHT_M + 0.03),
        0.015,
        C3["heading"],
    )

    # カメラの見える範囲
    pos, _ = sim.camera()
    poly = sim.view_polygon()
    if len(poly):
        closed = np.vstack([poly, poly[:1]])
        ov.polyline(closed, 0.006, C3["frustum"], px=2.5)
        n = len(poly) // 4
        for k in range(4):
            q = poly[k * n]
            ov.line(pos, (q[0], q[1], 0.006), C3["frustum"], 1.5)

    # 超音波の円すい
    c = sim.car
    fx, fy = c.to_world(CAR_FRONT_M, 0)
    reach = sim.sonar_m if sim.sonar_m is not None else SONAR_RANGE_M
    ang = np.radians(np.linspace(-SONAR_HALF_DEG, SONAR_HALF_DEG, 9)) + c.yaw
    edge = [(fx + reach * math.cos(a), fy + reach * math.sin(a)) for a in ang]
    ov.line((fx, fy, SONAR_H_M), (*edge[0], SONAR_H_M), C3["sonar"], 2)
    ov.line((fx, fy, SONAR_H_M), (*edge[-1], SONAR_H_M), C3["sonar"], 2)
    ov.polyline(edge, SONAR_H_M, C3["sonar"], px=2)

    # 計画：Dubins 経路・行き先・最初の 1 歩
    if sim.plan is not None:
        ov.polyline(
            [(p.x, p.y) for p in sim.plan.path[::3]] + [(sim.plan.path[-1].x, sim.plan.path[-1].y)],
            0.012,
            C3["path"],
            r=0.008,
        )
        t = sim.plan.target
        ov.sphere((t.x, t.y, 0.03), 0.03, C3["target"])
        ov.arrow(
            (t.x, t.y, 0.03),
            (t.x + 0.2 * math.cos(t.yaw), t.y + 0.2 * math.sin(t.yaw), 0.03),
            0.014,
            C3["target"],
        )
    if sim.step_arc:
        ov.polyline(_camera_path(sim.step_arc), 0.018, C3["step"], r=0.012)

    # clear_cm の通り道（前の左右の角の線と、間を埋める板）
    poly = sim.corridor
    if len(poly) >= 4:
        n = len(poly) // 2
        left, right = poly[:n], poly[n:][::-1]
        ov.polyline(left, 0.01, C3["step"], px=2)
        ov.polyline(right, 0.01, C3["step"], px=2)
        ov.line((*left[-1], 0.01), (*right[-1], 0.01), C3["step"], 3)
        for k in range(0, n - 1, 2):
            a = (left[k] + right[k]) / 2
            b = (left[min(k + 2, n - 1)] + right[min(k + 2, n - 1)]) / 2
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            length = float(np.linalg.norm(b - a))
            width = float(np.linalg.norm(left[k] - right[k]))
            mid = (a + b) / 2
            ov.box(
                mid[0] - length / 2,
                mid[1] - width / 2,
                mid[0] + length / 2,
                mid[1] + width / 2,
                0.007,
                0.001,
                C3["corridor"],
                yaw,
            )

    # snoo の位置（検出されていれば緑の輪も）
    s = sim.snoo.pose
    ov.disk(s.x, s.y, 0.27, 0.004, C3["snoo"])
    if sim.detection is not None:
        ov.disk(s.x, s.y, 0.31, 0.003, C3["seen"])

    # いま撮ったコマの光線の当たった点（多すぎるので間引く）
    if sim.phase == "scan" and sim.last_hits is not None and len(sim.last_hits.points):
        pts, kind = sim.last_hits.points, sim.last_hits.kind
        sel = np.linspace(0, len(pts) - 1, min(len(pts), 700)).astype(int)
        colors = [(0.2, 0.9, 0.2, 1), (1, 0.15, 0.1, 1), (0.25, 0.55, 1, 1)]
        for p, k in zip(pts[sel], kind[sel], strict=True):
            ov.add(mujoco.mjtGeom.mjGEOM_BOX, [0.006] * 3, p, colors[int(k)])


# --- 上から見た地図（2D） ---


class TopDown:
    """格子のセルを整数の画素にして、部屋を w x h に収める。"""

    def __init__(self, sim: Sim, w: int, h: int):
        g = sim.grid
        self.i0, self.i1, self.j0, self.j1 = _window(sim)
        nx, ny = self.i1 - self.i0, self.j1 - self.j0
        self.ppc = max(1, int(min((w - 20) / nx, (h - 60) / ny)))
        self.scale = self.ppc / g.res  # 画素 / m
        self.w, self.h = w, h
        self.ox = (w - nx * self.ppc) // 2
        self.oy = (h - 40 - ny * self.ppc) // 2
        self.x0 = g.origin + self.i0 * g.res
        self.y1 = g.origin + self.j1 * g.res

    def px(self, x, y) -> tuple[int, int]:
        return (
            int(round(self.ox + (x - self.x0) * self.scale)),
            int(round(self.oy + (self.y1 - y) * self.scale)),
        )

    def pts(self, xy) -> np.ndarray:
        return np.array([self.px(x, y) for x, y in xy], np.int32)

    def draw(self, sim: Sim) -> np.ndarray:
        img = np.full((self.h, self.w, 3), C2["bg"], np.uint8)
        g = sim.grid
        state = g.state()
        sub = state[self.i0 : self.i1, self.j0 : self.j1]
        cells = np.empty((*sub.shape, 3), np.uint8)
        cells[:] = C2["unknown"]
        cells[sub == FREE] = C2["free"]
        cells[sub == OCCUPIED] = C2["occ"]
        hi = (sim.high[self.i0 : self.i1, self.j0 : self.j1] > 0) & (sub != OCCUPIED)
        cells[hi & (sub == FREE)] = np.array(C2["high"]) * 0.45 + np.array(C2["free"]) * 0.55
        cells[hi & (sub == UNKNOWN)] = C2["high"]
        cells[frontiers(state)[self.i0 : self.i1, self.j0 : self.j1]] = C2["frontier"]
        pic = np.ascontiguousarray(cells.transpose(1, 0, 2)[::-1])
        pic = cv2.resize(pic, None, fx=self.ppc, fy=self.ppc, interpolation=cv2.INTER_NEAREST)
        img[self.oy : self.oy + pic.shape[0], self.ox : self.ox + pic.shape[1]] = pic

        # 1 m ごとの線
        for xm in range(math.ceil(self.x0), math.floor(self.x0 + (self.i1 - self.i0) * g.res) + 1):
            u, _ = self.px(xm, 0)
            cv2.line(img, (u, self.oy), (u, self.oy + pic.shape[0]), C2["grid"], 1)
        y0 = self.y1 - (self.j1 - self.j0) * g.res
        for ym in range(math.ceil(y0), math.floor(self.y1) + 1):
            _, v = self.px(0, ym)
            cv2.line(img, (self.ox, v), (self.ox + pic.shape[1], v), C2["grid"], 1)

        # 本当の部屋の箱（輪郭だけ。地図と比べるため）
        if sim.room.source_state is None:
            for b in sim.room.blocks:
                low = b.lo[2] < CAR_HEIGHT_M
                cv2.rectangle(
                    img,
                    self.px(b.lo[0], b.hi[1]),
                    self.px(b.hi[0], b.lo[1]),
                    C2["truth_low" if low else "truth_high"],
                    1,
                )

        # 半透明の面：カメラの見える範囲・超音波・通り道
        layer = img.copy()
        poly = sim.view_polygon()
        if len(poly):
            cv2.fillPoly(layer, [self.pts(poly)], C2["frustum"])
        c = sim.car
        fx, fy = c.to_world(CAR_FRONT_M, 0)
        reach = sim.sonar_m if sim.sonar_m is not None else SONAR_RANGE_M
        ang = np.radians(np.linspace(-SONAR_HALF_DEG, SONAR_HALF_DEG, 13)) + c.yaw
        cone = [(fx, fy)] + [(fx + reach * math.cos(a), fy + reach * math.sin(a)) for a in ang]
        if len(sim.corridor) >= 4:
            cv2.fillPoly(layer, [self.pts(sim.corridor)], C2["corridor"])
        img = cv2.addWeighted(layer, 0.28, img, 0.72, 0)
        layer = img.copy()
        cv2.fillPoly(layer, [self.pts(cone)], C2["sonar"])
        img = cv2.addWeighted(layer, 0.15, img, 0.85, 0)
        if len(poly):
            cv2.polylines(img, [self.pts(poly)], True, C2["frustum"], 1, cv2.LINE_AA)
        cv2.polylines(img, [self.pts(cone)], True, C2["sonar"], 1, cv2.LINE_AA)
        if len(sim.corridor) >= 4:
            cv2.polylines(img, [self.pts(sim.corridor)], True, C2["corridor"], 1, cv2.LINE_AA)

        # いま撮ったコマの点
        if sim.phase == "scan" and sim.last_hits is not None:
            colors = (C2["hit_floor"], C2["hit_occ"], C2["hit_high"])
            pts, kind = sim.last_hits.points, sim.last_hits.kind
            sel = np.linspace(0, len(pts) - 1, min(len(pts), 3000)).astype(int)
            for p, k in zip(pts[sel], kind[sel], strict=True):
                cv2.circle(img, self.px(p[0], p[1]), 1, colors[int(k)], -1)

        # 計画
        if sim.plan is not None:
            cv2.polylines(
                img,
                [self.pts([(p.x, p.y) for p in sim.plan.path])],
                False,
                C2["path"],
                2,
                cv2.LINE_AA,
            )
            t = sim.plan.target
            a = self.px(t.x, t.y)
            b = self.px(t.x + 0.25 * math.cos(t.yaw), t.y + 0.25 * math.sin(t.yaw))
            cv2.circle(img, a, 7, C2["target"], 2, cv2.LINE_AA)
            cv2.arrowedLine(img, a, b, C2["target"], 2, cv2.LINE_AA, tipLength=0.3)
        if sim.step_arc:
            cv2.polylines(
                img, [self.pts(_camera_path(sim.step_arc))], False, C2["step"], 3, cv2.LINE_AA
            )

        # ステーションと跡
        if len(sim.trail) > 1:
            cv2.polylines(
                img,
                [self.pts([(p.x, p.y) for p in sim.trail])],
                False,
                C2["station"],
                1,
                cv2.LINE_AA,
            )
        for k, s in enumerate(sim.stations):
            p = s.pose
            a = self.px(p.x, p.y)
            b = self.px(p.x + 0.18 * math.cos(p.yaw), p.y + 0.18 * math.sin(p.yaw))
            cv2.circle(img, a, 6, C2["station"], 2, cv2.LINE_AA)
            cv2.arrowedLine(img, a, b, C2["station"], 2, cv2.LINE_AA, tipLength=0.35)
            cv2.putText(
                img, str(k + 1), (a[0] + 7, a[1] - 7), 0, 0.45, C2["station"], 1, cv2.LINE_AA
            )

        # snoo（跡と体）
        tr = sim.snoo_trail[::4]
        for p in tr:
            cv2.circle(img, self.px(p.x, p.y), 1, C2["snoo"], -1)
        s = sim.snoo.pose
        cs, sn = math.cos(s.yaw), math.sin(s.yaw)
        body = [
            (SNOO_FRONT_M, SNOO_HALF_W_M),
            (SNOO_REAR_M + 0.1, SNOO_HALF_W_M),
            (SNOO_REAR_M + 0.1, -SNOO_HALF_W_M),
            (SNOO_FRONT_M, -SNOO_HALF_W_M),
        ]
        bw = [(s.x + cs * a - sn * b, s.y + sn * a + cs * b) for a, b in body]
        cv2.fillPoly(img, [self.pts(bw)], C2["snoo"], cv2.LINE_AA)
        tail = self.pts(
            [
                (s.x + cs * (SNOO_REAR_M + 0.1), s.y + sn * (SNOO_REAR_M + 0.1)),
                (s.x + cs * SNOO_REAR_M, s.y + sn * SNOO_REAR_M),
            ]
        )
        cv2.line(img, tuple(tail[0]), tuple(tail[1]), C2["snoo"], 2, cv2.LINE_AA)
        head = self.px(s.x + cs * 0.15, s.y + sn * 0.15)
        cv2.circle(img, head, max(2, int(0.055 * self.scale)), C2["snoo"], -1, cv2.LINE_AA)
        if sim.detection is not None:
            cv2.circle(img, self.px(s.x, s.y), int(0.3 * self.scale), C2["detect"], 1, cv2.LINE_AA)
        cv2.putText(img, "snoo", (head[0] + 6, head[1] - 6), 0, 0.45, C2["snoo"], 1, cv2.LINE_AA)

        # 車（外形・カメラ・向き）
        fp = footprint(c.x, c.y, c.yaw)
        cv2.fillPoly(img, [self.pts(fp)], C2["car"], cv2.LINE_AA)
        cv2.polylines(img, [self.pts(fp)], True, (255, 255, 255), 1, cv2.LINE_AA)
        cam = c.camera_pose()
        a = self.px(cam.x, cam.y)
        b = self.px(cam.x + 0.3 * math.cos(cam.yaw), cam.y + 0.3 * math.sin(cam.yaw))
        cv2.arrowedLine(img, a, b, C2["heading"], 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(img, a, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 縮尺
        u0, v0 = 14, self.h - 18
        cv2.line(img, (u0, v0), (u0 + int(self.scale), v0), C2["text"], 2)
        cv2.putText(
            img, "1 m", (u0 + int(self.scale) + 6, v0 + 4), 0, 0.45, C2["text"], 1, cv2.LINE_AA
        )
        return img


# --- 車のカメラ ---


def draw_camera_hud(img: np.ndarray, sim: Sim) -> np.ndarray:
    """車のカメラの絵に、通り道（床に落とした帯）と snoo の検出の箱を描く。"""
    pos, mat = sim.camera()
    if len(sim.corridor) >= 4:
        pts = np.column_stack([sim.corridor, np.zeros(len(sim.corridor))])
        u, v, front = project(pos, mat, pts)
        if front.all():
            poly = np.stack([u, v], 1).astype(np.int32)
            layer = img.copy()
            cv2.fillPoly(layer, [poly], C2["corridor"])
            img = cv2.addWeighted(layer, 0.25, img, 0.75, 0)
            cv2.polylines(img, [poly], True, C2["corridor"], 2, cv2.LINE_AA)
    if sim.detection is not None:
        u0, v0, u1, v1, seen = sim.detection
        cv2.rectangle(img, (int(u0), int(v0)), (int(u1), int(v1)), C2["detect"], 2)
        cv2.putText(
            img,
            f"snoo {seen:.2f}",
            (int(u0), max(int(v0) - 6, 14)),
            0,
            0.6,
            C2["detect"],
            2,
            cv2.LINE_AA,
        )
    c = sim.car
    lines = [
        f"pan {c.pan_deg:+.0f}  tilt {c.tilt_deg:+.0f}  steer {c.steer_deg:+.1f}",
        f"sonar {'-' if sim.sonar_m is None else f'{sim.sonar_m * 100:.0f} cm'}"
        f"  clear {sim.clear_m * 100:.0f} cm",
    ]
    cv2.rectangle(img, (0, 0), (430, 12 + 26 * len(lines)), (0, 0, 0), -1)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (10, 28 + 26 * k), 0, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --- まとめて 1 コマにする ---


def _text(img, lines, x, y, scale=0.55, step=24, color=None) -> None:
    for k, t in enumerate(lines):
        cv2.putText(img, t, (x, y + k * step), 0, scale, color or C2["text"], 1, cv2.LINE_AA)


def _legend(img, x, y) -> None:
    items = [
        ("free", "floor (map)"),
        ("occ", "obstacle"),
        ("high", "high layer"),
        ("unknown", "unknown"),
        ("frontier", "frontier"),
        ("station", "stations"),
        ("frustum", "camera view"),
        ("sonar", "ultrasonic"),
        ("path", "Dubins path"),
        ("target", "next view"),
        ("step", "1st step / clear"),
        ("snoo", "snoo"),
        ("truth_low", "truth: low"),
        ("truth_high", "truth: high"),
    ]
    for k, (key, name) in enumerate(items):
        cx, cy = x + (k % 4) * 225, y + (k // 4) * 26
        cv2.rectangle(img, (cx, cy - 12), (cx + 16, cy + 2), C2[key], -1)
        cv2.putText(img, name, (cx + 22, cy), 0, 0.5, C2["text"], 1, cv2.LINE_AA)


class SimRenderer:
    """シミュレーターの 1 コマ（1920x1080、BGR）を作る。"""

    def __init__(self, sim: Sim):
        self.sim = sim
        self.r3d = mujoco.Renderer(sim.model, 540, 960, max_geom=20000)
        self.rcam = mujoco.Renderer(sim.model, IMG_H, IMG_W, max_geom=2000)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[GROUP_DIM] = 1  # 見ていない所を暗くする板も描く
        # 車のカメラには車そのもの（前の超音波の板など）を映さない。実機のカメラは
        # もっと上・前にあるようで、自分の車体は映らない
        self.cam_opt = mujoco.MjvOption()
        self.cam_opt.geomgroup[GROUP_DIM] = 1
        self.cam_opt.geomgroup[GROUP_CAR] = 0
        self.cam = mujoco.MjvCamera()
        x0, y0, x1, y1 = sim.room.bounds
        self.center = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])
        # 横は x の幅、縦は y の奥行き（斜めに見るので縮む）が収まる距離
        self.cam.distance = 0.82 * max(x1 - x0, (y1 - y0) * 1.45) + 0.5
        self.cam.elevation = -55
        self.cam.azimuth = 100
        self.topdown = TopDown(sim, 960, 800)

    def close(self) -> None:
        self.r3d.close()
        self.rcam.close()

    def view3d(self) -> np.ndarray:
        sim = self.sim
        self.cam.lookat[:] = self.center
        self.r3d.update_scene(sim.data, self.cam, self.opt)
        add_scene_overlays(self.r3d.scene, sim)
        return cv2.cvtColor(self.r3d.render(), cv2.COLOR_RGB2BGR)

    def car_view(self) -> np.ndarray:
        self.rcam.update_scene(self.sim.data, "car_cam", self.cam_opt)
        img = cv2.cvtColor(self.rcam.render(), cv2.COLOR_RGB2BGR)
        return draw_camera_hud(np.ascontiguousarray(img), self.sim)

    def frame(self, title: str = "") -> np.ndarray:
        sim = self.sim
        out = np.zeros((H, W, 3), np.uint8)
        out[:] = C2["bg"]
        out[0:540, 0:960] = self.view3d()
        cam = cv2.resize(self.car_view(), (720, 540), interpolation=cv2.INTER_AREA)
        out[540:1080, 0:720] = cam
        out[0:800, 960:1920] = self.topdown.draw(sim)
        cv2.line(out, (960, 0), (960, H), (90, 90, 90), 2)
        cv2.line(out, (0, 540), (960, 540), (90, 90, 90), 2)
        cv2.line(out, (960, 800), (W, 800), (90, 90, 90), 2)
        cv2.putText(out, "3D bird's-eye", (10, 24), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            out, "car camera 640x480 (HFOV 54)", (10, 1068), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(out, "top-down map (5 cm grid)", (970, 24), 0, 0.6, C2["text"], 1, cv2.LINE_AA)

        # 右下：数字
        s = sim.summary()
        a = sim.action or {}
        p = sim.car.camera_pose()
        sp = sim.snoo.pose
        _text(
            out,
            [
                title or f"snoo_sim  {sim.room.name}",
                f"cycle {sim.cycle}  phase {sim.phase}  t {sim.t:.0f} s",
                f"stations {s.stations}  driven {s.distance_m:.2f} m",
                f"bumps {s.bumps}  stops {s.stops}  reverses {s.reverses}",
            ],
            975,
            830,
        )
        nxt = f"next: {a.get('action', '-')}" + (
            f" steer {a['steer_deg']:+.1f} speed {a['speed']} dist {a['distance_cm']:.0f} cm"
            if a.get("action") == "move"
            else f" ({a.get('reason', '')})"
        )
        _text(
            out,
            [
                f"car (camera) x {p.x:+.2f} y {p.y:+.2f} yaw {math.degrees(p.yaw):+.0f}",
                f"snoo x {sp.x:+.2f} y {sp.y:+.2f} {'walking' if sp.moving else 'resting'}",
                f"floor {s.free_m2:.2f} m2  known {s.known_m2:.2f} m2  coverage {s.coverage:.0%}",
                nxt,
            ],
            1395,
            830,
        )
        _legend(out, 975, 945)
        # 左下の右：カメラの横の説明
        _text(
            out,
            [
                "detection box:",
                "projected snoo",
                "(ray-checked)",
                "",
                "orange band:",
                "clear_cm corridor",
                "(steer arc x width)",
            ],
            730,
            580,
            0.5,
            22,
        )
        return out


class VideoWriter:
    """ffmpeg に BGR のコマを流して MP4（H.264）にする。"""

    def __init__(self, path: str | Path, fps: int = 15, size: tuple[int, int] = (W, H)):
        self.path = Path(path)
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{size[0]}x{size[1]}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, img: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg が失敗した（{self.path}）")
