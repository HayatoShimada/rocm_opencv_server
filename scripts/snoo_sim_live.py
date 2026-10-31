"""MuJoCo のシミュレーター（app/snoo_sim.py）を実機の速さで回し、猫（snoo）をゲームパッドで動かす。

    make sim-live                       # http://127.0.0.1:8731/sim
    make sim-live ARGS="--room home:3"  # 作った 30 坪の家（app/snoo_homes.py）
    make sim-live ARGS="--room session:explore_20261030_190244"  # 実走の部屋

車は地図のサーバーと同じ決め方（app/snoo_mapper.decide。segments・reverse・gear_segments・
long_step を受ける Pi として）で探る。時間は実機と同じ（app/snoo_sim.REAL_PACE：10 コマの
首振りに 14.4 秒、地図の返事を 11 秒待つ）。猫は ControlledSnoo で、/sim の画面の左のスティックで
歩かせ、ボタンで姿勢（座る・伏せる・寝る・立って見る・歩く）を決める。5 秒指令が来なければ、
自分で動く猫（--fallback、既定は録画から作った BehaviorSnoo）に任せる。

Pi が猫に合わせてすること（snoo_trucar の controller.py）も、ここでまねる：
- explore で snoo が explore_snoo_near_cm（150 cm）より近くに見えたら、止まってカメラで追う
  （watch）。
  見えなくなって 2 秒（lost_s）経ったら探索に戻る
- 自分から 30 cm より近づかない（guard。見守っている間は動かないので、近づくことも無い）

- GET  /sim：画面（app/static/sim/。入力は /controller/controller.js で /drive と同じ）
- POST /sim/cat：{forward, turn, posture?}。forward・turn は -1..1（turn は左回りが +）
- GET  /sim/frame.jpg?view=fpv|top|3d：車のカメラ・上から見た地図・3D。頼まれたときに描き、
  view ごとに FRAME_FPS より速くは描き直さない（前の絵を返す）
- GET  /sim/state：猫・車・次の 1 歩とその理由・Pi のまねの反応・ぶつかった数・調べた割合
- POST /sim/reset：{room, seed}。room は synthetic・home:<seed>・session:<名前か置き場>
- GET  /sim/rooms：選べる部屋

描くのは 1 本のスレッドだけ（EGL の文脈をスレッドの間で渡さない）。GPU の計算（CUDA・ROCm）は
使わない。
送り元の制限は地図のサーバーと同じやり方（SNOO_SIM_ALLOW。既定は SNOO_MAP_ALLOW と同じ）。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import logging
import math
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.roommap import LONG_STEP_M  # noqa: E402
from app.snoo import _allowed, parse_allow  # noqa: E402
from app.snoo_mapper import MapParams  # noqa: E402
from app.snoo_sim import (  # noqa: E402
    CAM_H_M,
    CAT_POSTURES,
    PAN_LIMIT_DEG,
    REAL_PACE,
    SNOO_RADIUS_M,
    ControlledSnoo,
    Room,
    Sim,
    SnooPose,
    find_free_spot,
    make_snoo,
    room_from_home,
    room_from_map,
    synthetic_room,
)
from app.snoo_teleop import NO_STORE, add_controller_static, static_file  # noqa: E402

log = logging.getLogger("snoo_sim_live")

HOST = os.getenv("SNOO_SIM_HOST", "0.0.0.0")
PORT = int(os.getenv("SNOO_SIM_PORT", "8731"))
ALLOW = os.getenv(
    "SNOO_SIM_ALLOW", os.getenv("SNOO_MAP_ALLOW", "192.168.0.188,100.73.14.8,127.0.0.1")
)
DATA = os.getenv("SNOO_MAP_DATA", "data/snoo_map/sessions")
# 1 刻み（Sim の秒）。猫と描く絵が飛ばないよう、Sim の advance をこれより細かく分ける
SUB_S = 0.05
# Pi のまね（snoo_trucar の config.py・guard.py と同じ値）
NEAR_CM = 150.0  # explore_snoo_near_cm
LOST_S = 2.0  # lost_s
GUARD_GAP_CM = 30.0  # guard の min_gap_cm
# view ごとの描き直しの上限（回/秒）
FRAME_FPS = {"fpv": 10.0, "top": 5.0, "3d": 8.0}
JPEG_QUALITY = 80
SESSION_MAX_STATIONS = 2000  # 地図のサーバーと同じ（探り続ける）


class Stopped(Exception):
    """reset・終えるときに、回しているシミュレーターを抜ける。"""


class RobotCmd(BaseModel):
    """A ボタン：車のマッピングを始める（true）・止める（false）。"""

    mapping: bool


class LiveSim(Sim):
    """Sim の時計（Sim.t）を実機の時計に合わせて回す。advance のたびに眠り、その間はロックを放す
    （画面の絵と state は、そのときに読む）。Pi が猫を見て止まる（watch）のもここでまねる。"""

    def __init__(self, room: Room, *args, lock, stop: threading.Event, speed=1.0, near_cm=NEAR_CM,
                 **kw):  # fmt: skip
        self._lock = lock
        self._stop = stop
        self.speed = speed
        self.near_cm = near_cm
        self._wall0 = time.monotonic()
        self._t0 = 0.0
        self.watching = False
        self.watches = 0
        # A でマッピングを始めるまで、車はその場で待つ（猫だけ動く）。B で止める
        self.mapping = False
        self.snoo_dist_cm: float | None = None
        super().__init__(room, *args, **kw)

    # --- 時計 ---

    def _pace(self) -> None:
        target = self._wall0 + (self.t - self._t0) / self.speed
        now = time.monotonic()
        wait = target - now
        if wait < -1.0:  # 重い計算で遅れたら、追いつこうと速く回さず、そこから数え直す
            self._wall0, self._t0, wait = now, self.t, 0.0
        self._lock.release()
        try:
            # 速く回すとき（試験）も少しは眠り、画面がロックを取れるようにする
            self._stop.wait(max(wait, 0.001))
        finally:
            self._lock.acquire()
        if self._stop.is_set():
            raise Stopped

    def advance(self, dt: float) -> None:
        left = dt
        while left > 1e-9:
            h = min(SUB_S, left)
            left -= h
            super().advance(h)
            if left > 1e-9:
                self.sync()  # 次の刻みの絵と猫の見え方のため（最後の刻みは呼ぶ側が sync する）
            self._pace()
            if self.phase in ("scan", "drive", "reverse") and self._snoo_near():
                self._watch()

    def think(self) -> dict:
        """次の 1 歩を決める前に、マッピングが止められていれば、その場で待つ（猫は動く）。"""
        phase = self.phase
        while not self.mapping:
            self.phase = "paused"
            self.advance(0.5)  # 車は止まっている（撮った後・決める前）。猫と時計だけ進む
        self.phase = phase
        return super().think()

    # --- 猫を見る（Pi のまね） ---

    def _snoo_dist_cm(self) -> float | None:
        """車のカメラから、見えている猫の体までの床の距離（cm）。見えていなければ None。"""
        if self.detection is None:
            return None
        cam = self.car.camera_pose()
        s = self.snoo.pose
        return max(0.0, math.dist((cam.x, cam.y), (s.x, s.y)) - SNOO_RADIUS_M) * 100

    def _snoo_near(self) -> bool:
        self.snoo_dist_cm = self._snoo_dist_cm()
        return (
            bool(self.near_cm)
            and self.snoo_dist_cm is not None
            and self.snoo_dist_cm < self.near_cm
        )

    def _watch(self) -> None:
        """止まって猫をカメラで追う。見えなくなって（遠くなって）LOST_S 経ったら戻る。"""
        c = self.car
        phase, pan, tilt = self.phase, c.pan_deg, c.tilt_deg
        self.phase, self.watching = "watch", True
        self.watches += 1
        lost = 0.0
        try:
            while lost < LOST_S:
                Sim.advance(self, SUB_S)
                s, cam = self.snoo.pose, c.camera_pose()
                bearing = math.atan2(s.y - cam.y, s.x - cam.x) - c.yaw
                bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
                c.pan_deg = max(-PAN_LIMIT_DEG, min(PAN_LIMIT_DEG, -math.degrees(bearing)))
                c.tilt_deg = 0.0
                self.sync()
                self._pace()
                lost = 0.0 if self._snoo_near() else lost + SUB_S
        finally:
            self.phase, self.watching = phase, False
            c.pan_deg, c.tilt_deg = pan, tilt
            self.sync()


# --- 部屋 ---


def resolve_session(name: str, data: Path) -> Path:
    p = Path(name)
    if p.is_dir() or p.suffix == ".json":
        return p
    return data / name


def make_room(spec: str, data: Path) -> Room:
    """synthetic・home:<seed>・session:<名前か置き場>。"""
    if spec == "synthetic":
        return synthetic_room()
    kind, _, arg = spec.partition(":")
    if kind == "home":
        from app.snoo_homes import make_home

        return room_from_home(make_home(int(arg or 0)))
    if kind == "session":
        path = resolve_session(arg, data)
        if not (path / "room.json").is_file() and not path.is_file():
            raise ValueError(f"{path} に room.json が無い")
        # 見ていない所も床にして壁で囲む（実走の間取りは狭いので、探る先が無くならないように）
        return room_from_map(path, unseen_floor=True)
    raise ValueError("room は synthetic・home:<seed>・session:<名前> のどれか")


@functools.cache
def home_seeds(n: int = 6, upto: int = 30) -> tuple[int, ...]:
    """地図（12 m 四方）に収まる、作った家のくじ。収まらない家もある（細長いとき）。"""
    from app.snoo_homes import make_home

    out = []
    for seed in range(1, upto):
        try:
            room_from_home(make_home(seed))
        except ValueError:
            continue
        out.append(seed)
        if len(out) >= n:
            break
    return tuple(out)


def list_rooms(data: Path) -> list[dict]:
    rooms = [{"room": "synthetic", "label": "作り物の居間（4 x 3 m）"}]
    rooms += [{"room": f"home:{s}", "label": f"作った 30 坪の家（くじ {s}）"} for s in home_seeds()]
    if data.is_dir():
        for d in sorted(data.iterdir(), reverse=True):
            if (d / "room.json").is_file():
                rooms.append({"room": f"session:{d.name}", "label": f"実走の部屋 {d.name}"})
    return rooms


# --- 回す ---


class Live:
    """回している Sim と、その周り（スレッド・描く・設定）。reset で作り直す。"""

    def __init__(
        self,
        room: str = "synthetic",
        seed: int = 0,
        rays: tuple[int, int] = (160, 120),
        speed: float = 1.0,
        fallback: str = "behavior",
        timeout_s: float = 5.0,
        server_s: float | None = None,
        near_cm: float = NEAR_CM,
        data: Path | str = DATA,
    ):
        self.rays, self.speed, self.fallback = rays, speed, fallback
        self.timeout_s, self.near_cm = timeout_s, near_cm
        self.pace = REAL_PACE if server_s is None else replace(REAL_PACE, server_s=server_s)
        self.data = Path(data)
        self.lock = threading.Lock()
        self.reset_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None
        self.room_spec, self.seed = room, seed
        self.render = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="render")
        self._renderers: dict = {}
        self._render_sim = None
        self._cache: dict[str, tuple[float, bytes]] = {}
        self.sim = self._build(room, seed, self.stop_event)

    def _build(self, spec: str, seed: int, stop: threading.Event) -> LiveSim:
        room = make_room(spec, self.data)
        rng = np.random.default_rng(seed)
        s = room.start
        # 猫は車の 2 m ほど前（近すぎると、車は始めから見守るだけになる）。立てる所を探す
        ahead = (s.x + 2.0 * math.cos(s.yaw), s.y + 2.0 * math.sin(s.yaw))
        start = find_free_spot(room, ahead, rng)
        start = SnooPose(start.x, start.y, start.yaw, False, "alert")
        fb = None if self.fallback == "none" else make_snoo(room, start, self.fallback, seed + 1)
        cat = ControlledSnoo(room, start, fb, timeout_s=self.timeout_s)
        params = MapParams(
            camera_height_m=CAM_H_M,
            segments=True,
            reverse=True,
            gear_segments=True,
            max_step_m=LONG_STEP_M,
            max_stations=SESSION_MAX_STATIONS,
        )
        sim = LiveSim(
            room, cat, rays=self.rays, params=params, seed=seed, sonar_map=True, pace=self.pace,
            lock=self.lock, stop=stop, speed=self.speed, near_cm=self.near_cm,
        )  # fmt: skip
        return sim

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run, args=(self.sim, self.stop_event), daemon=True, name="sim"
        )
        self.thread.start()

    def _run(self, sim: LiveSim, stop: threading.Event) -> None:
        self.lock.acquire()
        try:
            sim._wall0, sim._t0 = time.monotonic(), sim.t
            while not stop.is_set():
                # 調べ終えた・動けなくなったら返ってくる。少し待って撮り直す
                sim.explore(10**9)
                sim.advance(2.0)
        except Stopped:
            pass
        except Exception as e:  # 画面に出す（サーバーは落とさない）
            log.exception("シミュレーターが止まった")
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.lock.release()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)

    def reset(self, room: str | None = None, seed: int | None = None) -> None:
        with self.reset_lock:
            room = self.room_spec if room is None else room
            seed = self.seed if seed is None else seed
            stop = threading.Event()
            sim = self._build(room, seed, stop)  # 作れなければ ValueError（今のは止めない）
            self.stop()
            with self.lock:
                self.sim, self.stop_event, self.error = sim, stop, None
                self.room_spec, self.seed = room, seed
                self._cache.clear()
            self.start()

    def close(self) -> None:
        self.stop()
        self.render.submit(self._close_renderers).result(timeout=10)
        self.render.shutdown()

    # --- 描く（描くスレッドの中だけで呼ぶ） ---

    def _close_renderers(self) -> None:
        for r in self._renderers.values():
            if hasattr(r, "close"):
                r.close()
        self._renderers = {}
        self._render_sim = None

    def _draw(self, view: str) -> bytes:
        import mujoco

        from app.snoo_sim_render import GROUP_CAR, GROUP_DIM, TopDown, add_scene_overlays
        from app.snoo_sim_render import draw_camera_hud as hud

        sim = self.sim
        if self._render_sim is not sim:
            self._close_renderers()
            self._render_sim = sim
        r = self._renderers
        with self.lock:
            if view == "top":
                if "top" not in r:
                    r["top"] = TopDown(sim, 640, 560)
                img = r["top"].draw(sim)
            elif view == "fpv":
                if "fpv" not in r:
                    r["fpv"] = mujoco.Renderer(sim.model, 480, 640, max_geom=2000)
                    opt = mujoco.MjvOption()
                    opt.geomgroup[GROUP_DIM] = 1
                    opt.geomgroup[GROUP_CAR] = 0  # 車のカメラには車そのものを映さない
                    r["fpv_opt"] = opt
                r["fpv"].update_scene(sim.data, "car_cam", r["fpv_opt"])
                img = cv2.cvtColor(r["fpv"].render(), cv2.COLOR_RGB2BGR)
                img = hud(np.ascontiguousarray(img), sim)
            else:
                if "3d" not in r:
                    r["3d"] = mujoco.Renderer(sim.model, 360, 640, max_geom=20000)
                    opt = mujoco.MjvOption()
                    opt.geomgroup[GROUP_DIM] = 1
                    r["3d_opt"] = opt
                    r["3d_cam"] = mujoco.MjvCamera()
                cam = r["3d_cam"]
                c, s = sim.car, sim.snoo.pose
                # 車と猫の真ん中を、少し離れて斜めから見る（家は広いので、全体ではなく近くを）
                cam.lookat[:] = [(c.x + s.x) / 2, (c.y + s.y) / 2, 0.0]
                cam.distance = 1.6 + 1.2 * math.dist((c.x, c.y), (s.x, s.y))
                # azimuth は見る向き。車の後ろから、車の向きに見る
                cam.elevation, cam.azimuth = -50, math.degrees(c.yaw)
                r["3d"].update_scene(sim.data, cam, r["3d_opt"])
                add_scene_overlays(r["3d"].scene, sim)
                img = cv2.cvtColor(r["3d"].render(), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes()

    def frame(self, view: str) -> bytes:
        now = time.monotonic()
        hit = self._cache.get(view)
        if hit is not None and now - hit[0] < 1.0 / FRAME_FPS[view]:
            return hit[1]
        data = self.render.submit(self._draw, view).result(timeout=20)
        self._cache[view] = (time.monotonic(), data)
        return data

    # --- 様子 ---

    def state(self) -> dict:
        with self.lock:
            sim = self.sim
            s = sim.summary()
            c, cam, cat = sim.car, sim.car.camera_pose(), sim.snoo
            p = cat.pose
            a = sim.action or {}
            dist = sim._snoo_dist_cm()
            since = cat.since_command_s() if isinstance(cat, ControlledSnoo) else None
            return {
                "room": self.room_spec,
                "room_name": sim.room.name,
                "seed": self.seed,
                "running": self.thread is not None and self.thread.is_alive(),
                "error": self.error,
                "t": round(sim.t, 1),
                "cycle": sim.cycle,
                "phase": sim.phase,
                "phase_text": PHASE_TEXT.get(sim.phase, sim.phase),
                "robot": {
                    "x": round(cam.x, 3),
                    "y": round(cam.y, 3),
                    "yaw_deg": round(math.degrees(cam.yaw), 1),
                    "pan_deg": round(c.pan_deg, 1),
                    "tilt_deg": round(c.tilt_deg, 1),
                    "steer_deg": round(c.steer_deg, 1),
                },
                "action": {k: v for k, v in a.items() if k in ACTION_KEYS},
                "action_text": action_text(a),
                "cat": {
                    "x": round(p.x, 3),
                    "y": round(p.y, 3),
                    "yaw_deg": round(math.degrees(p.yaw), 1),
                    "posture": p.posture,
                    "moving": bool(p.moving),
                    "controlled": bool(getattr(cat, "controlled", False)),
                    "since_command_s": None if since is None else round(since, 1),
                    "blocked": bool(getattr(cat, "blocked", False)),
                    "fallback": self.fallback,
                },
                "reaction": {
                    "watching": sim.watching,
                    "mapping": sim.mapping,
                    "watches": sim.watches,
                    "snoo_seen": sim.detection is not None,
                    "snoo_dist_cm": None if dist is None else round(dist, 0),
                    "near_cm": sim.near_cm,
                    "guard": "hold" if dist is not None and dist < GUARD_GAP_CM else "ok",
                    "text": reaction_text(sim, dist),
                },
                "sonar_cm": None if sim.sonar_m is None else round(sim.sonar_m * 100, 0),
                "clear_cm": round(sim.clear_m * 100, 0),
                "stations": s.stations,
                "bumps": s.bumps,
                "bumps_snoo": sum(1 for b in sim.bump_log if b.get("snoo")),
                "stops": s.stops,
                "reverses": s.reverses,
                "distance_m": s.distance_m,
                "known_m2": s.known_m2,
                "covered_m2": s.covered_m2,
                "coverage": s.coverage,
                "kinds": dict(sim.kinds),
            }


ACTION_KEYS = {
    "action", "reason", "steer_deg", "speed", "distance_cm", "segments", "reverse", "gap",
    "stuck", "planner", "target",
}  # fmt: skip
PHASE_TEXT = {
    "start": "始めるところ",
    "scan": "止まって首を振って撮っている",
    "plan": "地図を直して次の 1 歩を決めている（地図のサーバーの返事を待つ間）",
    "drive": "前へ走っている",
    "reverse": "後ろへ走っている",
    "watch": "snoo が近くに見えるので、止まってカメラで追っている",
    "done": "止まっている",
}
REASON_TEXT = {
    "no_frontier": "見る所（境目）がもう無い",
    "unreachable": "行ける所が無い",
    "step_limit": "ステーションの数が上限",
    "stuck": "動けないのが続いた",
}


def action_text(a: dict) -> str:
    """次の 1 歩と、そう決めた理由（人が読む）。"""
    if not a:
        return "まだ決めていない"
    if a.get("action") != "move":
        r = str(a.get("reason", ""))
        return f"終える（{REASON_TEXT.get(r.split(':')[0], r)}）"
    backs = {bool(x.get("reverse", a.get("reverse"))) for x in a.get("segments") or [a]}
    if len(backs) == 2:
        what = "前と後ろを切り替えて向きを変える（切り返し・3 点ターン）"
    elif backs == {True}:
        what = "後ろへ下がる"
    else:
        what = "まだ見ていない所・確かめたい所が見える位置へ向かう"
    if a.get("gap"):
        what += "（狭い隙間を通る）"
    if a.get("stuck"):
        what = "動けないのが続くので、広く探した 1 歩：" + what
    # segments があれば合わせた長さ（distance_cm は最初の区間だけ）。ハンドルは最初の区間
    segs = a.get("segments") or [{"distance_cm": a.get("distance_cm", 0)}]
    cm = sum(float(x.get("distance_cm", 0)) for x in segs)
    steer = float(a.get("steer_deg", 0))
    return f"{what}（{cm:.0f} cm、ハンドル {steer:+.0f}°）"


def reaction_text(sim: LiveSim, dist: float | None) -> str:
    if sim.watching:
        return "snoo が 150 cm より近いので、探索を止めて見守っている（離れて 2 秒で戻る）"
    if dist is None:
        return "snoo は見えていない"
    if dist < GUARD_GAP_CM:
        return "snoo が 30 cm より近い（自分からは近づかない）"
    return f"snoo が {dist:.0f} cm に見えている"


# --- HTTP ---


class CatCommand(BaseModel):
    forward: float = Field(0.0, ge=-1, le=1)
    turn: float = Field(0.0, ge=-1, le=1)
    posture: str | None = None


class ResetBody(BaseModel):
    room: str | None = Field(None, max_length=200)
    seed: int | None = Field(None, ge=0, le=10**6)


def create_app(live: Live, allow: list | None = None, autostart: bool = True) -> FastAPI:
    from app.snoo_map_server import refused_page

    allow = parse_allow(ALLOW) if allow is None else allow
    app = FastAPI(title="snoo sim live")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.client.host if request.client else ""
        try:
            ok = _allowed((host, 0), allow)
        except ValueError:
            ok = False
        if not ok:
            if "text/html" in request.headers.get("accept", ""):
                return HTMLResponse(
                    refused_page(host).replace("SNOO_MAP_ALLOW", "SNOO_SIM_ALLOW"), 403
                )
            return JSONResponse({"detail": "この送り元からは受けない"}, 403)
        return await call_next(request)

    add_controller_static(app)

    @app.get("/sim", include_in_schema=False)
    def page():
        return static_file("sim", "index.html")

    @app.get("/sim/static/{name}", include_in_schema=False)
    def page_static(name: str):
        return static_file("sim", name)

    @app.post("/sim/cat")
    def cat(c: CatCommand):
        model = live.sim.snoo
        if not isinstance(model, ControlledSnoo):
            raise HTTPException(409, "この猫は手で動かせない")
        if c.posture is not None and c.posture not in CAT_POSTURES:
            raise HTTPException(400, f"posture は {', '.join(CAT_POSTURES)} のどれか")
        model.command(c.forward, c.turn, c.posture)
        p = model.pose
        return JSONResponse(
            {"ok": True, "posture": model.posture, "x": round(p.x, 3), "y": round(p.y, 3)},
            headers=NO_STORE,
        )

    @app.get("/sim/frame.jpg")
    def frame(view: str = "fpv"):
        if view not in FRAME_FPS:
            raise HTTPException(400, f"view は {', '.join(FRAME_FPS)} のどれか")
        try:
            data = live.frame(view)
        except Exception as e:
            log.exception("描けなかった")
            raise HTTPException(503, f"描けなかった（MUJOCO_GL=egl が要る）: {e}") from e
        return Response(data, media_type="image/jpeg", headers=NO_STORE)

    @app.get("/sim/state")
    def state():
        return JSONResponse(live.state(), headers=NO_STORE)

    @app.post("/sim/robot")
    def robot(c: RobotCmd):
        """A ボタン：車のマッピング（探る）を始める・止める。"""
        live.sim.mapping = bool(c.mapping)
        return {"mapping": live.sim.mapping}

    @app.post("/sim/reset")
    def reset(body: ResetBody | None = None):
        body = body or ResetBody()
        try:
            live.reset(body.room, body.seed)
        except (ValueError, OSError, KeyError) as e:
            raise HTTPException(400, f"部屋を作れなかった: {e}") from e
        return JSONResponse(live.state(), headers=NO_STORE)

    @app.get("/sim/rooms")
    def rooms():
        return {"rooms": list_rooms(live.data), "now": live.room_spec, "seed": live.seed}

    if autostart:
        live.start()
    return app


def _rays(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--room", default="synthetic", help="synthetic・home:<seed>・session:<名前>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rays", type=_rays, default=(160, 120), help="1 コマの光線（横x縦）")
    ap.add_argument("--speed", type=float, default=1.0, help="実機の何倍の速さで回すか")
    ap.add_argument("--fallback", default="behavior", choices=("behavior", "random", "none"),
                    help="手を離して 5 秒経ったときの猫")  # fmt: skip
    ap.add_argument("--timeout-s", type=float, default=5.0, help="この秒だけ指令が無ければ任せる")
    ap.add_argument("--server-s", type=float, help="地図のサーバーの返事を待つ秒（既定 11）")
    ap.add_argument(
        "--near-cm", type=float, default=NEAR_CM, help="この近さに見えたら見守る（0 で切る）"
    )
    ap.add_argument("--data", default=DATA, help="session:<名前> を探す置き場")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    live = Live(args.room, args.seed, args.rays, args.speed, args.fallback, args.timeout_s,
                args.server_s, args.near_cm, args.data)  # fmt: skip
    app = create_app(live)
    log.info("http://%s:%d/sim（部屋 %s、受ける送り元 %s）", args.host, args.port, args.room, ALLOW)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        live.close()


if __name__ == "__main__":
    main()
