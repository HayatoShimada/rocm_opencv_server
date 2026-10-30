"""PiCar-X を人が手で動かす（teleop）ための API と、ブラウザの操縦の画面（/drive）。

ブラウザ（ゲームパッド・キーボード・スマホの指）が 10 回/秒で指令を送り、Pi が 10 回/秒で
いちばん新しい指令を取りに来る。指令はメモリに 1 つだけ持つ（並べない。古い指令を後から
走らせない）。
Pi 側の決まり（teleop のモード・deadman・いつも効く安全）は docs/teleop.md。

- POST /v1/snoo/teleop：{seq, steer, speed, pan, tilt, mode?}。steer・speed・pan・tilt は -1..1。
  ここで Pi の単位に直し、上限で切る（下の TELEOP_*）。seq が前に受けたもの以下なら捨てる
  （届く順が入れ替わったとき、古い指令で上書きしない）
- POST /v1/snoo/teleop/stop：すぐ止める（speed 0。首はそのまま）。seq を見ない
- GET  /v1/snoo/teleop/latest?after=<seq>：Pi が 10 回/秒で問い合わせる。いちばん新しい指令と、
  受けてからの秒（age_s）。DEADMAN_S より古ければ止まる指令（speed 0。首はそのまま）を返す。
  Pi は今のモード・電池・clear_cm・実際に書いた指令をクエリで送る（画面に「Pi が受けているか」を
  出す）
- GET  /v1/snoo/teleop/status：画面が読む。操縦する人がいるか・Pi の様子・上限
- GET  /drive：操縦の画面（app/static/drive/）。GET /controller/<名前>：画面が共に使う JS
  （app/static/controller/。シミュレーターの /sim も同じものを使う）

秒はどれもこのサーバーの時計で数えるので、Pi とブラウザの時計を合わせなくてよい。
送り元の制限（SNOO_MAP_ALLOW）は地図のサーバーと同じ。スマホで開くときは、その IP を足す。
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

PREFIX = "/v1/snoo/teleop"
NO_STORE = {"Cache-Control": "no-store"}

# --- 上限（Pi の単位）。Pi 側でも同じ上限をもう一度当てる（docs/teleop.md） ---
# 前へ 15（約 25 cm/s。explore と同じ速さ）、後ろへ 10（後ろにはセンサーが無いので遅く）
TELEOP_MAX_SPEED = 15.0
TELEOP_MAX_REVERSE = 10.0
# ハンドル（度、右が +）。Pi の steer_max は 30 だが、いっぱいに切ると車輪がこすれるので少し手前
TELEOP_MAX_STEER_DEG = 28.0
# 首（度）。パンは右が +、チルトは上が +（0 が水平）。下を向きすぎると車の前しか映らない
TELEOP_PAN_DEG = 35.0
TELEOP_TILT_MIN_DEG = -20.0
TELEOP_TILT_MAX_DEG = 30.0
# 受けてからこれより古い指令は止まる指令にする。ブラウザは 0.1 秒ごとに送るので、2 回落ちても走る
DEADMAN_S = 0.3
# Pi に勧める問い合わせの回数（回/秒）
POLL_HZ = 10.0
# ブラウザの指令がこの秒のうちにあれば「操縦する人がいる」
OPERATOR_S = 2.0
# Pi がこの秒のうちに問い合わせていれば、Pi はつながっている
PI_SEEN_S = 1.0
# 画面から頼めるモード（Pi が supports に ui_mode を書いたときだけ Pi が受ける）
UI_MODES = ("teleop", "idle", "explore", "map")  # A ボタンでマッピング（explore）を始める
# 頼んだモードは、この秒を過ぎたら Pi に渡さない（前に押したボタンで、後からモードが変わらない）
MODE_REQUEST_TTL_S = 5.0
LIMITS = ("", "deadman", "poll", "guard", "battery", "sonar", "clear", "mode", "stopped")


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def to_pi(steer: float, speed: float, pan: float, tilt: float) -> dict:
    """-1..1 の指令を Pi の単位（度・picarx の速さ）にする。外れた値は端に寄せる。"""
    steer, speed = _clip(steer, -1, 1), _clip(speed, -1, 1)
    pan, tilt = _clip(pan, -1, 1), _clip(tilt, -1, 1)
    return {
        "steer_deg": round(steer * TELEOP_MAX_STEER_DEG, 1),
        "speed": round(speed * (TELEOP_MAX_SPEED if speed >= 0 else TELEOP_MAX_REVERSE), 1),
        "pan_deg": round(pan * TELEOP_PAN_DEG, 1),
        "tilt_deg": round(tilt * (TELEOP_TILT_MAX_DEG if tilt >= 0 else -TELEOP_TILT_MIN_DEG), 1),
    }


class TeleopCommand(BaseModel):
    seq: int = Field(ge=0)
    steer: float = Field(0.0, ge=-1, le=1)
    speed: float = Field(0.0, ge=-1, le=1)
    pan: float = Field(0.0, ge=-1, le=1)
    tilt: float = Field(0.0, ge=-1, le=1)
    # Pi に頼むモード（teleop・idle）。無ければ頼まない
    mode: str | None = None


class StopBody(BaseModel):
    reason: str = Field("", max_length=64)


def _num(v) -> float | None:
    return float(v) if isinstance(v, int | float) and math.isfinite(v) else None


class Teleop:
    """いちばん新しい指令と、Pi・ブラウザの様子。メモリだけに持つ（10 回/秒で来るので SSD に
    書かない）。"""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        # seq はサーバーが数える通し番号（受けた指令と stop のたびに 1 足す）。Pi は after に渡す
        self.seq = 0
        self.client_seq = -1  # ブラウザの seq（並びの入れ替わりを捨てる）
        self.cmd = to_pi(0, 0, 0, 0)
        self.raw = {"steer": 0.0, "speed": 0.0, "pan": 0.0, "tilt": 0.0}
        self.t: float | None = None  # 受けた時刻（None はまだ受けていない）
        self.stopped = True
        self.stop_reason = ""
        self.last_post: float | None = None
        self.mode_req: tuple[str, float] | None = None  # (頼んだモード, 時刻)
        # 頼んだ回の番号。Pi は前に受けた番号より大きいときだけモードを変える（同じ頼みを
        # 5 秒の間に何度も受けて、USR の長押しで idle に戻したのを teleop に戻さないように）
        self.mode_req_id = 0
        self.pi: dict[str, Any] = {}

    # --- ブラウザから ---

    def command(self, c: TeleopCommand) -> dict:
        with self.lock:
            now = self.clock()
            self.last_post = now
            if c.mode is not None:
                if c.mode not in UI_MODES:
                    raise ValueError(f"mode は {', '.join(UI_MODES)} のどれか")
                self.mode_req = (c.mode, now)
                self.mode_req_id += 1
            if c.seq <= self.client_seq:
                return {"accepted": False, "reason": "old_seq", **self._status(now)}
            self.client_seq = c.seq
            self.seq += 1
            self.raw = {"steer": c.steer, "speed": c.speed, "pan": c.pan, "tilt": c.tilt}
            self.cmd = to_pi(c.steer, c.speed, c.pan, c.tilt)
            self.t = now
            self.stopped = False
            self.stop_reason = ""
            return {"accepted": True, **self._status(now)}

    def stop(self, reason: str = "") -> dict:
        with self.lock:
            now = self.clock()
            self.last_post = now
            self.seq += 1
            self.cmd = {**self.cmd, "speed": 0.0}
            self.raw = {**self.raw, "speed": 0.0}
            self.t = now
            self.stopped = True
            self.stop_reason = reason or "stop"
            return {"accepted": True, **self._status(now)}

    # --- Pi から ---

    def latest(self, after: int, pi: dict) -> dict:
        with self.lock:
            now = self.clock()
            self.pi = {**pi, "t": now}
            age = None if self.t is None else now - self.t
            stale = age is None or age > DEADMAN_S
            cmd = dict(self.cmd)
            if stale or self.stopped:
                cmd["speed"] = 0.0  # 首とハンドルはそのまま（止まったときに首が跳ねない）
            mode = None
            if self.mode_req is not None and now - self.mode_req[1] <= MODE_REQUEST_TTL_S:
                mode = self.mode_req[0]
            return {
                "seq": self.seq,
                "changed": self.seq > after,
                "command": cmd,
                "age_s": None if age is None else round(age, 3),
                "stale": stale,
                "stop": stale or self.stopped,
                "reason": "deadman" if stale and not self.stopped else self.stop_reason,
                "operator": self._operator(now),
                "request_mode": mode,
                "request_id": self.mode_req_id,
                "deadman_s": DEADMAN_S,
                "poll_hz": POLL_HZ,
            }

    # --- 画面へ ---

    def _operator(self, now: float) -> dict:
        age = None if self.last_post is None else now - self.last_post
        return {
            "present": age is not None and age <= OPERATOR_S,
            "last_post_age_s": None if age is None else round(age, 2),
        }

    def _pi(self, now: float) -> dict | None:
        if not self.pi:
            return None
        age = now - self.pi["t"]
        out = {k: v for k, v in self.pi.items() if k != "t"}
        out["age_s"] = round(age, 2)
        out["connected"] = age <= PI_SEEN_S
        # teleop のモードで問い合わせていれば、Pi は手の指令で走る
        out["accepted"] = out["connected"] and out.get("mode") == "teleop"
        return out

    def _status(self, now: float) -> dict:
        age = None if self.t is None else now - self.t
        return {
            "seq": self.seq,
            "command": dict(self.cmd),
            "raw": dict(self.raw),
            "age_s": None if age is None else round(age, 3),
            "stopped": self.stopped,
            "operator": self._operator(now),
            "pi": self._pi(now),
        }

    def status(self) -> dict:
        with self.lock:
            now = self.clock()
            return {
                **self._status(now),
                "request_mode": None
                if self.mode_req is None or now - self.mode_req[1] > MODE_REQUEST_TTL_S
                else self.mode_req[0],
                "limits": LIMITS_INFO,
            }


LIMITS_INFO = {
    "max_speed": TELEOP_MAX_SPEED,
    "max_reverse": TELEOP_MAX_REVERSE,
    "max_steer_deg": TELEOP_MAX_STEER_DEG,
    "pan_deg": TELEOP_PAN_DEG,
    "tilt_min_deg": TELEOP_TILT_MIN_DEG,
    "tilt_max_deg": TELEOP_TILT_MAX_DEG,
    "deadman_s": DEADMAN_S,
    "poll_hz": POLL_HZ,
    "cm_s_per_speed": 1.7,
}


def create_router(teleop: Teleop | None = None) -> APIRouter:
    t = teleop or Teleop()
    router = APIRouter(prefix=PREFIX)

    @router.post("")
    def post_command(c: TeleopCommand):
        try:
            out = t.command(c)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return JSONResponse(out, headers=NO_STORE)

    @router.post("/stop")
    def post_stop(body: StopBody | None = None):
        return JSONResponse(t.stop(body.reason if body else ""), headers=NO_STORE)

    @router.get("/latest")
    def get_latest(
        after: int = Query(0, ge=0),
        mode: str | None = Query(None, max_length=32),
        battery_v: float | None = None,
        clear_cm: float | None = None,
        sonar_cm: float | None = None,
        steer_deg: float | None = None,
        speed: float | None = None,
        pan_deg: float | None = None,
        tilt_deg: float | None = None,
        limit: str | None = Query(None, max_length=64),
        supports: str | None = Query(None, max_length=200),
    ):
        """Pi が 10 回/秒で問い合わせる。steer_deg・speed・pan_deg・tilt_deg は Pi が今
        実際に書いている値（安全の層を通した後）、limit は手の指令より弱めた理由。"""
        pi = {
            "mode": mode,
            "battery_v": _num(battery_v),
            "clear_cm": _num(clear_cm),
            "sonar_cm": _num(sonar_cm),
            "applied": {
                "steer_deg": _num(steer_deg),
                "speed": _num(speed),
                "pan_deg": _num(pan_deg),
                "tilt_deg": _num(tilt_deg),
            },
            "limit": [x for x in (limit or "").split(",") if x],
            "supports": [s for s in (supports or "").split(",") if s],
        }
        return JSONResponse(t.latest(after, pi), headers=NO_STORE)

    @router.get("/status")
    def get_status():
        return JSONResponse(t.status(), headers=NO_STORE)

    return router


# --- 画面 ---

STATIC = Path(__file__).parent / "static"
MEDIA = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}
_STATIC_NAME = re.compile(r"[a-z0-9_-]{1,40}\.(js|css|html)")


def static_file(folder: str, name: str) -> FileResponse:
    """app/static/<folder>/<name> を返す。名前は決まった形だけ（../ などで外を読ませない）。"""
    if not _STATIC_NAME.fullmatch(name):
        raise HTTPException(404, "無い")
    path = STATIC / folder / name
    if not path.is_file():
        raise HTTPException(404, "無い")
    return FileResponse(path, media_type=MEDIA[path.suffix], headers=NO_STORE)


def add_controller_static(app: FastAPI) -> None:
    """GET /controller/<名前>：ゲームパッド・キーボード・指の入力と送る周りの JS（共に使う）。"""

    @app.get("/controller/{name}", include_in_schema=False)
    def controller_static(name: str):
        return static_file("controller", name)


def add_teleop(app: FastAPI, teleop: Teleop | None = None) -> Teleop:
    """地図のサーバーに teleop の API と /drive の画面を足す。"""
    teleop = teleop or Teleop()
    app.include_router(create_router(teleop))
    add_controller_static(app)

    @app.get("/drive", include_in_schema=False)
    def drive_page():
        return static_file("drive", "index.html")

    @app.get("/drive/static/{name}", include_in_schema=False)
    def drive_static(name: str):
        return static_file("drive", name)

    return teleop
