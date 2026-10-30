"""AI のエージェント（Claude Code など）が PiCar-X を見て・印を付けて・動かすための API。

人が手で何度も走らせなくても、エージェントが自分でロボットを動かして、部屋の意味のあるもの
（家具・ケーブル・snoo の好きな所）と、困りごと（見落とし・地図の誤り）を見つけられるようにする。
地図のサーバー（app/snoo_map_server.py）に /v1/snoo/agent/ として足す。読み込めなくても
地図のサーバーは動かす（create_app で包む）。送り元の制限（SNOO_MAP_ALLOW）は地図のサーバーと同じ。
仕様と Pi 側の作り方は docs/agent_robot.md。

- GET  state：Pi の今（検出サーバーの今のフレーム・ステーションの meta・Pi の問い合わせ）、
  セッション・ステーション・最後の返事・SLAM の位置。分からない欄は null
- GET  frame.jpg・frame.json：今のカメラの画像と、その欄（猫の距離と向きも）
- GET  map.png・map.json：セッションの間取りを上から見た絵と、絵の画素と座標の対応
- POST annotate・GET annotations：部屋の印（data/snoo_map/annotations.jsonl）
- POST issue：困りごとの下書き（data/snoo_agent/issues/<時刻>/。GitHub には出さない）
- POST say・GET say/next・GET say/<名前>.wav：日本語の声（16 kHz モノラルの wav）と、
  Pi が取りに来て鳴らす順番待ち
- POST command・GET commands/next・POST command/<id>/result・GET command/<id>・GET commands：
  Pi への指令（look・move・scan・stop）。move はここで確かめてから並べる

指令はまだ Pi が実行しない（Pi が commands/next を問い合わせるまで pi_support は false）。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from app import atomic
from app.clearance import Camera, floor_point
from app.roommap import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    Grid,
    Pose,
    blocked_cells,
    clip_pieces,
    frontiers,
    moves_are_clear,
)
from app.snoo_map_server import BadRequest, check_id, read_poses, station_dir, station_ids
from app.snoo_mapper import pose_dict
from app.snoo_twin import _SENDER, _read_json, list_sessions, read_live, slam_state

log = logging.getLogger("snoo_agent")

PREFIX = "/v1/snoo/agent"
# エージェントの置き場（指令・声・困りごとの下書き）
AGENT_DATA = os.getenv("SNOO_AGENT_DATA", "data/snoo_agent")
NO_STORE = {"Cache-Control": "no-store"}

# --- move の決まり（docs/agent_robot.md の「安全」）。Pi 側でも同じ決まりをもう一度当てる ---
# 1 回の move の長さの上限。サーバーの 1 歩（40 cm）より短くし、こまめに見直させる
MAX_MOVE_CM = 30.0
# 速さの上限（picarx の 0..100）。explore は 15、snoo のそばでゆっくり動くときが 12
MAX_MOVE_SPEED = 10.0
# ハンドルの上限（度）。Pi の限り
MAX_STEER_DEG = 30.0
# 猫がこれより近くに見えていたら動かさない
CAT_NEAR_CM = 150.0
# 猫とみなす確からしさ（Pi と同じ。snoo かどうかは問わない。どの猫にも近づかない）
CAT_MIN = 0.35
# snoo とみなす確からしさ（Pi の [control] と同じ）
SNOO_MIN = 0.6
# 今のフレームがこれより古ければ、猫が近くにいないと言えないので動かさない
FRESH_FRAME_S = 3.0
# SLAM の位置を今の位置とみなすのは、status.json がこれより新しいときだけ
SLAM_FRESH_S = 30.0
# 首の向きの限り（Pi の config の pan_min..pan_max・tilt_min..tilt_max）
PAN_LIMIT_DEG = 35.0
TILT_LIMIT_DEG = 35.0
# 並べた指令は、この秒数のうちに Pi が受けなければ実行しない（確かめた時の様子と変わるため）
COMMAND_TTL_S = 15.0
STOP_TTL_S = 60.0
# Pi がこの秒数のうちに commands/next を問い合わせていれば、指令を受けられる Pi とみなす
PI_SEEN_S = 10.0
# Pi がこの秒数より長く問い合わせなければ、動いている指令を止める（docs の「heartbeat」）
HEARTBEAT_S = 3.0
# 床からレンズの中心まで（cm。地図のサーバーの DEFAULT_CAMERA_HEIGHT_CM と同じ）
CAMERA_HEIGHT_CM = 8.0

# --- 声 ---
SAY_MAX_CHARS = 200
SAY_RATE = 16000
# 鳴らす順番待ちの声は、この秒数を過ぎたら Pi に渡さない（古い知らせを今鳴らさない）
SAY_TTL_S = 120.0

# --- 困りごとの下書き・印 ---
MAX_ISSUE_IMAGES = 8
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TEXT = 20000

KINDS = ("look", "move", "scan", "stop")
DONE_STATES = {"done", "refused", "failed", "expired", "cancelled"}
RESULT_STATES = {"accepted", "running"} | DONE_STATES
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SNAPSHOT = re.compile(r"[A-Za-z0-9_-]{1,64}\.jpg")
_WAV = re.compile(r"[0-9a-f]{16}\.wav")


# --- 読む（どれも無ければ None） ---


def _finite(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _num(v) -> float | None:
    return float(v) if _finite(v) else None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def latest_live(live_dir: Path | None, sender: str | None = None) -> dict | None:
    """検出サーバーが書いた、いちばん新しいフレームの欄（sender を決めればその送り元）。"""
    rows = read_live(live_dir)
    if sender is not None:
        rows = [r for r in rows if r.get("sender") == sender]
    return rows[0] if rows else None


def cat_place(det: dict, live: dict) -> dict:
    """猫の箱の下の端（足もと）が床にあるとして、カメラからの距離と向きを見積もる。

    箱の下の真ん中の画素を床に当てる（app/clearance.floor_point。カメラの高さ 8 cm）。
    箱が画面の下の端で切れていれば、足もとはもっと手前なので、距離は多めに出る（clipped）。
    水平より上なら（家具の上・遠い）距離は null。bearing_deg は車の前から右が +
    （Pi の pan と同じ）。
    """
    box = det.get("box")
    out = {
        "box": box,
        "cat": det.get("cat"),
        "snoo": det.get("snoo"),
        "distance_cm": None,
        "bearing_deg": None,
        "clipped": None,
    }
    cmd = live.get("cmd") or {}
    pan, tilt = _num(cmd.get("pan_deg")), _num(cmd.get("tilt_deg"))
    size = live.get("size") or [640, 480]
    if not (isinstance(box, list) and len(box) == 4 and all(_finite(v) for v in box)):
        return out
    w, h = float(size[0]), float(size[1])
    u = (box[0] + box[2]) / 2 * w
    out["clipped"] = box[3] >= 0.98
    if pan is None or tilt is None:  # 版 1 のヘッダー：首の向きが分からない
        return out
    cam = Camera.from_pi(tilt, pan, height_cm=CAMERA_HEIGHT_CM, width=int(w), height=int(h))
    fx, fy = floor_point(cam, u, min(box[3], 1.0) * h)
    fx, fy = float(fx), float(fy)
    if math.isfinite(fx) and math.isfinite(fy):
        out["distance_cm"] = round(math.hypot(fx, fy), 1)
        out["bearing_deg"] = round(-math.degrees(math.atan2(fy, fx)), 1)
    else:  # 向きだけは出せる（箱の真ん中の画素の向き）
        cx = (w - 1) / 2
        out["bearing_deg"] = round(pan + math.degrees(math.atan2(u - cx, cam.f)), 1)
    return out


def cats_in(live: dict | None) -> list[dict]:
    if not live:
        return []
    dets = live.get("dets") if isinstance(live.get("dets"), list) else []
    return [
        cat_place(d, live)
        for d in dets
        if isinstance(d, dict) and _finite(d.get("cat")) and d["cat"] >= CAT_MIN
    ]


def best_snoo(cats: list[dict]) -> dict | None:
    """Pi と同じ選び方：cat ≥ 0.35 かつ snoo ≥ 0.6 の中で cat × snoo がいちばん大きい箱。"""
    ok = [c for c in cats if _finite(c.get("snoo")) and c["snoo"] >= SNOO_MIN]
    return max(ok, key=lambda c: c["cat"] * c["snoo"]) if ok else None


def current_session(root: Path, session: str | None) -> Path | None:
    """session を決めればそれ、無ければいちばん新しく書いたセッション。"""
    if session is not None:
        try:
            check_id(session)
        except BadRequest as e:
            raise HTTPException(400, str(e)) from e
        d = root / session
        if not d.is_dir():
            raise HTTPException(404, "そのセッションは無い")
        return d
    rows = list_sessions(root)
    return root / rows[0]["id"] if rows else None


def last_reply(session: Path, ids: list[int]) -> tuple[int | None, dict | None]:
    """いちばん新しい reply.json（地図のサーバーが Pi に返したもの）と、そのステーション。"""
    for i in reversed(ids[-5:]):
        r = _read_json(station_dir(session, i) / "reply.json")
        if isinstance(r, dict):
            return i, r
    return None, None


def slam_pose(session: Path, now: float) -> dict | None:
    sl = slam_state(session)
    if not sl or not isinstance(sl.get("pose"), dict):
        return None
    p = sl["pose"]
    status = _read_json(session / "slam" / "status.json") or {}
    updated = _num(status.get("last_update"))
    return {
        "state": sl.get("state"),
        "x": _num(p.get("x")),
        "y": _num(p.get("y")),
        "yaw_deg": None if _num(p.get("yaw")) is None else round(math.degrees(p["yaw"]), 2),
        "t_capture_ns": p.get("t_capture_ns"),
        # 最後のキーフレームを撮ってからの秒数（status.json を書いた後に過ぎた分も足す）
        "age_s": None
        if updated is None or _num(sl.get("age_s")) is None
        else round(sl["age_s"] + max(0.0, now - updated), 2),
        "updated_age_s": None if updated is None else round(max(0.0, now - updated), 2),
        "rmse_m": sl.get("rmse_m"),
    }


def robot_pose(session: Path, now: float) -> tuple[Pose | None, str]:
    """move を確かめるときの今の位置（セッションの座標）と、どこから取ったか。

    SLAM が ok で新しければその位置（走った後も追える）、無ければ最後のステーションの位置。
    """
    sl = slam_pose(session, now)
    if (
        sl
        and sl["state"] == "ok"
        and sl["x"] is not None
        and sl["y"] is not None
        and sl["yaw_deg"] is not None
        and sl["updated_age_s"] is not None
        and sl["updated_age_s"] <= SLAM_FRESH_S
    ):
        return Pose(sl["x"], sl["y"], math.radians(sl["yaw_deg"])), "slam"
    poses = read_poses(session)
    if poses:
        last = max(poses)
        return poses[last], f"station:{last}"
    return None, "none"


# --- 間取り ---


class SessionGrid:
    """地図のサーバーが最後に書いた間取り（grid.png・high.png・room.json）。次の 1 歩を決めた格子と
    同じもの（ステーションで見た所・走った所・SLAM の分を足した後）。"""

    def __init__(self, session: Path):
        room = _read_json(session / "room.json")
        img = cv2.imread(str(session / "grid.png"), cv2.IMREAD_GRAYSCALE)
        if not isinstance(room, dict) or img is None:
            raise FileNotFoundError("grid.png・room.json がまだ無い")
        window = room.get("window") or {}
        if window.get("map") != "session":  # 古い版の窓の座標の間取りは使わない
            raise FileNotFoundError("間取りがセッションの座標で書かれていない（古い版）")
        res = float(room["resolution_cm"]) / 100
        n = int(room["size"][0])
        # grid_png の逆（PNG の行 0 が +y の端、列 0 が -x の端）
        cells = img[::-1].T
        if cells.shape != (n, n):
            raise FileNotFoundError("grid.png の大きさが room.json と合わない")
        self.state = np.full(cells.shape, UNKNOWN, np.uint8)
        self.state[cells == 255] = FREE
        self.state[cells == 0] = OCCUPIED
        self.grid = Grid(size_m=n * res, res=res)
        self.grid.origin = float(room["origin"][0])
        high = cv2.imread(str(session / "high.png"), cv2.IMREAD_GRAYSCALE)
        self.high = high[::-1].T > 0 if high is not None and high.shape == img.shape else None
        self.version = max(_mtime(session / "grid.png") or 0, _mtime(session / "room.json") or 0)


def check_move(
    grid: SessionGrid, pose: Pose, steer_deg: float, distance_cm: float, reverse: bool
) -> list[dict]:
    """間取りで move を確かめる。前へは clip_pieces（ぶつかる手前で縮めたら通らない）と
    moves_are_clear の両方、後ろへは moves_are_clear（後ろの頭 15 cm だけは見ていない所も通す）。"""
    m = distance_cm / 100
    out = []
    if not reverse:
        blocked = blocked_cells(grid.grid, grid.state, pose)
        clipped = clip_pieces(grid.grid, blocked, pose, [(steer_deg, m)])
        clear_m = sum(x for _, x in clipped)
        out.append(
            {
                "name": "clip_pieces",
                "ok": clear_m >= m - 0.005,
                "detail": f"ぶつからずに走れるのは {clear_m * 100:.0f} cm（頼んだのは "
                f"{distance_cm:.0f} cm）",
            }
        )
    ok = moves_are_clear(grid.grid, grid.state, pose, [(steer_deg, m, reverse)])
    out.append(
        {
            "name": "moves_are_clear",
            "ok": bool(ok),
            "detail": "車の真ん中が障害物・見ていない所に入らない"
            if ok
            else "車の真ん中が障害物（車の大きさ分広げたもの）か、見ていない所に入る",
        }
    )
    return out


def render_map(
    grid: SessionGrid,
    stations: dict[int, Pose],
    robot: Pose | None,
    robot_source: str,
    trail: list,
    target: Pose | None,
    annotations: list[dict],
    px: int = 6,
    margin_m: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """上から見た絵（BGR）と、画素と座標の対応。色は snoo_mapper.render_topdown に合わせる
    （白 = 床、黒 = 障害物、青っぽい = 上の層、灰 = 見ていない）。橙 = 境目（frontiers）、
    青い丸 = ステーション、赤 = 車の今の位置、緑の丸 = 次に撮る位置、紫 = エージェントの印。
    cv2 は日本語を描けないので、印は番号だけ描き、名前は map.json に載せる。"""
    g, state = grid.grid, grid.state
    known = np.argwhere(state != UNKNOWN)
    pts = [list(g.cell(p.x, p.y)) for p in stations.values()]
    if robot is not None:
        pts.append(list(g.cell(robot.x, robot.y)))
    pts = np.array([[int(a), int(b)] for a, b in pts] + known.tolist() or [[g.n // 2] * 2])
    m = int(margin_m / g.res)
    x0, y0 = np.maximum(pts.min(0) - m, 0)
    x1, y1 = np.minimum(pts.max(0) + m + 1, g.n)
    sub = state[x0:x1, y0:y1]
    img = np.full((*sub.shape, 3), 150, np.uint8)
    img[sub == FREE] = (235, 235, 235)
    img[sub == OCCUPIED] = (30, 30, 30)
    if grid.high is not None:
        hi = grid.high[x0:x1, y0:y1]
        img[hi & (sub != OCCUPIED)] = (200, 140, 90)
    img[frontiers(state)[x0:x1, y0:y1]] = (0, 150, 255)
    img = np.ascontiguousarray(img.transpose(1, 0, 2)[::-1])
    img = cv2.resize(img, None, fx=px, fy=px, interpolation=cv2.INTER_NEAREST)

    def to_px(x, y):
        ix = (x - g.origin) / g.res - x0
        iy = (y - g.origin) / g.res - y0
        return int(round(ix * px)), int(round((y1 - y0 - iy) * px))

    for xm in range(math.ceil(g.origin + x0 * g.res), math.floor(g.origin + x1 * g.res) + 1):
        u, _ = to_px(xm, 0)
        cv2.line(img, (u, 0), (u, img.shape[0]), (120, 180, 120), 1)
    for ym in range(math.ceil(g.origin + y0 * g.res), math.floor(g.origin + y1 * g.res) + 1):
        _, v = to_px(0, ym)
        cv2.line(img, (0, v), (img.shape[1], v), (120, 180, 120), 1)
    if len(trail) > 1:
        cv2.polylines(
            img, [np.array([to_px(x, y) for x, y in trail], np.int32)], False, (160, 100, 160), 1
        )

    def arrow(p: Pose, color, r: int, length: float = 0.25):
        a = to_px(p.x, p.y)
        b = to_px(p.x + length * math.cos(p.yaw), p.y + length * math.sin(p.yaw))
        cv2.circle(img, a, r, color, 2)
        cv2.arrowedLine(img, a, b, color, 2, tipLength=0.3)
        return a

    for i, p in sorted(stations.items()):
        a = arrow(p, (220, 120, 0), 6)
        cv2.putText(img, str(i), (a[0] + 7, a[1] - 7), 0, 0.4, (220, 120, 0), 1)
    if target is not None:
        arrow(target, (0, 160, 0), 6)
    for k, a in enumerate(annotations, 1):
        u, v = to_px(a["x"], a["y"])
        cv2.drawMarker(img, (u, v), (200, 0, 200), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(img, f"A{k}", (u + 7, v + 14), 0, 0.45, (200, 0, 200), 1)
    if robot is not None:
        arrow(robot, (0, 0, 230), 9, 0.3)
    cv2.putText(img, f"1 grid = 1 m  robot: {robot_source}", (6, 16), 0, 0.45, (0, 0, 200), 1)
    info = {
        "px_per_cell": px,
        "resolution_m": g.res,
        # 画素 (u, v) の座標: x = x_at_u0 + u / px_per_m、y = y_at_v0 - v / px_per_m
        "px_per_m": px / g.res,
        "x_at_u0": round(g.origin + x0 * g.res, 4),
        "y_at_v0": round(g.origin + y1 * g.res, 4),
        "size_px": [int(img.shape[1]), int(img.shape[0])],
        "frontier_cells": int(frontiers(state).sum()),
    }
    return img, info


# --- 声 ---


def resample_to(x: np.ndarray, sr: int, rate: int = SAY_RATE) -> np.ndarray:
    """sr の音を rate にする。整数分の 1 なら、窓をかけた sinc の低域通過をかけて間引く。"""
    x = np.asarray(x, np.float64)
    if sr == rate:
        return x
    cutoff = 0.45 * rate / sr  # 新しいナイキストの 90%（sr に対する比）
    n = np.arange(-50, 51)
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(len(n))
    y = np.convolve(x, h / h.sum(), mode="same")
    if sr % rate == 0:
        return y[:: sr // rate]
    t = np.arange(0, len(x) / sr, 1 / rate)
    return np.interp(t, np.arange(len(x)) / sr, y)


def pyopenjtalk_wav(text: str) -> bytes:
    """pyopenjtalk で読み上げ、16 kHz・モノラル・16 ビットの wav にする。
    初めて使うときに辞書（23 MB）を GitHub から取ってくる。"""
    import pyopenjtalk

    x, sr = pyopenjtalk.tts(text)  # int16 の幅の float64、48 kHz
    return wav_bytes(resample_to(x, int(sr)), SAY_RATE)


def wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    pcm = np.clip(np.round(samples), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# --- 置き場 ---


class JsonLines:
    """1 行 1 件の JSON のファイル。書くのは足すだけ（同じプロセスの中はロックで守る）。"""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def read(self) -> list[dict]:
        try:
            text = self.path.read_text()
        except OSError:
            return []
        out = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:  # 書きかけの最後の行など
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    def append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def next_id(self) -> int:
        """ロックの中で呼ぶ。今までの id の最大 + 1（1 から）。"""
        return max([int(r.get("id", 0)) for r in self.read()] + [0]) + 1


# --- 受ける形 ---


class Annotation(BaseModel):
    session: str | None = None
    x: float
    y: float
    label: str = Field(min_length=1, max_length=80)
    note: str = Field("", max_length=2000)
    # "live"（今のフレームを写しておく）・"station:<session>/<id>/<file>"・"snapshot:<名前>.jpg"・
    # そのほかの文字列（そのまま残す）
    image_ref: str | None = Field(None, max_length=300)


class IssueImage(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    data_b64: str


class Issue(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    body: str = Field("", max_length=MAX_TEXT)
    # 参照（"live"・"map"・"snapshot:<名前>.jpg"・"station:<session>/<id>/<file>"）か、画像そのもの
    images: list[str | IssueImage] = Field(default_factory=list, max_length=MAX_ISSUE_IMAGES)
    session: str | None = None


class Say(BaseModel):
    text: str = Field(min_length=1, max_length=SAY_MAX_CHARS)
    queue: bool = True  # Pi が鳴らす順番待ちに入れるか


class Command(BaseModel):
    kind: str
    session: str | None = None
    # look
    pan_deg: float | None = None
    tilt_deg: float | None = None
    # move
    steer_deg: float | None = None
    distance_cm: float | None = None
    speed: float | None = None
    reverse: bool = False
    note: str = Field("", max_length=500)


class Result(BaseModel):
    status: str
    reason: str | None = Field(None, max_length=500)
    # 実際に走った長さ・止めた理由など、Pi が付けたいもの
    detail: dict[str, Any] = Field(default_factory=dict)


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _bad(msg: str) -> HTTPException:
    return HTTPException(400, msg)


def create_router(
    root: Path,
    live_dir: Path | None,
    agent_dir: Path | str = AGENT_DATA,
    annotations_path: Path | None = None,
    tts: Callable[[str], bytes] | None = None,
    clock: Callable[[], float] = time.time,
) -> APIRouter:
    """root はセッションの置き場（地図のサーバーの DATA）。印は既定で root の隣の
    annotations.jsonl（data/snoo_map/annotations.jsonl）に書く。"""
    root = Path(root)
    agent = Path(agent_dir)
    notes = JsonLines(annotations_path or root.parent / "annotations.jsonl")
    commands = JsonLines(agent / "commands.jsonl")
    results = JsonLines(agent / "command_results.jsonl")
    says = JsonLines(agent / "say.jsonl")
    tts_dir = agent / "tts"
    snapshots = agent / "snapshots"
    tts = tts or pyopenjtalk_wav
    tts_lock = threading.Lock()  # pyopenjtalk は同時に呼ばない
    # Pi が commands/next で書いてきた今の様子（メモリだけ。1 秒に何回も来るので SSD に書かない）
    pi: dict[str, Any] = {}
    router = APIRouter(prefix=PREFIX)

    def pi_status() -> dict | None:
        if not pi:
            return None
        return {**pi, "age_s": round(max(0.0, clock() - pi["t"]), 2)}

    def pi_support() -> bool:
        return bool(pi) and clock() - pi["t"] <= PI_SEEN_S

    # --- state ---

    def build_state(session_id: str | None) -> dict:
        now = clock()
        live = latest_live(live_dir)
        cats = cats_in(live)
        cmd = (live or {}).get("cmd") or {}
        session = current_session(root, session_id)
        ids = station_ids(session) if session is not None else []
        meta = _read_json(station_dir(session, ids[-1]) / "meta.json") if ids else None
        meta = meta if isinstance(meta, dict) else {}
        reply_station, reply = last_reply(session, ids) if session is not None else (None, None)
        sp = pi_status()
        fresh = sp is not None and sp["age_s"] <= PI_SEEN_S

        def pick(key_pi: str, from_meta):
            if fresh and sp.get(key_pi) is not None:
                return sp[key_pi], "pi"
            return (from_meta, "station_meta") if from_meta is not None else (None, None)

        mode, mode_src = pick("mode", meta.get("mode"))
        battery, battery_src = pick("battery_v", _num(meta.get("battery_v")))
        sonar, sonar_src = pick("sonar_cm", _num(meta.get("ultrasonic_cm")))
        pose, pose_src = robot_pose(session, now) if session is not None else (None, "none")
        return {
            "now": round(now, 3),
            "pi_support": pi_support(),
            "mode": mode,
            "battery_v": battery,
            "sonar_cm": sonar,
            "clear_cm": (live or {}).get("clear_cm"),
            "clear_conf": (live or {}).get("clear_conf"),
            "pan_deg": _num(cmd.get("pan_deg")),
            "tilt_deg": _num(cmd.get("tilt_deg")),
            "steer_deg": _num(cmd.get("steer_deg")),
            "speed": _num(cmd.get("speed")),
            "moving": cmd.get("moving"),
            "snoo": best_snoo(cats),
            "cats": cats,
            "frame": None
            if live is None
            else {
                "sender": live.get("sender"),
                "seq": live.get("seq"),
                "age_s": live.get("age_s"),
                "size": live.get("size"),
                "infer_ms": live.get("infer_ms"),
                "url": f"{PREFIX}/frame.jpg?sender={live.get('sender')}",
            },
            "frame_age_s": None if live is None else live.get("age_s"),
            "sender": None if live is None else live.get("sender"),
            "session": None if session is None else session.name,
            "station": ids[-1] if ids else None,
            "station_age_s": None
            if not ids
            else round(now - (_mtime(station_dir(session, ids[-1]) / "meta.json") or now), 1),
            "last_reply": reply,
            "last_reply_station": reply_station,
            "slam": slam_pose(session, now) if session is not None else None,
            "pose": None if pose is None else {**pose_dict(pose), "source": pose_src},
            "pi_status": sp,
            "sources": {
                "mode": mode_src,
                "battery_v": battery_src,
                "sonar_cm": sonar_src,
                "frame": None if live is None else "live",
            },
        }

    @router.get("/state")
    def get_state(session: str | None = None):
        return JSONResponse(build_state(session), headers=NO_STORE)

    # --- frame ---

    def live_file(sender: str | None) -> tuple[dict, Path]:
        if live_dir is None:
            raise HTTPException(404, "今のフレームの置き場が無い（SNOO_MAP_LIVE_DIR）")
        if sender is not None and not _SENDER.fullmatch(sender):
            raise _bad("sender の形が違う")
        live = latest_live(live_dir, sender)
        if live is None:
            raise HTTPException(404, "今のフレームが無い")
        p = live_dir / f"{live['sender']}.jpg"
        if not p.is_file():
            raise HTTPException(404, "今のフレームの画像が無い")
        return live, p

    @router.get("/frame.jpg")
    def get_frame(sender: str | None = None):
        live, p = live_file(sender)
        headers = {
            **NO_STORE,
            "X-Frame-Seq": str(live.get("seq")),
            "X-Frame-Age-S": str(live.get("age_s")),
        }
        return Response(p.read_bytes(), media_type="image/jpeg", headers=headers)

    @router.get("/frame.json")
    def get_frame_json(sender: str | None = None):
        live, _ = live_file(sender)
        out = {**live, "cats": cats_in(live)}
        out["snoo"] = best_snoo(out["cats"])
        out["url"] = f"{PREFIX}/frame.jpg?sender={live['sender']}"
        return JSONResponse(out, headers=NO_STORE)

    # --- map ---

    def map_parts(session_id: str | None):
        session = current_session(root, session_id)
        if session is None:
            raise HTTPException(404, "セッションが無い")
        try:
            grid = SessionGrid(session)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
            raise HTTPException(404, f"間取りが無い: {e}") from e
        now = clock()
        pose, src = robot_pose(session, now)
        sl = slam_state(session) or {}
        plan = _read_json(session / "plan.json") or {}
        target = None
        p = plan.get("plan") if isinstance(plan.get("plan"), dict) else None
        window = plan.get("window") or {}
        if p and isinstance(p.get("target"), dict) and window.get("map") == "session":
            t = p["target"]
            target = Pose(t["x"], t["y"], math.radians(t["yaw_deg"]))
        marks = [a for a in notes.read() if a.get("session") == session.name]
        img, info = render_map(
            grid, read_poses(session), pose, src, sl.get("trail") or [], target, marks
        )
        info.update(
            {
                "session": session.name,
                "frame": "session",
                "robot": None if pose is None else {**pose_dict(pose), "source": src},
                "target": None if target is None else pose_dict(target),
                "annotations": [{"mark": f"A{k}", **a} for k, a in enumerate(marks, 1)],
                "grid_version": grid.version,
                "legend": {
                    "white": "床",
                    "black": "障害物",
                    "gray": "見ていない",
                    "light_blue": "車より高い物の下（上の層）",
                    "orange": "境目（frontier）",
                    "blue_circle": "ステーション（番号）",
                    "red": "車の今の位置",
                    "green_circle": "地図のサーバーが次に撮る位置",
                    "magenta": "印（A 番号）",
                    "purple_line": "SLAM の道筋",
                },
            }
        )
        return img, info

    @router.get("/map.png")
    def get_map(session: str | None = None):
        img, _ = map_parts(session)
        ok, png = cv2.imencode(".png", img)
        return Response(png.tobytes(), media_type="image/png", headers=NO_STORE)

    @router.get("/map.json")
    def get_map_json(session: str | None = None):
        _, info = map_parts(session)
        return JSONResponse(info, headers=NO_STORE)

    # --- 印 ---

    def snapshot_live(name: str) -> str:
        """今のフレームを snapshots/<name>.jpg に写す（live の画像は 1 秒ごとに上書きされる）。"""
        _, p = live_file(None)
        snapshots.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, snapshots / f"{name}.jpg")
        return f"snapshot:{name}.jpg"

    @router.post("/annotate")
    def annotate(a: Annotation):
        if not (_finite(a.x) and _finite(a.y)) or abs(a.x) > 50 or abs(a.y) > 50:
            raise _bad("x・y はセッションの座標（m）の 50 m までの数にする")
        session = current_session(root, a.session)
        if session is None:
            raise HTTPException(404, "セッションが無い（session を決める）")
        now = clock()
        with notes.lock:
            aid = notes.next_id()
            ref = a.image_ref
            if ref == "live":
                ref = snapshot_live(f"annotation_{aid}")
            pose, src = robot_pose(session, now)
            row = {
                "id": aid,
                "t": round(now, 3),
                "session": session.name,
                # セッションの座標：原点はそのセッションの最初のステーションの床、x はそのときの
                # 車の向き、y は左（m）。セッションごとに違うので、部屋の座標に直すときは
                # セッションどうしの位置合わせが要る
                "frame": "session",
                "x": round(a.x, 3),
                "y": round(a.y, 3),
                "label": a.label,
                "note": a.note,
                "image_ref": ref,
                "robot_pose": None if pose is None else {**pose_dict(pose), "source": src},
                "source": "agent",
            }
            notes.append(row)
        return row

    @router.get("/annotations")
    def get_annotations(session: str | None = None, label: str | None = None):
        rows = notes.read()
        if session is not None:
            rows = [r for r in rows if r.get("session") == session]
        if label is not None:
            rows = [r for r in rows if r.get("label") == label]
        return JSONResponse({"annotations": rows, "frame_note": FRAME_NOTE}, headers=NO_STORE)

    @router.get("/snapshots/{name}")
    def get_snapshot(name: str):
        if not _SNAPSHOT.fullmatch(name) or not (snapshots / name).is_file():
            raise HTTPException(404, "無い")
        return FileResponse(snapshots / name, media_type="image/jpeg")

    # --- 困りごとの下書き ---

    def ref_image(ref: str, session_id: str | None) -> tuple[str, bytes]:
        """参照の画像の (ファイル名, 中身)。"""
        if ref == "live":
            live, p = live_file(None)
            return f"live_{live['sender']}_{live.get('seq')}.jpg", p.read_bytes()
        if ref == "map":
            img, _ = map_parts(session_id)
            return "map.png", cv2.imencode(".png", img)[1].tobytes()
        if ref.startswith("snapshot:"):
            name = ref.split(":", 1)[1]
            if not _SNAPSHOT.fullmatch(name) or not (snapshots / name).is_file():
                raise _bad(f"{ref} は無い")
            return name, (snapshots / name).read_bytes()
        if ref.startswith("station:"):
            parts = ref.split(":", 1)[1].split("/")
            if len(parts) != 3 or not parts[1].isdigit():
                raise _bad("station: は station:<session>/<station_id>/<file> にする")
            s = current_session(root, parts[0])
            p = (station_dir(s, int(parts[1])) / parts[2]).resolve()
            if not p.is_relative_to(s.resolve()) or not p.is_file():
                raise _bad(f"{ref} は無い")
            return f"station_{parts[1]}_{p.name}", p.read_bytes()
        raise _bad(f"画像の参照 {ref!r} が分からない（live・map・snapshot:・station:）")

    @router.post("/issue")
    def issue(it: Issue):
        now = clock()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        slug = re.sub(r"[^A-Za-z0-9]+", "-", it.title).strip("-").lower()[:32]
        base = agent / "issues"
        d = base / (f"{stamp}_{slug}" if slug else stamp)
        k = 1
        while d.exists():
            k += 1
            d = base / f"{stamp}_{slug}_{k}"
        files: list[tuple[str, bytes]] = []
        for im in it.images:
            if isinstance(im, str):
                files.append(ref_image(im, it.session))
            else:
                if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", im.name) or im.name.startswith("."):
                    raise _bad(f"画像の名前 {im.name!r} は英数字・_・-・. にする")
                try:
                    data = base64.b64decode(im.data_b64, validate=True)
                except ValueError as e:
                    raise _bad(f"{im.name} の data_b64 が読めない") from e
                files.append((im.name, data))
        names = []
        for name, data in files:
            if len(data) > MAX_IMAGE_BYTES:
                raise HTTPException(413, f"{name} が大きすぎる")
            if name in names:
                stem, dot, ext = name.rpartition(".")
                name = f"{stem}_{len(names)}.{ext}" if dot else f"{name}_{len(names)}"
            names.append(name)
        state = build_state(it.session)
        d.mkdir(parents=True)
        for name, (_, data) in zip(names, files, strict=True):
            (d / name).write_bytes(data)
        lines = [
            f"# {it.title}",
            "",
            it.body,
            "",
            "## 画像",
            "",
            *([f"![{n}]({n})" for n in names] or ["（なし）"]),
            "",
            "## そのときの様子",
            "",
            f"- 時刻: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
            f"- セッション: {state['session']}・ステーション: {state['station']}",
            f"- 位置: {state['pose']}",
            "",
            "```json",
            json.dumps(
                {
                    k: state[k]
                    for k in (
                        "mode",
                        "battery_v",
                        "sonar_cm",
                        "snoo",
                        "frame",
                        "last_reply",
                        "slam",
                    )
                },
                ensure_ascii=False,
                indent=1,
            ),
            "```",
            "",
            "（エージェントが書いた下書き。人が読んでから GitHub に出す）",
            "",
        ]
        (d / "issue.md").write_text("\n".join(lines))
        return {"path": str(d), "files": ["issue.md", *names], "posted": False}

    # --- 声 ---

    @router.post("/say")
    def say(s: Say):
        text = s.text.strip()
        if not text:
            raise _bad("text が空")
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        path = tts_dir / f"{key}.wav"
        if not path.is_file():
            try:
                with tts_lock:
                    data = tts(text)
            except ImportError as e:
                raise HTTPException(
                    503, "pyopenjtalk が無い（uv sync --group agent か uv pip install pyopenjtalk）"
                ) from e
            tts_dir.mkdir(parents=True, exist_ok=True)
            atomic.write_bytes(path, data)
        sid = None
        if s.queue:
            with says.lock:
                sid = says.next_id()
                now = clock()
                says.append(
                    {
                        "id": sid,
                        "t": round(now, 3),
                        "text": text,
                        "wav": f"{key}.wav",
                        "url": f"{PREFIX}/say/{key}.wav",
                        "expires_at": round(now + SAY_TTL_S, 3),
                    }
                )
        headers = {"X-Say-Id": "" if sid is None else str(sid), "X-Say-Wav": f"{key}.wav"}
        return Response(path.read_bytes(), media_type="audio/wav", headers=headers)

    @router.get("/say/next")
    def say_next(since: int = Query(0, ge=0)):
        now = clock()
        for row in says.read():
            if int(row.get("id", 0)) > since and row.get("expires_at", 0) >= now:
                return JSONResponse({"say": row, "now": round(now, 3)}, headers=NO_STORE)
        return JSONResponse({"say": None, "now": round(now, 3)}, headers=NO_STORE)

    @router.get("/say/{name}")
    def say_wav(name: str):
        if not _WAV.fullmatch(name) or not (tts_dir / name).is_file():
            raise HTTPException(404, "無い")
        return FileResponse(tts_dir / name, media_type="audio/wav")

    # --- 指令 ---

    def status_of(c: dict, res: dict[int, dict], now: float) -> str:
        r = res.get(int(c["id"]))
        if r is not None:
            return r["status"]
        return "expired" if c.get("expires_at", 0) < now else "queued"

    def latest_results() -> dict[int, dict]:
        out: dict[int, dict] = {}
        for r in results.read():
            out[int(r.get("id", 0))] = r
        return out

    def validate(c: Command, now: float) -> tuple[dict, list[dict]]:
        """(Pi に渡す欄, 確かめた結果)。形が違えば 400、決まりに反すれば ok が False の行。"""
        if c.kind not in KINDS:
            raise _bad(f"kind は {', '.join(KINDS)} のどれか")
        checks: list[dict] = []
        if c.kind == "stop":
            return {}, [_check("stop", True, "止めるのはいつでも受ける")]
        if c.kind == "scan":
            return {}, [_check("scan", True, "その場で首を振って撮る（走らない）")]
        if c.kind == "look":
            if c.pan_deg is None and c.tilt_deg is None:
                raise _bad("look には pan_deg か tilt_deg が要る")
            params = {}
            for key, v, lim in (
                ("pan_deg", c.pan_deg, PAN_LIMIT_DEG),
                ("tilt_deg", c.tilt_deg, TILT_LIMIT_DEG),
            ):
                if v is None:
                    continue
                ok = _finite(v) and abs(v) <= lim
                checks.append(
                    _check(
                        key,
                        ok,
                        f"{v} は ±{lim:.0f}° の内" if ok else f"{v} は ±{lim:.0f}° を超える",
                    )
                )
                params[key] = round(float(v), 1) if _finite(v) else v
            return params, checks

        # move
        d, steer, speed = c.distance_cm, c.steer_deg or 0.0, c.speed
        if d is None or speed is None:
            raise _bad("move には distance_cm と speed が要る（steer_deg は省くと 0）")
        params = {
            "steer_deg": round(float(steer), 1),
            "distance_cm": round(float(d), 1),
            "speed": round(float(speed), 1),
            "reverse": bool(c.reverse),
        }
        ok = _finite(d) and 0 < d <= MAX_MOVE_CM
        checks.append(_check("distance", ok, f"{d} cm（0 より長く {MAX_MOVE_CM:.0f} cm まで）"))
        ok = _finite(speed) and 0 < speed <= MAX_MOVE_SPEED
        checks.append(_check("speed", ok, f"{speed}（0 より大きく {MAX_MOVE_SPEED:.0f} まで）"))
        ok = _finite(steer) and abs(steer) <= MAX_STEER_DEG
        checks.append(_check("steer", ok, f"{steer}°（±{MAX_STEER_DEG:.0f}° まで）"))
        if not all(x["ok"] for x in checks):
            return params, checks

        res = latest_results()
        pending = [
            x
            for x in commands.read()
            if x.get("kind") == "move"
            and status_of(x, res, now) in ("queued", "accepted", "running")
        ]
        checks.append(
            _check(
                "one_move_at_a_time",
                not pending,
                "前の move が終わっていない（結果を待つか stop する）"
                if pending
                else "前の move は終わっている",
            )
        )
        sp = pi_status()
        if (
            sp is not None
            and sp["age_s"] <= PI_SEEN_S
            and sp.get("mode") not in (None, "idle", "agent")
        ):
            checks.append(
                _check(
                    "pi_mode",
                    False,
                    f"Pi が {sp.get('mode')} で動いている（idle か agent のときだけ）",
                )
            )

        live = latest_live(live_dir)
        age = None if live is None else live.get("age_s")
        fresh = age is not None and age <= FRESH_FRAME_S
        checks.append(
            _check(
                "fresh_frame",
                fresh,
                f"今のフレームは {age} 秒前"
                if live
                else "今のフレームが無い（猫がいないと言えない）",
            )
        )
        if live is not None and (live.get("cmd") or {}).get("moving"):
            checks.append(_check("robot_still", False, "車が走っている"))
        if fresh:
            near = [
                x
                for x in cats_in(live)
                if x["distance_cm"] is None or x["distance_cm"] <= CAT_NEAR_CM
            ]
            if near:
                desc = ", ".join(
                    f"{'?' if x['distance_cm'] is None else round(x['distance_cm'])} cm"
                    for x in near
                )
                checks.append(
                    _check(
                        "cat_far",
                        False,
                        f"猫が {CAT_NEAR_CM:.0f} cm より近いか距離が分からない（{desc}）",
                    )
                )
            else:
                checks.append(_check("cat_far", True, f"{CAT_NEAR_CM:.0f} cm の内に猫は見えない"))

        session = current_session(root, c.session)
        if session is None:
            checks.append(_check("grid", False, "セッションが無い"))
            return params, checks
        try:
            grid = SessionGrid(session)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
            checks.append(_check("grid", False, f"間取りが無いので確かめられない: {e}"))
            return params, checks
        checks.append(_check("grid", True, f"{session.name} の間取り"))
        pose, src = robot_pose(session, now)
        if pose is None:
            checks.append(_check("pose", False, "今の位置が分からない"))
            return params, checks
        checks.append(_check("pose", True, f"{src} の位置 {pose_dict(pose)}"))
        params["session"] = session.name
        params["from_pose"] = {**pose_dict(pose), "source": src}
        checks += check_move(grid, pose, float(steer), float(d), bool(c.reverse))
        return params, checks

    @router.post("/command")
    def command(c: Command):
        now = clock()
        with commands.lock:
            params, checks = validate(c, now)
            ok = all(x["ok"] for x in checks)
            cid = None
            if ok:
                cid = commands.next_id()
                ttl = STOP_TTL_S if c.kind == "stop" else COMMAND_TTL_S
                commands.append(
                    {
                        "id": cid,
                        "t": round(now, 3),
                        "kind": c.kind,
                        **params,
                        "expires_at": round(now + ttl, 3),
                        "note": c.note,
                        "checks": checks,
                    }
                )
        support = pi_support()
        return {
            "queued": ok,
            "id": cid,
            "kind": c.kind,
            "params": params,
            "checks": checks,
            "pi_support": support,
            "note": "並べた。Pi が受けるまで動かない"
            if ok and support
            else "並べたが、Pi はまだ指令を受けない（pi_support: false）。何も動かない"
            if ok
            else "決まりに反するので並べなかった（checks を見る）",
        }

    @router.get("/commands/next")
    def commands_next(
        after: int = Query(0, ge=0),
        mode: str | None = Query(None, max_length=32),
        battery_v: float | None = None,
        sonar_cm: float | None = None,
        running: int | None = None,
        supports: str | None = Query(None, max_length=200),
    ):
        """Pi が 0.5 秒ごとに問い合わせる。ここに来ること自体が heartbeat。"""
        now = clock()
        pi.clear()
        pi.update(
            {
                "t": now,
                "mode": mode,
                "battery_v": _num(battery_v),
                "sonar_cm": _num(sonar_cm),
                "running": running,
                "supports": [s for s in (supports or "").split(",") if s],
            }
        )
        res = latest_results()
        pending = [
            x
            for x in commands.read()
            if int(x["id"]) > after and status_of(x, res, now) == "queued"
        ]
        stops = [x for x in pending if x.get("kind") == "stop"]
        # stop があれば、それより前の指令は飛ばして stop を先に渡す
        nxt = stops[-1] if stops else (pending[0] if pending else None)
        if nxt is not None:
            nxt = {k: v for k, v in nxt.items() if k != "checks"}
        return JSONResponse(
            {"now": round(now, 3), "heartbeat_s": HEARTBEAT_S, "command": nxt},
            headers=NO_STORE,
        )

    @router.post("/command/{cid}/result")
    def command_result(cid: int, r: Result):
        if r.status not in RESULT_STATES:
            raise _bad(f"status は {', '.join(sorted(RESULT_STATES))} のどれか")
        if not any(int(x["id"]) == cid for x in commands.read()):
            raise HTTPException(404, "その指令は無い")
        row = {
            "id": cid,
            "t": round(clock(), 3),
            "status": r.status,
            "reason": r.reason,
            "detail": r.detail,
        }
        with results.lock:
            results.append(row)
        return row

    @router.get("/command/{cid}")
    def command_get(cid: int):
        for x in commands.read():
            if int(x["id"]) == cid:
                res = latest_results()
                return {
                    **x,
                    "status": status_of(x, res, clock()),
                    "result": res.get(cid),
                    "pi_support": pi_support(),
                }
        raise HTTPException(404, "その指令は無い")

    @router.get("/commands")
    def commands_list(limit: int = Query(20, ge=1, le=200)):
        now = clock()
        res = latest_results()
        rows = commands.read()[-limit:]
        return {
            "commands": [
                {**x, "status": status_of(x, res, now), "result": res.get(int(x["id"]))}
                for x in rows
            ],
            "pi_support": pi_support(),
        }

    return router


FRAME_NOTE = (
    "x・y はセッションの座標（m）。原点はそのセッションの最初のステーションの床、x はそのときの"
    "車の向き、y は左、yaw は左回りが +。セッションごとに原点が違うので、session と一緒に読む"
)
