"""PiCar-X と部屋の「デジタルツイン」をブラウザで見る画面（地図のサーバーの /twin）。

地図のサーバー（app/snoo_map_server.py）に足す。画面は app/static/twin/ の素の JS と Canvas で、
組み立て（ビルド）も外の CDN も要らない（家の外へ出られなくても見られるように）。

- GET /twin：画面
- GET /twin/static/<名前>：画面の JS・CSS（app/static/twin/ にあるものだけ）
- GET /twin/api/sessions：セッションの一覧（新しい順）
- GET /twin/api/<session>/state：画面に要るものをまとめた JSON。ブラウザは 1.5 秒ごとに読む。
  地図（grid.png）や次の目標・Dubins の道は窓の座標（窓のいちばん古いステーションが原点）で
  書かれているので、ここでセッションの座標（最初のステーションが原点）に直して返す。
  次の 1 歩の区間は、Pi と同じ自転車モデル（app/roommap.py）で走らせた道にして返す
- GET /twin/api/<session>/map/<grid.png・high.png・topdown.png>
- GET /twin/api/<session>/station/<station_id>/<画像>：止まって撮った画像
- GET /twin/api/live：検出サーバー（Docker の snoo）が書いた、送り元ごとの最新の返事
  （app/snoo_live.py。SNOO_LIVE_DIR に書いたもの）
- GET /twin/api/live/<送り元>.jpg：そのフレーム

ファイルを返すのは、名前を決まった形に限ったうえで、置き場の中にあるものだけにする
（../ などで外を読ませない）。送り元の制限（SNOO_MAP_ALLOW）は地図のサーバーと同じものがかかる。
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.roommap import (
    CAM_X_M,
    CAR_FRONT_M,
    CAR_REAR_M,
    SONAR_HALF_DEG,
    SONAR_RANGE_M,
    WHEELBASE_M,
    Pose,
    camera_of,
    drive_poses,
)
from app.snoo_map_server import (
    _FILE,
    BadRequest,
    check_id,
    read_poses,
    read_reverses,
    station_dir,
    station_ids,
    to_session,
    ultrasonic_cm,
)
from app.snoo_mapper import pose_dict

STATIC = Path(__file__).parent / "static" / "twin"
MEDIA = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
MAPS = {"grid.png", "high.png", "topdown.png"}
# 送り元の名前（app/snoo_live.py の sender_name。IP の : は - にしてある）
_SENDER = re.compile(r"[0-9A-Za-z_-][0-9A-Za-z._-]{0,63}")
# 車の形（画面に描く）。幅は写真から見積もった（約 19 cm。測っていない）。カメラの画角は
# app/clearance.py と同じ 54°
CAR = {
    "wheelbase_m": WHEELBASE_M,
    "cam_x_m": CAM_X_M,
    "front_m": CAR_FRONT_M,
    "rear_m": CAR_REAR_M,
    "width_m": 0.19,
    "height_m": 0.12,
    "hfov_deg": 54.0,
    "sonar_half_deg": SONAR_HALF_DEG,
    "sonar_range_m": SONAR_RANGE_M,
}
NO_STORE = {"Cache-Control": "no-store"}


def _read_json(path: Path):
    """読めなければ None（書いている途中・まだ無い）。"""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _pose(d) -> Pose | None:
    if not isinstance(d, dict):
        return None
    try:
        return Pose(float(d["x"]), float(d["y"]), math.radians(float(d["yaw_deg"])))
    except (KeyError, TypeError, ValueError):
        return None


def list_sessions(root: Path) -> list[dict]:
    """セッションの一覧。いちばん新しく書いたものが先。"""
    out = []
    if not root.is_dir():
        return out
    for d in root.iterdir():
        if not d.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", d.name):
            continue
        ids = station_ids(d)
        times = [_mtime(d), _mtime(d / "plan.json")]
        if ids:
            times.append(_mtime(station_dir(d, ids[-1])))
        last = max(times)
        out.append(
            {"id": d.name, "stations": len(ids), "latest": ids[-1] if ids else None, "mtime": last}
        )
    return sorted(out, key=lambda s: (-s["mtime"], s["id"]))


def station_info(session: Path, i: int, poses: dict[int, Pose]) -> dict:
    d = station_dir(session, i)
    meta = _read_json(d / "meta.json") or {}
    frames = meta.get("frames") if isinstance(meta.get("frames"), list) else []
    try:
        sonar = ultrasonic_cm(d)
    except (OSError, ValueError, KeyError, TypeError):
        sonar = None
    return {
        "id": i,
        "pose": pose_dict(poses[i]) if i in poses else None,
        "battery_v": meta.get("battery_v"),
        "ultrasonic_cm": None if sonar is None else round(sonar, 1),
        "prev_step": meta.get("prev_step"),
        "supports": meta.get("supports"),
        "mtime": _mtime(d / "meta.json"),
        "frames": [
            {"file": f.get("file"), "pan_deg": f.get("pan_deg"), "tilt_deg": f.get("tilt_deg")}
            for f in frames
            if isinstance(f, dict) and not f.get("moving") and isinstance(f.get("file"), str)
        ],
    }


def step_path(reply: dict) -> dict | None:
    """Pi に返した 1 歩を、区間ごとにカメラの真下が通る道（セッションの座標）にする。"""
    if not isinstance(reply, dict) or reply.get("action") != "move":
        return None
    start = _pose(reply.get("pose"))
    if start is None:
        return None
    segs = reply.get("segments")
    if not isinstance(segs, list) or not segs:
        segs = [{"steer_deg": reply.get("steer_deg"), "distance_cm": reply.get("distance_cm")}]
    reverse = bool(reply.get("reverse"))
    out = []
    here = start
    for s in segs:
        try:
            steer, m = float(s["steer_deg"]), float(s["distance_cm"]) / 100
        except (KeyError, TypeError, ValueError):
            return None
        poses = [camera_of(p) for _, p in drive_poses(here, [(steer, m)], reverse=reverse)]
        out.append(
            {
                "steer_deg": steer,
                "distance_cm": round(m * 100, 1),
                "path": [[round(p.x, 3), round(p.y, 3)] for p in poses],
            }
        )
        here = poses[-1]
    return {"start": pose_dict(start), "end": pose_dict(here), "reverse": reverse, "segments": out}


def session_state(session: Path) -> dict:
    ids = station_ids(session)
    poses = read_poses(session)
    room = _read_json(session / "room.json")
    plan = _read_json(session / "plan.json") or {}
    window = plan.get("window") or (room or {}).get("window") or {}
    anchor = _pose(window.get("origin_pose")) or Pose(0.0, 0.0, 0.0)

    grid = None
    if isinstance(room, dict) and (session / "grid.png").is_file():
        high = (room.get("layers") or {}).get("high") or {}
        grid = {
            "resolution_m": room["resolution_cm"] / 100,
            "origin": room["origin"],
            "size": room["size"],
            "origin_pose": pose_dict(anchor),
            "high": (session / "high.png").is_file(),
            "high_min_m": (high.get("legend") or {}).get("min_height_m"),
            "version": max(_mtime(session / "grid.png"), _mtime(session / "high.png")),
        }

    # 古い plan.json には reply が無い（そのときは返事と next が同じだった）
    reply = plan.get("reply") or plan.get("next")
    if reply is None and "action" in plan:
        reply = {k: plan[k] for k in ("action", "reason") if k in plan}
    nbv = None
    if isinstance(plan.get("plan"), dict):
        p = plan["plan"]
        target = _pose(p.get("target"))
        nbv = {
            "target": pose_dict(to_session(anchor, target)) if target else None,
            "gain_cells": p.get("gain_cells"),
            "length_m": p.get("length_m"),
            "path": [
                [round(q.x, 3), round(q.y, 3)]
                for q in (to_session(anchor, Pose(x, y, 0.0)) for x, y in p.get("path") or [])
            ],
        }
    stations = [station_info(session, i, poses) for i in ids]
    updated = max(
        [_mtime(session / "plan.json"), _mtime(session / "poses.json")]
        + [s["mtime"] for s in stations[-1:]]
    )
    return {
        "session": session.name,
        "updated": updated,
        "now": time.time(),
        "stations": stations,
        "grid": grid,
        "reply": reply,
        "next": plan.get("next"),
        "plan": nbv,
        "step": step_path(reply) if reply else None,
        "window": window or None,
        # 返事を作ったステーション（窓の最後）。電池が少ない・上限のときは地図を作らずに返すので、
        # plan.json は前のステーションのままになる
        "reply_station": (window.get("stations") or [None])[-1],
        "timing_s": plan.get("timing_s"),
        "reverses": {str(k): v for k, v in read_reverses(session).items()},
        "car": CAR,
    }


def read_live(live_dir: Path | None) -> list[dict]:
    """検出サーバーが書いた送り元ごとの最新の返事。新しい順。"""
    if live_dir is None or not live_dir.is_dir():
        return []
    now = time.time()
    out = []
    for p in live_dir.glob("*.json"):
        if not _SENDER.fullmatch(p.stem):
            continue
        info = _read_json(p)
        if not isinstance(info, dict):
            continue
        info["sender"] = p.stem
        t = info.get("t_recv")
        # 同じマシンの時計なので負にはならないはずだが、なっても 0 にする
        info["age_s"] = round(max(0.0, now - t), 2) if isinstance(t, int | float) else None
        info["has_image"] = (live_dir / f"{p.stem}.jpg").is_file()
        out.append(info)
    return sorted(out, key=lambda i: -(i.get("t_recv") or 0))


def _inside(path: Path, base: Path) -> Path:
    """path が base の中にあれば、その本当の場所。外（リンクで外を指すものも）なら 404。"""
    real = path.resolve()
    if not real.is_relative_to(base.resolve()) or not real.is_file():
        raise HTTPException(404, "無い")
    return real


def add_twin(app: FastAPI, root: Path, live_dir: Path | None) -> None:
    static = {p.name: p for p in STATIC.iterdir() if p.is_file() and p.suffix in MEDIA}

    def _session(session_id: str) -> Path:
        try:
            check_id(session_id)
        except BadRequest as e:
            raise HTTPException(400, str(e)) from e
        d = root / session_id
        if not d.is_dir():
            raise HTTPException(404, "そのセッションは無い")
        return d

    @app.get("/twin", include_in_schema=False)
    def twin_page():
        return FileResponse(static["index.html"], media_type=MEDIA[".html"], headers=NO_STORE)

    @app.get("/twin/static/{name}", include_in_schema=False)
    def twin_static(name: str):
        if name not in static:  # 決まったファイルだけ（名前で外を指させない）
            raise HTTPException(404, "無い")
        p = static[name]
        return FileResponse(p, media_type=MEDIA[p.suffix], headers={"Cache-Control": "no-cache"})

    @app.get("/twin/api/sessions")
    def twin_sessions():
        return JSONResponse({"sessions": list_sessions(root)}, headers=NO_STORE)

    @app.get("/twin/api/live")
    def twin_live():
        return JSONResponse(
            {"enabled": live_dir is not None, "senders": read_live(live_dir)}, headers=NO_STORE
        )

    @app.get("/twin/api/live/{name}.jpg")
    def twin_live_image(name: str):
        if live_dir is None or not _SENDER.fullmatch(name):
            raise HTTPException(404, "無い")
        p = _inside(live_dir / f"{name}.jpg", live_dir)
        return FileResponse(p, media_type="image/jpeg", headers=NO_STORE)

    @app.get("/twin/api/{session_id}/state")
    def twin_state(session_id: str):
        return JSONResponse(session_state(_session(session_id)), headers=NO_STORE)

    @app.get("/twin/api/{session_id}/map/{name}")
    def twin_map(session_id: str, name: str):
        if name not in MAPS:
            raise HTTPException(404, f"{name} は無い（{', '.join(sorted(MAPS))} のどれか）")
        d = _session(session_id)
        p = _inside(d / name, d)
        return FileResponse(p, media_type="image/png", headers=NO_STORE)

    @app.get("/twin/api/{session_id}/station/{station_id}/{name}")
    def twin_station_image(session_id: str, station_id: int, name: str):
        if not 0 <= station_id <= 999 or not _FILE.fullmatch(name):
            raise HTTPException(404, "無い")
        d = _session(session_id)
        p = _inside(station_dir(d, station_id) / name, d)
        media = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        # 同じ station_id が送り直されると置き換わるので、長くは覚えさせない
        return FileResponse(p, media_type=media, headers={"Cache-Control": "max-age=10"})
