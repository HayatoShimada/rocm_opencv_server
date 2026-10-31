"""PiCar-X（85pi4 の snoo_trucar）が止まって首を振って撮った画像（1 ステーション）を HTTP で受け、
間取りを作り直して次の 1 歩を返す（間取りは app/snoo_mapper.py で作る）。

    cd ~/orca/rocm_opencv_server && .venv/bin/python -m app.snoo_map_server
VGGT は main の .venv にだけ入れてある（git から入れた。uv.lock にも Docker にも無い）ので、
コンテナではなくホストのプロセスとして動かす。

- POST /v1/snoo/map/station（multipart）：meta（scan.json に session_id・station_id などを足した
  JSON）と images（meta の frames[].file と同じ名前の画像）。返事は次の 1 歩（action=move）か、
  やめる理由（action=done）。Pi は 25 秒待って返事が無ければ送り直して待つ（自分では道を決めない）
  ので、1 ステーションを 20 秒以内に返す
  - meta の ultrasonic_cm（と止まって撮ったコマごとの読み）は、地図の弱い証拠としてだけ足す
    （測れた近い値の弧に「たぶん障害物」。止まる判断・ぶつかる判断には使わない）
  - 次の 1 歩はサーバーが道をすべて決める（app/snoo_maneuver.py。Pi はそのとおり走るだけ）。
    形は meta の supports（Pi ができること）で決める。詳しくは docs/snoo_explore_protocol.md
    - 無し：1 本の弧（前へだけ）{"action": "move", steer_deg, speed, distance_cm}
    - "segments"：segments: [{steer_deg, distance_cm}, ...]（3 つまで、1 歩の中は同じ向き）
    - "reverse"：後ろへの 1 歩も返す（"reverse": true。後ろへは合わせて 30 cm まで）
    - "gear_segments"（と "reverse"・"segments"）：区切りごとに前・後ろを変えられる（5 つまで。
      切り返し・3 点ターン・隙間へ寄せる）。区切りに reverse・end_pose（着く姿勢）・
      min_clear_cm（前への区切りの clear_cm の見込みの下限）、返事に reverse_speed を付ける
    - "long_step"（と "segments"）：前への合計を 60 cm まで
  - 調べ終えた（境目が無く、確信度の低い縁も無い）なら done no_frontier、行ける所が無ければ
    done unreachable、直近のステーションが同じ所に固まったまま行く道が無ければ done stuck
- GET /v1/snoo/map/<session_id>/topdown.png・room.json・plan.json：いちばん新しい結果（人が見る用）
- SNOO_HOME_MAP=1 なら、セッションを家の地図（app/snoo_home.py、SNOO_HOME_DIR）に合わせて足し、
  合えば次の 1 歩を家の地図で埋めた格子で決める。GET /v1/snoo/map/home.png・home.json で
  家の地図とセッションごとの変換を返す。既定は使わない（今までと同じに動く）
- /v1/snoo/agent/...：AI のエージェントがロボットを見て・印を付けて・動かすための API
  （app/snoo_agent_api.py・docs/agent_robot.md）。読み込めなくても地図のサーバーは動く
- GET /twin：車と部屋をブラウザで見る画面（app/snoo_twin.py）。地図・ステーション・次の 1 歩・
  撮った画像と、検出サーバーの今のフレーム（SNOO_MAP_LIVE_DIR）を 1〜2 秒ごとに読み直して出す。
  このマシンのブラウザなら http://127.0.0.1:8720/twin で開ける。tailnet や LAN の別の機械から
  見るときは、その機械の IP を SNOO_MAP_ALLOW に足して起動し直す（既定には入れていない。
  認証が無く、撮った部屋の画像が見えるため）。例：
      SNOO_MAP_ALLOW=192.168.0.188,100.73.14.8,127.0.0.1,100.x.y.z \\
          .venv/bin/python -m app.snoo_map_server
  入っていない機械のブラウザには、足し方を書いた 403 の画面を返す
- GET /drive・/v1/snoo/teleop/...：人がゲームパッド・キーボード・スマホで車を動かす画面と API
  （app/snoo_teleop.py。Pi 側の決まりは docs/teleop.md）

受けたステーションは <SNOO_MAP_DATA>/<session_id>/<station_id:03d>/ に画像と meta.json で残し、
結果（vggt.npz・room.json・topdown.png・plan.json など）はセッションの置き場に書く。

VGGT は 48 コマで 9 秒ほどかかる（RX 7900 XTX）。実データ（23 枚のステーション）で、受けてから
返すまで 1 か所目 5.7 秒、窓が 48 コマのとき 10〜12 秒（VGGT 7.5〜9.5 秒・間取り 2 秒）だった。
ステーションが増えても時間と GPU のメモリが増えないよう、VGGT に入れるのは最後のステーションと、
その近くのステーション（合わせて WINDOW 個）だけにする。そのときの間取りの原点は窓のいちばん
古いステーションになるので、前に求めたそのステーションの位置を足して、返す位置をセッションの
座標（最初のステーションが原点）に直す。
最後のステーションで見たもの（床・障害物・上の層のセルと超音波）は、セッションの座標にして
<station>/obs.npz に 1 回だけ残し、次の 1 歩は全部のステーションの分を足した間取りで決める
（窓の外で見た所も忘れない）。room.json・topdown.png もセッションの座標で書く
（window.origin_pose は (0, 0, 0)、VGGT の窓の原点は window.vggt_origin_pose）。
SNOO_POSE_GRAPH=1（既定）なら、ステーションの位置は位置のグラフ（app/snoo_posegraph.py）で
決める。ステーションごとに、窓の中の全部の組の辺と前のステーションから走った辺を
graph_edges.jsonl に足し、全部を解き直して graph_poses.json と poses.json に書く。前に残した
obs.npz は、読むときにグラフの位置へ置き直す。窓で位置合わせが崩れても、グラフで位置が
決まれば戻らずにそのまま探る（見たものは足さない）。グラフで落ちたら今までの決め方で続ける。
SNOO_ROMA_EDGES=1（既定は 0）なら、前のステーションと近くに戻ってきた組の候補 1 つを RoMa v2 に
通した辺（app/snoo_roma.py、kind "roma"）もグラフに足す。RoMa v2 は使うときに読む。

認証が無いので、SNOO_MAP_ALLOW に書いた送り元（Pi の LAN と tailscale の IP）以外には 403 を返す
（app/snoo.py の検出サーバーと同じやり方）。GPU は猫の検出（Docker の snoo）と分け合うので、
VGGT は 1 つずつ回す（同時に来たものは順に待たせる）。
"""

import dataclasses
import functools
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app import atomic
from app import snoo_posegraph as pg
from app import snoo_roma as sr
from app.roommap import (
    FIT_TOL_M,
    LONG_STEP_M,
    MAX_SEGMENTS,
    MAX_STEP_M,
    REVERSE_MAX_M,
    Pose,
    camera_of,
    drive_moves,
    fit_moves,
    mark_blocked_ahead,
    mark_driven_moves,
    take_moves,
)
from app.snoo import _allowed, parse_allow
from app.snoo_maneuver import GEAR_MAX_SEGMENTS, moves_ok
from app.snoo_maneuver import MAX_STEER_DEG as MANEUVER_STEER_DEG
from app.snoo_mapper import (
    CAT_MASK_MIN,
    MAX_FRAMES,
    RECENT_STATIONS,
    FrameRef,
    MapParams,
    Observation,
    Station,
    action_moves,
    build_map,
    choose_window,
    decide,
    frame_colors,
    limits,
    load_station,
    load_vggt_model,
    mark_slam,
    mask_cats,
    moves_action,
    pose_dict,
    registration_is_bad,
    run_vggt,
    save_vggt,
    select_frames,
    session_grid,
    to_frame,
    unproject,
    write_outputs,
    xy_to_frame,
)
from app.snoo_mapper import (
    WINDOW as MAPPER_WINDOW,
)

HOST = os.getenv("SNOO_MAP_HOST", "0.0.0.0")
PORT = int(os.getenv("SNOO_MAP_PORT", "8720"))
# 受ける送り元（IP か CIDR をカンマで区切る）。Pi の LAN・Pi の tailscale・このマシン
ALLOW = os.getenv("SNOO_MAP_ALLOW", "192.168.0.188,100.73.14.8,127.0.0.1")
DATA = os.getenv("SNOO_MAP_DATA", "data/snoo_map/sessions")
# 検出サーバー（Docker の snoo）が今のフレームを書く所（compose.yaml の SNOO_LIVE_DIR と同じ所）。
# 空なら /twin に今のフレームを出さない
LIVE_DIR = os.getenv("SNOO_MAP_LIVE_DIR", "/run/snoo/live")
# VGGT に使わせる GPU の CU（ROCm の HSA_CU_MASK の形「GPU 番号:CU の範囲」）。空なら全部使う。
# VGGT が GPU を全部使うと、同じ GPU の猫の検出（Docker の snoo）が 1 枚 4.4 ms → p50 37 ms・
# p95 185 ms（60 fps → 15 fps）まで遅れた。RX 7900 XTX（96 CU）で 72 CU（0-71）にすると、検出は
# p50 8.7 ms・p95 11.5 ms（55 fps）、VGGT は 40 コマで 6.5 → 7.4 秒（2026-10-29 に測った）。
# 48 CU（0-47）なら検出 p50 6.2 ms・p95 8.5 ms だが、VGGT が 9.9 秒になり、48 コマの窓では
# Pi の待ち（20 秒）に近づくので、余裕のある 0-71 にする。
# HSA_CU_MASK を直に付けて起動したときは、そちらを使う
CU_MASK = os.getenv("SNOO_MAP_CU_MASK", "0:0-71")
# 1 回の POST の上限。Pi の画像は 640x480 の JPEG で 1 枚 30 KB ほど、1 ステーション 24 枚ほど
MAX_REQUEST_BYTES = 40 * 1024 * 1024
# VGGT に入れるのは、最後のステーションの全部のコマ（10 コマ）と、近くの前の WINDOW - 1 か所の
# OLD_FRAMES コマずつ（パン -30・0・30 × チルト 2 段）。前のステーションは位置がもう決まって
# いて、見たものも obs.npz に残してあるので、最後のステーションを同じ座標に置く手がかりで足りる。
# 3 回目の実走の 15 か所を流し直して測った（scripts/snoo_map_replay.py、72 CU、2026-10-29）：
#   窓 6・等分（36 コマ）：VGGT 6.5 秒、合わせて 7.4 秒（いちばん遅いとき 9.8 秒）
#   窓 4・前は 6 コマ（28 コマ）：VGGT 4.5 秒、合わせて 5.3 秒（7.8 秒）。位置の差は
#     窓 6 と比べて中央値 0.5 cm・いちばん大きいとき 11 cm、向き 3° まで
#   窓 4・前は 2 コマ（16 コマ）：合わせて 3.1 秒だが、1 か所で位置が 24 cm・向きが 12° ずれ、
#     1 か所は位置合わせが崩れた。使わない
WINDOW = MAPPER_WINDOW
OLD_FRAMES: int | None = 6
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGES = 64
# これより電池の電圧が低ければ探るのをやめる（2 セルのリチウムイオンで、走ると電圧が落ちる）
LOW_BATTERY_V = 6.8
# 床からレンズの中心まで。2026-10-29 に SD カードの長い辺（32 mm）2.5 枚分と測った
DEFAULT_CAMERA_HEIGHT_CM = 8.0
# 超音波がこれより短く読んだら測れなかったとする（PiCar-X の超音波は 2 cm から）
MIN_ULTRASONIC_CM = 2.0
# Pi の speed 1 あたりの速さ（cm/s）。1 回目の実走の VGGT の位置から測った（speed 12 で約 20 cm/s）
PI_CM_PER_S_PER_SPEED = 1.7

# 置き場の名前になるので、/ や . を含めない（../ で外に書かせない）
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_FILE = re.compile(r"[A-Za-z0-9_+-][A-Za-z0-9_.+-]{0,127}\.(jpg|jpeg|png)", re.IGNORECASE)
_STATION_DIR = re.compile(r"\d{3}")
OUTPUTS = {
    "topdown.png": "image/png",
    "room.json": "application/json",
    "plan.json": "application/json",
}

log = logging.getLogger("snoo_map")


class BadRequest(ValueError):
    pass


# --- 受けたものを調べる ---


def _number(v, name: str, allow_none: bool = True) -> float | None:
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise BadRequest(f"{name} が数ではない")
    return float(v)


def check_id(session_id) -> str:
    if not isinstance(session_id, str) or not _ID.fullmatch(session_id):
        raise BadRequest("session_id は英数字・_・- の 64 文字までにする")
    return session_id


def parse_meta(text: str | None) -> dict:
    """meta（JSON の文字列）を読んで調べる。おかしければ BadRequest。"""
    if not text:
        raise BadRequest("meta が無い")
    try:
        meta = json.loads(text)
    except json.JSONDecodeError as e:
        raise BadRequest(f"meta が JSON として読めない: {e}") from e
    if not isinstance(meta, dict):
        raise BadRequest("meta が JSON のオブジェクトではない")
    check_id(meta.get("session_id"))
    sid = meta.get("station_id")
    if isinstance(sid, bool) or not isinstance(sid, int) or not 0 <= sid <= 999:
        raise BadRequest("station_id は 0〜999 の整数にする")
    _number(meta.get("battery_v"), "battery_v")
    _number(meta.get("tilt_offset_deg"), "tilt_offset_deg")
    _number(meta.get("ultrasonic_cm"), "ultrasonic_cm")
    h = _number(meta.get("camera_height_cm"), "camera_height_cm")
    if h is not None and not 0 < h <= 100:
        raise BadRequest("camera_height_cm は 0 より大きく 100 以下にする")
    if meta.get("prev_step") is not None and not isinstance(meta["prev_step"], dict):
        raise BadRequest("prev_step はオブジェクトか null にする")
    sup = meta.get("supports")
    if sup is not None and (
        not isinstance(sup, list) or len(sup) > 16 or not all(isinstance(x, str) for x in sup)
    ):
        raise BadRequest("supports は文字列のリストにする")
    frames = meta.get("frames")
    if not isinstance(frames, list) or not frames:
        raise BadRequest("meta の frames が無い")
    seen = set()
    for f in frames:
        if not isinstance(f, dict):
            raise BadRequest("frames の中身がオブジェクトではない")
        name = f.get("file")
        if not isinstance(name, str) or not _FILE.fullmatch(name):
            raise BadRequest(f"frames の file（{name!r}）は英数字の .jpg・.png の名前にする")
        if name in seen:
            raise BadRequest(f"frames に {name} が 2 つある")
        seen.add(name)
        _number(f.get("pan_deg"), f"{name} の pan_deg", allow_none=False)
        _number(f.get("tilt_deg"), f"{name} の tilt_deg", allow_none=False)
    if all(f.get("moving") for f in frames):
        raise BadRequest("止まって撮ったコマ（moving でないもの）が 1 つも無い")
    return meta


def read_images(images: list[UploadFile] | None, meta: dict) -> dict[str, bytes]:
    """画像を読んで、meta の frames と名前が合うか・読める画像かを調べる。"""
    if not images:
        raise BadRequest("images が無い")
    if len(images) > MAX_IMAGES:
        raise BadRequest(f"images は {MAX_IMAGES} 枚までにする")
    frames = {f["file"]: f for f in meta["frames"]}
    out: dict[str, bytes] = {}
    for up in images:
        name = up.filename or ""
        if name not in frames:
            raise BadRequest(f"画像 {name!r} が meta の frames に無い")
        if name in out:
            raise BadRequest(f"画像 {name} が 2 つある")
        data = up.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(
                413, f"画像 {name} が {MAX_IMAGE_BYTES // 1024 // 1024} MB を超える"
            )
        if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8) is None:
            raise BadRequest(f"画像 {name} が読めない")
        out[name] = data
    missing = [n for n, f in frames.items() if not f.get("moving") and n not in out]
    if missing:
        raise BadRequest(f"止まって撮ったコマの画像が無い: {', '.join(missing[:5])}")
    return out


# --- 置き場 ---


def station_dir(session: Path, station_id: int) -> Path:
    return session / f"{station_id:03d}"


def save_station(session: Path, station_id: int, meta: dict, images: dict[str, bytes]) -> Path:
    """画像と meta.json を書く。同じ station_id が来たら（送り直し）置き換える。
    途中で落ちても半端な置き場を残さないよう、別の名前で書いてから名前を変える。
    """
    d = station_dir(session, station_id)
    tmp = session / f".tmp_{station_id:03d}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name, data in images.items():
        (tmp / name).write_bytes(data)
    (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    return d


def station_ids(session: Path) -> list[int]:
    return sorted(
        int(d.name) for d in session.iterdir() if d.is_dir() and _STATION_DIR.fullmatch(d.name)
    )


def load_still(d: Path) -> Station:
    """止まって撮ったコマだけのステーション。走りながら撮ったコマはまだ VGGT に入れない
    （ぶれていて、首振りのコマと一緒に入れると位置合わせが崩れるかもしれない）。
    """
    st = load_station(d, "meta.json")
    return Station(st.dir, [f for f in st.frames if not f.get("moving")])


def read_meta(d: Path) -> dict:
    """ステーションの meta.json。1 回のリクエストで同じものを何度も読むので、時刻と大きさが
    同じ間は読み直さない（送り直されて書き換わったら読み直す）。返す dict は書き換えないこと。"""
    st = (d / "meta.json").stat()
    return _read_meta(str(d / "meta.json"), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=512)
def _read_meta(path: str, mtime_ns: int, size: int) -> dict:
    return json.loads(Path(path).read_text())


def supports(d: Path) -> set[str]:
    """Pi ができると書いてきたこと（"reverse"・"segments"）。書いていない Pi は今までどおり。"""
    return set(read_meta(d).get("supports") or [])


def camera_height_m(d: Path) -> float:
    h = read_meta(d).get("camera_height_cm")
    return (h or DEFAULT_CAMERA_HEIGHT_CM) / 100


def station_time(d: Path) -> float | None:
    """ステーションを撮った時刻（秒）。止まって撮ったコマの t_capture_ns のいちばん遅いもの
    （Pi の monotonic の時計。同じセッションの中で比べられる。make replay でも同じ値になる）。"""
    try:
        frames = read_meta(d).get("frames") or []
    except (OSError, ValueError):  # meta.json が無い・壊れている（試験の置き場など）
        return None
    ts = [f.get("t_capture_ns") for f in frames if isinstance(f, dict) and not f.get("moving")]
    ts = [float(t) for t in ts if isinstance(t, (int, float)) and math.isfinite(t)]
    return max(ts) / 1e9 if ts else None


# 人・snoo の箱の足もとを床に当てた所が、カメラからこれより遠ければ使わない（家具の上・遠い）
CAT_PLACE_MAX_M = 3.0


def station_cats(d: Path, pose: Pose) -> np.ndarray:
    """止まって撮ったコマの検出の猫の箱（frames[].dets）の足もとを床に当てた所 (K, 3)：
    [x, y, 撮った時刻（秒）]。pose はそのステーションの位置（カメラの真下）。箱の下の真ん中の
    画素を、首の向きとカメラの高さで床に当てる（app/snoo_agent_api.cat_place と同じ）。
    後ろへ下がる区切りを決めるのに使う（見た後に車の後ろへ回り込んでいるかもしれない）。"""
    from app.clearance import Camera, floor_point

    h_cm = camera_height_m(d) * 100
    out = []
    for f in read_meta(d).get("frames") or []:
        dets = f.get("dets")
        if f.get("moving") or not isinstance(dets, list):
            continue
        try:  # Pi から届いたもの。形が違えば使わない（500 にしない）
            cam = Camera.from_pi(float(f["tilt_deg"]), float(f["pan_deg"]), height_cm=h_cm)
            t = float(f["t_capture_ns"]) / 1e9
        except (KeyError, TypeError, ValueError):
            continue
        for det in dets:
            try:
                if float(det.get("cat", 0)) < CAT_MASK_MIN:
                    continue
                x1, _, x2, y2 = (min(1.0, max(0.0, float(v))) for v in det["box"])
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            fx, fy = floor_point(cam, (x1 + x2) / 2 * cam.width, y2 * cam.height)
            fx, fy = float(fx) / 100, float(fy) / 100
            if (
                not (math.isfinite(fx) and math.isfinite(fy))
                or math.hypot(fx, fy) > CAT_PLACE_MAX_M
            ):
                continue
            c, sn = math.cos(pose.yaw), math.sin(pose.yaw)
            out.append((pose.x + fx * c - fy * sn, pose.y + fx * sn + fy * c, t))
    return np.array(out, np.float64).reshape(-1, 3)


def ultrasonic_cm(d: Path) -> float | None:
    """そのステーションの前の超音波の読み（cm）。meta の ultrasonic_cm と、止まって撮ったコマごとの
    読みの中央値にする（1 回の読みはときどき跳ねる。実走で 34 cm の中に 61 cm が混じった）。
    MIN_ULTRASONIC_CM より短い・負の値は、測れなかった（Pi が -1・-2 を返すことがある）とする。
    """
    meta = read_meta(d)
    values = [meta.get("ultrasonic_cm")]
    values += [f.get("ultrasonic_cm") for f in meta["frames"] if not f.get("moving")]
    ok = [
        float(v)
        for v in values
        if isinstance(v, int | float)
        and not isinstance(v, bool)
        and math.isfinite(v)
        and v >= MIN_ULTRASONIC_CM
    ]
    return float(np.median(ok)) if ok else None


def read_poses(session: Path) -> dict[int, Pose]:
    """前に求めたステーションの位置（セッションの座標）。"""
    p = session / "poses.json"
    if not p.exists():
        return {}
    return {
        int(k): Pose(v["x"], v["y"], math.radians(v["yaw_deg"]))
        for k, v in json.loads(p.read_text()).items()
    }


def write_poses(session: Path, poses: dict[int, Pose]) -> None:
    data = {str(k): pose_dict(v) for k, v in sorted(poses.items())}
    tmp = session / ".poses.json"
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(session / "poses.json")


# --- 座標 ---


def to_session(anchor: Pose, p: Pose) -> Pose:
    """窓の座標（原点は窓のいちばん古いステーション）の位置を、セッションの座標に直す。"""
    return to_frame(anchor, p)


def obs_path(session: Path, station_id: int) -> Path:
    return station_dir(session, station_id) / "obs.npz"


def read_observations(
    session: Path, upto: int, poses: dict[int, Pose] | None = None
) -> list[Observation]:
    """upto までのステーションで見たもの（置き場の obs.npz）。無いステーションは飛ばす。

    poses（位置のグラフで決め直した位置）があれば、そこに位置のあるステーションの見たものを
    置き直す（obs.npz は置いたときの位置と一緒に残してあるので、new ∘ old⁻¹ で動かすだけ。
    新しく残す物は無い）。
    """
    out = []
    for i in station_ids(session):
        if i <= upto and obs_path(session, i).is_file():
            o = Observation.load(obs_path(session, i))
            if o.t_s is None:  # 前の版の obs.npz：時刻はステーションの meta から
                o.t_s = station_time(station_dir(session, i))
            if poses is not None and i in poses and poses[i] != o.pose:
                o = pg.replace_observation(o, poses[i])
            out.append(o)
    return out


# --- VGGT ---


@dataclass
class Recon:
    """VGGT の点（app.snoo_mapper.build_map に渡すもの）。"""

    extrinsic: np.ndarray
    points: np.ndarray
    conf: np.ndarray
    colors: np.ndarray | None


# refs・vggt.npz の置き場・時間を書く dict を受け、Recon を返す。
# テストでは GPU を使わない作り物にする
Reconstruct = Callable[[list[FrameRef], Path, dict], Recon]


# VGGT を読んだまま、この秒数ステーションが来なければプロセスを終えて GPU を手放す
# （systemd がすぐ起動し直し、次のステーションで読み直す。読むのは 7 秒ほど）。
# RX 7900 XTX は GPU を握ったプロセスが 2 つ以上あると、何もしていなくても休めず
# 120 W ほど食う（2026-10-30 に測った。1 つなら 6 W）。0 なら手放さない
GPU_IDLE_S = float(os.getenv("SNOO_MAP_GPU_IDLE_S", "120"))


class VggtReconstruct:
    """VGGT を GPU に載せて持つ（ステーションごとに読み直すと 5 秒以上かかる）。

    起動したときには読まず、最初のステーションで読む。GPU_IDLE_S 秒使わなければ、
    プロセスごと終えて GPU を手放す（GPU を握ったままだと、検出と合わせて GPU が休めない）。
    """

    def __init__(self, idle_s: float = GPU_IDLE_S):
        self.model = None
        self.idle_s = idle_s
        self.last_used = time.monotonic()
        # 受けている途中のリクエストの数と、終えると決めたか（self._lock で守る）。
        # 受けてから VGGT を呼ぶまでの間（画像を調べる・保存・GPU を待つ）に終えて切らないため
        self._lock = threading.Lock()
        self._in_flight = 0
        self._exiting = False
        if idle_s > 0:
            threading.Thread(target=self._watch_idle, daemon=True, name="vggt-idle").start()

    def enter(self) -> bool:
        """リクエストを受け始める。

        終えると決めた後なら False（503 を返す。Pi は自分で 1 歩動く）。
        """
        with self._lock:
            if self._exiting:
                return False
            self._in_flight += 1
            self.last_used = time.monotonic()
            return True

    def leave(self) -> None:
        with self._lock:
            self._in_flight -= 1
            self.last_used = time.monotonic()

    def _should_exit(self) -> bool:
        with self._lock:
            idle = time.monotonic() - self.last_used > self.idle_s
            if self.model is not None and idle and self._in_flight == 0:
                self._exiting = True
            return self._exiting

    def _watch_idle(self) -> None:
        while True:
            time.sleep(5)
            if self._should_exit():
                log.info("VGGT を %.0f 秒使っていないので、終えて GPU を手放す", self.idle_s)
                logging.shutdown()
                os._exit(0)

    def load(self) -> None:
        t = time.perf_counter()
        self.model = load_vggt_model()
        log.info("VGGT を読んだ（%.1f 秒）", time.perf_counter() - t)

    def __call__(self, refs: list[FrameRef], npz: Path, marks: dict) -> Recon:
        import torch

        self.last_used = time.monotonic()
        if self.model is None:
            self.load()
        params = MapParams()
        paths = [r.path for r in refs]
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        v = run_vggt(paths, self.model)
        marks["vggt"] = time.perf_counter() - t
        marks["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        t = time.perf_counter()
        save_vggt(v, npz)
        points, conf = unproject(v, params.stride, params.border_px)
        colors = frame_colors(paths, v.depth.shape[1:], params.stride)
        marks["points"] = time.perf_counter() - t
        self.last_used = time.monotonic()
        return Recon(v.extrinsic, points, conf, colors)


# ステーションごとに残す VGGT の点（scripts/snoo_replay.py がどのステーションでも決め直すため）。
# セッションの vggt.npz は毎回上書きされ、最後のステーションしか決め直せなかった。
# 深度の全部（56 MB）ではなく、build_map に渡す点だけを残す。半精度にすると、次の 1 歩の
# 候補の点数が僅差のときに選ぶものが変わり、同じ返事にならなかったので float32 のまま
RECON_FILE = "recon.npz"


def save_recon(path: Path, refs: list[FrameRef], rec: Recon) -> None:
    """refs の並びと、その VGGT の点を残す。書けなくても止めない。"""
    try:
        np.savez_compressed(
            path,
            keys=np.array([f"{Path(r.path).parent.name}/{Path(r.path).name}" for r in refs]),
            extrinsic=rec.extrinsic.astype(np.float32),
            points=rec.points.astype(np.float32),
            conf=rec.conf.astype(np.float32),
            colors=np.zeros((0,), np.uint8) if rec.colors is None else rec.colors,
        )
    except OSError as e:
        log.warning("%s を書けない: %s", path, e)


def load_recon(path: Path) -> tuple[list[str], Recon]:
    d = np.load(path)
    colors = d["colors"] if d["colors"].size else None
    recon = Recon(
        d["extrinsic"], d["points"].astype(np.float32), d["conf"].astype(np.float32), colors
    )
    return [str(k) for k in d["keys"]], recon


# --- 1 ステーションを処理する ---


# 走りながらの SLAM（app/snoo_slam.py）の点と道筋を間取りに足すか
USE_SLAM = os.getenv("SNOO_MAP_USE_SLAM", "1") == "1"
# VGGT を回している間置いておくファイル。走りながらの SLAM が同じ GPU で重なると、
# 1 か所に止まるたびに両方が遅れ、Pi の待ち（25 秒）を越えかねない（2026-10-30）
GPU_BUSY_FILE = os.getenv("SNOO_GPU_BUSY_FILE", "/run/snoo/map_busy")


@contextmanager
def gpu_busy():
    """GPU_BUSY_FILE を置いてから中を走らせ、終われば（落ちても）消す。書けなくても止めない。"""
    path = Path(GPU_BUSY_FILE) if GPU_BUSY_FILE else None
    try:
        if path is not None:
            path.write_text(str(os.getpid()))
    except OSError:
        path = None
    try:
        yield
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


# --- 位置のグラフ（app/snoo_posegraph.py） ---

# ステーションの位置を、窓の答えと走った 1 歩を重ねた位置のグラフで決めるか。
# 窓の原点に足すだけ（今までの決め方）では、1 回の VGGT が崩れるとそのステーションの位置が無くなり、
# 窓の原点のずれも後へ積もる。保存した explore_20261030_190244（75 か所）で比べると、位置のある
# ステーションが 66 → 75 か所、走った 1 歩との差（中央値・90 %）が 15.0・49.9 → 6.8・14.6 cm、
# 壁の重なりが 0.75 → 0.90 になった（scripts/snoo_pose_bakeoff.py の pg_live）。
# SNOO_POSE_GRAPH=0 で今までの決め方に戻す
POSE_GRAPH = os.getenv("SNOO_POSE_GRAPH", "1") == "1"
GRAPH_EDGES = "graph_edges.jsonl"
GRAPH_POSES = pg.GRAPH_POSES
# グラフで落ちたことを書いたセッション（同じセッションでは 1 回だけ書く）
_graph_failed: set[str] = set()


@dataclass
class GraphPoses:
    """位置のグラフを解いた答え。poses は fixed からたどれる（位置の決まった）ステーションだけ。"""

    poses: dict[int, Pose]
    # 自分の窓で位置合わせが崩れたステーション（見たものを地図に足さない）
    bad: set[int]
    info: dict


def graph_failed(session: Path, what: str) -> None:
    """グラフで落ちたことを、セッションごとに 1 回だけ書く（落ちても今までの決め方で続ける）。"""
    if session.name not in _graph_failed:
        _graph_failed.add(session.name)
        log.exception(
            "%s: 位置のグラフ（%s）で落ちたので、今までの決め方で続ける", session.name, what
        )


def odom_edge_for(session: Path, station_id: int) -> pg.Edge | None:
    """1 つ前のステーションから station_id まで走った辺（prev_step の区切りを、Pi が見積もった
    長さで切ったもの。区切りごとの前・後ろも入れる）。前のステーションが無い・prev_step が無い・
    長さが分からなければ None。
    """
    a = station_id - 1
    if a < 0 or not (station_dir(session, a) / "meta.json").is_file():
        return None
    prev = read_meta(station_dir(session, station_id)).get("prev_step")
    if not isinstance(prev, dict):
        return None
    moves, dist = step_moves(prev, sent_back(session, a)), pi_distance_m(prev)
    if moves is None or dist is None:
        return None
    return pg.odom_edge_moves(a, station_id, take_moves(moves, dist))


def read_graph_edges(session: Path) -> list[dict]:
    p = session / GRAPH_EDGES
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def write_graph_edges(session: Path, station_id: int, edges: list[pg.Edge]) -> list[dict]:
    """station_id の辺を graph_edges.jsonl に足す（送り直しなら、前の分を置き換える）。
    辺は 1 か所 7 本ほどと小さいので、毎回全部を書き直す（書きかけを読ませない）。
    """
    rows = [d for d in read_graph_edges(session) if d.get("station") != station_id]
    rows += [pg.edge_to_dict(e, station_id) for e in edges]
    atomic.write_text(session / GRAPH_EDGES, "".join(json.dumps(d) + "\n" for d in rows))
    return rows


def read_graph_state(session: Path) -> dict:
    p = session / GRAPH_POSES
    return json.loads(p.read_text()) if p.is_file() else {}


def solve_graph(
    session: Path,
    station_id: int,
    window: list[int],
    results: list,
    heights: list[float | None],
    known: dict[int, Pose],
    extra: list[pg.Edge] | None = None,
) -> GraphPoses:
    """このステーションの窓の辺と走った辺（と extra。RoMa の辺）を足し、全部の辺で位置を解き直す。

    results は窓の座標のステーションの位置（build_map の答えを to_session で直す前のもの）。
    known は今までの決め方の位置（セッションの座標）。グラフは最初のステーションを止めて解き、
    その位置を known の位置に合わせる（最初のステーションの位置は、最初の窓の VGGT のコマの
    向きで決まっていて (0, 0, 0) ではない。合わせないと、グラフを使うかで座標がずれる）。
    前の答え（graph_poses.json）から始めるので、1 か所ごとに解き直しても速く、答えも揺れない。
    """
    w = pg.Window(f"live:{station_id:03d}", list(window), list(results), list(heights))
    edges = pg.window_edges(w)
    odom = odom_edge_for(session, station_id)
    if odom is not None:
        edges.append(odom)
    edges += extra or []
    rows = write_graph_edges(session, station_id, edges)
    state = read_graph_state(session)
    bad = {int(i) for i in state.get("bad", [])} - {station_id}
    last = results[-1]
    if not last.frames or registration_is_bad(last, heights[-1]):
        bad.add(station_id)
    all_edges = [pg.edge_from_dict(d) for d in rows]
    if state.get("fixed") is not None and state.get("base"):
        fixed, base = int(state["fixed"]), Pose(*map(float, state["base"]))
    else:
        # ふつうは最初のステーション。セッションの途中でグラフを使い始めた（サーバーを入れ替えた）
        # ときは、辺に出てくる位置の分かったステーションのうち最初のものを止める
        # （最初のステーションには辺が無く、ほかがどれもつながらないため）
        fixed = station_ids(session)[0]
        linked = sorted(i for e in all_edges for i in (e.a, e.b) if i in known)
        if all_edges and fixed not in {i for e in all_edges for i in (e.a, e.b)} and linked:
            fixed = linked[0]
        base = known.get(fixed, Pose(0.0, 0.0, 0.0))
    back = pg.inverse(base)
    sol = None
    solved = {fixed: Pose(0.0, 0.0, 0.0)}  # グラフの座標（fixed が原点）
    if all_edges:
        ids = sorted({e.a for e in all_edges} | {e.b for e in all_edges} | {fixed})
        start = {}
        if state.get("fixed") == fixed:
            start = {int(k): to_frame(back, Pose(*map(float, v)))
                     for k, v in state.get("poses", {}).items()}  # fmt: skip
        init = pg.chain_init(ids, all_edges, fixed, start=start)
        sol = pg.optimize(all_edges, fixed, init=init)
        solved = sol.poses
    ok = pg.connected(sol.edges, fixed) if sol else {fixed}
    every = {i: to_frame(base, p) for i, p in solved.items()}
    poses = {i: p for i, p in every.items() if i in ok}
    outliers = [
        {"a": e.a, "b": e.b, "kind": e.kind, "source": e.source, "chi": round(e.chi, 1)}
        for e in (sol.edges if sol else []) if e.outlier
    ]  # fmt: skip
    info = {
        "fixed": fixed,
        "edges": len(all_edges),
        "outliers": len(outliers),
        "cost": None if sol is None else round(sol.cost, 4),
        "localized": len(poses),
        "station_localized": station_id in poses,
        # 走った辺だけで決めた（このステーションの窓の辺がどれも無いか外れた）
        "station_odom_only": station_id in poses and (sol is None or station_id not in sol.seen),
    }
    data = {
        **info,
        "base": [base.x, base.y, base.yaw],
        # セッションの座標（poses.json と同じ）。localized に無いものは仮の位置
        "poses": {str(i): [p.x, p.y, p.yaw] for i, p in sorted(every.items())},
        "localized": sorted(poses),
        "bad": sorted(bad),
        "outlier_edges": outliers,
    }
    atomic.write_text(session / GRAPH_POSES, json.dumps(data, indent=1))
    return GraphPoses(poses, bad, info)


def graph_poses(
    session: Path,
    station_id: int,
    window: list[int],
    results: list,
    heights: list,
    known: dict[int, Pose],
    extra: list[pg.Edge] | None = None,
) -> GraphPoses | None:
    """solve_graph。落ちたら None（今までの決め方で続ける。地図のサーバーを 500 にしない）。"""
    try:
        return solve_graph(session, station_id, window, results, heights, known, extra)
    except Exception:
        graph_failed(session, "解く")
        return None


# --- RoMa v2 の辺（app/snoo_roma.py） ---

# 新しいステーションと前のステーション、近くに戻ってきた組の候補 1 つを RoMa v2 に通し、
# 位置のグラフに "roma" の辺を足すか（SNOO_POSE_GRAPH=1 のときだけ）。既定は使わない。
# VGGT が白い壁・暗い所で崩れたステーションでも、向きを 2〜3° で測れる。保存した
# explore_20261030_190244 で、壁の重なりが 0.74 → 0.84〜0.86、確かめに取っておいた戻ってきた組
# との差が 6.7・20.5 → 1.7・15.0 cm（中央値・90 %）になった（scripts/snoo_roma_eval.py）。
# この口を 1 か所ずつ流し直すと（scripts/snoo_graph_replay.py --roma）、その差は 5.7・12.0 →
# 2.1・5.8 cm、走った辺だけで決まったステーションは 9 → 2 か所
# RoMa v2 は venv に入れずに、clone した src（SNOO_ROMA_SRC）から読む。重みは約 1.5 GB、
# GPU のメモリは 1 組ずつで 3.5 GB ほど。1 組 0.6 秒ほどで、1 か所 2 ジョブ・8 組まで
ROMA_EDGES = os.getenv("SNOO_ROMA_EDGES", "0") == "1"
# 1 組ごとに、かかった時間の何倍休むか（家のブレーカーが落ちたので、GPU の電力の山を下げる）
ROMA_REST = float(os.getenv("SNOO_ROMA_REST", "1.0"))
_roma_model = None
_roma_failed: set[str] = set()


def roma_match(pairs: list[tuple[str, str]]) -> list[dict]:
    """コマの組を 1 組ずつ RoMa v2 に通す。最初に呼ばれたときに読む。"""
    global _roma_model
    if _roma_model is None:
        t = time.perf_counter()
        _roma_model = sr.load_model()
        log.info("RoMa v2 を読んだ（%.1f 秒）", time.perf_counter() - t)
    out = []
    for k, p in enumerate(pairs):
        t = time.perf_counter()
        out += sr.match_pairs(_roma_model, [p])
        if ROMA_REST > 0 and k < len(pairs) - 1:
            time.sleep(ROMA_REST * (time.perf_counter() - t))
    return out


def roma_edges_for(session: Path, station_id: int, match=None) -> tuple[list[pg.Edge], list]:
    """station_id の RoMa の辺（前のステーションと、戻ってきた組の候補 1 つまで）と記録。
    今の位置は graph_poses.json（前のステーションまで）に、走った辺を足したもの。
    落ちたら空（セッションごとに 1 回だけ書く。地図のサーバーは止めない）。
    """
    try:
        state = read_graph_state(session)
        prior = {int(k): Pose(*map(float, v)) for k, v in state.get("poses", {}).items()}
        odom = odom_edge_for(session, station_id)
        if odom is not None and station_id - 1 in prior:
            prior[station_id] = to_frame(prior[station_id - 1], odom.z)
        need = {station_id - 1, station_id, sr.revisit_candidate(station_id, prior)}
        dirs = {i: station_dir(session, i) for i in need if i is not None and i >= 0}
        dirs = {i: d for i, d in dirs.items() if (d / "meta.json").is_file()}
        if station_id not in dirs:
            return [], []
        stills = {i: load_still(d).frames for i, d in dirs.items()}
        dyaw = math.degrees(odom.z.yaw) if odom is not None else None
        return sr.station_edges(stills, dirs, station_id, prior, match or roma_match, dyaw)
    except Exception:
        if session.name not in _roma_failed:
            _roma_failed.add(session.name)
            log.exception("%s: RoMa の辺で落ちたので、RoMa の辺なしで続ける", session.name)
        return [], []


def bad_stations(session: Path) -> set[int]:
    """自分の窓で位置合わせが崩れたステーション（graph_poses.json。無ければ空）。"""
    try:
        return {int(i) for i in read_graph_state(session).get("bad", [])}
    except (OSError, ValueError):
        return set()


def plan_station(
    session: Path,
    station_id: int,
    reconstruct: Reconstruct,
    marks: dict,
    home_root: Path | None = None,
) -> dict:
    """セッションのステーションから間取りを作り直し、次の 1 歩を返す（Pi に返す形）。

    VGGT は窓（WINDOW か所）だけに通し、最後のステーションの位置と見たものをセッションの座標に
    直して置き場に残す（obs.npz）。次の 1 歩は、全部のステーションの見たものを足した
    セッションの間取り（session_grid）で決める。窓の外で見た所も覚えているので、見た所へ
    戻らない。room.json などもセッションの座標で書く。

    home_root（SNOO_HOME_MAP=1 のとき）があれば、セッションを家の地図に合わせて足し、合っていれば
    次の 1 歩をセッションと家を合わせた格子で決める（use_home）。
    """
    ids = station_ids(session)
    poses = read_poses(session)
    window = choose_window(ids, station_id, poses, WINDOW)
    dirs = [station_dir(session, i) for i in window]
    stations = [load_still(d) for d in dirs]
    can = supports(dirs[-1])
    params = MapParams(
        camera_height_m=[camera_height_m(d) for d in dirs],
        ultrasonic_cm=[ultrasonic_cm(d) for d in dirs],
        segments="segments" in can,
        reverse="reverse" in can,
        gear_segments={"gear_segments", "reverse", "segments"} <= can,
        max_step_m=LONG_STEP_M if {"long_step", "segments"} <= can else MAX_STEP_M,
        max_stations=SESSION_MAX_STATIONS,
        seed=station_id,
    )
    refs = select_frames(stations, MAX_FRAMES, old_frames=OLD_FRAMES)
    with gpu_busy():  # SLAM（app/snoo_slam.py）は、この間サブマップを回さずに待つ
        rec = reconstruct(refs, session / "vggt.npz", marks)
    # 写っていた猫を間取りに入れない（Pi が止まって撮ったコマごとに frames[].dets を付けたとき）
    dets = {(d.name, f["file"]): f.get("dets") for d, st in zip(dirs, stations, strict=True)
            for f in st.frames}  # fmt: skip
    marks["cat_points"] = mask_cats(
        rec.conf, [dets.get((Path(r.path).parent.name, Path(r.path).name)) for r in refs]
    )
    # 猫を消した後の点を残す（make replay で決め直すと、同じ返事になるように）
    save_recon(station_dir(session, station_id) / RECON_FILE, refs, rec)

    t = time.perf_counter()
    # 窓の原点（いちばん古いステーション）の位置。最初のステーションなら (0, 0, 0)
    anchor = poses.get(window[0], Pose(0.0, 0.0, 0.0))
    r = build_map(
        stations, refs, rec.extrinsic, rec.points, rec.conf, params, rec.colors,
        anchor=anchor, plan=False,
    )  # fmt: skip
    for i, s, h in zip(window, r.stations, params.camera_height_m, strict=True):
        # 一度求めた位置は変えない（変えると、それを原点にした後の窓と座標がずれる）
        if i not in poses and s.frames and not registration_is_bad(s, h):
            poses[i] = to_session(anchor, s.pose)
    graph = None
    if POSE_GRAPH:
        extra = []
        if ROMA_EDGES:
            t_roma = time.perf_counter()
            with gpu_busy():
                extra, jobs = roma_edges_for(session, station_id)
            marks["roma"] = time.perf_counter() - t_roma
            marks["roma_edges"] = float(len(extra))  # marks は数だけ（ログで %.2f にする）
            if jobs:
                log.info("%s/%03d: RoMa %s", session.name, station_id, jobs)
        # 窓の座標のままのステーションの位置（to_session で直す前）で、窓の辺を作る
        t_graph = time.perf_counter()
        graph = graph_poses(
            session, station_id, window, r.stations, params.camera_height_m, poses, extra
        )
        marks["graph"] = time.perf_counter() - t_graph
    # 自分の窓で崩れたステーションは、後の窓で位置が合っても見たものを足さない（点が当てにならない）
    bad = graph.bad if graph is not None else bad_stations(session) if POSE_GRAPH else set()
    for i, o in zip(window, r.observations, strict=True):
        # 見たものも 1 回だけ残す（ふつうは最後のステーション。前の版で受けたステーションも）。
        # 置いた位置も一緒に残る（Observation.pose）。グラフが位置を直したら、読むときに置き直す
        if i in poses and o is not None and i not in bad and not obs_path(session, i).is_file():
            d = station_dir(session, i)
            o.t_s, o.cats = station_time(d), station_cats(d, o.pose)
            o.save(obs_path(session, i))
    if graph is not None:
        # グラフで位置の決まったステーションは、グラフの位置にする（窓で崩れたものも、
        # 走った辺とほかの窓から決まる）。決まらないもの（つながらない）は今までの位置のまま
        poses.update(graph.poses)
    write_poses(session, poses)

    # 窓の結果をセッションの座標に直し、セッションの間取りで次の 1 歩を決める
    localized = set(graph.poses) if graph is not None else set()
    r.stations = [
        dataclasses.replace(s, pose=poses[i] if i in localized else to_session(anchor, s.pose))
        for i, s in zip(window, r.stations, strict=True)
    ]
    # 窓で崩れても、グラフで位置の決まったステーションでは止めない（戻らずにそのまま探る）
    trusted = frozenset(s.name for i, s in zip(window, r.stations, strict=True) if i in localized)
    if graph is not None:
        r.pose_graph = {**graph.info, "trusted": sorted(trusted)}
    r.points = xy_to_frame(anchor, r.points) if len(r.points) else r.points
    if r.floor.inliers:
        observations = None
        if graph is not None:
            try:  # グラフが直した位置へ、前に見たものを置き直す（地図ごと動く）
                observations = read_observations(session, station_id, graph.poses)
            except Exception:
                graph_failed(session, "見たものを置き直す")
        if observations is None:
            observations = read_observations(session, station_id)
        r.grid, r.high = session_grid(observations)
        # いまの時刻（後ろへ下がる区切りで、古い床と最近見た人・snoo を見る。Pi の時計）
        r.grid.now_s = station_time(dirs[-1])
        # 今のステーションで見えた人・snoo は、位置合わせが崩れて obs.npz を残さなくても足す
        # （見た直後に車の後ろへ回り込んでいるかもしれない）
        if station_id in poses:
            r.grid.cats.extend(map(tuple, station_cats(dirs[-1], poses[station_id])))
        if params.driven_free:
            mark_steps_driven(r.grid, session, station_id, poses)
        if USE_SLAM:  # 走りながらの SLAM で見た所も足す（app/snoo_slam.py が書く）
            r.slam = mark_slam(r.grid, r.high, session / "slam", params)
        if home_root is not None:
            t_home = time.perf_counter()
            r.grid, r.home = use_home(home_root, session, station_id, r.grid, r.high)
            marks["home"] = time.perf_counter() - t_home
        recent = recent_poses(session, station_id, poses)
        r.action, r.plan = decide(
            r.grid, r.stations, params, params.camera_height_m, trusted, recent
        )
        r.step_path = []
        if r.plan is not None:
            start = r.stations[-1].pose
            r.step_path = [camera_of(p) for _, p, _, _ in drive_moves(start, r.plan.moves)]
    marks["map"] = time.perf_counter() - t

    t = time.perf_counter()
    raw = r.action
    action = dict(raw)
    pose = r.stations[-1].pose
    back = retrace_after_bad(session, station_id, raw, r.stations[-1].name, params)
    if back:
        # このステーションの位置合わせが崩れ、グラフでも位置が決まらない。探索を終えずに、いま
        # 走った 1 歩を逆にたどって（区切りの順と向きを逆に）、位置の分かっている前のステーションで
        # 撮り直す。走った所は床と分かっているので、後ろへも通れる
        action = moves_action(back, params)
        action["retrace"] = True
        r.step_path = [camera_of(p) for _, p, _, _ in drive_moves(pose, back)]
        r.action = dict(action)
    override = take_override(root_of(session), session.name, r.grid, pose, params)
    if override is not None:
        # エージェント（画像を見て決める AI）が予約した 1 歩。地図で確かめて通ったものだけ
        action = override
        r.step_path = [camera_of(p) for _, p, _, _ in drive_moves(pose, action_moves(action))]
        r.action = dict(action)
    if action["action"] == "move":
        here = poses.get(station_id)
        if here is None:  # 位置合わせが崩れていれば decide が done にするので、ふつうは来ない
            here = r.stations[-1].pose
        action["pose"] = pose_dict(here)
        action["station_index"] = ids.index(station_id)
        action["map_url"] = f"/v1/snoo/map/{session.name}/topdown.png"
    else:
        # bad_registration:<名前> の名前は Pi には要らない（plan.json には残す）
        action = {"action": "done", "reason": action["reason"].split(":")[0]}
    write_results(session, r, window, anchor, raw, marks, action)
    (station_dir(session, station_id) / "reply.json").write_text(json.dumps(action))
    marks["write"] = time.perf_counter() - t
    return action


# --- エージェントが予約した次の 1 歩 ---

# 予約が古ければ使わない（見た画像のときから車が動いている）
OVERRIDE_TTL_S = 90.0
OVERRIDE_MAX_CM = 40.0


def root_of(session: Path) -> Path:
    return session.parent


def override_path(root: Path) -> Path:
    return root.parent / "snoo_agent" / "override.json"


def take_override(root: Path, session: str, grid, pose: Pose, params: MapParams) -> dict | None:
    """エージェントが POST /v1/snoo/agent/next で予約した 1 歩を、このステーションの返事にする。

    予約は 1 回使えば消す。古い（OVERRIDE_TTL_S）・別のセッション・長すぎる・地図で通らない
    （計画と同じ決まり。app/snoo_maneuver.moves_ok）ものは使わず、理由を
    override_result.jsonl に残して、いつもの決め方に任せる。
    """
    path = override_path(root)
    try:
        o = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    path.unlink(missing_ok=True)
    why = None
    steer, cm, back = o.get("steer_deg"), o.get("distance_cm"), bool(o.get("reverse"))
    if time.time() - float(o.get("t", 0)) > OVERRIDE_TTL_S:
        why = "古い"
    elif o.get("session") not in (None, session):
        why = "別のセッション"
    elif not _finite(steer) or not _finite(cm) or not 0 < cm <= OVERRIDE_MAX_CM:
        why = "長さかハンドルがおかしい"
    elif abs(steer) > MAX_STEER_DEG + 0.5:
        why = "ハンドルが大きすぎる"
    elif grid is None:
        why = "地図が無い"
    else:
        why = moves_ok(grid, pose, [(float(steer), float(cm) / 100, back)], limits(params))
    log_path = path.parent / "override_result.jsonl"
    row = {"t": time.time(), "session": session, "override": o, "used": why is None, "why": why}
    with log_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if why is not None:
        log.info("エージェントの 1 歩を使わない（%s）: %s", why, o)
        return None
    action = moves_action([(float(steer), float(cm) / 100, back)], params)
    action["agent"] = True
    if o.get("note"):
        action["note"] = str(o["note"])[:200]
    return action


# --- 家の地図（app/snoo_home.py。SNOO_HOME_MAP=1 のとき） ---

# 家の地図を使うか。まだ試しなので既定は使わない
HOME_MAP = os.getenv("SNOO_HOME_MAP", "0") == "1"
HOME_DIR = os.getenv("SNOO_HOME_DIR", "data/snoo_map/home")


def use_home(home_root: Path, session: Path, station_id: int, grid, high):
    """セッションを家の地図に合わせて足し、(次の 1 歩に使う格子, plan.json に残す様子) を返す。

    合っていれば、セッションで見ていない所を家の地図で埋めた格子（障害物はどちらかで障害物なら
    障害物）。見た所へまた行かず、家の地図の境目へ向かう。合っていない・落ちたときは
    セッションの格子のまま（家の地図が無いときと同じに動く）。
    """
    from app.snoo_home import on_station

    try:
        union, info = on_station(home_root, session, station_id, grid, high)
    except Exception as e:  # 家の地図で探索を止めない
        log.exception("%s/%03d: 家の地図を使えなかった", session.name, station_id)
        return grid, {"error": f"{type(e).__name__}: {e}"}
    return (union if union is not None else grid), info


# 地図のサーバーの 1 セッションのステーションの上限（MapParams の既定 50 では 10 分ほどで終える）
SESSION_MAX_STATIONS = int(os.environ.get("SNOO_MAX_STATIONS", "2000"))
MAX_STEER_DEG = round(MANEUVER_STEER_DEG, 1)


# --- 位置合わせが崩れたら、来た道を逆にたどって撮り直す ---

# これより短く走っただけなら、戻らずに終える（同じ所で撮り直しても同じになりやすい）
RETRACE_MIN_M = 0.05


def retrace_after_bad(
    session: Path, station_id: int, raw: dict, last_name: str, params: MapParams
) -> list[tuple[float, float, bool]] | None:
    """最後のステーションの位置合わせが崩れたとき、いま走った 1 歩を逆にたどる区切り。

    同じハンドルで逆の向きに走れば同じ弧を戻る（自転車モデル）ので、区切りの順と向きを逆にすれば、
    走ったばかりの床の上だけを通って前のステーションへ戻れる。続けて崩れたら（前の返事も
    戻る 1 歩なら）・Pi が走れない形（向きの混じった列を gear_segments の無い Pi へ、後ろへを
    reverse の無い Pi へ）なら None。後ろへの合計は 30 cm まで。
    """
    if raw.get("reason") != f"bad_registration:{last_name}":
        return None
    prev = read_meta(station_dir(session, station_id)).get("prev_step")
    if not isinstance(prev, dict):
        return None
    before = station_dir(session, station_id - 1) / "reply.json"
    if before.is_file() and json.loads(before.read_text()).get("retrace"):
        return None
    moves, moved = step_moves(prev, sent_back(session, station_id - 1)), pi_distance_m(prev)
    if not moves or moved is None or moved < RETRACE_MIN_M:
        return None
    if sum(m for _, m, _ in moves) <= 0:  # 長さの無い 1 本の弧（古い Pi）は、走った長さにする
        moves = [(moves[0][0], moved, moves[0][2])]
    out = [(st, m, not b) for st, m, b in reversed(take_moves(moves, moved)) if m > 1e-3]
    rev, kept = 0.0, []
    for st, m, b in out:
        cut = b and m > REVERSE_MAX_M - rev
        if cut:
            m = REVERSE_MAX_M - rev
        rev += m if b else 0.0
        if m > 1e-3:
            kept.append((st, m, b))
        if cut:
            # 後ろへを途中で切ったら、そこで終える（その先の区切りは、走った道から外れた所から
            # 走ることになる）
            break
    if not kept or len(kept) > (GEAR_MAX_SEGMENTS if params.gear_segments else MAX_SEGMENTS):
        return None
    if len({b for _, _, b in kept}) > 1 and not params.gear_segments:
        return None
    if any(b for _, _, b in kept) and not params.reverse:
        return None
    return kept


# --- 走った 1 歩（prev_step）を読む ---


def step_moves(prev: dict, back: bool | None = None) -> list[tuple[float, float, bool]] | None:
    """prev_step の区切り [(ハンドル°, m, 後ろへか), ...]。segments（返した形のまま、走った分）が
    あればそれ、無ければ steer_deg の 1 本の弧（長さは分からないので 0 にし、測った分だけ伸ばす）。
    区切りの reverse が無ければ prev_step の reverse、それも無ければ back（古い Pi のために、
    そのとき返した返事から決める）。"""
    rev = prev.get("reverse")
    rev = bool(back) if rev is None else bool(rev)
    segs = prev.get("segments")
    if isinstance(segs, list) and segs:
        out = []
        for x in segs:
            st = x.get("steer_deg") if isinstance(x, dict) else None
            cm = x.get("distance_cm") if isinstance(x, dict) else None
            if not _finite(st) or not _finite(cm) or cm < 0:
                return None
            r = x.get("reverse")
            out.append((float(st), float(cm) / 100, rev if r is None else bool(r)))
        return out
    st = prev.get("steer_deg")
    return [(float(st), 0.0, rev)] if _finite(st) else None


def _finite(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def pi_distance_m(prev: dict) -> float | None:
    """Pi が見積もった、走った長さ（m）。est_distance_cm、無ければ 秒 × speed × 1.7 cm/s。"""
    est = prev.get("est_distance_cm")
    if _finite(est) and est >= 0:
        return float(est) / 100
    sec, speed = prev.get("seconds"), prev.get("speed")
    if _finite(sec) and _finite(speed) and sec >= 0:
        return float(sec) * abs(float(speed)) * PI_CM_PER_S_PER_SPEED / 100
    return None


def moved_fit(prev: dict, moves, a: Pose | None, b: Pose | None) -> tuple[float | None, bool]:
    """(区切りに沿って実際に走った長さ m, VGGT の位置で確かめられたか)。

    VGGT の 2 つの位置に、区切りを自転車モデルで走らせた位置を合わせて測る（Pi の見積もりより
    確か。1 回目の実走では Pi の見積もりは本当の 0.6 倍だった）。合わなければ（位置が無い・
    ずれが大きい）Pi の見積もりを使う。
    """
    if a is not None and b is not None:
        s, err = fit_moves(a, b, moves, hint=pi_distance_m(prev))
        if err <= FIT_TOL_M:
            return s, True
    return pi_distance_m(prev), False


def sent_back(session: Path, station_id: int) -> bool:
    """station_id で返した 1 歩が後ろへだったか（prev_step に reverse を書かない古い Pi のため。
    前の版の reverse.json も読む）。"""
    p = station_dir(session, station_id) / "reply.json"
    if p.is_file():
        try:
            return bool(json.loads(p.read_text()).get("reverse"))
        except ValueError:
            return False
    return station_id in read_reverses(session)


def steps_driven(
    session: Path, station_id: int, poses: dict[int, Pose], fitted_only: bool = False
) -> list[tuple[int, list[tuple[float, float, bool]] | None]]:
    """station_id までの、ステーションの間に走った 1 歩ごとの (前のステーション, 区切り)。
    区切りは実際に走った長さまで。どう動いたか分からない所は None。
    fitted_only なら、VGGT の位置で確かめられなかった歩も None にする（床にするとき。位置が
    8 cm を超えてずれていると、車の外形分が別の所で床になる。コードレビュー）。
    """
    out: list[tuple[int, list[tuple[float, float, bool]] | None]] = []
    before = None
    for i in station_ids(session):
        if i > station_id:
            break
        prev = read_meta(station_dir(session, i)).get("prev_step")
        if before is None or i != before + 1 or not isinstance(prev, dict):
            out.append((i, None))
        else:
            moves = step_moves(prev, sent_back(session, before))
            s, fitted = (
                (None, False)
                if moves is None
                else moved_fit(prev, moves, poses.get(before), poses.get(i))
            )
            if s is None or (fitted_only and not fitted):
                out.append((before, None))
            else:
                out.append((before, take_moves(moves, s)))
        before = i
    return out


def mark_steps_driven(grid, session: Path, station_id: int, poses: dict[int, Pose]) -> None:
    """走ってきた道（車の外形の下）を、セッションの間取りの床にする（roommap.mark_driven）。
    Pi が途中で止めた（clear・stuck）1 歩は、止まった所の先をたぶん障害物にする
    （roommap.mark_blocked_ahead。足もとは写らないので、そこにある物が地図に入らなかった）。
    止まった区切りは、Pi の見積もりの長さまで走った区切りの最後（古い Pi は返した区切りを全部
    返す）。足すのは直近 BLOCKED_KEEP_STATIONS か所で止まった分だけ（作り直すたびに足すので、
    前を横切った猫で止まった所が、ずっと障害物に寄ったままにならない）。"""
    for before, moves in steps_driven(session, station_id, poses, fitted_only=True):
        if moves and before in poses:
            # 走った時刻は、走り出したステーションの時刻とする（古めに見る）
            mark_driven_moves(
                grid, poses[before], moves, station_time(station_dir(session, before))
            )
    for i in station_ids(session):
        if i > station_id or i <= station_id - BLOCKED_KEEP_STATIONS or i not in poses:
            continue
        prev = read_meta(station_dir(session, i)).get("prev_step")
        if isinstance(prev, dict) and prev.get("stop_reason") in ("clear", "stuck"):
            moves = step_moves(prev, sent_back(session, i - 1))
            if moves:
                moved = pi_distance_m(prev)
                driven = take_moves(moves, moved) if moved is not None else moves
                st, _, back = (driven or moves[:1])[-1]
                mark_blocked_ahead(grid, poses[i], st, back)


# Pi が止まった所の先を「たぶん障害物」にするのは、直近のこれだけのステーションで止まった分
BLOCKED_KEEP_STATIONS = RECENT_STATIONS


def recent_poses(session: Path, station_id: int, poses: dict[int, Pose]) -> list[Pose]:
    """直近 RECENT_STATIONS か所（station_id まで）の位置（動けていないか・戻るかを見る）。"""
    ids = [i for i in station_ids(session) if i <= station_id and i in poses]
    return [poses[i] for i in ids[-RECENT_STATIONS:]]


def read_reverses(session: Path) -> dict[int, dict]:
    """前の版が残した、後ろへ戻れと返したステーションと返事（reverse.json）。今は書かない。"""
    p = session / "reverse.json"
    if not p.exists():
        return {}
    return {int(k): v for k, v in json.loads(p.read_text()).items()}


def write_results(
    session: Path, r, window: list[int], anchor: Pose, raw: dict, marks: dict, reply: dict
):
    """room.json などを書く。ブラウザで見ている途中に半端なファイルを読ませないよう、
    別の置き場に書いてから 1 つずつ名前を変える。
    """
    tmp = session / ".out"
    shutil.rmtree(tmp, ignore_errors=True)
    write_outputs(r, tmp)
    # 間取り・ステーション・目標はセッションの座標（原点 (0, 0, 0)）で書く。
    # vggt_origin_pose は VGGT の窓の原点（vggt.npz の座標）
    info = {
        "stations": window,
        "origin_pose": pose_dict(Pose(0.0, 0.0, 0.0)),
        "vggt_origin_pose": pose_dict(anchor),
        "map": "session",
    }
    room = json.loads((tmp / "room.json").read_text())
    room["window"] = info
    (tmp / "room.json").write_text(json.dumps(room, ensure_ascii=False, indent=1))
    plan = json.loads((tmp / "plan.json").read_text())
    plan["next"] = raw
    plan["reply"] = reply  # Pi に返したもの（後ろへ戻るときは next と違う）
    plan["window"] = info
    plan["timing_s"] = {k: round(v, 3) for k, v in marks.items()}
    plan["slam"] = getattr(r, "slam", None)  # 走りながらの SLAM を間取りに足したか
    if getattr(r, "home", None) is not None:  # 家の地図に合わせたか（SNOO_HOME_MAP=1）
        plan["home"] = r.home
    if getattr(r, "pose_graph", None) is not None:  # 位置のグラフで決めたか（SNOO_POSE_GRAPH=1）
        plan["pose_graph"] = r.pose_graph
    (tmp / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    for f in tmp.iterdir():
        f.replace(session / f.name)
    tmp.rmdir()


# --- HTTP ---


# 重いファイルを残す、新しいセッションの数
PRUNE_KEEP = int(os.getenv("SNOO_MAP_PRUNE_KEEP", "5"))


def _prune_old(root: Path) -> None:
    from scripts.snoo_prune import prune

    try:
        n, freed = prune(root, PRUNE_KEEP)
        if n:
            log.info("古いセッションの重いファイルを %d 個消した（%.0f MB）", n, freed / 1e6)
    except OSError:
        log.exception("古いセッションを片付けられなかった")


def create_app(
    root: Path | str = DATA,
    allow: list | None = None,
    reconstruct: Reconstruct | None = None,
    live_dir: Path | str | None = LIVE_DIR,
    home_dir: Path | str | None = None,
) -> FastAPI:
    """home_dir を渡すか SNOO_HOME_MAP=1 なら、家の地図（app/snoo_home.py）を使う。"""
    from app.snoo_twin import add_twin  # snoo_twin がこのモジュールを使うので、ここで読む

    root = Path(root)
    if home_dir is None and HOME_MAP:
        home_dir = HOME_DIR
    home_root = Path(home_dir) if home_dir is not None else None
    allow = parse_allow(ALLOW) if allow is None else allow
    reconstruct = VggtReconstruct() if reconstruct is None else reconstruct
    # GPU は 1 つずつ使う。同時に来たリクエストはここで順に待つ
    gpu = threading.Lock()
    refused: set[str] = set()
    # 受けている数と最後に返した時刻。make deploy-map が、Pi が走っている間（返した直後）を
    # 待って起動し直し、探索を止めずにコードを入れ替えるのに使う（GET /v1/snoo/map/busy）
    busy = {"in_flight": 0, "last_reply": 0.0, "last": None}
    busy_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # VGGT は最初のステーションで読む（7 秒ほど。Pi は 25 秒待つ）。起動したときに読むと、
        # Pi が探っていない間も GPU を握り、検出と合わせて GPU が休めない
        # 古いセッションの重いファイル（recon.npz など）を消す
        # （scripts/snoo_prune.py。別のスレッドで）
        threading.Thread(target=_prune_old, args=(root,), daemon=True, name="prune").start()
        yield

    app = FastAPI(title="snoo map", lifespan=lifespan)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.client.host if request.client else ""
        try:
            ok = _allowed((host, 0), allow)
        except ValueError:  # IP でない（テストの "testclient" など）
            ok = False
        if not ok:
            if host not in refused:  # 同じ送り元は 1 回だけ書く
                refused.add(host)
                log.warning("%s は受ける送り元に無いので断る", host)
            if "text/html" in request.headers.get("accept", ""):
                # ブラウザ（スマホで /drive を開いたときなど）には、足し方を日本語で見せる
                return HTMLResponse(refused_page(host), 403)
            return JSONResponse({"detail": "この送り元からは受けない"}, 403)
        if request.method == "POST":
            # 読み始める前に大きさで断る（multipart はいったん全部読まれてしまう）
            n = request.headers.get("content-length")
            if n is None or not n.isdigit():
                return JSONResponse({"detail": "Content-Length が要る"}, 411)
            if int(n) > MAX_REQUEST_BYTES:
                mb = MAX_REQUEST_BYTES // 1024 // 1024
                return JSONResponse({"detail": f"{mb} MB を超えるので受けない"}, 413)
        return await call_next(request)

    @app.post("/v1/snoo/map/station")
    def station(
        meta: Annotated[str | None, Form()] = None,
        images: Annotated[list[UploadFile] | None, File()] = None,
    ):
        t0 = time.perf_counter()
        enter = getattr(reconstruct, "enter", None)
        if enter is not None and not enter():
            raise HTTPException(503, "GPU を手放して終えるところ（すぐ起動し直す）")
        with busy_lock:
            busy["in_flight"] += 1
        try:
            action = handle(meta, images, t0)
            with busy_lock:
                busy["last"] = action.get("action")
            return action
        finally:
            with busy_lock:
                busy["in_flight"] -= 1
                busy["last_reply"] = time.time()
            leave = getattr(reconstruct, "leave", None)
            if leave is not None:
                leave()

    @app.get("/v1/snoo/map/busy")
    def busy_state():
        """受けている数・最後に返してからの秒・最後の返事（move なら Pi は今走って撮っている）。"""
        with busy_lock:
            age = time.time() - busy["last_reply"] if busy["last_reply"] else None
            return {"in_flight": busy["in_flight"], "since_reply_s": age, "last": busy["last"]}

    def handle(meta, images, t0: float) -> dict:
        try:
            m = parse_meta(meta)
            files = read_images(images, m)
        except BadRequest as e:
            raise HTTPException(400, str(e)) from e
        session = root / m["session_id"]
        sid = m["station_id"]
        t = time.perf_counter()
        # 保存も GPU のロックの中でする。Pi が同じ station_id を送り直したとき、前のリクエストが
        # 読んでいる置き場を消して 500 にしていた（コードレビュー）
        with gpu:
            marks = {"wait": time.perf_counter() - t}
            save_station(session, sid, m, files)
            marks["save"] = time.perf_counter() - t0 - marks["wait"]

            battery = m.get("battery_v")
            # 0 V 以下は読めなかった（2 回目の実走の最後で 0.0 が来た）ので、電池切れにしない
            if battery is not None and 0 < battery < LOW_BATTERY_V:
                log.info("%s/%03d: 電池が %.2f V なのでやめる", session.name, sid, battery)
                return {"action": "done", "reason": "low_battery"}
            if len(station_ids(session)) >= SESSION_MAX_STATIONS:
                log.info(
                    "%s/%03d: %d か所を超えたのでやめる", session.name, sid, SESSION_MAX_STATIONS
                )
                return {"action": "done", "reason": "step_limit"}
            try:
                action = plan_station(session, sid, reconstruct, marks, home_root)
            except Exception as e:
                log.exception("%s/%03d を処理できなかった", session.name, sid)
                raise HTTPException(500, f"間取りを作れなかった: {e}") from e
        marks["total"] = time.perf_counter() - t0
        log.info(
            "%s/%03d: %s｜秒: %s",
            session.name,
            sid,
            json.dumps(action, ensure_ascii=False),
            " ".join(f"{k} {v:.2f}" for k, v in marks.items()),
        )
        return action

    @app.get("/v1/snoo/map/{session_id}/{name}")
    def output(session_id: str, name: str):
        if name not in OUTPUTS:
            raise HTTPException(404, f"{name} は無い（{', '.join(OUTPUTS)} のどれか）")
        try:
            check_id(session_id)
        except BadRequest as e:
            raise HTTPException(400, str(e)) from e
        path = root / session_id / name
        if not path.is_file():
            raise HTTPException(404, "まだ作っていない")
        return FileResponse(path, media_type=OUTPUTS[name], headers={"Cache-Control": "no-store"})

    if home_root is not None:
        add_home(app, home_root)
    add_twin(app, root, Path(live_dir) if live_dir else None)
    add_agent(app, root, Path(live_dir) if live_dir else None)
    add_teleop_api(app)
    return app


def refused_page(host: str) -> str:
    """送り元の制限で断ったときにブラウザへ見せる、足し方の説明（HTML）。"""
    import html

    h = html.escape(host or "?")
    return f"""<!DOCTYPE html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>snoo：この機械は入れない</title>
<body style="font:15px/1.6 system-ui,sans-serif;background:#0e1114;color:#d5dbe2;
padding:16px;max-width:40em">
<h1 style="font-size:18px">この機械（{h}）からは開けない</h1>
<p>地図のサーバーには認証が無く、部屋の画像が見えて車も動かせるので、決めた送り元（SNOO_MAP_ALLOW）
だけを受けています。この機械で開くには、サーバーのマシンで次のようにして IP を足してください。</p>
<ol>
<li><code>~/.config/snoo-map.env</code> の <code>SNOO_MAP_ALLOW=</code> の行の終わりに
<code>,{h}</code> を足す（無ければ
<code>SNOO_MAP_ALLOW=192.168.0.188,100.73.14.8,127.0.0.1,{h}</code> と書く）</li>
<li><code>make deploy-map</code>（地図のサービスを起動し直す）</li>
</ol>
<p>スマホは家の Wi-Fi か tailscale の IP（100.x.y.z）で開くと IP が変わりにくい。</p>
</body></html>"""


def add_teleop_api(app: FastAPI) -> None:
    """手で動かす API と /drive の画面（app/snoo_teleop.py）を足す。読み込めなくても
    地図のサーバーは動かす。"""
    try:
        from app.snoo_teleop import add_teleop

        add_teleop(app)
    except Exception:
        log.exception("teleop の API を足せなかった（地図のサーバーはそのまま動かす）")


HOME_OUTPUTS = {"home.png": "image/png", "home.json": "application/json"}


def add_home(app: FastAPI, home_root: Path) -> None:
    """GET /v1/snoo/map/home.png・home.json：家の地図と、セッションごとの変換・島の一覧。"""

    def serve(name: str):
        path = home_root / name
        if not path.is_file():
            raise HTTPException(404, "家の地図はまだ無い")
        return FileResponse(path, media_type=HOME_OUTPUTS[name],
                            headers={"Cache-Control": "no-store"})  # fmt: skip

    @app.get("/v1/snoo/map/home.png")
    def home_png():
        return serve("home.png")

    @app.get("/v1/snoo/map/home.json")
    def home_json():
        return serve("home.json")


def add_agent(app: FastAPI, root: Path, live_dir: Path | None) -> None:
    """エージェントの API（app/snoo_agent_api.py、/v1/snoo/agent/）を足す。読み込めない・作れない
    ときはログに書くだけにして、Pi のステーションを受ける地図のサーバーはそのまま動かす。"""
    try:
        from app.snoo_agent_api import create_router

        app.include_router(create_router(root, live_dir))
    except Exception:
        log.exception("エージェントの API を足せなかった（地図のサーバーはそのまま動かす）")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # GPU の実行時（HSA）は torch を初めて使うときに環境変数を読むので、VGGT を読む前に決める
    if CU_MASK:
        os.environ.setdefault("HSA_CU_MASK", CU_MASK)
    log.info(
        "%s:%d で待ち受け（受ける送り元: %s、置き場: %s、VGGT の CU: %s）",
        HOST,
        PORT,
        ALLOW or "すべて",
        DATA,
        os.environ.get("HSA_CU_MASK") or "すべて",
    )
    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
