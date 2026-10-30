"""PiCar-X（85pi4 の snoo_trucar）が止まって首を振って撮った画像（1 ステーション）を HTTP で受け、
間取りを作り直して次の 1 歩を返す（間取りは app/snoo_mapper.py で作る）。

    cd ~/orca/rocm_opencv_server && .venv/bin/python -m app.snoo_map_server
VGGT は main の .venv にだけ入れてある（git から入れた。uv.lock にも Docker にも無い）ので、
コンテナではなくホストのプロセスとして動かす。

- POST /v1/snoo/map/station（multipart）：meta（scan.json に session_id・station_id などを足した
  JSON）と images（meta の frames[].file と同じ名前の画像）。返事は次の 1 歩（action=move）か、
  やめる理由（action=done）。Pi は 20 秒待って返事が無ければ自分で探るので、1 ステーションを
  20 秒以内に返す
  - meta の ultrasonic_cm（と止まって撮ったコマごとの読み）は、そのステーションの前の超音波として
    格子に足す
  - meta の supports（Pi ができること）に "segments" があれば、move に
    segments: [{steer_deg, distance_cm}, ...]（曲がる → まっすぐ、合わせて 40 cm まで）を付ける。
    上の steer_deg・distance_cm は最初の区間
  - supports に "reverse" があれば、前へ行ける所が無い（unreachable）とき、来た道を戻る
    {"action": "move", "reverse": true, steer_deg, speed, distance_cm（30 cm まで）, [segments]}
    を返す（続けて 2 回まで）。見たい所が後ろ（100° より後ろ）のときも、前へ輪を描く代わりに
    切り返し（いっぱいに切って下がる）を返すことがある。そのときは "turn": true が付く
    （形は reverse の返事と同じ）
  - supports に "long_step"（と "segments"）があれば、まっすぐの区間が地図で床と分かっている
    ときは、1 歩を 60 cm まで延ばす（40 cm を超えるのは、最後のまっすぐの区間だけ）。
    Pi は今までどおり超音波・clear_cm・動かないときに止まる
  - supports が無い Pi への返事は今までと同じ
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
from fastapi.responses import FileResponse, JSONResponse

from app import snoo_mapper
from app.roommap import (
    FIT_TOL_M,
    LONG_STEP_M,
    MAX_STEP_M,
    TURN_RADIUS_M,
    WHEELBASE_M,
    Pose,
    blocked_cells,
    camera_of,
    clip_pieces,
    consume,
    drive_poses,
    fit_driven,
    mark_driven,
    moves_are_clear,
    random_step,
    take,
    turning_reverse,
)
from app.snoo import _allowed, parse_allow
from app.snoo_mapper import (
    CLUSTER_STATIONS,
    MAX_FRAMES,
    MAX_RECENT_REVERSES,  # noqa: F401  REVERSE_SPEED などはここからも使う
    MAX_REVERSES,  # noqa: F401
    RECENT_STATIONS,
    REVERSE_SPEED,  # noqa: F401
    FrameRef,
    History,
    MapParams,
    Observation,
    Station,
    build_map,
    choose_window,
    decide,
    frame_colors,
    load_station,
    load_vggt_model,
    mark_slam,
    mask_cats,
    or_reverse,
    pose_dict,
    random_action,
    registration_is_bad,
    retrace_step,
    reverse_action,
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


def read_observations(session: Path, upto: int) -> list[Observation]:
    """upto までのステーションで見たもの（置き場の obs.npz）。無いステーションは飛ばす。"""
    out = []
    for i in station_ids(session):
        if i <= upto and obs_path(session, i).is_file():
            out.append(Observation.load(obs_path(session, i)))
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


def clustered(poses: dict[int, Pose], station_id: int) -> bool:
    """station_id の直前 CLUSTER_STATIONS か所が狭い所に固まっているか（snoo_mapper.clustered）。"""
    recent = [poses[i] for i in range(station_id - CLUSTER_STATIONS, station_id) if i in poses]
    return snoo_mapper.clustered(recent)


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
        force_random=clustered(poses, station_id),
        reverse="reverse" in can,
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
    for i, s, h, o in zip(window, r.stations, params.camera_height_m, r.observations, strict=True):
        # 一度求めた位置は変えない（変えると、それを原点にした後の窓と座標がずれる）
        if i not in poses and s.frames and not registration_is_bad(s, h):
            poses[i] = to_session(anchor, s.pose)
        # 見たものも 1 回だけ残す（ふつうは最後のステーション。前の版で受けたステーションも）
        if i in poses and o is not None and not obs_path(session, i).is_file():
            o.save(obs_path(session, i))
    write_poses(session, poses)

    # 窓の結果をセッションの座標に直し、セッションの間取りで次の 1 歩を決める
    r.stations = [dataclasses.replace(s, pose=to_session(anchor, s.pose)) for s in r.stations]
    r.points = xy_to_frame(anchor, r.points) if len(r.points) else r.points
    if r.floor.inliers:
        r.grid, r.high = session_grid(read_observations(session, station_id))
        if params.driven_free:
            mark_steps_driven(r.grid, session, station_id, poses)
        if USE_SLAM:  # 走りながらの SLAM で見た所も足す（app/snoo_slam.py が書く）
            r.slam = mark_slam(r.grid, r.high, session / "slam", params)
        if home_root is not None:
            t_home = time.perf_counter()
            r.grid, r.home = use_home(home_root, session, station_id, r.grid, r.high)
            marks["home"] = time.perf_counter() - t_home
        r.action, r.plan = decide(r.grid, r.stations, params, params.camera_height_m)
        r.step_path = []
        if r.plan is not None:
            pieces = r.plan.segments or [(r.plan.steer_deg, r.plan.step_m)]
            r.step_path = [camera_of(p) for _, p in drive_poses(r.stations[-1].pose, pieces)]
    marks["map"] = time.perf_counter() - t

    t = time.perf_counter()
    raw = r.action
    action = dict(raw)
    pose = r.stations[-1].pose
    stalls = stalled_steps(session, station_id)
    if stalls >= STALL_DONE:
        # 前へ出せない（Pi が 1 cm も走らなかった）が続いたら、待たせ続けずに終える
        action = {"action": "done", "reason": "stalled"}
    planned = action
    if (
        stalls
        and stalls < STALL_DONE
        and action.get("action") == "move"
        and not action.get("reverse")
    ):
        # 前の前へ行く 1 歩を Pi が 1 cm も走らなかった（超音波・clear_cm で止まった）。同じ所から
        # 前へ出し続けても走らないので、まず下がる 1 歩にする（下がれなければ、下で元の 1 歩に戻す）
        action = {"action": "done", "reason": "unreachable"}
    back = (
        retrace_after_bad(session, station_id, raw, r.stations[-1].name) if params.reverse else None
    )
    if back:
        # このステーションの位置合わせが崩れた。探索を終えずに、いま走った 1 歩を同じ道で戻り、
        # 位置の分かっている前のステーションの所で撮り直す
        action = reverse_action(back, params.segments)
        action["retrace"] = True
    override = take_override(root_of(session), session.name, r.grid, pose, params)
    if override is not None:
        # エージェント（画像を見て決める AI）が予約した 1 歩。地図で確かめて通ったものだけ
        action = override
    action, pieces = or_reverse(action, r.grid, pose, history(session, station_id, poses), params)
    if action is not planned and action == {"action": "done", "reason": "unreachable"}:
        # 下がれなかった。1〜2 回止まっただけで終えると早すぎる（18:55 の探索は 11 か所目で
        # 終えた）。決めた 1 歩をそのまま試し、続けて止まれば STALL_DONE で終える
        action = planned
    if action["action"] == "done" and KEEP_EXPLORING and action.get("reason") not in FINISHED:
        # 止まらずに探り続けて、地図を全部埋める（ユーザーの目標）。終えるのは調べ終えたときだけ
        alt = keep_going(r.grid, pose, params, station_id)
        if alt is not None:
            alt["why"] = str(action.get("reason", "")).split(":")[0]
            action = alt
    if pieces:  # 戻る弧を上から見た絵に描き、戻った先を target にする
        r.step_path = [camera_of(p) for _, p in drive_poses(pose, pieces, reverse=True)]
        action["target"] = pose_dict(r.step_path[-1])
        remember_reverse(session, station_id, action)
        r.action = dict(action)  # room.json の next と絵の見出しは、Pi に返すほうにする
    if action.get("reverse") and station_id not in read_reverses(session):
        # 向きを変える 1 歩（turn）も後ろへ走る。prev_step に reverse を書かない Pi のために残す
        # （残さないと、後ろへの turn を前へ走ったものとして来た道と床を逆側にしていた）
        remember_reverse(session, station_id, action)
    if action["action"] == "move":
        here = poses.get(station_id)
        if here is None:  # 位置合わせが崩れていれば decide が done にするので、ふつうは来ない
            here = r.stations[-1].pose
        action["pose"] = pose_dict(here)
        if not action.get("reverse") and r.plan is not None:  # でたらめな 1 歩には目標が無い
            action["target"] = pose_dict(r.plan.target)
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
    （前へは clip_pieces と moves_are_clear、後ろへは moves_are_clear）ものは使わず、理由を
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
        piece = [(float(steer), float(cm) / 100)]
        if not back:
            blocked = blocked_cells(grid, grid.state(), pose)
            run = sum(m for _, m in clip_pieces(grid, blocked, pose, piece))
            if run < piece[0][1] - 0.005:
                why = f"前に {run * 100:.0f} cm しか空いていない"
        if why is None and not moves_are_clear(grid, grid.state(), pose, [(*piece[0], back)]):
            why = "車の真ん中が障害物・見ていない所に入る"
    log_path = path.parent / "override_result.jsonl"
    row = {"t": time.time(), "session": session, "override": o, "used": why is None, "why": why}
    with log_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if why is not None:
        log.info("エージェントの 1 歩を使わない（%s）: %s", why, o)
        return None
    if back:
        action = reverse_action(piece, params.segments)
    else:
        action = random_action(piece[0], pose, params)
        action.pop("random", None)
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


# --- 止まらずに探り続ける ---

# 終えてよい理由（調べ終えた）。ほかの理由（行けない・止まったまま・位置合わせ・上限）では終えず、
# 切り返し・でたらめな 1 歩で場所と向きを変えて撮り直す。SNOO_KEEP_EXPLORING=0 で前の決まりに戻す
KEEP_EXPLORING = os.environ.get("SNOO_KEEP_EXPLORING", "1") != "0"
FINISHED = ("no_frontier",)
# 地図のサーバーの 1 セッションのステーションの上限（MapParams の既定 50 では 10 分ほどで終える）
SESSION_MAX_STATIONS = int(os.environ.get("SNOO_MAX_STATIONS", "2000"))
# 地図で確かめられないときの、見ないで走る短い切り返し（m）。後ろは見えないので短く
BLIND_TURN_M = 0.10
MAX_STEER_DEG = round(math.degrees(math.atan(WHEELBASE_M / TURN_RADIUS_M)), 1)


def keep_going(grid, pose: Pose, params: MapParams, station_id: int) -> dict | None:
    """終える代わりの 1 歩。でたらめな前への 1 歩 → 切り返し → 見ないで短く切り返す、の順。

    どれも Pi の guard・clear_cm・超音波（近い値が続いたとき）で止まる。ハンドルの向きは
    ステーションごとに左右を替え、同じ所で同じ向きに回り続けないようにする。
    """
    rng = np.random.default_rng(station_id)
    if grid is not None:
        step = random_step(grid, pose, rng)
        if step is not None:
            action = random_action(step, pose, params)
            action["keep_going"] = True
            return action
        if params.reverse:
            for length in (0.30, 0.15):
                pieces = turning_reverse(grid, pose, length=length)
                if pieces:
                    action = reverse_action(pieces, params.segments)
                    action["keep_going"] = True
                    return action
    if not params.reverse:
        return None
    side = 1.0 if station_id % 2 == 0 else -1.0
    action = reverse_action([(side * MAX_STEER_DEG, BLIND_TURN_M)], params.segments)
    action["keep_going"] = True
    action["blind"] = True
    return action


# --- 位置合わせが崩れたら、来た道を戻って撮り直す ---

# これより短く走っただけなら、戻らずに終える（同じ所で撮り直しても同じになりやすい）
RETRACE_MIN_M = 0.05


def retrace_after_bad(
    session: Path, station_id: int, raw: dict, last_name: str
) -> list[tuple[float, float]] | None:
    """最後のステーションの位置合わせが崩れたとき、いま走った 1 歩を戻る区間（後ろへ走る順）。

    18:49 の探索では 4 か所目で VGGT がコマの位置を 1〜16 cm に散らし、1 回の bad_registration で
    探索を終えていた。同じハンドルで後ろへ走れば同じ弧を戻るので、走ったばかりの床（空いている）
    を通って前のステーションへ戻れる。続けて崩れたら（前の返事も戻る 1 歩なら）終える。
    """
    if raw.get("reason") != f"bad_registration:{last_name}":
        return None
    prev = read_meta(station_dir(session, station_id)).get("prev_step")
    if not isinstance(prev, dict) or prev.get("reverse"):
        return None
    before = station_dir(session, station_id - 1) / "reply.json"
    if before.is_file() and json.loads(before.read_text()).get("retrace"):
        return None
    pieces, moved = step_pieces(prev), pi_distance_m(prev)
    if not pieces or moved is None or moved < RETRACE_MIN_M:
        return None
    if sum(m for _, m in pieces) <= 0:  # 長さの無い 1 本の弧（古い Pi）は、走った長さにする
        pieces = [(pieces[0][0], moved)]
    out, left = [], moved
    for steer, m in pieces:  # 走った分だけ（止まった所まで）
        take = min(m, left)
        if take > 1e-3:
            out.append((steer, take))
        left -= take
    return list(reversed(out)) or None


# --- 前へ出せないのを見る ---

# 前へ行く 1 歩を 1 cm も走らなかったら「出せなかった」とする
STALL_CM = 1.0
# 直近 STALL_WINDOW か所で、出せなかったのが STALL_DONE 回になったら終える
STALL_WINDOW = 6
STALL_DONE = 3


def _stalled(prev: dict | None) -> bool:
    """前へ行く 1 歩を、Pi が 1 cm も走らずに止めたか。"""
    if not isinstance(prev, dict) or prev.get("reverse"):
        return False
    if not prev.get("stop_reason"):
        return False
    try:
        return float(prev.get("est_distance_cm") or 0.0) < STALL_CM
    except (TypeError, ValueError):
        return False


def stalled_steps(session: Path, station_id: int) -> int:
    """直近の前へ出せなかった数。最後（このステーションの prev_step）が出せていれば 0。

    超音波が当てにならない（-1 が続く）と、Pi は前へ 1 cm も走らずに止まる。17:57 の探索では
    20 か所のうち 16 か所がこれで、同じ所からでたらめな 1 歩を出し続けていた。
    """
    ids = [i for i in station_ids(session) if i <= station_id][-STALL_WINDOW:]
    flags = [_stalled(read_meta(station_dir(session, i)).get("prev_step")) for i in ids]
    if not flags or not flags[-1]:
        return 0
    return sum(flags)


# --- 行き場が無いときに来た道を戻る ---


def step_pieces(prev: dict) -> list[tuple[float, float]] | None:
    """prev_step の区間 [(ハンドル°, m), ...]。segments（返した形のまま）があればそれ、
    無ければ steer_deg の 1 本の弧（長さは分からないので 0 にし、測った分だけ伸ばす）。
    """
    segs = prev.get("segments")
    if isinstance(segs, list) and segs:
        out = []
        for x in segs:
            st = x.get("steer_deg") if isinstance(x, dict) else None
            cm = x.get("distance_cm") if isinstance(x, dict) else None
            if not _finite(st) or not _finite(cm) or cm < 0:
                return None
            out.append((float(st), float(cm) / 100))
        return out
    st = prev.get("steer_deg")
    return [(float(st), 0.0)] if _finite(st) else None


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


def moved_fit(
    prev: dict, pieces, a: Pose | None, b: Pose | None, reverse: bool
) -> tuple[float | None, bool]:
    """(moved_m の長さ, VGGT の位置で確かめられたか)。"""
    if a is not None and b is not None:
        s, err = fit_driven(a, b, pieces, reverse)
        if err <= FIT_TOL_M:
            return s, True
    return pi_distance_m(prev), False


def moved_m(prev: dict, pieces, a: Pose | None, b: Pose | None, reverse: bool) -> float | None:
    """ステーション a から b まで、区間に沿って実際に走った長さ（m）。

    VGGT の 2 つの位置に、区間を自転車モデルで走らせた位置を合わせて測る（Pi の見積もりより
    確か。1 回目の実走では Pi の見積もりは本当の 0.6 倍だった）。合わなければ（位置が無い・
    ずれが大きい）Pi の見積もりを使う。
    """
    if a is not None and b is not None:
        s, err = fit_driven(a, b, pieces, reverse)
        if err <= FIT_TOL_M:
            return s
    return pi_distance_m(prev)


def driven_trail(
    session: Path, station_id: int, poses: dict[int, Pose]
) -> tuple[list[tuple[float, float]], int]:
    """station_id までに前へ走った道（走った順の区間）と、続けて後ろへ戻った回数。

    ステーションの meta の prev_step を順に見る。前へ走ったら区間を足し、後ろへ戻ったら
    その分を終わりから除く。どう動いたか分からない所（prev_step が無い・ステーションが
    飛んでいる・長さが測れない）があれば、そこより前へは戻らない。
    prev_step に reverse が無ければ（古い Pi）、そのとき後ろへ戻れと返したかで決める。
    """
    trail: list[tuple[float, float]] = []
    backs = 0
    for _, pieces, reverse in steps_driven(session, station_id, poses):
        if pieces is None:
            trail, backs = [], 0
        elif reverse:
            trail, backs = consume(trail, sum(m for _, m in pieces)), backs + 1
        else:
            trail, backs = trail + pieces, 0
    return trail, backs


def steps_driven(
    session: Path, station_id: int, poses: dict[int, Pose], fitted_only: bool = False
) -> list[tuple[int, list[tuple[float, float]] | None, bool]]:
    """station_id までの、ステーションの間に走った 1 歩ごとの (前のステーション, 区間, 後ろへか)。
    区間は実際に走った長さまで（moved_m）。どう動いたか分からない所は区間が None。
    fitted_only なら、VGGT の位置で確かめられなかった歩も None にする（床にするとき。位置が
    8 cm を超えてずれていると、車の外形分が別の所で床になる。コードレビュー）。
    prev_step に reverse が無ければ（古い Pi）、そのとき後ろへ戻れと返したかで決める。
    """
    sent = read_reverses(session)
    out: list[tuple[int, list[tuple[float, float]] | None, bool]] = []
    before = None
    for i in station_ids(session):
        if i > station_id:
            break
        prev = read_meta(station_dir(session, i)).get("prev_step")
        pieces = step_pieces(prev) if isinstance(prev, dict) else None
        if before is None or i != before + 1 or pieces is None:
            out.append((i, None, False))
        else:
            reverse = prev.get("reverse")
            reverse = before in sent if reverse is None else bool(reverse)
            s, fitted = moved_fit(prev, pieces, poses.get(before), poses.get(i), reverse)
            if s is None or (fitted_only and not fitted):
                out.append((before, None, reverse))
            else:
                out.append((before, take(pieces, s), reverse))
        before = i
    return out


def mark_steps_driven(grid, session: Path, station_id: int, poses: dict[int, Pose]) -> None:
    """走ってきた道（車の外形の下）を、セッションの間取りの床にする（roommap.mark_driven）。"""
    for before, pieces, reverse in steps_driven(session, station_id, poses, fitted_only=True):
        if pieces and before in poses:
            mark_driven(grid, poses[before], pieces, reverse)


def history(session: Path, station_id: int, poses: dict[int, Pose]) -> History:
    """station_id までの走り方（来た道・続けて戻った回数・直近で戻った回数）。"""
    trail, backs = driven_trail(session, station_id, poses)
    return History(trail, backs, recent_reverses(session, station_id))


def backtrack(
    session: Path, station_id: int, poses: dict[int, Pose], segments: bool
) -> list[tuple[float, float]]:
    """前へ行ける所が無いときに、来た道を戻る区間（戻る順）。戻れなければ空
    （app.snoo_mapper.retrace_step）。
    """
    return retrace_step(history(session, station_id, poses), segments)


def recent_reverses(session: Path, station_id: int) -> int:
    """直近 RECENT_STATIONS か所（station_id まで）のうち、後ろへ戻って着いたステーションの数。"""
    sent = read_reverses(session)
    count = 0
    for i in station_ids(session):
        if station_id - RECENT_STATIONS < i <= station_id:
            prev = read_meta(station_dir(session, i)).get("prev_step")
            if isinstance(prev, dict):
                reverse = prev.get("reverse")
                count += bool(reverse) if reverse is not None else (i - 1) in sent
    return count


def read_reverses(session: Path) -> dict[int, dict]:
    """後ろへ戻れと返したステーションと、その返事（reverse.json）。"""
    p = session / "reverse.json"
    if not p.exists():
        return {}
    return {int(k): v for k, v in json.loads(p.read_text()).items()}


def remember_reverse(session: Path, station_id: int, action: dict) -> None:
    data = read_reverses(session)
    data[station_id] = {k: v for k, v in action.items() if k != "map_url"}
    tmp = session / ".reverse.json"
    tmp.write_text(json.dumps({str(k): v for k, v in sorted(data.items())}, indent=1))
    tmp.replace(session / "reverse.json")


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
    return app


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
