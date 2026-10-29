"""PiCar-X（85pi4 の snoo_trucar）が止まって首を振って撮った画像（1 ステーション）を HTTP で受け、
間取りを作り直して次の 1 歩を返す（間取りは app/snoo_mapper.py で作る）。

    cd ~/orca/rocm_opencv_server && .venv/bin/python -m app.snoo_map_server
VGGT は main の .venv にだけ入れてある（git から入れた。uv.lock にも Docker にも無い）ので、
コンテナではなくホストのプロセスとして動かす。

- POST /v1/snoo/map/station（multipart）：meta（scan.json に session_id・station_id などを足した
  JSON）と images（meta の frames[].file と同じ名前の画像）。返事は次の 1 歩（action=move）か、
  やめる理由（action=done）。Pi は 20 秒待って返事が無ければ自分で探るので、1 ステーションを
  20 秒以内に返す
- GET /v1/snoo/map/<session_id>/topdown.png・room.json・plan.json：いちばん新しい結果（人が見る用）

受けたステーションは <SNOO_MAP_DATA>/<session_id>/<station_id:03d>/ に画像と meta.json で残し、
結果（vggt.npz・room.json・topdown.png・plan.json など）はセッションの置き場に書く。

VGGT は 48 コマで 9 秒ほどかかる（RX 7900 XTX）。実データ（23 枚のステーション）で、受けてから
返すまで 1 か所目 5.7 秒、窓が 48 コマのとき 10〜12 秒（VGGT 7.5〜9.5 秒・間取り 2 秒）だった。
ステーションが増えても時間と GPU のメモリが増えないよう、VGGT に入れるのは最後のステーションと、
その近くのステーション（合わせて WINDOW 個）だけにする。そのときの間取りの原点は窓のいちばん
古いステーションになるので、前に求めたそのステーションの位置を足して、返す位置をセッションの
座標（最初のステーションが原点）に直す。
窓の外のステーションは間取りに入らない（room.json・topdown.png は窓の中だけ）。

認証が無いので、SNOO_MAP_ALLOW に書いた送り元（Pi の LAN と tailscale の IP）以外には 403 を返す
（app/snoo.py の検出サーバーと同じやり方）。GPU は猫の検出（Docker の snoo）と分け合うので、
VGGT は 1 つずつ回す（同時に来たものは順に待たせる）。
"""

import json
import logging
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.roommap import Pose
from app.snoo import _allowed, parse_allow
from app.snoo_mapper import (
    MAX_FRAMES,
    MAX_STATIONS,
    FrameRef,
    MapParams,
    Station,
    build_map,
    frame_colors,
    load_station,
    load_vggt_model,
    pose_dict,
    registration_is_bad,
    run_vggt,
    save_vggt,
    select_frames,
    unproject,
    write_outputs,
)

HOST = os.getenv("SNOO_MAP_HOST", "0.0.0.0")
PORT = int(os.getenv("SNOO_MAP_PORT", "8720"))
# 受ける送り元（IP か CIDR をカンマで区切る）。Pi の LAN・Pi の tailscale・このマシン
ALLOW = os.getenv("SNOO_MAP_ALLOW", "192.168.0.188,100.73.14.8,127.0.0.1")
DATA = os.getenv("SNOO_MAP_DATA", "data/snoo_map/sessions")
# 1 回の POST の上限。Pi の画像は 640x480 の JPEG で 1 枚 30 KB ほど、1 ステーション 24 枚ほど
MAX_REQUEST_BYTES = 40 * 1024 * 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGES = 64
# これより電池の電圧が低ければ探るのをやめる（2 セルのリチウムイオンで、走ると電圧が落ちる）
LOW_BATTERY_V = 6.8
# 1 回に VGGT に入れるステーションの数。48 コマを 6 か所で分けると 1 か所 8 コマ（下向きと
# 正面の 2 段 × パン 4 つ）。これより多いと 1 か所のコマが少なすぎて向きと位置がぶれる
WINDOW = 6
# 床からレンズの中心まで。2026-10-29 に SD カードの長い辺（32 mm）2.5 枚分と測った
DEFAULT_CAMERA_HEIGHT_CM = 8.0
# 超音波がこれより短く読んだら測れなかったとする（PiCar-X の超音波は 2 cm から）
MIN_ULTRASONIC_CM = 2.0

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


def supports(d: Path) -> set[str]:
    """Pi ができると書いてきたこと（"reverse"・"segments"）。書いていない Pi は今までどおり。"""
    return set(json.loads((d / "meta.json").read_text()).get("supports") or [])


def camera_height_m(d: Path) -> float:
    h = json.loads((d / "meta.json").read_text()).get("camera_height_cm")
    return (h or DEFAULT_CAMERA_HEIGHT_CM) / 100


def ultrasonic_cm(d: Path) -> float | None:
    """そのステーションの前の超音波の読み（cm）。meta の ultrasonic_cm と、止まって撮ったコマごとの
    読みの中央値にする（1 回の読みはときどき跳ねる。実走で 34 cm の中に 61 cm が混じった）。
    MIN_ULTRASONIC_CM より短い・負の値は、測れなかった（Pi が -1・-2 を返すことがある）とする。
    """
    meta = json.loads((d / "meta.json").read_text())
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
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    yaw = anchor.yaw + p.yaw
    return Pose(
        anchor.x + c * p.x - s * p.y,
        anchor.y + s * p.x + c * p.y,
        math.atan2(math.sin(yaw), math.cos(yaw)),
    )


def choose_window(
    ids: list[int], latest: int, poses: dict[int, Pose], k: int = WINDOW
) -> list[int]:
    """VGGT に入れるステーション。最後のステーションと、位置が分かっている前のステーションのうち、
    1 つ前の位置（今の車の近く）に近い k - 1 個。原点にする窓のいちばん古いステーションの位置が
    分かっていないと座標を直せないので、位置の分からないステーションは入れない。
    """
    known = [i for i in ids if i < latest and i in poses]
    if not known:
        return [latest]
    here = poses[max(known)]
    near = sorted(known, key=lambda i: (math.hypot(poses[i].x - here.x, poses[i].y - here.y), -i))
    return sorted(near[: k - 1]) + [latest]


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


class VggtReconstruct:
    """VGGT を GPU に載せたまま持つ（ステーションごとに読み直すと 5 秒以上かかる）。
    GPU を 5 GB 使い続けるが、検出（Docker の snoo）は 2 GB ほどなので、24 GB に収まる。
    """

    def __init__(self):
        self.model = None

    def load(self) -> None:
        t = time.perf_counter()
        self.model = load_vggt_model()
        log.info("VGGT を読んだ（%.1f 秒）", time.perf_counter() - t)

    def __call__(self, refs: list[FrameRef], npz: Path, marks: dict) -> Recon:
        import torch

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
        return Recon(v.extrinsic, points, conf, colors)


# --- 1 ステーションを処理する ---


def plan_station(session: Path, station_id: int, reconstruct: Reconstruct, marks: dict) -> dict:
    """セッションのステーションから間取りを作り直し、次の 1 歩を返す（Pi に返す形）。"""
    ids = station_ids(session)
    poses = read_poses(session)
    window = choose_window(ids, station_id, poses)
    dirs = [station_dir(session, i) for i in window]
    stations = [load_still(d) for d in dirs]
    can = supports(dirs[-1])
    params = MapParams(
        camera_height_m=[camera_height_m(d) for d in dirs],
        ultrasonic_cm=[ultrasonic_cm(d) for d in dirs],
        segments="segments" in can,
    )
    refs = select_frames(stations, MAX_FRAMES)
    rec = reconstruct(refs, session / "vggt.npz", marks)

    t = time.perf_counter()
    r = build_map(stations, refs, rec.extrinsic, rec.points, rec.conf, params, rec.colors)
    marks["map"] = time.perf_counter() - t

    t = time.perf_counter()
    # 窓の原点（いちばん古いステーション）の位置。最初のステーションなら (0, 0, 0)
    anchor = poses.get(window[0], Pose(0.0, 0.0, 0.0))
    for i, s, h in zip(window, r.stations, params.camera_height_m, strict=True):
        # 一度求めた位置は変えない（変えると、それを原点にした後の窓と座標がずれる）
        if i not in poses and s.frames and not registration_is_bad(s, h):
            poses[i] = to_session(anchor, s.pose)
    write_poses(session, poses)

    action = dict(r.action)
    if action["action"] == "move":
        here = poses.get(station_id)
        if here is None:  # 位置合わせが崩れていれば decide が done にするので、ふつうは来ない
            here = to_session(anchor, r.stations[-1].pose)
        action["pose"] = pose_dict(here)
        action["target"] = pose_dict(to_session(anchor, r.plan.target))
        action["station_index"] = ids.index(station_id)
        action["map_url"] = f"/v1/snoo/map/{session.name}/topdown.png"
    else:
        # bad_registration:<名前> の名前は Pi には要らない（plan.json には残す）
        action = {"action": "done", "reason": action["reason"].split(":")[0]}
    write_results(session, r, window, anchor, r.action, marks)
    marks["write"] = time.perf_counter() - t
    return action


def write_results(session: Path, r, window: list[int], anchor: Pose, raw: dict, marks: dict):
    """room.json などを書く。ブラウザで見ている途中に半端なファイルを読ませないよう、
    別の置き場に書いてから 1 つずつ名前を変える。
    """
    tmp = session / ".out"
    shutil.rmtree(tmp, ignore_errors=True)
    write_outputs(r, tmp)
    info = {"stations": window, "origin_pose": pose_dict(anchor)}
    room = json.loads((tmp / "room.json").read_text())
    room["window"] = info
    (tmp / "room.json").write_text(json.dumps(room, ensure_ascii=False, indent=1))
    plan = json.loads((tmp / "plan.json").read_text())
    plan["next"] = raw
    plan["window"] = info
    plan["timing_s"] = {k: round(v, 3) for k, v in marks.items()}
    (tmp / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    for f in tmp.iterdir():
        f.replace(session / f.name)
    tmp.rmdir()


# --- HTTP ---


def create_app(
    root: Path | str = DATA,
    allow: list | None = None,
    reconstruct: Reconstruct | None = None,
) -> FastAPI:
    root = Path(root)
    allow = parse_allow(ALLOW) if allow is None else allow
    reconstruct = VggtReconstruct() if reconstruct is None else reconstruct
    # GPU は 1 つずつ使う。同時に来たリクエストはここで順に待つ
    gpu = threading.Lock()
    refused: set[str] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # 最初のステーションも 20 秒に収まるよう、VGGT は起動したときに読んでおく
        load = getattr(reconstruct, "load", None)
        if load is not None:
            load()
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
        try:
            m = parse_meta(meta)
            files = read_images(images, m)
        except BadRequest as e:
            raise HTTPException(400, str(e)) from e
        session = root / m["session_id"]
        sid = m["station_id"]
        save_station(session, sid, m, files)
        marks = {"save": time.perf_counter() - t0}

        battery = m.get("battery_v")
        if battery is not None and battery < LOW_BATTERY_V:
            log.info("%s/%03d: 電池が %.2f V なのでやめる", session.name, sid, battery)
            return {"action": "done", "reason": "low_battery"}
        if len(station_ids(session)) >= MAX_STATIONS:
            log.info("%s/%03d: %d か所を超えたのでやめる", session.name, sid, MAX_STATIONS)
            return {"action": "done", "reason": "step_limit"}

        t = time.perf_counter()
        with gpu:
            marks["wait"] = time.perf_counter() - t
            try:
                action = plan_station(session, sid, reconstruct, marks)
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

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info(
        "%s:%d で待ち受け（受ける送り元: %s、置き場: %s）", HOST, PORT, ALLOW or "すべて", DATA
    )
    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
