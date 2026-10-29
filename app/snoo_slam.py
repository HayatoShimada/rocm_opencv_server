"""PiCar-X が走っている間に、カメラの位置と点を VGGT-SLAM 2.0 で求め続ける（ホストのサービス）。

VGGT-SLAM 2.0（MIT-SPARK、BSD-2）は、数コマずつ（サブマップ）VGGT に通し、隣のサブマップと
重なる 1 コマで SL(4) の向き・大きさをつなぎ、GTSAM で全体を合わせ直す。似た画像（SALAD）が
見つかればループを閉じる。torch・gtsam・vggt_slam は main の .venv ではなく別の venv
（~/orca/vggt-slam-venv。入れ方は deploy/snoo-slam.service）にあるので、使うときだけ読む。
このファイルの残り（置き場の形・座標合わせ・書き出し）は numpy だけで動き、main の .venv でも
読める。

    ~/orca/vggt-slam-venv/bin/python -m app.snoo_slam   # サービス（deploy/snoo-slam.service）

入るもの:
- 走っている間のフレーム：検出サーバーが /run/snoo/slam/<送り元>/ に書く（app/snoo_slam_tap.py。
  形はそちらの先頭）。名前（撮った時刻）で読んだ所を覚え、消すのは書く側に任せる
- 止まって首を振ったフレーム：地図のサーバーが <SNOO_MAP_DATA>/<セッション>/<NNN>/ に残す
  meta.json の frames[]（moving が false のもの）

いちばん新しいセッション（置き場の更新時刻）を今のセッションとし、その最初のステーションより後に
撮ったフレームだけを使う。サブマップは submap_size コマ（既定 8）たまるか、flush_s 秒（既定 1.5）
新しいフレームが来なければ回す。猫の箱（dets）の所は灰色で塗り、点にも使わない。

セッションの座標（m。原点はステーション 0 の真下の床、x はステーション 0 の車の向き、y 左、z 上。
app/roommap.py・poses.json と同じ）への合わせ方:
- 床：SLAM の点から RANSAC で平面を探す（カメラの「上」から 25° 以内で、カメラがみな上にあるもの）
- 大きさ：止まって撮ったコマのカメラの床からの高さの中央値を camera_height_m（8 cm）とする
- 床の上の位置と向き：poses.json のステーションの位置に、回転と平行移動だけで合わせる
  （ステーションの広がりが 15 cm 未満なら、ステーションの向きの差の平均で回す）。
  ステーションの SLAM の位置は、止まって撮ったコマのカメラの中心の平均（無ければ、止まる直前に
  走りながら撮ったコマ）。車の向きは tilt 0 に近くパン 0 に近いコマの光軸を床に落とし、パンの分戻す
- poses.json が無ければ、最初のコマを原点・その向きを x にする（仮。status の alignment.method）

追跡が崩れたら（VGGT の値が有限でない・サブマップの間の大きさの比が 1/4〜4 を外れる・
つなぎで例外）、SLAM を作り直して新しい区間（segment）を始める。区間ごとに座標を合わせるので、
新しい区間はステーションのコマが入るまで lost のまま。

書くもの（<SNOO_MAP_DATA>/<セッション>/slam/。読む側が半端なファイルを読まないよう、別の名前で
書いてから名前を変える）:
- trajectory.jsonl：1 行 1 キーフレーム、撮った順。サブマップを回すたびに全体を書き直す
  （ループを閉じると前の位置も変わるため）。欄:
  t_capture_ns・file（frames/ の名前）・source（tap / still / moving）・station（止まって撮った
  コマだけ）・segment・x・y（m）・yaw（rad、-π..π）・z（カメラの床からの高さ、m）・
  pan_deg・tilt_deg・latency_s（届いてから初めて位置が出るまで）
- points.npz：xyz（float32 (N, 3)、セッションの座標、z は床からの高さ）・rgb（uint8 (N, 3)）・
  count（そのボクセルに入った点の数）・voxel_m・t_update（UNIX 時刻）。voxel_m（既定 3 cm）ごとに
  1 点に間引く。床（|z| < 2 cm）・障害物（2 cm〜車の高さ）・上の層は読む側が z で分ける
- status.json：session・state（init / ok / lost / no_floor）・keyframes・submaps・segments・
  loop_closures・failures・last_keyframe_t_capture_ns・last_keyframe_age_s（最後に位置が出た
  キーフレームが届いてから）・last_update・last_latency_s・timing（submap_ms の p50・最大）・
  alignment（method・scale・floor_inliers・stations・rmse_m）・pose（最後のキーフレームの
  x・y・yaw・t_capture_ns）・vram_gb
- frames/<t_capture_ns>.jpg と frames.jsonl：使ったフレーム（猫は塗ったもの）と、その欄。
  後で同じものを流し直せる

地図のサーバーへの JSON の案（まだ作っていない。地図のサーバーが上のファイルを読んで返す）:
- GET /v1/snoo/map/<session>/slam/pose → {"state", "t_capture_ns", "age_s", "x", "y", "yaw_deg",
  "segment"}（今の車の位置。/twin の車の絵と、次の 1 歩の出発点に使う）
- GET /v1/snoo/map/<session>/slam/trajectory → {"poses": [{t_capture_ns, x, y, yaw_deg, source,
  station}, ...]}
- GET /v1/snoo/map/<session>/slam/points?since=<t_update> → points.npz をそのまま
  （application/octet-stream）か、z で分けた格子のセル {"floor": [[ix, iy], ...], "obstacle": ...}
間取りに足すときは、points.npz の点を app/roommap.Grid に、ステーションの VGGT の点と同じ高さの
分け方で足す（1 セル MIN_POINTS_PER_CELL 点以上）。SLAM の点は走った道の両脇も埋める。
state が ok でない間・alignment.rmse_m が大きい間は使わない。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("snoo_slam")

QUEUE_DIR = os.getenv("SNOO_SLAM_QUEUE_DIR", "/run/snoo/slam")
DATA = os.getenv("SNOO_MAP_DATA", "data/snoo_map/sessions")
OUT_NAME = "slam"
# 床からカメラの中心まで（snoo_mapper の MapParams.camera_height_m と同じ）
CAMERA_HEIGHT_M = float(os.getenv("SNOO_SLAM_CAMERA_HEIGHT_M", "0.08"))
# 1 つのサブマップに入れる新しいコマ。前のサブマップと重ねる 1 コマが足される。
# 8 なら 9 コマで VGGT が 1.2〜2 秒（72 CU、検出と同時）
SUBMAP_SIZE = int(os.getenv("SNOO_SLAM_SUBMAP_SIZE", "8"))
# 地図のサーバーが VGGT を回している間に置くファイル（app/snoo_map_server.py の GPU_BUSY_FILE）。
# ある間はサブマップを回さない。地図のサーバーが落ちて残ったときのため、古いものは見ない
MAP_BUSY_FILE = os.getenv("SNOO_GPU_BUSY_FILE", "/run/snoo/map_busy")
MAP_BUSY_MAX_S = 60.0


def map_is_busy() -> bool:
    try:
        return time.time() - os.path.getmtime(MAP_BUSY_FILE) < MAP_BUSY_MAX_S
    except OSError:
        return False


# 新しいコマがこの秒数来なければ、submap_size に足りなくても回す（止まったとき）
FLUSH_S = float(os.getenv("SNOO_SLAM_FLUSH_S", "1.5"))
# ループを閉じるか（SALAD を使う。GPL-3.0）
LOOP = os.getenv("SNOO_SLAM_LOOP", "1") == "1"
VOXEL_M = 0.03
# 点を取り出すときの画素の間引き（392x518 → 98x130）
POINT_STRIDE = 4
# 点を使うカメラからの距離の上限（m）。遠いほどぶれる
MAX_RANGE_M = 3.0
# この時間（秒）より前に更新したセッションは、今のセッションとしない（ステーションは 30 秒ほどごとに
# 増えるので、10 分止まっていれば探索は終わっている。サービスを起動し直したとき、終わった
# セッションを最初から回し直さない）
SESSION_MAX_AGE_S = float(os.getenv("SNOO_SLAM_SESSION_MAX_AGE_S", "600"))
# サブマップの間の大きさの比がこれを外れたら追跡が崩れたとする
MAX_SCALE_JUMP = 4.0
# ステーションの広がりがこれより小さければ、位置ではなく向きで回す（m）
MIN_SPREAD_M = 0.15
FLOOR_TOL_M = 0.02

# VGGT に入れる大きさ（vggt.utils.load_fn の crop: 幅 518、高さは 14 の倍数）
VGGT_W = 518


@dataclass
class KeyFrame:
    t_capture_ns: int
    image: Path  # 元の画像
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    steer_deg: float = 0.0
    speed: float = 0.0
    moving: bool = False
    station: int | None = None  # 止まって撮ったコマだけ
    # tap（検出サーバーから）/ still（ステーション）/ moving（ステーションの置き場の走行中のコマ）
    source: str = "tap"
    boxes: list = field(default_factory=list)  # 猫の箱（幅・高さに対する比）
    t_recv: float | None = None  # 届いた時刻（UNIX。遅れを測る）
    file: str = ""  # frames/ の名前（取り込んだ後）

    def row(self) -> dict:
        return {
            "t_capture_ns": self.t_capture_ns,
            "file": self.file,
            "source": self.source,
            "station": self.station,
            "pan_deg": round(self.pan_deg, 2),
            "tilt_deg": round(self.tilt_deg, 2),
            "steer_deg": round(self.steer_deg, 2),
            "speed": round(self.speed, 2),
            "moving": self.moving,
            "boxes": self.boxes,
        }


# --- 入るもの ---


def read_queue(sender_dir: Path, after_ns: int = -1) -> list[KeyFrame]:
    """app/snoo_slam_tap.py が書いたもののうち、撮った時刻が after_ns より後のもの（撮った順）。"""
    out = []
    for p in sorted(sender_dir.glob("*.json")):
        if p.name.startswith("."):
            continue
        try:
            t = int(p.stem)
        except ValueError:
            continue
        if t <= after_ns:
            continue
        try:
            info = json.loads(p.read_text())
        except (OSError, ValueError):
            continue  # 書く側が消した・書きかけ
        img = sender_dir / info.get("image", f"{p.stem}.jpg")
        if not img.exists():
            continue
        out.append(
            KeyFrame(
                t_capture_ns=int(info["t_capture_ns"]),
                image=img,
                pan_deg=float(info.get("pan_deg") or 0.0),
                tilt_deg=float(info.get("tilt_deg") or 0.0),
                steer_deg=float(info.get("steer_deg") or 0.0),
                speed=float(info.get("speed") or 0.0),
                moving=bool(info.get("moving", True)),
                source="tap",
                boxes=[d["box"] for d in info.get("dets", []) if "box" in d],
                t_recv=info.get("t_recv"),
            )
        )
    return out


def station_frames(d: Path, still: bool = True, moving: bool = False) -> list[KeyFrame]:
    """ステーションの置き場（地図のサーバーが残す NNN/）のコマ（撮った順）。"""
    meta = json.loads((d / "meta.json").read_text())
    sid = int(meta.get("station_id", int(d.name)))
    out = []
    for f in meta["frames"]:
        mv = bool(f.get("moving"))
        if (mv and not moving) or (not mv and not still):
            continue
        out.append(
            KeyFrame(
                t_capture_ns=int(f["t_capture_ns"]),
                image=d / f["file"],
                pan_deg=float(f.get("pan_deg") or 0.0),
                tilt_deg=float(f.get("tilt_deg") or 0.0),
                steer_deg=float(f.get("steer_deg") or 0.0),
                speed=float(f.get("speed") or 0.0),
                moving=mv,
                station=None if mv else sid,
                source="moving" if mv else "still",
            )
        )
    return sorted(out, key=lambda k: k.t_capture_ns)


def station_dirs(session: Path) -> list[Path]:
    return sorted(
        d
        for d in session.iterdir()
        if d.is_dir() and len(d.name) == 3 and d.name.isdigit() and (d / "meta.json").exists()
    )


def read_station_poses(session: Path) -> dict[int, tuple[float, float, float]]:
    """poses.json（地図のサーバーが書く）。{ステーション: (x, y, yaw rad)}。"""
    p = session / "poses.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return {int(k): (v["x"], v["y"], math.radians(v["yaw_deg"])) for k, v in data.items()}


def current_session(data: Path, now: float | None = None) -> Path | None:
    """いちばん新しく更新したセッション（max_age 秒より古いものは除く）。"""
    now = time.time() if now is None else now
    best = None
    if not data.exists():
        return None
    for s in data.iterdir():
        if not s.is_dir() or s.name.startswith("."):
            continue
        try:
            m = s.stat().st_mtime
        except OSError:
            continue
        if now - m > SESSION_MAX_AGE_S:
            continue
        if best is None or m > best[0]:
            best = (m, s)
    return None if best is None else best[1]


def mask_boxes(image: np.ndarray, boxes: list) -> np.ndarray:
    """猫の箱を灰色で塗る（VGGT が猫を部屋の一部と見ないように）。"""
    if not boxes:
        return image
    out = image.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        a, b = int(max(0, math.floor(x1 * w))), int(min(w, math.ceil(x2 * w)))
        c, d = int(max(0, math.floor(y1 * h))), int(min(h, math.ceil(y2 * h)))
        out[c:d, a:b] = 127
    return out


def box_mask(boxes: list, shape: tuple[int, int]) -> np.ndarray:
    """VGGT の出力の大きさ（高さ, 幅）で、猫の箱の所が True の画素。"""
    m = np.zeros(shape, bool)
    h, w = shape
    for x1, y1, x2, y2 in boxes:
        a, b = int(max(0, math.floor(x1 * w))), int(min(w, math.ceil(x2 * w)))
        c, d = int(max(0, math.floor(y1 * h))), int(min(h, math.ceil(y2 * h)))
        m[c:d, a:b] = True
    return m


# --- 座標合わせ（numpy だけ） ---


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """dst ≈ s R src + t となる s・R・t（最小二乗。Umeyama 1991）。src・dst は (n, d)。"""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, sig, vt = np.linalg.svd(cov)
    sign = np.eye(src.shape[1])
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1
    r = u @ sign @ vt
    var = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(sig) @ sign) / var) if with_scale and var > 0 else 1.0
    t = mu_d - s * r @ mu_s
    return s, r, t


def camera_up(r_c2w: np.ndarray, tilt_deg) -> np.ndarray:
    """各コマの「上」の見込み（SLAM の座標）。上を tilt° 向いたカメラでは、カメラの座標で
    (0, -cos tilt, sin tilt)（app/snoo_mapper.up_in_world と同じ）。r_c2w は (n, 3, 3)。
    """
    t = np.radians(np.asarray(tilt_deg, float))
    up = np.stack([np.zeros_like(t), -np.cos(t), np.sin(t)], -1)
    return np.einsum("nij,nj->ni", r_c2w, up)


@dataclass
class Floor:
    normal: np.ndarray  # 上向きの単位ベクトル（SLAM の座標）
    offset: float  # 床からの高さ = normal・p + offset
    inliers: int


def fit_floor(
    points: np.ndarray,
    up_prior: np.ndarray,
    cams: np.ndarray,
    max_tilt_deg: float = 25.0,
    iters: int = 600,
    rng: np.random.Generator | None = None,
) -> Floor | None:
    """点から床の平面を RANSAC で探す。向きが up_prior から max_tilt_deg 以内で、カメラがみな
    上にあるもの。許す外れは、カメラの高さの中央値の 15 %（大きさが決まっていないため）を、
    はじめはカメラから点までの距離の中央値の 0.5 % で見積もる。
    """
    rng = rng or np.random.default_rng(0)
    points = np.asarray(points, float)
    if len(points) < 3:
        return None
    if len(points) > 100_000:
        points = points[rng.choice(len(points), 100_000, replace=False)]
    up = up_prior / np.linalg.norm(up_prior)
    thr = 0.005 * float(np.median(np.linalg.norm(points - cams.mean(0), axis=1)))
    cos_max = math.cos(math.radians(max_tilt_deg))
    tri = points[rng.integers(0, len(points), (iters, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-12
    n[ok] /= norm[ok, None]
    n *= np.where(n @ up < 0, -1.0, 1.0)[:, None]
    ok &= n @ up > cos_max
    best = None
    for i in np.flatnonzero(ok):
        off = -float(n[i] @ tri[i, 0])
        if (cams @ n[i] + off).min() <= 0:  # カメラより上の平面は床ではない
            continue
        count = int((np.abs(points @ n[i] + off) < thr).sum())
        if best is None or count > best[0]:
            best = (count, n[i], off)
    if best is None:
        return None
    _, nn, off = best
    for _ in range(3):  # 床に近い点で当て直す
        h = float(np.median(cams @ nn + off))
        inl = points[np.abs(points @ nn + off) < max(thr, 0.15 * h)]
        if len(inl) < 3:
            break
        c = inl.mean(0)
        n2 = np.linalg.svd(inl - c, full_matrices=False)[2][2]
        nn = n2 if n2 @ nn > 0 else -n2
        off = -float(nn @ c)
    h = float(np.median(cams @ nn + off))
    count = int((np.abs(points @ nn + off) < max(thr, 0.15 * h)).sum())
    return Floor(nn, off, count)


def _rotate_about(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def car_heading(r_c2w: np.ndarray, normal: np.ndarray, pan_deg: float) -> np.ndarray:
    """そのコマの車の向き（床に沿う単位ベクトル、SLAM の座標）。光軸を床に落とし、パン（右が +）の
    分だけ上向きの軸のまわりに左へ戻す。
    """
    fwd = r_c2w[:, 2] - (r_c2w[:, 2] @ normal) * normal
    fwd = _rotate_about(fwd, normal, math.radians(pan_deg))
    return fwd / np.linalg.norm(fwd)


@dataclass
class SessionFrame:
    """SLAM の座標 → セッションの座標（m、z は床からの高さ）：p = scale * M q + b。"""

    scale: float
    m: np.ndarray  # (3, 3)。行が x・y・z の軸（SLAM の座標の単位ベクトル）
    b: np.ndarray  # (3,)

    def apply(self, q: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(q, float) @ self.m.T + self.b

    def yaw(self, heading: np.ndarray) -> float:
        d = self.m @ heading
        return math.atan2(d[1], d[0])


@dataclass
class Alignment:
    frame: SessionFrame | None
    method: str  # stations / heading / first_frame / none
    floor: Floor | None = None
    stations: list[int] = field(default_factory=list)
    rmse_m: float | None = None
    sim2_scale: float | None = None  # 位置だけで大きさも合わせたときの大きさ（床の大きさとの比）

    def summary(self) -> dict:
        return {
            "method": self.method,
            "scale": None if self.frame is None else round(self.frame.scale, 5),
            "floor_inliers": None if self.floor is None else self.floor.inliers,
            "stations": self.stations,
            "rmse_m": None if self.rmse_m is None else round(self.rmse_m, 4),
            "sim2_scale_ratio": None if self.sim2_scale is None else round(self.sim2_scale, 4),
        }


def heading_index(frames: list[KeyFrame], idx: list[int]) -> int:
    """車の向きを出すのに使うコマ：tilt 0 に近い段の、パンがいちばん 0 に近いもの。"""
    return min(idx, key=lambda i: (round(abs(frames[i].tilt_deg)), abs(frames[i].pan_deg)))


def station_anchors(
    frames: list[KeyFrame], station_start_ns: dict[int, int], max_gap_s: float = 3.0
) -> dict[int, list[int]]:
    """ステーションごとに、その位置を表すコマ。止まって撮ったコマがあればそれ、無ければ
    そのステーションで止まる直前（max_gap_s 秒以内）に走りながら撮った最後のコマ。
    """
    out: dict[int, list[int]] = {}
    for i, f in enumerate(frames):
        if f.station is not None:
            out.setdefault(f.station, []).append(i)
    for s, t0 in station_start_ns.items():
        if s in out:
            continue
        before = [
            i
            for i, f in enumerate(frames)
            if f.station is None and t0 - max_gap_s * 1e9 <= f.t_capture_ns <= t0
        ]
        if before:
            out[s] = [max(before, key=lambda i: frames[i].t_capture_ns)]
    return out


def align_session(
    cams: np.ndarray,
    rots: np.ndarray,
    frames: list[KeyFrame],
    points: np.ndarray,
    station_poses: dict[int, tuple[float, float, float]],
    station_start_ns: dict[int, int] | None = None,
    height_m: float = CAMERA_HEIGHT_M,
    rng: np.random.Generator | None = None,
) -> Alignment:
    """SLAM のカメラ（中心 cams (n, 3)、向き rots (n, 3, 3) カメラ → 世界）と点を、セッションの
    座標に合わせる。大きさは床とカメラの高さ、床の上の位置と向きは poses.json のステーション。
    """
    if len(frames) == 0:
        return Alignment(None, "none")
    up = camera_up(rots, [f.tilt_deg for f in frames]).mean(0)
    floor = fit_floor(points, up, cams, rng=rng) if len(points) >= 3 else None
    if floor is None:
        return Alignment(None, "none")
    n = floor.normal
    h = cams @ n + floor.offset
    still = np.array([f.station is not None for f in frames])
    use = still if still.sum() >= 3 else np.ones(len(frames), bool)
    hh = h[use & (h > 0)]
    if len(hh) == 0:
        return Alignment(None, "none", floor)
    scale = height_m / float(np.median(hh))
    # 床の上の 2 つの軸（e1 × e2 = n で右手系）
    e1 = car_heading(rots[0], n, frames[0].pan_deg)
    e2 = np.cross(n, e1)
    basis = np.stack([e1, e2])
    o = cams[0] - h[0] * n  # 最初のコマの真下の床（SLAM の座標）

    def flat(q):
        return scale * (np.asarray(q) - o) @ basis.T

    anchors = station_anchors(frames, station_start_ns or {})
    common = sorted(s for s in anchors if s in station_poses)
    p_slam, p_ses, dyaw = [], [], []
    for s in common:
        idx = anchors[s]
        p_slam.append(flat(cams[idx].mean(0)))
        p_ses.append(station_poses[s][:2])
        k = heading_index(frames, idx)
        v = basis @ car_heading(rots[k], n, frames[k].pan_deg)
        dyaw.append(station_poses[s][2] - math.atan2(v[1], v[0]))
    p_slam, p_ses = np.array(p_slam).reshape(-1, 2), np.array(p_ses).reshape(-1, 2)
    sim2 = None
    if len(common) >= 2 and np.linalg.norm(p_slam - p_slam.mean(0), axis=1).max() >= MIN_SPREAD_M:
        _, r2, t2 = umeyama(p_slam, p_ses, with_scale=False)
        method = "stations"
        sim2 = umeyama(p_slam, p_ses, with_scale=True)[0]
    elif common:
        th = math.atan2(np.mean(np.sin(dyaw)), np.mean(np.cos(dyaw)))
        r2 = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        t2 = (p_ses - p_slam @ r2.T).mean(0)
        method = "heading"
    else:
        r2, t2, method = np.eye(2), np.zeros(2), "first_frame"
    m2 = r2 @ basis
    m = np.stack([m2[0], m2[1], n])
    b = np.concatenate([t2 - scale * m2 @ o, [scale * floor.offset]])
    frame = SessionFrame(scale, m, b)
    rmse = None
    if common:
        fit = p_slam @ r2.T + t2
        rmse = float(np.sqrt(((fit - p_ses) ** 2).sum(1).mean()))
    return Alignment(frame, method, floor, common, rmse, sim2)


def frame_poses(
    al: Alignment, cams: np.ndarray, rots: np.ndarray, frames: list[KeyFrame]
) -> list[tuple[float, float, float, float]]:
    """コマごとの (x, y, yaw, z)：カメラの真下の床の位置・車の向き・カメラの床からの高さ。"""
    p = al.frame.apply(cams)
    out = []
    for i, f in enumerate(frames):
        yaw = al.frame.yaw(car_heading(rots[i], al.floor.normal, f.pan_deg))
        out.append((float(p[i, 0]), float(p[i, 1]), yaw, float(p[i, 2])))
    return out


def voxel_down(xyz: np.ndarray, rgb: np.ndarray, voxel: float):
    """voxel ごとに 1 点（平均）にする。返すのは xyz・rgb・count。"""
    if len(xyz) == 0:
        return xyz.astype(np.float32), rgb.astype(np.uint8), np.zeros(0, np.int32)
    key = np.floor(xyz / voxel).astype(np.int64)
    _, inv, count = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    k = len(count)
    out = np.zeros((k, 3))
    col = np.zeros((k, 3))
    np.add.at(out, inv, xyz)
    np.add.at(col, inv, rgb.astype(float))
    out /= count[:, None]
    col /= count[:, None]
    return out.astype(np.float32), col.round().astype(np.uint8), count.astype(np.int32)


# --- 書き出し ---


def _replace(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_trajectory(out: Path, rows: list[dict]) -> None:
    _replace(out / "trajectory.jsonl", "".join(json.dumps(r) + "\n" for r in rows).encode())


def write_points(out: Path, xyz, rgb, count, voxel: float) -> None:
    tmp = out / ".points.npz"
    with tmp.open("wb") as f:
        np.savez(f, xyz=xyz, rgb=rgb, count=count, voxel_m=voxel, t_update=time.time())
    tmp.replace(out / "points.npz")


def write_status(out: Path, status: dict) -> None:
    _replace(out / "status.json", json.dumps(status, ensure_ascii=False, indent=1).encode())


def read_trajectory(out: Path) -> list[dict]:
    p = out / "trajectory.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# --- VGGT-SLAM（GPU。別の venv） ---


def _install_stubs(loop: bool) -> None:
    """vggt_slam の Viewer（viser のサーバーを立てる）を読まないようにする。ループを閉じないときは
    SALAD（GPL-3.0）も読まない。
    """
    sys.modules.setdefault("vggt_slam.viewer", types.SimpleNamespace(Viewer=lambda *a, **k: None))
    if not loop and "vggt_slam.loop_closure" not in sys.modules:
        import torch

        class _NoLoop:
            def get_all_submap_embeddings(self, submap):
                return torch.zeros((submap.get_all_frames().shape[0], 1))

            def find_loop_closures(self, *a, **k):
                return []

        sys.modules["vggt_slam.loop_closure"] = types.SimpleNamespace(ImageRetrieval=_NoLoop)


def load_vggt_model():
    """VGGT（facebook/VGGT-1B）を bf16 で GPU に載せる（2.5 GB）。1 回通して準備しておく。"""
    import torch
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(torch.bfloat16).cuda().eval()
    with torch.no_grad():
        model(torch.zeros((1, 2, 3, 392, VGGT_W), device="cuda"))
    return model


class VggtSlam:
    """vggt_slam.solver.Solver を、コマを足しながら回す。"""

    def __init__(self, model=None, loop: bool = LOOP, conf_threshold: float = 25.0, lc_thres=0.95):
        _install_stubs(loop)
        import vggt_slam.solver as vs

        self._vs = vs
        self._scales: list[float] = []
        orig = vs.estimate_scale_pairwise

        def scale_hook(x, y, DEBUG=False):  # noqa: N803 元の名前のまま
            s = orig(x, y)
            self._scales.append(float(s[0]))
            return s

        vs.estimate_scale_pairwise = scale_hook
        self.loop = loop
        self.model = model if model is not None else load_vggt_model()
        self.solver = vs.Solver(init_conf_threshold=conf_threshold, lc_thres=lc_thres)
        self.overlap: Path | None = None
        self.segment = 0
        self.submaps = 0
        self.loops = 0
        self.failures = 0
        self.timing: list[dict] = []
        self.frozen: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}  # 前の区間の位置

    def reset(self, keep: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """追跡が崩れたとき。今の区間の位置（keep。崩れる前のもの）を残して、新しい区間を始める。"""
        for name, (c, r) in keep.items():
            self.frozen[name] = (c, r, self.segment)
        from vggt_slam.frame_overlap import FrameTracker
        from vggt_slam.graph import PoseGraph
        from vggt_slam.map import GraphMap

        s = self.solver
        s.map, s.graph, s.flow_tracker = GraphMap(), PoseGraph(), FrameTracker()
        s.current_working_submap = None
        self.overlap = None
        self.segment += 1
        self.failures += 1

    def _predict(self, names: list[str], masks: dict[int, list]):
        import torch

        for attempt in range(2):
            pred = self.solver.run_predictions(names, self.model, 1 if self.loop else 0, None, None)
            if all(
                np.isfinite(pred[k]).all()
                for k in ("depth", "depth_conf", "extrinsic", "intrinsic")
            ):
                break
            log.warning("VGGT の値が有限でない（%d 回目）", attempt + 1)
            torch.cuda.empty_cache()
        else:
            return None
        h, w = pred["depth_conf"].shape[1:3]
        for i, boxes in masks.items():
            pred["depth_conf"][i][box_mask(boxes, (h, w))] = 0.0
        return pred

    def add(self, paths: list[Path], boxes: list[list] | None = None, retry: bool = True) -> dict:
        """新しいコマを 1 つのサブマップとして足す。前のサブマップの最後の 1 コマを先頭に重ねる。
        返すのは回った時間などの dict（ok が False なら追跡が崩れて区間を作り直した）。
        """
        import torch

        boxes = boxes or [[] for _ in paths]
        names = [str(p) for p in paths]
        masks = {i: b for i, b in enumerate(boxes) if b}
        if self.overlap is not None:
            names = [str(self.overlap), *names]
            masks = {i + 1: b for i, b in masks.items()}
        if len(names) < 2:  # 1 コマだけでは VGGT の向きが決まらない。次に回す
            self.overlap = Path(names[0])
            return {"ok": True, "frames": 0, "skipped": True}
        before = self.poses()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_scales = len(self._scales)
        loops_before = self.solver.graph.get_num_loops()
        pred = self._predict(names, masks)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ok = pred is not None
        if ok:
            try:
                self.solver.add_points(pred)
                self.solver.graph.optimize()
            except (RuntimeError, ValueError) as e:  # SL(4) が作れない（大きさが nan）など
                log.warning("サブマップをつなげない: %s", e)
                ok = False
        new = self._scales[n_scales:]
        if (
            ok
            and new
            and (not math.isfinite(new[0]) or not 1 / MAX_SCALE_JUMP < new[0] < MAX_SCALE_JUMP)
        ):
            log.warning("サブマップの間の大きさの比 %.3f が外れている", new[0])
            ok = False
        t2 = time.perf_counter()
        info = {
            "ok": ok,
            "frames": len(names),
            "vggt_ms": round((t1 - t0) * 1000, 1),
            "backend_ms": round((t2 - t1) * 1000, 1),
            "submap_ms": round((t2 - t0) * 1000, 1),
            "scale": new[0] if new else None,
            "loop": False,
            "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        if not ok:
            first = 1 if self.overlap is not None else 0
            self.reset(before)
            # 崩れたサブマップの新しいコマから、新しい区間を始める（1 回だけ）
            if retry and len(names) - first >= 2:
                rest = [Path(n) for n in names[first:]]
                info["retry"] = self.add(
                    rest, [masks.get(i, []) for i in range(first, len(names))], False
                )
            return info
        self.submaps += 1
        if self.solver.graph.get_num_loops() > loops_before:
            self.loops += 1
            info["loop"] = True
        self.overlap = Path(names[-1])
        self.timing.append(info)
        return info

    def poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """今の区間のコマごとのカメラの中心と向き（カメラ → 世界）。
        重なったコマは前のサブマップの値。
        """
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        g = self.solver.graph
        for sm in self.solver.map.ordered_submaps_by_key():
            if sm.get_lc_status():
                continue
            ps = sm.get_all_poses_world(g)
            for name, p in zip(sm.img_names, ps, strict=False):
                out.setdefault(str(name), (p[:3, 3].copy(), p[:3, :3].copy()))
        return out

    def points(self, stride: int = POINT_STRIDE) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """今の区間の点（SLAM の座標）と色（RGB）と、どのコマの点か（names の番号）と names。"""
        g = self.solver.graph
        xyz, rgb, src, names = [], [], [], []
        for sm in self.solver.map.ordered_submaps_by_key():
            if sm.get_lc_status():
                continue
            for i in range(len(sm.poses)):
                pc = sm.pointclouds[i][::stride, ::stride].reshape(-1, 3)
                ok = (sm.conf_masks[i][::stride, ::stride] > sm.conf_threshold).reshape(-1)
                pc = pc[ok]
                if len(pc) == 0:
                    continue
                hmat = g.get_homography(sm.get_id() + i)
                ph = np.hstack([pc, np.ones((len(pc), 1))]) @ hmat.T
                xyz.append(ph[:, :3] / ph[:, 3:])
                rgb.append(sm.colors[i][::stride, ::stride].reshape(-1, 3)[ok])
                src.append(np.full(len(pc), len(names), np.int32))
                names.append(str(sm.img_names[i]))
        if not xyz:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, np.int32), []
        return np.concatenate(xyz), np.concatenate(rgb), np.concatenate(src), names


# --- サービス ---


def _p50(v: list[float]) -> float | None:
    return None if not v else round(float(np.median(v)), 3)


class SlamService:
    """置き場を見て、今のセッションのコマを SLAM に足し、結果を書く。
    slam_factory を渡せば VGGT を使わずに試せる（テスト）。
    """

    def __init__(
        self,
        data: str | Path = DATA,
        queue: str | Path = QUEUE_DIR,
        submap_size: int = SUBMAP_SIZE,
        flush_s: float = FLUSH_S,
        loop: bool = LOOP,
        station_moving: bool = False,
        still: bool = True,
        slam_factory=None,
        clock=time.time,
    ):
        self.data = Path(data)
        self.queue = Path(queue)
        self.submap_size = submap_size
        self.flush_s = flush_s
        self.loop = loop
        # ステーションの置き場の走行中のコマも使う（タップが無いとき）
        self.station_moving = station_moving
        self.still = still  # 止まって撮ったコマを使う
        self.clock = clock
        self._model = None
        self.slam_factory = slam_factory or self._new_slam
        self.session: Path | None = None
        self._last_status = -1e18

    def _new_slam(self):
        spare, self._spare = getattr(self, "_spare", None), None
        if spare is not None:
            return spare
        if self._model is None:
            self._model = load_vggt_model()
        return VggtSlam(self._model, loop=self.loop)

    def preload(self) -> None:
        """VGGT と SALAD を先に読んでおく（最初のサブマップを 5〜10 秒待たせない）。"""
        self._spare = self._new_slam()

    def start(self, session: Path) -> None:
        """セッションごとの状態を作り直す。"""
        log.info("セッション %s を始める", session.name)
        self.session = session
        self.out = session / OUT_NAME
        self.frames_dir = self.out / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        old = self.out / "frames.jsonl"
        if old.exists():  # 起動し直したとき。前の欄は退ける（frames/ の画像はそのまま使う）
            old.replace(self.out / f"frames.{int(time.time())}.old.jsonl")
        self.slam = None
        self.kfs: dict[str, KeyFrame] = {}
        self.pending: list[KeyFrame] = []
        self.seen_queue: dict[str, int] = {}
        self.seen_stations: set[int] = set()
        self.station_start: dict[int, int] = {}
        self.t_min: int | None = None
        self.last_new = self.clock()
        self.done_at: dict[str, float] = {}
        self.latency: dict[str, float] = {}
        self.rows: list[dict] = []
        self.old_rows: list[dict] = []
        self.points_now: tuple[np.ndarray, np.ndarray] | None = None
        self.points_old: list[tuple[np.ndarray, np.ndarray]] = []
        self.alignment: Alignment | None = None
        self.last_info: dict = {}
        self.state = "init"

    def ingest(self, k: KeyFrame) -> bool:
        name = f"{k.t_capture_ns:020d}.jpg"
        if name in self.kfs:
            return False
        dst = self.frames_dir / name
        try:
            if k.boxes:
                import cv2

                img = cv2.imread(str(k.image))
                if img is None:
                    return False
                cv2.imwrite(str(dst), mask_boxes(img, k.boxes))
            else:
                shutil.copyfile(k.image, dst)
        except OSError:  # 書く側が消した
            return False
        k.file = name
        k.t_recv = k.t_recv if k.t_recv is not None else self.clock()
        self.kfs[name] = k
        self.pending.append(k)
        self.last_new = self.clock()
        with (self.out / "frames.jsonl").open("a") as f:
            f.write(json.dumps({**k.row(), "t_recv": k.t_recv}) + "\n")
        return True

    def scan(self) -> int:
        """新しいステーションと、タップの新しいコマを取り込む。取り込んだ数を返す。"""
        n = 0
        for d in station_dirs(self.session):
            sid = int(d.name)
            if sid in self.seen_stations:
                continue
            try:
                fr = station_frames(d, still=self.still, moving=self.station_moving)
                meta_still = station_frames(d, still=True, moving=False)
            except (OSError, ValueError, KeyError):
                continue  # 書きかけ
            self.seen_stations.add(sid)
            if meta_still:
                self.station_start[sid] = meta_still[0].t_capture_ns
                if self.t_min is None:
                    self.t_min = meta_still[0].t_capture_ns
            n += sum(self.ingest(k) for k in fr)
        if self.queue.exists() and self.t_min is not None:
            for sd in sorted(p for p in self.queue.iterdir() if p.is_dir()):
                seen = self.seen_queue.get(sd.name, self.t_min)
                for k in read_queue(sd, seen):
                    n += self.ingest(k)
                    seen = max(seen, k.t_capture_ns)
                self.seen_queue[sd.name] = seen
        return n

    def due(self) -> bool:
        if not self.pending or map_is_busy():
            return False
        return len(self.pending) >= self.submap_size or self.clock() - self.last_new >= self.flush_s

    def step(self) -> dict:
        """たまったコマで 1 つサブマップを回し、結果を書く。"""
        if self.slam is None:
            self.slam = self.slam_factory()
        self.pending.sort(key=lambda k: k.t_capture_ns)
        batch, self.pending = self.pending[: self.submap_size], self.pending[self.submap_size :]
        seg = self.slam.segment
        info = self.slam.add([self.frames_dir / k.file for k in batch], [k.boxes for k in batch])
        if self.slam.segment != seg:  # 区間を作り直した。前の区間の結果は残す
            self.old_rows += self.rows
            self.rows = []
            if self.points_now is not None:
                self.points_old.append(self.points_now)
            self.points_now = None
        now = self.clock()
        for k in batch:
            self.done_at.setdefault(k.file, now)
        self.last_info = info
        self.update()
        return info

    def update(self) -> None:
        poses = self.slam.poses()
        names = sorted(poses, key=lambda n: Path(n).name)
        frames = [self.kfs[Path(n).name] for n in names]
        if not frames:
            self.state = "lost" if self.slam.segment else "init"
            self.write_status()
            return
        cams = np.stack([poses[n][0] for n in names])
        rots = np.stack([poses[n][1] for n in names])
        xyz, rgb, owner, owner_names = self.slam.points()
        sub = xyz[:: max(1, len(xyz) // 60000)]
        al = align_session(
            cams, rots, frames, sub, read_station_poses(self.session), self.station_start
        )
        self.alignment = al
        if al.frame is None:
            self.state = "no_floor"
            self.write_status()
            return
        if self.slam.segment > 0 and al.method == "first_frame":
            self.state = "lost"  # 新しい区間がまだステーションに結び付いていない
        else:
            self.state = "ok" if self.last_info.get("ok", True) else "lost"
        rows = []
        for f, (x, y, yaw, z) in zip(frames, frame_poses(al, cams, rots, frames), strict=True):
            if f.file not in self.latency and f.file in self.done_at:
                self.latency[f.file] = round(self.done_at[f.file] - f.t_recv, 3)
            rows.append(
                {
                    "t_capture_ns": f.t_capture_ns,
                    "file": f.file,
                    "source": f.source,
                    "station": f.station,
                    "segment": self.slam.segment,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "yaw": round(yaw, 4),
                    "z": round(z, 4),
                    "pan_deg": round(f.pan_deg, 2),
                    "tilt_deg": round(f.tilt_deg, 2),
                    "latency_s": self.latency.get(f.file),
                }
            )
        self.rows = rows
        # 点：カメラから MAX_RANGE_M 以内だけ
        p = al.frame.apply(xyz)
        idx = {n: i for i, n in enumerate(names)}
        cam_of = np.array([idx.get(n, 0) for n in owner_names], int)
        cpos = al.frame.apply(cams)
        near = np.linalg.norm(p - cpos[cam_of[owner]], axis=1) < MAX_RANGE_M if len(p) else []
        self.points_now = (p[near], rgb[near])
        write_trajectory(self.out, sorted(self.old_rows + rows, key=lambda r: r["t_capture_ns"]))
        pts = [*self.points_old, self.points_now]
        pxyz = np.concatenate([c[0] for c in pts])
        prgb = np.concatenate([c[1] for c in pts])
        write_points(self.out, *voxel_down(pxyz, prgb, VOXEL_M), VOXEL_M)
        self.write_status()

    def status(self) -> dict:
        allrows = sorted(self.old_rows + self.rows, key=lambda r: r["t_capture_ns"])
        last = allrows[-1] if allrows else None
        lat = [v for v in self.latency.values() if v is not None]
        sm = [t["submap_ms"] for t in (self.slam.timing if self.slam else [])]
        age = None if last is None else round(self.clock() - self.kfs[last["file"]].t_recv, 2)
        slam = self.slam
        return {
            "session": self.session.name,
            "state": self.state,
            "keyframes": len(allrows),
            "received": len(self.kfs),
            "pending": len(self.pending),
            "submaps": slam.submaps if slam else 0,
            "segments": slam.segment + 1 if slam else 0,
            "loop_closures": slam.loops if slam else 0,
            "failures": slam.failures if slam else 0,
            "last_keyframe_t_capture_ns": last["t_capture_ns"] if last else None,
            "last_keyframe_age_s": age,
            "last_update": round(self.clock(), 3),
            "last_latency_s": lat[-1] if lat else None,
            "latency_s": {"p50": _p50(lat), "max": max(lat) if lat else None},
            "timing": {"submap_ms_p50": _p50(sm), "submap_ms_max": max(sm) if sm else None},
            "alignment": self.alignment.summary() if self.alignment else None,
            "pose": None
            if last is None
            else {k: last[k] for k in ("t_capture_ns", "x", "y", "yaw", "segment")},
            "vram_gb": self.last_info.get("vram_gb"),
        }

    def write_status(self) -> None:
        if self.session is None:
            return
        write_status(self.out, self.status())
        self._last_status = self.clock()

    def poll(self) -> bool:
        """1 回見る。サブマップを回したら True。"""
        s = current_session(self.data, self.clock())
        if s is None:
            return False
        if s != self.session:
            self.start(s)
        self.scan()
        if self.due():
            self.step()
            return True
        if self.clock() - self._last_status >= 1.0:
            self.write_status()
        return False

    def drain(self) -> None:
        """たまったものをみな回す（流し直しの終わり）。"""
        while self.pending:
            self.step()

    def run_forever(self, interval: float = 0.2) -> None:
        while True:
            try:
                if not self.poll():
                    time.sleep(interval)
            except Exception:  # 1 つのセッションの失敗でサービスを止めない
                log.exception("SLAM で失敗した。このセッションをやめて、次を待つ")
                self.session = None
                time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DATA, help="セッションの置き場（SNOO_MAP_DATA）")
    parser.add_argument("--queue", default=QUEUE_DIR, help="タップの置き場（SNOO_SLAM_QUEUE_DIR）")
    parser.add_argument("--submap-size", type=int, default=SUBMAP_SIZE)
    parser.add_argument("--flush-s", type=float, default=FLUSH_S)
    parser.add_argument(
        "--loop", action=argparse.BooleanOptionalAction, default=LOOP, help="ループを閉じる"
    )
    parser.add_argument(
        "--station-moving",
        action="store_true",
        help="ステーションの置き場の走行中のコマも使う（タップを使わないとき）",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    quiet_vggt_slam()
    svc = SlamService(
        args.data, args.queue, args.submap_size, args.flush_s, args.loop, args.station_moving
    )
    svc.preload()
    log.info("待ち受け（置き場 %s、タップ %s）", args.data, args.queue)
    svc.run_forever()


def quiet_vggt_slam() -> None:
    """vggt_slam・vggt は 1 サブマップごとに何十行も print するので、標準出力を捨てる
    （ログは標準エラーに出す）。
    """
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115


if __name__ == "__main__":
    main()
