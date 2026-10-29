"""PiCar-X が止まって首を振って撮った画像（1 か所 = 1 ステーション）から間取りを作り、
次の 1 歩を決める。

    uv run python -m scripts.snoo_map_build data/snoo_map/map_A data/snoo_map/map_B \\
        --camera-height-cm 10 --out out/
車には車輪の回転計が無いので、位置も画像から出す。前の超音波は、測れたステーションでは
格子に足す（MapParams.ultrasonic_cm）。
全ステーションの画像をまとめて 1 回 VGGT に通し、同じ座標に並べる（別々に通すと座標がそろわない）。
VGGT の座標は大きさ（m）も向きも決まっていないので、次のように決める。
- 床：点の中から RANSAC で平面を探す。カメラより下で、カメラの「上」に近い向きの平面だけを見る
- 大きさ：カメラの中心から床までの距離を、カメラの高さ（camera_height_m）とする。
  高さはまだ測っていない（約 10 cm）
- 向き：原点は最初のステーションの真下の床、x は最初の車の向き、y は左、z は上
  （app/roommap.py と同じ）
車の向きは、tilt 0 でパンがいちばん 0 に近いコマの光軸を床に落とし、パンの分だけ戻して出す。
パンのサーボは指令の 0.9〜0.97 倍しか回らないが、パン 5° 前後のコマを使うので 1° 未満しか違わない。

床からの高さで点を分け、格子（app/roommap.Grid）に足す。
- ±2 cm：床（走れる）
- 2 cm〜車の高さ + 3 cm：障害物（ぶつかる）
- それより上：「上」の層（机の天板など。車は下をくぐれるが、上にいる猫は見えない）。
  走れるかには使わない
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.roommap import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    Grid,
    Plan,
    Pose,
    frontiers,
    mark_sonar,
    next_best_view,
)

# 1 回に VGGT に入れるコマの数の上限（24 GB の GPU で 60 枚ほどまで。
# 検出サーバーと分け合うので余裕を見る）
MAX_FRAMES = 48
# 次の位置へ走る速さ（Pi の forward の値）。床が見えている所しか走らないが、ゆっくり行く
SPEED = 12
# ステーションの数の上限（これ以上は探らない）
MAX_STATIONS = 30
# 床・障害物と見る点の、格子 1 マスあたりの最少の数（1 回の撮影で）。
# VGGT の点は外れ値があるので 1 点では決めない
MIN_POINTS_PER_CELL = 3
# 床が見えたセルがこれより少なければ、床の当てはめを信じない
MIN_FREE_CELLS = 30
# 1 か所で首を振ったときのカメラの中心のずれの上限。首振りの軸のずれは数 cm なので、
# これより大きければ VGGT の位置合わせが崩れている
MAX_STATION_SPREAD_M = 0.10

_NAME = re.compile(r"pan([+-]\d+)_tilt([+-]\d+)")


@dataclass
class MapParams:
    # 床からカメラの中心まで。ステーションごとに変えてもよい。None のステーションは
    # 大きさを決めるのに使わず、ほかのステーションで決めた大きさで高さを測る
    # （机の上から撮ったときなど）
    camera_height_m: float | list[float | None] = 0.08  # SD カード 2.5 枚分と測った
    car_height_m: float = 0.12
    floor_tol_m: float = 0.02  # 床と見る高さの幅（±）
    obstacle_margin_m: float = 0.03  # 車の高さに足す余裕
    conf_percentile: float = 50.0  # VGGT の確からしさの下から何 % を捨てるか
    max_range_m: float = 3.0  # ステーションからこれより遠い点は使わない（遠いほど位置がぶれる）
    stride: int = 2  # 深度の画素を何画素おきに使うか
    # 画像の縁の何画素（VGGT の幅 518 で）を使わないか。縁の深度は崩れやすく、下の隅の点が
    # 車のすぐ前の「障害物」になって、どこへも行けなくなった
    border_px: int = 8
    max_stations: int = MAX_STATIONS
    # ステーションごとの前の超音波の読み（cm。None は測れなかった）。そのステーションの位置から
    # 格子に足す（app/roommap.mark_sonar）。VGGT は白い壁や近すぎる物を見落とすことがあり、
    # 1 回目の実走（explore_20261029_210008）では 3 歩とも超音波で止まった
    ultrasonic_cm: list[float | None] | None = None


@dataclass
class FrameRef:
    """VGGT に入れた 1 コマ。station は何番目のステーションか。"""

    path: Path
    station: int
    pan_deg: float
    tilt_deg: float  # 校正済み（tilt_offset_deg を足した）値


@dataclass
class Station:
    dir: Path
    frames: list[dict]  # scan.json の frames（tilt_deg は校正済みに直してある）

    @property
    def name(self) -> str:
        return self.dir.name


def load_station(d: str | Path, scan_file: str = "scan.json") -> Station:
    """撮影の置き場（scan.json と画像）を読む。tilt_offset_deg は tilt に足して校正済みにする。
    HTTP で受けたステーション（app/snoo_map_server.py）は meta.json に置くので、名前を変えられる。
    """
    d = Path(d)
    scan = json.loads((d / scan_file).read_text())
    offset = float(scan.get("tilt_offset_deg") or 0.0)
    frames = []
    for f in scan["frames"]:
        frames.append(
            {**f, "tilt_deg": float(f["tilt_deg"]) + offset, "pan_deg": float(f["pan_deg"])}
        )
    return Station(d, frames)


def select_frames(
    stations: list[Station], max_frames: int = MAX_FRAMES, drop_tilt_above: float = 5.0
) -> list[FrameRef]:
    """VGGT に入れるコマを選ぶ。上向き（tilt > drop_tilt_above）の段は床がほとんど写らないので外し、
    それでも 1 ステーションの割り当て（max_frames ÷ ステーション数）を超えれば、パンを
    1 つおき、2 つおき…に間引く。それでも多ければ tilt 0 から遠い段から外す。
    パン 0 に近いコマ（向きを出すのに使う）は残す。
    """
    budget = max(1, max_frames // max(1, len(stations)))
    refs: list[FrameRef] = []
    for si, st in enumerate(stations):
        frames = [f for f in st.frames if f["tilt_deg"] <= drop_tilt_above] or list(st.frames)
        rows: dict[int, list[dict]] = {}
        for f in frames:
            rows.setdefault(int(round(f["tilt_deg"])), []).append(f)
        for row in rows.values():
            row.sort(key=lambda f: f["pan_deg"])
        k = 1
        while sum(len(_thin(r, k)) for r in rows.values()) > budget and k < 8:
            k += 1
        kept = {t: _thin(r, k) for t, r in rows.items()}
        while sum(len(r) for r in kept.values()) > budget and len(kept) > 1:
            del kept[max(kept, key=lambda t: (abs(t), t))]
        for t in sorted(kept):
            for f in kept[t]:
                refs.append(FrameRef(st.dir / f["file"], si, f["pan_deg"], f["tilt_deg"]))
    return refs


def _thin(row: list[dict], k: int) -> list[dict]:
    """パンの並びを k 個に 1 個にする。パン 0 にいちばん近いコマを必ず残す。"""
    if k <= 1:
        return row
    c = int(np.argmin([abs(f["pan_deg"]) for f in row]))
    return [f for i, f in enumerate(row) if (i - c) % k == 0]


# --- VGGT ---


@dataclass
class VggtOut:
    """VGGT の出力。extrinsic は世界 → カメラ（OpenCV の向き：x 右・y 下・z 前）。
    世界は最初のコマのカメラ。
    """

    extrinsic: np.ndarray  # (F, 3, 4)
    intrinsic: np.ndarray  # (F, 3, 3)
    depth: np.ndarray  # (F, H, W)
    depth_conf: np.ndarray  # (F, H, W)
    files: list[str]


def load_vggt_model():
    """VGGT（facebook/VGGT-1B）を GPU に載せる。読むのに 5〜7 秒かかり、GPU を 5 GB 使う。
    torch・vggt は重いので、使うときだけ読む。
    """
    from vggt.models.vggt import VGGT

    return VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()


def run_vggt(paths: list[Path], model=None) -> VggtOut:
    """画像をまとめて VGGT に通す。GPU を使う。
    model を渡さなければ、その場で読んで、終わったら捨てる（コマンドで 1 回だけ使うとき）。
    サーバーは読んだものを持ち続けて渡す（ステーションごとに読み直すと 5 秒以上かかる）。
    """
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    own = model is None
    if own:
        model = load_vggt_model()
    imgs = load_and_preprocess_images([str(p) for p in paths]).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(imgs)
    ext, K = pose_encoding_to_extri_intri(out["pose_enc"], imgs.shape[-2:])
    result = VggtOut(
        extrinsic=ext[0].float().cpu().numpy(),
        intrinsic=K[0].float().cpu().numpy(),
        depth=out["depth"][0, ..., 0].float().cpu().numpy(),
        depth_conf=out["depth_conf"][0].float().cpu().numpy(),
        files=[str(Path(p).resolve()) for p in paths],
    )
    del out, imgs
    if own:
        del model
    torch.cuda.empty_cache()
    return result


def save_vggt(v: VggtOut, path: Path) -> None:
    np.savez(
        path,
        extrinsic=v.extrinsic,
        intrinsic=v.intrinsic,
        depth=v.depth,
        depth_conf=v.depth_conf,
        files=np.array(v.files),
    )


def load_vggt(path: Path) -> VggtOut:
    """保存した VGGT の出力を読む（前に scratchpad の vggt_scan.py で作ったものも読める）。"""
    d = np.load(path)
    return VggtOut(d["extrinsic"], d["intrinsic"], d["depth"], d["depth_conf"], list(d["files"]))


def match_frames(v: VggtOut, stations: list[Station]) -> list[FrameRef]:
    """保存した VGGT の出力のコマを、ステーション（置き場の名前）と scan.json のコマに
    結びつける。
    """
    by_name = {st.name: (i, {f["file"]: f for f in st.frames}) for i, st in enumerate(stations)}
    refs = []
    for p in v.files:
        p = Path(p)
        if p.parent.name not in by_name:
            raise ValueError(f"{p} はどのステーションの画像でもない")
        si, frames = by_name[p.parent.name]
        f = frames[p.name]
        refs.append(FrameRef(stations[si].dir / p.name, si, f["pan_deg"], f["tilt_deg"]))
    return refs


def unproject(v: VggtOut, stride: int = 2, border_px: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """深度とカメラから、各コマの点（VGGT の世界座標、(F, h, w, 3)）と
    確からしさ（(F, h, w)）を作る。
    VGGT の論文どおり、点の頭（world_points）より深度 + カメラのほうが正確なのでこちらを使う。
    画像の縁 border_px の点は確からしさを 0 にする（使わない）。
    """
    F, H, W = v.depth.shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride].astype(np.float32)
    pts = np.empty((F, *uu.shape, 3), np.float32)
    for i in range(F):
        K, E = v.intrinsic[i], v.extrinsic[i]
        z = v.depth[i, ::stride, ::stride]
        cam = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], -1)
        pts[i] = (cam - E[:, 3]) @ E[:, :3]  # R^T (cam - t)
    conf = v.depth_conf[:, ::stride, ::stride].copy()
    edge = (uu < border_px) | (uu >= W - border_px) | (vv < border_px) | (vv >= H - border_px)
    conf[:, edge] = 0
    return pts, conf


def frame_colors(paths: list[Path], shape: tuple[int, int], stride: int = 2) -> np.ndarray:
    """点の色（BGR、(F, h, w, 3)）。VGGT の前処理と同じく幅 518 に縮めた画像から取る。"""
    H, W = shape
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            img = np.zeros((H, W, 3), np.uint8)
        out.append(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)[::stride, ::stride])
    return np.stack(out)


# --- 床と座標 ---


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    R, t = extrinsic[:, :, :3], extrinsic[:, :, 3]
    return -np.einsum("fji,fj->fi", R, t)


def up_in_world(extrinsic: np.ndarray, tilt_deg: np.ndarray) -> np.ndarray:
    """各コマの「上」（床の法線の見込み）を VGGT の世界座標で。上を tilt° 向いたカメラでは、
    上はカメラ座標で (0, -cos tilt, sin tilt)。
    """
    t = np.radians(tilt_deg)
    up_cam = np.stack([np.zeros_like(t), -np.cos(t), np.sin(t)], -1)
    return np.einsum("fji,fj->fi", extrinsic[:, :, :3], up_cam)


@dataclass
class Floor:
    normal: np.ndarray  # 上向きの単位ベクトル（VGGT の座標）
    offset: float  # 床からの高さ = normal・p + offset
    inliers: int
    candidates: int

    def height(self, p: np.ndarray) -> np.ndarray:
        return p @ self.normal + self.offset


def fit_floor(
    points: np.ndarray,
    up_prior: np.ndarray,
    cams: np.ndarray,
    max_tilt_deg: float = 25.0,
    iters: int = 800,
    rng: np.random.Generator | None = None,
) -> Floor | None:
    """点から床の平面を RANSAC で探す。向きが up_prior から max_tilt_deg 以内で、
    カメラがみな上にあるものだけ。
    大きさが決まっていないので、許す外れはカメラから点までの距離の中央値の 0.5 % にする
    （カメラ 10 cm・点まで 2 m なら 1 cm）。見つけた後、カメラの高さの 20 % 以内の点で当て直す。
    """
    rng = rng or np.random.default_rng(0)
    up_prior = up_prior / np.linalg.norm(up_prior)
    if len(points) > 150_000:
        points = points[rng.choice(len(points), 150_000, replace=False)]
    if len(points) < 3:
        return None
    ref = float(np.median(np.linalg.norm(points - cams.mean(0), axis=1)))
    thr = 0.005 * ref
    cos_max = math.cos(math.radians(max_tilt_deg))
    tri = points[rng.integers(0, len(points), (iters, 3))]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    ok = norm > 1e-12
    normals[ok] /= norm[ok, None]
    normals *= np.where(normals @ up_prior < 0, -1.0, 1.0)[:, None]
    ok &= normals @ up_prior > cos_max
    best = None
    for i in np.flatnonzero(ok):
        n = normals[i]
        off = -float(n @ tri[i, 0])
        if (cams @ n + off).min() <= 0:  # カメラより上の平面（天板の裏など）は床ではない
            continue
        count = int((np.abs(points @ n + off) < thr).sum())
        if best is None or count > best[0]:
            best = (count, n, off)
    if best is None:
        return None
    _, n, off = best
    for _ in range(3):
        h = float(np.median(cams @ n + off))
        inl = points[np.abs(points @ n + off) < max(thr, 0.2 * h)]
        if len(inl) < 3:
            break
        c = inl.mean(0)
        n2 = np.linalg.svd(inl - c, full_matrices=False)[2][2]
        n = n2 if n2 @ n > 0 else -n2
        off = -float(n @ c)
    h = float(np.median(cams @ n + off))
    count = int((np.abs(points @ n + off) < max(thr, 0.2 * h)).sum())
    return Floor(n, off, count, len(points))


@dataclass
class WorldFrame:
    """VGGT の座標 → 間取りの座標（m。原点は最初のステーションの真下の床、x 前・y 左・z 上）。"""

    rotation: np.ndarray  # 行が x・y・z の軸（VGGT の座標）
    origin: np.ndarray  # VGGT の座標
    scale: float  # m / VGGT の長さ

    def apply(self, p: np.ndarray) -> np.ndarray:
        return self.scale * ((p - self.origin) @ self.rotation.T)

    def direction(self, d: np.ndarray) -> np.ndarray:
        return d @ self.rotation.T


def _rotate_about(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """v を axis のまわりに angle（右ねじ）回す（Rodrigues）。"""
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def heading_frame(refs: list[FrameRef], station: int) -> int:
    """そのステーションで車の向きを出すのに使うコマ。
    tilt 0 に近い段の、パンがいちばん 0 に近いもの。
    """
    idx = [i for i, r in enumerate(refs) if r.station == station]
    return min(idx, key=lambda i: (round(abs(refs[i].tilt_deg)), abs(refs[i].pan_deg)))


def camera_heading(extrinsic: np.ndarray, floor: Floor, pan_deg: float) -> np.ndarray:
    """そのコマから見た車の向き（床に沿う単位ベクトル、VGGT の座標）。パンは右が + なので、
    光軸を上向きの軸のまわりに左回りへパンの分だけ戻す。
    """
    fwd = extrinsic[2, :3]  # R の 3 行目 = 光軸（世界座標）
    fwd = fwd - (fwd @ floor.normal) * floor.normal
    fwd = _rotate_about(fwd, floor.normal, math.radians(pan_deg))
    return fwd / np.linalg.norm(fwd)


def make_world(
    floor: Floor,
    extrinsic: np.ndarray,
    refs: list[FrameRef],
    heights_m: list[float | None],
) -> tuple[WorldFrame, np.ndarray]:
    """床と最初のステーションから座標を決める。大きさは、ステーションごとのカメラの中心（コマの平均）の
    床からの高さが heights_m に合うよう最小二乗で決める（None のステーションは使わない）。
    返すのは座標と、ステーションごとの高さ（VGGT）。
    """
    cams = camera_centers(extrinsic)
    n_st = max(r.station for r in refs) + 1
    centers = np.stack([cams[[r.station == s for r in refs]].mean(0) for s in range(n_st)])
    h = floor.height(centers)
    known = np.array([x is not None for x in heights_m])
    if not known.any():
        raise ValueError("カメラの高さが分かっているステーションが 1 つも無い")
    H = np.array([x or 0.0 for x in heights_m])[known]
    scale = float((h[known] * H).sum() / (h[known] ** 2).sum())
    k = heading_frame(refs, 0)
    ex = camera_heading(extrinsic[k], floor, refs[k].pan_deg)
    ez = floor.normal / np.linalg.norm(floor.normal)
    ey = np.cross(ez, ex)
    origin = centers[0] - h[0] * ez
    return WorldFrame(np.stack([ex, ey, ez]), origin, scale), h


# --- 地図 ---


@dataclass
class StationResult:
    name: str
    pose: Pose
    height_m: float  # 床からカメラまで（求めたもの）
    spread_m: float  # 首を振ったときのカメラの中心のばらつき（最大）
    frames: int
    floor_points: int = 0
    obstacle_points: int = 0
    high_points: int = 0


@dataclass
class MapResult:
    grid: Grid
    high: np.ndarray  # (n, n) の int。「上」の層の点が見つかった回数
    stations: list[StationResult]
    floor: Floor
    world: WorldFrame
    action: dict
    plan: Plan | None
    params: MapParams
    # 上から見た絵に使う点（間取りの座標、m）と色（BGR）。間引いてある
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))


def _cells_with(grid: Grid, x: np.ndarray, y: np.ndarray, min_count: int):
    """min_count 点以上あるセルの中心を返す（外れ値の 1 点でセルを決めない）。"""
    ix, iy = grid.cell(x, y)
    ok = grid.inside(ix, iy)
    if not ok.any():
        return np.zeros(0), np.zeros(0)
    cells, counts = np.unique(np.stack([ix[ok], iy[ok]], 1), axis=0, return_counts=True)
    cells = cells[counts >= min_count]
    return grid.center(cells[:, 0], cells[:, 1])


def build_map(
    stations: list[Station],
    refs: list[FrameRef],
    extrinsic: np.ndarray,
    points: np.ndarray,
    conf: np.ndarray,
    params: MapParams | None = None,
    colors: np.ndarray | None = None,
) -> MapResult:
    """VGGT の点（コマごと、(F, ..., 3)）から床・座標・ステーションの位置・格子を作り、
    次の 1 歩を決める。
    GPU は使わない（VGGT の出力を差し替えて試せるよう、ここは numpy だけ）。
    """
    params = params or MapParams()
    F = len(refs)
    pts = points.reshape(F, -1, 3)
    cf = conf.reshape(F, -1)
    cols = colors.reshape(F, -1, 3) if colors is not None else None
    n_st = len(stations)
    heights = params.camera_height_m
    heights = list(heights) if isinstance(heights, list | tuple) else [heights] * n_st
    if len(heights) != n_st:
        raise ValueError("camera_height_m の数がステーションの数と合わない")
    if params.ultrasonic_cm is not None and len(params.ultrasonic_cm) != n_st:
        raise ValueError("ultrasonic_cm の数がステーションの数と合わない")

    finite = np.isfinite(pts).all(-1) & (cf > 0)
    good = finite & (cf >= np.percentile(cf[finite], params.conf_percentile))
    # 床の向きの見込みは最初のステーションのコマから（tilt が校正済みの、床の高さのステーション）
    first = np.array([r.station == 0 for r in refs])
    up = up_in_world(extrinsic[first], np.array([r.tilt_deg for r in refs])[first]).mean(0)
    cams = camera_centers(extrinsic)
    floor = fit_floor(pts[good], up, cams)
    if floor is None:
        return _failed(stations, refs, params, "no_floor")
    world, h_vggt = make_world(floor, extrinsic, refs, heights)

    cams_w = world.apply(cams)
    grid = Grid()
    high = np.zeros_like(grid.logodds, np.int32)
    top = params.car_height_m + params.obstacle_margin_m
    results: list[StationResult] = []
    keep_pts, keep_cols = [], []
    rng = np.random.default_rng(0)
    for s, st in enumerate(stations):
        idx = [i for i, r in enumerate(refs) if r.station == s]
        c = cams_w[idx]
        center = c.mean(0)
        k = heading_frame(refs, s)
        hd = world.direction(camera_heading(extrinsic[k], floor, refs[k].pan_deg))
        pose = Pose(float(center[0]), float(center[1]), math.atan2(hd[1], hd[0]))
        res = StationResult(
            st.name,
            pose,
            float(world.scale * h_vggt[s]),
            float(np.linalg.norm(c - center, axis=1).max()),
            len(idx),
        )
        results.append(res)
        if registration_is_bad(res, heights[s]):
            continue  # 位置の合っていない点は地図に入れない（decide で止める）
        p = world.apply(pts[idx][good[idx]])
        near = np.hypot(p[:, 0] - pose.x, p[:, 1] - pose.y) <= params.max_range_m
        p = p[near]
        z = p[:, 2]
        is_floor = np.abs(z) <= params.floor_tol_m
        is_obst = (z > params.floor_tol_m) & (z <= top)
        is_high = z > top
        fx, fy = _cells_with(grid, p[is_floor, 0], p[is_floor, 1], MIN_POINTS_PER_CELL)
        grid.mark_free(fx, fy)
        ox, oy = _cells_with(grid, p[is_obst, 0], p[is_obst, 1], MIN_POINTS_PER_CELL)
        grid.mark_occupied(ox, oy)
        hx, hy = _cells_with(grid, p[is_high, 0], p[is_high, 1], MIN_POINTS_PER_CELL)
        hix, hiy = grid.cell(hx, hy)
        high[hix, hiy] += 1
        if params.ultrasonic_cm is not None and params.ultrasonic_cm[s] is not None:
            mark_sonar(grid, pose, params.ultrasonic_cm[s] / 100)
        res.floor_points, res.obstacle_points = int(is_floor.sum()), int(is_obst.sum())
        res.high_points = int(is_high.sum())
        if cols is not None:
            sel = rng.choice(len(p), min(len(p), 60_000), replace=False)
            keep_pts.append(p[sel])
            keep_cols.append(cols[idx][good[idx]][near][sel])

    action, plan = decide(grid, results, params, heights)
    return MapResult(
        grid,
        high,
        results,
        floor,
        world,
        action,
        plan,
        params,
        np.concatenate(keep_pts) if keep_pts else np.zeros((0, 3), np.float32),
        np.concatenate(keep_cols) if keep_cols else np.zeros((0, 3), np.uint8),
    )


def _failed(stations, refs, params, reason) -> MapResult:
    grid = Grid()
    return MapResult(
        grid,
        np.zeros_like(grid.logodds, np.int32),
        [StationResult(st.name, Pose(0, 0, 0), 0.0, 0.0, 0) for st in stations],
        Floor(np.array([0.0, 0.0, 1.0]), 0.0, 0, 0),
        WorldFrame(np.eye(3), np.zeros(3), 1.0),
        {"action": "done", "reason": reason},
        None,
        params,
    )


def pose_dict(p: Pose) -> dict:
    # + 0.0 で -0.0 を 0.0 にする
    return {
        "x": round(p.x, 3) + 0.0,
        "y": round(p.y, 3) + 0.0,
        "yaw_deg": round(math.degrees(p.yaw), 1) + 0.0,
    }


def registration_is_bad(st: StationResult, height_m: float | None) -> bool:
    """首を振っただけでカメラが大きく動いた・床からの高さが見込みと大きく違う
    → VGGT の位置合わせが崩れている。
    """
    if st.spread_m > MAX_STATION_SPREAD_M:
        return True
    return height_m is not None and abs(st.height_m - height_m) > 0.5 * height_m


def decide(
    grid: Grid, stations: list[StationResult], params: MapParams, heights: list[float | None]
) -> tuple[dict, Plan | None]:
    """最後のステーションから次の 1 歩を決める。危なそうなら止める（done）。"""
    if len(stations) >= params.max_stations:
        return {"action": "done", "reason": "step_limit"}, None
    for st, H in zip(stations, heights, strict=True):
        if registration_is_bad(st, H):
            return {"action": "done", "reason": f"bad_registration:{st.name}"}, None
    if int((grid.state() == FREE).sum()) < MIN_FREE_CELLS:
        return {"action": "done", "reason": "too_little_floor"}, None
    pose = stations[-1].pose
    if not frontiers(grid.state()).any():
        return {"action": "done", "reason": "no_frontier"}, None
    plan = next_best_view(grid, pose)
    if plan is None:  # 境目はあるが、見えている床を通って行ける所が無い
        return {"action": "done", "reason": "unreachable"}, None
    return {
        "action": "move",
        "steer_deg": round(plan.steer_deg, 1),
        "speed": SPEED,
        "distance_cm": round(plan.step_m * 100, 1),
        "pose": pose_dict(pose),
        "target": pose_dict(plan.target),
    }, plan


# --- 書き出し ---

# 格子の PNG の画素値（ROS の map_server に近い：白 = 走れる、黒 = ふさがり、灰 = 見ていない）
PNG_VALUE = {UNKNOWN: 128, FREE: 255, OCCUPIED: 0}


def grid_png(state: np.ndarray) -> np.ndarray:
    """格子 [ix, iy] を PNG の向き（行 0 が +y の端、列 0 が -x の端）にする。"""
    img = np.full(state.shape, PNG_VALUE[UNKNOWN], np.uint8)
    img[state == FREE] = PNG_VALUE[FREE]
    img[state == OCCUPIED] = PNG_VALUE[OCCUPIED]
    return img.T[::-1]


def room_json(r: MapResult) -> dict:
    g = r.grid
    legend = {
        "pixel": "行 0 が +y の端、列 0 が -x の端。1 画素 = 1 セル",
        "values": {str(v): k for k, v in (("unknown", 128), ("free", 255), ("occupied", 0))},
    }
    return {
        "schema_version": 1,
        "resolution_cm": round(g.res * 100, 3),
        "origin": [g.origin, g.origin],
        "size": [g.n, g.n],
        "start_pose": pose_dict(Pose(0, 0, 0)),
        "stations": [
            {
                "name": s.name,
                "pose": pose_dict(s.pose),
                "camera_height_m": round(s.height_m, 3),
                "frames": s.frames,
            }
            for s in r.stations
        ],
        "layers": {
            "static": {"png": "grid.png", "legend": legend},
            "high": {
                "png": "high.png",
                "legend": {
                    "pixel": legend["pixel"],
                    "values": {"255": "high", "0": "none"},
                    "min_height_m": round(r.params.car_height_m + r.params.obstacle_margin_m, 3),
                },
            },
            "movable": [],
            "occluders": [],
            "snoo_spots": [],
        },
        "scale_m_per_unit": round(r.world.scale, 5),
        "next": r.action,
    }


def plan_json(r: MapResult) -> dict:
    out = dict(r.action)
    if r.plan is not None:
        out["plan"] = {
            "target": pose_dict(r.plan.target),
            "gain_cells": r.plan.gain,
            "length_m": round(r.plan.length_m, 3),
            "path": [[round(p.x, 3), round(p.y, 3)] for p in r.plan.path],
        }
    return out


def write_outputs(r: MapResult, out: Path) -> None:
    """room.json・grid.png・high.png・plan.json・topdown.png を書く。"""
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "grid.png"), grid_png(r.grid.state()))
    cv2.imwrite(str(out / "high.png"), np.where(r.high > 0, 255, 0).astype(np.uint8).T[::-1])
    (out / "room.json").write_text(json.dumps(room_json(r), ensure_ascii=False, indent=1))
    (out / "plan.json").write_text(json.dumps(plan_json(r), ensure_ascii=False, indent=1))
    cv2.imwrite(str(out / "topdown.png"), render_topdown(r))


def render_topdown(r: MapResult, px_per_cell: int = 6, margin_m: float = 0.5) -> np.ndarray:
    """上から見た絵。左は格子（白 = 床、黒 = 障害物、青 = 上の層、灰 = 見ていない）に、
    ステーション（丸と向き）、道のり（緑）、次の位置（赤）を描く。右は点の色を上から
    見たもの。格子の線は 1 m ごと。
    """
    g = r.grid
    state = g.state()
    known = np.argwhere((state != UNKNOWN) | (r.high > 0))
    st_cells = [g.cell(s.pose.x, s.pose.y) for s in r.stations]
    pts = np.array([[int(a), int(b)] for a, b in st_cells] + known.tolist() or [[g.n // 2] * 2])
    m = int(margin_m / g.res)
    x0, y0 = np.maximum(pts.min(0) - m, 0)
    x1, y1 = np.minimum(pts.max(0) + m + 1, g.n)

    def to_px(x, y):
        ix = (x - g.origin) / g.res - x0
        iy = (y - g.origin) / g.res - y0
        return int(round(ix * px_per_cell)), int(round((y1 - y0 - iy) * px_per_cell))

    sub = state[x0:x1, y0:y1]
    left = np.full((*sub.shape, 3), 150, np.uint8)
    left[sub == FREE] = (235, 235, 235)
    left[sub == OCCUPIED] = (30, 30, 30)
    hi = r.high[x0:x1, y0:y1] > 0
    left[hi & (sub != OCCUPIED)] = (200, 140, 90)
    left = np.ascontiguousarray(left.transpose(1, 0, 2)[::-1])
    left = cv2.resize(left, None, fx=px_per_cell, fy=px_per_cell, interpolation=cv2.INTER_NEAREST)

    # 点の色：セルごとに床の点（高さ 0〜車の高さ）の色の平均
    right = np.zeros_like(left)
    if len(r.points):
        p, c = r.points, r.colors.astype(np.float32)
        low = p[:, 2] <= r.params.car_height_m + r.params.obstacle_margin_m
        ix, iy = g.cell(p[low, 0], p[low, 1])
        ok = (ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)
        acc = np.zeros((x1 - x0, y1 - y0, 3), np.float32)
        cnt = np.zeros((x1 - x0, y1 - y0), np.float32)
        np.add.at(acc, (ix[ok] - x0, iy[ok] - y0), c[low][ok])
        np.add.at(cnt, (ix[ok] - x0, iy[ok] - y0), 1)
        img = (acc / np.maximum(cnt, 1)[..., None]).astype(np.uint8)
        img = np.ascontiguousarray(img.transpose(1, 0, 2)[::-1])
        right = cv2.resize(
            img, None, fx=px_per_cell, fy=px_per_cell, interpolation=cv2.INTER_NEAREST
        )

    for canvas in (left, right):
        for xm in range(math.ceil(g.origin + x0 * g.res), math.floor(g.origin + x1 * g.res) + 1):
            u, _ = to_px(xm, 0)
            cv2.line(canvas, (u, 0), (u, canvas.shape[0]), (120, 180, 120), 1)
        for ym in range(math.ceil(g.origin + y0 * g.res), math.floor(g.origin + y1 * g.res) + 1):
            _, v = to_px(0, ym)
            cv2.line(canvas, (0, v), (canvas.shape[1], v), (120, 180, 120), 1)
        if r.plan is not None:
            path = np.array([to_px(p.x, p.y) for p in r.plan.path], np.int32)
            cv2.polylines(canvas, [path], False, (0, 170, 0), 2)
            t = r.plan.target
            a, b = (
                to_px(t.x, t.y),
                to_px(t.x + 0.25 * math.cos(t.yaw), t.y + 0.25 * math.sin(t.yaw)),
            )
            cv2.circle(canvas, a, 6, (0, 0, 220), 2)
            cv2.arrowedLine(canvas, a, b, (0, 0, 220), 2, tipLength=0.3)
        for i, s in enumerate(r.stations):
            p = s.pose
            a, b = (
                to_px(p.x, p.y),
                to_px(p.x + 0.25 * math.cos(p.yaw), p.y + 0.25 * math.sin(p.yaw)),
            )
            cv2.circle(canvas, a, 7, (220, 120, 0), 2)
            cv2.arrowedLine(canvas, a, b, (220, 120, 0), 2, tipLength=0.3)
            cv2.putText(canvas, str(i + 1), (a[0] + 8, a[1] - 8), 0, 0.5, (220, 120, 0), 1)
    sep = np.full((left.shape[0], 4, 3), 255, np.uint8)
    img = np.concatenate([left, sep, right], 1)
    text = f"1 grid = 1 m  next: {r.action.get('action')} {r.action.get('reason', '')}"
    cv2.putText(img, text, (6, 18), 0, 0.5, (0, 0, 200), 1)
    return img
