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
    MAX_SEGMENTS,
    MIN_STEP_M,
    OCCUPIED,
    REVERSE_MAX_M,
    UNKNOWN,
    Grid,
    Plan,
    Pose,
    camera_of,
    drive_poses,
    frontiers,
    mark_sonar,
    next_best_view,
    random_step,
    retrace,
    turning_reverse,
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
    # 1 歩を「曲がる → まっすぐ」の区間（segments）で返す。Pi が meta の supports に
    # "segments" を書いたときだけ（古い Pi は 1 本の弧しか走れない）
    segments: bool = False
    # 行き場が無いときに後ろへ下がってよい（Pi が supports に "reverse" を書いたとき。
    # 下がり方は reverse_step で決める）
    reverse: bool = False
    # 走ってきた道（車の外形の下）を床にする（roommap.mark_driven。地図のサーバーと
    # シミュレーターが間取りに足す）
    driven_free: bool = True
    # 次の 1 歩を、この割合ででたらめな 1 歩にする（ルンバのように行き詰まりから抜け出す）
    random_p: float = 0.15
    # 同じ所に固まっていれば、必ずでたらめな 1 歩にする（地図のサーバーが決める）
    force_random: bool = False
    seed: int = 0


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
    stations: list[Station],
    max_frames: int = MAX_FRAMES,
    drop_tilt_above: float = 5.0,
    old_frames: int | None = None,
) -> list[FrameRef]:
    """VGGT に入れるコマを選ぶ。上向き（tilt > drop_tilt_above）の段は床がほとんど写らないので外し、
    それでも 1 ステーションの割り当て（max_frames ÷ ステーション数）を超えれば、パンを
    1 つおき、2 つおき…に間引く。それでも多ければ tilt 0 から遠い段から外す。
    パン 0 に近いコマ（向きを出すのに使う）は残す。

    old_frames を渡すと、最後でないステーションは old_frames コマまで、最後のステーションは
    残り（max_frames - 前の分）までにする。前のステーションは、位置がもう決まっていて、
    最後のステーションを同じ座標に置くための手がかりでしかない。見たものも前に残してある
    （obs.npz）。最後のステーションのコマを減らさず、VGGT に入れるコマを減らせる。
    """
    n = max(1, len(stations))
    budgets = [max(1, max_frames // n)] * n
    if old_frames is not None:
        budgets = [old_frames] * (n - 1) + [max(1, max_frames - old_frames * (n - 1))]
    refs: list[FrameRef] = []
    for si, st in enumerate(stations):
        budget = budgets[si]
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
class Observation:
    """1 ステーションで見たもの。セッションの座標（最初のステーションが原点）の格子のセルの中心
    (K, 2)。地図のサーバーはステーションごとに 1 回だけ（そのステーションが最後のとき）作って
    置き場に残し、全部のステーションの分を足してセッションの間取りにする（session_grid）。
    窓（WINDOW か所）だけで間取りを作ると、窓の外で見た所を忘れて、そこへまた行こうとする。
    """

    free: np.ndarray
    occupied: np.ndarray
    high: np.ndarray
    pose: Pose  # ステーションの位置（セッションの座標）
    sonar_m: float | None = None

    def save(self, path: Path) -> None:
        p = self.pose
        np.savez(
            path,
            free=self.free.astype(np.float32),
            occupied=self.occupied.astype(np.float32),
            high=self.high.astype(np.float32),
            pose=np.array([p.x, p.y, p.yaw]),
            sonar_m=np.array(np.nan if self.sonar_m is None else self.sonar_m),
        )

    @classmethod
    def load(cls, path: Path) -> Observation:
        d = np.load(path)
        sonar = float(d["sonar_m"])
        return cls(
            d["free"], d["occupied"], d["high"], Pose(*map(float, d["pose"])),
            None if math.isnan(sonar) else sonar,
        )  # fmt: skip


def session_grid(observations: list[Observation]) -> tuple[Grid, np.ndarray]:
    """ステーションごとに見たものを順に足した格子と「上」の層（build_map と同じ足し方）。"""
    grid = Grid()
    high = np.zeros_like(grid.logodds, np.int32)
    for o in observations:
        grid.mark_free(o.free[:, 0], o.free[:, 1])
        grid.mark_occupied(o.occupied[:, 0], o.occupied[:, 1])
        hix, hiy = grid.cell(o.high[:, 0], o.high[:, 1])
        ok = grid.inside(hix, hiy)
        high[hix[ok], hiy[ok]] += 1
        if o.sonar_m is not None:
            mark_sonar(grid, o.pose, o.sonar_m)
    return grid, high


def to_frame(anchor: Pose, p: Pose) -> Pose:
    """anchor を原点にした座標の位置を、anchor のある座標に直す（窓 → セッション）。"""
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    yaw = anchor.yaw + p.yaw
    return Pose(
        anchor.x + c * p.x - s * p.y,
        anchor.y + s * p.x + c * p.y,
        math.atan2(math.sin(yaw), math.cos(yaw)),
    )


def xy_to_frame(anchor: Pose, xy: np.ndarray) -> np.ndarray:
    """点 (N, 2 以上) の x・y を to_frame と同じに直す（3 列目からはそのまま）。"""
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    out = np.array(xy, dtype=np.float64, copy=True)
    out[:, 0] = anchor.x + c * xy[:, 0] - s * xy[:, 1]
    out[:, 1] = anchor.y + s * xy[:, 0] + c * xy[:, 1]
    return out.astype(np.asarray(xy).dtype)


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
    # 次の 1 歩で車（カメラの真下）が通る所。上から見た絵に描く（後ろへ戻るときはサーバーが入れる）
    step_path: list[Pose] = field(default_factory=list)
    # build_map に anchor を渡したときの、ステーションごとに見たもの（セッションの座標）。
    # 位置合わせが崩れたステーションは None
    observations: list[Observation | None] = field(default_factory=list)


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
    anchor: Pose | None = None,
    plan: bool = True,
) -> MapResult:
    """VGGT の点（コマごと、(F, ..., 3)）から床・座標・ステーションの位置・格子を作り、
    次の 1 歩を決める（plan が False なら決めない。地図のサーバーはセッションの間取りで決める）。
    anchor（最初のステーションのセッションの座標での位置）を渡すと、ステーションごとに見たものを
    セッションの座標の格子のセルにして observations に入れる（点を直してからセルにする。
    セルの中心を回すと、セルの抜けや重なりができる）。
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
    observations: list[Observation | None] = []
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
        observations.append(None)
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
        sonar = None
        if params.ultrasonic_cm is not None and params.ultrasonic_cm[s] is not None:
            sonar = params.ultrasonic_cm[s] / 100
            mark_sonar(grid, pose, sonar)
        if anchor is not None:
            q = xy_to_frame(anchor, p)
            cells = []
            for sel in (is_floor, is_obst, is_high):
                cx, cy = _cells_with(grid, q[sel, 0], q[sel, 1], MIN_POINTS_PER_CELL)
                cells.append(np.stack([cx, cy], 1).reshape(-1, 2))
            observations[-1] = Observation(*cells, to_frame(anchor, pose), sonar)
        res.floor_points, res.obstacle_points = int(is_floor.sum()), int(is_obst.sum())
        res.high_points = int(is_high.sum())
        if cols is not None:
            sel = rng.choice(len(p), min(len(p), 60_000), replace=False)
            keep_pts.append(p[sel])
            keep_cols.append(cols[idx][good[idx]][near][sel])

    action, nbv = decide(grid, results, params, heights) if plan else ({}, None)
    step_path = []
    if nbv is not None:
        pieces = nbv.segments or [(nbv.steer_deg, nbv.step_m)]
        step_path = [camera_of(p) for _, p in drive_poses(results[-1].pose, pieces)]
    return MapResult(
        grid,
        high,
        results,
        floor,
        world,
        action,
        nbv,
        params,
        np.concatenate(keep_pts) if keep_pts else np.zeros((0, 3), np.float32),
        np.concatenate(keep_cols) if keep_cols else np.zeros((0, 3), np.uint8),
        step_path,
        observations,
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
    rng = np.random.default_rng(params.seed)
    plan = None
    if not params.force_random and rng.random() >= params.random_p:
        plan = next_best_view(grid, pose, segments=params.segments)
    if plan is None:
        # 行き詰まった・同じ所に固まった・ときどき：でたらめな 1 歩
        step = random_step(grid, pose, rng)
        if step is None:  # 境目はあるが、見えている床を通って行ける所が無い
            return {"action": "done", "reason": "unreachable"}, None
        steer, run = step
        action = {
            "action": "move",
            "steer_deg": round(steer, 1),
            "speed": SPEED,
            "distance_cm": round(run * 100, 1),
            "pose": pose_dict(pose),
            "random": True,
        }
        if params.segments:
            action["segments"] = segments_json([(steer, run)])
        return action, None
    action = {
        "action": "move",
        "steer_deg": round(plan.steer_deg, 1),
        "speed": SPEED,
        "distance_cm": round(plan.step_m * 100, 1),
        "pose": pose_dict(pose),
        "target": pose_dict(plan.target),
    }
    if params.segments:
        # 上の steer_deg・distance_cm は最初の区間（segments を知らない Pi でも同じに動く）
        action["segments"] = segments_json(plan.segments)
    return action, plan


def segments_json(pieces: list[tuple[float, float]]) -> list[dict]:
    return [{"steer_deg": round(st, 1) + 0.0, "distance_cm": round(m * 100, 1)} for st, m in pieces]


# --- 行き場が無いときに後ろへ下がる（地図のサーバーとシミュレーターで同じ決め方にする） ---

# 行き場が無いときに来た道を戻るのは、続けてこの回数まで
MAX_REVERSES = 2
# 間に前へ走っても、直近 RECENT_STATIONS か所で戻るのはこの回数まで。戻る → 前へ → 戻るを
# 繰り返して同じ所を回り続けないため（シミュレーターで周の多くを使った）
MAX_RECENT_REVERSES = 3
RECENT_STATIONS = 6
# 後ろにはセンサーが無く、シミュレーターでは後ろに来た snoo に当たることがあった。
# 当たっても弱くなるよう、前（12）より遅く戻る（約 14 cm/s）
REVERSE_SPEED = 8
# 直近 CLUSTER_STATIONS か所が、この半径に固まっていたら、でたらめな 1 歩で抜け出す
CLUSTER_STATIONS = 4
CLUSTER_RADIUS_M = 0.30


# 1 回に VGGT に入れるステーションの数。48 コマを 6 か所で分けると 1 か所 8 コマ（下向きと
# 正面の 2 段 × パン 4 つ）。これより多いと 1 か所のコマが少なすぎて向きと位置がぶれる
WINDOW = 6


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


@dataclass
class History:
    """前のステーションまでの走り方。地図のサーバーは置き場（meta の prev_step・reverse.json）
    から、シミュレーターは覚えたものから作る。
    """

    # 前へ走った道（走った順の区間 (ハンドル°, m)）。来た道を戻るのに使う
    trail: list[tuple[float, float]] = field(default_factory=list)
    backs: int = 0  # 続けて後ろへ戻った回数
    recent_reverses: int = 0  # 直近 RECENT_STATIONS か所のうち、後ろへ戻って着いた数


def clustered(recent: list[Pose]) -> bool:
    """直近のステーション（CLUSTER_STATIONS か所）が狭い所に固まっているか。"""
    if len(recent) < CLUSTER_STATIONS:
        return False
    cx = sum(p.x for p in recent) / len(recent)
    cy = sum(p.y for p in recent) / len(recent)
    return all(math.hypot(p.x - cx, p.y - cy) <= CLUSTER_RADIUS_M for p in recent)


def retrace_step(history: History, segments: bool) -> list[tuple[float, float]]:
    """来た道を戻る区間（戻る順）。続けて戻りすぎていれば・戻れる長さが無ければ空。

    前にしか走らないと、壁に向いて止まったときに回れずに終わる（1 回目の実走の 4 か所目。
    シミュレーターでも角で 74 % で止まり、30 cm 下がると 95 % まで調べられた）。
    segments を受けない Pi には、最後に走った区間の 1 本だけを返す。
    """
    if history.backs >= MAX_REVERSES or history.recent_reverses >= MAX_RECENT_REVERSES:
        return []
    pieces = retrace(history.trail, REVERSE_MAX_M, MAX_SEGMENTS if segments else 1)
    return pieces if sum(m for _, m in pieces) >= MIN_STEP_M else []


def reverse_step(
    grid: Grid, pose: Pose, history: History, segments: bool
) -> list[tuple[float, float]]:
    """前へ行ける所が無い（unreachable）ときに後ろへ下がる区間。下がれなければ空。

    まず切り返し（曲がりながら下がって向きを変える）、だめなら来た道を戻る。
    後ろは見ていないので 30 cm 下がれなければ、見ていない所に入る 15 cm だけ下がる。
    """
    pieces: list[tuple[float, float]] = []
    if history.recent_reverses < MAX_RECENT_REVERSES:
        pieces = turning_reverse(grid, pose) or turning_reverse(grid, pose, length=0.15)
    return pieces or retrace_step(history, segments)


def reverse_action(pieces: list[tuple[float, float]], segments: bool) -> dict:
    action = {
        "action": "move",
        "reverse": True,
        "steer_deg": round(pieces[0][0], 1) + 0.0,
        "speed": REVERSE_SPEED,
        "distance_cm": round(pieces[0][1] * 100, 1),
    }
    if segments:
        action["segments"] = segments_json(pieces)
    return action


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
    ステーション（丸と向き）、道のり（緑）、次の位置（赤）、次の 1 歩（紫）を描く。
    右は点の色を上から見たもの。格子の線は 1 m ごと。
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
        if len(r.step_path) > 1:  # 次の 1 歩（Pi が走る所）
            step = np.array([to_px(p.x, p.y) for p in r.step_path], np.int32)
            cv2.polylines(canvas, [step], False, (200, 0, 200), 3)
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
    what = "reverse" if r.action.get("reverse") else r.action.get("reason", "")
    text = f"1 grid = 1 m  next: {r.action.get('action')} {what}"
    cv2.putText(img, text, (6, 18), 0, 0.5, (0, 0, 200), 1)
    return img
