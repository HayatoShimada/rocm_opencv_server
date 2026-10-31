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
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.roommap import (
    CAR_HEIGHT_M,
    FREE,
    MAX_SEGMENTS,
    MAX_STEP_M,
    OCCUPIED,
    UNKNOWN,
    Grid,
    Pose,
    big_frontiers,
    camera_of,
    drive_moves,
    mark_sonar,
    mark_station,
    next_best_view,
)
from app.snoo_belief import (
    CONF_DONE_MAX_CELLS,
    LANDMARK_TEX,
    Confidence,
    Texture,
    confidence_of,
    confidence_png,
    texture_map,
    texture_png,
    unsure_edges,
)
from app.snoo_maneuver import (
    GEAR_MAX_SEGMENTS,
    Limits,
    Maneuver,
    Scene,
    Weights,
    plan_maneuver,
)

# 1 回に VGGT に入れるコマの数の上限（24 GB の GPU で 60 枚ほどまで。
# 検出サーバーと分け合うので余裕を見る）
MAX_FRAMES = 48
# 次の位置へ走る速さ（Pi の forward の値）。床が見えている所しか走らないが、ゆっくり行く
SPEED = 12
# ステーションの数の上限（これ以上は探らない）
# 1 周が 17 秒ほどになり、30 か所では 8 分で止まるので、電池（10〜20 分）に合わせる
MAX_STATIONS = 50
# 床・障害物と見る点の、格子 1 マスあたりの最少の数（1 回の撮影で）。
# VGGT の点は外れ値があるので 1 点では決めない
MIN_POINTS_PER_CELL = 3
# 床が見えたセルがこれより少なければ、床の当てはめを信じない
MIN_FREE_CELLS = 30
# 1 か所で首を振ったときのカメラの中心のずれの上限。首振りの軸のずれは数 cm なので、
# これより大きければ VGGT の位置合わせが崩れている
MAX_STATION_SPREAD_M = 0.10
# ステーションの中で外れたコマとみなす、カメラの中心の中央値からの距離と、残す割合の下限
OUTLIER_M = 0.06
OUTLIER_KEEP = 0.6

_NAME = re.compile(r"pan([+-]\d+)_tilt([+-]\d+)")


def _env_float(name: str, default: float | None) -> float | None:
    v = os.getenv(name)
    if v is None:
        return default
    return None if v.strip().lower() in ("", "none", "off") else float(v)


# 上の層（車の高さ＋余白より上の点）の高さの見込みのずれ。VGGT の点は遠いほどずれるので、
# 物の下の端が「車の高さ＋余白＋ずれ」より低ければ障害物にする（ソファ・ベッドの下に首が挟まる）。
# 遠くから見た低い隙間は通れない側に倒し、近くで見直して高さが確かなら通す
OVERHEAD_SIGMA_M = 0.01
OVERHEAD_SIGMA_PER_M = 0.02


def overhead_top_m(car_height_m: float, margin_m: float, dist_m):
    """これより低い点は障害物、高い点は上の層（下をくぐれる）とする高さ（m）。dist_m は撮った
    所からの距離（配列でもよい）。"""
    return car_height_m + margin_m + OVERHEAD_SIGMA_M + OVERHEAD_SIGMA_PER_M * np.asarray(dist_m)


@dataclass
class MapParams:
    # 床からカメラの中心まで。ステーションごとに変えてもよい。None のステーションは
    # 大きさを決めるのに使わず、ほかのステーションで決めた大きさで高さを測る
    # （机の上から撮ったときなど）
    camera_height_m: float | list[float | None] = 0.08  # SD カード 2.5 枚分と測った
    car_height_m: float = CAR_HEIGHT_M  # 測っていない（SNOO_CAR_HEIGHT_CM）
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
    # 後ろへ走る区切りを返してよい（Pi が supports に "reverse" を書いたとき）
    reverse: bool = False
    # 1 歩の中で区切りごとに前・後ろを変えてよい（Pi が supports に "gear_segments" を
    # 書いたとき。区切りは 5 つまで。切り返し・3 点ターン・隙間へ寄せる）
    gear_segments: bool = False
    # 走ってきた道（車の外形の下）を床にする（roommap.mark_driven。地図のサーバーと
    # シミュレーターが間取りに足す）
    driven_free: bool = True
    # 1 歩の長さの上限。Pi が supports に "long_step"（と "segments"）を書いたときは
    # roommap.LONG_STEP_M（60 cm）まで。超えた分はまっすぐの区間だけ
    max_step_m: float = MAX_STEP_M
    seed: int = 0
    # 計画器のコストの重みを場面から出す方策（app/snoo_maneuver_policy.py）の npz。None なら
    # 固定の重み。既定は環境変数 SNOO_MANEUVER_POLICY（無ければ使わない）
    policy: str | None = field(default_factory=lambda: os.getenv("SNOO_MANEUVER_POLICY") or None)


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


# 猫の箱を広げる割合（毛やぼけで箱の外に点がはみ出す分）と、箱とみなす確からしさの下限
CAT_MASK_PAD = 0.15
CAT_MASK_MIN = 0.5


def mask_cats(conf: np.ndarray, boxes: list[list]) -> int:
    """各コマの、猫の箱の中の点の確からしさを 0 にする（間取りに入れない）。消した点の数を返す。

    snoo は動くので、撮った時の場所に「障害物」として残ると、どいた後もそこを通れない幽霊の壁になる
    （Nav2 は光線で消し、RTAB-Map は弱い当たりで少しずつ確かめる。ここは検出の箱で先に除く）。
    boxes はコマごとの [{box: [x1, y1, x2, y2]（0..1）, cat}, ...]。Pi が止まって撮ったコマごとに
    meta の frames[].dets に入れる。VGGT は横を 518 に縮めて縦を切りそろえるので、比のまま当てる。
    """
    if conf.ndim != 3 or not boxes:  # 画素の並びが無い（作り物の点）なら消せない
        return 0
    n = 0
    F, h, w = conf.shape
    for i in range(min(F, len(boxes))):
        dets = boxes[i] if isinstance(boxes[i], list) else []
        for det in dets:
            try:  # Pi から届いたもの。形が違えば使わない（500 にしない）
                if float(det.get("cat", 0)) < CAT_MASK_MIN:
                    continue
                x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in det["box"])
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            dx, dy = (x2 - x1) * CAT_MASK_PAD, (y2 - y1) * CAT_MASK_PAD
            c0, c1 = max(0, int((x1 - dx) * w)), min(w, int(np.ceil((x2 + dx) * w)))
            r0, r1 = max(0, int((y1 - dy) * h)), min(h, int(np.ceil((y2 + dy) * h)))
            n += int((conf[i, r0:r1, c0:c1] > 0).sum())
            conf[i, r0:r1, c0:c1] = 0
    return n


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


def heading_frame(refs: list[FrameRef], station: int, only: list[int] | None = None) -> int:
    """そのステーションで車の向きを出すのに使うコマ。
    tilt 0 に近い段の、パンがいちばん 0 に近いもの（only があれば、その中から）。
    """
    idx = only if only else [i for i, r in enumerate(refs) if r.station == station]
    return min(idx, key=lambda i: (round(abs(refs[i].tilt_deg)), abs(refs[i].pan_deg)))


def station_inliers(cams: np.ndarray) -> np.ndarray:
    """首を振っただけのステーションのコマのうち、カメラの中心がまとまっているもの（bool）。

    首を振るだけならカメラはほぼ動かない（1〜2 cm）。VGGT が 1〜2 コマだけ大きく外すことがあり
    （18:49 の探索の 004 は 16 cm）、それだけでステーション全体を位置合わせの失敗として探索を
    終えていた。真ん中（中央値）から OUTLIER_M より離れたコマを外し、残りが OUTLIER_KEEP 以上
    あれば、残りで位置とばらつきを出す。足りなければ全部を返す（ばらつきで失敗と分かる）。
    """
    if len(cams) < 3:
        return np.ones(len(cams), bool)
    d = np.linalg.norm(cams - np.median(cams, 0), axis=1)
    keep = d <= OUTLIER_M
    if keep.sum() < max(3, math.ceil(OUTLIER_KEEP * len(cams))):
        return np.ones(len(cams), bool)
    return keep


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
    # 障害物と「上」の層のセルの手ざわり (K, 3)：[x, y, 0〜1]（app/snoo_belief.Texture）。
    # 前の版の obs.npz には無い（None）
    texture: np.ndarray | None = None
    # 撮った時刻（秒。Pi の時計）と、見えた人・snoo の足もと (K, 3)：[x, y, 見た時刻]。
    # 後ろへ下がる区切りを決めるのに使う（app/snoo_maneuver.Scene）。前の版の obs.npz には無い
    t_s: float | None = None
    cats: np.ndarray | None = None

    def save(self, path: Path) -> None:
        p = self.pose
        extra = {} if self.texture is None else {"texture": self.texture.astype(np.float32)}
        if self.t_s is not None:
            extra["t_s"] = np.array(self.t_s)
        if self.cats is not None:
            extra["cats"] = np.asarray(self.cats, np.float64).reshape(-1, 3)
        np.savez(
            path,
            free=self.free.astype(np.float32),
            occupied=self.occupied.astype(np.float32),
            high=self.high.astype(np.float32),
            pose=np.array([p.x, p.y, p.yaw]),
            sonar_m=np.array(np.nan if self.sonar_m is None else self.sonar_m),
            **extra,
        )

    @classmethod
    def load(cls, path: Path) -> Observation:
        d = np.load(path)
        sonar = float(d["sonar_m"])
        return cls(
            d["free"], d["occupied"], d["high"], Pose(*map(float, d["pose"])),
            None if math.isnan(sonar) else sonar,
            d["texture"] if "texture" in d.files else None,
            float(d["t_s"]) if "t_s" in d.files else None,
            d["cats"] if "cats" in d.files else None,
        )  # fmt: skip


def session_grid(observations: list[Observation]) -> tuple[Grid, np.ndarray]:
    """ステーションごとに見たものを順に足した格子と「上」の層（build_map と同じ足し方）。
    確信度（grid.confidence）と手ざわり（grid.texture）の層も一緒に作る（app/snoo_belief.py）。"""
    grid = Grid()
    grid.confidence = Confidence(grid.n)
    grid.texture = Texture(grid.n)
    high = np.zeros_like(grid.logodds, np.int32)
    for o in observations:
        add_observation(grid, high, o)
    return grid, high


def add_observation(grid: Grid, high: np.ndarray, o: Observation) -> None:
    """1 ステーションで見たものを格子と層に足す。超音波は弱い証拠だけ（roommap.mark_sonar）。
    撮った時刻があれば、確信度の層に時刻を付け、見えた人・snoo を grid.cats に足す。"""
    mark_station(grid, o.pose)
    if o.cats is not None:
        grid.cats.extend((float(x), float(y), float(t)) for x, y, t in o.cats)
    grid.mark_free(o.free[:, 0], o.free[:, 1])
    grid.mark_occupied(o.occupied[:, 0], o.occupied[:, 1])
    hix, hiy = grid.cell(o.high[:, 0], o.high[:, 1])
    ok = grid.inside(hix, hiy)
    high[hix[ok], hiy[ok]] += 1
    if grid.confidence is not None:
        grid.confidence.add(grid, o.free, o.occupied, o.pose, t=o.t_s)
    if grid.texture is not None and o.texture is not None:
        grid.texture.add(grid, o.texture)
    if o.sonar_m is not None:
        mark_sonar(grid, o.pose, o.sonar_m)
        if grid.confidence is not None:
            grid.confidence.add_sonar(grid, o.pose, o.sonar_m)


# SLAM（app/snoo_slam.py）の結果を間取りに足すのは、座標合わせのずれがこれより小さいときだけ
SLAM_MAX_RMSE_M = 0.08
# SLAM の点のボクセル（3 cm）に、これより少ない点しか入っていなければ使わない（ノイズ）
SLAM_MIN_COUNT = 2


def mark_slam(grid: Grid, high: np.ndarray, slam_dir: Path, params: MapParams) -> dict:
    """走りながらの SLAM の点と道筋を、セッションの間取りに足す。足したセルの数を返す。

    止まって撮った VGGT だけでは、走った道の両脇と、ステーションの間で見えた所が地図に入らない。
    SLAM は 4 枚/秒で走りながら見ているので、そこを埋める。高さの分け方は build_map と同じ。
    state が ok でないとき・座標合わせが崩れているときは足さない（壊れた点で道をふさがない）。
    """
    try:
        status = json.loads((slam_dir / "status.json").read_text())
        alignment = status.get("alignment") or {}
        rmse = alignment.get("rmse_m")
        # 位置で合わせた（ステーション 2 か所以上）ときだけ使う。向きだけで合わせたときは、
        # ずれを確かめられず、向きが 5° ずれると 3 m 先で 26 cm ずれた障害物が入る
        if (
            status.get("state") != "ok"
            or alignment.get("method") != "stations"
            or rmse is None
            or rmse > SLAM_MAX_RMSE_M
        ):
            return {
                "used": False,
                "reason": f"state={status.get('state')} "
                f"method={alignment.get('method')} rmse={rmse}",
            }
        d = np.load(slam_dir / "points.npz")
        xyz, count = d["xyz"], d["count"]
    except (OSError, ValueError, KeyError) as e:
        return {"used": False, "reason": type(e).__name__}
    xyz = xyz[count >= SLAM_MIN_COUNT]
    z = xyz[:, 2]
    # 撮った所からの距離は分からないので、近い所のずれだけ足す
    top = float(overhead_top_m(params.car_height_m, params.obstacle_margin_m, 0.0))
    floor = xyz[np.abs(z) <= params.floor_tol_m]
    obst = xyz[(z > params.floor_tol_m) & (z <= top)]
    above = xyz[z > top]
    grid.mark_free(floor[:, 0], floor[:, 1])
    grid.mark_occupied(obst[:, 0], obst[:, 1])
    hix, hiy = grid.cell(above[:, 0], above[:, 1])
    ok = grid.inside(hix, hiy)
    high[hix[ok], hiy[ok]] += 1
    # 車が通った所（カメラの真下）は床
    path = []
    try:
        for line in (slam_dir / "trajectory.jsonl").read_text().splitlines():
            row = json.loads(line)
            path.append((row["x"], row["y"]))
    except (OSError, ValueError, KeyError):
        pass
    if path:
        xy = np.array(path)
        grid.mark_free(xy[:, 0], xy[:, 1])
    return {"used": True, "floor": len(floor), "obstacle": len(obst), "high": len(above),
            "path": len(path), "rmse_m": rmse}  # fmt: skip


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
    plan: Maneuver | None
    params: MapParams
    # 上から見た絵に使う点（間取りの座標、m）と色（BGR）。間引いてある
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    # 次の 1 歩で車（カメラの真下）が通る所。上から見た絵に描く（後ろへ戻るときはサーバーが入れる）
    step_path: list[Pose] = field(default_factory=list)
    # build_map に anchor を渡したときの、ステーションごとに見たもの（セッションの座標）。
    # 位置合わせが崩れたステーションは None
    observations: list[Observation | None] = field(default_factory=list)
    # 走りながらの SLAM を足したか（mark_slam の答え。地図のサーバーが入れる）
    slam: dict | None = None


def _cells_with(grid: Grid, x: np.ndarray, y: np.ndarray, min_count: int):
    """min_count 点以上あるセルの中心を返す（外れ値の 1 点でセルを決めない）。"""
    ix, iy = grid.cell(x, y)
    ok = grid.inside(ix, iy)
    if not ok.any():
        return np.zeros(0), np.zeros(0)
    cells, counts = np.unique(np.stack([ix[ok], iy[ok]], 1), axis=0, return_counts=True)
    cells = cells[counts >= min_count]
    return grid.center(cells[:, 0], cells[:, 1])


def cell_texture(grid: Grid, xy: np.ndarray, tex: np.ndarray) -> np.ndarray:
    """点の手ざわりをセルごとに平均した (K, 3)：[セルの中心 x, y, 手ざわり]。
    MIN_POINTS_PER_CELL 点より少ないセルは使わない。"""
    if not len(xy):
        return np.zeros((0, 3), np.float32)
    ix, iy = grid.cell(xy[:, 0], xy[:, 1])
    ok = grid.inside(ix, iy)
    key = ix[ok] * grid.n + iy[ok]
    uniq, inv, count = np.unique(key, return_inverse=True, return_counts=True)
    mean = np.bincount(inv, weights=tex[ok], minlength=len(uniq)) / count
    keep = count >= MIN_POINTS_PER_CELL
    cx, cy = grid.center(uniq[keep] // grid.n, uniq[keep] % grid.n)
    return np.stack([cx, cy, mean[keep]], 1).astype(np.float32)


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
    # 点ごとの手ざわり（写真のその所の勾配。白い壁は低い）。目印の層に使う
    texs = (
        np.stack([texture_map(c) for c in colors.reshape(F, *colors.shape[-3:])]).reshape(F, -1)
        if colors is not None and colors.ndim >= 4
        else None
    )
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
    results: list[StationResult] = []
    observations: list[Observation | None] = []
    keep_pts, keep_cols = [], []
    rng = np.random.default_rng(0)
    for s, st in enumerate(stations):
        idx = [i for i, r in enumerate(refs) if r.station == s]
        # 外れたコマ（VGGT が位置を外した）は、位置・向き・地図の点に使わない
        idx = [i for i, ok in zip(idx, station_inliers(cams_w[idx]), strict=True) if ok]
        c = cams_w[idx]
        center = c.mean(0)
        k = heading_frame(refs, s, idx)
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
        dist = np.hypot(p[:, 0] - pose.x, p[:, 1] - pose.y)
        near = dist <= params.max_range_m
        p, dist = p[near], dist[near]
        z = p[:, 2]
        # 車の高さ＋余白より下は障害物。遠い点ほど高さがずれるので、余白を足す（overhead_top_m）
        top_p = overhead_top_m(params.car_height_m, params.obstacle_margin_m, dist)
        is_floor = np.abs(z) <= params.floor_tol_m
        is_obst = (z > params.floor_tol_m) & (z <= top_p)
        is_high = z > top_p
        fx, fy = _cells_with(grid, p[is_floor, 0], p[is_floor, 1], MIN_POINTS_PER_CELL)
        mark_station(grid, pose)
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
            tex = None
            if texs is not None:
                t = texs[idx][good[idx]][near]
                tex = cell_texture(grid, q[is_obst | is_high], t[is_obst | is_high])
            observations[-1] = Observation(*cells, to_frame(anchor, pose), sonar, tex)
        res.floor_points, res.obstacle_points = int(is_floor.sum()), int(is_obst.sum())
        res.high_points = int(is_high.sum())
        if cols is not None:
            sel = rng.choice(len(p), min(len(p), 60_000), replace=False)
            keep_pts.append(p[sel])
            keep_cols.append(cols[idx][good[idx]][near][sel])

    action, nbv = decide(grid, results, params, heights) if plan else ({}, None)
    step_path = []
    if nbv is not None:
        step_path = [camera_of(p) for _, p, _, _ in drive_moves(results[-1].pose, nbv.moves)]
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
        # 地図のサーバーはステーションと同じ数の見たものを並べて回すので、数を合わせる
        # （空だと zip(strict=True) で落ち、done の代わりに 500 を返していた）
        observations=[None] * len(stations),
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


def stop_reason(
    grid: Grid,
    stations: list[StationResult],
    params: MapParams,
    heights: list[float | None],
    trusted: frozenset[str] = frozenset(),
    state: np.ndarray | None = None,
    conf: np.ndarray | None = None,
) -> dict | None:
    """探るのをやめる返事（done）か、続けるなら None。決め方（今まで・学んだもの）によらない。

    trusted のステーション（名前）は、この窓で位置合わせが崩れても止めない。地図のサーバーが
    位置のグラフ（app/snoo_posegraph.py）で、走った辺とほかの窓から位置を決めたもの。
    state・conf は grid.state()・confidence_of の値（decide が一度だけ出して渡す）。
    """
    if len(stations) >= params.max_stations:
        return {"action": "done", "reason": "step_limit"}
    for st, H in zip(stations, heights, strict=True):
        if st.name not in trusted and registration_is_bad(st, H):
            return {"action": "done", "reason": f"bad_registration:{st.name}"}
    state = grid.state() if state is None else state
    if int((state == FREE).sum()) < MIN_FREE_CELLS:
        return {"action": "done", "reason": "too_little_floor"}
    # 次に撮る位置と同じ境目で見る（床の中の小さな穴だけが残ったら、調べ終えたとする）。
    # 境目が無くても、確信度の低い縁（近づいて確かめる行き先）が残っていれば続ける
    if not big_frontiers(state).any():
        conf = confidence_of(grid, state) if conf is None else conf
        unsure = int(unsure_edges(grid, state, conf).sum())
        if unsure < CONF_DONE_MAX_CELLS:
            return {"action": "done", "reason": "no_frontier"}
    return None


def limits(params: MapParams, stuck: bool = False) -> Limits:
    """Pi が走れる 1 歩の形。"""
    if params.gear_segments:
        n = GEAR_MAX_SEGMENTS
    else:
        n = MAX_SEGMENTS if params.segments else 1
    return Limits(
        gears=params.gear_segments,
        reverse=params.reverse,
        max_segments=n,
        max_fwd_m=params.max_step_m,
        wide=stuck,
    )


def decide(
    grid: Grid,
    stations: list[StationResult],
    params: MapParams,
    heights: list[float | None],
    trusted: frozenset[str] = frozenset(),
    recent: list[Pose] | None = None,
    weights: Weights | None = None,
) -> tuple[dict, Maneuver | None]:
    """最後のステーションから次の 1 歩を決める。止める理由があれば done。

    1 歩は app/snoo_maneuver.plan_maneuver（MPPI 風。名目の列は next_best_view の前への 1 歩・
    Hybrid A*・切り返しのひな形）で、Pi が走れる形（limits）の区切りの列にする。
    recent は直近のステーションの位置（同じ所へ戻らない・動けていないかを見る）。動けていなければ
    広く探し（区切りを長く・前後の切り替えを安く）、それでも無ければ done "stuck"。
    trusted は stop_reason と同じ（位置のグラフで位置が決まったステーションの名前）。
    計画器の行き先（撮った所のそばを除いた境目・確信度の低い縁の手前・next_best_view の行き先）が
    一つも無ければ done "no_frontier"（stop_reason の境目は残っていても、計画器が使えない）。
    weights は計画器のコストの重み（学ぶときに外から渡す）。無くて params.policy があれば、
    その方策が場面から出す（app/snoo_maneuver_policy.py）。どちらも無ければ固定の重み。
    """
    state = grid.state()
    conf = confidence_of(grid, state)
    stop = stop_reason(grid, stations, params, heights, trusted, state, conf)
    if stop is not None:
        return stop, None
    pose = stations[-1].pose
    recent = list(recent if recent is not None else [s.pose for s in stations[-RECENT_STATIONS:]])
    stuck = is_stuck(recent)
    lim = limits(params, stuck or is_still(recent))
    nominal = None
    plan = next_best_view(grid, pose, segments=params.segments, max_step_m=params.max_step_m)
    if plan is not None:
        nominal = [(st, m, False) for st, m in (plan.segments or [(plan.steer_deg, plan.step_m)])]
    goal = plan.target if plan is not None else None
    policy_info = None
    if weights is None and params.policy:
        from app.snoo_maneuver_policy import explain, features, load_policy

        f = features(grid, pose, recent, state, conf, len(stations))
        weights = load_policy(params.policy).weights(f)
        policy_info = explain(f, weights)
    # 同じ格子・姿勢なので、広く探し直すときも同じ Scene を使う
    scene = Scene(grid, pose, True, recent[:-1], lim.wide, goal, state, conf, weights)
    if not scene.targets.any():
        # 残っている境目は撮った所のそば（足もと・車の横。どこから撮っても写らない）だけ。
        # 撮り直しを続けて stuck にしない
        return {"action": "done", "reason": "no_frontier"}, None
    man = plan_maneuver(grid, pose, lim, nominal, params.seed, scene=scene)
    if man is None and not lim.wide:
        # 行ける列が見つからない：切り返しを安くして広く探し直す
        lim = limits(params, True)
        man = plan_maneuver(grid, pose, lim, nominal, params.seed + 1, scene=scene)
    if stuck and (man is None or not man.info.get("connected", True)):
        # 動けていないのに、行き先までつながる道が無い：同じ所で切り返し続けない
        return {"action": "done", "reason": "stuck"}, None
    if man is None:
        # 走れる列が無い（地図の上で囲まれている）。動かずに撮り直す（VGGT の窓が変わると点の
        # ぶれが変わり、猫もどくかもしれない）。同じ所で続けば上の stuck で終える
        return rescan_action(pose), None
    action = maneuver_action(man, pose, params)
    if lim.wide:
        action["stuck"] = True
    if policy_info is not None:
        action["policy"] = policy_info
    return action, man


# 直近 RECENT_STATIONS か所が、この半径に収まっていれば「動けていない」
RECENT_STATIONS = 6
STUCK_RADIUS_M = 0.20
# 続けてこれしか動けなければ（3 か所）、動けていないとする
STILL_M = 0.03
# 下がるときの速さ（後ろにはセンサーが無いので、前（12）より遅く。約 14 cm/s）
REVERSE_SPEED = 8


def rescan_action(pose: Pose) -> dict:
    """動かずにその場で撮り直す 1 歩（長さ 0）。どの Pi でも、走らずに次の station を撮る。"""
    return {
        "action": "move", "steer_deg": 0.0, "speed": SPEED, "distance_cm": 0.0, "segments": [],
        "pose": pose_dict(pose), "rescan": True, "why": "unreachable",
    }  # fmt: skip


def is_stuck(recent: list[Pose]) -> bool:
    """直近のステーション（今を含めて RECENT_STATIONS か所）の正味の動きが小さいか。
    2026-10-31 の実走の 024〜032 は、10 cm の切り返しを左右交互に 9 回して同じ所に居た。"""
    if len(recent) < RECENT_STATIONS:
        return False
    last = recent[-RECENT_STATIONS:]
    cx = sum(p.x for p in last) / len(last)
    cy = sum(p.y for p in last) / len(last)
    return all(math.hypot(p.x - cx, p.y - cy) <= STUCK_RADIUS_M for p in last)


def is_still(recent: list[Pose]) -> bool:
    """続けて 2 回ほとんど動けなかった（前の 1 歩を Pi が止めた・猫が前に居る）か。広く探して、
    同じ 1 歩を出し続けない（終えはしない）。"""
    return len(recent) >= 3 and all(
        math.hypot(p.x - recent[-1].x, p.y - recent[-1].y) <= STILL_M for p in recent[-3:]
    )


def maneuver_action(man: Maneuver, pose: Pose, params: MapParams) -> dict:
    """Pi に返す 1 歩（docs/snoo_explore_protocol.md）。形は moves_action。gear_segments の
    Pi には、区切りごとに着く姿勢（end_pose）・前の区切りの clear_cm の下限（min_clear_cm）も
    付ける。"""
    action = moves_action(man.moves, params, man.ends, man.min_clear_cm)
    action["pose"] = pose_dict(pose)
    action["target"] = pose_dict(man.end)
    action["planner"] = man.kind
    if man.gap:
        action["gap"] = True
    return action


def moves_action(
    moves: list[tuple[float, float, bool]],
    params: MapParams,
    ends: list[Pose] | None = None,
    clears: list[float | None] | None = None,
) -> dict:
    """区切りの列 [(ハンドル°, m, 後ろへか), ...] の 1 歩の返事（計画器の 1 歩・来た道を戻る・
    エージェントの 1 歩で同じ形）。上の steer_deg・distance_cm・reverse は最初の区切り
    （segments を知らない Pi でも同じに動く）。gear_segments の Pi には、区切りごとに reverse と、
    ends・clears があれば end_pose・min_clear_cm を付ける。"""
    first = moves[0]
    all_back = all(b for _, _, b in moves)
    action = {
        "action": "move",
        "steer_deg": round(first[0], 1) + 0.0,
        "speed": REVERSE_SPEED if all_back else SPEED,
        "distance_cm": round(first[1] * 100, 1),
    }
    if all_back:
        action["reverse"] = True
    if params.gear_segments:
        action["reverse_speed"] = REVERSE_SPEED
        segs = []
        for i, (st, m, back) in enumerate(moves):
            seg = {"steer_deg": round(st, 1) + 0.0, "distance_cm": round(m * 100, 1),
                   "reverse": back}  # fmt: skip
            if ends is not None:
                seg["end_pose"] = pose_dict(ends[i])
            if clears is not None:
                seg["min_clear_cm"] = clears[i]
            segs.append(seg)
        action["segments"] = segs
    elif params.segments:
        action["segments"] = segments_json([(st, m) for st, m, _ in moves])
    return action


def action_moves(action: dict) -> list[tuple[float, float, bool]]:
    """返事の区切り [(ハンドル°, m, 後ろへか), ...]。区切りの reverse が無ければ返事の reverse。"""
    back = bool(action.get("reverse"))
    segs = action.get("segments") or [action]
    return [(float(x["steer_deg"]), float(x["distance_cm"]) / 100, bool(x.get("reverse", back)))
            for x in segs]  # fmt: skip


def segments_json(pieces: list[tuple[float, float]]) -> list[dict]:
    return [{"steer_deg": round(st, 1) + 0.0, "distance_cm": round(m * 100, 1)} for st, m in pieces]


# 1 回に VGGT に入れるステーションの数（地図のサーバーもこれを使う）。最後のステーションの
# 全部のコマと、近くの前の 3 か所の 6 コマずつで 28 コマ（VGGT 4.5 秒ほど）。6 か所・48 コマでは
# 6.5 秒かかり、位置はほとんど変わらなかった
WINDOW = 4


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
            # 確信度（app/snoo_belief.py）：0 = 見ていない、1〜255 = 確信度 0〜1
            "confidence": {
                "png": "confidence.png",
                "legend": {"pixel": legend["pixel"], "values": {"0": "unknown", "1..255": "0..1"}},
            },
            # 手ざわり（目印）：0〜255 = 0〜1。LANDMARK_TEX 以上を目印にする
            "texture": {
                "png": "texture.png",
                "legend": {"pixel": legend["pixel"], "landmark_min": LANDMARK_TEX},
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
    """room.json・grid.png・high.png・confidence.png・texture.png・plan.json・topdown.png
    を書く。"""
    out.mkdir(parents=True, exist_ok=True)
    state = r.grid.state()
    cv2.imwrite(str(out / "grid.png"), grid_png(state))
    cv2.imwrite(str(out / "confidence.png"), confidence_png(confidence_of(r.grid, state), state))
    tex = getattr(r.grid, "texture", None)
    tv = tex.value() if tex is not None else np.zeros(state.shape, np.float32)
    cv2.imwrite(str(out / "texture.png"), texture_png(tv))
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
