"""単眼の metric の深度（cm）と、床の幾何での尺度の確かめ。

PiCar-X のカメラは床から 8 cm と低く、学習の写真（人の目の高さ）とは見え方が違う。
metric の深度のモデルがこの高さでも cm を正しく出すかは、床で確かめられる。床の画素の
本当の奥行きは、カメラの高さと tilt だけで決まる（app/clearance.floor_point と同じ幾何）。
外のセンサー（LiDAR など）がまだ無いので、これを主な答え合わせにする。

ここは GPU を使わない計算（床の奥行き・比べる数・尺度の確かめ・clear_cm）と、
HF のモデルを包む MonoDepth（使うときだけ torch を読む）に分けてある。
比べた結果と使い方の案は scripts/snoo_monodepth_eval.py の頭に書いた。
まだ検出（app/snoo.py）にはつないでいない。
"""

import math
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from app.clearance import (
    Camera,
    Clearance,
    Grid,
    Path,
    _blind,
    clearance_from_z,
    fit_floor,
    pixel_rays,
)

# 試したモデル（HF の名前）。短い名前はスクリプトの表に使う
MODELS = {
    "da2-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "da2-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "depthpro": "apple/DepthPro-hf",
    "zoedepth": "Intel/zoedepth-nyu-kitti",
}

# 尺度を確かめる床: カメラから FLOOR_CHECK_CM まで。遠い床は tilt の小さな誤差で奥行きが
# 大きく変わる（8 cm の高さで 1° 違うと、1 m 先の床は 20 cm 以上ずれる）ので、近くだけを使う
FLOOR_CHECK_CM = 60.0
# 尺度が合っているとみなす比の幅（本当 / 見積もり が 1/1.25..1.25）
SCALE_TOL = 1.25


def floor_depth(cam: Camera) -> np.ndarray:
    """各画素に床が映っているとしたときの奥行き（cm、カメラの前への距離）。形は (高さ, 幅)。

    単眼の深度のモデルが返すのも同じ「前への距離」（光線の長さではない）。
    水平より上を向く画素は nan。
    """
    v, u = np.mgrid[0 : cam.height, 0 : cam.width]
    down = -pixel_rays(cam, u, v)[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(down > 1e-9, cam.height_cm / down, np.nan)


@dataclass
class DepthStats:
    """見積もり（pred）を本当（gt）と比べた数。

    ratio は gt / pred の中央値（1 より大きいと近すぎ、小さいと遠すぎる）。
    """

    n: int
    abs_rel: float  # |pred - gt| / gt の平均
    rmse_cm: float
    delta1: float  # max(pred/gt, gt/pred) < 1.25 の割合
    ratio: float  # 尺度のずれ（gt / pred の中央値）
    abs_rel_scaled: float  # ratio をかけて尺度を合わせた後の AbsRel（形だけの良さ）

    def to_json(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _valid(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    ok = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    return ok if mask is None else ok & mask


def median_ratio(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """gt / pred の中央値。比べられる画素が無ければ None。"""
    ok = _valid(pred, gt, mask)
    if not ok.any():
        return None
    return float(np.median(gt[ok] / pred[ok]))


def depth_stats(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None, min_n: int = 20
) -> DepthStats | None:
    """深度の推定でよく使う数（AbsRel・RMSE・δ<1.25）と、尺度のずれ。画素が少なければ None。"""
    ok = _valid(pred, gt, mask)
    n = int(ok.sum())
    if n < min_n:
        return None
    p, g = pred[ok].astype(np.float64), gt[ok].astype(np.float64)
    ratio = float(np.median(g / p))
    return DepthStats(
        n=n,
        abs_rel=float(np.mean(np.abs(p - g) / g)),
        rmse_cm=float(np.sqrt(np.mean((p - g) ** 2))),
        delta1=float(np.mean(np.maximum(p / g, g / p) < 1.25)),
        ratio=ratio,
        abs_rel_scaled=float(np.mean(np.abs(p * ratio - g) / g)),
    )


@dataclass
class ScaleCheck:
    """床の近くで見た尺度のずれ。ratio は 本当 / 見積もり（そのままかけると合う）。"""

    ratio: float | None
    n: int  # 使った床の画素
    inliers: float  # ratio の ±25 % に入った割合（低いと床に物が載っているか、形が崩れている）

    @property
    def ok(self) -> bool:
        """尺度が SCALE_TOL の中で、床の大半がそれに合っている。"""
        if self.ratio is None:
            return False
        return 1 / SCALE_TOL <= self.ratio <= SCALE_TOL and self.inliers >= 0.5


def floor_scale_check(
    depth_cm: np.ndarray,
    cam: Camera,
    mask: np.ndarray | None = None,
    max_cm: float = FLOOR_CHECK_CM,
) -> ScaleCheck:
    """近くの床（カメラから max_cm まで）で、見積もった深度の尺度を確かめる。

    床の上に物があると、その画素の見積もりは床より近い（比が大きい）。中央値なので、
    床が半分より多ければ外れない。cam は depth_cm と同じ大きさにしたもの。
    """
    floor = floor_depth(cam)
    near = np.isfinite(floor) & (floor <= max_cm)
    if mask is not None:
        near &= mask
    ok = _valid(depth_cm, floor, near)
    n = int(ok.sum())
    if n < 50:
        return ScaleCheck(None, n, 0.0)
    r = floor[ok] / depth_cm[ok]
    ratio = float(np.median(r))
    inl = float(np.mean(np.abs(r / ratio - 1) < 0.25))
    return ScaleCheck(ratio, n, inl)


def floor_affine(
    depth_cm: np.ndarray, cam: Camera, max_cm: float = FLOOR_CHECK_CM
) -> tuple[np.ndarray, float] | None:
    """逆数の深度を近くの床に一次式で合わせ直す（1/z = a/pred + b を床で決める）。

    今の検出（相対の深度を床に合わせる clearance_from_disparity）と同じ直し方を metric の
    深度にかけたもの。尺度だけ（floor_scale_check）では、遠くほど見積もりのずれ方が変わる
    のを直せなかったので比べる。(直した深度, 床に当てはまった割合) を返す。合わなければ None。
    """
    floor = floor_depth(cam)
    near = np.isfinite(floor) & (floor <= max_cm) & np.isfinite(depth_cm) & (depth_cm > 0)
    if near.sum() < 50:
        return None
    a, b, inliers = fit_floor(1 / floor[near], 1 / depth_cm[near].astype(np.float64))
    if a <= 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = 1 / depth_cm.astype(np.float64)
        z = np.where(disp - b > 1e-9, a / (disp - b), np.inf)
    return z.astype(np.float32), inliers


def clearance_from_metric(
    depth_cm: np.ndarray, cam: Camera, path: Path, scale: float = 1.0, confidence: float = 1.0
) -> Clearance:
    """metric の深度（cm）から clear_cm を出す。障害物の決め方は app/clearance と同じ。

    scale は深度にかける数（floor_scale_check の ratio を渡すと、床で尺度を合わせた答えになる）。
    分からない画素（nan・0 以下）は使わない。
    """
    h, w = depth_cm.shape
    grid = Grid.cached(cam.scaled(w, h), path)
    if (none := _blind(grid, "metric")) is not None:
        return none
    z = np.where(np.isfinite(depth_cm) & (depth_cm > 0), depth_cm * scale, np.inf)
    c = clearance_from_z(z, grid, path, confidence)
    return Clearance(c.clear_cm, c.confidence, c.blind_cm, "metric")


class MonoDepth:
    """HF の metric の深度のモデルを包む。depth_cm(画像) で入力と同じ大きさの深度（cm）を返す。

    Depth Anything V2 は前処理を GPU で行う（HF の前処理は PIL で、CPU に 10 ms ほどかかる）。
    ほかのモデルは HF の前処理と後処理をそのまま使う。DepthPro は画角も推定するが、
    PiCar-X の画角は測ってあるので、known_fov=True なら測った焦点距離で尺度を決める。
    """

    def __init__(
        self,
        name: str,
        device: str | None = None,
        half: bool = True,
        size: tuple[int, int] | None = None,
        known_fov: bool = False,
        hfov_deg: float = Camera().hfov_deg,
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.name = name
        self.repo = MODELS.get(name, name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if half and self.device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(self.repo)
        model = AutoModelForDepthEstimation.from_pretrained(self.repo, dtype=self.dtype)
        self.model = model.to(self.device).eval()
        self.kind = self.model.config.model_type
        self.known_fov, self.hfov_deg = known_fov, hfov_deg
        # DA V2 の入力の大きさ（幅, 高さ）。14 の倍数。既定は HF の前処理と同じ（短い辺 518）
        self.size = size or (686, 518)
        # DA V2 を GPU で前処理するときの平均とばらつき（DepthPro の前処理は 1 つの数で持つ）
        mean = np.broadcast_to(getattr(self.processor, "image_mean", 0.5), 3).tolist()
        std = np.broadcast_to(getattr(self.processor, "image_std", 0.5), 3).tolist()
        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.last_fov_deg: float | None = None
        self.timing: dict[str, float] = {}

    def _sync(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def depth_cm(self, image_bgr: np.ndarray) -> np.ndarray:
        """BGR の画像の深度（cm、float32、入力と同じ大きさ）。時間（ms）は self.timing に残す。"""
        torch = self.torch
        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            if self.kind == "depth_anything":
                x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0)
                x = x.to(self.dtype).div_(255)
                sw, sh = self.size
                x = torch.nn.functional.interpolate(x, (sh, sw), mode="bicubic", antialias=True)
                inputs = {"pixel_values": (x - self.mean) / self.std}
            else:
                inputs = self.processor(images=rgb, return_tensors="pt")
                inputs = {k: v.to(self.device, self.dtype) for k, v in inputs.items()}
            self._sync()
            t1 = time.perf_counter()
            out = self.model(**inputs)
            self._sync()
            t2 = time.perf_counter()
            depth_m = self._post(out, h, w)
            self._sync()
        t3 = time.perf_counter()
        self.timing = {
            "pre_ms": (t1 - t0) * 1e3,
            "model_ms": (t2 - t1) * 1e3,
            "post_ms": (t3 - t2) * 1e3,
            "total_ms": (t3 - t0) * 1e3,
        }
        return depth_m

    def _post(self, out, h: int, w: int) -> np.ndarray:
        torch = self.torch
        if self.kind == "zoedepth":
            d = self.processor.post_process_depth_estimation(out, source_sizes=[(h, w)])
            return (d[0]["predicted_depth"].float() * 100).cpu().numpy()
        if self.kind == "depth_pro":
            canon = out.predicted_depth[0].float()  # 尺度の決まっていない逆数の深度
            fov = float(out.field_of_view[0])
            self.last_fov_deg = fov
            use = self.hfov_deg if self.known_fov else fov
            focal = 0.5 * w / math.tan(math.radians(use) / 2)
            canon = torch.nn.functional.interpolate(canon[None, None], (h, w), mode="bilinear")[
                0, 0
            ]
            return (focal / torch.clamp(canon * w, min=1e-4) * 100).cpu().numpy()
        d = out.predicted_depth.float().unsqueeze(1)
        d = torch.nn.functional.interpolate(d, (h, w), mode="bilinear", align_corners=False)
        return (d[0, 0] * 100).cpu().numpy()
