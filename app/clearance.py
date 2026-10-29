"""PiCar-X のカメラ画像から「進む先があと何 cm 空いているか」（clear_cm）を見積もる。

超音波のセンサーが壊れていて、走っている間は前が見えない。スキャンで見えなかった細いケーブルや
出てきた snoo に当たらないよう、走る間はカメラで前を見る。

幾何（GPU は使わない）
- 床は平ら、カメラは床から height_cm の高さにある。tilt（上が正）と pan（右が正）は
  Pi から受け取る。画素を通る光線と床の交点が、その画素に映る床の位置になる（floor_point）。
  逆も出せる（project）。
- 車の位置は「カメラの真下」を原点に、前が x、左が y、上が z（cm）。
- 進む道（Path）は、切った舵（右が正）で決まる円弧。車の幅に余裕を足した帯を通り道とし、
  帯の中の点を (s, lat) で表す。s は「車があと何 cm 進むと前の端がそこに着くか」、lat は
  道の中心から左への距離。曲がるときは前の外側の角が大きく振れるので、外側の帯を広げる。
- 画面の下の端より手前と、帯の幅が画角に入りきらない所は見えない。帯の幅の 90% が映る所から
  を見えているとし、その手前を blind_cm として返す（tilt 0° で前の端から約 17 cm、-15° で
  約 13 cm）。手前はスキャンや前のフレームで見えていたはず、という前提。曲がると帯はすぐ
  画面の横へ出るので、見えている端（clear_cm の上限）が近くなる。曲がるときは pan を曲がる
  向きに振るとよい。
- tilt は TILT_GAIN・TILT_OFFSET_DEG で直せる。Pi が 10/29 20:20 ごろに tilt の校正を直したので、
  それより後のフレームは直さなくてよい（既定は直さない）。前のフレームは、指令 0° で
  約 4.5° 下を向いていた。

見積もりの方法（二つ試して比べた）
(a) DepthClearance: 単眼の深度（Depth Anything V2 Small の相対版、Apache-2.0）。床の画素で視差を
    床の幾何に当てはめて尺度を決め、画素ごとに点の位置と高さを出す。通り道の帯の中で床から
    2〜15 cm の点を障害物とする。GPU で 3.4 ms（compile）＋ CPU で 1.2 ms。
(b) FloorClearance: 幾何と床の見た目。通り道の帯を真上から見た格子（1 cm 角）に画像を写し、
    道に沿って床の色が急に変わる所（物の足もと）を探す。CPU で 0.5 ms。
(a) は高さで見るので、床と色の近い物（白い箱）も見えるが、高さ 2 cm 未満の寝たケーブルは
見えない。(b) はケーブルが見えるが、色の近い物を見落とし、家具の影の縁で止まりすぎる。
地図のフレーム 23 枚と、近くに脚・箱・ケーブルを貼った合成 24 枚で比べると、
(a) 見落とし: 実物 3（床に置いた低い機械。高さ 2 cm 未満かもしれない）・ケーブル 8/8、止まりすぎ 0。
(b) 見落とし: 箱 5/8・脚 1/8、止まりすぎ: 実物 3（ソファの影）。
二つの近い方（combine）は、見落とし 0、止まりすぎ 実物 3・脚 1・箱 1。走るときは combine を使う。

検出の返事への載せ方は app/snoo.py（ClearanceWorker）。走っている間だけ、検出の返事を送った後に
別のスレッドで最新のフレームを Estimator にかけ、次の返事に載せる。
"""

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import numpy as np

from app import roommap

# 横の画角（測った値）。縦は 640x480 の比から決まる（約 42°）
HFOV_DEG = 54.0
# カメラの床からの高さ（まだ測っていない。距離はこれに比例して変わる）
CAM_HEIGHT_CM = float(os.getenv("SNOO_CAMERA_HEIGHT_CM", "8"))  # SD カード 2.5 枚分と測った
# Pi の tilt の指令から、床に対する実際の向きを出す（実際 = 指令 × GAIN + OFFSET）。
# 試作のときの地図のフレーム（VGGT の深度）では、水平線がどの tilt でも 40〜50 画素上にあった
# （= カメラが約 4.5° 下を向いていた）。同じソファの脚が tilt 0° と -15° で同じ位置に来るのは、
# -15° が実際は -20.7° のとき（GAIN 1.08、OFFSET -4.5）。Pi はこれを受けて 10/29 20:20 ごろに
# tilt の校正を 26.5° から 22.0° に直したので、今のフレームは直さない（1.0 と 0.0）。
# 直す前に撮った画像（data/snoo_map/map_20261029_192220 など）を試すときだけ、環境変数で戻す
TILT_GAIN = float(os.getenv("SNOO_TILT_GAIN", "1.0"))
TILT_OFFSET_DEG = float(os.getenv("SNOO_TILT_OFFSET_DEG", "0.0"))
# これより先は見ない（40 cm の弧を走って止まるまでの分に余裕を足した長さ）
MAX_CM = 100.0
# 見えない手前がこれより長いときは、clear_cm を出さない（カメラが上を向きすぎている）
MAX_BLIND_CM = 40.0


@dataclass(frozen=True)
class Camera:
    height_cm: float = CAM_HEIGHT_CM
    tilt_deg: float = 0.0  # 上が正（0° で水平）
    pan_deg: float = 0.0  # 右が正
    hfov_deg: float = HFOV_DEG
    width: int = 640
    height: int = 480

    @property
    def f(self) -> float:
        """焦点距離（画素）。画素は正方形とする。"""
        return self.width / 2 / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def center(self) -> tuple[float, float]:
        return (self.width - 1) / 2, (self.height - 1) / 2

    def rotation(self) -> np.ndarray:
        """カメラの向き（OpenCV の右・下・前）を車の向き（前・左・上）に直す行列。

        pan の台の上に tilt の台が載っているので、tilt を回してから pan を回す。
        """
        t, p = math.radians(self.tilt_deg), math.radians(self.pan_deg)
        to_car = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        tilt = np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])
        pan = np.array([[math.cos(p), math.sin(p), 0], [-math.sin(p), math.cos(p), 0], [0, 0, 1]])
        return pan @ tilt @ to_car

    @classmethod
    def from_pi(cls, tilt_deg: float, pan_deg: float, **kw) -> "Camera":
        """Pi が送る tilt・pan の指令から。tilt は TILT_GAIN・TILT_OFFSET_DEG で直す。"""
        return cls(tilt_deg=tilt_deg * TILT_GAIN + TILT_OFFSET_DEG, pan_deg=pan_deg, **kw)

    def scaled(self, width: int, height: int) -> "Camera":
        """同じ画角のまま、画像の大きさだけ変えたもの（縮めた画像で計算するとき）。"""
        return Camera(self.height_cm, self.tilt_deg, self.pan_deg, self.hfov_deg, width, height)


def pixel_rays(cam: Camera, u, v) -> np.ndarray:
    """画素 (u, v) を通る光線の向き（車の向き。カメラの前への成分が 1 の長さ）。形は (..., 3)。"""
    cx, cy = cam.center
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    d = np.stack([(u - cx) / cam.f, (v - cy) / cam.f, np.ones_like(u)], axis=-1)
    return d @ cam.rotation().T


def floor_point(cam: Camera, u, v) -> tuple[np.ndarray, np.ndarray]:
    """画素 (u, v) に映る床の位置 (x, y) cm。水平より上を向く画素は nan。"""
    d = pixel_rays(cam, u, v)
    down = -d[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(down > 1e-9, cam.height_cm / down, np.nan)
    return t * d[..., 0], t * d[..., 1]


def project(cam: Camera, points) -> tuple[np.ndarray, np.ndarray]:
    """車の向きの点 (x, y, z) cm を画素 (u, v) にする。カメラの後ろの点は nan。"""
    p = np.asarray(points, np.float64) - np.array([0.0, 0.0, cam.height_cm])
    c = p @ cam.rotation()  # 回転の逆（転置）をかける
    cx, cy = cam.center
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(c[..., 2] > 1e-9, c[..., 2], np.nan)
        return cx + cam.f * c[..., 0] / z, cy + cam.f * c[..., 1] / z


def floor_row(cam: Camera, x_cm: float, y_cm: float = 0.0) -> float:
    """床の (x, y) が映る画像の行。"""
    return float(project(cam, [x_cm, y_cm, 0.0])[1])


@dataclass(frozen=True)
class Car:
    """PiCar-X の寸法（cm）。値は app/roommap.py にまとめてある（前後の位置はまだ測っていない）。

    前はここだけ後ろの車軸まで 12 cm・ホイールベース 12 cm で、計画の弧（16 cm・14 cm）より
    通り道の帯の曲がりが 15 % ほどきつかった（コードレビュー）。
    """

    width_cm: float = roommap.CAR_WIDTH_M * 100
    height_cm: float = roommap.CAR_HEIGHT_M * 100
    margin_cm: float = 2.0  # 両脇に足す余裕
    # カメラの真下から前の端まで
    cam_to_front_cm: float = (roommap.CAR_FRONT_M - roommap.CAM_X_M) * 100
    # カメラの真下から後ろの車軸まで（曲がるときの中心の位置）
    cam_to_rear_axle_cm: float = roommap.CAM_X_M * 100
    wheelbase_cm: float = roommap.WHEELBASE_M * 100  # 前と後ろの車軸の間


@dataclass(frozen=True)
class Path:
    """舵 steer_deg（右が正）のまま進む円弧と、その通り道の帯。"""

    car: Car = Car()
    steer_deg: float = 0.0

    @property
    def radius(self) -> float:
        """後ろの車軸の中心が描く円の半径。左に曲がると正、右は負、まっすぐは inf。"""
        if abs(self.steer_deg) < 0.5:
            return math.inf
        return self.car.wheelbase_cm / math.tan(math.radians(-self.steer_deg))

    @property
    def _front(self) -> float:
        """後ろの車軸から前の端まで。"""
        return self.car.cam_to_rear_axle_cm + self.car.cam_to_front_cm

    @property
    def max_s(self) -> float:
        """扱う s の上限。曲がるときは半周まで（舵 30° だと半径 21 cm で、約 47 cm）。"""
        r = self.radius
        if math.isinf(r):
            return math.inf
        return abs(r) * (math.pi - math.atan2(self._front, abs(r)))

    def to_path(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """床の (x, y) を道の (s, lat) にする。s は前の端がそこに着くまでに進む距離。"""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        r = self.radius
        if math.isinf(r):
            return x - self.car.cam_to_front_cm, y
        big, sign = abs(r), math.copysign(1.0, r)
        bx, by = x + self.car.cam_to_rear_axle_cm, sign * y  # 左に曲がる形に揃える
        dist = np.hypot(bx, big - by)
        angle = np.arctan2(bx, big - by)
        return big * (angle - math.atan2(self._front, big)), sign * (big - dist)

    def from_path(self, s, lat) -> tuple[np.ndarray, np.ndarray]:
        s, lat = np.asarray(s, np.float64), np.asarray(lat, np.float64)
        r = self.radius
        if math.isinf(r):
            return s + self.car.cam_to_front_cm, lat
        big, sign = abs(r), math.copysign(1.0, r)
        angle = s / big + math.atan2(self._front, big)
        dist = big - sign * lat
        bx, by = dist * np.sin(angle), big - dist * np.cos(angle)
        return bx - self.car.cam_to_rear_axle_cm, sign * by

    def bounds(self) -> tuple[float, float]:
        """帯の lat の範囲（右の端, 左の端）。曲がるときは外側を前の角が振れる分だけ広げる。"""
        half = self.car.width_cm / 2 + self.car.margin_cm
        r = self.radius
        if math.isinf(r):
            return -half, half
        outer = math.hypot(abs(r) + half, self._front) - abs(r)
        return (-outer, half) if r > 0 else (-half, outer)

    def inside(self, s, lat, max_cm: float = MAX_CM) -> np.ndarray:
        lo, hi = self.bounds()
        s, lat = np.asarray(s), np.asarray(lat)
        top = min(max_cm, self.max_s)
        return (s >= 0) & (s <= top) & (lat >= lo) & (lat <= hi)


@dataclass
class Clearance:
    clear_cm: float | None  # 前の端から最初の障害物まで。何も無ければ見えている端（最大 MAX_CM）
    confidence: float  # 0..1
    blind_cm: float  # 前の端からこの距離までは見えていない（帯の幅の 90% が映るまで）
    method: str

    @property
    def out_of_view(self) -> bool:
        """通り道がそもそも画面に入っていない（大きく曲がっている・上を向いている）。

        確かめられなかったのではなく、見ていないので、Pi に止まれ（null）とは言わない。
        """
        return self.clear_cm is None and self.blind_cm > MAX_BLIND_CM

    def to_json(self) -> dict | None:
        if self.clear_cm is None:
            return None
        return {
            "cm": round(self.clear_cm, 1),
            "conf": round(self.confidence, 2),
            "blind_cm": round(self.blind_cm, 1),
        }


class Grid:
    """通り道の帯を真上から見た格子（cell_cm 角）と、各マスに映る画素の位置。

    tilt・pan・舵が同じなら使い回せるので、作ったものを覚えておく（cached）。
    """

    def __init__(self, cam: Camera, path: Path, max_cm: float = MAX_CM, cell_cm: float = 1.0):
        lo, hi = path.bounds()
        self.cam, self.path, self.cell_cm = cam, path, cell_cm
        self.s = np.arange(0.0, min(max_cm, path.max_s) + 1e-6, cell_cm)
        self.lat = np.arange(lo + cell_cm / 2, hi, cell_cm)
        s, lat = np.meshgrid(self.s, self.lat, indexing="ij")
        x, y = path.from_path(s, lat)
        u, v = project(cam, np.stack([x, y, np.zeros_like(x)], axis=-1))
        self.valid = (u >= 0) & (u <= cam.width - 1) & (v >= 0) & (v <= cam.height - 1)
        self.map_u = np.where(self.valid, u, -1).astype(np.float32)
        self.map_v = np.where(self.valid, v, -1).astype(np.float32)
        # 帯の幅のほとんど（90%）が映っている行だけを「見えている」とする。画角が 54° なので、
        # まっすぐでも帯の幅（23 cm）がおさまるのは前の端から 18 cm ほど先から。曲がると帯は
        # すぐ画面の横へ出ていくので、見えている端が近くなる（clear_cm もそこまで）
        rows = self.valid.mean(axis=1) >= 0.9
        if rows.any():
            first = int(np.argmax(rows))
            last = first + int(np.argmin(rows[first:])) if not rows[first:].all() else len(rows)
            self.blind_cm = float(self.s[first])
            self.seen_cm = float(self.s[last - 1])  # 手前から続けて見えている端
        else:
            self.blind_cm = self.seen_cm = math.inf

    _cache: dict = {}

    @classmethod
    def cached(cls, cam: Camera, path: Path, max_cm: float = MAX_CM, cell_cm: float = 1.0):
        # 角度は 0.5° に丸めて覚える（Pi の値は細かく揺れる）
        key = (
            round(cam.tilt_deg * 2) / 2,
            round(cam.pan_deg * 2) / 2,
            round(path.steer_deg * 2) / 2,
            cam.height_cm,
            cam.hfov_deg,
            cam.width,
            cam.height,
            path.car,
            max_cm,
            cell_cm,
        )
        grid = cls._cache.get(key)
        if grid is None:
            if len(cls._cache) > 256:
                cls._cache.clear()
            cam = Camera(cam.height_cm, key[0], key[1], cam.hfov_deg, cam.width, cam.height)
            grid = cls._cache[key] = cls(cam, Path(path.car, key[2]), max_cm, cell_cm)
        return grid

    def sample(self, image: np.ndarray) -> np.ndarray:
        """画像を格子に写す（映らないマスは 0）。形は (s の数, lat の数, チャンネル)。"""
        return cv2.remap(image, self.map_u, self.map_v, cv2.INTER_LINEAR, borderValue=0)

    def rays(self) -> dict:
        """深度の計算に使う、画素ごとの光線と床までの奥行き（作るのは 1 回だけ）。"""
        if getattr(self, "_rays", None) is None:
            cam = self.cam
            v, u = np.mgrid[0 : cam.height, 0 : cam.width]
            ray = pixel_rays(cam, u, v)
            down = -ray[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                zfloor = np.where(down > 1e-9, cam.height_cm / down, np.nan)
            s, lat = self.path.to_path(zfloor * ray[..., 0], zfloor * ray[..., 1])
            lo, hi = self.path.bounds()
            # 尺度を合わせる床の候補: 通り道とその両脇の、手前 FIT_CM まで
            fit = np.isfinite(s) & (s >= -10) & (s <= FIT_CM) & (lat >= lo - 15) & (lat <= hi + 15)
            self._rays = {"ray": ray.astype(np.float32), "zfloor": zfloor, "fit": fit}
        return self._rays


def first_blocked(weight: np.ndarray, s: np.ndarray, min_weight: float = 3) -> float | None:
    """格子（s の向き × lat の向き）の重みから、最初にふさがる s を返す。何も無ければ None。

    1 マスだけの点はノイズとみなし、周り 3x3 の重みの和が min_weight 以上の所だけを数える。
    床の見た目では重み = ふさがったマス（横に寝たケーブルも 3 マス並ぶので残る）、
    深度では重み = そのマスに落ちた点の数（立った脚の点は同じマスに積み重なる）。
    """
    weight = np.asarray(weight, np.float32)
    if not weight.any():
        return None
    sums = cv2.boxFilter(weight, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    rows = np.flatnonzero(((weight > 0) & (sums >= min_weight)).any(axis=1))
    return float(s[rows[0]]) if len(rows) else None


def _blind(grid: Grid, method: str) -> Clearance | None:
    """手前が見えなさすぎるときの答え（clear_cm は null）。見えていれば None。"""
    if grid.blind_cm > MAX_BLIND_CM:
        return Clearance(None, 0.0, min(grid.blind_cm, MAX_CM), method)
    return None


# 床の見た目: 違いの尺度は手前 REF_CM の床から取る。各マスは同じ列の LOCAL_CM（何 cm 手前から
# 何 cm 手前まで）の床と比べる。明るさは影や照り返しで変わるので軽く見る
REF_CM = 10.0
LOCAL_CM = (6.0, 12.0)
LAB_WEIGHT = np.array([0.5, 1.0, 1.0], np.float32)
LAB_MIN_SPREAD = np.array([8.0, 3.0, 3.0], np.float32)


class FloorClearance:
    """(b) 幾何と床の見た目。通り道に沿って、床の色が急に変わる所を探す。

    GPU は使わない。半分に縮めた画像で 0.5 ms ほど。各マスを、同じ列の LOCAL_CM 手前の床の
    平均と比べる（Lab、明るさは半分の重み）。手前の床 1 か所を見本にすると、奥へ行くにつれて
    暗くなる床や照り返しで止まりすぎた（実物 15 枚のうち 10 枚）。手前と比べれば、ゆっくり
    変わる影では変わらず、物の足もとのような急な変わり目だけが残る（3 枚）。
    違いの尺度（ばらつき）は手前 REF_CM の床から取る。
    床と色の近い物（暖かい光の下の白い箱、木の椅子の脚）は見落とすので、これだけには頼らない。
    深度では見えない、床に寝たケーブル（高さ 2 cm 未満）を見つけるのに使う。
    """

    method = "floor"

    def __init__(self, threshold: float = 4.0, scale: float = 0.5, max_cm: float = MAX_CM):
        self.threshold, self.scale, self.max_cm = threshold, scale, max_cm

    def blocked(self, image: np.ndarray, grid: Grid) -> tuple[np.ndarray, float]:
        """格子のマスごとに床らしくないか（と、確からしさ 0..1）。"""
        lab = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2Lab)
        cells = grid.sample(lab).astype(np.float32)
        valid = grid.valid
        ref = valid & (grid.s[:, None] < grid.blind_cm + REF_CM)
        if not ref.any():
            return np.zeros(valid.shape, bool), 0.0
        r = cells[ref]
        med = np.median(r, axis=0)
        spread = np.maximum(1.4826 * np.median(np.abs(r - med), axis=0), LAB_MIN_SPREAD)
        # 同じ列の LOCAL_CM 手前の平均（映らないマスは数えない）。格子の行は手前から奥へ並ぶ
        near, far = (round(c / grid.cell_cm) for c in LOCAL_CM)
        kernel = np.zeros((far + 1, 1), np.float32)
        kernel[: far - near + 1] = 1
        v = valid.astype(np.float32)
        den = cv2.filter2D(v, -1, kernel, anchor=(0, far), borderType=cv2.BORDER_CONSTANT)
        local = np.stack(
            [
                cv2.filter2D(cells[..., i] * v, -1, kernel, anchor=(0, far),
                             borderType=cv2.BORDER_CONSTANT)
                for i in range(3)
            ],
            axis=-1,
        ) / np.maximum(den, 1e-6)[..., None]  # fmt: skip
        dist = np.sqrt((((cells - local) / spread * LAB_WEIGHT) ** 2).sum(axis=-1))
        blocked = valid & (den >= 3) & (dist > self.threshold)
        # 見えている範囲でふさがったマスが多すぎる = 床ではない物で画面が埋まっているかもしれない
        busy = float(blocked[valid].mean()) if valid.any() else 1.0
        # 暗いと色が当てにならない（L は 0..255）
        dark = float(np.clip((med[0] - 20) / 40, 0.2, 1.0))
        return blocked, float(np.clip(1.0 - 2 * busy, 0.0, 1.0)) * dark

    def estimate(
        self, image: np.ndarray, cam: Camera, steer_deg: float = 0.0, car: Car | None = None
    ) -> Clearance:
        h, w = image.shape[:2]
        sw, sh = round(w * self.scale), round(h * self.scale)
        small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        grid = Grid.cached(cam.scaled(sw, sh), Path(car or Car(), steer_deg), self.max_cm)
        if (none := _blind(grid, self.method)) is not None:
            return none
        blocked, conf = self.blocked(small, grid)
        hit = first_blocked(blocked, grid.s)
        clear = grid.seen_cm if hit is None else min(hit, grid.seen_cm)
        return Clearance(clear, conf, grid.blind_cm, self.method)


# 深度: 床に合わせる範囲（前の端から何 cm まで）と、障害物とみなす高さ（cm）
FIT_CM = 60.0
MIN_HEIGHT_CM = 2.0
HEIGHT_MARGIN_CM = 3.0


def fit_floor(inv_z: np.ndarray, disp: np.ndarray, iters: int = 4) -> tuple[float, float, float]:
    """床の画素で disp = a / z + b を当てる（相対の深度は視差の一次式なので）。

    手前に物があっても床が多ければ外れるよう、残差の大きい点を除いて当て直す。
    (a, b, 当てはまった点の割合) を返す。
    """
    keep = np.ones(len(inv_z), bool)
    a = b = 0.0
    for _ in range(iters):
        if keep.sum() < 10:
            return 0.0, 0.0, 0.0
        # 2 つの数の当てはめなので式で解く（lstsq より 1 桁速い）
        x, y = inv_z[keep], disp[keep]
        dx = x - x.mean()
        a = float((dx * (y - y.mean())).sum() / max(float((dx * dx).sum()), 1e-12))
        b = float(y.mean() - a * x.mean())
        r = np.abs(disp - (a * inv_z + b))
        mad = 1.4826 * float(np.median(r[keep]))
        keep = r < max(3 * mad, 0.01 * abs(float(np.median(disp))))
    return float(a), float(b), float(keep.mean())


def clearance_from_disparity(
    disp: np.ndarray, cam: Camera, path: Path, max_cm: float = MAX_CM, min_points: int = 4
) -> Clearance:
    """相対の深度（視差。大きいほど近い）から clear_cm を出す。GPU は使わない。

    床の幾何で尺度を合わせ、画素ごとに点の位置と床からの高さを出し、通り道の中で
    MIN_HEIGHT_CM〜車の高さ + 余裕 の点が min_points 以上集まった所を障害物とする。
    """
    h, w = disp.shape
    grid = Grid.cached(cam.scaled(w, h), path, max_cm)
    if (none := _blind(grid, "depth")) is not None:
        return none
    g = grid.rays()
    fit = g["fit"]
    if fit.sum() < 50:
        return Clearance(None, 0.0, grid.blind_cm, "depth")
    a, b, inliers = fit_floor(1 / g["zfloor"][fit], disp[fit].astype(np.float64))
    if a <= 0:
        return Clearance(None, 0.0, grid.blind_cm, "depth")
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(disp - b > 1e-6, a / (disp - b), np.inf)
    ray = g["ray"]
    near = np.isfinite(z) & (z < 3 * max_cm)
    z = z[near]
    height = cam.height_cm + z * ray[..., 2][near]
    s, lat = path.to_path(z * ray[..., 0][near], z * ray[..., 1][near])
    car = path.car
    ob = (height >= MIN_HEIGHT_CM) & (height <= car.height_cm + HEIGHT_MARGIN_CM)
    ob &= path.inside(s, lat, max_cm)
    ns, nl = grid.valid.shape
    si = np.clip(np.round((s[ob] - grid.s[0]) / grid.cell_cm).astype(int), 0, ns - 1)
    li = np.clip(np.round((lat[ob] - grid.lat[0]) / grid.cell_cm).astype(int), 0, nl - 1)
    counts = np.bincount(si * nl + li, minlength=ns * nl).reshape(ns, nl)
    hit = first_blocked(counts, grid.s, min_points)
    clear = grid.seen_cm if hit is None else min(hit, grid.seen_cm)
    return Clearance(clear, inliers, grid.blind_cm, "depth")


class DepthClearance:
    """(a) 単眼の深度（Depth Anything V2 Small、相対。Apache-2.0）。

    metric（Hypersim で学習した indoor）の版も試したが、目の高さで撮った部屋を覚えているので、
    床から 10 cm の視点では奥行きを 2〜7 倍に見積もり、遠くの床を持ち上げてしまった。
    相対の版の視差を床の幾何に当てはめる方が、床がまっすぐ平らに出た。
    torch.compile（HIP グラフ）で 434x322 が 3.5 ms（compile なしは 8 ms）。

    検出と GPU を分け合うので、GPU の仕事は自分の stream に流す。同じ（既定の）stream に流すと、
    検出の仕事がこちらの仕事の後ろに並んで待たされる。compile したときの HIP グラフは、記録した
    スレッドでしか使い回せないので、作るのも呼ぶのも同じスレッドで行う。
    """

    MODEL = os.getenv("SNOO_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
    method = "depth"

    def __init__(self, model: str = MODEL, size: tuple[int, int] = (434, 322), compile=False):
        import torch
        from transformers import AutoModelForDepthEstimation

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(model)
        self.model = self.model.to(self.device, self.dtype).eval()
        self.size = size  # (幅, 高さ)。14 の倍数（ViT のパッチ）
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.stream = torch.cuda.Stream() if self.device == "cuda" else None
        self.forward = self.model
        if compile and self.device == "cuda":
            self.forward = torch.compile(self.model, mode="reduce-overhead", dynamic=False)
        for _ in range(3 if compile else 1):
            self.disparity(np.zeros((480, 640, 3), np.uint8))

    def disparity(self, image: np.ndarray) -> np.ndarray:
        """BGR の画像の相対の視差（半分の大きさ。大きいほど近い）。"""
        torch = self.torch
        stream = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        with stream, torch.inference_mode():
            x = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(self.device)
            x = x.permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255)
            w, h = self.size
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", antialias=True)
            d = self.forward(pixel_values=(x - self.mean) / self.std).predicted_depth
            # 点の計算は CPU でするので、半分にして渡す量と計算を減らす
            d = torch.nn.functional.avg_pool2d(d.unsqueeze(1).float(), 2)
            return d[0, 0].cpu().numpy()  # この stream が終わるまで待つ

    def estimate(
        self, image: np.ndarray, cam: Camera, steer_deg: float = 0.0, car: Car | None = None
    ) -> Clearance:
        path = Path(car or Car(), steer_deg)
        return clearance_from_disparity(self.disparity(image), cam, path)


def combine(depth: Clearance, floor: Clearance) -> Clearance:
    """(a) と (b) の近い方。深度では見えない寝たケーブルを (b) で拾う。

    深度が出せない（見えない・床に合わない）ときは、(b) だけでは箱を見落とすので null にする。
    """
    if depth.clear_cm is None or floor.clear_cm is None:
        return Clearance(None, 0.0, max(depth.blind_cm, floor.blind_cm), "both")
    return Clearance(
        min(depth.clear_cm, floor.clear_cm),
        min(depth.confidence, floor.confidence),
        max(depth.blind_cm, floor.blind_cm),
        "both",
    )


class Estimator:
    """走っている間に使う見積もり。(a) と (b) の近い方（combine）。

    Pi が送る tilt・pan（指令）と舵から、カメラの向きと通り道を決める。探索中の Pi は
    曲がる向きに pan を振る（pan = 舵、±30° まで）ので、pan は必ずヘッダーの値を使う。
    """

    def __init__(self, compile: bool = False, height_cm: float = CAM_HEIGHT_CM):
        self.height_cm = height_cm
        self.depth = DepthClearance(compile=compile)
        self.floor = FloorClearance()

    def estimate(
        self, image: np.ndarray, tilt_deg: float, pan_deg: float, steer_deg: float
    ) -> Clearance:
        h, w = image.shape[:2]
        cam = Camera.from_pi(tilt_deg, pan_deg, height_cm=self.height_cm, width=w, height=h)
        return combine(
            self.depth.estimate(image, cam, steer_deg), self.floor.estimate(image, cam, steer_deg)
        )
