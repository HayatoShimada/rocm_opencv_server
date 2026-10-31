"""次の 1 歩の計画器（app/snoo_maneuver.py）のコストの重みを、場面から出す小さな方策。

計画器は一つのコスト関数で 1 歩を選ぶ。重みは手で決めた固定の値（snoo_maneuver.FIXED）で、
白い壁の部屋でも、動けないときも、猫がそばに居るときも同じだった。ここでは、場面の特徴
（FEATURES）から、重みに掛ける倍率を出す。

- 学ぶもの（KNOBS）：行き先までの道のり・見えるもの・目印が見えない罰・同じ所へ戻る罰・
  前後の切り替え・下がる長さ・前へ走る長さ・隙間を抜ける得・たぶん空きの床の得の重みと、
  確信度の低い縁の後回しの道のり、MPPI の温度。倍率は exp(z)、z は ±Z_MAX まで
- 学ばないもの：安全の項（ぶつかる・後ろの確かでない所・前の見ていない所・Pi の clear_cm で
  止まる道・人と snoo のそばへ下がらない）。計画器の中で固定のまま
- 形：z = Z_MAX × tanh(W f + b)。f は FEATURES の数（どれも 0〜1 ほど）。W・b が 0 なら
  倍率は 1 で、固定の重みと同じ（学ぶ前の形）
- 推論は numpy だけ（地図のサーバーで torch を読まない）。重みは npz（W・b・features・knobs）。
  scripts/snoo_explore_train.py が書く。地図のサーバーは環境変数 SNOO_MANEUVER_POLICY の
  ファイルを読む（app/snoo_mapper.MapParams.policy。無ければ固定の重み）
"""

from __future__ import annotations

import dataclasses
import functools
import math
from pathlib import Path

import numpy as np

from app.roommap import FREE, HALF_VIEW_DEG, UNKNOWN, VIEW_RANGE_M, Grid, Pose
from app.snoo_belief import CONF_OK, LANDMARK_MIN, visible_landmarks
from app.snoo_maneuver import CAT_RECENT_S, FIXED, Weights

# 重みの名前（snoo_maneuver.Weights の欄）。この順に z を並べる
KNOBS = ("goal", "info", "lost", "revisit", "switch", "rev", "fwd", "gap", "maybe",
         "lowconf_offset_m", "lam")  # fmt: skip
N_KNOBS = len(KNOBS)
# 倍率 exp(z) の z の上限（exp(1.5) = 4.5 倍、exp(-1.5) = 0.22 倍）
Z_MAX = 1.5
FEATURES = (
    "conf_mean",  # 床と見たセルの確信度の平均
    "lowconf_frac",  # 床と見たセルのうち、確信度の低い（CONF_OK 未満）割合
    "landmarks_here",  # 今の所から見える目印の数 / LANDMARK_MIN（2 まで）
    "landmarks_all",  # 地図の目印の数 / 50（2 まで）
    "stuck",  # 動けていない（同じ所に固まっている）
    "still",  # 続けてほとんど動けなかった
    "unknown_near",  # 1.5 m 以内の見ていない割合
    "known_m2",  # 床と分かった広さ / 50 m²（2 まで）
    "cat_near",  # 最近見た人・snoo までの近さ exp(-d / 1 m)（見ていなければ 0）
    "stations",  # 撮った所の数 / 40（2 まで）
)
N_FEATURES = len(FEATURES)
NEAR_M = 1.5


def features(
    grid: Grid,
    pose: Pose,
    recent: list[Pose],
    state: np.ndarray,
    conf: np.ndarray,
    n_stations: int,
) -> np.ndarray:
    """場面の特徴（N_FEATURES,）。recent は今を含む直近のステーションの位置。"""
    from app.snoo_mapper import is_still, is_stuck

    floor = state == FREE
    nf = int(floor.sum())
    conf_mean = float(conf[floor].mean()) if nf else 0.0
    low = float((conf[floor] < CONF_OK).mean()) if nf else 1.0
    tex = getattr(grid, "texture", None)
    lm = tex.landmarks(grid) if tex is not None else np.zeros((0, 2))
    here = float(visible_landmarks(lm, pose.x, pose.y, pose.yaw, HALF_VIEW_DEG, VIEW_RANGE_M))
    ix, iy = grid.cell(pose.x, pose.y)
    r = int(math.ceil(NEAR_M / grid.res))
    x0, x1 = max(int(ix) - r, 0), min(int(ix) + r + 1, grid.n)
    y0, y1 = max(int(iy) - r, 0), min(int(iy) + r + 1, grid.n)
    xs, ys = np.mgrid[x0:x1, y0:y1]
    disk = (xs - ix) ** 2 + (ys - iy) ** 2 <= r * r
    sub = state[x0:x1, y0:y1]
    unk = float((sub[disk] == UNKNOWN).mean()) if disk.any() else 1.0
    now = getattr(grid, "now_s", None)
    cats = [(x, y) for x, y, t in getattr(grid, "cats", None) or []
            if now is not None and now - t <= CAT_RECENT_S]  # fmt: skip
    d = min((math.hypot(x - pose.x, y - pose.y) for x, y in cats), default=math.inf)
    f = [
        conf_mean,
        low,
        min(here / LANDMARK_MIN, 2.0),
        min(len(lm) / 50.0, 2.0),
        float(is_stuck(recent)),
        float(is_still(recent)),
        unk,
        min(nf * grid.res**2 / 50.0, 2.0),
        math.exp(-d) if math.isfinite(d) else 0.0,
        min(n_stations / 40.0, 2.0),
    ]
    return np.asarray(f, np.float64)


def to_weights(z: np.ndarray, base: Weights = FIXED) -> Weights:
    """z（N_KNOBS,）を重みに：固定の重み × exp(z)。z は ±Z_MAX に収める。"""
    z = np.clip(np.asarray(z, float), -Z_MAX, Z_MAX)
    return dataclasses.replace(
        base, **{k: getattr(base, k) * math.exp(float(v)) for k, v in zip(KNOBS, z, strict=True)}
    )


class WeightPolicy:
    """z = Z_MAX × tanh(W f + b) の方策（numpy だけ）。"""

    def __init__(self, W: np.ndarray | None = None, b: np.ndarray | None = None):
        self.W = np.zeros((N_KNOBS, N_FEATURES)) if W is None else np.asarray(W, float)
        self.b = np.zeros(N_KNOBS) if b is None else np.asarray(b, float)
        if self.W.shape != (N_KNOBS, N_FEATURES) or self.b.shape != (N_KNOBS,):
            raise ValueError(f"重みの形が合わない: W {self.W.shape}・b {self.b.shape}")

    # 平らにした重み（学ぶとき）
    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b])

    @classmethod
    def from_theta(cls, theta: np.ndarray) -> WeightPolicy:
        theta = np.asarray(theta, float)
        n = N_KNOBS * N_FEATURES
        return cls(theta[:n].reshape(N_KNOBS, N_FEATURES), theta[n:])

    def mean(self, f: np.ndarray) -> np.ndarray:
        return Z_MAX * np.tanh(self.W @ f + self.b)

    def weights(self, f: np.ndarray) -> Weights:
        return to_weights(self.mean(f))

    def save(self, path: str | Path, **extra) -> None:
        path = Path(path)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, W=self.W, b=self.b, features=np.array(FEATURES), knobs=np.array(KNOBS),
                 **extra)  # fmt: skip
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> WeightPolicy:
        with np.load(path) as d:
            if tuple(d["features"]) != FEATURES or tuple(d["knobs"]) != KNOBS:
                raise ValueError(f"特徴・重みの名前が今の版と違う: {path}")
            return cls(d["W"], d["b"])


@functools.lru_cache(maxsize=4)
def load_policy(path: str) -> WeightPolicy:
    """重みのファイルを 1 回だけ読む（地図のサーバーが 1 歩ごとに読み直さない）。"""
    return WeightPolicy.load(path)


def explain(f: np.ndarray, w: Weights) -> dict:
    """返事・ログに付ける、特徴と固定からの倍率（小さく丸める）。"""
    return {
        "features": {k: round(float(v), 3) for k, v in zip(FEATURES, f, strict=True)},
        "scale": {k: round(getattr(w, k) / getattr(FIXED, k), 3) for k in KNOBS},
    }
