"""RoMa v2 の密な対応から、2 か所のステーションの相対の位置（SE(2)）を出し、位置のグラフの辺にする。

VGGT の窓は、白い壁と暗い所で位置合わせが崩れる（explore_20261030_190244 では自分の窓で崩れた
ステーションが 11 か所）。崩れたステーションは走った辺だけで決まり、向きがいちばん当てにならない
（走った 1 歩の向きのずれは中央値で 14° ほど）。RoMa v2 は、少ない模様でも密に対応を出すので、
ステーションどうしの向きを別に測れる。

1 組のコマ（a のコマと b のコマ）から:
- 対応（画素）から基本行列を USAC_MAGSAC で求め、カメラの相対の回転と、向きだけの動き（長さ 1）
  を出す
- コマのパン・チルトで、カメラの座標を車の座標（x は前、y は左、z は上。原点はカメラの真下の床）
  に直す
- 大きさは床で決める：対応した点のうち地平線より下の点を三角測量し、いちばん多く乗る水平な面を床と
  みなして、カメラから床までが 8 cm（camera_height_cm）になるよう伸ばす。走った長さ（Pi の見積もり）
  は 0.6 倍のこともあったので使わない（使うと走った辺と同じずれが入り、辺を足す意味が薄れる）
- 首を振っただけでほとんど動いていない組（視差が小さい）は、基本行列の動きの向きが決まらないので、
  回転だけを使い、動きは 0（確からしさは大きめ）とする
- 車の座標の相対の位置を、床の上の (x, y, yaw) にする

ステーションの組では、重なりそうなコマの組（パンの差が向きの見込みに近いもの）を数組通し、
組ごとの答えを重み付きの中央値でまとめる。組どうしのばらつきと、残った対応の数から確からしさを
決める。辺の kind は "roma"。窓の辺と同じに外れ値として外せる（app/snoo_posegraph.optimize）。

位置を出す所は numpy と OpenCV だけ（GPU を使わないのでテストできる）。RoMa v2 を回す所
（load_model・match_pairs）は torch を使うときだけ読む。地図のサーバー（SNOO_ROMA_EDGES=1）は
同じプロセスで、比べるとき（scripts/snoo_roma_match.py）は別の venv（~/orca/roma-venv）で回す。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

from app.roommap import Pose

# カメラ（PiCar-X の OV5647）。640x480、横の画角 54°、画素は正方形、中心は画像の中心
IMAGE_W, IMAGE_H = 640, 480
HFOV_DEG = 54.0
CAMERA_HEIGHT_M = 0.08
# 基本行列の RANSAC のしきい値（画素）
RANSAC_PX = 1.0
# 1 組のコマで、残った対応がこれより少なければ使わない
MIN_INLIERS = 60
# 床とみなす面：法線が真上から MAX_FLOOR_TILT_DEG 以内、点の数が MIN_FLOOR_POINTS 以上
MAX_FLOOR_TILT_DEG = 15.0
MIN_FLOOR_POINTS = 25
# 床の面から、カメラまでの距離のこの割合以内の点を面に乗っているとみなす
FLOOR_TOL = 0.06
# 床の大きさを決める点：カメラから見て地平線よりこの割合（sin）以上下にあるもの
FLOOR_MIN_DROP = 0.05
# 床の大きさの決め方。vertical は床を水平と決める、plane は 3 点の面を RANSAC で探す
FLOOR_MODE = "vertical"
# 回転だけで説明できる（視差がこれ未満）組は、動きを使わない
MIN_PARALLAX_DEG = 1.0
# 回転だけかを見るとき：K⁻¹HK の特異値の比の上限と、回転で説明できるとみなす角度（°）
ROTATION_SV_RATIO = 1.05
ROTATION_TOL_DEG = 0.3
# 車の座標の相対の回転の、水平からの傾きがこれを超える組は捨てる（チルトの校正と合わない）
MAX_ROLL_PITCH_DEG = 12.0
# 確からしさ（1σ）
ROMA_SIGMA_M = 0.03
ROMA_SIGMA_PER_M = 0.10
ROMA_SIGMA_DEG = 1.5
ROMA_SIGMA_DEG_PER_SQRT_N = 40.0  # 残った対応の数 n の √ で割る
ROTATION_ONLY_SIGMA_M = 0.06
# ステーションの組の答えにするのに要る、向きのそろったコマの組の数と、対応の数の合計
MIN_AGREEING_PAIRS = 2
MIN_TOTAL_INLIERS = 2000
AGREE_DEG = 10.0


def intrinsics(w: int = IMAGE_W, h: int = IMAGE_H, hfov_deg: float = HFOV_DEG) -> np.ndarray:
    f = (w / 2) / math.tan(math.radians(hfov_deg / 2))
    return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


def cam_to_car(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """カメラの座標（OpenCV：x は右、y は下、z は前）を、車の座標（x は前、y は左、z は上）へ
    回す行列。
    パンは右が +（Pi と同じ）、チルトは上が +。パンの台の上にチルトが載っている。
    """
    base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    t, p = math.radians(tilt_deg), math.radians(-pan_deg)
    # チルト：y 軸（左）まわりに -t（前を上へ）
    ry = np.array(
        [[math.cos(t), 0.0, -math.sin(t)], [0.0, 1.0, 0.0], [math.sin(t), 0.0, math.cos(t)]]
    )
    rz = np.array(
        [[math.cos(p), -math.sin(p), 0.0], [math.sin(p), math.cos(p), 0.0], [0.0, 0.0, 1.0]]
    )
    return rz @ ry @ base


@dataclass
class PairEstimate:
    """1 組のコマから出した、a の車から見た b の車の位置。"""

    pose: Pose
    inliers: int
    floor_points: int
    parallax_deg: float
    rotation_only: bool
    scale: float  # 長さ 1 の動きに掛けた大きさ（m）
    tilt_deg: float  # 車の相対の回転の、水平からの傾き（確かめ用）


def _floor_scale(Y: np.ndarray, height_m: float, rng: np.random.Generator) -> tuple[float, int]:
    """Y（車の向きに回した、カメラからの点。大きさは決まっていない）から床の面を探し、
    カメラから床までの距離が height_m になる大きさと、床に乗った点の数を返す。
    見つからなければ (0, 0)。
    床はいちばん低い大きな面なので、法線が上向きで、点がカメラより下にある面だけを見る。
    """
    below = Y[Y[:, 2] < 0]
    if len(below) < MIN_FLOOR_POINTS:
        return 0.0, 0
    cos_max = math.cos(math.radians(MAX_FLOOR_TILT_DEG))
    best_n, best_d, best_cnt = None, 0.0, 0
    for _ in range(200):
        p = below[rng.choice(len(below), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        if n[2] < 0:
            n = -n
        if n[2] < cos_max:
            continue
        d = -float(n @ p[0])  # 面は n·x + d = 0。カメラ（原点）から面までの距離が d
        if d <= 0:
            continue
        cnt = int((np.abs(below @ n + d) < FLOOR_TOL * d).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    if best_n is None or best_cnt < MIN_FLOOR_POINTS:
        return 0.0, 0
    # 乗った点で面を合わせ直す（最小二乗の面。3 点の面の傾きのずれが、そのまま大きさのずれになる）
    n, d = best_n, best_d
    for _ in range(2):
        on = below[np.abs(below @ n + d) < FLOOR_TOL * d]
        if len(on) < MIN_FLOOR_POINTS:
            return 0.0, 0
        m = on.mean(0)
        n = np.linalg.svd(on - m, full_matrices=False)[2][2]
        n = -n if n[2] < 0 else n
        if n[2] < cos_max:
            return 0.0, 0
        d = -float(n @ m)
        if d <= 0:
            return 0.0, 0
    return height_m / d, len(on)


def _floor_scale_vertical(Y: np.ndarray, height_m: float) -> tuple[float, int]:
    """床が水平（法線が真上）と決めて大きさを出す。地平線より下の点ごとに「この点が床なら」の
    大きさ h / (-z) を出し、いちばん多く集まる大きさを取る。床より上の物（壁の下・箱）の点は
    大きさが大きい側へ散るので、集まりにはならない。面の向きを決めなくてよいので、
    3 点の面（_floor_scale）より点が少なくても崩れにくい。
    """
    r = np.linalg.norm(Y, axis=1)
    below = Y[Y[:, 2] < -FLOOR_MIN_DROP * r]
    if len(below) < MIN_FLOOR_POINTS:
        return 0.0, 0
    ls = np.log(height_m / -below[:, 2])
    # ±FLOOR_TOL の幅の窓を滑らせ、いちばん多く入る所
    ls.sort()
    w = math.log(1 + FLOOR_TOL)
    hi = np.searchsorted(ls, ls + 2 * w)
    k = int(np.argmax(hi - np.arange(len(ls))))
    sel = ls[k : hi[k]]
    if len(sel) < MIN_FLOOR_POINTS:
        return 0.0, 0
    return float(math.exp(np.median(sel))), len(sel)


def pair_pose(
    ka: np.ndarray,
    kb: np.ndarray,
    pan_a: float,
    tilt_a: float,
    pan_b: float,
    tilt_b: float,
    K: np.ndarray | None = None,
    height_m: float = CAMERA_HEIGHT_M,
    seed: int = 0,
) -> PairEstimate | None:
    """a のコマの画素 ka と b のコマの画素 kb（同じ順の対応）から、a の車から見た b の車の位置。
    使えなければ None（対応が少ない・床が見つからない・回転が水平でない）。
    """
    K = intrinsics() if K is None else K
    ka = np.asarray(ka, np.float64).reshape(-1, 2)
    kb = np.asarray(kb, np.float64).reshape(-1, 2)
    if len(ka) < MIN_INLIERS:
        return None
    Ra, Rb = cam_to_car(pan_a, tilt_a), cam_to_car(pan_b, tilt_b)
    c = np.array([0.0, 0.0, height_m])  # 車の原点（カメラの真下の床）から見たカメラ
    Kinv = np.linalg.inv(K)
    ba = (Kinv @ np.c_[ka, np.ones(len(ka))].T).T
    bb = (Kinv @ np.c_[kb, np.ones(len(kb))].T).T
    ba /= np.linalg.norm(ba, axis=1, keepdims=True)
    bb /= np.linalg.norm(bb, axis=1, keepdims=True)

    def angles(R: np.ndarray) -> np.ndarray:
        """a の向きを R で回したものと b の向きの角度（°）。回転だけで説明できる対応は 0 に近い。"""
        return np.degrees(np.arccos(np.clip(np.sum((ba @ R.T) * bb, 1), -1, 1)))

    E, mask = cv2.findEssentialMat(ka, kb, K, method=cv2.USAC_MAGSAC, prob=0.9999,
                                   threshold=RANSAC_PX, maxIters=5000)  # fmt: skip
    rot = _rotation_only(ka, kb, K, ba, bb)
    if rot is not None:
        R, keep = rot
        # 回転だけで、基本行列と同じくらいの対応が説明できる：動きは決まらないので、
        # カメラが同じ所にいるとする（首を振っただけ・その場で回っただけ）
        n_e = 0 if mask is None else int(mask.sum())
        if keep.sum() >= max(MIN_INLIERS, 0.8 * n_e):
            return _from_rotation(R, int(keep.sum()), float(np.median(angles(R)[keep])), Ra, Rb, c)
    if E is None or E.shape != (3, 3):
        return None
    n_in, R, t, mask2 = cv2.recoverPose(E, ka, kb, K, mask=mask.copy())
    good = mask2.ravel() > 0
    if n_in < MIN_INLIERS:
        return None
    parallax = float(np.median(angles(R)[good]))
    if parallax < MIN_PARALLAX_DEG:
        return _from_rotation(R, int(n_in), parallax, Ra, Rb, c)
    Rab = Ra @ R.T @ Rb.T  # 車の座標の相対の回転（a の車から見た b の車の向き）
    tilt = _tilt(Rab)
    if tilt > MAX_ROLL_PITCH_DEG:
        return None
    yaw = math.atan2(Rab[1, 0], Rab[0, 0])
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, ka[good].T, kb[good].T)
    X = (X[:3] / X[3]).T
    front = (X[:, 2] > 0) & ((X @ R.T + t.reshape(1, 3))[:, 2] > 0)
    Y = X[front] @ Ra.T  # 車の向き、原点はカメラ
    if FLOOR_MODE == "vertical":
        scale, n_floor = _floor_scale_vertical(Y, height_m)
    else:
        scale, n_floor = _floor_scale(Y, height_m, np.random.default_rng(seed))
    if scale <= 0:
        return None
    # b のカメラの位置（a のカメラの座標）は -Rᵀt
    tr = c - Rab @ c + scale * (Ra @ (-R.T @ t.reshape(3)))
    return PairEstimate(Pose(float(tr[0]), float(tr[1]), yaw), int(n_in), n_floor, parallax,
                        False, float(scale), tilt)  # fmt: skip


def _tilt(Rab: np.ndarray) -> float:
    return math.degrees(math.acos(min(1.0, max(-1.0, float(Rab[2, 2])))))


def _rotation_only(ka, kb, K, ba, bb) -> tuple[np.ndarray, np.ndarray] | None:
    """対応が回転だけで説明できるか。ホモグラフィ H を求め、K⁻¹HK が回転に近ければ
    (その回転, 回転で説明できる対応)。床だけが写っていても H は合うが、そのときは K⁻¹HK が
    回転にならない（特異値がそろわない）ので区別できる。
    """
    H, hm = cv2.findHomography(ka, kb, cv2.USAC_MAGSAC, 2.0, maxIters=5000, confidence=0.9999)
    if H is None:
        return None
    M = np.linalg.inv(K) @ H @ K
    U, S, Vt = np.linalg.svd(M)
    if S[0] / S[2] > ROTATION_SV_RATIO:
        return None
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    # 回転を対応で合わせ直す（Kabsch）
    for _ in range(2):
        keep = np.degrees(np.arccos(np.clip(np.sum((ba @ R.T) * bb, 1), -1, 1))) < ROTATION_TOL_DEG
        if keep.sum() < MIN_INLIERS:
            return None
        u, _, vt = np.linalg.svd(bb[keep].T @ ba[keep])
        R = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    keep = np.degrees(np.arccos(np.clip(np.sum((ba @ R.T) * bb, 1), -1, 1))) < ROTATION_TOL_DEG
    return R, keep


def _from_rotation(R, n, parallax, Ra, Rb, c) -> PairEstimate | None:
    Rab = Ra @ R.T @ Rb.T
    tilt = _tilt(Rab)
    if tilt > MAX_ROLL_PITCH_DEG:
        return None
    yaw = math.atan2(Rab[1, 0], Rab[0, 0])
    tr = c - Rab @ c
    return PairEstimate(Pose(float(tr[0]), float(tr[1]), yaw), n, 0, parallax, True, 0.0, tilt)


@dataclass
class StationEstimate:
    """ステーションの組の答え（コマの組をまとめたもの）。"""

    pose: Pose
    sigma_m: float
    sigma_rad: float
    pairs: int  # 使えたコマの組
    inliers: int  # 使えた組の対応の数の合計
    rotation_only: bool


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def combine(
    estimates: list[PairEstimate | None],
    min_pairs: int = MIN_AGREEING_PAIRS,
    min_inliers: int = MIN_TOTAL_INLIERS,
) -> StationEstimate | None:
    """コマの組の答えを、残った対応の数で重み付けた中央値でまとめる。
    向きが中央値から AGREE_DEG 以内の組だけを使い、それが min_pairs 組・対応が合わせて
    min_inliers に足りなければ None（1 組だけの答えは、似た別の所との取り違えのことがある。
    保存したセッションでは、近くに戻ってきた組の候補のうち 1 組だけのものは半分以上が
    外れ値だった）。
    動きの決まった組があれば、回転だけの組は向きにだけ使う。
    """
    est = [e for e in estimates if e is not None]
    if not est:
        return None
    w = np.array([e.inliers for e in est], float)
    yaws = np.array([e.pose.yaw for e in est])
    y0 = _wmedian(yaws, w)
    yaws = y0 + np.arctan2(np.sin(yaws - y0), np.cos(yaws - y0))  # 中央値のまわりに折り返す
    agree = np.abs(yaws - y0) <= math.radians(AGREE_DEG)
    est = [e for e, ok in zip(est, agree, strict=True) if ok]
    w, yaws = w[agree], yaws[agree]
    if len(est) < min_pairs or w.sum() < min_inliers:
        return None
    yaw = _wmedian(yaws, w)
    moved = [e for e in est if not e.rotation_only]
    rot_only = not moved
    use = moved or est
    wu = np.array([e.inliers for e in use], float)
    x = _wmedian(np.array([e.pose.x for e in use]), wu)
    y = _wmedian(np.array([e.pose.y for e in use]), wu)
    n = int(w.sum())
    # 組どうしのばらつき（1 組だけなら 0）
    yaw_spread = float(np.sqrt(np.average((yaws - yaw) ** 2, weights=w))) if len(est) > 1 else 0.0
    xy = np.array([[e.pose.x, e.pose.y] for e in use])
    t_spread = (float(np.sqrt(np.average(np.sum((xy - [x, y]) ** 2, 1), weights=wu)))
                if len(use) > 1 else 0.0)  # fmt: skip
    dist = math.hypot(x, y)
    sm = ROMA_SIGMA_M + ROMA_SIGMA_PER_M * dist + t_spread
    if rot_only:
        sm = max(sm, ROTATION_ONLY_SIGMA_M)
    if len(use) == 1:
        sm *= 1.5  # 確かめる相手が無い
    sr = (
        math.radians(ROMA_SIGMA_DEG + ROMA_SIGMA_DEG_PER_SQRT_N / math.sqrt(max(n, 1))) + yaw_spread
    )
    return StationEstimate(Pose(x, y, math.atan2(math.sin(yaw), math.cos(yaw))), sm, sr,
                           len(est), n, rot_only)  # fmt: skip


def pick_frame_pairs(
    frames_a: list[dict], frames_b: list[dict], dyaw_deg: float, k: int = 4
) -> list[tuple[dict, dict]]:
    """重なりそうなコマの組を k 組。カメラの向き（車の向き − パン）がそろう組ほど重なるので、
    同じチルトで、パンの差が向きの見込み dyaw_deg（b の車の向き − a の車の向き、左が +）に
    近いものから選ぶ。チルトの段ごとに交互に取る（床の見える下の段と、遠くの見える段の両方を使う）。
    """
    by_tilt: dict[int, list[tuple[float, dict, dict]]] = {}
    for fa in frames_a:
        for fb in frames_b:
            if round(fa["tilt_deg"]) != round(fb["tilt_deg"]):
                continue
            # 同じ向きを見るには pan_b = pan_a + dyaw
            cost = abs(fb["pan_deg"] - fa["pan_deg"] - dyaw_deg) + 0.05 * abs(fa["pan_deg"])
            by_tilt.setdefault(round(fa["tilt_deg"]), []).append((cost, fa, fb))
    rows = [sorted(v, key=lambda t: t[0]) for _, v in sorted(by_tilt.items())]
    out: list[tuple[dict, dict]] = []
    used: set[tuple[str, str]] = set()
    i = 0
    while len(out) < k and any(i < len(r) for r in rows):
        for r in rows:
            if i < len(r) and len(out) < k:
                _, fa, fb = r[i]
                key = (fa["file"], fb["file"])
                if key not in used:
                    used.add(key)
                    out.append((fa, fb))
        i += 1
    return out


def to_edge(a: int, b: int, st: StationEstimate, source: str = ""):
    """ステーションの組の答えを、位置のグラフの辺（kind "roma"）にする。"""
    from app.snoo_posegraph import Edge

    return Edge(a, b, st.pose, st.sigma_m, st.sigma_rad, "roma", source or f"roma:{a:03d}_{b:03d}")


# --- RoMa v2 を回す（GPU。torch は使うときだけ読む） ---

ROMA_SRC = os.path.expanduser(os.getenv("SNOO_ROMA_SRC", "~/orca/ext/romav2-src/src"))
# 640 の正方形に縮めて通す。320（turbo）では 12 組中 3 組で違う辺を出した（2026-10-31）。
# 800 + 1280（precise）は 1 組 2.2 秒・10 GB で、地図のサーバーの待ちに入らない
ROMA_SETTING = os.getenv("SNOO_ROMA_SETTING", "base")


SETTINGS = ("turbo", "fast", "base", "precise")


def load_model(setting: str = ROMA_SETTING):
    """RoMa v2 を読む。venv に入っていなければ ROMA_SRC（clone した src）から読む
    （地図のサーバーの .venv には torch・einops・pillow・rich・tqdm がもうあるので、入れずに済む）。
    重み（約 1.5 GB）は torch.hub の置き場に 1 回だけ落とす。
    """
    import sys

    try:
        import romav2  # noqa: F401
    except ImportError:
        if ROMA_SRC not in sys.path:
            sys.path.insert(0, ROMA_SRC)
    from romav2 import RoMaV2

    model = RoMaV2(RoMaV2.Cfg(setting=setting))
    return model


def _image(path: str):
    import torch
    from PIL import Image

    img = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0), img.shape[:2]


def match_pairs(model, pairs: list[tuple[str, str]], samples: int = 5000) -> list[dict]:
    """コマの組ごとの対応（画素）。組はまとめて 1 回で通せる（同じ大きさの画像なので）。
    まとめると GPU の電力が上がるので、ふつうは 1 組ずつ渡す。"""
    import torch
    import torch.nn.functional as F
    from romav2.romav2 import _map_confidence

    if not pairs:
        return []
    dev = next(model.parameters()).device
    A, B, sizes = [], [], []
    for pa, pb in pairs:
        ia, sa = _image(pa)
        ib, sb = _image(pb)
        A.append(ia)
        B.append(ib)
        sizes.append((sa, sb))
    A = torch.stack(A).to(dev)
    B = torch.stack(B).to(dev)

    def resize(x, h, w):
        return F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)

    from torch.nn.attention import SDPBackend, sdpa_kernel

    # ROCm（RX 7900 XTX）では、flash・mem_efficient の注意が DINOv3 の形で hipErrorInvalidValue に
    # なる。math だけにする。with の中だけにして、同じプロセスの VGGT は速いほうのままにする
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        a_lr, b_lr = resize(A, model.H_lr, model.W_lr), resize(B, model.H_lr, model.W_lr)
        a_hr = b_hr = None
        if model.H_hr is not None:
            a_hr, b_hr = resize(A, model.H_hr, model.W_hr), resize(B, model.H_hr, model.W_hr)
        p = model(a_lr, b_lr, img_A_hr=a_hr, img_B_hr=b_hr)
        out = []
        for k, ((ha, wa), (hb, wb)) in enumerate(sizes):
            ov_ab, pr_ab = _map_confidence(confidence=p["confidence_AB"][k : k + 1],
                                           threshold=model.threshold)  # fmt: skip
            pred = {"warp_AB": p["warp_AB"][k : k + 1], "overlap_AB": ov_ab, "precision_AB": pr_ab}
            if model.bidirectional:
                ov_ba, pr_ba = _map_confidence(confidence=p["confidence_BA"][k : k + 1],
                                               threshold=model.threshold)  # fmt: skip
                pred.update(warp_BA=p["warp_BA"][k : k + 1], overlap_BA=ov_ba, precision_BA=pr_ba)
            m, cert, _, _ = model.sample(pred, samples)
            ka, kb = model.to_pixel_coordinates(m, ha, wa, hb, wb)
            out.append({
                "ka": ka.float().cpu().numpy(), "kb": kb.float().cpu().numpy(),
                "cert": cert.float().cpu().numpy(),
                # 重なりの見込み（a の画素のうち、b に写っていそうな割合）
                "overlap": float(ov_ab.mean().item()),
            })  # fmt: skip
    return out


# --- ステーションごとの辺（地図のサーバーの SNOO_ROMA_EDGES=1） ---

# RoMa の重なりの確からしさがこれ未満の対応は使わない
MIN_CERT = 0.05
# 1 組のステーションに通すコマの組
FRAME_PAIRS = 4
# 戻ってきた組の候補：今の位置でこれより近く、番号が LOOP_MIN_GAP 以上離れたもの（1 か所に 1 組）
NEAR_M = 0.6
LOOP_MIN_GAP = 4


def estimate(matches: list[dict], pans: list, loop: bool):
    """コマの組ごとの対応（match_pairs の答え）と、そのパン・チルト (pan_a, tilt_a, pan_b, tilt_b)
    から、ステーションの組の答え（無ければ None）と、コマの組ごとの答え。
    戻ってきた組（loop）は取り違えが多いので、向きのそろう組を 1 つ多く求める。
    """
    est = []
    for k, (m, (pa, ta, pb, tb)) in enumerate(zip(matches, pans, strict=True)):
        ok = np.asarray(m["cert"]) >= MIN_CERT
        ka, kb = np.asarray(m["ka"])[ok], np.asarray(m["kb"])[ok]
        est.append(pair_pose(ka, kb, pa, ta, pb, tb, seed=k))
    return combine(est, min_pairs=MIN_AGREEING_PAIRS + int(loop)), est


def relative_yaw(a: Pose, b: Pose) -> float:
    return math.atan2(math.sin(b.yaw - a.yaw), math.cos(b.yaw - a.yaw))


def revisit_candidate(b: int, prior: dict[int, Pose]) -> int | None:
    """b に近い、番号の離れたステーション（今の位置で NEAR_M より近い、いちばん近いもの）。"""
    if b not in prior:
        return None
    pb = prior[b]
    near = [(math.hypot(p.x - pb.x, p.y - pb.y), a) for a, p in prior.items()
            if a <= b - LOOP_MIN_GAP]  # fmt: skip
    if not near:
        return None
    d, a = min(near)
    return a if d < NEAR_M else None


def station_edges(
    stills: dict[int, list[dict]],
    dirs: dict,
    b: int,
    prior: dict[int, Pose],
    match,
    dyaw_prev_deg: float | None = None,
    revisit: bool = True,
) -> tuple[list, dict]:
    """新しいステーション b の RoMa の辺：前のステーション（b - 1）と、戻ってきた組の候補 1 つまで。

    stills はステーションごとの止まって撮ったコマ（meta.json の frames。tilt_deg は校正済み）、
    dirs はステーションの置き場（Path）。prior は今の位置（b は前の位置に走った辺を足したもの）。
    dyaw_prev_deg は前から b への向きの見込み（走った辺。無ければ prior から）。
    match は match_pairs と同じ形の関数（コマの組の列 → 対応の列）。答えは (辺, 記録)。
    """
    jobs = []
    if b - 1 in stills:
        dyaw = dyaw_prev_deg
        if dyaw is None and b - 1 in prior and b in prior:
            dyaw = math.degrees(relative_yaw(prior[b - 1], prior[b]))
        jobs.append((b - 1, dyaw or 0.0, False))
    a = revisit_candidate(b, prior) if revisit else None
    if a is not None and a in stills and a != b - 1:
        jobs.append((a, math.degrees(relative_yaw(prior[a], prior[b])), True))
    edges, info = [], []
    for a, dyaw, loop in jobs:
        fp = pick_frame_pairs(stills[a], stills[b], dyaw, FRAME_PAIRS)
        pairs = [(str(dirs[a] / fa["file"]), str(dirs[b] / fb["file"])) for fa, fb in fp]
        pans = [(fa["pan_deg"], fa["tilt_deg"], fb["pan_deg"], fb["tilt_deg"]) for fa, fb in fp]
        st, est = estimate(match(pairs), pans, loop)
        info.append({"a": a, "loop": loop, "pairs": len(fp), "edge": st is not None,
                     "ok_pairs": sum(e is not None for e in est)})  # fmt: skip
        if st is not None:
            edges.append(to_edge(a, b, st, f"roma:{a:03d}_{b:03d}"))
    return edges, info
