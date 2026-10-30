"""snoo の写真・動画の関節（DeepLabCut）に、シミュレーターの猫の関節の角度を合わせる。

    MUJOCO_GL=egl .venv/bin/python -m scripts.snoo_posture_fit --out out/snoo_postures
姿勢ごとに 1 枚（SOURCES）を選び、app/cat_pose.py で 39 点を取る。GPU は使わない
（地図や学習が GPU を使っていることがあり、GPU を開くのは 1 つのプロセスに限るため）。
次に、体（app/snoo_body.py）の点（kp_ の site）を、写したカメラに近いカメラで画面に落とし、
画面の上のずれが小さくなる関節の角度とカメラを探す（減衰付きのガウス・ニュートン）。
カメラ 1 台では奥行きが決まらないので、手で決めた初めの形に近いこと・左右の脚が
そろうこと・床に着く所（足先や腰）が床にあること・床より下に行かないことを一緒に効かせる。
合わなかった姿勢は手で決めた形（HAND）を使い、"source": "hand" と書く。
結果は app/snoo_postures.json と、写真と合わせた形を並べた絵（--out）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from app import snoo_body as sb  # noqa: E402
from app.cat_pose import NAMES  # noqa: E402

VIDEO = Path("data/snoo_video/IMG_5903.mov")
PHOTOS = Path("data/snoo_raw")
# iPhone の広角（26 mm 相当）の長い辺の画角（度）。写真も縦の動画も長い辺が縦
FOV_LONG_DEG = 66.0
# 写真は長い辺をこの画素に縮めてから関節を取る
PHOTO_PX = 1280
# 使う点の確からしさの下限
CONF_MIN = 0.4
# 点ごとの重み。膝と首の付け根は、DeepLabCut の点とこの体の関節の位置が少し違うので軽く
WEIGHT = {"neck_base": 0.4, "back_base": 0.6, "throat_base": 0.5, "belly_bottom": 0.5} | {
    f"{a}_{b}_knee": 0.5 for a in ("front", "back") for b in ("left", "right")
}
LEG_PAIRS = [("fl", "fr", ("shoulder", "elbow")), ("bl", "br", ("hip", "knee"))]


@dataclass
class Source:
    image: str  # data/snoo_raw の番号か、動画のフレーム（v123）
    init: dict[str, float]  # 手で決めた初めの形（度、snoo_z は m）
    contacts: tuple[str, ...]  # 床に着く体（body の名前）
    symmetric: float = 25.0  # 左右の脚の角度の差の目安（度）
    note: str = ""
    azimuths: tuple[float, ...] = tuple(range(0, 360, 45))
    tail_free: bool = False  # しっぽは写真で外に出ているので、点が無ければ初めの形のまま


PAWS = tuple(f"snoo_{leg}_lower" for leg in ("fl", "fr", "bl", "br"))
SOURCES = {
    "walk": Source(
        "262",
        {
            "snoo_neck_pitch": 15, "snoo_head_pitch": 10,
            "snoo_tail1_pitch": 70, "snoo_tail2_pitch": 20, "snoo_tail3_pitch": -20,
            "snoo_tail4_pitch": -30,
            "snoo_fl_shoulder": -10, "snoo_fr_shoulder": 10, "snoo_bl_hip": 10, "snoo_br_hip": -10,
        },
        PAWS, symmetric=60, note="床の上を、しっぽを立てて斜め前へ歩く（写真 262）",
    ),
    "alert": Source(
        "80",
        {
            "snoo_neck_pitch": -15, "snoo_neck_yaw": 30, "snoo_head_yaw": 20,
            "snoo_tail1_pitch": 75, "snoo_tail2_pitch": 10, "snoo_tail3_pitch": -10,
        },
        PAWS, note="立ち止まって、首をこちらに向けて見る（写真 80）",
    ),
    "sit": Source(
        "150",
        {
            "snoo_z": -0.1, "snoo_pitch": -50, "snoo_spine1_pitch": -5, "snoo_spine2_pitch": -5,
            "snoo_neck_pitch": 20, "snoo_head_pitch": 35,
            "snoo_fl_shoulder": 60, "snoo_fr_shoulder": 60,
            "snoo_bl_hip": 20, "snoo_br_hip": 20, "snoo_bl_knee": -100, "snoo_br_knee": -100,
            "snoo_tail1_pitch": 50, "snoo_tail1_yaw": 30, "snoo_tail2_yaw": 40,
            "snoo_tail3_yaw": 30,
        },
        ("snoo_hips", "snoo_fl_lower", "snoo_fr_lower", "snoo_bl_lower", "snoo_br_lower"),
        note="前足をそろえて座る（写真 150）", tail_free=True,
    ),
    "lie": Source(
        "163",
        {
            "snoo_z": -0.1, "snoo_neck_pitch": -25, "snoo_head_pitch": 15,
            "snoo_fl_shoulder": -80, "snoo_fr_shoulder": -80, "snoo_fl_elbow": 10,
            "snoo_fr_elbow": 10, "snoo_bl_hip": -80, "snoo_br_hip": -80,
            "snoo_bl_knee": -60, "snoo_br_knee": -60,
            "snoo_tail1_pitch": -15, "snoo_tail2_yaw": 20, "snoo_tail3_yaw": 20,
        },
        ("snoo_hips", "snoo_waist", "snoo_chest", "snoo_fl_lower", "snoo_fr_lower"),
        note="床に伏せて、前足を前に出し頭を上げる（写真 163）",
    ),
    "sleep": Source(
        "3",
        {
            # 右を下に横に倒れ（roll）、背骨を前後（pitch）に曲げて丸まる。横になると
            # pitch の軸が縦になるので、床の上で輪になる
            "snoo_z": -0.092, "snoo_roll": 80, "snoo_pitch": 0,
            "snoo_spine1_pitch": 35, "snoo_spine2_pitch": 35,
            "snoo_neck_pitch": 50, "snoo_head_pitch": 40, "snoo_head_roll": -20,
            "snoo_fl_shoulder": -50, "snoo_fr_shoulder": -40, "snoo_fl_elbow": 100,
            "snoo_fr_elbow": 90, "snoo_bl_hip": -70, "snoo_br_hip": -60,
            "snoo_bl_knee": -80, "snoo_br_knee": -70,
            "snoo_tail1_pitch": -50, "snoo_tail2_pitch": -60, "snoo_tail3_pitch": -60,
            "snoo_tail4_pitch": -50,
        },
        ("snoo_hips", "snoo_waist", "snoo_chest"),
        note="横に倒れて丸まり、顔をしっぽの方へ埋めて寝る（写真 3）",
    ),
}  # fmt: skip

# 合わせた結果を見て、手で決めた形を使う姿勢（理由を書く）
HAND: dict[str, str] = {
    "sleep": "顔が体に埋もれて点がぶれ（確からしさ 0.6 前後）、合わせると頭が上がった形になった",
}


# --- 関節を取る（CPU） ---


def load_image(key: str) -> np.ndarray:
    if key.startswith("v"):
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(key[1:]))
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"{VIDEO} の {key} が読めない")
        return img
    img = cv2.imread(str(PHOTOS / f"img_{int(key):04d}.jpg"))
    if img is None:
        raise SystemExit(f"{PHOTOS} の {key} が無い")
    s = PHOTO_PX / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def cpu_pose_models():
    """猫の検出（RT-DETR）と関節（DeepLabCut）を CPU で読む。GPU のプロセスを増やさない。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    from app.cat_pose import CatPose
    from app.snoo import CatDetector

    det = CatDetector(cat_min=0.3)
    if det.device != "cpu":
        raise SystemExit("GPU を使わないはずが、検出が GPU に載った")
    return det, CatPose(device="cpu")


def keypoints(keys: list[str], cache: Path) -> dict[str, dict]:
    """画像ごとの {box, size, keypoints (39, 3)}。cache にあれば使い回す。"""
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [k for k in keys if k not in have]
    if todo:
        det, pose = cpu_pose_models()
        for k in todo:
            img = load_image(k)
            dets = det.detect(img)
            if not dets:
                raise SystemExit(f"{k} に猫が見つからない")
            kp = pose.estimate(img, dets[0].box)
            have[k] = {
                "box": [float(v) for v in dets[0].box],
                "size": [img.shape[1], img.shape[0]],
                "keypoints": kp.round(3).tolist(),
            }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have))
    return have


# --- 体を画面に落とす ---


class Body:
    """猫だけの MuJoCo のモデル。関節の角度から、点と geom の位置を計算する。"""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(sb.standalone_mjcf())
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in sb.JOINT_NAMES])
        self.data.qpos[3] = 1.0  # 根元の向き（四元数）
        self.site = {n: self.model.site(f"kp_{n}").id for n in NAMES if self._has(f"kp_{n}")}
        cat = self.model.geom_group == 1
        self.geoms = np.flatnonzero(cat)
        self.geom_body = {
            g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g])
            for g in self.geoms
        }

    def _has(self, name: str) -> bool:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0

    def pose(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)

    def sites(self) -> dict[str, np.ndarray]:
        return {n: self.data.site_xpos[i] for n, i in self.site.items()}

    def bottoms(self) -> dict[str, float]:
        """body ごとの、いちばん低い geom の下の端の高さ。"""
        m, d = self.model, self.data
        out: dict[str, float] = {}
        for g in self.geoms:
            R = d.geom_xmat[g].reshape(3, 3)
            s = m.geom_size[g]
            c = d.geom_xpos[g]
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                z = c[2] - math.sqrt(float(((R[2, :] * s) ** 2).sum()))
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                z = c[2] - abs(R[2, 2]) * s[1] - s[0]
            else:
                z = c[2] - s[0]
            b = self.geom_body[g]
            out[b] = min(out.get(b, math.inf), z)
        return out


def camera_frame(cam: np.ndarray):
    """カメラ（注視点 x, y, z・距離・方位角・仰角（度））の位置と、右・上・前の向き。

    MuJoCo の自由なカメラ（MjvCamera）と同じ決め方にして、そのまま絵にも使えるようにする。
    """
    lx, ly, lz, dist, az, el = cam
    a, e = math.radians(az), math.radians(el)
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    pos = np.array([lx, ly, lz]) - dist * fwd
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, right, up, fwd


def project(pts: np.ndarray, cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    f = (max(w, h) / 2) / math.tan(math.radians(FOV_LONG_DEG) / 2)
    pos, right, up, fwd = camera_frame(cam)
    d = pts - pos
    z = np.maximum(d @ fwd, 1e-3)
    return np.stack([w / 2 + f * (d @ right) / z, h / 2 - f * (d @ up) / z], -1)


# --- 合わせる ---


@dataclass
class Fit:
    q: np.ndarray
    cam: np.ndarray
    rms: float  # 点のずれの二乗平均（猫の箱の対角線に対する比）
    used: list[str] = field(default_factory=list)


def fit_posture(body: Body, src: Source, kp: dict, iters: int = 120) -> Fit:
    w, h = kp["size"]
    x0, y0, x1, y1 = kp["box"]
    diag = math.hypot((x1 - x0) * w, (y1 - y0) * h)
    pts = np.array(kp["keypoints"])
    used = [n for n, (_, _, c) in zip(NAMES, pts, strict=True) if c >= CONF_MIN and n in body.site]
    target = np.array([pts[NAMES.index(n), :2] for n in used])
    wt = np.array([pts[NAMES.index(n), 2] * WEIGHT.get(n, 1.0) for n in used])
    if src.tail_free:
        wt *= np.array([0.3 if n == "tail_end" else 1.0 for n in used])
    q_init = sb.to_vector(src.init)
    lo, hi = sb.limits()
    sigma = np.full(len(q_init), math.radians(35))
    sigma[sb.JOINT_INDEX["snoo_z"]] = 0.08
    pair = [
        (sb.JOINT_INDEX[f"snoo_{a}_{j}"], sb.JOINT_INDEX[f"snoo_{b}_{j}"])
        for a, b, js in LEG_PAIRS
        for j in js
    ]
    sym = math.radians(src.symmetric)

    def residual(p: np.ndarray) -> np.ndarray:
        q, cam = p[: len(q_init)], p[len(q_init) :]
        body.pose(q)
        s = body.sites()
        uv = project(np.array([s[n] for n in used]), cam, (w, h))
        r = [((uv - target) / diag * np.sqrt(wt)[:, None]).ravel()]
        r.append((q - q_init) / sigma * 0.15)
        r.append(np.array([(q[i] - q[j]) / sym * 0.1 for i, j in pair]))
        bot = body.bottoms()
        # 床に着く所は床に、ほかは床より下に行かない（2 cm で 1 の重さ）
        r.append(np.array([bot[c] / 0.02 for c in src.contacts]) * 0.3)
        r.append(np.array([min(z, 0.0) / 0.01 for z in bot.values()]) * 0.3)
        return np.concatenate(r)

    best: Fit | None = None
    best_cost = math.inf
    for az in src.azimuths:
        dist0 = (max(w, h) / 2) / math.tan(math.radians(FOV_LONG_DEG) / 2) * 0.45 / diag * 1.2
        cam0 = np.array([0.0, 0.0, 0.12, dist0, az, -10.0])
        p = np.concatenate([q_init, cam0])
        plo = np.concatenate([lo, [-0.4, -0.4, -0.05, 0.25, az - 120, -80]])
        phi = np.concatenate([hi, [0.4, 0.4, 0.4, 5.0, az + 120, 15]])
        p = _levenberg_marquardt(residual, p, plo, phi, iters)
        q, cam = p[: len(q_init)], p[len(q_init) :]
        body.pose(q)
        s = body.sites()
        uv = project(np.array([s[n] for n in used]), cam, (w, h))
        err = np.linalg.norm(uv - target, axis=1) / diag
        rms = float(np.sqrt((err**2 * wt).sum() / wt.sum()))
        cost = float((residual(p) ** 2).sum())
        if cost < best_cost:
            best, best_cost = Fit(q.copy(), cam.copy(), rms, used), cost
    assert best is not None
    return best


def _levenberg_marquardt(f, p, lo, hi, iters: int) -> np.ndarray:
    """境界付きの減衰ガウス・ニュートン（scipy を入れないで済むよう、数値微分で）。"""
    p = np.clip(p, lo, hi)
    r = f(p)
    cost = float(r @ r)
    lam = 1e-2
    eps = 1e-4
    for _ in range(iters):
        J = np.empty((len(r), len(p)))
        for i in range(len(p)):
            dp = p.copy()
            step = eps * max(1.0, abs(p[i])) if i >= len(p) - 6 else eps
            dp[i] += step
            J[:, i] = (f(dp) - r) / step
        g = J.T @ r
        H = J.T @ J
        improved = False
        for _ in range(8):
            delta = np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-6 * np.eye(len(p))), -g)
            pn = np.clip(p + delta, lo, hi)
            rn = f(pn)
            cn = float(rn @ rn)
            if cn < cost:
                p, r, cost, lam = pn, rn, cn, max(lam / 3, 1e-6)
                improved = True
                break
            lam *= 4
        if not improved or float(np.abs(delta).max()) < 1e-6:
            break
    return p


# --- 絵 ---


def draw_keypoints(img: np.ndarray, kp: dict, fit: Fit | None, body: Body) -> np.ndarray:
    out = img.copy()
    pts = np.array(kp["keypoints"])
    for x, y, c in pts:
        if c >= CONF_MIN:
            cv2.circle(out, (int(x), int(y)), 5, (60, 230, 60), -1, cv2.LINE_AA)
    if fit is not None:
        body.pose(fit.q)
        s = body.sites()
        uv = project(np.array([s[n] for n in fit.used]), fit.cam, tuple(kp["size"]))
        for n, (u, v) in zip(fit.used, uv, strict=True):
            x, y = pts[NAMES.index(n), :2]
            cv2.line(out, (int(x), int(y)), (int(u), int(v)), (40, 40, 240), 1, cv2.LINE_AA)
            cv2.circle(out, (int(u), int(v)), 4, (40, 40, 240), -1, cv2.LINE_AA)
    return out


def render_view(body: Body, q: np.ndarray, cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """合わせたカメラから見た体。size は (幅, 高さ)。カメラは model の fit に書いて使う。"""
    w, h = size
    body.pose(q)
    m, k = body.model, body.model.camera("fit").id
    pos, right, up, fwd = camera_frame(cam)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([right, up, -fwd], 1).ravel())
    m.cam_pos[k], m.cam_quat[k] = pos, quat
    f = (max(w, h) / 2) / math.tan(math.radians(FOV_LONG_DEG) / 2)
    m.cam_fovy[k] = math.degrees(2 * math.atan(h / 2 / f))
    mujoco.mj_kinematics(m, body.data)
    mujoco.mj_camlight(m, body.data)
    s = 720 / h  # 描く大きさ（縦 720）
    with mujoco.Renderer(m, 720, round(w * s)) as r:
        r.update_scene(body.data, "fit")
        return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 21), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit_height(img: np.ndarray, h: int) -> np.ndarray:
    return cv2.resize(
        img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("out/snoo_postures"))
    parser.add_argument("--only", nargs="+", choices=list(SOURCES))
    parser.add_argument("--no-write", action="store_true", help="app/snoo_postures.json は書かない")
    args = parser.parse_args()

    names = args.only or list(SOURCES)
    kps = keypoints([SOURCES[n].image for n in names], args.out / "keypoints.json")
    body = Body()
    library = json.loads(sb.POSTURES_PATH.read_text()) if sb.POSTURES_PATH.exists() else {}
    postures = library.get("postures", {})
    for name in names:
        src = SOURCES[name]
        kp = kps[src.image]
        fit = fit_posture(body, src, kp)
        hand = name in HAND
        q = sb.to_vector(src.init) if hand else fit.q.copy()
        if name == "walk":
            # 歩く振りは SnooAnimator が足すので、脚は左右をそろえた真ん中の形にしておく
            for a, b, js in LEG_PAIRS:
                for j in js:
                    i, k = sb.JOINT_INDEX[f"snoo_{a}_{j}"], sb.JOINT_INDEX[f"snoo_{b}_{j}"]
                    q[i] = q[k] = (q[i] + q[k]) / 2
        postures[name] = {
            "source": "hand" if hand else "fit",
            "image": src.image
            if src.image.startswith("v")
            else f"snoo_raw/img_{int(src.image):04d}",
            "note": src.note + (f"。手で決めた：{HAND[name]}" if hand else ""),
            "rms": round(fit.rms, 3),
            "camera": [round(float(v), 3) for v in fit.cam],
            "joints": sb.to_angles(q),
        }
        print(f"{name}: {src.image} rms {fit.rms:.3f} 点 {len(fit.used)} "
              f"cam {np.round(fit.cam, 2).tolist()} {'hand' if hand else 'fit'}")  # fmt: skip
        img = load_image(src.image)
        size = tuple(kp["size"])
        panels = [
            label(
                draw_keypoints(img, kp, fit, body), f"{name}: photo {src.image}, DLC green, fit red"
            ),
            label(render_view(body, fit.q, fit.cam, size), f"fit rms {fit.rms:.3f}"),
        ]
        if hand:
            panels.append(label(render_view(body, q, fit.cam, size), "hand-tuned (used)"))
        args.out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(args.out / f"fit_{name}.jpg"), np.hstack([fit_height(p, 720) for p in panels])
        )
    if not args.no_write:
        library = {
            "about": "snoo の姿勢の本。joints は度（snoo_z は m）。"
            "scripts/snoo_posture_fit.py で作る",
            "postures": {n: postures[n] for n in sb.POSTURES if n in postures},
        }
        sb.POSTURES_PATH.write_text(json.dumps(library, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
