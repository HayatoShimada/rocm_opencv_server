"""シミュレーターの snoo が、本物の snoo にどれだけ似ているかを確かめる。

    MUJOCO_GL=egl .venv/bin/python -m scripts.snoo_realism_check --out out/realism --tag after \\
        --detect --compare --video
- --detect：車のカメラ（床から 8 cm）の絵を、姿勢 × 距離 3 つ × 向き 3 つで撮り、猫の検出
  （RT-DETR、CPU）の cat の確からしさと、DeepLabCut の点の確からしさを JSON に書く。
  検出の下限は 0.01 まで下げて、見つけられなかった絵の値も残す（0 か 1 かでなく近さを見る）。
- --compare：ロボットの録画（data/snoo_rec）で snoo が写ったコマと、同じ距離・向き・首の
  角度に置いたシミュレーターの絵を横に並べる。箱の幅・高さ（画素）も並べて書く。
- --video：歩く・座る・伏せる・寝る・起きるを、車のカメラと斜め上から撮った MP4。
GPU は使わない（ほかの仕事が GPU を使っていることがあるため。CUDA も HIP も見えなくする）。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from app import snoo_body as sb  # noqa: E402
from app.cat_pose import NAMES  # noqa: E402
from app.snoo_sim import CAM_H_M as CAM_H  # noqa: E402
from app.snoo_sim import Sim, SnooPose, synthetic_room  # noqa: E402

# 長毛でも取りやすい点（scripts/snoo_posture_views.py と同じ）
EASY = (
    "nose", "left_eye", "right_eye", "left_earend", "right_earend", "tail_end",
    "front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw",
)  # fmt: skip
TRACKS = Path("data/snoo_behavior/tracks")
REC = Path("data/snoo_rec")
# 録画のコマと、シミュレーターで置く姿勢・向き（0 は右を向いた横向き、90 はこちら向き）。
# 距離・方位・首の角度は録画の記録（dist_ground_m・bearing_deg・camera）から取る
REAL = (
    ("sit", "20261029_220238", "0000055845.jpg", "sit", 95),
    ("walk", "20261029_220420", "0000061303.jpg", "walk", 215),
    ("lie", "20261029_220623", "0000068563.jpg", "lie", 50),
    ("sleep", "20261029_220522", "0000065178.jpg", "sleep", 40),
    ("sit_far", "20261029_220440", "0000062569.jpg", "sit", 80),
)


@dataclasses.dataclass
class ShowPose(SnooPose):
    posture: str = "walk"


class StillSnoo:
    def __init__(self, pose):
        self.pose = pose

    def step(self, dt, robot):
        return self.pose


def postures() -> tuple[str, ...]:
    return tuple(sb.POSTURES)


class Stage:
    """作り物の部屋に車と snoo を置いて、車のカメラと斜め上から撮る。"""

    def __init__(self):
        self.sim = Sim(synthetic_room(), rays=(64, 48), seed=0)
        self.sim.car.pan_deg = self.sim.car.tilt_deg = 0.0
        m = self.sim.model
        self.rcam = mujoco.Renderer(m, 480, 640, max_geom=2000)
        self.r34 = mujoco.Renderer(m, 480, 640, max_geom=2000)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[2] = 0  # 車は描かない

    def place(self, posture, dist, facing_deg, bearing_deg=0.0, pan=0.0, tilt=0.0, settle=5.0):
        """カメラの dist m 先、方位 bearing_deg（車から見て左が +。録画の記録と同じ）に置く。"""
        car = self.sim.car
        car.pan_deg, car.tilt_deg = pan, tilt
        cam = car.camera_pose()
        a = cam.yaw + math.radians(bearing_deg)
        x, y = cam.x + dist * math.cos(a), cam.y + dist * math.sin(a)
        # facing 0 は、カメラから見て右を向く（猫の前がカメラの右）。90 はこちら向き
        yaw = a - math.pi / 2 - math.radians(facing_deg)
        self.sim.snoo = StillSnoo(ShowPose(x, y, yaw, False, posture))
        self.sim.snoo_anim.update(settle, self.sim.snoo.pose)
        self.sim.sync()

    def ground_dist(self, tilt: float) -> float | None:
        """検出の箱の下の端から、録画と同じやり方（scripts/snoo_behavior_extract.py）で距離を出す。"""
        if self.sim.detection is None:
            return None
        v1 = self.sim.detection[3]
        f = 320 / math.tan(math.radians(27))
        below = math.degrees(math.atan((v1 - 240) / f)) - tilt
        return CAM_H / math.tan(math.radians(below)) if below > 1.0 else None

    def place_like_real(self, posture, dist, facing, bearing, pan, tilt) -> None:
        """録画の距離は猫の下の端（手前の足や毛）までなので、同じ測り方で同じ距離になるよう置く。"""
        d = dist
        for _ in range(6):
            self.place(posture, d, facing, bearing, pan, tilt)
            g = self.ground_dist(tilt)
            if g is None:
                break
            d += dist - g
            if abs(dist - g) < 0.005:
                break
        self.place(posture, d, facing, bearing, pan, tilt)

    def car_cam(self) -> np.ndarray:
        self.rcam.update_scene(self.sim.data, "car_cam", self.opt)
        return cv2.cvtColor(self.rcam.render(), cv2.COLOR_RGB2BGR)

    def view34(self, az=135.0, dist=0.95, el=-22.0) -> np.ndarray:
        s = self.sim.snoo.pose
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [s.x, s.y, 0.12]
        cam.distance, cam.elevation = dist, el
        cam.azimuth = math.degrees(s.yaw) + az
        self.r34.update_scene(self.sim.data, cam, self.opt)
        return cv2.cvtColor(self.r34.render(), cv2.COLOR_RGB2BGR)


def models():
    from app.cat_pose import CatPose
    from app.snoo import CatDetector

    det = CatDetector(cat_min=0.01)
    if det.device != "cpu":
        raise SystemExit("GPU を使わないはずが、検出が GPU に載った")
    return det, CatPose(device="cpu")


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 24), 0, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def detect(stage: Stage, out: Path, tag: str) -> dict:
    det, pose = models()
    rows = []
    easy = [NAMES.index(n) for n in EASY]
    for name in postures():
        for dist in (0.5, 0.8, 1.2):
            for facing in (0, 45, 90):
                stage.place(name, dist, facing)
                box = stage.sim.detection
                if box is None:
                    continue
                img = stage.car_cam()
                dets = det.detect(img)
                u0, v0, u1, v1, _ = box
                kp = pose.estimate(img, (u0 / 640, v0 / 480, u1 / 640, v1 / 480))
                best = max(dets, key=lambda d: d.cat, default=None)
                rows.append(
                    {
                        "posture": name, "dist": dist, "facing": facing,
                        "cat": round(best.cat if best else 0.0, 3),
                        "conf_all": round(float(kp[:, 2].mean()), 3),
                        "conf_easy": round(float(kp[easy, 2].mean()), 3),
                        "sim_box_px": [round(u1 - u0), round(v1 - v0)],
                    }
                )  # fmt: skip
                if dist == 0.8:
                    vis = img.copy()
                    if best is not None:
                        b = best.box
                        p0 = (int(b[0] * 640), int(b[1] * 480))
                        cv2.rectangle(vis, p0, (int(b[2] * 640), int(b[3] * 480)), (0, 200, 0), 2)
                    for x, y, c in kp:
                        if c >= 0.4:
                            cv2.circle(vis, (int(x), int(y)), 3, (60, 230, 60), -1)
                    text = f"{tag} {name} f{facing} cat={rows[-1]['cat']:.2f}"
                    cv2.imwrite(str(out / f"det_{tag}_{name}_{facing}.jpg"), label(vis, text))
    keys = ("cat", "conf_all", "conf_easy")

    def mean(rs):
        return {k: round(float(np.mean([r[k] for r in rs])), 3) for k in keys} | {
            "cat_ge_0.3": round(float(np.mean([r["cat"] >= 0.3 for r in rs])), 3)
        }

    result = {
        "mean": mean(rows),
        "by_posture": {n: mean([r for r in rows if r["posture"] == n]) for n in postures()},
        "runs": rows,
    }
    (out / f"detect_{tag}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"mean": result["mean"], "by_posture": result["by_posture"]}, indent=1))
    return result


def track_row(session: str, file: str) -> dict:
    for line in (TRACKS / f"{session}_192.168.0.188.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["file"] == file:
            return r
    raise SystemExit(f"{session} に {file} が無い")


def real_image(session: str, file: str) -> np.ndarray:
    day, rest = session.split("_", 1)
    path = glob.glob(str(REC / day / f"{rest}_*" / file))
    img = cv2.imread(path[0]) if path else None
    if img is None:
        raise SystemExit(f"{session}/{file} の絵が無い")
    return img


def compare(stage: Stage, out: Path, tag: str) -> list[dict]:
    det = None
    try:
        det, _ = models()
    except Exception as e:  # noqa: BLE001  検出が読めなくても絵は作る
        print("検出を読めない:", e)
    rows, panels = [], []
    for key, session, file, posture, facing in REAL:
        r = track_row(session, file)
        cat, cam = r["cat"], r["camera"] or {}
        dist = cat.get("dist_ground_m") or cat.get("dist_m")
        pan, tilt = cam.get("pan_deg", 0.0), cam.get("tilt_deg", 0.0)
        stage.place_like_real(posture, dist, facing, cat.get("bearing_deg", 0.0), pan, tilt)
        sim_img = stage.car_cam()
        real = real_image(session, file)
        b = cat["box"]
        rb = (int(b[0] * 640), int(b[1] * 480), int(b[2] * 640), int(b[3] * 480))
        cv2.rectangle(real, rb[:2], rb[2:], (0, 220, 0), 2)
        row = {
            "key": key, "dist_m": dist, "tilt": tilt, "real_cat": cat["cat"],
            "real_box_px": [rb[2] - rb[0], rb[3] - rb[1]],
        }  # fmt: skip
        if det is not None:
            dets = det.detect(sim_img)
            best = max(dets, key=lambda d: d.cat, default=None)
            row["sim_cat"] = round(best.cat, 3) if best else 0.0
            if best is not None:
                sb_ = best.box
                q = (int(sb_[0] * 640), int(sb_[1] * 480), int(sb_[2] * 640), int(sb_[3] * 480))
                row["sim_det_box_px"] = [q[2] - q[0], q[3] - q[1]]
                cv2.rectangle(sim_img, q[:2], q[2:], (0, 220, 0), 2)
        if stage.sim.detection is not None:
            u0, v0, u1, v1, _ = stage.sim.detection
            row["sim_geom_box_px"] = [round(u1 - u0), round(v1 - v0)]
            cv2.rectangle(sim_img, (int(u0), int(v0)), (int(u1), int(v1)), (0, 160, 255), 1)
        rows.append(row)
        left = label(
            real, f"real {key} d={dist:.2f}m cat={cat['cat']:.2f} box={row['real_box_px']}"
        )
        right = label(
            sim_img,
            f"sim {posture} cat={row.get('sim_cat', -1):.2f} box={row.get('sim_det_box_px')}",
        )
        q34 = label(stage.view34(), "sim 3/4")
        pair = np.hstack([left, right, q34])
        cv2.imwrite(str(out / f"realsim_{tag}_{key}.jpg"), pair)
        panels.append(pair)
    cv2.imwrite(str(out / f"realsim_{tag}.jpg"), np.vstack(panels))
    (out / f"compare_{tag}.json").write_text(json.dumps(rows, indent=1))
    for row in rows:
        print(row)
    return rows


def video(stage: Stage, out: Path, tag: str, fps: int = 20) -> Path:
    """横切って歩き、止まって嗅ぎ、座り、毛づくろいし、伏せ、寝て、起きて歩き去る。

    左が車のカメラ（首は猫を追う）、右が斜め上から。猫の向きは段ごとの向きへゆっくり回す。
    """
    from app.snoo_sim_render import VideoWriter

    sim = stage.sim
    path = out / f"snoo_{tag}.mp4"
    writer = VideoWriter(path, fps=fps, size=(1280, 480))
    car = sim.car
    car.pan_deg = car.tilt_deg = 0.0
    cam = car.camera_pose()
    fwd = np.array([math.cos(cam.yaw), math.sin(cam.yaw)])
    left = np.array([-fwd[1], fwd[0]])
    base = np.array([cam.x, cam.y])
    across = math.atan2(-left[1], -left[0])  # カメラから見て右へ
    toward = math.atan2(-fwd[1], -fwd[0])  # カメラの方へ
    away = math.atan2(fwd[1], fwd[0])
    # (秒, 姿勢, 動くか, 向き)
    script = [
        (2.6, "walk", True, across),
        (2.0, "sniff", False, across),
        (2.5, "sit", False, toward + math.radians(40)),
        (3.0, "groom", False, toward + math.radians(40)),
        (3.0, "lie", False, toward + math.radians(60)),
        (5.0, "sleep", False, toward + math.radians(60)),
        (3.0, "lie", False, toward + math.radians(60)),
        (2.5, "alert", False, toward + math.radians(20)),
        (3.5, "walk", True, away + math.radians(30)),
    ]
    p = base + 0.95 * fwd + 0.42 * left
    heading = across
    pose = ShowPose(float(p[0]), float(p[1]), heading, False, "walk")
    sim.snoo = StillSnoo(pose)
    sim.snoo_anim.update(5.0, pose)
    speed, dt = 0.25, 1.0 / fps
    t_all = 0.0
    for seconds, posture, moving, want in script:
        n = int(seconds * fps)
        for k in range(n):
            turn = (want - heading + math.pi) % (2 * math.pi) - math.pi
            heading += float(np.clip(turn, -1.2 * dt, 1.2 * dt))
            if moving:
                p = p + speed * dt * np.array([math.cos(heading), math.sin(heading)])
            pose = ShowPose(float(p[0]), float(p[1]), heading, moving, posture)
            sim.snoo = StillSnoo(pose)
            # 首は猫を追う（実機もそうする）
            rel = p - base
            bearing = math.degrees(math.atan2(rel @ left, rel @ fwd))
            car.pan_deg = float(np.clip(-bearing, -60, 60))
            sim.advance(dt)
            sim.sync()
            a = label(stage.car_cam(), f"robot camera 8 cm  {posture}  t={t_all:.1f}s")
            b = label(stage.view34(az=150, dist=1.0, el=-18), "3/4 view")
            writer.write(np.hstack([a, b]))
            t_all += dt
            if k == n // 2:
                cv2.imwrite(str(out / f"video_{tag}_{posture}.jpg"), np.hstack([a, b]))
    writer.close()
    print("video", path, writer.frames, "frames")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("out/realism"))
    parser.add_argument("--tag", default="after")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stage = Stage()
    if args.compare:
        compare(stage, args.out, args.tag)
    if args.detect:
        detect(stage, args.out, args.tag)
    if args.video:
        video(Stage(), args.out, args.tag)


if __name__ == "__main__":
    main()
