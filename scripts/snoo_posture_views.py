"""シミュレーターの snoo の姿勢を、元の写真と並べて絵にする（形を目で確かめるため）。

    MUJOCO_GL=egl .venv/bin/python -m scripts.snoo_posture_views --out out/snoo_postures
姿勢ごとに、写真と DeepLabCut の点（scripts/snoo_posture_fit.py が残した keypoints.json）・
車のカメラ（床から 8 cm、検出の箱つき）・斜め上から見た絵を横に並べる。
--dlc を付けると、車のカメラの絵に DeepLabCut（CPU）をかけ、点の確からしさを JSON に書く。
シミュレーターの絵で猫の関節の見つけ方を試せるか（学習や試験に使えるか）の目安にする。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from app.cat_pose import NAMES  # noqa: E402
from app.roommap import Pose  # noqa: E402
from app.snoo_body import POSTURES, load_postures  # noqa: E402
from app.snoo_sim import Sim, SnooPose, synthetic_room  # noqa: E402
from app.snoo_sim_render import draw_camera_hud  # noqa: E402
from scripts.snoo_posture_fit import SOURCES, fit_height, label, load_image  # noqa: E402

# 鼻・目・耳・足先・しっぽの先など、長毛でも取りやすい点（app/cat_pose.py の説明から）
EASY = (
    "nose", "left_eye", "right_eye", "left_earend", "right_earend", "tail_end",
    "front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw",
)  # fmt: skip


@dataclasses.dataclass
class ShowPose(SnooPose):
    posture: str = "walk"


class StillSnoo:
    """決めた所で、決めた姿勢のまま動かない snoo（絵を撮るため）。"""

    def __init__(self, pose: SnooPose):
        self.pose = pose

    def step(self, dt: float, robot: Pose | None) -> SnooPose:
        return self.pose


def place(sim: Sim, posture: str, dist: float, facing_deg: float) -> None:
    """車のカメラの dist m 先に、カメラから見て facing_deg（0 は右を向いた横向き、90 は
    こちら向き）の向きで snoo を置き、姿勢を移し終えるまで時間を進める。"""
    cam = sim.car.camera_pose()
    x, y = cam.x + dist * math.cos(cam.yaw), cam.y + dist * math.sin(cam.yaw)
    yaw = cam.yaw - math.pi / 2 + math.radians(facing_deg)
    sim.snoo = StillSnoo(ShowPose(x, y, yaw, False, posture))
    sim.snoo_anim.update(5.0, sim.snoo.pose)
    sim.sync()


class Views:
    def __init__(self):
        self.sim = Sim(synthetic_room(), rays=(64, 48), seed=0)
        self.sim.car.pan_deg = self.sim.car.tilt_deg = 0.0
        m = self.sim.model
        self.rcam = mujoco.Renderer(m, 480, 640, max_geom=2000)
        self.r34 = mujoco.Renderer(m, 480, 640, max_geom=2000)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[2] = 0  # 車は描かない

    def car_cam(self, hud: bool = True) -> np.ndarray:
        self.rcam.update_scene(self.sim.data, "car_cam", self.opt)
        img = cv2.cvtColor(self.rcam.render(), cv2.COLOR_RGB2BGR)
        return draw_camera_hud(np.ascontiguousarray(img), self.sim) if hud else img

    def three_quarter(self) -> np.ndarray:
        s = self.sim.snoo.pose
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [s.x, s.y, 0.12]
        cam.distance, cam.elevation = 0.95, -25
        cam.azimuth = math.degrees(s.yaw) + 135
        self.r34.update_scene(self.sim.data, cam, self.opt)
        return cv2.cvtColor(self.r34.render(), cv2.COLOR_RGB2BGR)


def photo_panel(name: str, keypoints: dict) -> np.ndarray:
    key = SOURCES[name].image
    img = load_image(key)
    kp = keypoints.get(key)
    if kp is not None:
        for (x, y, c), n in zip(kp["keypoints"], NAMES, strict=True):
            if c >= 0.4:
                col = (
                    (60, 230, 60)
                    if "left" in n
                    else (60, 60, 240)
                    if "right" in n
                    else (240, 220, 60)
                )
                cv2.circle(img, (int(x), int(y)), 6, col, -1, cv2.LINE_AA)
    return label(img, f"{name}: snoo photo {key} + DeepLabCut")


def dlc_on_renders(views: Views, out: Path, tag: str) -> dict:
    """車のカメラの絵（距離 3 つ × 向き 3 つ × 姿勢）に、検出と DeepLabCut を CPU でかける。"""
    from scripts.snoo_posture_fit import cpu_pose_models

    det, pose = cpu_pose_models()
    rows = []
    for name in POSTURES:
        for dist in (0.5, 0.8, 1.2):
            for facing in (0, 45, 90):
                place(views.sim, name, dist, facing)
                box = views.sim.detection
                if box is None:
                    continue
                img = views.car_cam(hud=False)
                dets = det.detect(img)
                u0, v0, u1, v1, _ = box
                kp = pose.estimate(img, (u0 / 640, v0 / 480, u1 / 640, v1 / 480))
                easy = [NAMES.index(n) for n in EASY]
                rows.append(
                    {
                        "posture": name, "dist": dist, "facing": facing,
                        "cat": round(max((d.cat for d in dets), default=0.0), 3),
                        "conf_all": round(float(kp[:, 2].mean()), 3),
                        "conf_easy": round(float(kp[easy, 2].mean()), 3),
                    }
                )  # fmt: skip
                if dist == 0.8:
                    vis = img.copy()
                    for x, y, c in kp:
                        if c >= 0.4:
                            cv2.circle(vis, (int(x), int(y)), 3, (60, 230, 60), -1)
                    cv2.imwrite(str(out / f"dlc_{tag}_{name}_{facing}.jpg"), vis)
    summary = {
        k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("cat", "conf_all", "conf_easy")
    }
    by_posture = {
        n: {
            k: round(float(np.mean([r[k] for r in rows if r["posture"] == n])), 3)
            for k in ("cat", "conf_all", "conf_easy")
        }
        for n in POSTURES
    }
    return {"mean": summary, "by_posture": by_posture, "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("out/snoo_postures"))
    parser.add_argument("--dlc", action="store_true", help="車のカメラの絵に DeepLabCut をかける")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "keypoints.json"
    keypoints = json.loads(cache.read_text()) if cache.exists() else {}
    library = json.loads(Path("app/snoo_postures.json").read_text())["postures"]
    load_postures()  # 姿勢の本が読めることを先に確かめる
    views = Views()
    rows = []
    for name in POSTURES:
        place(views.sim, name, 1.0, 30)
        cam = label(views.car_cam(), f"robot camera 8 cm ({library[name]['source']})")
        q34 = label(views.three_quarter(), "3/4 view")
        row = np.hstack([fit_height(photo_panel(name, keypoints), 480), cam, q34])
        cv2.imwrite(str(args.out / f"posture_{name}.jpg"), row)
        rows.append(row)
    w = max(r.shape[1] for r in rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], 0) for r in rows])
    cv2.imwrite(str(args.out / "postures.jpg"), sheet)
    if args.dlc:
        result = dlc_on_renders(views, args.out, "new")
        (args.out / "dlc_sim.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({"mean": result["mean"], "by_posture": result["by_posture"]}))


if __name__ == "__main__":
    main()
