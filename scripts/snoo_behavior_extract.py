"""実機の録画と動画から、snoo の動きをフレームごとに取り出す（猫の動きのモデルの素材）。

  .venv/bin/python -m scripts.snoo_behavior_extract
シミュレーターの猫（app/snoo_behavior.py）を本物の snoo に近づけるため、録画のフレームごとに
猫の箱・関節・状態・カメラと車の姿勢・車からの距離を data/snoo_behavior/tracks/ に書く。

- 録画：data/snoo_rec/<日付>/<時刻>_<送り元>/（app/snoo_record.py。jpg と frames.jsonl）。
  検出は録ったときの返事ではなく、ここで同じ検出器（app/snoo.CatDetector）をかけ直す。
  録った日の返事は見本（gallery）が無い時間もあり、snoo らしさがそろわないため
- 関節：app/cat_pose.CatPose（DeepLabCut SuperAnimal-Quadruped。非商用の重み、家で使うだけ）
- 状態：data/snoo/state_head.npz（CLIP の埋め込みの上の線形の分類器。scripts/snoo_state_head.py）
- カメラ：frames.jsonl の cmd（SNO2 の指令のパン・チルト）。チルトのサーボは 19:20 と 20:20 に
  直したので、その前のフレームは TILT_FIXES の分だけ下げる。cmd の無い（SNO1）フレームは不明
- 車：cmd の speed・steer_deg から速さ。位置は、同じ時刻の探索のセッション（data/snoo_map/
  sessions。Pi の CLOCK_MONOTONIC が同じ）があればステーションの位置から、無ければ場面の始めを
  原点にして指令を足し合わせる（推測航法。ずれる）
- 距離：箱の下の端が床に着いていれば、カメラの高さと角度から（ground）。着いていない（猫が
  家具の上・箱が画面の下で切れる）ときは、箱の大きさと猫の大きさから（size）
- 先生（--teacher）：取り出した後に、snoo のフレームを --teacher-every 秒おきに Clef
  （scripts/snoo_teacher.py と同じ質問）でも見て、cat.teacher に書く。状態の分類器は写真
  （data/snoo_raw）で学んだもので、録画の低いカメラでは座っている猫を寝そべりと言うことが多い。
  Clef は 1 枚 0.4 秒・VRAM 約 21 GB なので、間引いて別に回す
- 動画：data/snoo_video/*.mov（床の高さで撮ったスマホの動画）。カメラの向きも距離も分からないので、
  速さは体の長さ（鼻から しっぽの付け根）あたりで出す

GPU は 1 つのプロセスだけで使う（/sys/class/kfd/kfd/proc を先に確かめる）。全部で 5 分ほど。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import cv2
import numpy as np

from app.roommap import Pose, arc, camera_of, rear_axle

REC = Path("data/snoo_rec")
VIDEO = Path("data/snoo_video")
SESSIONS = Path("data/snoo_map/sessions")
OUT = Path("data/snoo_behavior")
STATE_HEAD = Path("data/snoo/state_head.npz")
GALLERY = "data/snoo/gallery.npz"
# 箱を取る下限は検出のサーバーと同じ。snoo とみなすのは録画と同じ下限（app/snoo_record.py）
CAT_MIN, SNOO_CAT_MIN, SNOO_MIN = 0.25, 0.5, 0.3
# カメラ（app/snoo_sim.py と同じ値。あちらは mujoco を読み込むので、ここでは読まない）
CAM_H_M, HFOV_DEG = 0.08, 54.0
SPEED_M_S = 0.017
# (この時刻（家の時計）より前なら, チルトにこれだけ足す)。サーボを直す前の録画のため
TILT_FIXES = ((dt.time(19, 20), -26.5), (dt.time(20, 20), -4.5))
# 箱の大きさから距離を出すときの snoo の大きさ（m）。長毛で丸く見えるので、体の高さは
# 立って 27 cm（app/snoo_sim の SNOO_HEIGHT_M）、座って 30 cm、寝そべって 15 cm にする。
# 横から見た長さはしっぽを除いて 45 cm。箱の高さと幅のうち、向きで変わりにくい高さを主に使う
CAT_HEIGHT_M = {"sleeping": 0.13, "lying": 0.15, "sitting": 0.30, "grooming": 0.22}
CAT_HEIGHT_DEFAULT_M = 0.27
# 箱の下の端が画面の下からこれより内側なら、床に着いた足が写っているとみなす
BOTTOM_CUT = 0.97


def tilt_fix(t_recv: float) -> float:
    local = dt.datetime.fromtimestamp(t_recv).time()
    for until, fix in TILT_FIXES:
        if local < until:
            return fix
    return 0.0


def focal_px(width: int) -> float:
    return width / 2 / math.tan(math.radians(HFOV_DEG / 2))


def box_geometry(
    box, size, pan_deg: float | None, tilt_deg: float | None, activity: str | None
) -> dict:
    """箱から、車から見た向き（左が +）・距離・猫の下の端の高さを見積もる。

    ground：箱の下の端の画素の向きが床と交わる所。角度が浅いと少しの誤差で大きく変わるので、
    水平より 2° 以上下を向いているときだけ使う。size：箱の高さと猫の大きさから。
    """
    w, h = size
    f = focal_px(w)
    u = (box[0] + box[2]) / 2 * w
    out: dict = {"bearing_deg": None, "dist_ground_m": None, "dist_size_m": None}
    off = math.degrees(math.atan((u - w / 2) / f))
    if pan_deg is not None:
        out["bearing_deg"] = round(-(pan_deg + off), 1)
    box_h = max((box[3] - box[1]) * h, 1.0)
    real_h = CAT_HEIGHT_M.get(activity or "", CAT_HEIGHT_DEFAULT_M)
    cut = box[3] >= BOTTOM_CUT or box[1] <= 1 - BOTTOM_CUT
    if not cut:
        out["dist_size_m"] = round(f * real_h / box_h, 2)
    if tilt_deg is not None and box[3] < BOTTOM_CUT:
        below = math.degrees(math.atan((box[3] * h - h / 2) / f)) - tilt_deg
        if below > 2.0:
            out["dist_ground_m"] = round(CAM_H_M / math.tan(math.radians(below)), 2)
        elif out["dist_size_m"] is not None:
            # 床に着いていない（家具の上）。size の距離で、下の端の高さを出す
            out["floor_z_m"] = round(
                CAM_H_M + out["dist_size_m"] * math.tan(math.radians(-below)), 2
            )
    g, s = out["dist_ground_m"], out["dist_size_m"]
    # ground があれば ground（形に頼らない）。size しか無ければ size
    out["dist_m"] = g if g is not None else s
    out["dist_src"] = "ground" if g is not None else ("size" if s is not None else None)
    return out


class StateHead:
    """data/snoo/state_head.npz の線形の分類器（質問ごとに確率）。"""

    def __init__(self, path: Path):
        d = np.load(path)
        self.q = {
            str(q): (d[f"{q}_w"], d[f"{q}_b"], [str(c) for c in d[f"{q}_classes"]])
            for q in d["questions"]
        }

    def __call__(self, emb: np.ndarray) -> dict:
        out = {}
        for q, (w, b, classes) in self.q.items():
            z = emb @ w * 10 + b
            p = np.exp(z - z.max())
            p /= p.sum()
            out[q] = {c: round(float(v), 3) for c, v in zip(classes, p, strict=True)}
        return out


class Models:
    def __init__(self):
        import torch

        from app.cat_pose import CatPose
        from app.snoo import CatDetector

        self.torch = torch
        self.det = CatDetector(cat_min=CAT_MIN, gallery=GALLERY)
        self.pose = CatPose(self.det.device)
        self.head = StateHead(STATE_HEAD)

    def run(self, image: np.ndarray) -> dict | None:
        """いちばん snoo らしい箱の 1 匹の箱・関節・状態。猫がいなければ None。"""
        from app.snoo import to_tensor

        dets = self.det.detect(image)
        if not dets:
            return None
        best = max(dets, key=lambda d: (d.cat >= SNOO_CAT_MIN and (d.snoo or 0) >= SNOO_MIN, d.cat))
        x = to_tensor(image, self.det.device, self.det.dtype)
        emb = self.det.scorer.embed(x, [best.box])[0].cpu().numpy()
        kp = self.pose.estimate(image, best.box)
        return {
            "box": [round(v, 4) for v in best.box],
            "cat": round(best.cat, 3),
            "snoo": round(best.snoo, 3) if best.snoo is not None else None,
            "is_snoo": bool(best.cat >= SNOO_CAT_MIN and (best.snoo or 0) >= SNOO_MIN),
            "n_cats": len(dets),
            "keypoints": [
                [round(float(a), 1), round(float(b), 1), round(float(c), 3)] for a, b, c in kp
            ],
            "state": self.head(emb),
        }


def station_tracks(root: Path) -> list[tuple[int, int, Pose, str]]:
    """探索のセッションのステーションごとの (最初のフレームの t_capture_ns, 最後の, 位置, 名前)。"""
    out = []
    for session in sorted(root.glob("explore_*")):
        poses = (
            json.loads((session / "poses.json").read_text())
            if (session / "poses.json").exists()
            else {}
        )
        for meta_path in sorted(session.glob("[0-9][0-9][0-9]/meta.json")):
            meta = json.loads(meta_path.read_text())
            sid = str(meta.get("station_id", int(meta_path.parent.name)))
            ts = [f["t_capture_ns"] for f in meta.get("frames", []) if "t_capture_ns" in f]
            if not ts or sid not in poses:
                continue
            p = poses[sid]
            pose = Pose(p["x"], p["y"], math.radians(p["yaw_deg"]))
            out.append((min(ts), max(ts), pose, f"{session.name}#{sid}"))
    return out


def robot_poses(rows: list[dict], stations) -> list[tuple[Pose | None, str | None]]:
    """フレームごとの車（カメラの真下）の姿勢。ステーションの位置から指令を足し合わせる。

    同じ時刻の探索のセッションが無ければ、場面の始めを原点にした推測航法（station は None）。
    """
    out: list[tuple[Pose | None, str | None]] = []
    t0 = rows[0]["t_capture_ns"]
    # 場面の始めより前の、いちばん近いステーション（60 秒以内）
    before = [s for s in stations if s[0] <= t0 + 60e9 and t0 - s[1] < 60e9]
    src = "odom"
    cur = rear_axle(Pose(0.0, 0.0, 0.0))
    if before:
        s = max(before, key=lambda s: s[0])
        cur, src = rear_axle(s[2]), "station"
    last_t = t0
    for r in rows:
        t = r["t_capture_ns"]
        # ステーションに着いた（撮り始めた）なら、そこの位置に合わせ直す
        for s0, s1, pose, _ in stations:
            if s0 <= t <= s1:
                cur, src = rear_axle(pose), "station"
        cmd = r.get("cmd")
        if cmd is None:
            out.append((None, None))
            continue
        ds = cmd["speed"] * SPEED_M_S * (t - last_t) / 1e9
        if abs(ds) > 0:
            x, y, yaw = arc(cur.x, cur.y, cur.yaw, cmd["steer_deg"], ds)
            cur = Pose(x, y, yaw)
        last_t = t
        out.append((camera_of(cur), src))
    return out


def extract_session(path: Path, models: Models, stations) -> list[dict]:
    rows = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: r["t_capture_ns"])
    poses = robot_poses(rows, stations)
    out = []
    for r, (pose, src) in zip(rows, poses, strict=True):
        image = cv2.imread(str(path / r["file"]))
        if image is None:
            continue
        cmd = r.get("cmd")
        rec: dict = {
            "t": round(r["t_capture_ns"] / 1e9, 3),
            "t_recv": r["t_recv"],
            "file": r["file"],
            "size": r["size"],
            "camera": None,
            "robot": None,
        }
        if cmd is not None:
            tilt = cmd["tilt_deg"] + tilt_fix(r["t_recv"])
            rec["camera"] = {
                "pan_deg": round(cmd["pan_deg"], 2),
                "tilt_deg": round(tilt, 2),
                "height_m": CAM_H_M,
            }
            rec["robot"] = {
                "speed_m_s": round(cmd["speed"] * SPEED_M_S, 3),
                "steer_deg": cmd["steer_deg"],
                "moving": bool(cmd["flags"] & 1),
                "reversing": bool(cmd["flags"] & 2),
                "pose": None
                if pose is None
                else [round(pose.x, 3), round(pose.y, 3), round(math.degrees(pose.yaw), 1)],
                "pose_src": src,
            }
        cat = models.run(image)
        if cat is not None:
            act = max(cat["state"]["activity"], key=cat["state"]["activity"].get)
            cam = rec["camera"] or {}
            geo = box_geometry(cat["box"], r["size"], cam.get("pan_deg"), cam.get("tilt_deg"), act)
            cat |= geo
            if pose is not None and geo["dist_m"] is not None and geo["bearing_deg"] is not None:
                # 車のカメラの真下から見た猫の位置を、部屋の座標に
                b = pose.yaw + math.radians(geo["bearing_deg"])
                cat["world"] = [
                    round(pose.x + geo["dist_m"] * math.cos(b), 3),
                    round(pose.y + geo["dist_m"] * math.sin(b), 3),
                ]
        rec["cat"] = cat
        out.append(rec)
    return out


def extract_video(path: Path, models: Models) -> list[dict]:
    capture = cv2.VideoCapture(str(path))  # 縦に撮った動画は向きを直して読む
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    out, k = [], 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        cat = models.run(image)
        if cat is not None:
            kp = np.array(cat["keypoints"])
            # 体の長さ（画素）：鼻としっぽの付け根が両方とれていれば、その間
            nose, tail = kp[0], kp[22]
            if nose[2] > 0.3 and tail[2] > 0.3:
                cat["body_px"] = round(float(np.hypot(*(nose[:2] - tail[:2]))), 1)
        out.append(
            {
                "t": round(k / fps, 3),
                "file": f"{path.name}#{k}",
                "size": [image.shape[1], image.shape[0]],
                "camera": None,
                "robot": None,
                "cat": cat,
            }
        )
        k += 1
    return out


def add_teacher(tracks: Path, rec: Path, video: Path, every_s: float) -> None:
    """snoo のフレームを every_s 秒おきに Clef に見せて、cat.teacher に確率を書く。"""
    from app import clef
    from app.config import settings
    from scripts.snoo_teacher import QUESTIONS, STATE, crop

    teacher = clef.from_settings(settings)
    for path in sorted(tracks.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        capture, frame_no = None, -1
        if rows and "#" in rows[0]["file"]:
            capture = cv2.VideoCapture(str(video / rows[0]["file"].split("#")[0]))
        day, _, sess = path.stem.partition("_")
        last, n = -1e9, 0
        for r in rows:
            c = r["cat"]
            image = None
            if capture is not None:
                k = int(r["file"].split("#")[1])
                while frame_no < k:
                    ok, image = capture.read()
                    frame_no += 1
            if not c or not c["is_snoo"] or r["t"] - last < every_s:
                continue
            if capture is None:
                image = cv2.imread(str(rec / day / sess / r["file"]))
            if image is None:
                continue
            jpeg = cv2.imencode(".jpg", crop(image, c["box"]), [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
            answers = teacher.ask(STATE, QUESTIONS, [(jpeg.tobytes(), "image/jpeg")])
            c["teacher"] = {q: answers[q]["probabilities"] for q in QUESTIONS}
            last, n = r["t"], n + 1
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(path.stem, f"Clef {n} 枚", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rec", type=Path, default=REC)
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", help="この文字を含む場面だけ（試すとき）")
    parser.add_argument(
        "--teacher", action="store_true", help="書いた tracks に Clef の印を足すだけ"
    )
    parser.add_argument("--teacher-every", type=float, default=0.5)
    args = parser.parse_args()
    if args.teacher:
        add_teacher(args.out / "tracks", args.rec, args.video, args.teacher_every)
        return

    models = Models()
    stations = station_tracks(SESSIONS)
    tracks = args.out / "tracks"
    tracks.mkdir(parents=True, exist_ok=True)
    summary = {}
    jobs = [("rec", p) for p in sorted(args.rec.glob("*/*/")) if (p / "frames.jsonl").exists()]
    jobs += [("video", p) for p in sorted(args.video.glob("*.mov"))]
    for kind, path in jobs:
        name = f"{path.parent.name}_{path.name}" if kind == "rec" else path.stem
        if args.only and args.only not in name:
            continue
        rows = (
            extract_session(path, models, stations)
            if kind == "rec"
            else extract_video(path, models)
        )
        with (tracks / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        snoo = sum(bool(r["cat"] and r["cat"]["is_snoo"]) for r in rows)
        summary[name] = {
            "kind": kind,
            "frames": len(rows),
            "snoo_frames": snoo,
            "camera": sum(r["camera"] is not None for r in rows),
            "moving": sum(bool(r["robot"] and r["robot"]["moving"]) for r in rows),
            "pose_src": sorted(
                {r["robot"]["pose_src"] for r in rows if r["robot"] and r["robot"]["pose_src"]}
            ),
        }
        print(name, json.dumps(summary[name]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
