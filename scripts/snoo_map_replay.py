"""実走のセッションの画像を、地図のサーバーの処理（plan_station）に 1 か所ずつ流し直して、
1 ステーションの時間と位置を測る。GPU（VGGT）を使う。

    .venv/bin/python -m scripts.snoo_map_replay data/snoo_map/sessions/explore_20261029_223944 \\
        --out data/replay/w4 --window 4 --old-frames 6
走っている地図のサーバー（:8720）とは別のプロセスで、別の置き場（--out）に書く。
GPU の CU はサーバーと同じ（SNOO_MAP_CU_MASK、既定 0:0-71）にする（猫の検出と分け合う）。
VGGT の窓・コマの数を変えて、時間（vggt・map・合わせて）と、位置（元のセッションの
poses.json との差）を比べる。返事（次の 1 歩）は実走と違ってよい（Pi はその通りには
走っていない）。位置の差は、窓の選び方が同じでも VGGT のぶれで数 cm ある。
"""

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, help="VGGT に入れるステーションの数")
    parser.add_argument("--max-frames", type=int, help="VGGT に入れるコマの数の上限")
    parser.add_argument("--old-frames", type=int, help="最後でないステーションのコマの数")
    parser.add_argument("--stations", type=int, help="この数のステーションまで")
    args = parser.parse_args()

    import app.snoo_map_server as server

    os.environ.setdefault("HSA_CU_MASK", server.CU_MASK)
    if args.window:
        server.WINDOW = args.window
    if args.max_frames:
        server.MAX_FRAMES = args.max_frames
    if args.old_frames:
        server.OLD_FRAMES = args.old_frames
    ref = server.read_poses(args.session)
    session = args.out / args.session.name
    shutil.rmtree(session, ignore_errors=True)
    session.mkdir(parents=True)
    rec = server.VggtReconstruct()
    rec.load()
    rows = []
    ids = server.station_ids(args.session)[: args.stations]
    for i in ids:
        src = server.station_dir(args.session, i)
        dst = server.station_dir(session, i)
        dst.mkdir()
        for f in src.iterdir():
            if f.is_file() and f.name != "obs.npz":
                shutil.copy2(f, dst / f.name)
        marks: dict = {}
        t = time.perf_counter()
        action = server.plan_station(session, i, rec, marks)
        marks["total"] = time.perf_counter() - t
        poses = server.read_poses(session)
        row = {"station": i, **{k: round(v, 3) for k, v in marks.items()}}
        row["window"] = len(json.loads((session / "plan.json").read_text())["window"]["stations"])
        row["frames"] = int(np.load(session / "vggt.npz")["files"].size)
        if i in poses and i in ref:
            a, b = poses[i], ref[i]
            row["pose_diff_m"] = round(math.hypot(a.x - b.x, a.y - b.y), 3)
            row["yaw_diff_deg"] = round(
                abs(math.degrees(math.remainder(a.yaw - b.yaw, 2 * math.pi))), 1
            )
        row["action"] = action.get("action") + ("/reverse" if action.get("reverse") else "")
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    tot = [r["total"] for r in rows[1:]]
    diffs = [r["pose_diff_m"] for r in rows if "pose_diff_m" in r]
    summary = {
        "window": server.WINDOW,
        "max_frames": server.MAX_FRAMES,
        "old_frames": server.OLD_FRAMES,
        "total_s_mean": round(float(np.mean(tot)), 2) if tot else None,
        "total_s_max": round(float(np.max(tot)), 2) if tot else None,
        "vggt_s_mean": round(float(np.mean([r["vggt"] for r in rows[1:]])), 2) if tot else None,
        "pose_diff_m_median": round(float(np.median(diffs)), 3) if diffs else None,
        "pose_diff_m_max": round(float(np.max(diffs)), 3) if diffs else None,
        "registered": sum("pose_diff_m" in r for r in rows),
    }
    print(json.dumps(summary, ensure_ascii=False))
    (args.out / "replay.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1)
    )


if __name__ == "__main__":
    main()
