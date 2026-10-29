"""保存した探索のセッションを、ロボット無しで SLAM の道筋（タップ → 置き場 → SLAM → 結果）に
流し直し、ステーションの位置（poses.json）と比べる。VGGT を使うので SLAM の venv で動かす（GPU）。

    HSA_CU_MASK=0:0-71 ~/orca/vggt-slam-venv/bin/python -m scripts.snoo_slam_replay \\
        data/snoo_map/sessions/explore_20261029_223944 --out /tmp/slam --mode all

- 走りながら撮ったコマ（meta.json の moving が true。5 枚/秒）は、SNO2 のパケットに組み直して
  app/snoo_slam_tap.SlamTap に渡す（検出サーバーと同じ）。撮った時刻の間隔どおりに渡す
  （--speed で速める。止まっている間は 2 秒に縮める）
- 止まって撮ったコマは、ステーションの最後のコマの時刻に、仮のセッションの置き場にステーションの
  置き場を置く（地図のサーバーが受けて残すのと同じ）。poses.json もそこまでのステーションを書く
- --mode all：両方。moving：走りながらのコマだけ（ステーションの位置は止まる直前のコマ）。
  still：止まって撮ったコマだけ

比べるもの（ステーションごと。SLAM のステーションの位置は止まって撮ったコマの平均、moving では
止まる直前のコマ）:
- live：SLAM が書いた位置そのまま（大きさは床とカメラの高さ、向きと位置は poses.json に
  合わせたもの）
- anchor0：ステーション 0 だけで合わせ直したもの（ほかのステーションの答えを使わない）
- sim2：位置で大きさも合わせたもの（形がどれだけ合うか）
poses.json も VGGT（窓ごと）から出したもので、本当の位置ではない。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from app.snoo import HEADER_V2, MAGIC_V2, MOVING, REVERSING, parse_frame
from app.snoo_slam import (
    SlamService,
    quiet_vggt_slam,
    read_station_poses,
    read_trajectory,
    station_dirs,
    station_frames,
    umeyama,
)
from app.snoo_slam_tap import SlamTap

SENDER = ("192.168.0.188", 40000)
# 止まっている間（首振り・地図のサーバーの返事待ち）の長さの上限（秒、流し直しの時間）
MAX_GAP_S = 2.0


def packet(seq: int, k, jpeg: bytes) -> bytes:
    flags = MOVING | (REVERSING if k.speed < 0 else 0)
    head = HEADER_V2.pack(
        MAGIC_V2, seq, k.t_capture_ns, 640, 480, k.steer_deg, k.speed, k.pan_deg, k.tilt_deg, flags
    )
    return head + jpeg


def events(session: Path) -> list[tuple[int, str, object]]:
    """(撮った時刻, 種類, 中身) を時刻の順に。"""
    ev = []
    for d in station_dirs(session):
        stills = station_frames(d, still=True, moving=False)
        for k in station_frames(d, still=False, moving=True):
            ev.append((k.t_capture_ns, "moving", k))
        if stills:
            ev.append((stills[-1].t_capture_ns + 1, "station", d))
    return sorted(ev, key=lambda e: e[0])


def write_poses(dst: Path, poses: dict, upto: int) -> None:
    data = {k: v for k, v in poses.items() if int(k) <= upto}
    (dst / "poses.json").write_text(json.dumps(data, indent=1))


def replay(args) -> dict:
    src = Path(args.session).resolve()
    out = Path(args.out).resolve() / f"{src.name}_{args.mode}"
    if out.exists():
        shutil.rmtree(out)
    data, queue = out / "data", out / "queue"
    fake = data / src.name
    fake.mkdir(parents=True)
    queue.mkdir()
    all_poses = json.loads((src / "poses.json").read_text())
    tap = SlamTap(queue, hz=args.tap_hz, max_files=10_000)
    svc = SlamService(
        data,
        queue,
        submap_size=args.submap_size,
        flush_s=args.flush_s,
        loop=args.loop,
        still=args.mode != "moving",
    )
    svc.preload()
    ev = events(src)
    t_wall0, t_prev, t_sim = time.time(), None, 0.0
    seq = 0
    for t, kind, item in ev:
        if t_prev is not None:
            t_sim += min((t - t_prev) / 1e9, MAX_GAP_S) / max(args.speed, 1e-9)
        t_prev = t
        while args.speed > 0 and time.time() - t_wall0 < t_sim:
            if not svc.poll():
                time.sleep(0.01)
        if kind == "moving" and args.mode != "still":
            seq += 1
            frame = parse_frame(packet(seq, item, item.image.read_bytes()))
            tap.offer(SENDER, frame, [])
            tap.flush()
        elif kind == "station":
            (fake / item.name).symlink_to(item)
            write_poses(fake, all_poses, int(item.name))
            fake.touch()
        if args.speed <= 0:
            svc.poll()
    while svc.session is None or svc.pending or svc.scan():
        svc.poll()
        svc.drain()
    svc.update()
    wall = time.time() - t_wall0
    return evaluate(src, svc, out, wall, args)


def station_estimates(rows: list[dict], session: Path, mode: str) -> dict[int, dict]:
    """SLAM の行から、ステーションごとの位置と向き。"""
    out = {}
    for d in station_dirs(session):
        sid = int(d.name)
        stills = station_frames(d)
        if not stills:
            continue
        if mode != "moving":
            mine = [r for r in rows if r["station"] == sid]
        else:
            t0 = stills[0].t_capture_ns
            before = [
                r for r in rows if r["station"] is None and t0 - 3e9 <= r["t_capture_ns"] < t0
            ]
            mine = before[-1:]
        if not mine:
            continue
        head = min(mine, key=lambda r: (round(abs(r["tilt_deg"])), abs(r["pan_deg"])))
        out[sid] = {
            "x": float(np.mean([r["x"] for r in mine])),
            "y": float(np.mean([r["y"] for r in mine])),
            "yaw": head["yaw"],
            "segment": head["segment"],
        }
    return out


def _wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def errors(est: dict, ref: dict, ids: list[int]) -> dict:
    if not ids:
        return {"n": 0}
    p = np.array([[est[i]["x"], est[i]["y"]] for i in ids])
    q = np.array([ref[i][:2] for i in ids])
    dy = _wrap([est[i]["yaw"] - ref[i][2] for i in ids])
    d = np.linalg.norm(p - q, axis=1)
    return {
        "n": len(ids),
        "rmse_cm": round(float(np.sqrt((d**2).mean())) * 100, 1),
        "max_cm": round(float(d.max()) * 100, 1),
        "heading_mae_deg": round(float(np.degrees(np.abs(dy)).mean()), 1),
        "heading_max_deg": round(float(np.degrees(np.abs(dy)).max()), 1),
    }


def _stats(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "p50": round(float(np.median(v)), 2),
        "p90": round(float(np.percentile(v, 90)), 2),
        "max": round(max(v), 2),
    }


def evaluate(src: Path, svc: SlamService, out: Path, wall: float, args) -> dict:
    slam_out = svc.out
    rows = read_trajectory(slam_out)
    status = json.loads((slam_out / "status.json").read_text())
    ref = read_station_poses(src)
    est = station_estimates(rows, src, args.mode)
    ids = sorted(set(est) & set(ref))
    res = {"session": src.name, "mode": args.mode, "loop": args.loop, "wall_s": round(wall, 1)}
    res["live"] = errors(est, ref, ids)
    # ステーション 0 だけで合わせ直す（区間 0 のみ）
    if 0 in est:
        e0 = est[0]
        c, s = math.cos(-e0["yaw"]), math.sin(-e0["yaw"])
        a0 = {}
        for i in ids:
            if est[i]["segment"] != e0["segment"]:
                continue
            dx, dy = est[i]["x"] - e0["x"], est[i]["y"] - e0["y"]
            a0[i] = {"x": c * dx - s * dy, "y": s * dx + c * dy, "yaw": est[i]["yaw"] - e0["yaw"]}
        res["anchor0"] = errors(a0, ref, sorted(a0))
    if len(ids) >= 3:
        p = np.array([[est[i]["x"], est[i]["y"]] for i in ids])
        q = np.array([ref[i][:2] for i in ids])
        sc, r, t = umeyama(p, q, with_scale=True)
        fit = sc * p @ r.T + t
        th = math.atan2(r[1, 0], r[0, 0])
        sim = {
            i: {"x": fit[j, 0], "y": fit[j, 1], "yaw": est[i]["yaw"] + th}
            for j, i in enumerate(ids)
        }
        res["sim2"] = {**errors(sim, ref, ids), "scale": round(sc, 3)}
    timing = svc.slam.timing
    sub_ms = [t["submap_ms"] for t in timing]
    frames = [t["frames"] for t in timing]
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    res["stations"] = len(ids)
    res["keyframes"] = len(rows)
    res["submaps"] = status["submaps"]
    res["segments"] = status["segments"]
    res["failures"] = status["failures"]
    res["loop_closures"] = status["loop_closures"]
    res["submap_ms"] = {
        "p50": round(float(np.median(sub_ms)), 0) if sub_ms else None,
        "max": round(max(sub_ms), 0) if sub_ms else None,
    }
    vggt = [t["vggt_ms"] for t in timing]
    res["vggt_ms_p50"] = round(float(np.median(vggt)), 0) if vggt else None
    res["ms_per_keyframe"] = round(sum(sub_ms) / max(1, sum(frames) - len(frames)), 0)
    res["latency_s"] = _stats(lat)
    # 走りながらのコマ（タップ）と止まって撮ったコマ（10 枚まとめて届く）で分ける
    for src_name in ("tap", "still"):
        v = [r["latency_s"] for r in rows if r["source"] == src_name and r.get("latency_s")]
        res[f"latency_{src_name}_s"] = _stats(v)
    res["vram_gb_max"] = max((t["vram_gb"] for t in timing), default=None)
    res["alignment"] = status["alignment"]
    res["per_station"] = {
        i: {
            "slam": [
                round(est[i]["x"], 3),
                round(est[i]["y"], 3),
                round(math.degrees(est[i]["yaw"]), 1),
            ],
            "ref": [round(ref[i][0], 3), round(ref[i][1], 3), round(math.degrees(ref[i][2]), 1)],
        }
        for i in ids
    }
    (out / "result.json").write_text(json.dumps(res, indent=1))
    if args.plot:
        plot(rows, slam_out, est, ref, Path(args.plot) / f"{src.name}_{args.mode}.png", res)
    return res


def plot(rows, slam_out: Path, est, ref, path: Path, res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.load(slam_out / "points.npz")
    xyz = pts["xyz"]
    fig, ax = plt.subplots(figsize=(8, 8))
    floor = np.abs(xyz[:, 2]) < 0.02
    obst = (xyz[:, 2] >= 0.02) & (xyz[:, 2] < 0.15)
    ax.scatter(xyz[floor, 0], xyz[floor, 1], s=1, c="#cccccc", lw=0, label="点：床（±2 cm）")
    ax.scatter(xyz[obst, 0], xyz[obst, 1], s=1, c="#eb6834", lw=0, label="点：障害物（2〜15 cm）")
    segs = sorted({r["segment"] for r in rows})
    for g in segs:
        rr = [r for r in rows if r["segment"] == g]
        ax.plot(
            [r["x"] for r in rr],
            [r["y"] for r in rr],
            "-",
            c="#2a78d6",
            lw=1.5,
            label="SLAM の道筋" if g == segs[0] else None,
        )
    for i, (x, y, yaw) in ref.items():
        ax.plot(x, y, "s", c="#333333", ms=7, label="poses.json" if i == min(ref) else None)
        ax.annotate(
            "",
            (x + 0.12 * math.cos(yaw), y + 0.12 * math.sin(yaw)),
            (x, y),
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.5},
        )
        ax.text(x + 0.02, y + 0.02, str(i), fontsize=8, color="#333333")
    for i, e in est.items():
        ax.plot(
            e["x"],
            e["y"],
            "o",
            c="#2a78d6",
            ms=6,
            mfc="white",
            mew=2,
            label="SLAM のステーション" if i == min(est) else None,
        )
        ax.annotate(
            "",
            (e["x"] + 0.12 * math.cos(e["yaw"]), e["y"] + 0.12 * math.sin(e["yaw"])),
            (e["x"], e["y"]),
            arrowprops={"arrowstyle": "->", "color": "#2a78d6", "lw": 1.5},
        )
    allx = [r["x"] for r in rows] + [v[0] for v in ref.values()]
    ally = [r["y"] for r in rows] + [v[1] for v in ref.values()]
    if obst.sum() > 100:  # 障害物の点の 2〜98 % も入れる
        allx += list(np.percentile(xyz[obst, 0], [2, 98]))
        ally += list(np.percentile(xyz[obst, 1], [2, 98]))
    if allx:
        ax.set_xlim(min(allx) - 0.3, max(allx) + 0.3)
        ax.set_ylim(min(ally) - 0.3, max(ally) + 0.3)
    ax.set_aspect("equal")
    ax.grid(True, color="#eeeeee", lw=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    live = res.get("live", {})
    ax.set_title(
        f"{res['session']} / {res['mode']}  RMSE {live.get('rmse_cm')} cm, "
        f"heading {live.get('heading_mae_deg')}°, "
        f"seg {res['segments']}, loop {res['loop_closures']}",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["all", "moving", "still"], default="all")
    parser.add_argument("--tap-hz", type=float, default=0.0, help="0 なら保存したコマをみな渡す")
    parser.add_argument("--submap-size", type=int, default=8)
    parser.add_argument("--flush-s", type=float, default=1.5)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed", type=float, default=1.0, help="流し直しの速さ（0 は待たない）")
    parser.add_argument("--plot", default="", help="上から見た絵を置く所")
    args = parser.parse_args()
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    quiet_vggt_slam()
    res = replay(args)
    import sys

    print(
        json.dumps({k: v for k, v in res.items() if k != "per_station"}, ensure_ascii=False),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
