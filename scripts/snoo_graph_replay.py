"""保存した探索のセッションを、地図のサーバーの plan_station に 1 か所ずつ順に流し直す
（GPU は使わない）。位置のグラフ（SNOO_POSE_GRAPH）を使うときと使わないときで、
返事・位置・間取りを比べるため。

    .venv/bin/python scripts/snoo_graph_replay.py data/snoo_map/sessions/explore_20261030_190244 \\
        --out /tmp/pg_replay/on --graph 1 [--json on.json] [--png on.png]

VGGT の代わりに、ステーションごとに残した recon.npz（地図のサーバーがそのとき回した窓の点）を使う。
窓に入れるステーションは、そのときのものに合わせる（グラフで位置が変わると、選ぶ近くの
ステーションが変わり、残した点では決め直せないため）。つまり「同じ窓の VGGT の答えで、位置の
決め方だけを変えたら」を見る。Pi の 1 歩（prev_step）は実走のままなので、返事が変わっても
次のステーションは変わらない。

もとのセッションには書かない。--out に meta.json だけを写して流す（画像は読まない）。
--roma にすると、RoMa v2 の辺（SNOO_ROMA_EDGES）も足す。そのときだけ画像へのリンクを置く
（RoMa は GPU を使う。1 か所 2 ジョブ・8 組まで）。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

import app.snoo_map_server as server
from app import snoo_posegraph as pg
from app.roommap import FREE, OCCUPIED, Pose, take
from app.snoo_mapper import session_grid


def window_of(keys: list[str]) -> list[int]:
    return sorted({int(k.split("/")[0]) for k in keys})


def odom_edges(session: Path, ids: list[int]) -> list[pg.Edge]:
    """続く 2 か所の間を走った辺（scripts/snoo_pose_bakeoff.py と同じ）。"""
    sent = server.read_reverses(session)
    have = set(ids)
    out = []
    for b in ids:
        if b - 1 not in have:
            continue
        prev = json.loads((session / f"{b:03d}" / "meta.json").read_text()).get("prev_step")
        if not isinstance(prev, dict):
            continue
        pieces, dist = server.step_pieces(prev), server.pi_distance_m(prev)
        if pieces is None or dist is None:
            continue
        rev = prev.get("reverse")
        rev = (b - 1) in sent if rev is None else bool(rev)
        out.append(pg.odom_edge(b - 1, b, take(pieces, dist), rev))
    return out


def odom_agreement(poses: dict[int, Pose], odom: list[pg.Edge]) -> dict:
    dt, dyaw = [], []
    for e in odom:
        if e.a in poses and e.b in poses:
            r = pg.relative(poses[e.a], poses[e.b])
            dt.append(math.hypot(r.x - e.z.x, r.y - e.z.y))
            dyaw.append(abs(math.degrees(pg.wrap(r.yaw - e.z.yaw))))
    q = lambda v, p: round(float(np.percentile(v, p)), 3) if v else None  # noqa: E731
    return {
        "n": len(dt),
        "t_med_cm": None if not dt else round(100 * q(dt, 50), 1),
        "t_p90_cm": None if not dt else round(100 * q(dt, 90), 1),
        "yaw_med_deg": q(dyaw, 50),
        "yaw_p90_deg": q(dyaw, 90),
    }


def speckle(state: np.ndarray) -> float:
    """障害物のセルのうち、まわり 8 セルに障害物が無い（ぽつんと 1 つの）ものの割合。"""
    occ = state == OCCUPIED
    k = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D(occ.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) - occ
    return round(float((occ & (nb == 0)).sum()) / max(1, int(occ.sum())), 3)


def render(grid, poses: dict[int, Pose], title: str, px: int = 4, half_m: float = 3.5):
    state = grid.state()
    n = int(2 * half_m / grid.res)
    c = grid.cell(np.array([-half_m]), np.array([-half_m]))
    i0, j0 = int(c[0][0]), int(c[1][0])
    sub = state[i0 : i0 + n, j0 : j0 + n]
    img = np.full((n, n, 3), 128, np.uint8)
    img[sub == FREE] = (235, 235, 235)
    img[sub == OCCUPIED] = (30, 30, 30)
    img = img.transpose(1, 0, 2)[::-1].copy()  # 行 = y（上が +）、列 = x
    img = cv2.resize(img, (n * px, n * px), interpolation=cv2.INTER_NEAREST)

    def to_px(x, y):
        return int((x + half_m) / grid.res * px), int((half_m - y) / grid.res * px)

    pts = [to_px(poses[i].x, poses[i].y) for i in sorted(poses)]
    for a, b in zip(pts, pts[1:], strict=False):
        cv2.line(img, a, b, (200, 140, 60), 1, cv2.LINE_AA)
    for i in sorted(poses):
        p = poses[i]
        u, v = to_px(p.x, p.y)
        cv2.circle(img, (u, v), 4, (0, 0, 220), -1, cv2.LINE_AA)
        e = to_px(p.x + 0.12 * math.cos(p.yaw), p.y + 0.12 * math.sin(p.yaw))
        cv2.line(img, (u, v), e, (0, 0, 220), 1, cv2.LINE_AA)
    for color, w in (((0, 0, 0), 2), ((255, 255, 255), 1)):
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, w, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--graph", choices=["0", "1"], default="1")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--png", type=Path)
    ap.add_argument("--title", default="")
    ap.add_argument("--roma", action="store_true", help="RoMa v2 の辺も足す（GPU）")
    args = ap.parse_args()
    src = Path(args.session).resolve()
    out = args.out
    if out.resolve() == src or src in out.resolve().parents:
        raise SystemExit("出力先がもとのセッションの中")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    use_graph = args.graph == "1"
    server.POSE_GRAPH = use_graph  # 古い版（POSE_GRAPH が無い）でも害は無い
    server.ROMA_EDGES = args.roma
    from scripts.snoo_replay import recon_reconstruct

    windows: dict[int, list[int]] = {}
    server.choose_window = lambda ids, latest, poses, k=4: windows[latest]
    ids = server.station_ids(src)
    rows = []
    for i in ids:
        d = out / f"{i:03d}"
        d.mkdir()
        shutil.copy2(src / f"{i:03d}" / "meta.json", d / "meta.json")
        if args.roma:
            for f in (src / f"{i:03d}").glob("*.jpg"):
                (d / f.name).symlink_to(f)
        recon = src / f"{i:03d}" / server.RECON_FILE
        if not recon.is_file():
            rows.append({"station": i, "skipped": "recon.npz が無い"})
            continue
        keys, rec = server.load_recon(recon)
        windows[i] = window_of(keys)
        marks: dict = {}
        t = time.perf_counter()
        reply = server.plan_station(out, i, recon_reconstruct(keys, rec), marks)
        marks["total"] = time.perf_counter() - t
        plan = json.loads((out / "plan.json").read_text())
        rows.append({
            "station": i, "reply": reply, "next": plan.get("next"),
            "marks": {k: round(v, 4) for k, v in marks.items()},
            "pose_graph": plan.get("pose_graph"),
        })  # fmt: skip
    poses = server.read_poses(out)
    graph = pg.read_graph_poses(out) if use_graph else {}
    last = ids[-1]
    try:
        obs = server.read_observations(out, last, graph or None)
    except TypeError:  # 古い版の read_observations は poses を受けない
        obs = server.read_observations(out, last)
    grid, _ = session_grid(obs)
    state = grid.state()
    done = [r for r in rows if "reply" in r]

    def count(f):
        return sum(1 for r in done if f(r))

    summary = {
        "graph": use_graph,
        "stations": len(ids),
        "replayed": len(done),
        "with_pose": len(poses),
        "bad_registration": count(
            lambda r: str((r["next"] or {}).get("reason", "")).startswith("bad_registration")
        ),
        "retrace": count(lambda r: r["reply"].get("retrace")),
        "keep_going": count(lambda r: r["reply"].get("keep_going")),
        "keep_going_after_bad": count(
            lambda r: r["reply"].get("keep_going") and r["reply"].get("why") == "bad_registration"
        ),
        "done": count(lambda r: r["reply"].get("action") == "done"),
        "odom": odom_agreement(poses, odom_edges(out, ids)),
        "free_cells": int((state == FREE).sum()),
        "occupied_cells": int((state == OCCUPIED).sum()),
        "speckle": speckle(state),
        "graph_s_med": _q([r["marks"].get("graph") for r in done], 50),
        "graph_s_max": _q([r["marks"].get("graph") for r in done], 100),
        "map_s_med": _q([r["marks"].get("map") for r in done], 50),
        "map_s_max": _q([r["marks"].get("map") for r in done], 100),
        "graph_s_last10_med": _q([r["marks"].get("graph") for r in done[-10:]], 50),
        "roma_s_med": _q([r["marks"].get("roma") for r in done], 50),
        "roma_s_max": _q([r["marks"].get("roma") for r in done], 100),
        "roma_edges": sum(r["marks"].get("roma_edges", 0) for r in done),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows,
                                         "poses": {str(k): server.pose_dict(v)
                                                   for k, v in poses.items()}},
                                        ensure_ascii=False, indent=1))  # fmt: skip
    if args.png:
        title = args.title or ("graph on" if use_graph else "graph off")
        cv2.imwrite(str(args.png), render(grid, poses, title))


def _q(v, p):
    v = [x for x in v if x is not None]
    return round(float(np.percentile(v, p)), 3) if v else None


if __name__ == "__main__":
    main()
