"""実走のセッションを、今の決め方（app/snoo_mapper.decide）で決め直して、返した 1 歩と比べる。

    .venv/bin/python -m scripts.snoo_plan_replay data/snoo_map/sessions/explore_20261031_075053 \\
        --stations 4-10 24-32 --out out/replay/075053.json

VGGT は回さない（GPU を使わない）。置き場の obs.npz（見たもの）・poses.json（位置）・
meta.json（走った
1 歩）から、そのステーションまでの間取りを作り直し、Pi が gear_segments を受けるとして決める。
ステーションごとに、写真の手ざわり（app/snoo_belief.image_texture_score）・位置のグラフの辺の数
（window・roma・odom）・位置合わせが崩れたか（graph_poses.json の bad）も出す。白い壁ばかりの所で
位置を外したかを見るため。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import app.snoo_map_server as server
from app.roommap import camera_of, drive_moves
from app.snoo_belief import image_texture_score
from app.snoo_mapper import MapParams, StationResult, decide, session_grid


def ranges(texts: list[str]) -> list[int]:
    out = []
    for t in texts:
        a, _, b = t.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def texture_of(d: Path) -> dict:
    """止まって撮ったコマの手ざわりの平均。"""
    meta = server.read_meta(d)
    rows = [
        image_texture_score(d / f["file"]) for f in meta.get("frames", []) if not f.get("moving")
    ]
    if not rows:
        return {}
    return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}


def edges_of(session: Path) -> dict[int, dict]:
    """ステーションごとの、位置のグラフの辺の数（そのステーションにつながるもの）。"""
    out: dict[int, dict] = {}
    p = session / server.GRAPH_EDGES
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        for i in {e["a"], e["b"]}:
            c = out.setdefault(i, {"window": 0, "roma": 0, "odom": 0})
            c[e["kind"]] = c.get(e["kind"], 0) + 1
    return out


def moves_text(a: dict) -> str:
    if a.get("action") != "move":
        return f"done {a.get('reason')}"
    if a.get("rescan"):
        return "その場で撮り直す（rescan）"
    segs = a.get("segments") or [a]
    return " → ".join(
        f"{'後' if x.get('reverse', a.get('reverse')) else '前'}{float(x['steer_deg']):+.0f}°"
        f" {float(x['distance_cm']):.0f}cm"
        for x in segs
    )


def replay(session: Path, sid: int) -> dict:
    poses = server.read_poses(session)
    d = server.station_dir(session, sid)
    obs = server.read_observations(session, sid, poses)
    grid, _ = session_grid(obs)
    server.mark_steps_driven(grid, session, sid, poses)
    pose = poses[sid]
    st = [StationResult(f"{sid:03d}", pose, 0.08, 0.0, 10)]
    can = server.supports(d) | {"gear_segments"}
    params = MapParams(
        segments="segments" in can, reverse="reverse" in can, gear_segments=True,
        max_step_m=0.6 if {"long_step", "segments"} <= can else 0.4,
        max_stations=10**6, seed=sid,
    )  # fmt: skip
    recent = server.recent_poses(session, sid, poses)
    action, man = decide(grid, st, params, [0.08], recent=recent)
    old = json.loads((d / "reply.json").read_text()) if (d / "reply.json").is_file() else {}
    end = None
    if man is not None:
        end = camera_of(drive_moves(pose, man.moves)[-1][1])
    return {
        "station": sid,
        "pose": server.pose_dict(pose),
        "old": moves_text(old),
        "old_flags": {
            k: old[k] for k in ("keep_going", "blind", "why", "turn", "random") if k in old
        },
        "new": moves_text(action),
        "new_planner": action.get("planner"),
        "new_gap": bool(action.get("gap")),
        "new_stuck": bool(action.get("stuck")),
        "new_end": server.pose_dict(end) if end else None,
        "new_moved_m": round(math.hypot(end.x - pose.x, end.y - pose.y), 3) if end else 0.0,
        "info": man.info if man is not None else None,
        "action": action,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("--stations", nargs="+", default=None, help="例：4-10 24-32")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    ids = ranges(a.stations) if a.stations else server.station_ids(a.session)
    graph = json.loads((a.session / "graph_poses.json").read_text()) if (
        a.session / "graph_poses.json").is_file() else {}  # fmt: skip
    bad = set(graph.get("bad", []))
    edges = edges_of(a.session)
    rows = []
    for sid in server.station_ids(a.session):
        tex = texture_of(server.station_dir(a.session, sid))
        row = {"station": sid, "texture": tex, "edges": edges.get(sid, {}), "bad": sid in bad}
        if sid in ids:
            row |= replay(a.session, sid)
            print(f"{sid:03d} 前: {row['old']} {row['old_flags'] or ''}")
            print(
                f"    新: {row['new']}（{row['new_planner']}、隙間 {row['new_gap']}、"
                f"動けない {row['new_stuck']}、{row['new_moved_m']} m 先へ）{row['info']}"
            )
        print(f"    手ざわり {tex}  辺 {row['edges']}  崩れた {row['bad']}")
        rows.append(row)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
