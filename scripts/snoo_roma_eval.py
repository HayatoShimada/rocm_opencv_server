"""RoMa v2 の辺（app/snoo_roma.py）を位置のグラフに足したときを、保存したセッションで比べる。

    # 1) ジョブを書く（CPU）。続く 2 か所と、戻ってきた組の候補
    .venv/bin/python -m scripts.snoo_roma_eval jobs <セッション> --bake <bakeoff> --out <dir>
    # 2) RoMa を回す（GPU、別の venv）
    ~/orca/roma-venv/bin/python scripts/snoo_roma_match.py --jobs <dir>/jobs.json --out <dir>/match
    # 3) 比べる（CPU）
    .venv/bin/python -m scripts.snoo_roma_eval compare <セッション> --bake <bakeoff> --out <dir>

<bakeoff> は scripts/snoo_pose_bakeoff.py --gpu の --out。その答え
（live_windows.pkl・loops.json・extra/）を使う。
戻ってきた組の VGGT の窓のうち、偶数番目（bakeoff が確かめに使う半分）は RoMa でも使わない
（確かめる組は、どの出し方にも入れない）。

比べるもの:
- pg_live：今の位置のグラフ（窓の辺と走った辺）
- pg_roma_seq：pg_live に、続く 2 か所の RoMa の辺を足したもの
- pg_roma_loop：pg_roma_seq に、戻ってきた組（ORB の候補のうち確かめに使わない半分と、
  pg_live の位置で近い組）の RoMa の辺を足したもの
- pg_full：bakeoff と同じ（VGGT の 2 か所の窓と、戻ってきた組の半分）。RoMa の代わりに VGGT を
  足したときの物差し

RoMa そのものの確からしさとして、続く 2 か所の VGGT の窓（bakeoff の pair_*）が崩れていない組で、
窓の相対の位置と RoMa の辺の差も出す。自分の窓で崩れたステーション（VGGT の失敗）では、RoMa の辺が
出たか・走った辺とどれだけ合うかを出す。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from app import snoo_posegraph as pg
from app import snoo_roma as sr
from app.roommap import Pose
from app.snoo_mapper import load_station
from scripts import snoo_pose_bakeoff as bk

NEAR_M = sr.NEAR_M  # 戻ってきた組の候補：pg_live の位置でこれより近いもの
NEAR_PER_STATION = 1
FRAME_PAIRS = sr.FRAME_PAIRS


def still_frames(session: Path, i: int) -> list[dict]:
    st = load_station(session / f"{i:03d}", "meta.json")
    return [f for f in st.frames if not f.get("moving")]


def heading_prior(a: int, b: int, prior: dict[int, Pose], odom: dict[tuple[int, int], pg.Edge]):
    """b の車の向き − a の車の向き の見込み（°）。走った辺があればそれ、無ければ pg_live の位置。"""
    if (a, b) in odom:
        return math.degrees(odom[(a, b)].z.yaw)
    if a in prior and b in prior:
        return math.degrees(pg.relative(prior[a], prior[b]).yaw)
    return 0.0


def make_jobs(session: Path, bake: Path) -> tuple[list[dict], dict]:
    ids = bk.station_list(session)
    odom = {(e.a, e.b): e for e in bk.odom_edges(session, ids)}
    live = bk.live_windows(session, bake / "live_windows.pkl")
    sol = pg.optimize([e for w in live for e in pg.window_edges(w)] + list(odom.values()), ids[0])
    prior = sol.poses
    cand = bk.loop_candidates(session, ids, bake / "loops.json")
    held = {(a, b) for k, (a, b, _) in enumerate(cand) if k % 2 == 0}  # bakeoff が確かめる半分
    orb_use = [(a, b) for k, (a, b, _) in enumerate(cand) if k % 2 == 1]
    # pg_live の位置で近い組（番号の離れたもの）。1 か所につき近い順に NEAR_PER_STATION 組
    near = []
    for b in ids:
        best = sorted(
            (math.hypot(prior[a].x - prior[b].x, prior[a].y - prior[b].y), a)
            for a in ids
            if a <= b - bk.LOOP_MIN_GAP and a in prior and b in prior
        )
        for d, a in best[:NEAR_PER_STATION]:
            if d < NEAR_M and (a, b) not in held and (a, b) not in orb_use:
                near.append((a, b))
    seq = [(a, b) for a, b in zip(ids, ids[1:], strict=False) if b == a + 1]
    frames = {i: still_frames(session, i) for i in ids}
    jobs = []
    for kind, pairs in (("seq", seq), ("orb", orb_use), ("near", near)):
        for a, b in pairs:
            dyaw = heading_prior(a, b, prior, odom)
            fp = sr.pick_frame_pairs(frames[a], frames[b], dyaw, FRAME_PAIRS)
            jobs.append({
                "name": f"{kind}_{a:03d}_{b:03d}", "kind": kind, "a": a, "b": b,
                "pairs": [[str(session / f"{a:03d}" / fa["file"]),
                           str(session / f"{b:03d}" / fb["file"])] for fa, fb in fp],
                "pans": [[fa["pan_deg"], fa["tilt_deg"], fb["pan_deg"], fb["tilt_deg"]]
                         for fa, fb in fp],
            })  # fmt: skip
    info = {"seq": len(seq), "orb": len(orb_use), "near": len(near), "held": sorted(held)}
    return jobs, info


def job_estimate(job: dict, d: dict):
    """1 ジョブの RoMa の対応から、ステーションの組の答えと、コマの組ごとの答え
    （地図のサーバーと同じ app/snoo_roma.estimate）。"""
    n = int(d["n"])
    matches = [{"ka": d[f"ka_{k}"], "kb": d[f"kb_{k}"], "cert": d[f"cert_{k}"]} for k in range(n)]
    return sr.estimate(matches, job["pans"][:n], job["kind"] != "seq")


def roma_edges(jobs: list[dict], match: Path):
    edges, rows = [], []
    for job in jobs:
        p = match / f"{job['name']}.npz"
        if not p.is_file():
            continue
        d = dict(np.load(p))
        t = time.perf_counter()
        st, est = job_estimate(job, d)
        cpu = time.perf_counter() - t
        row = {"name": job["name"], "kind": job["kind"], "a": job["a"], "b": job["b"],
               "gpu_s": float(d["seconds"]), "cpu_s": cpu, "pairs": len(est),
               "ok_pairs": sum(e is not None for e in est),
               "overlap": [float(d[f"overlap_{k}"]) for k in range(int(d["n"]))]}  # fmt: skip
        if st is not None:
            row.update(z=[st.pose.x, st.pose.y, math.degrees(st.pose.yaw)], sigma_m=st.sigma_m,
                       sigma_deg=math.degrees(st.sigma_rad), inliers=st.inliers,
                       rotation_only=st.rotation_only)  # fmt: skip
            edges.append(sr.to_edge(job["a"], job["b"], st, f"roma:{job['name']}"))
        rows.append(row)
    return edges, rows


def _q(v, q):
    return None if not len(v) else round(float(np.percentile(v, q)), 2)


def compare_to(edges: list[pg.Edge], ref: dict[tuple[int, int], Pose]) -> dict:
    dt, dy = [], []
    for e in edges:
        z = ref.get((e.a, e.b))
        if z is None:
            continue
        dt.append(100 * math.hypot(e.z.x - z.x, e.z.y - z.y))
        dy.append(abs(math.degrees(pg.wrap(e.z.yaw - z.yaw))))
    return {"n": len(dt), "t_med_cm": _q(dt, 50), "t_p90_cm": _q(dt, 90),
            "yaw_med_deg": _q(dy, 50), "yaw_p90_deg": _q(dy, 90)}  # fmt: skip


def evaluate(session: Path, bake: Path, out: Path, jobs: list[dict]) -> dict:
    ids = bk.station_list(session)
    odom = bk.odom_edges(session, ids)
    live = bk.live_windows(session, bake / "live_windows.pkl")
    cand = bk.loop_candidates(session, ids, bake / "loops.json")
    extra_jobs = [(f"pair_{a:03d}_{b:03d}_b", a, b, True)
                  for a, b in zip(ids, ids[1:], strict=False) if b == a + 1]  # fmt: skip
    extra_jobs += [(f"loop_{a:03d}_{b:03d}", a, b, False) for a, b, _ in cand]
    extra = [bk.extra_window(session, *j, bake / "extra") for j in extra_jobs
             if (bake / "extra" / f"{j[0]}.npz").is_file()]  # fmt: skip
    pairs = [w for w in extra if w.source.startswith("pair_")]
    loop_wins = [w for w in extra if w.source.startswith("loop_")]
    loop_use, loop_eval = [], []
    for k, w in enumerate(loop_wins):
        es = pg.window_edges(w)
        for e in es:
            e.kind = "loop"
        (loop_use if k % 2 else loop_eval).extend(es)
    live_edges = [e for w in live for e in pg.window_edges(w)]
    pair_edges = [e for w in pairs for e in pg.window_edges(w)]

    t = time.perf_counter()
    redges, rows = roma_edges(jobs, out / "match")
    roma_cpu = time.perf_counter() - t
    held = {(e.a, e.b) for e in loop_eval}
    redges = [e for e in redges if (e.a, e.b) not in held]  # 念のため（ジョブでも外している）
    seq = [e for e in redges if e.source.startswith("roma:seq_")]
    loops = [e for e in redges if not e.source.startswith("roma:seq_")]

    fixed = ids[0]
    methods = {}

    def run(name, edges, wins):
        t = time.perf_counter()
        sol = pg.optimize([replace(e) for e in edges], fixed)
        sec = time.perf_counter() - t
        sp = bk.best_spread(wins, sol)
        out_by = {}
        for e in sol.edges:
            if e.outlier:
                out_by[e.kind] = out_by.get(e.kind, 0) + 1
        methods[name] = (sol, sp, {"solve_s": round(sec, 2), "edges": len(sol.edges),
                                   "outliers": out_by, "seen": len(sol.seen)})  # fmt: skip

    run("pg_live", live_edges + odom, live)
    run("pg_roma_seq", live_edges + seq + odom, live)
    run("pg_roma_loop", live_edges + seq + loops + odom, live)
    run("pg_roma_only", seq + loops + odom, [])
    run("pg_full", live_edges + pair_edges + loop_use + odom, live + pairs + loop_wins)
    run("pg_full_roma", live_edges + pair_edges + loop_use + seq + loops + odom,
        live + pairs + loop_wins)  # fmt: skip

    clouds = bk.best_clouds(live + pairs + loop_wins)
    agree_pairs = [(a, b) for a, b in zip(ids, ids[1:], strict=False) if b == a + 1]
    agree_pairs += [(a, b) for a, b, _ in cand]
    table = {}
    for name, (sol, spread, info) in methods.items():
        poses = sol.poses
        m = bk.metrics(poses, spread, ids, odom, loop_eval)
        # 「位置がまとまっている」：見た辺（外れ値でない窓・RoMa・見直し）があるステーションの割合
        m["seen_frac"] = round(len(sol.seen) / len(ids), 3)
        obs = [(i, pg.observation(clouds[i], poses[i])) for i in ids if i in poses and i in clouds]
        mm, _ = bk.map_metrics(obs, poses, agree_pairs)
        table[name] = {**m, **mm, "info": info}

    # RoMa の辺そのもの
    win_ref = {}
    for w in pairs:  # VGGT の 2 か所の窓が崩れていない組
        es = pg.window_edges(w)
        for e in es:
            win_ref[(e.a, e.b)] = e.z
    loop_ref = {(e.a, e.b): e.z for e in loop_use + loop_eval}
    odom_ref = {(e.a, e.b): e.z for e in odom}
    # VGGT の失敗：自分の窓で崩れたステーションと、地図のサーバーが位置を出せなかった
    # （poses.json に無い）ステーション
    failed = {w.ids[-1] for w in live if not w.good(len(w.ids) - 1)}
    failed = sorted(failed | (set(ids) - set(bk.read_poses(session))))
    seq_failed = [e for e in seq if e.b in failed or e.a in failed]
    fail_rows = []
    for i in failed:
        r = [x for x in rows if x["kind"] == "seq" and i in (x["a"], x["b"])]
        fail_rows.append({"station": i, "edges": [
            {k: x.get(k) for k in ("name", "ok_pairs", "pairs", "inliers", "z", "sigma_m",
                                   "sigma_deg", "rotation_only")} for x in r]})  # fmt: skip
    pg_live_poses = methods["pg_live"][0].poses
    pg_full_poses = methods["pg_full"][0].poses
    rel = {}
    for name, P in (("pg_live", pg_live_poses), ("pg_full", pg_full_poses)):
        rel[name] = {(e.a, e.b): pg.relative(P[e.a], P[e.b]) for e in redges
                     if e.a in P and e.b in P}  # fmt: skip
    quality = {
        "seq_jobs": sum(r["kind"] == "seq" for r in rows),
        "seq_edges": len(seq),
        "loop_jobs": sum(r["kind"] != "seq" for r in rows),
        "loop_edges": len(loops),
        "seq_vs_vggt_pair": compare_to(seq, win_ref),
        "seq_vs_odom": compare_to(seq, odom_ref),
        "loop_vs_vggt_loop": compare_to(loops, loop_ref),
        "seq_vs_pg_full": compare_to(seq, rel["pg_full"]),
        "loops_vs_pg_full": compare_to(loops, rel["pg_full"]),
        "failed_stations": failed,
        "failed_seq_jobs": sum(
            1 for r in rows if r["kind"] == "seq" and (r["a"] in failed or r["b"] in failed)
        ),  # fmt: skip
        "failed_seq_edges": len(seq_failed),
        "failed_vs_odom": compare_to(seq_failed, odom_ref),
        "failed_vs_vggt_pair": compare_to(seq_failed, win_ref),
        "failed_vs_pg_full": compare_to(seq_failed, rel["pg_full"]),
        "gpu_s_per_job_med": _q([r["gpu_s"] for r in rows], 50),
        "gpu_s_per_job_p90": _q([r["gpu_s"] for r in rows], 90),
        "cpu_s_per_job_med": _q([r["cpu_s"] for r in rows], 50),
        "roma_cpu_total_s": round(roma_cpu, 1),
    }
    out_roma = {}
    for name, (sol, _, _) in methods.items():
        out_roma[name] = [{"a": e.a, "b": e.b, "source": e.source, "chi": round(e.chi, 1)}
                          for e in sol.edges if e.kind == "roma" and e.outlier]  # fmt: skip
    (out / "roma_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "roma_outliers.json").write_text(json.dumps(out_roma, indent=1))
    (out / "failed.json").write_text(json.dumps(fail_rows, indent=1))
    res = {"table": table, "roma": quality}
    (out / "table.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


COLS = [("with_pose", "pose"), ("consistent", "cons<5cm"), ("seen_frac", "seen")] + bk.COLS[2:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["jobs", "compare"])
    ap.add_argument("session")
    ap.add_argument("--bake", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    session, bake, out = Path(args.session).resolve(), Path(args.bake), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.what == "jobs":
        jobs, info = make_jobs(session, bake)
        (out / "jobs.json").write_text(json.dumps(jobs, indent=1))
        print(f"ジョブ {len(jobs)}：{info}")
        return
    jobs = json.loads((out / "jobs.json").read_text())
    res = evaluate(session, bake, out, jobs)
    print("| method | " + " | ".join(h for _, h in COLS) + " |")
    print("|---" * (len(COLS) + 1) + "|")
    for name, m in res["table"].items():
        print(f"| {name} | " + " | ".join(str(m.get(k)) for k, _ in COLS) + " |")
    for name, m in res["table"].items():
        print(name, m["info"])
    print(json.dumps(res["roma"], indent=1))


if __name__ == "__main__":
    main()
