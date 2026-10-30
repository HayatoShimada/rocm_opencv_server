"""学んだ選び方を、学ぶのに使っていない家で、今の決め方（decide）と比べる。

    make eval-explore
    .venv/bin/python -m scripts.snoo_explore_eval --policy data/snoo_policy/best.npz --homes 48

家のくじは scripts/snoo_explore_train.TEST_SEED0 から（学ぶ家・確かめる家とは別）。
速いシミュレーター（app/snoo_gridsim.py）で、同じ家・同じ人と snoo の動きで回し、
5・10・20 分の時点で分かった床（m²）、行ける床のうちの割合、ぶつかった数を出す。
--plot で時間と広さの線の絵を、--snapshots で何軒かの最後の地図を書く。
GPU は使わない。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np

from scripts.snoo_explore_train import TEST_SEED0

MODES = ("decide", "heuristic", "policy")
MARKS_S = (300, 600, 1200)


def _init() -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    cv2.setNumThreads(1)


def _one(args: tuple) -> dict:
    from app.snoo_explore_rollout import run_episode

    seed, mode, weights, budget_s, snap = args
    r = run_episode(seed, mode, weights, budget_s=budget_s, keep_sim=bool(snap))
    e = r["episode"]
    row = {
        "seed": seed, "mode": mode, "reachable_m2": e.reachable_m2, "covered_m2": e.covered_m2,
        "coverage": e.coverage, "bumps": e.bumps, "agent_bumps": e.agent_bumps, "stops": e.stops,
        "distance_m": e.distance_m, "cycles": e.cycles, "reason": e.reason, "kinds": e.kinds,
        "timeline": e.timeline, **{f"m2_{t // 60}min": round(e.m2_at(t), 3) for t in MARKS_S},
    }  # fmt: skip
    if snap:
        from app.snoo_homes import render_home

        sim = r["sim"]
        img = render_home(sim.home, known=sim.grid.state(), t_s=sim.t)
        px = 4
        pts = [
            (int((p.x - sim.home.origin) / sim.home.res * px),
             int((sim.home.n - (p.y - sim.home.origin) / sim.home.res) * px))
            for p in sim.trail
        ]  # fmt: skip
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 0, 255), 2)
        cv2.putText(img, f"{mode} seed {seed}: {e.covered_m2:.1f} m2 ({100 * e.coverage:.0f} %)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # fmt: skip
        Path(snap).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(snap) / f"run_{seed}_{mode}.png"), img)
    return row


def summarize(rows: list[dict]) -> dict:
    out = {}
    for mode in MODES:
        mine = [r for r in rows if r["mode"] == mode]
        if not mine:
            continue
        keys = [f"m2_{t // 60}min" for t in MARKS_S] + ["coverage", "distance_m"]
        out[mode] = {k: round(float(np.mean([r[k] for r in mine])), 3) for k in keys}
        out[mode] |= {k: int(sum(r[k] for r in mine)) for k in ("bumps", "agent_bumps", "stops")}
        out[mode]["homes"] = len(mine)
    if "decide" in out and "policy" in out:
        a = {r["seed"]: r["m2_20min"] for r in rows if r["mode"] == "decide"}
        b = {r["seed"]: r["m2_20min"] for r in rows if r["mode"] == "policy"}
        diff = np.array([b[s] - a[s] for s in a if s in b])
        out["policy_minus_decide_20min"] = {
            "mean": round(float(diff.mean()), 3),
            # 家ごとの差の平均の標準誤差（家による違いが大きいので、対で比べる）
            "sem": round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 3) if len(diff) > 1 else 0,
            "wins": int((diff > 0).sum()),
            "homes": len(diff),
        }
    return out


def plot(rows: list[dict], path: Path, budget_s: float) -> None:
    """時間と、分かった床（家の平均）の線。cv2 で描く（matplotlib は入れていない）。"""
    W, H, m = 800, 500, 60
    img = np.full((H, W, 3), 255, np.uint8)
    ts = np.arange(0, budget_s + 1, 10.0)
    colors = {"decide": (60, 60, 200), "heuristic": (60, 160, 60), "policy": (200, 100, 30)}
    curves = {}
    for mode in MODES:
        mine = [r for r in rows if r["mode"] == mode]
        if not mine:
            continue
        ys = []
        for r in mine:
            tl = r["timeline"]
            ys.append([max([c for t, c, _ in tl if t <= x] or [0.0]) for x in ts])
        curves[mode] = np.mean(ys, axis=0)
    ymax = max(float(c.max()) for c in curves.values()) * 1.1 or 1.0

    def P(t, y):
        return int(m + (W - 2 * m) * t / budget_s), int(H - m - (H - 2 * m) * y / ymax)

    cv2.line(img, P(0, 0), P(budget_s, 0), (0, 0, 0), 1)
    cv2.line(img, P(0, 0), P(0, ymax), (0, 0, 0), 1)
    for minute in range(0, int(budget_s // 60) + 1, 5):
        x, y = P(minute * 60, 0)
        cv2.putText(img, f"{minute}", (x - 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for k in range(0, int(ymax) + 1, 5):
        x, y = P(0, k)
        cv2.putText(img, f"{k}", (x - 35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for i, (mode, c) in enumerate(curves.items()):
        pts = np.array([P(t, y) for t, y in zip(ts, c, strict=True)], np.int32)
        cv2.polylines(img, [pts], False, colors[mode], 2, cv2.LINE_AA)
        cv2.putText(img, f"{mode} {c[-1]:.1f} m2", (m + 20, m + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[mode], 2)  # fmt: skip
    cv2.putText(img, "minutes", (W // 2, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, default=Path("data/snoo_policy/best.npz"))
    parser.add_argument("--homes", type=int, default=48)
    parser.add_argument("--seed0", type=int, default=TEST_SEED0)
    parser.add_argument("--budget-s", type=float, default=1200.0)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", type=Path, help="結果の JSON")
    parser.add_argument("--plot", type=Path, help="時間と広さの線の絵（PNG）")
    parser.add_argument("--snapshots", type=Path, help="最初の 3 軒の最後の地図を書く置き場")
    args = parser.parse_args()

    weights = None
    if "policy" in args.modes:
        with np.load(args.policy) as d:
            weights = dict(d)
    jobs = []
    for k in range(args.homes):
        seed = args.seed0 + k
        for mode in args.modes:
            snap = str(args.snapshots) if args.snapshots and k < 3 else ""
            jobs.append((seed, mode, weights if mode == "policy" else None, args.budget_s, snap))
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    means = summarize(rows)
    for k, v in means.items():
        print(k, json.dumps(v))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        info = {"args": {k: str(v) for k, v in vars(args).items()}, "means": means, "runs": rows}
        args.out.write_text(json.dumps(info, ensure_ascii=False, indent=1))
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(rows, args.plot, args.budget_s)


if __name__ == "__main__":
    main()
