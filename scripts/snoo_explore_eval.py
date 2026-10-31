"""学んだ重みの方策を、学ぶのに使っていない家で、固定の重みと比べる。

    make eval-explore
    .venv/bin/python -m scripts.snoo_explore_eval --policy data/snoo_policy/best.npz --homes 12

scripts/snoo_plan_bench.py と同じ家（SEED0 から）・同じ数（覆った割合・広さ・見失った数・
ぶつかった数・地図の正しさ・決めるのにかかった時間など）で、ふつうの壁と白い壁の 2 通りを
回す。固定の重み（fixed）と学んだ重み（learned）を同じ家で回し、平均と、家ごとの差
（覆った割合）の平均・標準誤差・勝った数を出す。GPU は使わない。CPU は --workers 本（4 本まで）。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from scripts.snoo_plan_bench import FIELDS, SEED0, run_one

DEFAULT_POLICY = Path("data/snoo_policy/best.npz")


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for white in (False, True):
        part = "white" if white else "normal"
        for label in ("fixed", "learned"):
            mine = [r for r in rows if r["white"] == white and r["label"] == label]
            if mine:
                out.setdefault(part, {})[label] = {
                    k: round(float(np.mean([r[k] for r in mine])), 4) for k in FIELDS
                }
        a = {r["home"]: r for r in rows if r["white"] == white and r["label"] == "fixed"}
        b = {r["home"]: r for r in rows if r["white"] == white and r["label"] == "learned"}
        diff = np.array([b[h]["coverage"] - a[h]["coverage"] for h in a if h in b])
        if len(diff):
            out[part]["learned_minus_fixed_coverage"] = {
                "mean": round(float(diff.mean()), 4),
                "sem": round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 4)
                if len(diff) > 1
                else 0.0,
                "wins": int((diff > 0).sum()),
                "losses": int((diff < 0).sum()),
                "homes": len(diff),
            }
    return out


def _one(job: tuple) -> dict:
    label, args = job
    r = run_one(args)
    r["label"] = label
    r.pop("lost_events", None)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--homes", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--minutes", type=float, default=15.0, help="実機の時計で何分まで")
    ap.add_argument("--workers", type=int, default=3, help="同時に回すプロセスの数（4 本まで）")
    ap.add_argument("--snoo", default="random", choices=("random", "behavior", "sneak"))
    ap.add_argument("--out", type=Path, help="結果の JSON")
    a = ap.parse_args()
    if not a.policy.exists():
        print(f"方策のファイルが無い: {a.policy}（make train-explore で作る）", file=sys.stderr)
        return 1
    policy = str(a.policy.resolve())
    jobs = [(label, (a.seed0 + i, white, "new", a.minutes * 60, False, a.snoo,
                     policy if label == "learned" else None))
            for i in range(a.homes) for white in (False, True)
            for label in ("fixed", "learned")]  # fmt: skip
    rows = []
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as pool:
        for r in pool.imap_unordered(_one, jobs):
            rows.append(r)
            keys = ("label", "home", "white", "coverage", "lost", "bumps", "agent_bumps")
            print(json.dumps({k: r[k] for k in keys}), flush=True)
    means = summarize(rows)
    for part, v in means.items():
        print(part, json.dumps(v, ensure_ascii=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        info = {"args": {k: str(v) for k, v in vars(a).items()}, "means": means, "runs": rows}
        a.out.write_text(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
