"""次の 1 歩の決め方を、作った家と速いシミュレーター（app/snoo_gridsim.py）で測る。

    .venv/bin/python -m scripts.snoo_plan_bench --homes 12 --workers 3 --out out/bench/plan.json
    # 前の版と同じ家・同じくじで比べる（前の版のチェックアウトを PYTHONPATH に）
    PYTHONPATH=<前の版> .venv/bin/python <この版>/scripts/snoo_plan_bench.py --homes 12 --label main

GPU は使わない。家ごとに、ふつうの壁と白い壁（手ざわりの無い壁・白い家電）の 2 通りで回す。
出すのは家ごと・平均の次の数。
- coverage：行ける床のうち、床と分かった割合。m2：その広さ。t_s：止まった時刻（実機の時計）
- stations：撮った所の数。unreachable・stuck：決め方が「行ける所が無い」「動けない」と返した所の数
  （前の版は、その後で下がる 1 歩に替えることがある。数えるのは替える前。今の版の
  「その場で撮り直す」
  （rescan）も unreachable に数える）
- bumps：家具・壁にぶつかった数（--low-sofa なら、くぐれない低いソファの下に入ったのも）。
  agent_bumps：人・snoo
- gaps：両脇に車幅 + 6 cm より狭い所を通り抜けた 1 歩の数（本当の家の格子で数える）
- lost：位置を見失った所の数（撮った所から見えた目印（手ざわりの高い障害物）が少ない。見たものは
  地図に足さない。地図のサーバーの bad_registration と同じ）
- map_acc：見た所のうち、床・障害物が本当の家と合っている割合。wall_err_cm：障害物と見たセルから
  本当の障害物までの平均の距離
--planner hybrid は MPPI を使わず Hybrid A* だけにする（比べるため。地図のサーバーは使わない）。
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

# 白い壁・家具の手ざわり（app/snoo_homes.home_texture と同じ。前の版には無いのでここにも置く）
TEX_WALL, TEX_WHITE, TEX_FURNITURE, LANDMARK_TEX = 0.45, 0.05, 0.8, 0.35
WHITE_ITEMS = ("kitchen", "fridge", "wardrobe", "chest")
LOST_MIN = 6
NARROW_M = 0.06
SEED0 = 910_000


def texture(home, white: bool) -> np.ndarray:
    from app.snoo_homes import CLOSED, FLOOR, HIGH, LOW

    tex = np.zeros(home.kind.shape, np.float32)
    tex[home.kind == CLOSED] = TEX_WHITE if white else TEX_WALL
    tex[(home.kind == LOW) | (home.kind == HIGH)] = TEX_FURNITURE
    if white:
        for it in home.items:
            if it.name in WHITE_ITEMS:
                x0, y0, x1, y1 = it.rect
                ix0, iy0 = home.cell(x0, y0)
                ix1, iy1 = home.cell(x1 - 1e-6, y1 - 1e-6)
                sl = (slice(max(int(ix0), 0), int(ix1) + 1), slice(max(int(iy0), 0), int(iy1) + 1))
                tex[sl] = np.where(home.kind[sl] != FLOOR, TEX_WHITE, tex[sl])
    return tex


def solid_cells(sim, pose, t) -> np.ndarray:
    """見えた壁・家具のセル（app/snoo_gridsim.GridSim.observe の solid と同じ。前の版には無い）。"""
    from app.snoo_homes import CLOSED, LOW

    h = sim.home
    a = pose.yaw + sim._da[:, None]
    d = sim._d[None, :]
    ix, iy = h.cell(pose.x + d * np.cos(a), pose.y + d * np.sin(a))
    kind, height = sim._layers(t, ix, iy)
    low = kind == LOW
    full = (kind == CLOSED) | (low & (height >= 0.08))
    open_ = (np.cumsum(full, axis=1) - full) == 0
    m = (full | low) & open_
    cells = np.unique(np.stack([ix[m], iy[m]], 1), axis=0)
    ok = (cells >= 0).all(1) & (cells < h.n).all(1)
    return cells[ok]


def run_one(args: tuple) -> dict:
    home_seed, white, planner, budget_s, low_sofa = (*args, False)[:5]
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    import cv2

    cv2.setNumThreads(1)
    import app.snoo_gridsim as gs
    from app.roommap import CAR_CENTER_M, FREE, OCCUPIED, UNKNOWN, arc, rear_axle
    from app.snoo_homes import make_home

    if planner == "hybrid":
        import functools

        import app.snoo_mapper as mapper

        mapper.plan_maneuver = functools.partial(mapper.plan_maneuver, mode="hybrid")
    sofa = {"low_sofa": True} if low_sofa else {}
    home = make_home(home_seed, agent_t_s=budget_s + 120, **sofa)
    tex = texture(home, white)
    kw = {"texture": tex} if "texture" in inspect.signature(gs.GridSim.__init__).parameters else {}
    sim = gs.GridSim(home, **kw)
    counts = {"unreachable": 0, "stuck": 0, "gaps": 0, "lost": 0, "plan_ms": []}
    if not kw:  # 前の版：見失うのはここでまねる
        real_observe = sim.observe

        def observe(pose, t):
            free, occ = real_observe(pose, t)
            solid = solid_cells(sim, pose, t)
            seen = tex[solid[:, 0], solid[:, 1]] if len(solid) else np.zeros(0)
            if sim.stations and int((seen >= LANDMARK_TEX).sum()) < LOST_MIN:
                counts["lost"] += 1
                return free[:0], occ[:0]
            return free, occ

        sim.observe = observe
    real_decide = gs.decide

    def decide(*a, **k):
        t0 = time.perf_counter()
        out = real_decide(*a, **k)
        counts["plan_ms"].append(1000 * (time.perf_counter() - t0))
        reason = out[0].get("reason") if out[0].get("action") == "done" else out[0].get("why")
        if reason in ("unreachable", "stuck"):
            counts[reason] += 1
        return out

    gs.decide = decide
    block = sim.car_block
    hw = 0.095 + NARROW_M

    def narrow(x, y, yaw) -> bool:
        c, s = math.cos(yaw), math.sin(yaw)
        for ahead in (CAR_CENTER_M - 0.08, CAR_CENTER_M, CAR_CENTER_M + 0.08):
            mx, my = x + ahead * c, y + ahead * s
            sides = []
            for side in (1, -1):
                ix, iy = home.cell(mx - side * hw * s, my + side * hw * c)
                ok = 0 <= ix < home.n and 0 <= iy < home.n
                sides.append((not ok) or bool(block[ix, iy]))
            if all(sides):
                return True
        return False

    real_drive = sim.drive

    def drive(steer, speed, distance_m, reverse, *a, **k):
        r = rear_axle(sim.pose)
        result, moved = real_drive(steer, speed, distance_m, reverse, *a, **k)
        x, y, yaw = r.x, r.y, r.yaw
        hit = False
        for _ in range(int(moved / 0.02)):
            x, y, yaw = arc(x, y, yaw, max(-30.0, min(30.0, steer)), (-0.02 if reverse else 0.02))
            hit |= narrow(x, y, yaw)
        counts["gaps"] += hit
        return result, moved

    sim.drive = drive
    t0 = time.perf_counter()
    ep = sim.run(budget_s)
    cpu = time.perf_counter() - t0
    # 地図の正しさ（見た所だけ）
    st = sim.grid.state()
    known = st != UNKNOWN
    truth_occ = block
    match = ((st == OCCUPIED) & truth_occ) | ((st == FREE) & ~truth_occ)
    acc = float(match[known].mean()) if known.any() else 0.0
    import cv2

    edt = cv2.distanceTransform(np.where(truth_occ, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    occ_cells = st == OCCUPIED
    wall_err = float(edt[occ_cells].mean() * home.res * 100) if occ_cells.any() else 0.0
    lost = getattr(ep, "lost", None)
    if not kw:
        lost = counts["lost"]
    ms = counts["plan_ms"]
    return {
        "home": home_seed, "white": white, "planner": planner, "coverage": ep.coverage,
        "m2": ep.covered_m2, "t_s": ep.t_s, "stations": ep.cycles, "reason": ep.reason,
        "unreachable": counts["unreachable"], "stuck": counts["stuck"], "bumps": ep.bumps,
        "agent_bumps": ep.agent_bumps, "gaps": counts["gaps"], "lost": lost,
        "map_acc": round(acc, 4), "wall_err_cm": round(wall_err, 2), "distance_m": ep.distance_m,
        "kinds": ep.kinds, "plan_ms_med": round(float(np.median(ms)), 1) if ms else 0.0,
        "plan_ms_max": round(float(np.max(ms)), 1) if ms else 0.0, "cpu_s": round(cpu, 1),
    }  # fmt: skip


FIELDS = ("coverage", "m2", "t_s", "stations", "unreachable", "stuck", "bumps", "agent_bumps",
          "gaps", "lost", "map_acc", "wall_err_cm", "plan_ms_med", "plan_ms_max")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--homes", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--minutes", type=float, default=15.0, help="実機の時計で何分まで")
    ap.add_argument("--workers", type=int, default=3, help="同時に回すプロセスの数（4 本まで）")
    ap.add_argument("--planner", default="new", choices=("new", "hybrid"))
    ap.add_argument("--low-sofa", action="store_true",
                    help="下の端が 10〜18 cm の低いソファを 1 つ足した家で回す")  # fmt: skip
    ap.add_argument("--label", default=None, help="結果の名前（既定は --planner）")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    label = a.label or a.planner
    jobs = [(a.seed0 + i, white, a.planner, a.minutes * 60, a.low_sofa)
            for i in range(a.homes) for white in (False, True)]  # fmt: skip
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as pool:
        rows = []
        for r in pool.imap_unordered(run_one, jobs):
            r["label"] = label
            rows.append(r)
            print(json.dumps({k: r[k] for k in ("home", "white", *FIELDS, "reason")}), flush=True)
    means = {}
    for white in (False, True):
        mine = [r for r in rows if r["white"] == white]
        means["white" if white else "normal"] = {
            k: round(float(np.mean([r[k] for r in mine])), 4) for k in FIELDS
        }
    print(label, json.dumps(means, ensure_ascii=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        info = {"label": label, "args": vars(a) | {"out": str(a.out)}, "means": means, "runs": rows}
        a.out.write_text(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
