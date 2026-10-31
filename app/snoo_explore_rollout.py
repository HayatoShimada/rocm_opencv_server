"""家（app/snoo_homes.py）を 1 つ作り、速いシミュレーター（app/snoo_gridsim.py）で 1 回探る。
計画器の重みを方策（app/snoo_maneuver_policy.py）で決め、学ぶための周ごとの記録と点数を返す。

学ぶ（scripts/snoo_explore_train.py）ときに使う。numpy だけで動く（子のプロセスに torch を
読ませない。GPU も使わない）。

- 家の手ざわりは app/snoo_homes.home_texture（white なら白い壁・白い家電）。目印が少ない所で
  撮ると位置を見失い、見たものを地図に足さない（scripts/snoo_plan_bench.py と同じ）
- 決め方は地図のサーバーと同じ app/snoo_mapper.decide。重みだけを外から渡す
  （policy が None なら固定の重み）。sigma > 0 なら z に正規分布のゆらぎを足す（PPO が試す）
- 点数（score）：覆った割合を早く増やすほど良い（増えた割合 × (1 - 0.5 × 時刻 / 上限) × K_COV）。
  地図の間違い（本当の家と違うセル m²）・見失った数・ぶつかった数（人と snoo は重く）を引く。
  周ごとの報酬（reward）は、その周の 1 歩から次の周までに変わった分で、足すと score になる
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from app.roommap import FREE, OCCUPIED, UNKNOWN
from app.snoo_belief import confidence_of
from app.snoo_gridsim import CAM_H_M, GridSim
from app.snoo_homes import home_texture, make_home
from app.snoo_maneuver_policy import WeightPolicy, features, to_weights
from app.snoo_mapper import RECENT_STATIONS, decide

# 点数の重み
K_COV = 20.0  # 覆った割合 1 あたり（早いほど重い。上限の時刻では半分）
K_WRONG = 10.0  # 地図の間違い 1 m² あたり
K_LOST = 0.5  # 見失った 1 か所あたり（見たものを足さないので、覆う分でも損をしている）
K_BUMP = 2.0  # 家具・壁にぶつかった 1 回あたり
K_AGENT = 4.0  # 人・snoo にぶつかった 1 回あたり


@dataclass
class Step:
    f: np.ndarray  # 特徴
    z: np.ndarray  # 使った z（ゆらぎを足した後、±Z_MAX に収める前）
    t: float
    reward: float = 0.0


def wrong_m2(sim: GridSim) -> float:
    """見たセルのうち、本当の家と違う（床と見たのに障害物・障害物と見たのに床）広さ。"""
    st = sim.grid.state()
    block = sim.car_block
    bad = ((st == OCCUPIED) & ~block) | ((st == FREE) & block)
    return float((bad & (st != UNKNOWN)).sum() * sim.home.res**2)


def _snapshot(sim: GridSim) -> tuple:
    return (sim.t, sim.covered_m2() / max(sim.reach_m2, 1e-9), wrong_m2(sim), sim.lost,
            sim.bumps, sim.agent_bumps)  # fmt: skip


def gain(a: tuple, b: tuple, budget_s: float) -> float:
    """a から b までに得た点（b の時刻で覆った分を重みづけ）。"""
    early = max(0.5, 1.0 - 0.5 * min(b[0] / budget_s, 1.0))
    return (
        K_COV * (b[1] - a[1]) * early
        - K_WRONG * (b[2] - a[2])
        - K_LOST * (b[3] - a[3])
        - K_BUMP * (b[4] - a[4])
        - K_AGENT * (b[5] - a[5])
    )


def run_episode(
    home_seed: int,
    policy: WeightPolicy | None = None,
    sigma: np.ndarray | float = 0.0,
    white: bool = False,
    budget_s: float = 900.0,
    snoo: str = "random",
    rng_seed: int = 0,
) -> dict:
    """1 回探って、点数・結果（Episode）・周ごとの Step を返す。"""
    home = make_home(home_seed, agent_t_s=budget_s + 120)
    sim = GridSim(home, snoo=snoo, texture=home_texture(home, white))
    rng = np.random.default_rng([home_seed, rng_seed])
    sig = np.asarray(sigma, float)
    steps: list[Step] = []
    marks: list[tuple] = []

    def chooser(s: GridSim) -> dict:
        heights = [CAM_H_M] * len(s.stations)
        params = dataclasses.replace(s.params, seed=len(s.stations) - 1, policy=None)
        recent = [st.pose for st in s.stations[-RECENT_STATIONS:]]
        s.grid.now_s = s.t
        if policy is None:
            action, _ = decide(s.grid, s.stations, params, heights, recent=recent)
            return action
        state = s.grid.state()
        f = features(s.grid, s.pose, recent, state, confidence_of(s.grid, state), len(s.stations))
        z = policy.mean(f)
        if np.any(sig > 0):
            z = z + sig * rng.standard_normal(z.shape)
        steps.append(Step(f, z, s.t))
        marks.append(_snapshot(s))
        action, _ = decide(s.grid, s.stations, params, heights, recent=recent,
                           weights=to_weights(z))  # fmt: skip
        return action

    ep = sim.run(budget_s, chooser=chooser)
    end = _snapshot(sim)
    # 撮り終えるごとの覆った割合から、早く覆うほど重い点を足す（間違い・見失い・ぶつかりは最後に）
    score, prev = 0.0, 0.0
    for t, cov_m2, _ in sim.timeline:
        c = cov_m2 / max(sim.reach_m2, 1e-9)
        score += K_COV * (c - prev) * max(0.5, 1.0 - 0.5 * min(t / budget_s, 1.0))
        prev = c
    score -= K_WRONG * end[2] + K_LOST * end[3] + K_BUMP * end[4] + K_AGENT * end[5]
    for i, st in enumerate(steps):
        b = marks[i + 1] if i + 1 < len(marks) else end
        st.reward = gain(marks[i], b, budget_s)
    return {
        "seed": home_seed, "white": white, "snoo": snoo, "score": round(score, 4),
        "coverage": ep.coverage, "m2": ep.covered_m2, "t_s": ep.t_s, "stations": ep.cycles,
        "lost": ep.lost, "bumps": ep.bumps, "agent_bumps": ep.agent_bumps,
        "wrong_m2": round(end[2], 4), "reason": ep.reason, "steps": steps,
    }  # fmt: skip


def mean_rows(rows: list[dict]) -> dict:
    """家ごとの結果の平均（白い壁かどうかで分けたものも）。"""
    out = {}
    for name, mine in (("all", rows), ("normal", [r for r in rows if not r["white"]]),
                       ("white", [r for r in rows if r["white"]])):  # fmt: skip
        if not mine:
            continue
        out[name] = {k: round(float(np.mean([r[k] for r in mine])), 4)
                     for k in ("score", "coverage", "m2", "lost", "bumps", "agent_bumps",
                               "wrong_m2")}  # fmt: skip
    return out
