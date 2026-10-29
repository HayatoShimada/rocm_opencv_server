"""家（app/snoo_homes.py）を 1 つ作り、速いシミュレーター（app/snoo_gridsim.py）で 1 回探る。

学ぶ（scripts/snoo_explore_train.py）ときも、比べる（scripts/snoo_explore_eval.py）ときも
これを使う。numpy だけで動く（学ぶときの子のプロセスに torch を読み込ませない。GPU を
1 つのプロセスだけが使うように）。

決め方（mode）：
- decide：今の決め方（app/snoo_mapper.decide → or_reverse。でたらめな 1 歩も入る）
- heuristic：候補の中から、今の決め方の「でたらめ抜き」の 1 歩を選ぶ
- policy：学んだ点数で選ぶ（sample なら確率で、でなければいちばん高いもの）
"""

from __future__ import annotations

import dataclasses

import numpy as np

from app.snoo_explore_policy import Scorer, choose
from app.snoo_gridsim import CAM_H_M, Episode, GridSim
from app.snoo_homes import make_home
from app.snoo_mapper import CLUSTER_STATIONS, clustered, or_reverse, stop_reason

# 報酬：新しく分かった床（m²）から、時間（秒あたり）とぶつかった数を引く
TIME_COST = 0.002
BUMP_COST = 1.0
AGENT_BUMP_COST = 3.0


@dataclasses.dataclass
class Step:
    x: np.ndarray  # (K, F)
    index: int
    logp: float
    value: float
    heuristic: int
    reward: float = 0.0


def run_episode(
    home_seed: int,
    mode: str = "decide",
    weights: dict | None = None,
    sample: bool = False,
    budget_s: float = 1200.0,
    rng_seed: int = 0,
    keep_sim: bool = False,
) -> dict:
    """1 回探って、結果（Episode）と、点数で選んだ周ごとの Step（学ぶとき）を返す。"""
    home = make_home(home_seed, agent_t_s=budget_s + 120)
    sim = GridSim(home)
    steps: list[Step] = []
    marks: list[tuple[float, float, int, int]] = []  # 選んだときの (秒, 床 m², 家具, 人と猫)
    scorer = Scorer(weights) if weights is not None else None
    rng = np.random.default_rng([home_seed, rng_seed])

    def chooser(s: GridSim) -> dict:
        heights = [CAM_H_M] * len(s.stations)
        recent = [st.pose for st in s.stations[-CLUSTER_STATIONS - 1 : -1]]
        params = dataclasses.replace(
            s.params, seed=len(s.stations) - 1, force_random=clustered(recent)
        )
        stop = stop_reason(s.grid, s.stations, params, heights)
        if stop is not None:
            return stop
        pick = "heuristic" if mode == "heuristic" else "policy"
        c = choose(
            s.grid, s.stations, params, scorer, rng, sample=sample,
            clustered_now=params.force_random, pick=pick,
        )  # fmt: skip
        if c.x is not None:
            steps.append(Step(c.x, c.index, c.logp, c.value, c.heuristic))
            marks.append((s.t, s.covered_m2(), s.bumps, s.agent_bumps))
        action, _ = or_reverse(c.action, s.grid, s.pose, s.history(), params)
        return action

    ep: Episode = sim.run(budget_s, chooser=None if mode == "decide" else chooser)
    end = (sim.t, sim.covered_m2(), sim.bumps, sim.agent_bumps)
    for i, st in enumerate(steps):
        a, b = marks[i], marks[i + 1] if i + 1 < len(marks) else end
        st.reward = (
            (b[1] - a[1])
            - TIME_COST * (b[0] - a[0])
            - BUMP_COST * (b[2] - a[2])
            - AGENT_BUMP_COST * (b[3] - a[3])
        )
    out = {"episode": ep, "steps": steps}
    if keep_sim:
        out["sim"] = sim
    return out
