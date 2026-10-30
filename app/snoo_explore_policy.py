"""次の 1 歩を、候補の中から学んだ点数で選ぶ（学ぶのは選び方だけ）。

今の決め方（app/snoo_mapper.decide）は、次に撮る位置（Next-Best-View）のいちばん良いもの、
後ろなら切り返し、ときどきでたらめな 1 歩、と手で決めている。何千もの家
（app/snoo_homes.py）で回すと、どの場面でどれを選ぶと早く広く分かるかを学べる。

- 候補：今ある作り方そのまま。NBV の上位 K_NBV 個（行き先を 40 cm 以上離す）・向きを変える
  1 歩（plan_turn）・切り返し（turning_reverse）・でたらめな 1 歩（random_step）・決まった弧
  （まっすぐ・左右いっぱい）。どれも作るときに地図でぶつからないかを確かめている
  （clip_pieces・moves_are_clear）。学んだ点数はこの中から選ぶだけなので、安全の確かめは
  今と同じ
- 特徴：候補ごとに FEATURES の数（利得・道のり・曲がる角度・境目の塊の大きさ・障害物までの
  距離・行き先のまわりの見ていない割合・前に撮った所の近さ・今の決め方ならこれを選ぶか…）と、
  場面の数（ステーションの数・分かった広さ・固まっているか…）
- 点数：小さな MLP（FEATURES → 64 → 64 → 1、tanh）。numpy だけで計算する（地図のサーバーに
  torch を入れない）。重みは npz（scripts/snoo_explore_train.py が書く）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.roommap import (
    FREE,
    MAX_STEP_M,
    OCCUPIED,
    TURN_RADIUS_M,
    UNKNOWN,
    WHEELBASE_M,
    Grid,
    Plan,
    Pose,
    bearing,
    big_frontiers,
    blocked_cells,
    camera_of,
    clip_pieces,
    drive_poses,
    next_best_views,
    plan_turn,
    random_step,
    turning_reverse,
    view_gain,
)

# NBV の候補の数
K_NBV = 5
# 候補の数の上限（学ぶときに揃える）
K_MAX = 14
KINDS = ("nbv", "turn", "reverse", "random", "arc")
FEATURES = (
    *(f"kind_{k}" for k in KINDS),
    "heuristic",  # 今の決め方（でたらめを除く）ならこれを選ぶ
    "nbv_rank",
    "target_gain",
    "utility",
    "route_m",
    "step_m",
    "end_gain",  # 1 歩走った先で見えそうな未知のセル
    "heading_change",
    "target_bearing",
    "reverse",
    "frontier_size",  # 行き先にいちばん近い境目の塊の大きさ
    "frontier_dist",
    "end_clearance",  # 1 歩走った先から、障害物と見ているセルまで
    "target_unknown",  # 行き先の 1 m 四方の見ていない割合
    "end_visits",  # 1 歩走った先の 50 cm 以内で前に撮った数
    "end_novelty",  # 1 歩走った先から、前に撮ったいちばん近い所まで
    # 場面（どの候補も同じ値）
    "stations",
    "known_m2",
    "frontier_cells",
    "clustered",
    "recent_spread",
    "n_candidates",
    "last_move",  # 前のステーションからの距離（同じ所で止まり続けているか）
)
N_FEATURES = len(FEATURES)
HIDDEN = 64


@dataclass
class Candidate:
    kind: str
    action: dict
    plan: Plan | None
    end: Pose  # 1 歩走った先（カメラの真下）
    target: Pose  # 行き先（NBV のとき。ほかは end）
    pieces_m: float
    rank: int = -1  # NBV の順位
    gain: int = 0  # 行き先で見えそうな未知のセル（NBV）


def _arc_pieces(grid: Grid, blocked: np.ndarray, pose: Pose, steer: float, m: float):
    pieces = clip_pieces(grid, blocked, pose, [(steer, m)])
    return pieces if pieces and sum(x for _, x in pieces) >= 0.15 else []


def build_candidates(
    grid: Grid, stations, params, rng: np.random.Generator
) -> tuple[list[Candidate], int]:
    """候補の 1 歩と、今の決め方（でたらめを除く）が選ぶ候補の番号（無ければ -1）。"""
    from app.snoo_mapper import (
        SPEED,
        plan_action,
        pose_dict,
        random_action,
        reverse_action,
        turn_action,
        turn_instead,
    )

    pose = stations[-1].pose
    state = grid.state()
    blocked = blocked_cells(grid, state, pose)
    out: list[Candidate] = []
    plans = next_best_views(
        grid, pose, K_NBV, segments=params.segments, max_step_m=params.max_step_m
    )
    for r, p in enumerate(plans):
        pieces = p.segments or [(p.steer_deg, p.step_m)]
        end = camera_of(drive_poses(pose, pieces)[-1][1])
        out.append(Candidate("nbv", plan_action(p, pose, params), p, end, p.target,
                             sum(m for _, m in pieces), r, p.gain))  # fmt: skip
    heuristic = 0 if plans else -1
    if params.reverse and params.plan_turns:
        turns = []
        best = turn_instead(grid, pose, plans[0] if plans else None, params)
        if best is not None:
            turns.append(best)
        other = plan_turn(grid, pose, None)
        if other is not None and (best is None or other.moves != best.moves):
            turns.append(other)
        for k, t in enumerate(turns):
            if k == 0 and best is not None:
                heuristic = len(out)
            out.append(Candidate("turn", turn_action(t, pose, params), None, t.end, t.end,
                                 sum(m for _, m, _ in t.moves)))  # fmt: skip
    if params.reverse:
        for length in (0.30, 0.15):
            pieces = turning_reverse(grid, pose, length=length)
            if pieces:
                action = reverse_action(pieces, params.segments)
                action["pose"] = pose_dict(pose)
                end = camera_of(drive_poses(pose, pieces, reverse=True)[-1][1])
                out.append(Candidate("reverse", action, None, end, end, sum(m for _, m in pieces)))
                break
    step = random_step(grid, pose, rng)
    if step is not None:
        if heuristic < 0:
            heuristic = len(out)
        end = camera_of(drive_poses(pose, [step])[-1][1])
        out.append(Candidate("random", random_action(step, pose, params), None, end, end, step[1]))
    max_steer = math.degrees(math.atan(WHEELBASE_M / TURN_RADIUS_M))
    for steer in (0.0, -max_steer, max_steer):
        pieces = _arc_pieces(grid, blocked, pose, steer, MAX_STEP_M)
        if not pieces:
            continue
        st, m = pieces[0][0], sum(x for _, x in pieces)
        action = random_action((st, m), pose, params)
        action.pop("random")
        action["speed"] = SPEED
        end = camera_of(drive_poses(pose, pieces)[-1][1])
        out.append(Candidate("arc", action, None, end, end, m))
    return out[:K_MAX], heuristic if heuristic < K_MAX else -1


def featurize(
    grid: Grid, stations, cands: list[Candidate], heuristic: int, clustered_now: bool
) -> np.ndarray:
    """候補ごとの特徴 (K, N_FEATURES)。だいたい 0〜1 になるよう割っておく。"""
    state = grid.state()
    pose = stations[-1].pose
    front = big_frontiers(state)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
        front.astype(np.uint8), connectivity=8
    )
    fcells = np.argwhere(front)
    fx, fy = grid.center(fcells[:, 0], fcells[:, 1]) if len(fcells) else (np.zeros(0),) * 2
    occ = state == OCCUPIED
    dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * grid.res
    unknown = (state == UNKNOWN).astype(np.float32)
    integral = cv2.integral(unknown)
    known_m2 = float((state == FREE).sum() * grid.res**2)
    past = np.array([[s.pose.x, s.pose.y] for s in stations[:-1]] or [[1e9, 1e9]])
    recent = np.array([[s.pose.x, s.pose.y] for s in stations[-4:]])
    spread = float(np.hypot(*(recent - recent.mean(0)).T).max()) if len(recent) > 1 else 0.0
    glob = [
        len(stations) / 100,
        known_m2 / 80,
        len(fcells) / 2000,
        float(clustered_now),
        min(spread, 2.0),
        len(cands) / K_MAX,
        min(float(np.hypot(*(recent[-1] - recent[-2]))), 1.0) / 0.4 if len(recent) > 1 else 1.0,
    ]
    rows = []
    half = int(round(0.5 / grid.res))
    for i, c in enumerate(cands):
        one_hot = [float(c.kind == k) for k in KINDS]
        end_gain = view_gain(grid, state, c.end, rays=21)
        t = c.target
        if len(fcells):
            d2 = (fx - t.x) ** 2 + (fy - t.y) ** 2
            j = int(np.argmin(d2))
            fsize = stats[labels[fcells[j, 0], fcells[j, 1]], cv2.CC_STAT_AREA]
            fdist = math.sqrt(float(d2[j]))
        else:
            fsize, fdist = 0, 3.0
        ix, iy = grid.cell(c.end.x, c.end.y)
        clear = float(dist[ix, iy]) if grid.inside(ix, iy) else 0.0
        tx, ty = grid.cell(t.x, t.y)
        a0, a1 = max(tx - half, 0), min(tx + half + 1, grid.n)
        b0, b1 = max(ty - half, 0), min(ty + half + 1, grid.n)
        area = max((a1 - a0) * (b1 - b0), 1)
        unk = (integral[a1, b1] - integral[a0, b1] - integral[a1, b0] + integral[a0, b0]) / area
        dpast = np.hypot(past[:, 0] - c.end.x, past[:, 1] - c.end.y)
        gain = c.gain if c.kind == "nbv" else end_gain
        route = c.plan.length_m if c.plan is not None else c.pieces_m
        dyaw = abs(math.remainder(c.end.yaw - pose.yaw, 2 * math.pi))
        rows.append(
            one_hot
            + [
                float(i == heuristic),
                (c.rank + 1) / K_NBV if c.rank >= 0 else 0.0,
                gain / 400,
                gain * math.exp(-route / 1.5) / 400,
                min(route, 4.0) / 3,
                c.pieces_m / 0.6,
                end_gain / 150,
                dyaw / math.pi,
                abs(bearing(pose, t)) / math.pi if c.kind == "nbv" else 0.0,
                float(bool(c.action.get("reverse"))),
                min(fsize, 400) / 200,
                min(fdist, 3.0) / 2,
                min(clear, 1.0),
                float(unk),
                min(int((dpast < 0.5).sum()), 10) / 5,
                min(float(dpast.min()), 3.0) / 2,
            ]
            + glob
        )
    return np.asarray(rows, np.float32).reshape(len(cands), N_FEATURES)


# --- 点数（numpy だけ） ---


class Scorer:
    """候補の点数を出す MLP（と、学ぶときの場面の値）。重みは npz の pi_* と v_*。"""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = {k: np.asarray(v, np.float32) for k, v in weights.items()}
        if self.w["pi_w1"].shape[0] != N_FEATURES:
            raise ValueError(
                f"重みの特徴の数 {self.w['pi_w1'].shape[0]} が今の {N_FEATURES} と違う"
            )

    @classmethod
    def load(cls, path: str | Path) -> Scorer:
        with np.load(path) as d:
            return cls(dict(d))

    @classmethod
    def random(cls, seed: int = 0, hidden: int = HIDDEN) -> Scorer:
        rng = np.random.default_rng(seed)

        def lin(a, b, scale=1.0):
            return (rng.normal(0, scale / math.sqrt(a), (a, b))).astype(np.float32), np.zeros(
                b, np.float32
            )

        w = {}
        for pre, nin in (("pi", N_FEATURES), ("v", 2 * N_FEATURES)):
            w[f"{pre}_w1"], w[f"{pre}_b1"] = lin(nin, hidden)
            w[f"{pre}_w2"], w[f"{pre}_b2"] = lin(hidden, hidden)
            w[f"{pre}_w3"], w[f"{pre}_b3"] = lin(hidden, 1, 0.01)
        return cls(w)

    def _mlp(self, pre: str, x: np.ndarray) -> np.ndarray:
        w = self.w
        h = np.tanh(x @ w[f"{pre}_w1"] + w[f"{pre}_b1"])
        h = np.tanh(h @ w[f"{pre}_w2"] + w[f"{pre}_b2"])
        return (h @ w[f"{pre}_w3"] + w[f"{pre}_b3"])[..., 0]

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._mlp("pi", x)

    def value(self, x: np.ndarray) -> float:
        pooled = np.concatenate([x.mean(0), x.max(0)])
        return float(self._mlp("v", pooled[None])[0])


_SCORERS: dict[str, Scorer] = {}


def load_scorer(path: str | Path) -> Scorer:
    """重みを読む（同じファイルは 1 回だけ）。"""
    key = str(Path(path).resolve())
    if key not in _SCORERS:
        _SCORERS[key] = Scorer.load(path)
    return _SCORERS[key]


@dataclass
class Choice:
    action: dict
    plan: Plan | None
    x: np.ndarray | None = None  # (K, N_FEATURES)
    index: int = -1
    logp: float = 0.0
    value: float = 0.0
    heuristic: int = -1


def choose(
    grid: Grid,
    stations,
    params,
    scorer: Scorer | None,
    rng: np.random.Generator,
    sample: bool = False,
    clustered_now: bool = False,
    pick: str = "policy",
) -> Choice:
    """候補を作って選ぶ。pick は policy（点数）・heuristic（今の決め方のでたらめ抜き）。
    候補が無ければ unreachable（地図のサーバーは or_reverse で来た道を戻る）。"""
    cands, heuristic = build_candidates(grid, stations, params, rng)
    if not cands:
        return Choice({"action": "done", "reason": "unreachable"}, None)
    x = featurize(grid, stations, cands, heuristic, clustered_now)
    if pick == "heuristic" or scorer is None:
        i = heuristic if heuristic >= 0 else 0
        return Choice(_tag(cands[i]), cands[i].plan, x, i, 0.0, 0.0, heuristic)
    z = scorer.logits(x).astype(np.float64)
    z -= z.max()
    p = np.exp(z) / np.exp(z).sum()
    i = int(rng.choice(len(p), p=p)) if sample else int(np.argmax(z))
    return Choice(_tag(cands[i], float(p[i])), cands[i].plan, x, i, float(math.log(p[i] + 1e-12)),
                  scorer.value(x), heuristic)  # fmt: skip


def _tag(c: Candidate, p: float | None = None) -> dict:
    action = dict(c.action)
    action["policy"] = {"kind": c.kind} | ({"p": round(p, 3)} if p is not None else {})
    return action


def policy_decide(grid: Grid, stations, params) -> tuple[dict, Plan | None]:
    """app/snoo_mapper.decide から呼ぶ（params.policy の重みで、いちばん点数の高い候補）。"""
    # 固まっているか（force_random）は、今の決め方ではでたらめな 1 歩にするが、ここでは特徴にする
    c = choose(
        grid, stations, params, load_scorer(params.policy), np.random.default_rng(params.seed),
        clustered_now=params.force_random,
    )  # fmt: skip
    return c.action, c.plan
