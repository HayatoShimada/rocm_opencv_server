import json
import math

import numpy as np
import pytest

from app.roommap import Pose
from app.snoo_behavior import (
    MODEL_PATH,
    POSTURES,
    PRIOR_MODEL,
    BehaviorAgent,
    BehaviorPose,
    BehaviorSnoo,
    grid_free,
    load_model,
)
from app.snoo_gridsim import GridSim
from app.snoo_homes import make_home


def _open_room(x0=0.0, y0=0.0, x1=4.0, y1=4.0, r=0.15):
    def free(x, y):
        return x0 + r <= x <= x1 - r and y0 + r <= y <= y1 - r

    return free


def _sources(d, out):
    if isinstance(d, dict):
        if "source" in d:
            out.append(d["source"])
        for v in d.values():
            _sources(v, out)
    return out


def test_repo_model_loads_and_every_parameter_says_where_it_came_from():
    raw = json.loads(MODEL_PATH.read_text())
    assert len(MODEL_PATH.read_bytes()) < 50_000
    m = load_model()
    assert set(m["postures"]) == set(POSTURES)
    srcs = _sources(raw, [])
    assert srcs and set(srcs) <= {"data", "prior"}
    # 録画から作った数がある（全部 prior のままではない）
    assert "data" in srcs
    for p in POSTURES:
        d = m["postures"][p]["duration_s"]
        assert d["median"] > 0 and d["sigma"] > 0


@pytest.mark.parametrize("model", [PRIOR_MODEL, load_model()], ids=["prior", "repo"])
def test_posture_transitions_sum_to_one_and_never_stay(model):
    for p in POSTURES:
        nxt = model["postures"][p]["next"]["value"]
        assert sum(nxt.values()) == pytest.approx(1.0, abs=1e-2)
        assert nxt.get(p, 0.0) == 0.0
        assert set(nxt) <= set(POSTURES)
    for probs in model["reaction"]["choice"]["value"].values():
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-2)


def _run(seed, robot_path):
    snoo = BehaviorSnoo(BehaviorPose(2.0, 2.0, 0.0), _open_room(), np.random.default_rng(seed))
    out = []
    for robot in robot_path:
        p = snoo.step(0.25, robot)
        out.append((round(p.x, 6), round(p.y, 6), round(p.yaw, 6), p.posture))
    return out, snoo


def test_same_seed_same_robot_gives_the_same_cat():
    path = [Pose(0.5 + 0.01 * k, 1.0, 0.0) for k in range(400)]
    a, _ = _run(4, path)
    b, _ = _run(4, path)
    c, _ = _run(5, path)
    assert a == b
    assert a != c


def test_flees_when_the_robot_comes_fast_inside_the_flight_distance():
    m = load_model()
    # 気づいたときのくじでは逃げないようにして、飛び退く距離の決まりだけを確かめる
    for probs in m["reaction"]["choice"]["value"].values():
        probs["flee"] = 0.0
        probs["freeze"] = probs.get("freeze", 0.0) + 1e-3
    flight = m["reaction"]["flight_m"]["value"]
    fast = m["reaction"]["fast_m_s"]["value"]
    v = fast + 0.25
    for seed in range(5):
        snoo = BehaviorSnoo(
            BehaviorPose(2.0, 2.0, 0.0),
            _open_room(),
            np.random.default_rng(seed),
            model=m,
            posture="sit",
        )
        x = 2.0 - flight - 0.6
        for _ in range(100):
            x += v * 0.1
            if math.hypot(snoo.x - x, snoo.y - 2.0) < 0.35:
                break
            snoo.step(0.1, Pose(x, 2.0, 0.0))
        flee = [r for r in snoo.reactions if r["kind"] == "flee"]
        assert flee and flee[0]["dist_m"] <= flight and flee[0]["closing_m_s"] > fast
        # 逃げた後は飛び退く距離より遠い
        assert math.hypot(snoo.x - x, snoo.y - 2.0) > flight


def test_a_robot_far_away_does_not_scare_the_cat():
    snoo = BehaviorSnoo(
        BehaviorPose(3.5, 3.5, 0.0), _open_room(), np.random.default_rng(0), posture="sleep"
    )
    for _ in range(200):
        snoo.step(0.1, Pose(0.5, 0.5, 0.0))
    assert not snoo.reactions and snoo.posture == "sleep"


def test_never_walks_through_obstacles_or_into_the_robot():
    # 4 x 4 m の部屋の真ん中に壁（扉は 1 か所）
    room = _open_room()

    def free(x, y):
        wall = 1.85 <= x <= 2.15 and not (3.0 <= y <= 3.6)
        return room(x, y) and not wall

    snoo = BehaviorSnoo(BehaviorPose(1.0, 1.0, 0.0), free, np.random.default_rng(2), posture="walk")
    snoo.m["postures"]["walk"]["duration_s"] = {"median": 60.0, "sigma": 0.1}
    robot = Pose(1.0, 2.5, 0.0)
    sides = set()
    prev = prev_moved = (snoo.x, snoo.y)
    for k in range(6000):
        rx = 1.0 + 0.6 * math.sin(k / 50)  # 車が行き来する
        robot = Pose(rx, 2.5, 0.0)
        p = snoo.step(0.1, robot)
        assert free(p.x, p.y)
        # 1 刻みで壁を飛び越えない
        assert math.hypot(p.x - prev[0], p.y - prev[1]) < 0.3
        prev = (p.x, p.y)
        sides.add(p.x > 2.0)
        if p.moving and (p.x, p.y) != prev_moved:
            # 自分から車の円に入らない（車のほうから来るのは止められない）
            assert math.hypot(p.x - (robot.x - 0.08), p.y - robot.y) >= 0.32 - 1e-6
        prev_moved = (p.x, p.y)
    assert sides == {False, True}  # 扉を通って向こうへも行く


def test_grid_free_reads_cells():
    ok = np.zeros((10, 10), bool)
    ok[2:5, 3:6] = True
    free = grid_free(ok, -0.25, 0.05)
    assert free(-0.25 + 0.05 * 3.5, -0.25 + 0.05 * 4.5)
    assert not free(-0.25 + 0.05 * 6.5, -0.25 + 0.05 * 4.5)
    assert not free(10.0, 10.0)


def test_behavior_agent_only_moves_forward_in_time():
    snoo = BehaviorSnoo(
        BehaviorPose(2.0, 2.0, 0.0), _open_room(), np.random.default_rng(1), posture="walk"
    )
    a = BehaviorAgent(snoo, 0.15, 0.27)
    p5 = a.at(5.0)
    assert a.at(1.0) == p5  # 前の時刻はいまの位置
    a.at(10.0)
    assert a.t == 10.0


def test_grid_sim_with_the_behavior_cat_is_deterministic_and_reacts():
    a = GridSim(make_home(5, agent_t_s=400), snoo="behavior")
    b = GridSim(make_home(5, agent_t_s=400), snoo="behavior")
    ea, eb = a.run(240), b.run(240)
    assert ea.timeline == eb.timeline and ea.agent_bumps == eb.agent_bumps
    cat = next(x for x in a.agents if x.name == "snoo")
    assert isinstance(cat, BehaviorAgent) and cat.t > 200
    with pytest.raises(ValueError):
        GridSim(make_home(5, agent_t_s=10), snoo="tiger")
