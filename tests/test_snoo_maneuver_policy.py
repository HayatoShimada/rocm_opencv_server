"""計画器の重みを場面から出す方策（app/snoo_maneuver_policy.py）と、その学び方
（app/snoo_explore_rollout.py・scripts/snoo_explore_train.py）の試験。GPU も torch も使わない。"""

import dataclasses
import math
import subprocess
import sys

import numpy as np
import pytest

from app.snoo_belief import confidence_of
from app.snoo_gridsim import CAM_H_M, GridSim
from app.snoo_homes import home_texture, make_home
from app.snoo_maneuver import FIXED, GEAR_MAX_SEGMENTS, Limits, Scene, Weights, project, rollout
from app.snoo_maneuver_policy import (
    FEATURES,
    KNOBS,
    N_FEATURES,
    N_KNOBS,
    Z_MAX,
    WeightPolicy,
    features,
    to_weights,
)
from app.snoo_mapper import MapParams, decide


@pytest.fixture(scope="module")
def scanned():
    home = make_home(1, agent_t_s=200)
    sim = GridSim(home, texture=home_texture(home, True))
    sim.scan()
    return sim


def _random_policy(seed: int = 0, scale: float = 0.5) -> WeightPolicy:
    rng = np.random.default_rng(seed)
    return WeightPolicy(rng.normal(0, scale, (N_KNOBS, N_FEATURES)), rng.normal(0, scale, N_KNOBS))


def test_zero_policy_gives_the_fixed_weights():
    f = np.random.default_rng(0).random(N_FEATURES)
    assert WeightPolicy().weights(f) == FIXED
    assert to_weights(np.zeros(N_KNOBS)) == FIXED


def test_knobs_are_weights_and_safety_terms_are_not_learned():
    names = {f.name for f in dataclasses.fields(Weights)}
    assert set(KNOBS) <= names
    # 安全の項（ぶつかる・後ろの確かでない所・前の見ていない所）は Weights に無い
    assert not any(k in names for k in ("unk_rev", "unk_fwd", "seg"))


def test_weights_are_clipped_to_z_max():
    w = to_weights(np.full(N_KNOBS, 10.0))
    for k in KNOBS:
        assert getattr(w, k) == pytest.approx(getattr(FIXED, k) * math.exp(Z_MAX))
    z = _random_policy(scale=5.0).mean(np.ones(N_FEATURES))
    assert np.all(np.abs(z) <= Z_MAX + 1e-12)


def test_features_have_the_right_shape(scanned):
    sim = scanned
    state = sim.grid.state()
    f = features(sim.grid, sim.pose, [sim.pose], state, confidence_of(sim.grid, state), 1)
    assert f.shape == (len(FEATURES),) and np.isfinite(f).all()
    assert 0.0 <= f[FEATURES.index("unknown_near")] <= 1.0
    assert f[FEATURES.index("stuck")] == 0.0


def test_any_weights_keep_the_same_safe_sequences(scanned):
    """重みを変えても、走れる列（ぶつからない・後ろの確かでない所に入らない・clear で止まらない）は
    同じ。変わるのは点数だけ。"""
    sim = scanned
    rng = np.random.default_rng(0)
    K = 200
    steer = rng.uniform(-30, 30, (K, GEAR_MAX_SEGMENTS))
    dist = rng.uniform(-0.3, 0.3, (K, GEAR_MAX_SEGMENTS))
    steer, dist = project(steer, dist, Limits())
    base = rollout(Scene(sim.grid, sim.pose, True), steer, dist)
    for z in (np.full(N_KNOBS, Z_MAX), np.full(N_KNOBS, -Z_MAX), rng.normal(0, 1, N_KNOBS)):
        other = rollout(Scene(sim.grid, sim.pose, True, weights=to_weights(z)), steer, dist)
        assert (other.ok == base.ok).all()
    assert base.ok.any() and not base.ok.all()


def test_save_and_load(tmp_path):
    p = _random_policy(3)
    p.save(tmp_path / "w.npz")
    q = WeightPolicy.load(tmp_path / "w.npz")
    assert np.allclose(p.theta, q.theta)
    assert np.allclose(WeightPolicy.from_theta(p.theta).W, p.W)
    np.savez(tmp_path / "bad.npz", W=p.W, b=p.b, features=np.array(FEATURES[::-1]),
             knobs=np.array(KNOBS))  # fmt: skip
    with pytest.raises(ValueError):
        WeightPolicy.load(tmp_path / "bad.npz")


def test_policy_runs_without_torch(tmp_path):
    _random_policy(1).save(tmp_path / "w.npz")
    code = (
        "import sys, numpy as np\n"
        "from app.snoo_maneuver_policy import load_policy, N_FEATURES\n"
        f"p = load_policy({str(tmp_path / 'w.npz')!r})\n"
        "w = p.weights(np.full(N_FEATURES, 0.5))\n"
        "assert w.goal > 0 and w.lam > 0\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_map_params_read_the_policy_from_the_environment(monkeypatch):
    monkeypatch.delenv("SNOO_MANEUVER_POLICY", raising=False)
    assert MapParams().policy is None
    monkeypatch.setenv("SNOO_MANEUVER_POLICY", "/x/w.npz")
    assert MapParams().policy == "/x/w.npz"


def test_decide_uses_the_policy_only_when_given(scanned, tmp_path):
    sim = scanned
    sim.grid.now_s = sim.t
    params = MapParams(camera_height_m=CAM_H_M, segments=True, reverse=True, gear_segments=True,
                       policy=None)  # fmt: skip
    action, _ = decide(sim.grid, sim.stations, params, [CAM_H_M])
    assert action["action"] == "move" and "policy" not in action
    WeightPolicy(b=np.full(N_KNOBS, 0.5)).save(tmp_path / "w.npz")
    params = dataclasses.replace(params, policy=str(tmp_path / "w.npz"))
    action, _ = decide(sim.grid, sim.stations, params, [CAM_H_M])
    assert action["action"] == "move"
    scale = action["policy"]["scale"]
    assert set(scale) == set(KNOBS)
    assert scale["goal"] == pytest.approx(math.exp(Z_MAX * math.tanh(0.5)), abs=1e-3)


# --- 学び方 ---


def test_ppo_gradients_match_numeric():
    from scripts.snoo_explore_train import logp_and_grads

    rng = np.random.default_rng(1)
    n = N_KNOBS * (N_FEATURES + 1)
    th, ls = rng.normal(0, 0.3, n), rng.normal(-1, 0.2, N_KNOBS)
    F, Z = rng.random((4, N_FEATURES)), rng.normal(0, 1, (4, N_KNOBS))
    _, gt, gs = logp_and_grads(th, ls, F, Z)
    e = 1e-6
    for i in (0, 13, n - 1):
        d = np.zeros(n)
        d[i] = e
        num = (logp_and_grads(th + d, ls, F, Z)[0] - logp_and_grads(th - d, ls, F, Z)[0]) / (2 * e)
        assert np.allclose(num, gt[:, i], atol=1e-6)
    d = np.zeros(N_KNOBS)
    d[2] = e
    num = (logp_and_grads(th, ls + d, F, Z)[0] - logp_and_grads(th, ls - d, F, Z)[0]) / (2 * e)
    assert np.allclose(num, gs[:, 2], atol=1e-6)


def test_gae_with_no_value_is_the_discounted_sum():
    from scripts.snoo_explore_train import gae

    f = np.zeros(N_FEATURES)
    z = np.zeros(N_KNOBS)
    rows = [{"budget_s": 100.0, "steps": [(f, z, 0.0, 1.0), (f, z, 10.0, 0.0), (f, z, 20.0, 2.0)]}]
    F, Z, A, R, T = gae(rows, None, gamma=0.5, lam=1.0)
    assert np.allclose(A, [1 + 0.25 * 2, 0.5 * 2, 2.0]) and np.allclose(R, A)
    assert np.allclose(T, [0.0, 0.1, 0.2]) and F.shape == (3, N_FEATURES)


def test_cma_es_finds_the_minimum_of_a_bowl():
    from scripts.snoo_explore_train import cma_ask, cma_init, cma_tell

    rng = np.random.default_rng(0)
    es = cma_init(12, 0.5, 10)
    es["mean"][:] = 2.0
    for _ in range(120):
        xs = cma_ask(es, rng)
        cma_tell(es, xs, -(xs**2).sum(1))
    assert np.linalg.norm(es["mean"]) < 0.05


def test_curriculum_and_home_kinds():
    from scripts.snoo_explore_train import BUDGET0_S, BUDGET1_S, CURRICULUM, budget_at, home_kind

    assert budget_at(0) == BUDGET0_S and budget_at(CURRICULUM) == budget_at(999) == BUDGET1_S
    kinds = [home_kind(s) for s in range(10)]
    assert sum(w for w, _ in kinds) == 5
    assert {s for _, s in kinds} == {"random", "behavior", "sneak"}


def test_short_episode_scores_and_rewards():
    from app.snoo_explore_rollout import K_COV, run_episode

    r = run_episode(2, WeightPolicy(), 0.3, white=True, budget_s=90.0)
    steps = r["steps"]
    assert steps and math.isfinite(r["score"]) and r["score"] <= K_COV
    assert all(s.f.shape == (N_FEATURES,) and s.z.shape == (N_KNOBS,) for s in steps)
    assert all(math.isfinite(s.reward) for s in steps)
    fixed = run_episode(2, None, white=True, budget_s=90.0)
    assert fixed["steps"] == [] and fixed["stations"] >= 1
