"""計画器の重みを場面から出す方策（app/snoo_maneuver_policy.py）を、作った家で学ぶ。

    make train-explore                       # data/snoo_policy/<algo>/ の続きから
    .venv/bin/python -m scripts.snoo_explore_train --algo ppo --minutes 40 --workers 3

学ぶのは z = Z_MAX × tanh(W f + b) の W・b（11 × 10 + 11 = 121 個）。安全の項は計画器の中で
固定なので、どの重みでもぶつかる道・後ろの確かでない所は選ばない。

- 回す：子のプロセス（--workers、3 本まで）が、くじの違う家（app/snoo_homes.py）を作り、
  速いシミュレーター（app/snoo_gridsim.py）で 1 回ずつ探る（app/snoo_explore_rollout.py）。
  numpy だけで、GPU も torch も使わない（家のブレーカーが落ちたことがあるので、CPU も控えめに）
- 家：くじの半分は白い壁（目印が少なく、位置を見失いやすい）。snoo は 5 軒に 3 軒が
  でたらめに歩き、1 軒が車に反応し（behavior）、1 軒が撮るたびに後ろへ回り込む（sneak）
- カリキュラム：1 回の探索の長さを、はじめは BUDGET0_S（8 分）にし、CURRICULUM 回で
  BUDGET1_S（15 分。scripts/snoo_plan_bench.py と同じ）まで延ばす
- --algo ppo：1 歩ごとに z を正規分布で試し（ガウス方策）、周ごとの報酬で PPO（クリップ）。
  値は特徴の線形回帰、利得の見積もりは GAE（γ 0.95・λ 0.8）。勾配は手で書く（numpy）
- --algo cmaes：W・b を 1 つの点として、対角の CMA-ES（sep-CMA-ES）で探す。1 世代の候補は
  同じ家で比べる（家による違いを消す）
- 確かめ：--eval-every 回ごとに、確かめの家（VAL_SEED0 から、ふつうの壁と白い壁）で、ゆらぎの
  無い方策を固定の重みと比べ、いちばん良いものを best.npz に書く
- 止め方：SIGTERM・SIGINT を受けたら、チェックポイントを書いてから終わる（scripts/snoo_phase.sh
  robot が送る）。次に動かすと続きから。--minutes は今までの分も含めた合計の時間
- 書き出し（--out/<algo>/）：ckpt.npz・state.json（様子と確かめの歴史）・policy.npz（今の方策）・
  best.npz・train.log
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import numpy as np

OUT = Path("data/snoo_policy")
# 家のくじ。学ぶ家・確かめる家（途中で良いものを選ぶ）。比べる家は scripts/snoo_plan_bench.SEED0
TRAIN_SEED0 = 0
VAL_SEED0 = 500_000
# カリキュラム（1 回の探索の秒。実機の時計）
BUDGET0_S = 480.0
BUDGET1_S = 900.0
CURRICULUM = 12
MAX_WORKERS = 3

_stop = False
_LOG: Path | None = None


def _on_signal(signum, _frame) -> None:
    global _stop
    _stop = True
    log(f"signal {signal.Signals(signum).name}: チェックポイントを書いて終わる")


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    if _LOG is not None:
        with _LOG.open("a") as f:
            f.write(line + "\n")


def budget_at(it: int) -> float:
    return BUDGET0_S + (BUDGET1_S - BUDGET0_S) * min(it / CURRICULUM, 1.0)


def home_kind(seed: int) -> tuple[bool, str]:
    """家のくじから、白い壁か・snoo の動き。"""
    snoo = ("random", "random", "random", "behavior", "sneak")[seed % 5]
    return seed % 2 == 1, snoo


# --- 子のプロセス（torch を読まない） ---


def _init_worker() -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("SNOO_MANEUVER_POLICY", None)
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C は親が受けて片付ける
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    import cv2

    cv2.setNumThreads(1)  # 子どうしで CPU を取り合わない


def _job(args: tuple) -> dict:
    from app.snoo_explore_rollout import run_episode
    from app.snoo_maneuver_policy import WeightPolicy

    seed, white, snoo, theta, sigma, budget_s, tag = args
    pol = None if theta is None else WeightPolicy.from_theta(theta)
    r = run_episode(seed, pol, sigma, white=white, budget_s=budget_s, snoo=snoo)
    r["steps"] = [(s.f, s.z, s.t, s.reward) for s in r["steps"]]
    r["tag"] = tag
    r["budget_s"] = budget_s
    return r


def make_pool(workers: int):
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        return mp.get_context("spawn").Pool(workers, initializer=_init_worker)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def collect(pool, jobs: list[tuple]) -> list[dict] | None:
    """jobs を回して結果を集める（jobs の順）。止めるよう言われたら None（その分は捨てる）。
    1 秒ごとに止めるかを見る（子がまとめて SIGTERM で死ぬと、結果が来ないまま待ち続ける）。"""
    it = pool.imap(_job, jobs)
    out = []
    while len(out) < len(jobs):
        if _stop:
            return None
        try:
            out.append(it.next(timeout=1.0))
        except mp.TimeoutError:
            continue
    return out


def val_jobs(theta, homes: int) -> list[tuple]:
    return [(VAL_SEED0 + k, white, "random", theta, 0.0, BUDGET1_S, "val")
            for k in range(homes) for white in (False, True)]  # fmt: skip


def evaluate(pool, theta, homes: int) -> dict | None:
    from app.snoo_explore_rollout import mean_rows

    rows = collect(pool, val_jobs(theta, homes))
    return None if rows is None else mean_rows(rows)


# --- Adam（numpy） ---


class Adam:
    def __init__(self, n: int, lr: float):
        self.lr, self.m, self.v, self.t = lr, np.zeros(n), np.zeros(n), 0

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """grad は上る向き（大きくしたいものの勾配）。"""
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad**2
        mh = self.m / (1 - 0.9**self.t)
        vh = self.v / (1 - 0.999**self.t)
        return x + self.lr * mh / (np.sqrt(vh) + 1e-8)


# --- PPO（ガウス方策。numpy で勾配を書く） ---

LOG_STD0 = math.log(0.4)
LOG_STD_MIN, LOG_STD_MAX = math.log(0.05), math.log(0.8)


def value_feats(F: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.concatenate([F, T[:, None], np.ones((len(F), 1))], 1)


def gae(rows: list[dict], vw: np.ndarray | None, gamma: float, lam: float):
    """周ごとの (特徴, z, 利得の見積もり, 目標の値, 時刻の割合) を並べる。"""
    F, Z, A, R, T = [], [], [], [], []
    for r in rows:
        steps = r["steps"]
        if not steps:
            continue
        f = np.array([s[0] for s in steps])
        tt = np.array([s[2] for s in steps]) / r["budget_s"]
        rew = np.array([s[3] for s in steps])
        v = value_feats(f, tt) @ vw if vw is not None else np.zeros(len(steps))
        v = np.append(v, 0.0)
        adv = np.zeros(len(steps))
        g = 0.0
        for t in reversed(range(len(steps))):
            g = rew[t] + gamma * v[t + 1] - v[t] + gamma * lam * g
            adv[t] = g
        F.append(f)
        Z.append(np.array([s[1] for s in steps]))
        A.append(adv)
        R.append(adv + v[:-1])
        T.append(tt)
    return tuple(np.concatenate(x) for x in (F, Z, A, R, T))


def logp_and_grads(theta, log_std, F, Z):
    """各周の log π(z|f) と、theta・log_std についての勾配（周ごと）。"""
    from app.snoo_maneuver_policy import N_FEATURES, N_KNOBS, Z_MAX

    n = N_KNOBS * N_FEATURES
    W, b = theta[:n].reshape(N_KNOBS, N_FEATURES), theta[n:]
    th = np.tanh(F @ W.T + b)
    mu = Z_MAX * th
    sd = np.exp(log_std)
    d = (Z - mu) / sd
    logp = (-0.5 * d**2 - log_std).sum(1)
    dmu = d / sd  # ∂logp/∂μ
    dpre = dmu * Z_MAX * (1 - th**2)  # (B, K)
    gW = dpre[:, :, None] * F[:, None, :]  # (B, K, F)
    g_theta = np.concatenate([gW.reshape(len(F), -1), dpre], 1)
    g_logstd = d**2 - 1
    return logp, g_theta, g_logstd


def ppo_update(st: dict, F, Z, A, args, rng) -> dict:
    theta, log_std = st["theta"], st["log_std"]
    old, _, _ = logp_and_grads(theta, log_std, F, Z)
    adv = (A - A.mean()) / (A.std() + 1e-8)
    n = len(F)
    clipped = kl = 0.0
    count = 0
    for _ in range(args.epochs):
        perm = rng.permutation(n)
        for k in range(0, n, args.minibatch):
            idx = perm[k : k + args.minibatch]
            lp, gt, gs = logp_and_grads(theta, log_std, F[idx], Z[idx])
            ratio = np.exp(lp - old[idx])
            a = adv[idx]
            # クリップした目的関数で、勾配が残る周（比がクリップの内側か、損をする向き）
            live = np.where(a >= 0, ratio < 1 + args.clip, ratio > 1 - args.clip)
            c = (a * ratio * live)[:, None] / len(idx)
            g_theta = (c * gt).sum(0)
            g_ls = (c * gs).sum(0) + args.ent
            theta = st["adam"].step(theta, g_theta)
            log_std = np.clip(log_std + args.lr_std * g_ls, LOG_STD_MIN, LOG_STD_MAX)
            clipped += float((~live).mean())
            kl += float((old[idx] - lp).mean())
            count += 1
        if _stop:
            break
    st["theta"], st["log_std"] = theta, log_std
    return {"clip": round(clipped / max(count, 1), 3), "kl": round(kl / max(count, 1), 4),
            "std": round(float(np.exp(log_std).mean()), 3)}  # fmt: skip


def fit_value(F, T, R, ridge: float = 1e-2) -> np.ndarray:
    X = value_feats(F, T)
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ R)


# --- sep-CMA-ES ---


def cma_init(n: int, sigma0: float, popsize: int) -> dict:
    mu = popsize // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / (w**2).sum()
    cs = (mueff + 2) / (n + mueff + 5)
    cc = 4 / (n + 4)
    # 対角だけなので、学ぶ速さを (n + 2) / 3 倍にする（sep-CMA-ES）
    c1 = 2 / ((n + 1.3) ** 2 + mueff) * (n + 2) / 3
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff) * (n + 2) / 3)
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    return {"mean": np.zeros(n), "sigma": sigma0, "C": np.ones(n), "pc": np.zeros(n),
            "ps": np.zeros(n), "w": w, "mueff": mueff, "cs": cs, "cc": cc, "c1": c1,
            "cmu": cmu, "damps": damps, "gen": 0, "popsize": popsize}  # fmt: skip


def cma_ask(es: dict, rng) -> np.ndarray:
    z = rng.standard_normal((es["popsize"], len(es["mean"])))
    return es["mean"] + es["sigma"] * np.sqrt(es["C"]) * z


def cma_tell(es: dict, xs: np.ndarray, scores: np.ndarray) -> None:
    """scores は大きいほど良い。"""
    n = len(es["mean"])
    mu = len(es["w"])
    order = np.argsort(-scores)[:mu]
    old = es["mean"].copy()
    es["mean"] = (es["w"][:, None] * xs[order]).sum(0)
    y = (es["mean"] - old) / es["sigma"]
    cs, cc, mueff = es["cs"], es["cc"], es["mueff"]
    es["ps"] = (1 - cs) * es["ps"] + math.sqrt(cs * (2 - cs) * mueff) * y / np.sqrt(es["C"])
    es["gen"] += 1
    chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    ps_norm = np.linalg.norm(es["ps"]) / math.sqrt(1 - (1 - cs) ** (2 * es["gen"]))
    hsig = ps_norm < (1.4 + 2 / (n + 1)) * chi
    es["pc"] = (1 - cc) * es["pc"] + hsig * math.sqrt(cc * (2 - cc) * mueff) * y
    ys = (xs[order] - old) / es["sigma"]
    es["C"] = ((1 - es["c1"] - es["cmu"]) * es["C"] + es["c1"] * es["pc"] ** 2
               + es["cmu"] * (es["w"][:, None] * ys**2).sum(0))  # fmt: skip
    es["sigma"] *= math.exp(cs / es["damps"] * (np.linalg.norm(es["ps"]) / chi - 1))
    es["sigma"] = float(min(es["sigma"], 1.0))


# --- チェックポイント ---


def save(out: Path, state: dict, arrays: dict) -> None:
    from app.snoo_maneuver_policy import WeightPolicy

    tmp = out / "ckpt.tmp.npz"
    np.savez(tmp, **arrays)
    tmp.replace(out / "ckpt.npz")
    (out / "state.tmp.json").write_text(json.dumps(state, ensure_ascii=False, indent=0))
    (out / "state.tmp.json").replace(out / "state.json")
    WeightPolicy.from_theta(arrays["theta"]).save(out / "policy.npz")


def top_best(out: Path, theta: np.ndarray, margin: float, iteration: int) -> None:
    """どの学び方でも、確かめの家で固定の重みをいちばん上回ったものを --out/best.npz に写す
    （make eval-explore が読む）。上回った分は、その学び方の固定の重みの点と比べる。"""
    from app.snoo_maneuver_policy import WeightPolicy

    info = out.parent / "best.json"
    old = json.loads(info.read_text()) if info.exists() else None
    if old is not None and old["algo"] != out.name and old["margin"] >= margin:
        return
    WeightPolicy.from_theta(theta).save(out.parent / "best.npz")
    info.write_text(json.dumps({"algo": out.name, "iteration": iteration,
                                "margin": round(margin, 4)}))  # fmt: skip


def load(out: Path) -> tuple[dict, dict] | None:
    if not (out / "ckpt.npz").exists() or not (out / "state.json").exists():
        return None
    with np.load(out / "ckpt.npz", allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    return json.loads((out / "state.json").read_text()), arrays


# --- まとめ ---


def main() -> None:
    global _LOG
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--algo", choices=("ppo", "cmaes"), default="ppo",
                    help="既定は 2026-10-31 に確かめの家で良かった PPO")  # fmt: skip
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--minutes", type=float, default=40.0, help="今までの分も含めた時間の上限")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help="3 本まで")
    ap.add_argument("--episodes", type=int, default=24, help="PPO の 1 回に回す家の数")
    ap.add_argument("--popsize", type=int, default=10, help="CMA-ES の 1 世代の候補の数")
    ap.add_argument("--homes", type=int, default=3, help="CMA-ES の候補ごとに回す家の数")
    ap.add_argument("--sigma0", type=float, default=0.3, help="CMA-ES のはじめの広さ")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--lr-std", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.002)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-homes", type=int, default=6, help="確かめの家の数（× ふつう・白い壁）")
    ap.add_argument("--fresh", action="store_true", help="チェックポイントから続けない")
    args = ap.parse_args()
    args.workers = max(1, min(args.workers, MAX_WORKERS))
    out = args.out / args.algo
    out.mkdir(parents=True, exist_ok=True)
    _LOG = out / "train.log"
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    pool = make_pool(args.workers)
    try:
        run(args, out, pool)
    finally:
        pool.terminate()
        pool.join()


def run(args, out: Path, pool) -> None:
    from app.snoo_maneuver_policy import N_FEATURES, N_KNOBS, WeightPolicy

    n = N_KNOBS * (N_FEATURES + 1)
    rng = np.random.default_rng(0)
    got = None if args.fresh else load(out)
    if got is not None:
        state, arrays = got
        rng = np.random.default_rng([0, state["iteration"]])
        log(f"続ける：{state['iteration']} 回・{state['episodes']} 家・"
            f"{state['wall_s'] / 60:.1f} 分")  # fmt: skip
    else:
        state = {"algo": args.algo, "iteration": 0, "episodes": 0, "wall_s": 0.0, "best": None,
                 "baseline": None, "history": []}  # fmt: skip
        arrays = {"theta": np.zeros(n)}
        if args.algo == "ppo":
            arrays |= {"log_std": np.full(N_KNOBS, LOG_STD0), "m": np.zeros(n), "v": np.zeros(n),
                       "t": np.array(0), "vw": np.zeros(N_FEATURES + 2)}  # fmt: skip
        else:
            es = cma_init(n, args.sigma0, args.popsize)
            arrays |= {f"es_{k}": np.asarray(v) for k, v in es.items()}
        log(f"はじめから学ぶ（{args.algo}、子 {args.workers}）")
    t_start, wall0 = time.time(), state["wall_s"]

    def elapsed() -> float:
        return wall0 + time.time() - t_start

    if state["baseline"] is None:
        ev = evaluate(pool, None, args.eval_homes)
        if ev is None:
            return
        state["baseline"] = ev
        log(f"確かめの家で固定の重み：{json.dumps(ev, ensure_ascii=False)}")
    if args.algo == "ppo":
        st = {"theta": arrays["theta"], "log_std": arrays["log_std"], "adam": Adam(n, args.lr)}
        st["adam"].m, st["adam"].v, st["adam"].t = arrays["m"], arrays["v"], int(arrays["t"])
        vw = arrays["vw"] if state["iteration"] > 0 else None
    else:
        es = {k[3:]: (v.item() if v.ndim == 0 else v) for k, v in arrays.items()
              if k.startswith("es_")}  # fmt: skip
        es["gen"], es["popsize"] = int(es["gen"]), int(es["popsize"])

    def pack() -> dict:
        if args.algo == "ppo":
            return {"theta": st["theta"], "log_std": st["log_std"], "m": st["adam"].m,
                    "v": st["adam"].v, "t": np.array(st["adam"].t),
                    "vw": vw if vw is not None else np.zeros(N_FEATURES + 2)}  # fmt: skip
        return {"theta": es["mean"], **{f"es_{k}": np.asarray(v) for k, v in es.items()}}

    while not _stop:
        if elapsed() >= args.minutes * 60:
            log("時間の上限に達した")
            break
        it = state["iteration"]
        budget = budget_at(it)
        e0 = state["episodes"]
        t0 = time.time()
        if args.algo == "ppo":
            sigma = np.exp(st["log_std"])
            seeds = [TRAIN_SEED0 + e0 + k for k in range(args.episodes)]
            jobs = [(s, *home_kind(s), st["theta"], sigma, budget, "ppo") for s in seeds]
            rows = collect(pool, jobs)
            if rows is None:
                break
            F, Z, A, R, T = gae(rows, vw, args.gamma, args.lam)
            stats = ppo_update(st, F, Z, A, args, rng)
            vw = fit_value(F, T, R)
            stats["steps"] = len(F)
            theta = st["theta"]
        else:
            xs = cma_ask(es, rng)
            homes = [TRAIN_SEED0 + e0 + k for k in range(args.homes)]
            jobs = [(h, *home_kind(h), x, 0.0, budget, str(i))
                    for i, x in enumerate(xs) for h in homes]  # fmt: skip
            rows = collect(pool, jobs)
            if rows is None:
                break
            scores = np.array([np.mean([r["score"] for r in rows if r["tag"] == str(i)])
                               for i in range(len(xs))])  # fmt: skip
            cma_tell(es, xs, scores)
            stats = {"sigma": round(float(es["sigma"]), 4), "best": round(float(scores.max()), 3),
                     "mean_of_pop": round(float(scores.mean()), 3)}  # fmt: skip
            theta = es["mean"]
        state["iteration"] += 1
        state["episodes"] += len(rows) if args.algo == "ppo" else args.homes
        state["wall_s"] = elapsed()
        row = {
            "iteration": state["iteration"], "episodes": state["episodes"],
            "minutes": round(state["wall_s"] / 60, 2), "budget_s": budget,
            "score": round(float(np.mean([r["score"] for r in rows])), 3),
            "coverage": round(float(np.mean([r["coverage"] for r in rows])), 4),
            "lost": round(float(np.mean([r["lost"] for r in rows])), 2),
            "bumps": int(sum(r["bumps"] + r["agent_bumps"] for r in rows)),
            "rollout_s": round(time.time() - t0, 1), **stats,
        }  # fmt: skip
        if state["iteration"] % args.eval_every == 0 and not _stop:
            ev = evaluate(pool, theta, args.eval_homes)
            if ev is not None:
                row["val"] = ev
                best = state.get("best")
                if best is None or ev["all"]["score"] > best["all"]["score"]:
                    state["best"] = {**ev, "iteration": state["iteration"]}
                    WeightPolicy.from_theta(theta).save(out / "best.npz")
                    top_best(out, theta, ev["all"]["score"] - state["baseline"]["all"]["score"],
                             state["iteration"])  # fmt: skip
                state["wall_s"] = elapsed()
        state["history"].append(row)
        log(json.dumps(row, ensure_ascii=False))
        save(out, state, pack())
    state["wall_s"] = elapsed()
    save(out, state, pack())
    b = state.get("best")
    log(f"終わり：{state['iteration']} 回・{state['episodes']} 家・{state['wall_s'] / 60:.1f} 分。"
        f"固定 {state['baseline']['all']['score']}・いちばん良い "
        f"{b['all']['score'] if b else None}（{b['iteration'] if b else '-'} 回目）")  # fmt: skip


if __name__ == "__main__":
    main()
