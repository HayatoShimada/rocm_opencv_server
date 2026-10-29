"""次の 1 歩の選び方（app/snoo_explore_policy.py の MLP）を、作った家で PPO で学ぶ。

    make train-explore                 # data/snoo_policy/ の最後のチェックポイントから続ける
    .venv/bin/python -m scripts.snoo_explore_train --hours 4 --workers 18

- 回す（rollout）：子のプロセス（--workers）が、くじの違う家（app/snoo_homes.py）を作り、
  速いシミュレーター（app/snoo_gridsim.py）で 1 回ずつ探る。numpy だけで、GPU は見せない
  （HIP_VISIBLE_DEVICES=""。GPU を使うプロセスが 2 つ以上あると、何もしなくても 120 W ほど食う）
- 学ぶ（learner）：この親のプロセスだけが GPU（--device cuda、ROCm）で重みを直す。
  始める前に、ほかに GPU を使っているプロセスが無いか見る（あれば待って、ログに書く）
- はじめの --bc-iters 回は、今の決め方（でたらめ抜き）を軽くまねる（行動の複製）。まねすぎると
  選び方がほぼ決まってしまい（エントロピー 0.12）、PPO で試す幅が無くなった。そのあと PPO
- 報酬：新しく分かった床（m²）− 時間 − ぶつかった数（人と snoo は重く。app/snoo_explore_rollout.py）
- 止め方：SIGTERM・SIGINT を受けたら、チェックポイントを書いてから終わる（scripts/snoo_phase.sh が
  ロボットの時間に入るとき送る。120 秒以内に終わる）。次に動かすと、最後のチェックポイント
  （data/snoo_policy/ckpt_*.pt）から続ける。--hours は今までの分も含めた合計の時間
- 書き出し：data/snoo_policy/policy.npz（いまの重み）・best.npz（確かめ用の家でいちばん良かった
  もの）・train.log（チェックポイントごとの様子）・history.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import numpy as np

OUT = Path("data/snoo_policy")
# 家のくじ。学ぶ家・確かめる家（途中で良いものを選ぶ）・比べる家（scripts/snoo_explore_eval.py）
TRAIN_SEED0 = 0
VAL_SEED0 = 500_000
TEST_SEED0 = 900_000
KEEP_CKPTS = 3

_stop = False


def _on_signal(signum, _frame) -> None:
    global _stop
    _stop = True
    log(f"signal {signal.Signals(signum).name}: チェックポイントを書いて終わる")


_LOG: Path | None = None


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    if _LOG is not None:
        with _LOG.open("a") as f:
            f.write(line + "\n")


# --- 子のプロセス（torch を読まない） ---


def _init_worker() -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C は親が受けて片付ける
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    import cv2

    cv2.setNumThreads(1)  # 子どうしで CPU を取り合わない（cv2 は既定でコアの数だけ糸を立てる）


def _job(args: tuple) -> dict:
    from app.snoo_explore_rollout import run_episode

    seed, mode, weights, sample, budget_s = args
    r = run_episode(seed, mode, weights, sample=sample, budget_s=budget_s)
    e = r["episode"]
    return {
        "seed": seed,
        "covered": e.covered_m2,
        "coverage": e.coverage,
        "bumps": e.bumps,
        "agent_bumps": e.agent_bumps,
        "m2_5": e.m2_at(300),
        "m2_10": e.m2_at(600),
        "steps": [(s.x, s.index, s.logp, s.value, s.heuristic, s.reward) for s in r["steps"]],
    }


def make_pool(workers: int):
    """子のプロセスを作る。作る間だけ GPU を見えなくしておく（子は環境を受け継ぐ）。"""
    saved = {k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    saved |= {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    os.environ.update(
        HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )  # fmt: skip
    try:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return pool


def gpu_users() -> list[int]:
    """GPU（/dev/kfd）を開いているプロセスの番号（自分を除く）。"""
    root = Path("/sys/class/kfd/kfd/proc")
    if not root.is_dir():
        return []
    return [int(p.name) for p in root.iterdir() if p.name.isdigit() and int(p.name) != os.getpid()]


def collect(pool, jobs: list[tuple]) -> list[dict] | None:
    """jobs を回して結果を集める。途中で止めるよう言われたら None（その分は捨てる）。"""
    out = []
    for r in pool.imap_unordered(_job, jobs):
        out.append(r)
        if _stop:
            return None
    return out


# --- 学ぶ側（torch） ---


def build(torch, n_features: int, hidden: int):
    nn = torch.nn

    def mlp(nin):
        return nn.Sequential(
            nn.Linear(nin, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )  # fmt: skip

    pi, v = mlp(n_features), mlp(2 * n_features)
    with torch.no_grad():
        pi[4].weight.mul_(0.01)
        v[4].weight.mul_(0.1)
    return pi, v


def export(pi, v) -> dict[str, np.ndarray]:
    """torch の重みを app/snoo_explore_policy.Scorer の形（numpy）に。"""
    w = {}
    for pre, net in (("pi", pi), ("v", v)):
        for k, idx in ((1, 0), (2, 2), (3, 4)):
            w[f"{pre}_w{k}"] = net[idx].weight.detach().cpu().numpy().T.copy()
            w[f"{pre}_b{k}"] = net[idx].bias.detach().cpu().numpy().copy()
    return w


def batch_tensors(torch, episodes: list[dict], gamma: float, lam: float, device):
    """周ごとの候補を (B, K_MAX, F) に揃え、GAE で利得の見積もり（advantage）と目標の値を出す。"""
    from app.snoo_explore_policy import K_MAX, N_FEATURES

    xs, masks, acts, logps, advs, rets, heur = [], [], [], [], [], [], []
    for ep in episodes:
        steps = ep["steps"]
        if not steps:
            continue
        vals = np.array([s[3] for s in steps] + [0.0])
        rew = np.array([s[5] for s in steps])
        adv = np.zeros(len(steps))
        g = 0.0
        for t in reversed(range(len(steps))):
            delta = rew[t] + gamma * vals[t + 1] - vals[t]
            g = delta + gamma * lam * g
            adv[t] = g
        for t, (x, i, lp, _, h, _) in enumerate(steps):
            pad = np.zeros((K_MAX, N_FEATURES), np.float32)
            m = np.zeros(K_MAX, bool)
            pad[: len(x)], m[: len(x)] = x, True
            xs.append(pad)
            masks.append(m)
            acts.append(i)
            logps.append(lp)
            advs.append(adv[t])
            rets.append(adv[t] + vals[t])
            heur.append(h if h >= 0 else i)

    def T(a, dt=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=dt, device=device)

    return {
        "x": T(xs), "mask": T(masks, torch.bool), "act": T(acts, torch.long), "logp": T(logps),
        "adv": T(advs), "ret": T(rets), "heur": T(heur, torch.long),
    }  # fmt: skip


def forward(torch, pi, v, x, mask):
    logits = pi(x)[..., 0].masked_fill(~mask, -1e9)
    m = mask.float()[..., None]
    mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
    mx = x.masked_fill(~mask[..., None], -1e9).max(1).values
    value = v(torch.cat([mean, mx], -1))[..., 0]
    return torch.log_softmax(logits, -1), value


def ppo_update(torch, pi, v, opt, b, args) -> dict:
    n = len(b["act"])
    adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
    stats = {"kl": 0.0, "clip": 0.0, "ent": 0.0, "vloss": 0.0}
    count = 0
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=b["act"].device)
        for k in range(0, n, args.minibatch):
            idx = perm[k : k + args.minibatch]
            logp_all, value = forward(torch, pi, v, b["x"][idx], b["mask"][idx])
            logp = logp_all.gather(1, b["act"][idx, None])[:, 0]
            ratio = torch.exp(logp - b["logp"][idx])
            a = adv[idx]
            loss_pi = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            p = torch.exp(logp_all) * b["mask"][idx]
            ent = -(p * logp_all.masked_fill(~b["mask"][idx], 0)).sum(-1).mean()
            loss_v = ((value - b["ret"][idx]) ** 2).mean()
            loss = loss_pi + 0.5 * loss_v - args.ent * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*pi.parameters(), *v.parameters()], 0.5)
            opt.step()
            with torch.no_grad():
                stats["kl"] += float((b["logp"][idx] - logp).mean())
                stats["clip"] += float(((ratio - 1).abs() > args.clip).float().mean())
                stats["ent"] += float(ent)
                stats["vloss"] += float(loss_v)
            count += 1
        if _stop:
            break
    return {k: round(s / max(count, 1), 4) for k, s in stats.items()}


def bc_update(torch, pi, v, opt, b, epochs: int, minibatch: int) -> dict:
    """今の決め方の選んだ候補をまねる（交差エントロピー）。値も同時に合わせる。"""
    n = len(b["act"])
    acc = loss_sum = 0.0
    count = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=b["act"].device)
        for k in range(0, n, minibatch):
            idx = perm[k : k + minibatch]
            logp_all, value = forward(torch, pi, v, b["x"][idx], b["mask"][idx])
            ce = -logp_all.gather(1, b["heur"][idx, None])[:, 0].mean()
            loss = ce + 0.5 * ((value - b["ret"][idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                acc += float((logp_all.argmax(-1) == b["heur"][idx]).float().mean())
                loss_sum += float(ce)
            count += 1
        if _stop:
            break
    return {"ce": round(loss_sum / max(count, 1), 4), "acc": round(acc / max(count, 1), 4)}


def latest_ckpt(out: Path) -> Path | None:
    cks = sorted(out.glob("ckpt_*.pt"))
    return cks[-1] if cks else None


def save_ckpt(torch, out: Path, state: dict, pi, v, opt) -> Path:
    path = out / f"ckpt_{state['iteration']:05d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"state": state, "pi": pi.state_dict(), "v": v.state_dict(), "opt": opt.state_dict()}, tmp
    )
    tmp.replace(path)
    for old in sorted(out.glob("ckpt_*.pt"))[:-KEEP_CKPTS]:
        old.unlink()
    w = export(pi, v)
    np.savez(out / "policy.tmp.npz", **w)
    (out / "policy.tmp.npz").replace(out / "policy.npz")
    (out / "history.json").write_text(json.dumps(state["history"], indent=0))
    return path


def evaluate(pool, weights, seeds, budget_s) -> dict | None:
    rows = collect(pool, [(s, "policy", weights, False, budget_s) for s in seeds])
    if rows is None:
        return None
    return {
        "covered": round(float(np.mean([r["covered"] for r in rows])), 3),
        "coverage": round(float(np.mean([r["coverage"] for r in rows])), 4),
        "m2_10": round(float(np.mean([r["m2_10"] for r in rows])), 3),
        "bumps": int(sum(r["bumps"] for r in rows)),
        "agent_bumps": int(sum(r["agent_bumps"] for r in rows)),
    }


def main() -> None:
    global _LOG
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--hours", type=float, default=4.0, help="今までの分も含めた時間の上限")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--episodes", type=int, default=72, help="1 回に回す家の数")
    parser.add_argument("--budget-s", type=float, default=1200.0, help="1 回の探索（実機の秒）")
    parser.add_argument("--bc-iters", type=int, default=1)
    parser.add_argument("--bc-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-homes", type=int, default=36)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-wait-min", type=float, help="ほかが GPU を使っていたら、この分だけ待って CPU で学ぶ"
        "（既定は空くまで待つ）",
    )  # fmt: skip
    parser.add_argument("--max-iters", type=int, default=100_000)
    parser.add_argument("--fresh", action="store_true", help="チェックポイントから続けない")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _LOG = args.out / "train.log"
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    pool = make_pool(args.workers)
    try:
        run(args, pool)
    finally:
        pool.terminate()
        pool.join()


def run(args, pool) -> None:
    if args.device.startswith("cuda"):
        waited = 0.0
        while gpu_users() and not _stop:
            if args.gpu_wait_min is not None and waited >= args.gpu_wait_min * 60:
                log(f"GPU が {args.gpu_wait_min} 分空かないので CPU で学ぶ（{gpu_users()}）")
                args.device = "cpu"
                break
            log(f"GPU を使っているプロセスがある（{gpu_users()}）。30 秒待つ")
            time.sleep(30)
            waited += 30
        if _stop:
            return
    import torch

    from app.snoo_explore_policy import HIDDEN, N_FEATURES

    want_cuda = args.device.startswith("cuda")
    device = torch.device(args.device if not want_cuda or torch.cuda.is_available() else "cpu")
    if device.type != args.device.split(":")[0]:
        log(f"{args.device} が使えないので {device} で学ぶ")
    if device.type == "cpu":
        torch.set_num_threads(2)  # 子のプロセスに CPU を空けておく
    torch.manual_seed(0)
    pi, v = build(torch, N_FEATURES, HIDDEN)
    pi, v = pi.to(device), v.to(device)
    opt = torch.optim.Adam([*pi.parameters(), *v.parameters()], lr=args.lr)
    state = {"iteration": 0, "episodes": 0, "wall_s": 0.0, "best": None, "history": []}
    ck = None if args.fresh else latest_ckpt(args.out)
    if ck is not None:
        d = torch.load(ck, map_location=device, weights_only=False)
        pi.load_state_dict(d["pi"])
        v.load_state_dict(d["v"])
        opt.load_state_dict(d["opt"])
        state = d["state"]
        log(f"{ck.name} から続ける（{state['iteration']} 回目・{state['episodes']} 家・"
            f"{state['wall_s'] / 3600:.2f} 時間）")  # fmt: skip
    else:
        log(f"はじめから学ぶ（{device}、子 {args.workers}、1 回 {args.episodes} 家）")
    val_seeds = list(range(VAL_SEED0, VAL_SEED0 + args.eval_homes))
    t_start, wall0 = time.time(), state["wall_s"]
    while not _stop and state["iteration"] < args.max_iters:
        if wall0 + time.time() - t_start >= args.hours * 3600:
            log("時間の上限に達した")
            break
        it = state["iteration"]
        bc = it < args.bc_iters
        weights = None if bc else export(pi, v)
        e0 = state["episodes"]
        mode = "heuristic" if bc else "policy"
        jobs = [(TRAIN_SEED0 + e0 + k, mode, weights, not bc, args.budget_s)
                for k in range(args.episodes)]  # fmt: skip
        t0 = time.time()
        eps = collect(pool, jobs)
        if eps is None:
            break
        t_roll = time.time() - t0
        b = batch_tensors(torch, eps, args.gamma, args.lam, device)
        stats = (
            bc_update(torch, pi, v, opt, b, args.bc_epochs, args.minibatch)
            if bc
            else ppo_update(torch, pi, v, opt, b, args)
        )
        state["iteration"] += 1
        state["episodes"] += len(eps)
        state["wall_s"] = wall0 + time.time() - t_start
        row = {
            "iteration": state["iteration"], "episodes": state["episodes"],
            "hours": round(state["wall_s"] / 3600, 3), "mode": "bc" if bc else "ppo",
            "covered": round(float(np.mean([e["covered"] for e in eps])), 3),
            "coverage": round(float(np.mean([e["coverage"] for e in eps])), 4),
            "bumps": int(sum(e["bumps"] for e in eps)),
            "agent_bumps": int(sum(e["agent_bumps"] for e in eps)),
            "steps": len(b["act"]), "rollout_s": round(t_roll, 1), **stats,
        }  # fmt: skip
        if state["iteration"] % args.eval_every == 0 and not bc and not _stop:
            ev = evaluate(pool, export(pi, v), val_seeds, args.budget_s)
            if ev is not None:
                row["val"] = ev
                best = state.get("best")
                if best is None or ev["covered"] > best["covered"]:
                    state["best"] = {**ev, "iteration": state["iteration"]}
                    np.savez(args.out / "best.npz", **export(pi, v))
        state["history"].append(row)
        log(json.dumps(row, ensure_ascii=False))
        path = save_ckpt(torch, args.out, state, pi, v, opt)
        if "val" in row:
            log(f"checkpoint {path.name}: 確かめ {row['val']}、いちばん良い {state['best']}")
    state["wall_s"] = wall0 + time.time() - t_start
    path = save_ckpt(torch, args.out, state, pi, v, opt)
    log(f"終わり：{path.name}（{state['iteration']} 回・{state['episodes']} 家・"
        f"{state['wall_s'] / 3600:.2f} 時間）")  # fmt: skip


if __name__ == "__main__":
    main()
