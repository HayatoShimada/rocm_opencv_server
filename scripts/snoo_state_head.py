"""Clef の印（scripts/snoo_teacher.py）を、CLIP の埋め込みの上の小さな分類器に覚えさせる（蒸留）。

  uv run --group clip --group clef-local python -m scripts.snoo_state_head
質問（何をしているか・どこが見えているか・どちらを向いているか）ごとに線形の分類器を作り、
Clef の確率に近づける。検出の速い経路では snoo らしさと同じ埋め込みを使うので、足しても 1 ms 未満。
写真ごとに 5 つに分け、見ていない写真で Clef とどれだけ合うかを出してから、全部で学習し直す。
結果は data/snoo/state_head.npz に保存する。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.snoo import SnooScorer, to_tensor
from scripts.snoo_teacher import OUTPUT as TEACHER
from scripts.snoo_teacher import QUESTIONS, VERSION

OUTPUT = Path("data/snoo/state_head.npz")
FOLDS = 5


def fit(torch, x, y, steps: int = 300, weight_decay: float = 1e-3):
    """埋め込み x から Clef の確率 y への線形の分類器を、交差エントロピーで学習する。"""
    w = torch.zeros(x.shape[1], y.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros(y.shape[1], device=x.device, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=0.05, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        loss = -(y * torch.log_softmax(x @ w * 10 + b, dim=1)).sum(dim=1).mean()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    import torch

    labels = [v for v in json.loads(TEACHER.read_text()).values() if v["version"] == VERSION]
    scorer = SnooScorer("cuda", torch.float16)
    by_file = defaultdict(list)
    for v in labels:
        by_file[v["file"]].append(v)
    embeddings, rows = [], []
    for file, items in by_file.items():
        x = to_tensor(cv2.imread(file), "cuda", torch.float16)
        embeddings.extend(scorer.embed(x, [tuple(v["box"]) for v in items]))
        rows.extend(items)
    x = torch.stack(embeddings)
    files = sorted(by_file)
    fold_of = {f: i % FOLDS for i, f in enumerate(np.random.default_rng(0).permutation(files))}
    folds = torch.tensor([fold_of[r["file"]] for r in rows], device=x.device)

    saved = {"questions": np.array(list(QUESTIONS))}
    for q, spec in QUESTIONS.items():
        classes = list(spec["criteria"])
        y = torch.tensor([[r[q][c] for c in classes] for r in rows], device=x.device)
        y = y / y.sum(dim=1, keepdim=True)
        hits = 0
        for k in range(FOLDS):
            train, test = folds != k, folds == k
            w, b = fit(torch, x[train], y[train])
            hits += ((x[test] @ w * 10 + b).argmax(1) == y[test].argmax(1)).sum().item()
        # 多い答えをいつも返すだけのときの一致率（比べる基準）
        majority = (y.argmax(1) == y.argmax(1).mode().values).float().mean().item()
        print(f"{q}: Clef と一致 {hits / len(rows):.1%}（いつも多い答えなら {majority:.1%}）")
        w, b = fit(torch, x, y)
        saved[f"{q}_w"] = w.cpu().numpy()
        saved[f"{q}_b"] = b.cpu().numpy()
        saved[f"{q}_classes"] = np.array(classes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **saved)
    print(f"{args.output} に保存した（猫の箱 {len(rows)} 個）")


if __name__ == "__main__":
    main()
