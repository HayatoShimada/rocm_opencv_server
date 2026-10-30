"""RoMa v2 で、ステーションのコマの組の密な対応を出す（GPU）。地図のサーバーの venv では回さない。

RoMa v2 は torch.hub で DINOv3 のコードを取る。重みは RoMa v2 の 1 ファイル
（DINOv3 の分も入っている）。
地図のサーバーの .venv（VGGT）とは torch の入れ方が違ってもよいように、別の venv に入れる:

    uv venv --python 3.12 ~/orca/roma-venv
    uv pip install --python ~/orca/roma-venv/bin/python \\
        --index-url https://stable.repo.amd.com/rocm/pytorch/whl-next/ \\
        --extra-index-url https://stable.repo.amd.com/rocm/core/whl-next/ \\
        --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \\
        torch==2.13.0+rocm10.0.0 torchvision==0.28.0+rocm10.0.0 \\
        amd-torch-device-gfx1100 amd-torchvision-device-gfx1100
    uv pip install --python ~/orca/roma-venv/bin/python einops pillow rich tqdm \\
        opencv-python-headless scipy numpy
    git clone https://github.com/Parskatt/romav2 ~/orca/ext/romav2-src
    uv pip install --python ~/orca/roma-venv/bin/python --no-deps -e ~/orca/ext/romav2-src

fused-local-corr（CUDA だけの部品）は入れない。無ければ RoMa v2 は torch だけの相関に切り替わる。
ROCm（RX 7900 XTX）では、torch の scaled_dot_product_attention の flash・mem_efficient が
DINOv3 の形で hipErrorInvalidValue になるので、math だけにする（app/snoo_roma.match_pairs）。

使い方（リポジトリの直下で。ジョブは scripts/snoo_roma_eval.py が書く）:

    ~/orca/roma-venv/bin/python -m scripts.snoo_roma_match --jobs jobs.json --out <dir> \\
        [--setting base] [--samples 5000]

jobs.json は [{"name": "004_005", "pairs": [["<a のコマ>", "<b のコマ>"], ...]}, ...]。
<out>/<name>.npz に、組ごとの ka_k・kb_k（画素）・cert_k（RoMa の重なりの確からしさ）と、
組ごとの秒数を残す。あるものは飛ばす。

--serve にすると、地図のサーバーから呼べる小さな HTTP の口（POST /match、同じ形の 1 ジョブ）になる。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.snoo_roma import SETTINGS, load_model, match_pairs


def run_job(model, job: dict, samples: int, batch: int = 1, rest: float = 1.0) -> dict:
    """1 ジョブのコマの組を batch 組ずつ通す。1 回ごとに、かかった時間の rest 倍だけ休む
    （家のブレーカーが落ちたので、GPU の平均の電力を半分ほどに抑える）。休む時間は seconds に
    入れない。
    """
    import torch

    pairs = [tuple(p) for p in job["pairs"]]
    res, secs = [], []
    for i in range(0, len(pairs), batch):
        torch.cuda.synchronize()
        t = time.perf_counter()
        res += match_pairs(model, pairs[i : i + batch], samples)
        torch.cuda.synchronize()
        secs.append(time.perf_counter() - t)
        if rest > 0:
            time.sleep(rest * secs[-1])
    d: dict = {"seconds": np.array(sum(secs)), "n": np.array(len(res))}
    for k, r in enumerate(res):
        d[f"ka_{k}"], d[f"kb_{k}"], d[f"cert_{k}"] = r["ka"], r["kb"], r["cert"]
        d[f"overlap_{k}"] = np.array(r["overlap"])
        d[f"files_{k}"] = np.array(pairs[k])
    return d


def serve(model, port: int, samples: int) -> None:  # 地図のサーバーから 1 ジョブずつ
    """POST /match（本文はジョブ 1 つ）で対応を返す。答えは npz のバイト列。"""
    import io
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/match":
                self.send_error(404)
                return
            job = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            buf = io.BytesIO()
            np.savez(buf, **run_job(model, job, samples))
            body = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    HTTPServer(("127.0.0.1", port), H).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs")
    ap.add_argument("--out")
    ap.add_argument("--setting", default="base", choices=SETTINGS)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument(
        "--batch", type=int, default=1, help="1 回に通すコマの組（まとめると電力が上がる）"
    )
    ap.add_argument("--rest", type=float, default=1.0, help="1 回ごとに、かかった時間の何倍休むか")
    ap.add_argument("--serve", type=int, help="この番号の口で待つ（127.0.0.1）")
    args = ap.parse_args()
    t = time.perf_counter()
    model = load_model(args.setting)
    print(f"RoMa v2（{args.setting}）を読んだ：{time.perf_counter() - t:.1f} 秒", flush=True)
    if args.serve:
        serve(model, args.serve, args.samples)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(args.jobs).read_text())
    for job in jobs:
        dst = out / f"{job['name']}.npz"
        if dst.is_file():
            continue
        d = run_job(model, job, args.samples, args.batch, args.rest)
        np.savez_compressed(dst, **d)
        print(f"  {job['name']}: {int(d['n'])} 組 {float(d['seconds']):.2f} 秒", flush=True)


if __name__ == "__main__":
    main()
