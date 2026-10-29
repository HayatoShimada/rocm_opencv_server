#!/bin/bash
# SLAM（app/snoo_slam.py）の venv を作る。VGGT-SLAM 2.0 は torch 2.3（CUDA）と numpy 1 を前提にしているが、
# main の .venv と同じ ROCm 10 の torch 2.13 で動く（2026-10-29 に確かめた）。
# main の .venv には入れない（uv sync で消える。VGGT のフォークが地図のサーバーの VGGT とぶつかる）。
# 上流は python 3.11 で試しているので、3.12 の別の venv にする。
#
#   deploy/snoo-slam-venv.sh            # ~/orca/vggt-slam-venv と ~/orca/vggt-slam-src を作る
#
# 入れるもの（ライセンス）:
# - VGGT-SLAM 2.0（MIT-SPARK、BSD-2）・その VGGT のフォーク VGGT_SPARK（CC BY-NC 4.0。元の VGGT の
#   古いライセンスのまま）・SALAD のフォーク（GPL-3.0。ループを閉じるときだけ使う）
# - gtsam-develop（BSD）・open3d（MIT）・pytorch-lightning（Apache-2.0）・pytorch-metric-learning（MIT）・
#   faiss-cpu（MIT）・einops（MIT）・termcolor（MIT）・prettytable（BSD）・scipy（BSD）・matplotlib・
#   opencv-python-headless（Apache-2.0）・huggingface_hub・safetensors（Apache-2.0）
# - 重み：facebook/VGGT-1B（CC BY-NC 4.0。地図のサーバーと同じもの。~/.cache/huggingface にある）・
#   dino_salad.ckpt（SALAD、GPL-3.0）・DINOv2 ViT-B/14（Apache-2.0、torch.hub が取ってくる）
# CUDA だけの部品は無い（VGGT は F.scaled_dot_product_attention、GTSAM・open3d は CPU）。
set -euo pipefail
SRC=${SRC:-$HOME/orca/vggt-slam-src}
VENV=${VENV:-$HOME/orca/vggt-slam-venv}
mkdir -p "$SRC"
clone() {  # 名前 URL コミット
  [ -d "$SRC/$1" ] || git clone https://github.com/"$2".git "$SRC/$1"
  git -C "$SRC/$1" checkout -q "$3"
}
clone VGGT-SLAM MIT-SPARK/VGGT-SLAM 35327ac28b7d193df9ccc39ba6346052bb6f1207
clone salad Dominic101/salad 33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d
clone vggt MIT-SPARK/VGGT_SPARK 6e6e16107b88e8e76c751826af10d4295d87ecd2

uv venv --python 3.12 "$VENV"
# main の .venv と同じ AMD 配布の ROCm 10 の torch（RX 7900 XTX = gfx1100）
uv pip install --python "$VENV/bin/python" \
  "torch[device-gfx1100]==2.13.0+rocm10.0.0" "torchvision[device-gfx1100]==0.28.0+rocm10.0.0" \
  --index https://stable.repo.amd.com/rocm/pytorch/whl-next/ \
  --index https://stable.repo.amd.com/rocm/core/whl-next/ --index-strategy unsafe-best-match
# requirements.txt の torch==2.3.1・viser・gradio は入れない（viser の Viewer は app/snoo_slam.py が読まない）
uv pip install --python "$VENV/bin/python" gtsam-develop open3d einops safetensors huggingface_hub \
  opencv-python-headless scipy termcolor matplotlib tqdm pytorch-lightning pytorch-metric-learning \
  prettytable faiss-cpu
# 依存は上で入れたので --no-deps（vggt の numpy<2 の指定を無視する。numpy 2 で動く）
uv pip install --python "$VENV/bin/python" --no-deps \
  -e "$SRC/salad" -e "$SRC/vggt" -e "$SRC/VGGT-SLAM"
mkdir -p "$HOME/.cache/torch/hub/checkpoints"
CK="$HOME/.cache/torch/hub/checkpoints/dino_salad.ckpt"
[ -f "$CK" ] || curl -L -o "$CK" https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
"$VENV/bin/python" -c "import torch, gtsam, vggt_slam; print(torch.__version__, torch.cuda.is_available())"
