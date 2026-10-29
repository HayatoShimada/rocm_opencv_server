#!/usr/bin/env bash
# 学習のターンと、実機で確かめるターンを切り替える（GPU を握るプロセスを 1 つにするため）。
#   scripts/snoo_phase.sh train   … 検出・地図・SLAM・見張りを止め、GPU を学習だけに使う
#   scripts/snoo_phase.sh robot   … 学習を止め（チェックポイントを書いてから）、見張りを起こす
#   scripts/snoo_phase.sh status  … どちらのターンか・GPU を握っているプロセス
# RX 7900 XTX は GPU を握ったプロセスが 2 つ以上あると、何もしていなくても 120 W のまま休めない。
set -u
cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f compose.yaml -f compose.dev.yaml)
TRAIN_PATTERN='scripts.snoo_explore_train'

gpu_holders() {
  # kfd に GPU を開いているプロセス（pid と名前）
  for d in /sys/class/kfd/kfd/proc/*/; do
    [ -d "$d" ] || continue
    pid=$(basename "$d")
    printf '%s %s\n' "$pid" "$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null | cut -c1-80)"
  done
}

stop_training() {
  pids=$(pgrep -f "$TRAIN_PATTERN" | grep -v "^$$\$" || true)
  [ -z "$pids" ] && return 0
  echo "学習を止める（チェックポイントを書いてから終わる）: $pids"
  kill -TERM $pids
  for _ in $(seq 120); do
    pgrep -f "$TRAIN_PATTERN" | grep -qv "^$$\$" || return 0
    sleep 1
  done
  echo "学習が 120 秒で終わらない。手で確かめる"
  return 1
}

case "${1:-status}" in
  train)
    systemctl --user stop snoo-ondemand snoo-slam snoo-map
    "${COMPOSE[@]}" stop snoo >/dev/null 2>&1
    echo "学習のターン：検出・地図・SLAM・見張りを止めた。make train-explore で学習を始める（続きから）"
    ;;
  robot)
    stop_training || exit 1
    systemctl --user start snoo-ondemand
    echo "実機のターン：見張りを起こした（Pi が見えたら検出・地図・SLAM を起こす）。make status で確かめる"
    ;;
  status)
    if systemctl --user is-active -q snoo-ondemand; then echo "実機のターン"; else echo "学習のターン（または全部止めている）"; fi
    pgrep -fa "$TRAIN_PATTERN" | grep -v pgrep | cut -c1-100 || true
    echo "GPU を握っているプロセス:"
    gpu_holders
    ;;
  *)
    echo "使い方: $0 train|robot|status"; exit 2 ;;
esac
