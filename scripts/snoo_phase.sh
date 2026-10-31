#!/usr/bin/env bash
# 手元で GPU を使って比べるターンと、実機で確かめるターンを切り替える（GPU を握るプロセスを 1 つにするため）。
#   scripts/snoo_phase.sh offline … 検出・地図・SLAM・見張りを止め、GPU を 1 つの比べる仕事
#                                   （RoMa・単眼の深度・SLAM の流し直しなど）だけに使う
#   scripts/snoo_phase.sh robot   … 見張りを起こす（比べる仕事が GPU を握っていたら起こさない）
#   scripts/snoo_phase.sh status  … どちらのターンか・GPU を握っているプロセス
# RX 7900 XTX は GPU を握ったプロセスが 2 つ以上あると、何もしていなくても 120 W のまま休めない。
# 学習（PPO の scripts.snoo_explore_train）は #73 で消したので、ここでは止めない。
set -u
cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f compose.yaml -f compose.dev.yaml)
# 手元で GPU を使う比べる仕事（止めはしない。終わるのを待つか、手で止める）
OFFLINE_PATTERN='scripts\.snoo_(roma_match|roma_eval|monodepth_eval|slam_replay|pose_bakeoff)'

gpu_holders() {
  # kfd に GPU を開いているプロセス（pid と名前）
  for d in /sys/class/kfd/kfd/proc/*/; do
    [ -d "$d" ] || continue
    pid=$(basename "$d")
    printf '%s %s\n' "$pid" "$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null | cut -c1-80)"
  done
}

offline_jobs() {
  pgrep -fa "$OFFLINE_PATTERN" | grep -v -e pgrep -e snoo_phase | cut -c1-100 || true
}

case "${1:-status}" in
  offline)
    systemctl --user stop snoo-ondemand snoo-slam snoo-map
    "${COMPOSE[@]}" stop snoo >/dev/null 2>&1
    echo "比べるターン：検出・地図・SLAM・見張りを止めた（画像の API の server は止めない）。GPU を使う比べる仕事は 1 つずつ"
    ;;
  robot)
    jobs=$(offline_jobs)
    if [ -n "$jobs" ]; then
      echo "比べる仕事が動いている。終わってから（または止めてから）もう一度:"
      echo "$jobs"
      exit 1
    fi
    systemctl --user start snoo-ondemand
    echo "実機のターン：見張りを起こした（Pi が見えたら検出・地図・SLAM を起こす）。make status で確かめる"
    ;;
  status)
    if systemctl --user is-active -q snoo-ondemand; then echo "実機のターン"; else echo "比べるターン（または全部止めている）"; fi
    offline_jobs
    echo "GPU を握っているプロセス:"
    gpu_holders
    ;;
  *)
    echo "使い方: $0 offline|robot|status"; exit 2 ;;
esac
