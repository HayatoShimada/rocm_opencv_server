#!/usr/bin/env bash
# 85pi4 が起きている間だけ、snoo の検出（Docker）・地図のサーバー（snoo-map）・SLAM（snoo-slam）を動かす。
# Pi が見えない（ping が通らない）まま IDLE_S 秒たてば両方止め、見えたらすぐ起こす。
# 止めておけば、GPU も CPU も使わず、電気代がかからない（2026-10-29、消費電力がきつい）。
# 起こしてから検出が受け始めるまで 15〜25 秒、Pi は届かない間 tailscale に回るか待つだけ。
set -u
cd "$(dirname "$0")/.."
PI_ADDRS=${SNOO_PI_ADDRS:-"192.168.0.188 100.73.14.8"}
IDLE_S=${SNOO_IDLE_S:-300}
EVERY_S=${SNOO_CHECK_S:-10}
COMPOSE=(docker compose -f compose.yaml -f compose.dev.yaml)
last_seen=0
state=unknown

pi_up() {
  for a in $PI_ADDRS; do
    ping -c 1 -W 1 "$a" >/dev/null 2>&1 && return 0
  done
  return 1
}

while true; do
  now=$(date +%s)
  if pi_up; then
    last_seen=$now
    if [ "$state" != up ]; then
      echo "Pi が見えた：検出と地図のサーバーを起こす"
      "${COMPOSE[@]}" up -d --no-build snoo >/dev/null 2>&1 || echo "検出を起こせない"
      systemctl --user start snoo-map || echo "地図のサーバーを起こせない"
      systemctl --user start snoo-slam || echo "SLAM を起こせない"
      state=up
    fi
  elif [ "$state" != down ] && [ $((now - last_seen)) -ge "$IDLE_S" ]; then
    echo "Pi が ${IDLE_S} 秒見えない：検出と地図のサーバーを止める"
    "${COMPOSE[@]}" stop snoo >/dev/null 2>&1 || echo "検出を止められない"
    systemctl --user stop snoo-map || echo "地図のサーバーを止められない"
    systemctl --user stop snoo-slam || echo "SLAM を止められない"
    state=down
  fi
  sleep "$EVERY_S"
done
