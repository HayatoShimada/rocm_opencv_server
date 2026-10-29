#!/bin/bash
# snoo の検出（app/snoo.py）が動いている間だけ GPU の時計を上げ（high）、止まったら auto に戻す。
#
# 検出はフレームの合間に GPU が時計を下げるため、60 fps の間隔だと 7.8 ms かかる（休まず回すと 4.3 ms）。
# 時計を上げたままだと何もしていないときの消費電力が増えるので、検出がフレームを処理している間だけ上げる。
# 検出は処理している間、SNOO_ACTIVE_FILE を 1 秒ごとに触る。それが STALE_S 秒より古ければ auto に戻す。
# 戻し忘れが無いように:
# - コンテナが止まる・Pi が送るのをやめる → ファイルが古くなり、STALE_S 秒で auto
# - このスクリプトが TERM・INT で止まる → trap で auto
# - それ以外で落ちる（kill -9 など） → systemd の ExecStopPost（deploy/snoo-gpu-perf.service）で auto
# - 起動し直す → amdgpu は auto から始まる
set -u

ACTIVE=${SNOO_ACTIVE_FILE:-/run/snoo/active}
STALE_S=${STALE_S:-3}
INTERVAL_S=${INTERVAL_S:-1}
LEVEL_FILE=${GPU_PERF_FILE:-}
if [ -z "$LEVEL_FILE" ]; then
    # AMD（vendor 0x1002）の GPU の設定ファイルを探す
    for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
        if [ "$(cat "$(dirname "$f")/vendor" 2>/dev/null)" = "0x1002" ]; then
            LEVEL_FILE=$f
            break
        fi
    done
fi
if [ -z "$LEVEL_FILE" ] || [ ! -w "$LEVEL_FILE" ]; then
    echo "GPU の時計の設定ファイル（${LEVEL_FILE:-見つからない}）に書けない" >&2
    exit 1
fi

set_level() {
    if [ "$(cat "$LEVEL_FILE")" != "$1" ]; then
        echo "$1" > "$LEVEL_FILE" && echo "GPU の時計: $1（$LEVEL_FILE）"
    fi
}
trap 'set_level auto' EXIT
trap 'exit 0' TERM INT

while true; do
    if [ -f "$ACTIVE" ] && [ $(( $(date +%s) - $(stat -c %Y "$ACTIVE") )) -le "$STALE_S" ]; then
        set_level high
    else
        set_level auto
    fi
    sleep "$INTERVAL_S" &
    wait $!
done
