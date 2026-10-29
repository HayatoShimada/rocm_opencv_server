#!/usr/bin/env bash
# すべてのテストを、N 個の pytest に分けて同時に動かす（make test。分け方は tests/conftest.py）。
# 1 つの pytest で 4 分かかるもの（ほとんどが重い十数個）を、重さが揃うように分ける。
#   scripts/test_all.sh [N] [pytest の引数...]     N の既定は 6
# 終わった組から結果の行を出し、どれかが落ちたらその組の出力を全部出して 1 で終わる。
set -u
cd "$(dirname "$0")/.."
n=${1:-6}
shift || true
py=${PYTHON:-.venv/bin/python}
export MUJOCO_GL=${MUJOCO_GL:-egl}
dir=$(mktemp -d /tmp/snoo-test.XXXXXX)
start=$SECONDS
pids=()
for ((i = 0; i < n; i++)); do
  PYTEST_SHARD=$i/$n "$py" -m pytest -q -p no:cacheprovider --basetemp="$dir/tmp$i" "$@" \
    >"$dir/$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for ((i = 0; i < n; i++)); do
  if wait "${pids[$i]}"; then
    echo "[$((SECONDS - start)) 秒] 組 $i/$n: $(tail -1 "$dir/$i.log")"
  else
    rc=$?
    # 5 は「選んだテストが無い」（組が多すぎる）なので落ちたことにしない
    if [ "$rc" -eq 5 ]; then
      echo "[$((SECONDS - start)) 秒] 組 $i/$n: テストが無い"
      continue
    fi
    fail=1
    echo "[$((SECONDS - start)) 秒] 組 $i/$n: 落ちた（終了コード $rc）"
    cat "$dir/$i.log"
  fi
done
echo "合わせて $((SECONDS - start)) 秒（ログ: $dir）"
exit $fail
