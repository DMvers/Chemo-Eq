#!/usr/bin/env bash
# The PySR searches of the paper, on the same bins and splits as the LLM search.
#
#   bash pysr/run.sh <tracks>                       every search of the paper
#   bash pysr/run.sh <tracks> cei 1 "0 1 2"         CEI at x1 complexity penalty, splits 0-2
#
# Searches: cei and speed at x0.01, x0.1, x1, x10, x20, x30 and x100, and costheta at x1; splits 0-29 each.
# Step 1: run prepare.py to make the segments, splits and bin rows in pysr/data/.
# Step 2: one search.py per target, penalty and split, $JOBS (default 3) at a time.
# Results: pysr/runs/<target>_x<scale>_r<split>/ and its log, pysr/runs/<target>_x<scale>_r<split>.log.
# Done searches are skipped. Python: $PYTHON (default pysr/.venv/bin/python).
set -u
TRACKS=$(realpath "${1:?tracks folder}")
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PYTHON:-$HERE/.venv/bin/python}; JOBS=${JOBS:-3}
[ -f data/bins_r29.csv ] || "$PY" prepare.py "$TRACKS" data || exit 1
mkdir -p runs
{
  if [ -n "${2:-}" ]; then
    for t in $2; do for s in ${3:-1}; do for r in ${4:-$(seq 0 29)}; do echo "$t $s $r"; done; done; done
  else
    for r in $(seq 0 29); do
      for t in cei speed; do for s in 1 0_01 0_1 10 20 30 100; do echo "$t $s $r"; done; done
      echo "costheta 1 $r"
    done
  fi
} | while read -r t s r; do [ -f "runs/${t}_x${s}_r$r/best.py" ] || echo "$t $s $r"; done |
  xargs -P "$JOBS" -L 1 bash -c '
    if '"$PY"' search.py $0 $1 $2 > "runs/$0_x$1_r$2.log" 2>&1; then echo "done $0 x$1 r$2"
    else echo "FAILED $0 x$1 r$2 (runs/$0_x$1_r$2.log)"; fi'
