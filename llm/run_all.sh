#!/usr/bin/env bash
# The LLM searches of the paper (experiments/paper.py). Start the model server first: bash llm/scripts/prelaunch.sh
#
#   bash llm/run_all.sh <tracks>                        every search of the paper
#   bash llm/run_all.sh <tracks> "cei" "0 1 2"          CEI at x1, splits 0-2
#
# Searches: cei, speed and costheta at x1; cei and speed at x0.01, x0.1, x10, x20, x30, x100;
# cei_seed2, cei_seed3, cei_nobio_object, cei_nobio_object_vague, cei_nobio_dicty_vague (Fig. 5); splits 0-29 each.
# Results: $CHEMOEQ_RUNS (default ~/chemoeq-runs)/paper/<variant>_r<split>/. $JOBS (default 3) searches share the
# one server. Finished searches are skipped; an interrupted one resumes from its checkpoint.
set -u
export CHEMOTRACK_RETRACKED=$(cd "${1:?tracks folder}" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=.venv/bin/python; RUNS=${CHEMOEQ_RUNS:-$HOME/chemoeq-runs}; JOBS=${JOBS:-3}
ALL="cei speed costheta cei_x0_01 speed_x0_01 cei_x0_1 speed_x0_1 cei_x10 speed_x10
        cei_x20 speed_x20 cei_x30 speed_x30 cei_x100 speed_x100
        cei_seed2 cei_seed3 cei_nobio_object cei_nobio_object_vague cei_nobio_dicty_vague"
mkdir -p "$RUNS/logs"
$PY experiments/paper.py cei prepare || exit 1          # the segments and bins, once, before the parallel searches
{
  if [ -n "${2:-}" ]; then
    for v in $2; do for r in ${3:-$(seq 0 29)}; do echo "$v $r"; done; done
  else
    for v in $ALL; do for r in $(seq 0 29); do echo "$v $r"; done; done
  fi
} | while read -r v r; do [ -f "$RUNS/paper/${v}_r$r/best.py" ] || echo "$v $r"; done |
  xargs -P "$JOBS" -L 1 bash -c '
    if '"$PY"' experiments/paper.py $0 search --repeat $1 > "'"$RUNS"'/logs/paper_$0_r$1.log" 2>&1; then echo "done $0 r$1"
    else echo "FAILED $0 r$1 ('"$RUNS"'/logs/paper_$0_r$1.log)"; fi'
