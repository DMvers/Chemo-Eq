#!/usr/bin/env bash
# Start the model server (scripts/serve_qwen.sh) in the background and wait until it answers. After every reboot.
#
#   bash llm/scripts/prelaunch.sh            start it and wait until it is ready
#   bash llm/scripts/prelaunch.sh status     is it answering
#   bash llm/scripts/prelaunch.sh stop       stop it (frees the GPU)
#
# Log: ~/.chemoeq/vllm.log, next to its socket ~/.chemoeq/vllm.sock.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$HOME/.chemoeq"; SOCK="$DIR/vllm.sock"; LOG="$DIR/vllm.log"; PIDF="$DIR/vllm.pid"
WAIT="${WAIT:-1800}"

served() { curl -s --max-time 5 --unix-socket "$SOCK" http://localhost/v1/models 2>/dev/null | grep -q '"id"'; }
alive() { [[ -f "$PIDF" ]] && kill -0 "$(<"$PIDF")" 2>/dev/null; }

case "${1:-start}" in
  start)
    if served; then echo "model server already running"; exit 0; fi
    mkdir -p "$DIR" && chmod 700 "$DIR"
    if ! alive; then
      echo "starting the model server; log: $LOG"
      setsid nohup bash "$here/scripts/serve_qwen.sh" >>"$LOG" 2>&1 </dev/null &
      echo $! >"$PIDF"
    fi
    t0=$SECONDS
    until served; do
      alive || { echo "the model server stopped; last lines of $LOG:"; tail -n 30 "$LOG"; exit 1; }
      (( SECONDS - t0 < WAIT )) || { echo "not ready after $WAIT s"; exit 1; }
      sleep 10
    done
    echo "ready after $((SECONDS - t0)) s" ;;
  status)
    if served; then echo "model server: ready"; elif alive; then echo "model server: loading"; else echo "model server: not running"; fi ;;
  stop)
    if alive; then pid=$(<"$PIDF"); kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"; echo "model server stopped"
    else echo "no model server running"; fi
    rm -f "$PIDF" ;;
  *) echo "usage: bash $0 [start|status|stop]"; exit 1 ;;
esac
