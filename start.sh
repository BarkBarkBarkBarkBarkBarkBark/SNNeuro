#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Launch the full SNN Agent stack (pipeline + web UI).
#
# Usage:
#   ./start.sh                          # synthetic mode, 1 channel, port 8000
#   ./start.sh --mode synthetic --channels 4
#   ./start.sh --mode electrode
#   ./start.sh --mode lsl
#   ./start.sh --web-port 9000
#   ./start.sh --config data/a_best_config.json
#   ./start.sh --help
#
# The script starts two processes:
#   1. snn-serve  — asyncio pipeline server  (WS :8765, HTTP :8080)
#   2. daphne     — Django ASGI web server   (HTTP+WS :8000 by default)
#
# Browse to http://<host>:<web-port>/ from your machine.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODE="synthetic"
CHANNELS=1
WEB_PORT=8000
CONFIG=""
EXTRA_PIPELINE_ARGS=""
FREE_PORTS=0
WS_PORT=8765

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="$2";                 shift 2 ;;
    --channels)    CHANNELS="$2";            shift 2 ;;
    --web-port)    WEB_PORT="$2";            shift 2 ;;
    --ws-port)     WS_PORT="$2";             shift 2 ;;
    --config)      CONFIG="$2";              shift 2 ;;
    --free-ports)  FREE_PORTS=1;              shift 1 ;;
    --help|-h)
      sed -n '2,15p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $1  (use --help)"
      exit 1
      ;;
  esac
done

_port_pids() {
  local port="$1"
  lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
}

_free_port() {
  local port="$1"
  local pids
  pids="$(_port_pids "$port")"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "⚠  Port $port is already in use by PID(s): $pids"
  if [[ "$FREE_PORTS" != "1" ]]; then
    echo "   Re-run with: ./start.sh --free-ports (kills listeners on $port)"
    echo "   Or choose a different port: ./start.sh --web-port 8001"
    exit 1
  fi

  echo "   Freeing port $port …"
  # Try graceful, then force.
  kill $pids 2>/dev/null || true
  sleep 0.2
  pids="$(_port_pids "$port")"
  if [[ -n "$pids" ]]; then
    kill -9 $pids 2>/dev/null || true
    sleep 0.2
  fi
}

# ── Locate virtualenv ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
else
  echo "⚠  No .venv found at $VENV — using system Python"
fi

PYTHON="${VENV}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"

# ── Build snn-serve command ────────────────────────────────────────────────────
PIPELINE_CMD=("$PYTHON" -m snn_agent --mode "$MODE")

if [[ "$CHANNELS" -gt 1 ]]; then
  PIPELINE_CMD+=(--channels "$CHANNELS")
fi

if [[ -n "$CONFIG" ]]; then
  PIPELINE_CMD+=(--config "$CONFIG")
fi

# Ports (WS port must match the browser UI expectation)
PIPELINE_CMD+=(--ws-port "$WS_PORT")

# ── Build daphne command ───────────────────────────────────────────────────────
DAPHNE_CMD=(
  "$PYTHON" -m daphne
  -b 0.0.0.0
  -p "$WEB_PORT"
  snn_web.asgi:application
)

export DJANGO_SETTINGS_MODULE="snn_web.settings"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/src:${PYTHONPATH:-}"

# ── Resolve display address ────────────────────────────────────────────────────
# Try to find the LAN IP (exclude loopback); fall back to hostname.
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ -z "$HOST_IP" ]] && HOST_IP=$(hostname)

# ── Print banner ──────────────────────────────────────────────────────────────
echo ""
echo "⚡ SNN Agent"
echo "   Mode       : $MODE"
echo "   Channels   : $CHANNELS"
[[ -n "$CONFIG" ]] && echo "   Config     : $CONFIG"
echo ""
echo "   Pipeline   : ${PIPELINE_CMD[*]}"
echo "   Web server : ${DAPHNE_CMD[*]}"
echo ""
echo "   ──────────────────────────────────────────────────"
echo "   Browser →  http://${HOST_IP}:${WEB_PORT}/"
echo "   ──────────────────────────────────────────────────"
echo ""
echo "   VS Code Remote-SSH: forward ports ${WEB_PORT} and ${WS_PORT}, then open:"
echo "   http://localhost:${WEB_PORT}/"
echo ""
echo "   Press Ctrl+C to stop both processes."
echo ""

# ── Preflight: avoid confusing port-in-use failures ─────────────────────────
_free_port "$WEB_PORT"
_free_port "$WS_PORT"

# ── Start pipeline in background ──────────────────────────────────────────────
cd "$SCRIPT_DIR"
"${PIPELINE_CMD[@]}" &
PIPELINE_PID=$!

# Give the pipeline a moment to bind its WebSocket port
sleep 2

# ── Start web server (foreground) ─────────────────────────────────────────────
"${DAPHNE_CMD[@]}" &
DAPHNE_PID=$!

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down…"
  kill "$PIPELINE_PID" "$DAPHNE_PID" 2>/dev/null || true
  wait "$PIPELINE_PID" "$DAPHNE_PID" 2>/dev/null || true
  echo "Done."
}
trap cleanup INT TERM EXIT

# Wait for either process to exit
wait -n "$PIPELINE_PID" "$DAPHNE_PID" 2>/dev/null || wait
