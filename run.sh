#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run.sh — Launch helper for the SNN agent
#
#  Usage:
#    ./run.sh                  # start the agent (reads mode from config.py)
#    ./run.sh config           # open config.py in $EDITOR, then relaunch
#    ./run.sh lsl <ncs_path>   # start lsl_player + agent side-by-side
#    ./run.sh electrode        # start agent in electrode mode (+ test sender)
#    ./run.sh install          # create venv + install deps
#    ./run.sh help             # show this help
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
CONFIG="$DIR/config.py"

# ── Colors ────────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; R='\033[0;31m'; N='\033[0m'

info()  { echo -e "${G}▸${N} $*"; }
warn()  { echo -e "${Y}▸${N} $*"; }
err()   { echo -e "${R}✖${N} $*" >&2; }

ensure_venv() {
    if [[ ! -f "$PY" ]]; then
        warn "Virtual environment not found. Running install first…"
        do_install
    fi
}

# ── Commands ──────────────────────────────────────────────────────────────────

do_install() {
    info "Creating virtual environment…"
    python3 -m venv "$VENV"
    info "Installing dependencies…"
    "$PIP" install --upgrade pip -q
    "$PIP" install -r "$DIR/requirements.txt" -q
    info "Done. Key packages:"
    "$PIP" list --format=columns 2>/dev/null \
        | grep -Ei 'numpy|scipy|torch|snntorch|websockets|mne' || true
    echo ""
}

do_config() {
    local editor="${EDITOR:-${VISUAL:-nano}}"
    info "Opening config in ${C}${editor}${N}…"
    "$editor" "$CONFIG"
    echo ""
    info "Config saved. Launching agent…"
    echo ""
    do_agent
}

do_agent() {
    ensure_venv
    info "Starting SNN agent  (Ctrl+C to stop)"
    echo ""
    exec "$PY" "$DIR/server.py"
}

do_lsl() {
    local ncs_path="${1:-}"
    if [[ -z "$ncs_path" ]]; then
        err "Usage:  ./run.sh lsl <path-to-ncs-file-or-dir>"
        exit 1
    fi
    ensure_venv

    # Ensure config is set to LSL mode
    sed -i.bak 's/"mode":.*/"mode": "lsl",/' "$CONFIG" && rm -f "$CONFIG.bak"

    info "Starting LSL player + SNN agent (two processes)"
    info "  Player → ${C}${ncs_path}${N}"
    info "  Agent  → reads from LSL stream"
    echo ""

    "$PY" "$DIR/lsl_player.py" "$ncs_path" &
    PLAYER_PID=$!
    trap 'kill $PLAYER_PID 2>/dev/null; wait $PLAYER_PID 2>/dev/null' EXIT INT TERM

    sleep 1
    "$PY" "$DIR/server.py"
}

do_electrode() {
    ensure_venv

    # Ensure config is set to electrode mode
    sed -i.bak 's/"mode":.*/"mode": "electrode",/' "$CONFIG" && rm -f "$CONFIG.bak"

    info "Starting agent (electrode mode) + test sender (two processes)"
    echo ""

    "$PY" "$DIR/server.py" &
    AGENT_PID=$!
    trap 'kill $AGENT_PID 2>/dev/null; wait $AGENT_PID 2>/dev/null' EXIT INT TERM

    sleep 1
    "$PY" "$DIR/test_electrode.py" "$@"
}

do_synth() {
    ensure_venv
    info "Switching config to synthetic mode …"
    sed -i '' 's/"mode":.*$/"mode": "synthetic",/' "$DIR/config.py"
    info "Mode set to 'synthetic'"
    do_agent
}

do_evaluate() {
    ensure_venv
    local overrides="${1:-{}}"
    info "Running offline evaluation…"
    echo ""
    "$PY" "$DIR/evaluate.py" --overrides "$overrides"
}

do_optimize() {
    ensure_venv
    info "Launching hyperparameter optimization…"
    echo ""
    "$PY" "$DIR/optimize.py" "$@"
}

show_help() {
    echo ""
    echo -e "${G}SNN Agent — run script${N}"
    echo ""
    echo -e "${C}Usage:${N}"
    echo "  ./run.sh                    Start the agent (mode from config.py)"
    echo "  ./run.sh config             Edit config.py in \$EDITOR, then launch"
    echo "  ./run.sh lsl <ncs_path>     Start LSL player + agent together"
    echo "  ./run.sh synthetic          Run with SpikeInterface ground-truth data"
    echo "  ./run.sh electrode [args]   Start agent + synthetic test sender"
    echo "  ./run.sh evaluate [json]    Run offline pipeline evaluation"
    echo "  ./run.sh optimize [opts]    Hyperparameter optimization (Optuna)"
    echo "  ./run.sh install            Create .venv and install dependencies"
    echo "  ./run.sh help               Show all options"
    echo ""
    echo -e "${C}Examples:${N}"
    echo "  ./run.sh install"
    echo "  ./run.sh config                          # tweak, save, auto-launches"
    echo "  ./run.sh lsl raw_data/CSC285_0001.ncs    # replay a recording"
    echo "  ./run.sh synthetic                       # ground-truth benchmark"
    echo "  ./run.sh electrode --snr 3 --duration 30"
    echo "  ./run.sh evaluate '{\"l1_n_neurons\": 40}' # one-shot evaluation"
    echo "  ./run.sh optimize --n-trials 40          # run optimization"
    echo ""
    echo -e "${C}Typical workflow:${N}"
    echo -e "  1. ${Y}./run.sh install${N}         (first time only)"
    echo -e "  2. ${Y}./run.sh config${N}          (edit → save → agent starts)"
    echo -e "  3. Open ${C}http://localhost:8080${N} in browser"
    echo -e "  4. Ctrl+C to stop → tweak config → re-run"
    echo ""
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${1:-}" in
    install)            do_install ;;
    config|cfg|edit)    do_config ;;
    lsl)                shift; do_lsl "$@" ;;
    synthetic|synth)    do_synth ;;
    electrode|elec)     shift; do_electrode "$@" ;;
    evaluate|eval)      shift; do_evaluate "$@" ;;
    optimize|opt)       shift; do_optimize "$@" ;;
    help|-h|--help)     show_help ;;
    "")                 do_agent ;;
    *)                  err "Unknown command: $1"; show_help; exit 1 ;;
esac
