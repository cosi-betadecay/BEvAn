#!/usr/bin/env bash
# Launch the ablation study (ablations/main.py) as a detached background process.
#
# Nothing prints to the terminal: all output goes to
# logs/ablations_<timestamp>.log, and the shell gets the prompt back
# immediately. The process appears in top/htop as "bevan-ablations" rather than
# "python3" (top/htop show the executable's name, so the run goes through a
# named symlink to the venv interpreter).
#
# Extra arguments are passed through to ablations/main.py, e.g.:
#   scripts/run_ablations.sh --ablations no_ckd_order --datasets Max NCT
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROC_NAME="bevan-ablations"
VENV_BIN="$REPO_ROOT/.venv/bin"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/ablations_$(date +%Y-%m-%d_%H-%M-%S).log"

# ROOT/MEGAlib must be on the path before anything imports PyROOT. Checking
# ROOTSYS alone is not enough: macOS SIP strips DYLD_LIBRARY_PATH whenever an
# Apple binary (env, bash, nohup) is in the exec chain, while other exported
# variables survive — so re-source whenever the library path no longer covers
# ROOT's lib dir (its dylibs carry no rpaths and resolve only through it).
if [[ ":${DYLD_LIBRARY_PATH:-}:" != *":${ROOTSYS:-/nonexistent}/lib:"* ]]; then
    ENV_SCRIPT="${COSITOOLSDIR:+$COSITOOLSDIR/cositools-env.sh}"
    if [[ -z "$ENV_SCRIPT" || ! -f "$ENV_SCRIPT" ]]; then
        ENV_SCRIPT="$HOME/Documents/COSI/COSItools/cositools-env.sh"
    fi
    if [[ ! -f "$ENV_SCRIPT" ]]; then
        echo "error: DYLD_LIBRARY_PATH lacks ROOT's lib dir and cositools-env.sh was not found" >&2
        exit 1
    fi
    set +eu
    source "$ENV_SCRIPT"
    set -eu
fi

if [[ ! -x "$VENV_BIN/python3" ]]; then
    echo "error: $VENV_BIN/python3 not found — run 'uv sync' first" >&2
    exit 1
fi

# The named symlink lives inside .venv/bin so the interpreter still finds
# pyvenv.cfg next to it and the venv's site-packages resolve as usual.
ln -sf "$VENV_BIN/python3" "$VENV_BIN/$PROC_NAME"

mkdir -p "$LOG_DIR"
# The ablations discover datasets via data/ relative to the repo root.
cd "$REPO_ROOT"
# No nohup: /usr/bin/nohup is SIP-protected and would strip DYLD_LIBRARY_PATH
# again. trap '' HUP + disown + the redirects give the same detachment.
trap '' HUP
PYTHONUNBUFFERED=1 "$VENV_BIN/$PROC_NAME" ablations/main.py "$@" </dev/null >"$LOG_FILE" 2>&1 &
pid=$!
disown "$pid"

echo "$PROC_NAME started (pid $pid)"
echo "log: $LOG_FILE"
echo "follow with: tail -f $LOG_FILE"
