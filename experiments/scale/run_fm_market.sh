#!/usr/bin/env bash
# Gate public PDE foundation/surrogate checkpoints beyond Poseidon.
#
# This script intentionally runs the cheap question first: can the checkpoint be
# loaded as a deterministic differentiable map, and does its Jacobian have
# state-dependent low-rank geometry? A model that fails here should not receive
# s1/s2/s3 GPU time.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ] && [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
fi
if [ -z "$PYTHON" ]; then
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import torch" 2>/dev/null; then
      PYTHON="$(command -v "$cand")"; break
    fi
  done
fi
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || echo python3)}"

SIZES_DPOT="${SIZES_DPOT:-Ti S M}"
K="${K:-64}"
NSTATES="${NSTATES:-8}"
CHUNK="${CHUNK:-16}"
LEADS="${LEADS:-1,4}"
DEVICE="${DEVICE:-}"
DEV_ARG=""
[ -n "$DEVICE" ] && DEV_ARG="--device $DEVICE"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

log() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }

log "interpreter: $PYTHON"
"$PYTHON" - <<'PY'
import importlib.util, sys, torch
need = ["numpy", "huggingface_hub", "einops"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("FATAL: missing Python packages: " + " ".join(missing))
if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 4):
    sys.exit(f"FATAL: torch {torch.__version__} < 2.4")
PY

DPOT_DIR="${DPOT_PATH:-$PWD/third_party/dpot}"
if [ ! -f "$DPOT_DIR/models/dpot.py" ]; then
  log "DPOT source not found at $DPOT_DIR -- cloning"
  git clone --depth 1 https://github.com/HaoZhongkai/DPOT.git "$DPOT_DIR"
fi
export DPOT_PATH="$DPOT_DIR"

if [ -n "${PREFETCH:-}" ]; then
  log "prefetching DPOT checkpoints: $SIZES_DPOT"
  "$PYTHON" - $SIZES_DPOT <<'PY'
import sys
from huggingface_hub import hf_hub_download
for size in sys.argv[1:]:
    key = {"T": "Ti", "Tiny": "Ti"}.get(size, size)
    path = hf_hub_download("hzk17/DPOT", f"model_{key}.pth")
    print(f"  DPOT-{key} -> {path}")
PY
  log "Prefetch done."
  exit 0
fi

FAILED=0
run() {
  "$PYTHON" experiments/scale/check_fm.py "$@" $DEV_ARG
  local rc=$?
  [ "$rc" -ge 2 ] && FAILED=$((FAILED + 1))
  return 0
}

for size in $SIZES_DPOT; do
  BASE="--fm dpot --fm-size $size --k $K --n-states $NSTATES --chunk $CHUNK"
  log "dpot-$size -- all channels, repeated-history gate"
  run $BASE --fm-channels all --states fluid --lead-times "$LEADS" --tag main
  log "dpot-$size -- gaussian control"
  run $BASE --fm-channels all --states gaussian --lead-times "1" --tag ctrl
done

log "walrus -- documented gate"
"$PYTHON" experiments/scale/check_fm.py --fm walrus $DEV_ARG || true

log "pdeformer -- documented gate"
"$PYTHON" experiments/scale/check_fm.py --fm pdeformer $DEV_ARG || true

log "the_well -- external single-dataset controls, if installed"
for family in FNO TFNO UNetConvNext; do
  run --fm the_well --fm-family "$family" --fm-dataset active_matter \
      --k 16 --n-states 4 --chunk 4 --lead-times 1 --tag active_matter || true
done

if [ "$FAILED" -gt 0 ]; then
  log "FAILED: $FAILED configuration(s) could not load a model."
  exit 1
fi
log "Done. Results in results/scale/check_fm/"
