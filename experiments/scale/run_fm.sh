#!/usr/bin/env bash
# Everything check_fm can say about the pretrained foundation models, on a GPU.
#
# Nothing to set up: scOT is cloned on demand (a checkout, not an install --
# its pyproject pins transformers==4.29.2 and torch==2.0.1, while the package
# itself imports unchanged against current versions). Checkpoints come from the
# Hub on first use: T 83 MB, B 631 MB, L 2.5 GB.
#
# On a cluster whose compute nodes have no outbound network, do both fetches on
# the login node first and the batch job then runs offline:
#
#   PREFETCH=1 SIZES="T B L" ./experiments/scale/run_fm.sh
#
# Usage:
#   QUICK=1 ./experiments/scale/run_fm.sh          # ~10 min, calibrates the budget
#   ./experiments/scale/run_fm.sh                  # T and B, full grid
#   SIZES="T B L" ./experiments/scale/run_fm.sh    # add the 629M checkpoint
#
# Budgeting. Cost is 6*(k+oversample) + n_probe network passes per state per
# lead time -- 845 at k=128 -- times n_states, times the number of rows below.
# It does not depend on N, which is the entire point of the low-rank route.
# The first table printed reports `s/state` and whether torch.vmap survived the
# architecture (a sequential fallback is numerically identical and several
# times slower); multiply from there rather than trusting an estimate. Run
# QUICK=1 first and read those two numbers.
set -euo pipefail
cd "$(dirname "$0")/../.."

SIZES="${SIZES:-T B}"
K="${K:-128}"
NSTATES="${NSTATES:-16}"
CHUNK="${CHUNK:-32}"
# The cadence sweep, at full resolution for the two rows that carry the result
# and coarsened for the two controls -- their job is to bound a gap, not to
# trace a curve, and at 4x the cost per row that is where the budget goes.
LEADS="${LEADS:-0.25,0.5,1,2,4,8,16}"
LEADS_CTRL="${LEADS_CTRL:-0.25,1,16}"
DEVICE="${DEVICE:-}"

if [ -n "${QUICK:-}" ]; then
  SIZES="${SIZES:-T}"; K=32; NSTATES=6; LEADS="1,8"; LEADS_CTRL="1"
fi

DEV_ARG=""
[ -n "$DEVICE" ] && DEV_ARG="--device $DEVICE"

# Python block-buffers stdout when it is a file, so under sbatch every table
# appears only when its config's process exits -- a job that is working looks
# identical to one that is hung, for tens of minutes at a time. The bash `log`
# lines below are unaffected, which makes it worse: the log shows a config
# starting and then nothing.
export PYTHONUNBUFFERED=1

log() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }

# --- provisioning ----------------------------------------------------------
# All of this was once "setup steps in a comment", and a job that skipped it
# printed four "cannot load" stanzas and still exited 0. A run that could not
# load its model must not be indistinguishable from one that ran.

# Which interpreter. `python3` is not reliably the active environment's: a venv
# may provide only `python`, and a bare `python3` then falls through PATH to
# /usr/bin/python3 -- which is how an activated venv still produced "torch is
# not importable by /usr/bin/python3". Resolve it explicitly, in order of how
# direct the evidence is, and print the answer.
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
log "interpreter: $PYTHON"

if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "FATAL: torch is not importable by $PYTHON." >&2
  echo "  Neither python3 nor python on PATH could import it either." >&2
  echo "  Activate the environment that has torch, or name it explicitly:" >&2
  echo "    PYTHON=/path/to/venv/bin/python $0" >&2
  echo "  On a module-based cluster you may need e.g. module load anaconda3_gpu" >&2
  exit 1
fi
# Dependencies next, because the alternative failure modes are all expensive: a
# missing huggingface_hub surfaces halfway through a login-node prefetch, and a
# missing transformers surfaces inside a queued job after the model load -- by
# which point the allocation is spent. torch is deliberately not in the install
# line: none of these depends on it, and a cluster's torch is usually the one
# thing you must not let pip touch.
MISSING=$("$PYTHON" - <<'PY'
import importlib.util
need = ["numpy", "transformers", "huggingface_hub"]
print(" ".join(m for m in need if importlib.util.find_spec(m) is None))
PY
)
if [ -n "$MISSING" ]; then
  echo "FATAL: missing Python packages: $MISSING" >&2
  echo "  $PYTHON -m pip install $MISSING" >&2
  echo "  (none of these depends on torch, so your torch install is untouched)" >&2
  exit 1
fi
"$PYTHON" - <<'PY' || exit 1
import sys, torch, transformers
v = tuple(int(x) for x in torch.__version__.split(".")[:2])
if v < (2, 4):
    # probe_map batches torch.func.jvp under vmap with chunk_size; without it
    # every probe runs one at a time and the grid takes several times longer.
    sys.exit(f"FATAL: torch {torch.__version__} < 2.4 (see requirements-scale.txt)")
if int(transformers.__version__.split(".")[0]) >= 5:
    # scOT is written against transformers 4.29 and calls into internals that
    # v5 removed -- get_head_mask is the first, and it is not obviously the
    # last. adapters._patch_scot shims that one, but the whole pipeline was
    # verified on 4.57 and nothing was verified on v5, which is not a stack to
    # discover mid-job. Gate it here rather than six hours in.
    sys.exit(f"FATAL: transformers {transformers.__version__} is not supported.\n"
             f"  scOT needs the 4.x API:  {sys.executable} -m pip install "
             f"'transformers>=4.40,<5'")
PY

# HF_HOME needs a few GB free (T+B+L is ~3.2 GB) and must be writable. Not
# defaulted to /scratch/$USER: that path does not exist on every cluster -- on
# Delta scratch is /scratch/<project>/$USER -- and the failed mkdir is silent
# until the first download.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if ! mkdir -p "$HF_HOME" 2>/dev/null; then
  echo "FATAL: HF_HOME=$HF_HOME is not writable. Set it to a path that is," >&2
  echo "  e.g. \$(ls -d /scratch/*/\$USER 2>/dev/null | head -1)/hf" >&2
  exit 1
fi

SCOT_DIR="${SCOT_PATH:-$PWD/third_party/poseidon}"
if [ ! -f "$SCOT_DIR/scOT/model.py" ]; then
  log "scOT not found at $SCOT_DIR -- cloning"
  if ! git clone --depth 1 https://github.com/camlab-ethz/poseidon.git "$SCOT_DIR"; then
    echo "FATAL: could not clone scOT into $SCOT_DIR." >&2
    echo "  If this node has no outbound network, run PREFETCH=1 on a login" >&2
    echo "  node first, then resubmit." >&2
    exit 1
  fi
fi
export SCOT_PATH="$SCOT_DIR"

if [ -n "${PREFETCH:-}" ]; then
  log "prefetching checkpoints for: $SIZES  (HF_HOME=$HF_HOME)"
  # safetensors only: each repo also carries an equivalent pytorch_model.bin,
  # and pulling both doubles the transfer for nothing.
  "$PYTHON" - $SIZES <<'PY'
import sys
from huggingface_hub import snapshot_download
for s in sys.argv[1:]:
    p = snapshot_download(f"camlab-ethz/Poseidon-{s}",
                          allow_patterns=["*.json", "*.safetensors"])
    print(f"  Poseidon-{s} -> {p}")
PY
  log "Prefetch done. The batch job can now run with no network."
  exit 0
fi

FAILED=0
# check_fm exits 0 on a clean pass, 1 on a degenerate/failed-check *result*
# (which is a finding, not an error) and 2 when it could not load the model at
# all. Only the last is a run failure -- but it must be counted, not swallowed,
# so one unusable checkpoint neither aborts the grid nor passes for a result.
run() {
  "$PYTHON" experiments/scale/check_fm.py "$@" $DEV_ARG
  local rc=$?
  [ "$rc" -ge 2 ] && FAILED=$((FAILED + 1))
  return 0
}

for size in $SIZES; do
  BASE="--fm poseidon --fm-size $size --k $K --n-states $NSTATES --chunk $CHUNK"

  # The headline: on-manifold states, velocity channels only, cadence swept.
  # Poseidon takes lead time as an input, so this is the s5 table measured on a
  # pretrained model rather than on a resimulated dataset.
  log "poseidon-$size -- velocity channels, divergence-free states, cadence sweep"
  run $BASE --fm-channels velocity --states fluid --lead-times "$LEADS" --tag main

  # The one that decides whether s1 is alive. States from the model's own
  # rollout are the regime an autoregressive predictive distribution actually
  # conditions on, and the column to read is cross-state subspace overlap: if
  # it sits near 1 the leading geometry does not move, a single global
  # covariance reproduces it, and the captured-trace numbers are a fact about
  # dissipation rather than about per-state curvature.
  log "poseidon-$size -- rollout states (subspace overlap is the result here)"
  run $BASE --fm-channels velocity --states rollout --rollout-steps 4 \
      --lead-times "$LEADS" --tag main

  # Off-manifold control. Not a sanity check -- the gap to the rows above is
  # the measurement, and it is what says whether probing at randn (which is
  # what check_adapter used to do) was ever telling the truth.
  log "poseidon-$size -- gaussian control (off-manifold)"
  run $BASE --fm-channels velocity --states gaussian \
      --lead-times "$LEADS_CTRL" --tag ctrl

  # All four channels. On the incompressible corpus rho and p are constants, so
  # this deliberately includes directions the model never saw vary; the
  # difference from the velocity-only rows is the size of that mistake.
  log "poseidon-$size -- all 4 channels [rho,u,v,p]"
  run $BASE --fm-channels all --states fluid --lead-times "$LEADS_CTRL" --tag allch
done

# Documents why there is no OmniArch row: the obstruction is in what the
# project released, not here. Its exit 2 is the expected outcome, so it is run
# outside `run` and deliberately not counted as a failure.
log "omniarch"
"$PYTHON" experiments/scale/check_fm.py --fm omniarch $DEV_ARG || true

if [ "$FAILED" -gt 0 ]; then
  log "FAILED: $FAILED configuration(s) could not load a model. No result was"
  echo "  produced for them; see the 'cannot load' lines above." >&2
  exit 1
fi
log "Done. Results in results/scale/check_fm/<fm>_<size>_<states>_<tag>/"
