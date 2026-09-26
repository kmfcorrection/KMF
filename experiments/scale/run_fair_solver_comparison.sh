#!/usr/bin/env bash
# Fair solver comparison for the reviewer-facing latency audit.
#
# The Python runner writes two explicitly separated groups:
#   1. endpoint-conditioned refinement: raw FM, linear, KMF bridge, fusion
#   2. causal solver controls: RK2/RK4 from u0 only
# Calibration is recorded as offline cost and excluded from online latency.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
OUT_DIR="${OUT_DIR:-results/audit_experiments/fair_solver_comparison}"
FM_SIZE="${FM_SIZE:-B}"
GRID="${GRID:-128}"
N_CAL="${N_CAL:-20}"
N_TEST="${N_TEST:-50}"
SEED="${SEED:-20260925}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"
python3 -u experiments/scale/run_reviewer_closure_benchmarks.py \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --fm poseidon \
  --fm-size "${FM_SIZE}" \
  --grid "${GRID}" \
  --n-cal "${N_CAL}" \
  --n-test "${N_TEST}" \
  --solver-substeps 1 2 4 8 16 \
  --experiments pareto \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT_DIR}/fair_solver_comparison.log"

echo
echo "Fair solver comparison written to ${OUT_DIR}"
echo "Interpret endpoint-conditioned and causal-solver groups separately."
