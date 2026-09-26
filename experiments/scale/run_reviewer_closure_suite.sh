#!/usr/bin/env bash
# Reviewer-closure GPU suite: E1 stable solver Pareto, E2 mismatch,
# E3 calibration robustness, E4 cadence scaling.  No PhysicsCorrect result is
# fabricated; its separate compatibility gate is below.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
OUT_DIR="${OUT_DIR:-results/audit_experiments/reviewer_closure}"
FM_SIZE="${FM_SIZE:-B}"
GRID="${GRID:-128}"
N_CAL="${N_CAL:-20}"
N_TEST="${N_TEST:-50}"
CALIBRATION_SPLITS="${CALIBRATION_SPLITS:-50}"
EXPERIMENTS="${EXPERIMENTS:-pareto mismatch calibration cadence}"
SEED="${SEED:-20260917}"

if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  DEVICE=cuda
else
  DEVICE=cpu
fi

mkdir -p "${OUT_DIR}"
python3 -u experiments/scale/run_reviewer_closure_benchmarks.py \
  --data-root "${DATA_ROOT}" --out-dir "${OUT_DIR}" \
  --fm poseidon --fm-size "${FM_SIZE}" --grid "${GRID}" \
  --n-cal "${N_CAL}" --n-test "${N_TEST}" \
  --calibration-splits "${CALIBRATION_SPLITS}" \
  --experiments ${EXPERIMENTS} --device "${DEVICE}" --seed "${SEED}" \
  2>&1 | tee "${OUT_DIR}/reviewer_closure.log"
