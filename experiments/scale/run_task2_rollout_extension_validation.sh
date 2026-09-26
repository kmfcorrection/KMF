#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
OUT_DIR="${OUT_DIR:-results/audit_experiments/task2_rollout_extension}"
FM_SIZE="${FM_SIZE:-B}"
GRID="${GRID:-128}"
CALIBRATION_POOL="${CALIBRATION_POOL:-20}"
N_CAL="${N_CAL:-5}"
N_TEST="${N_TEST:-50}"
N_SPLITS="${N_SPLITS:-20}"
STEPS="${STEPS:-4}"
DT="${DT:-0.10}"
STRIDE="${STRIDE:-2}"
SEED="${SEED:-20260924}"
ALPHA_GRID="${ALPHA_GRID:-0.01 0.02 0.05 0.10 0.20}"

mkdir -p "${OUT_DIR}"

DEVICE="cpu"
if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then DEVICE="cuda"; fi

python3 experiments/scale/validate_task2_rollout_extension.py \
  --data-root "${DATA_ROOT}" --out-dir "${OUT_DIR}" \
  --fm poseidon --fm-size "${FM_SIZE}" --grid "${GRID}" --device "${DEVICE}" \
  --dt "${DT}" --stride "${STRIDE}" --steps "${STEPS}" \
  --calibration-pool "${CALIBRATION_POOL}" --n-cal "${N_CAL}" \
  --n-test "${N_TEST}" --n-splits "${N_SPLITS}" \
  --alpha-grid ${ALPHA_GRID} --seed "${SEED}" \
  2>&1 | tee "${OUT_DIR}/task2_rollout_extension_validation.log"
