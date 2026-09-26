#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-data/assembled}"
OUT_DIR="${OUT_DIR:-results/audit_experiments/task2_correction_suite}"
mkdir -p "${OUT_DIR}"
DEVICE=cpu; if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then DEVICE=cuda; fi
python3 experiments/scale/benchmark_task2_correction_suite.py \
  --data-root "${DATA_ROOT}" --out-dir "${OUT_DIR}" --fm poseidon --fm-size "${FM_SIZE:-B}" --device "${DEVICE}" \
  --grid "${GRID:-128}" --pc-grid "${PC_GRID:-16}" --dt "${DT:-0.05}" --stride "${STRIDE:-1}" --steps "${STEPS:-4}" \
  --n-cal "${N_CAL:-5}" --n-test "${N_TEST:-20}" --step-grid ${STEP_GRID:-0.01 0.02 0.05 0.10 0.20} \
  --gn-damping "${GN_DAMPING:-1e-3}" --flow-substeps "${FLOW_SUBSTEPS:-16}" --methods ${METHODS:-raw projection kmf gradient gn physicscorrect solver} --seed "${SEED:-20260924}" \
  2>&1 | tee "${OUT_DIR}/task2_correction_suite.log"
