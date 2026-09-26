#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
OUT_DIR="${OUT_DIR:-results/audit_experiments/fm_endpoint_refinement}"
mkdir -p "${OUT_DIR}"

DEVICE=cpu
if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then DEVICE=cuda; fi

python3 experiments/scale/validate_fm_endpoint_refinement.py \
  --data-root "${DATA_ROOT}" --out-dir "${OUT_DIR}" \
  --fm poseidon --fm-size "${FM_SIZE:-B}" --fm-channels velocity \
  --grid "${GRID:-128}" --native-stride "${NATIVE_STRIDE:-2}" --horizon "${HORIZON:-4}" \
  --n-cal "${N_CAL:-20}" --n-test "${N_TEST:-50}" --offset "${OFFSET:-19760}" \
  --device "${DEVICE}" --seed "${SEED:-20260924}" \
  2>&1 | tee "${OUT_DIR}/fm_endpoint_refinement.log"
