#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scale/audit_poseidon_strong_residual_reference.py \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --n-traj "${N_TRAJ:-32}" --offset "${OFFSET:-0}" \
  --device "${DEVICE:-cuda}" --tag "${TAG:-strong_residual_reference}" \
  --out-dir "${OUT_DIR:-results/scale/poseidon_strong_residual_reference}"
