#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scale/audit_poseidon_true_midpoint_residual.py \
  --data "${FM_DATA_PATH:?set FM_DATA_PATH}" --n-traj "${N_TRAJ:-32}" \
  --transitions "${TRANSITIONS:-5}" --offset "${OFFSET:-0}" \
  --device "${DEVICE:-cuda}" \
  --tag "${TAG:-true_midpoint}" \
  --out-dir "${OUT_DIR:-results/scale/poseidon_true_midpoint_residual}"
