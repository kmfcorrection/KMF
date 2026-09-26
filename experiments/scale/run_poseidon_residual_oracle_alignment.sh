#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scale/audit_poseidon_residual_oracle_alignment.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-8}" --calibration-offset "${CALIBRATION_OFFSET:-0}" \
  --n-traj "${N_TRAJ:-16}" --split-offset "${SPLIT_OFFSET:-0}" --flow-cfl "${FLOW_CFL:-0.5}" \
  --spectral-bands ${SPECTRAL_BANDS:-"0 2 4 8 16 32 64 128"} \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --seed "${SEED:-20260908}" --tag "${TAG:-residual_oracle_alignment}"
