#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scale/audit_poseidon_joint_midpoint_gradient.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-32}" --n-val-traj "${N_VAL_TRAJ:-8}" --n-test-traj "${N_TEST_TRAJ:-16}" \
  --lambda-grid ${LAMBDA_GRID:-"1e-4 1e-3 1e-2 0.1 1"} \
  --map-steps "${MAP_STEPS:-30}" --map-lr "${MAP_LR:-0.5}" \
  --divergence-weight "${DIVERGENCE_WEIGHT:-0}" \
  --residual-bias "${RESIDUAL_BIAS:-zero}" \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --seed "${SEED:-20260908}" --tag "${TAG:-joint_midpoint_gradient_alignment}"
