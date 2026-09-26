#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scale/s4_fm_autoregressive_composite.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-16}" --n-val-traj "${N_VAL_TRAJ:-8}" --n-test-traj "${N_TEST_TRAJ:-16}" \
  --methods ${METHODS:-"isotropic block_innovation"} \
  --residual-scheme "${RESIDUAL_SCHEME:-trapezoid}" \
  --lambda-grid ${LAMBDA_GRID:-"0 1e-5 1e-4 1e-3 1e-2 0.1 1"} \
  --map-steps "${MAP_STEPS:-40}" --map-lr "${MAP_LR:-0.5}" \
  --lbfgs-history "${LBFGS_HISTORY:-20}" \
  --map-grad-tol "${MAP_GRAD_TOL:-1e-8}" --map-change-tol "${MAP_CHANGE_TOL:-1e-11}" \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --seed "${SEED:-20260908}" --tag "${TAG:-autoregressive_composite}"
