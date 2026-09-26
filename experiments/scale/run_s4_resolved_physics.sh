#!/usr/bin/env bash
# Exact Gaussian S4 update using only coarse modes observed by AZEBAN physics.
set -euo pipefail

python -u experiments/scale/s4_resolved_physics_assimilation.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --data-source poseidon-native --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-16}" --n-val-traj "${N_VAL_TRAJ:-8}" --n-test-traj "${N_TEST_TRAJ:-16}" \
  --physics-resolution "${PHYSICS_RESOLUTION:-8}" --flow-cfl "${FLOW_CFL:-0.5}" \
  --k "${K:-64}" --oversample "${OVERSAMPLE:-16}" --chunk "${CHUNK:-16}" \
  --n-sketch "${N_SKETCH:-80}" --n-probe "${N_PROBE:-16}" --n-tail "${N_TAIL:-32}" \
  --methods ${METHODS:-"isotropic pushforward coarse_replace"} \
  --lambda-grid ${LAMBDA_GRID:-"0.03 0.1 0.3 1 3 10 30 100"} \
  --discrepancy-covariance "${DISCREPANCY_COVARIANCE:-diagonal}" \
  --discrepancy-shrink "${DISCREPANCY_SHRINK:-0.75}" \
  --discrepancy-ridge-rel "${DISCREPANCY_RIDGE_REL:-1e-4}" \
  --resolved-covariance-update "${RESOLVED_COVARIANCE_UPDATE:-frozen}" \
  --seed "${SEED:-0}" --tag "${TAG:-resolved_coarse_physics}"
