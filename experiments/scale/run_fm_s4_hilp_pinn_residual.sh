#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scale/s4_fm_hilp_pinn_residual.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-8}" --n-val-traj "${N_VAL_TRAJ:-4}" --n-test-traj "${N_TEST_TRAJ:-8}" \
  --methods ${METHODS:-"isotropic pushforward gauss_newton"} \
  --k "${K:-64}" --tau-rel "${TAU_REL:-0.01}" --oversample "${OVERSAMPLE:-16}" \
  --n-iter "${N_ITER:-2}" --chunk "${CHUNK:-16}" --n-tail "${N_TAIL:-16}" \
  --n-sketch "${N_SKETCH:-80}" --n-probe "${N_PROBE:-16}" \
  --lambda-grid ${LAMBDA_GRID:-"0 0.1 1 3 10 30 100 300 1000 3000"} \
  --map-steps "${MAP_STEPS:-30}" --map-lr "${MAP_LR:-0.5}" \
  --divergence-weight "${DIVERGENCE_WEIGHT:-1.0}" \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --seed "${SEED:-20260908}" --tag "${TAG:-original_hilp_pinn_residual}" ${EXTRA_FLAGS:-}

