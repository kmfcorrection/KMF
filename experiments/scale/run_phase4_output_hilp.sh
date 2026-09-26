#!/usr/bin/env bash
set -euo pipefail

# Phase 4 deliberately uses only output-space priors.  It excludes the
# direct-GN-as-output and latent-GN branches identified in the audit.
python -u experiments/scale/s4_fm_document_posterior.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels "${FM_CHANNELS:-velocity}" \
  --data-source poseidon-native --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --methods isotropic pushforward \
  --physics-likelihood "${PHYSICS_LIKELIHOOD:-midpoint_residual}" \
  --physics-resolution "${PHYSICS_RESOLUTION:-128}" --flow-cfl "${FLOW_CFL:-0.5}" \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-8}" --n-test-traj "${N_TEST_TRAJ:-8}" \
  --k "${K:-64}" --oversample "${OVERSAMPLE:-16}" --chunk "${CHUNK:-16}" \
  --n-tail "${N_TAIL:-16}" --n-sketch "${N_SKETCH:-80}" --n-probe "${N_PROBE:-16}" \
  --lambda-grid ${LAMBDA_GRID:-"0.01 0.03 0.1 0.3 1 3 10 30 100"} \
  --map-steps "${MAP_STEPS:-30}" --map-lr "${MAP_LR:-0.5}" \
  --trace-posterior --tag "${TAG:-phase4_output_hilp}"
