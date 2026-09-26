#!/usr/bin/env bash
# Dense latent-time, residual-only S4.  No RK/AZEBAN endpoint flow is used.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
LEAD=${LEAD:-1}
STEPS=${STEPS:-3}
DENSE_SUBSTEPS=${DENSE_SUBSTEPS:-2}
LATENT_BRIDGE=${LATENT_BRIDGE:-hermite_pde}
N_CAL_TRAJ=${N_CAL_TRAJ:-16}
N_VAL_TRAJ=${N_VAL_TRAJ:-8}
N_TEST_TRAJ=${N_TEST_TRAJ:-16}
LAMBDA_GRID=${LAMBDA_GRID:-"0 1e-6 3e-6 1e-5 3e-5 1e-4"}
MAP_STEPS=${MAP_STEPS:-25}
MAP_LR=${MAP_LR:-0.5}
DIVERGENCE_WEIGHT=${DIVERGENCE_WEIGHT:-1.0}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-dense_residual}
SEED=${SEED:-0}

python -u experiments/scale/s4_fm_dense_residual_window.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --dense-substeps "$DENSE_SUBSTEPS" \
  --latent-bridge "$LATENT_BRIDGE" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --lambda-grid $LAMBDA_GRID --map-steps "$MAP_STEPS" --map-lr "$MAP_LR" \
  --divergence-weight "$DIVERGENCE_WEIGHT" \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG"
