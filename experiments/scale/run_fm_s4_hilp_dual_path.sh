#!/usr/bin/env bash
# S4 Optimal 2D Spectral MMSE Deconvolution Benchmark:
# STRICTLY ZERO FORWARD ODE SOLVES, 100% NON-CHEATING CAUSAL PHYSICS
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
N_CAL_TRAJ=${N_CAL_TRAJ:-24}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
RIDGE=${RIDGE:-1e-3}
GAMMA_GRID=${GAMMA_GRID:-"0.0 0.2 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.5"}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_mmse_spectral_run}
SEED=${SEED:-20260908}

python -u experiments/scale/s4_fm_hilp_dual_path.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --ridge "$RIDGE" \
  --gamma-grid $GAMMA_GRID \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG"
