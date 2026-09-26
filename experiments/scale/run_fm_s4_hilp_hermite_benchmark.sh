#!/usr/bin/env bash
# S4 Hermite Kinematic & HILP Physical Benchmark on Delta GPU (128x128 NS-Gauss)
# STRICTLY ZERO FORWARD ODE SOLVES across all methods
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
ALPHA_GRID=${ALPHA_GRID:-"0.05 0.10 0.15 0.20 0.25 0.30 0.40"}
MMSE_GAMMA_GRID=${MMSE_GAMMA_GRID:-"0.0 0.2 0.4 0.6 0.8 1.0"}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_hermite_benchmark_run}
SEED=${SEED:-20260908}

python -u experiments/scale/s4_fm_hilp_hermite_benchmark.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --alpha-grid $ALPHA_GRID \
  --mmse-gamma-grid $MMSE_GAMMA_GRID \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG"
