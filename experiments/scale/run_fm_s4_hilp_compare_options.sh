#!/usr/bin/env bash
# S4 Physics Likelihood Head-to-Head: Option 1 vs Option 2
# Option 1: Instantaneous Differential Residual + Wiener Transfer Filter (Zero forward ODE solves)
# Option 2: Budgeted Coarse Predictor Likelihood (Fixed substeps=16)
# Baselines: raw_fm, raw_projected, pure_coarse_flow, pure_fine_flow (adaptive CFL 0.5)
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
COARSE_GRID=${COARSE_GRID:-32}
CFL=${CFL:-0.5}
GAMMA_GRID=${GAMMA_GRID:-"0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0"}
BETA_GRID=${BETA_GRID:-"0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0"}
WIENER_REG=${WIENER_REG:-1e-3}
EXTRA_FLAGS=${EXTRA_FLAGS:---project-incompressible}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_compare_options}
SEED=${SEED:-20260908}

python -u experiments/scale/s4_fm_hilp_compare_options.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --coarse-grid "$COARSE_GRID" --cfl "$CFL" \
  --gamma-grid $GAMMA_GRID --beta-grid $BETA_GRID \
  --wiener-reg "$WIENER_REG" \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG" \
  $EXTRA_FLAGS
