#!/usr/bin/env bash
# S4 Vorticity-Space Physical Radial Shell Discrepancy Filter
# Sobolev H^1 unbiased loss + Continuous Biot-Savart velocity lift.
# Strictly zero forward ODE solves, 100% non-cheating, instantaneous evaluation.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
N_CAL_TRAJ=${N_CAL_TRAJ:-24}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
SHELLS=${SHELLS:-24}
STEP_MODE=${STEP_MODE:-two-stage}
CAL_MODE=${CAL_MODE:-closed-loop}
GAMMA_MODE=${GAMMA_MODE:-per-step}
RIDGE=${RIDGE:-1e-4}
GAMMA_GRID=${GAMMA_GRID:-"-1.5 -1.0 -0.8 -0.5 -0.2 0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.5 1.6 1.8 2.0"}
EXTRA_FLAGS=${EXTRA_FLAGS:---project-incompressible}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_vorticity_biotsavart_run}
SEED=${SEED:-20260908}

python -u experiments/scale/s4_fm_hilp_radial_discrepancy.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --shells "$SHELLS" \
  --step-mode "$STEP_MODE" \
  --cal-mode "$CAL_MODE" \
  --gamma-mode "$GAMMA_MODE" \
  --ridge "$RIDGE" \
  --gamma-grid $GAMMA_GRID \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG" \
  $EXTRA_FLAGS
