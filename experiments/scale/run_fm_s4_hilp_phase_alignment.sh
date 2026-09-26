#!/usr/bin/env bash
# S4 Lie-Algebraic Advective Phase Alignment HILP.
# Directly targets on-manifold sub-pixel spatial vortex translation drift
# via continuous SE(2) Fourier phase shifting in ~0.81 seconds with zero forward simulation calls.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
METHODS=${METHODS:-"phase_align advective_phase_hybrid"}
GAMMA_GRID=${GAMMA_GRID:-"-2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 3.0 4.0"}
EXTRA_FLAGS=${EXTRA_FLAGS:---project-incompressible}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_phase_alignment}
SEED=${SEED:-20260908}

python -u experiments/scale/s4_fm_hilp_phase_alignment.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --methods $METHODS --gamma-grid $GAMMA_GRID \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG" \
  $EXTRA_FLAGS
