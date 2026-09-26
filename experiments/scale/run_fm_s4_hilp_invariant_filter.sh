#!/usr/bin/env bash
# Sequential Autoregressive Filtering with Integral Invariants (Pathway 2).
# Augments 4th-order Hermite-Simpson collocation with exact Kinetic Energy and Enstrophy balance penalties.
# Does not modify or overwrite any legacy scripts.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
K=${K:-64}
CHUNK=${CHUNK:-16}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
ORDER=${ORDER:-4}
RESIDUAL_BIAS=${RESIDUAL_BIAS:-calibration_mean}
ENERGY_WEIGHT=${ENERGY_WEIGHT:-1.0}
ENSTROPHY_WEIGHT=${ENSTROPHY_WEIGHT:-1.0}
DIVERGENCE_WEIGHT=${DIVERGENCE_WEIGHT:-10.0}
LAMBDA_GRID=${LAMBDA_GRID:-"0 0.1 0.3 1 3 10 30 100 300 1000 3000 10000"}
METHODS=${METHODS:-"isotropic"}
MAP_STEPS=${MAP_STEPS:-40}
MAP_LR=${MAP_LR:-0.5}
EXTRA_FLAGS=${EXTRA_FLAGS:---project-incompressible}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_invariant_filter}
SEED=${SEED:-20260908}

python -u experiments/scale/s4_fm_hilp_invariant_filter.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --k "$K" --chunk "$CHUNK" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --order "$ORDER" --residual-bias "$RESIDUAL_BIAS" \
  --energy-weight "$ENERGY_WEIGHT" --enstrophy-weight "$ENSTROPHY_WEIGHT" \
  --lambda-grid $LAMBDA_GRID --methods $METHODS \
  --divergence-weight "$DIVERGENCE_WEIGHT" \
  --map-steps "$MAP_STEPS" --map-lr "$MAP_LR" \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG" \
  $EXTRA_FLAGS
