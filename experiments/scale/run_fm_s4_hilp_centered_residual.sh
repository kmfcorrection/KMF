#!/usr/bin/env bash
# Whole-timescale joint HILP with calibrated discretization bias subtraction (Option 1).
# Does not modify or overwrite any legacy scripts.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
K=${K:-64}
JOINT_RANK=${JOINT_RANK:-}
TANGENT_SAMPLES=${TANGENT_SAMPLES:-}
JOINT_FLOOR_REL=${JOINT_FLOOR_REL:-0.001}
OVERSAMPLE=${OVERSAMPLE:-16}
CHUNK=${CHUNK:-16}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
LAMBDA_GRID=${LAMBDA_GRID:-"0 0.01 0.03 0.1 0.3 1 3 10 30 100 300 1000 3000 10000 30000"}
METHODS=${METHODS:-"joint_pushforward"}
RESIDUAL_MODE=${RESIDUAL_MODE:-collocation}
RESIDUAL_BIAS=${RESIDUAL_BIAS:-calibration_mean}
DIVERGENCE_WEIGHT=${DIVERGENCE_WEIGHT:-10.0}
MAP_STEPS=${MAP_STEPS:-40}
MAP_LR=${MAP_LR:-0.5}
EXTRA_FLAGS=${EXTRA_FLAGS:---project-incompressible}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_joint_centered_residual}
SEED=${SEED:-20260908}

OPTIONAL=()
if [[ -n "$JOINT_RANK" ]]; then OPTIONAL+=(--joint-rank "$JOINT_RANK"); fi
if [[ -n "$TANGENT_SAMPLES" ]]; then OPTIONAL+=(--tangent-samples "$TANGENT_SAMPLES"); fi

python -u experiments/scale/s4_fm_hilp_centered_residual.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --k "$K" --oversample "$OVERSAMPLE" --chunk "$CHUNK" \
  "${OPTIONAL[@]}" --joint-floor-rel "$JOINT_FLOOR_REL" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --lambda-grid $LAMBDA_GRID --methods $METHODS \
  --residual-mode "$RESIDUAL_MODE" --residual-bias "$RESIDUAL_BIAS" \
  --divergence-weight "$DIVERGENCE_WEIGHT" \
  --map-steps "$MAP_STEPS" --map-lr "$MAP_LR" \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG" \
  $EXTRA_FLAGS
