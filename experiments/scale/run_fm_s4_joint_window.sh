#!/usr/bin/env bash
# Joint-window, residual-only S4.  Does not alter legacy one-step S4 scripts.
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
N_TEST_TRAJ=${N_TEST_TRAJ:-8}
LAMBDA_GRID=${LAMBDA_GRID:-"0.03 0.1 0.3 1 3 10 30 100"}
METHODS=${METHODS:-"isotropic joint_pushforward"}
MAP_STEPS=${MAP_STEPS:-30}
MAP_LR=${MAP_LR:-0.5}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-joint_window}
SEED=${SEED:-0}

OPTIONAL=()
if [[ -n "$JOINT_RANK" ]]; then OPTIONAL+=(--joint-rank "$JOINT_RANK"); fi
if [[ -n "$TANGENT_SAMPLES" ]]; then OPTIONAL+=(--tangent-samples "$TANGENT_SAMPLES"); fi

python -u experiments/scale/s4_fm_joint_window.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --k "$K" --oversample "$OVERSAMPLE" --chunk "$CHUNK" \
  "${OPTIONAL[@]}" --joint-floor-rel "$JOINT_FLOOR_REL" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --lambda-grid $LAMBDA_GRID --methods $METHODS --map-steps "$MAP_STEPS" --map-lr "$MAP_LR" \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG"
