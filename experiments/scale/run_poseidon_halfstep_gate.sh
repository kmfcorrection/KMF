#!/usr/bin/env bash
# Test true FM-generated 0.05 half-steps before using them in an S4 residual.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
N_TRAJ=${N_TRAJ:-32}
OFFSET=${OFFSET:-0}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
SEED=${SEED:-0}
TAG=${TAG:-poseidon_halfstep_gate}
DIVERGENCE_WEIGHT=${DIVERGENCE_WEIGHT:-0}

python -u experiments/scale/poseidon_halfstep_gate.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --n-traj "$N_TRAJ" --offset "$OFFSET" --divergence-weight "$DIVERGENCE_WEIGHT" \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG"
