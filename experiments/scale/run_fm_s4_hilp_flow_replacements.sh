#!/usr/bin/env bash
# S4 Flow Replacements Benchmark: Option A (Foias-Prodi Determining Modes) vs Option B (ETD Duhamel)
# Fully equipped with diagnostic markers: substeps, cosine, div RMS, energy error, enstrophy error.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
COARSE_GRID=${COARSE_GRID:-48}
CFL=${CFL:-0.7}
BETA_GRID=${BETA_GRID:-"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"}
ALPHA_GRID=${ALPHA_GRID:-"0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5"}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_flow_replacements_run}
SEED=${SEED:-20260908}

python -u experiments/scale/s4_fm_hilp_flow_replacements.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --coarse-grid "$COARSE_GRID" \
  --cfl "$CFL" \
  --beta-grid $BETA_GRID \
  --alpha-grid $ALPHA_GRID \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG"
