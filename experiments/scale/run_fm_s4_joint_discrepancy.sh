#!/usr/bin/env bash
# Separate S4 experiment for calibrated innovation--residual discrepancy HILP.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
LEAD=${LEAD:-1}
STEPS=${STEPS:-3}
N_CAL_TRAJ=${N_CAL_TRAJ:-32}
N_VAL_TRAJ=${N_VAL_TRAJ:-16}
N_TEST_TRAJ=${N_TEST_TRAJ:-32}
FEATURE_MODES=${FEATURE_MODES:-4}
OBSERVATION=${OBSERVATION:-midpoint_fourier}
FLOW_DEFECT_SUBSTEPS=${FLOW_DEFECT_SUBSTEPS:-32}
R_SHRINK=${R_SHRINK:-0.5}
R_RIDGE_REL=${R_RIDGE_REL:-1e-3}
# Mixture weight for the whole joint covariance.  CROSS_SHRINK_GRID remains
# accepted for compatibility with the first runner revision.
JOINT_SHRINK_GRID=${JOINT_SHRINK_GRID:-${CROSS_SHRINK_GRID:-"0 0.1 0.25 0.5 0.75 1"}}
GAIN_GRID=${GAIN_GRID:-"0 0.1 0.3 0.5 0.75 1"}
BLEND_GRID=${BLEND_GRID:-"0 0.1 0.25 0.5 0.75 1"}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
SEED=${SEED:-0}
TAG=${TAG:-joint_discrepancy}

python -u experiments/scale/s4_fm_joint_discrepancy_hilp.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --observation "$OBSERVATION" --flow-defect-substeps "$FLOW_DEFECT_SUBSTEPS" \
  --feature-modes "$FEATURE_MODES" --r-shrink "$R_SHRINK" --r-ridge-rel "$R_RIDGE_REL" \
  --cross-shrink-grid $JOINT_SHRINK_GRID --gain-grid $GAIN_GRID --blend-grid $BLEND_GRID \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG"
