#!/usr/bin/env bash
# Frozen-split diagnostic protocol for joint innovation--residual HILP.
# It sweeps observation coverage only; every run uses the same trajectory
# split, fixed RK3-32 observation, and validation-only hyperparameter choices.
set -euo pipefail

MODE_COUNTS=${MODE_COUNTS:-"2 8 16"}
OBSERVATION=${OBSERVATION:-joint_fixed_rk3_flow_defect}
FLOW_DEFECT_SUBSTEPS=${FLOW_DEFECT_SUBSTEPS:-32}
TAG_PREFIX=${TAG_PREFIX:-joint_rk3_coverage}

for FEATURE_MODES in $MODE_COUNTS; do
  TAG="${TAG_PREFIX}_m${FEATURE_MODES}"
  export FEATURE_MODES OBSERVATION FLOW_DEFECT_SUBSTEPS TAG
  echo ">>> joint-discrepancy coverage: ${FEATURE_MODES} Fourier modes"
  ./experiments/scale/run_fm_s4_joint_discrepancy.sh
done
