#!/usr/bin/env bash
# Staged, residual-aligned Poseidon protocol.  This is intentionally separate
# from every midpoint-residual runner.
#
# Stage A verifies that the released data is consistent with the adaptive
# AZEBAN-like SSP-RK3/CFL endpoint map.  Stage B screens *fixed-budget* RK3
# endpoint maps as deliberately imperfect, cheap physical observations.  Stage
# C runs the calibration-only discrepancy-covariance / block-innovation HILP
# experiment for one pre-specified candidate budget.
#
# Neither stage B nor C is allowed to substitute a solver trajectory into the
# FM rollout.  The RK3-only and FM/RK3 blend rows are printed as controls.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STAGES=${STAGES:-"replay screen hilp"}
POSEIDON_SIZE=${POSEIDON_SIZE:-T}
LEAD=${LEAD:-1}
STEPS=${STEPS:-3}
N_CAL_TRAJ=${N_CAL_TRAJ:-32}
N_VAL_TRAJ=${N_VAL_TRAJ:-16}
N_TEST_TRAJ=${N_TEST_TRAJ:-32}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH to the assembled NS-Gauss.nc file}
FLOW_CFL=${FLOW_CFL:-0.5}
# These fixed budgets must be declared before looking at the held-out results.
CANDIDATE_SUBSTEPS=${CANDIDATE_SUBSTEPS:-"4 8 16 32"}
HILP_SUBSTEPS=${HILP_SUBSTEPS:-32}
FEATURE_MODES=${FEATURE_MODES:-8}
R_SHRINK=${R_SHRINK:-0.5}
R_RIDGE_REL=${R_RIDGE_REL:-1e-3}
JOINT_SHRINK_GRID=${JOINT_SHRINK_GRID:-"0 0.1 0.25 0.5 0.75 1"}
GAIN_GRID=${GAIN_GRID:-"0 0.03 0.1 0.3 0.5 0.75 1"}
BLEND_GRID=${BLEND_GRID:-"0 0.1 0.25 0.5 0.75 1"}
K=${K:-64}
OVERSAMPLE=${OVERSAMPLE:-16}
CHUNK=${CHUNK:-32}
SEED=${SEED:-20260908}
TAG_PREFIX=${TAG_PREFIX:-discrete_flow_protocol}

contains_stage() {
  [[ " $STAGES " == *" $1 "* ]]
}

if contains_stage replay; then
  echo ">>> Stage A: adaptive AZEBAN replay consistency gate"
  FM_DATA_PATH="$FM_DATA_PATH" LEAD="$LEAD" STEPS="$STEPS" \
  N_CAL_TRAJ="$N_CAL_TRAJ" N_TEST_TRAJ="$N_TEST_TRAJ" CFL="$FLOW_CFL" \
  SEED="$SEED" TAG="${TAG_PREFIX}_adaptive_replay" \
    ./experiments/scale/run_s4_azeban_flow_gate.sh
fi

if contains_stage screen; then
  echo ">>> Stage B: validation-only fixed-budget flow candidate screen"
  # Candidate selection cannot see held-out trajectories.  The eventual HILP
  # run prints the RK3-only control on held-out data for the chosen budget.
  python -u experiments/scale/audit_poseidon_fixed_flow_candidates.py \
    --fm-data-path "$FM_DATA_PATH" --steps "$STEPS" \
    --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" \
    --candidate-substeps $CANDIDATE_SUBSTEPS \
    --tag "${TAG_PREFIX}_candidate_screen"
fi

if contains_stage hilp; then
  echo ">>> Stage C: calibrated block-innovation HILP with fixed RK3-${HILP_SUBSTEPS} defect"
  # The joint observation is r_t=x_{t+1}-Psi(x_t) for every corrected
  # interval.  Its calibration-only eta mean/R covariance and optional
  # innovation--eta cross covariance are fit before validation.  No future
  # truth is available to the correction at validation or held-out time.
  POSEIDON_SIZE="$POSEIDON_SIZE" LEAD="$LEAD" STEPS="$STEPS" \
  N_CAL_TRAJ="$N_CAL_TRAJ" N_VAL_TRAJ="$N_VAL_TRAJ" N_TEST_TRAJ="$N_TEST_TRAJ" \
  OBSERVATION="joint_fixed_rk3_flow_defect" FLOW_DEFECT_SUBSTEPS="$HILP_SUBSTEPS" \
  FEATURE_MODES="$FEATURE_MODES" R_SHRINK="$R_SHRINK" R_RIDGE_REL="$R_RIDGE_REL" \
  JOINT_SHRINK_GRID="$JOINT_SHRINK_GRID" GAIN_GRID="$GAIN_GRID" BLEND_GRID="$BLEND_GRID" \
  FM_DATA_PATH="$FM_DATA_PATH" SEED="$SEED" TAG="${TAG_PREFIX}_hilp_rk3_${HILP_SUBSTEPS}" \
    ./experiments/scale/run_fm_s4_joint_discrepancy.sh
fi

echo ">>> Discrete-flow protocol completed: stages=[$STAGES]"
