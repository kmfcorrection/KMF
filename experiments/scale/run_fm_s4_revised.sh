#!/usr/bin/env bash
# Revised S4 HILP. It uses a block innovation precision, not a sampled covariance.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
LEAD=${LEAD:-1}
STEPS=${STEPS:-3}
N_CAL_TRAJ=${N_CAL_TRAJ:-16}
N_VAL_TRAJ=${N_VAL_TRAJ:-8}
N_TEST_TRAJ=${N_TEST_TRAJ:-16}
LAMBDA_GRID=${LAMBDA_GRID:-"0 0.03 0.1 0.3 1 3 10 30 100"}
MAP_STEPS=${MAP_STEPS:-30}
MAP_LR=${MAP_LR:-0.5}
PHYSICS_LIKELIHOOD=${PHYSICS_LIKELIHOOD:-midpoint_residual}
PHYSICS_SUBSTEPS=${PHYSICS_SUBSTEPS:-2}
TRANSPORT_JVP_TERMS=${TRANSPORT_JVP_TERMS:-1}
FIXED_RK3_SUBSTEPS=${FIXED_RK3_SUBSTEPS:-64}
FLOW_CFL=${FLOW_CFL:-0.5}
DIVERGENCE_WEIGHT=${DIVERGENCE_WEIGHT:-1.0}
INNOVATION_COVARIANCE=${INNOVATION_COVARIANCE:-isotropic}
INNOVATION_RANK=${INNOVATION_RANK:-32}
INNOVATION_SHRINK_GRID=${INNOVATION_SHRINK_GRID:-"0 0.1 0.25 0.5 0.75 1"}
PACKAGE_BATCH=${PACKAGE_BATCH:-1}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-revised_s4}
SEED=${SEED:-0}

EXTRA=()
if [[ "${ALLOW_RESIDUAL_GATE_FAIL:-0}" == "1" ]]; then EXTRA+=(--allow-residual-gate-fail); fi
if [[ "${VERIFY_PACKAGE_BATCH:-0}" == "1" ]]; then EXTRA+=(--verify-package-batch); fi

python -u experiments/scale/s4_fm_revised_hilp.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --lambda-grid $LAMBDA_GRID --map-steps "$MAP_STEPS" --map-lr "$MAP_LR" \
  --physics-likelihood "$PHYSICS_LIKELIHOOD" --physics-substeps "$PHYSICS_SUBSTEPS" \
  --transport-jvp-terms "$TRANSPORT_JVP_TERMS" \
  --physics-resolution 128 --fixed-rk3-substeps "$FIXED_RK3_SUBSTEPS" --flow-cfl "$FLOW_CFL" \
  --divergence-weight "$DIVERGENCE_WEIGHT" \
  --innovation-covariance "$INNOVATION_COVARIANCE" --innovation-rank "$INNOVATION_RANK" \
  --innovation-shrink-grid $INNOVATION_SHRINK_GRID \
  --package-batch "$PACKAGE_BATCH" \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG" "${EXTRA[@]}"
