#!/usr/bin/env bash
# Latent-state GN-HILP on the shared Poseidon NS-Gauss transfer corpus.
set -euo pipefail
cd "$(dirname "$0")/../.."

FM=${FM:-morph}
FM_SIZE=${FM_SIZE:-Ti}
LEAD=${LEAD:-1}
STEPS=${STEPS:-3}
K=${K:-64}
OVERSAMPLE=${OVERSAMPLE:-10}
CHUNK=${CHUNK:-16}
N_CAL_TRAJ=${N_CAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
LAMBDA_GRID=${LAMBDA_GRID:-"0.03 0.1 0.3 1 3 10 30 100"}
METHODS=${METHODS:-"latent_isotropic latent_gauss_newton latent_soft_gauss_newton"}
PHYSICS_RESOLUTION=${PHYSICS_RESOLUTION:-8}
FLOW_CFL=${FLOW_CFL:-0.5}
SOFT_INVERSE_ITERS=${SOFT_INVERSE_ITERS:-2}
SOFT_CG_ITERS=${SOFT_CG_ITERS:-24}
SOFT_SHIFT_REL=${SOFT_SHIFT_REL:-0.1}
FULL_GN_CG_ITERS=${FULL_GN_CG_ITERS:-24}
FULL_GN_CG_TOL=${FULL_GN_CG_TOL:-1e-3}
FULL_GN_DIAG_PROBES=${FULL_GN_DIAG_PROBES:-16}
TAU_REL=${TAU_REL:-0.01}
FULL_GN_TAU_GRID=${FULL_GN_TAU_GRID:-}
FULL_GN_CONVERGENCE_ONLY=${FULL_GN_CONVERGENCE_ONLY:-0}
MAP_STEPS=${MAP_STEPS:-30}
MAP_LR=${MAP_LR:-0.5}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH to Poseidon velocity_16.nc}
TAG=${TAG:-latent_gn}

extra=()
if [[ -n "$FULL_GN_TAU_GRID" ]]; then
  extra+=(--full-gn-tau-grid $FULL_GN_TAU_GRID)
fi
if [[ "$FULL_GN_CONVERGENCE_ONLY" == "1" ]]; then
  extra+=(--full-gn-convergence-only)
fi

python -u experiments/scale/s4_fm_latent_gn.py \
  --fm "$FM" --fm-size "$FM_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --k "$K" --oversample "$OVERSAMPLE" --chunk "$CHUNK" --tau-rel "$TAU_REL" --n-cal-traj "$N_CAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --lambda-grid $LAMBDA_GRID --methods $METHODS --physics-likelihood discrete_flow \
  --physics-resolution "$PHYSICS_RESOLUTION" --flow-cfl "$FLOW_CFL" \
  --soft-inverse-iters "$SOFT_INVERSE_ITERS" --soft-cg-iters "$SOFT_CG_ITERS" \
  --soft-shift-rel "$SOFT_SHIFT_REL" \
  --full-gn-cg-iters "$FULL_GN_CG_ITERS" --full-gn-cg-tol "$FULL_GN_CG_TOL" \
  --full-gn-diag-probes "$FULL_GN_DIAG_PROBES" \
  --map-steps "$MAP_STEPS" --map-lr "$MAP_LR" \
  --fm-data-path "$FM_DATA_PATH" --allow-native-transfer --tag "$TAG" "${extra[@]}"
