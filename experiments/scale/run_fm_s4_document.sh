#!/usr/bin/env bash
# Document-faithful S4: calibrated isotropic vs pushforward vs Gauss--Newton.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
LEAD=${LEAD:-3}
STEPS=${STEPS:-3}
K=${K:-64}
OVERSAMPLE=${OVERSAMPLE:-16}
CHUNK=${CHUNK:-16}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
N_VAL_TRAJ=${N_VAL_TRAJ:-0}
N_TEST_TRAJ=${N_TEST_TRAJ:-8}
LAMBDA_GRID=${LAMBDA_GRID:-"0.03 0.1 0.3 1 3 10 30 100 300 1000"}
DATA_SOURCE=${DATA_SOURCE:-poseidon-native}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH to velocity_16.nc}
TAG=${TAG:-document}
METHODS=${METHODS:-"isotropic pushforward gauss_newton"}
TRACE_POSTERIOR=${TRACE_POSTERIOR:-0}
PHYSICS_LIKELIHOOD=${PHYSICS_LIKELIHOOD:-midpoint}
FLOW_CFL=${FLOW_CFL:-0.5}
PHYSICS_RESOLUTION=${PHYSICS_RESOLUTION:-128}
FIXED_RK3_SUBSTEPS=${FIXED_RK3_SUBSTEPS:-64}
N_SKETCH=${N_SKETCH:-}
N_PROBE=${N_PROBE:-8}
VERIFY_BATCHED=${VERIFY_BATCHED:-0}
SEED=${SEED:-0}

TRACE_ARGS=()
if [[ "$TRACE_POSTERIOR" == "1" ]]; then TRACE_ARGS+=(--trace-posterior); fi
if [[ "$VERIFY_BATCHED" == "1" ]]; then TRACE_ARGS+=(--verify-batched); fi
SKETCH_ARGS=(--n-probe "$N_PROBE")
if [[ -n "$N_SKETCH" ]]; then SKETCH_ARGS+=(--n-sketch "$N_SKETCH"); fi

# -u is important on a cluster: when stdout is piped to `tee`, ordinary Python
# buffers progress until the process exits, which makes a long curvature run
# look stalled even while the GPU is busy.
python -u experiments/scale/s4_fm_document_posterior.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --k "$K" --oversample "$OVERSAMPLE" --chunk "$CHUNK" --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  "${SKETCH_ARGS[@]}" \
  --lambda-grid $LAMBDA_GRID --methods $METHODS --data-source "$DATA_SOURCE" --fm-data-path "$FM_DATA_PATH" \
  --physics-likelihood "$PHYSICS_LIKELIHOOD" --physics-resolution "$PHYSICS_RESOLUTION" --flow-cfl "$FLOW_CFL" \
  --fixed-rk3-substeps "$FIXED_RK3_SUBSTEPS" \
  --seed "$SEED" \
  --tag "$TAG" "${TRACE_ARGS[@]}"
