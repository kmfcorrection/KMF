#!/usr/bin/env bash
# Explicit OOD S4 transfer: a fixed-step external FM on Poseidon NS-Gauss.
# This uses the same document posterior and AZEBAN endpoint likelihood as S4,
# but records the cadence convention rather than claiming native-FM cadence.
set -euo pipefail
cd "$(dirname "$0")/../.."

FM=${FM:-morph}
FM_SIZE=${FM_SIZE:-Ti}
LEAD=${LEAD:-1}
STEPS=${STEPS:-3}
K=${K:-32}
CHUNK=${CHUNK:-4}
N_CAL_TRAJ=${N_CAL_TRAJ:-2}
N_TEST_TRAJ=${N_TEST_TRAJ:-2}
LAMBDA_GRID=${LAMBDA_GRID:-"0.1 1 10 100"}
METHODS=${METHODS:-"isotropic pushforward gauss_newton"}
PHYSICS_LIKELIHOOD=${PHYSICS_LIKELIHOOD:-discrete_flow}
FLOW_CFL=${FLOW_CFL:-0.5}
PHYSICS_RESOLUTION=${PHYSICS_RESOLUTION:-128}
PHYSICS_ONLY_CALIBRATION_GATE=${PHYSICS_ONLY_CALIBRATION_GATE:-0}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH to Poseidon velocity_16.nc}
TAG=${TAG:-transfer}

gate_args=()
if [[ "$PHYSICS_ONLY_CALIBRATION_GATE" == "1" ]]; then
  gate_args+=(--physics-only-calibration-gate)
fi

python -u experiments/scale/s4_fm_document_posterior.py \
  --fm "$FM" --fm-size "$FM_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --k "$K" --chunk "$CHUNK" --n-cal-traj "$N_CAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --lambda-grid $LAMBDA_GRID --methods $METHODS \
  --data-source poseidon-native --fm-data-path "$FM_DATA_PATH" \
  --physics-likelihood "$PHYSICS_LIKELIHOOD" --flow-cfl "$FLOW_CFL" \
  --physics-resolution "$PHYSICS_RESOLUTION" \
  --allow-native-transfer --tag "$TAG" "${gate_args[@]}"
