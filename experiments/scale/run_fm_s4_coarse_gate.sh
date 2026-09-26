#!/usr/bin/env bash
# Calibration-only selection gate for a frozen coarse AZEBAN likelihood.
# It deliberately never loads held-out trajectories.
set -euo pipefail
cd "$(dirname "$0")/../.."

RESOLUTIONS=${RESOLUTIONS:-"64 32 16"}
FM=${FM:-morph}
FM_SIZE=${FM_SIZE:-Ti}
LEAD=${LEAD:-1}
STEPS=${STEPS:-2}
K=${K:-32}
CHUNK=${CHUNK:-16}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH to Poseidon velocity_16.nc}
TAG=${TAG:-coarse_resolution_gate}

for physics_n in $RESOLUTIONS; do
  echo ">>> calibration gate: ${physics_n}x${physics_n} physics"
  FM="$FM" FM_SIZE="$FM_SIZE" LEAD="$LEAD" STEPS="$STEPS" K="$K" CHUNK="$CHUNK" \
  N_CAL_TRAJ="$N_CAL_TRAJ" PHYSICS_RESOLUTION="$physics_n" \
  PHYSICS_LIKELIHOOD=discrete_flow PHYSICS_ONLY_CALIBRATION_GATE=1 \
  FM_DATA_PATH="$FM_DATA_PATH" TAG="${TAG}_n${physics_n}" \
  ./experiments/scale/run_fm_s4_transfer.sh
done
