#!/usr/bin/env bash
set -euo pipefail

python -u experiments/scale/phase1_physics_controls.py \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --resolutions ${RESOLUTIONS:-"8 4"} \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-8}" --n-test-traj "${N_TEST_TRAJ:-8}" \
  --cfl "${FLOW_CFL:-0.5}" --device "${DEVICE:-cuda}" \
  --seed "${SEED:-0}" --tag "${TAG:-phase1}"
