#!/usr/bin/env bash
# Discrete-flow consistency gate; no checkpoint download or FM inference needed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" experiments/scale/s4_azeban_flow_gate.py \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH to velocity_16.nc}" \
  --lead-steps "${LEAD:-3}" \
  --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-8}" \
  --n-test-traj "${N_TEST_TRAJ:-8}" \
  --cfl "${CFL:-0.5}" \
  --device "${DEVICE:-cuda}" \
  --tag "${TAG:-}"
