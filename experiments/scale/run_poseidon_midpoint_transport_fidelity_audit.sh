#!/usr/bin/env bash
# Held-out residual-fidelity audit on actual Poseidon NS-Gauss trajectories.
set -euo pipefail

python -u experiments/scale/audit_poseidon_midpoint_transport.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-32}" --n-test-traj "${N_TEST_TRAJ:-32}" \
  --terms ${TRANSPORT_TERMS:-"0 1 2 4"} \
  --out-dir "${OUT_DIR:-results/scale/poseidon_midpoint_transport_fidelity}" \
  --seed "${SEED:-20260908}"
