#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scale/audit_poseidon_midpoint_bridges.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --lead-steps "${LEAD:-1}" \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --n-traj "${N_TRAJ:-32}" --offset "${OFFSET:-0}" \
  --seed "${SEED:-20260908}" --tag "${TAG:-midpoint_bridge_audit}" \
  --out-dir "${OUT_DIR:-results/scale/poseidon_midpoint_bridge_audit}"
