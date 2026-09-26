#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scale/audit_poseidon_effective_discrepancy.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-32}" --n-val-traj "${N_VAL_TRAJ:-16}" \
  --shells "${SHELLS:-8}" --ridge-grid ${RIDGE_GRID:-"1e-5 1e-4 1e-3 1e-2 1e-1 1 10"} \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" --seed "${SEED:-20260908}" --tag "${TAG:-effective_discrepancy_gate}" \
  --out-dir "${OUT_DIR:-results/scale/poseidon_effective_discrepancy_audit}"
