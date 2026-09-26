#!/usr/bin/env bash
# No endpoint solver: audit residual proxies only, against an offline oracle.
set -euo pipefail
read -r -a cadence_args <<< "${CADENCES:-0.01 0.05 0.1}"
read -r -a weak_mode_args <<< "${WEAK_MODES:-8 16 32}"
python3 experiments/scale/audit_local_residual_only.py \
  --data-root "${LOCAL_DATASET:-/Volumes/ExternalSSD/HILP/data/dense_ns2d_forced_n64_dt001}" \
  --out-dir "${LOCAL_RESULTS:-/Volumes/ExternalSSD/HILP/results/local_residual_only}" \
  --cadences "${cadence_args[@]}" \
  --n-cal-traj "${N_CAL_TRAJ:-32}" --n-test-traj "${N_TEST_TRAJ:-8}" \
  --transitions-per-traj "${TRANSITIONS_PER_TRAJ:-8}" \
  --weak-modes "${weak_mode_args[@]}"
