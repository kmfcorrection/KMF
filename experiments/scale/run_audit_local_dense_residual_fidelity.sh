#!/usr/bin/env bash
# Audit every available non-learned residual proxy on the local dense NS corpus.
set -euo pipefail

LOCAL_DATASET="${LOCAL_DATASET:-/Volumes/ExternalSSD/HILP/data/dense_ns2d_forced_n64_dt001}"
LOCAL_RESULTS="${LOCAL_RESULTS:-/Volumes/ExternalSSD/HILP/results/local_dense_residual_fidelity}"

python3 experiments/scale/audit_local_dense_residual_fidelity.py \
  --data-root "$LOCAL_DATASET" --out-dir "$LOCAL_RESULTS" \
  --cadences ${CADENCES:-"0.01 0.05 0.1"} \
  --n-cal-traj "${N_CAL_TRAJ:-16}" --n-test-traj "${N_TEST_TRAJ:-8}" \
  --transitions-per-traj "${TRANSITIONS_PER_TRAJ:-8}" --batch "${AUDIT_BATCH:-4}" \
  --fd-orders ${FD_ORDERS:-"2 4 6"} --weak-modes ${WEAK_MODES:-"8 16 32"} \
  --rk3-substeps ${RK3_SUBSTEPS:-"1 2 4 8 16"} \
  --les-resolutions ${LES_RESOLUTIONS:-"16 32"} --les-cs "${LES_CS:-0.17}" \
  --les-substeps "${LES_SUBSTEPS:-8}" --pod-ranks ${POD_RANKS:-"8 16 32"} \
  --pod-substeps "${POD_SUBSTEPS:-8}" --seed "${SEED:-20260909}"
