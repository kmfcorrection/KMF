#!/usr/bin/env bash
# Local large-J GMRES diagnostic; no PDE flow endpoint is computed.
set -euo pipefail
python3 -u experiments/scale/audit_local_gmres_transport.py \
  --data-root "${LOCAL_DATASET:-/Volumes/ExternalSSD/HILP/data/actual_like_ns2d_forced_n32_dt01_poc_2k}" \
  --out-dir "${LOCAL_RESULTS:-/Volumes/ExternalSSD/HILP/results/local_gmres_transport_stress}" \
  --n-cal-traj "${N_CAL_TRAJ:-32}" --n-test-traj "${N_TEST_TRAJ:-16}" \
  --transitions-per-traj "${TRANSITIONS_PER_TRAJ:-4}" --dt "${DT:-0.1}" \
  --error-strengths ${ERROR_STRENGTHS:-"0.1 0.25 0.5 1 2"} \
  --gmres-iters "${GMRES_ITERS:-48}" --gmres-tol "${GMRES_TOL:-1e-8}" \
  --seed "${SEED:-20260910}"
