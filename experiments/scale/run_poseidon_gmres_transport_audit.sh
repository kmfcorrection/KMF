#!/usr/bin/env bash
# Actual-Poseidon counterpart of the local large-J GMRES stress diagnostic.
# It does not run a PDE flow; it only evaluates midpoint RHS residuals/JVPs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python experiments/scale/audit_poseidon_gmres_transport.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-8}" --n-test-traj "${N_TEST_TRAJ:-4}" \
  --gmres-iters "${GMRES_ITERS:-96}" --gmres-tol "${GMRES_TOL:-1e-6}" \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --seed "${SEED:-20260908}" \
  --out-dir "results/scale/poseidon_gmres_transport_audit/poseidon_${POSEIDON_SIZE:-T}_${TAG:-gate}"
