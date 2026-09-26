#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
OUT_DIR="${OUT_DIR:-results/audit_experiments/fair_solver_all_pdes}"
FM_SIZE="${FM_SIZE:-B}"
GRID="${GRID:-128}"
N_CAL="${N_CAL:-20}"
N_TEST="${N_TEST:-50}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"
python3 -u experiments/scale/run_fair_solver_all_pdes.py \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --pdes FNS-KF ACE Wave-Gauss \
  --fm-size "${FM_SIZE}" \
  --grid "${GRID}" \
  --n-cal "${N_CAL}" \
  --n-test "${N_TEST}" \
  --solver-substeps 1 2 4 8 16 \
  --device "${DEVICE}" \
  2>&1 | tee "${OUT_DIR}/fair_solver_all_pdes.log"

echo "Fair all-PDE solver audit written to ${OUT_DIR}"
