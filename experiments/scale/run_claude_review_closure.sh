#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${OUT_ROOT:-results/audit_experiments/claude_review_closure}"
DATA_ROOT="${DATA_ROOT:-data/assembled}"
FM_SIZE="${FM_SIZE:-B}"
mkdir -p "${OUT_ROOT}"
DEVICE=cpu
if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then DEVICE=cuda; fi

echo "[1/5] Matched-cost solver, bootstrap, long-horizon, wave, and latency audit"
OUT_DIR="${OUT_ROOT}/audit_suite" GRID=128 N_TRAJ=60 \
  bash experiments/scale/run_audit_suite.sh

echo "[2/5] Reviewer closure: solver Pareto, mismatch, calibration, cadence"
DATA_ROOT="${DATA_ROOT}" OUT_DIR="${OUT_ROOT}/reviewer_closure" \
  FM_SIZE="${FM_SIZE}" N_CAL=20 N_TEST=50 CALIBRATION_SPLITS=50 \
  EXPERIMENTS="pareto mismatch calibration cadence" \
  bash experiments/scale/run_reviewer_closure_suite.sh

echo "[3/5] Causal correction-method comparison"
DATA_ROOT="${DATA_ROOT}" OUT_DIR="${OUT_ROOT}/task2_correction_suite" \
  FM_SIZE="${FM_SIZE}" DT=0.05 STRIDE=1 STEPS=4 N_CAL=5 N_TEST=20 PC_GRID=16 \
  METHODS="raw projection kmf gradient gn physicscorrect solver" \
  bash experiments/scale/run_task2_correction_suite.sh

echo "[4/6] PDE-operator latency audit"
python3 experiments/scale/profile_operator_latency.py \
  --grid 128 --device "${DEVICE}" \
  --out "${OUT_ROOT}/operator_latency/operator_latency.json"

echo "[5/6] Master-matrix outlier and contract audit"
python3 experiments/scale/analyze_review_outliers.py \
  --results-root results/scale --out-dir "${OUT_ROOT}/master_outliers"

echo "[6/6] Closure manifest"
python3 - <<'PY' "${OUT_ROOT}/closure_manifest.json"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
out.write_text(json.dumps({
    "suite": "KMF Claude review closure",
    "components": [
        "matched-cost solver Pareto and paired uncertainty",
        "operator mismatch and calibration robustness",
        "long-horizon and wave-regime diagnostics",
        "synchronized operator/projection latency",
        "causal correction-method comparison",
        "master-matrix outlier and unavailable-contract audit",
    ],
}, indent=2) + "\n")
PY

echo "Review-closure suite complete: ${OUT_ROOT}"
