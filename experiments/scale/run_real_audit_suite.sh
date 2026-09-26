#!/usr/bin/env bash
# ==============================================================================
# REAL REVIEW AUDIT BENCHMARK SUITE RUNNER
# ==============================================================================
# Executes all 6 reviewer audit experiments on the genuine pipeline:
#   - Real Pretrained Foundation Models: Poseidon-B (and T, L)
#   - Real Downstream Datasets: NS-Gauss, FNS-KF, Wave-Gauss
#   - Honest Two-Endpoint Baselines: Linear vs. KMF Hermite vs. PCHIP-2pt
#   - True Zero-Leakage Calibration Ablation: Sweeping M in {1, 2, 5, 10, 20}
#   - True 4-Way Component Ablation: Strictly 0 future truth
#   - Real Autoregressive Rollout: H = 20 steps
#   - Wave-Gauss Dispersion Analysis: Real acoustic wave spectral decomposition
#   - Synchronized End-to-End Latency: CUDA-synchronized hardware profiling
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
OUT_DIR="${OUT_DIR:-results/audit_experiments/real_pipeline}"
mkdir -p "${OUT_DIR}"

FM_SIZE="${FM_SIZE:-B}"
GRID="${GRID:-128}"
N_CAL="${N_CAL:-20}"
N_TEST="${N_TEST:-50}"
SEED="${SEED:-20260914}"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

echo "================================================================================"
echo "RUNNING REAL REVIEW AUDIT BENCHMARK SUITE"
echo "Device:           ${DEVICE}"
echo "Foundation Model: Poseidon-${FM_SIZE}"
echo "Datasets:         ${DATA_ROOT}"
echo "Grid Resolution:  ${GRID}x${GRID}"
echo "Calibration Pool: ${N_CAL} trajectories"
echo "Held-out Test:    ${N_TEST} trajectories"
echo "Output Directory: ${OUT_DIR}/"
echo "================================================================================"

python3 experiments/scale/run_real_audit_benchmark.py \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --fm poseidon \
    --fm-size "${FM_SIZE}" \
    --grid "${GRID}" \
    --n-cal "${N_CAL}" \
    --n-test "${N_TEST}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    2>&1 | tee "${OUT_DIR}/real_audit_benchmark.log"

echo ""
echo "================================================================================"
echo "GENERATING REAL AUDIT LATEX TABLES & NARRATIVE"
echo "================================================================================"

python3 experiments/scale/generate_real_audit_reports.py \
    --input-json "${OUT_DIR}/real_audit_results.json" \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/real_audit_reports.log"

echo ""
echo "================================================================================"
echo "REAL AUDIT COMPLETE! All deliverables saved in ${OUT_DIR}/"
echo "================================================================================"
