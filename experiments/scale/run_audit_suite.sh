#!/usr/bin/env bash
# ==============================================================================
# REVIEW AUDIT EXPERIMENT SUITE RUNNER
# ==============================================================================
# Executes all newly requested experiments from REVIEW_AUDIT_FOR_ANTIGRAVITY.md:
# 1. Primary-Contract Benchmark with Matched-Cost Baselines (Linear, PCHIP, Spline, RK4, Hermite)
# 2. Trajectory-Level Paired Bootstrap 95% Confidence Intervals & Wilcoxon Tests
# 3. Calibration Robustness & Sample Efficiency Ablation (M in {1, 2, 5, 10, 20})
# 4. Systematic Component Ablation & Physical Invariants (Divergence, Energy, Enstrophy)
# 5. Long-Horizon Rollouts (H = 20 Steps)
# 6. Hyperbolic Wave Regime Map (Dimensionless frequency sweep Omega = c*k*dt)
# 7. Synchronized Wall-Clock Component Latency Protocol
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-results/audit_experiments}"
mkdir -p "${OUT_DIR}"

GRID="${GRID:-128}"
N_TRAJ="${N_TRAJ:-60}"
SEED="${SEED:-20260914}"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

echo "================================================================================"
echo "RUNNING REVIEW AUDIT EXPERIMENT SUITE"
echo "Device:       ${DEVICE}"
echo "Grid:         ${GRID}x${GRID}"
echo "Trajectories: ${N_TRAJ}"
echo "Output:       ${OUT_DIR}/"
echo "================================================================================"

python3 experiments/scale/audit_experiments_suite.py \
    --out-dir "${OUT_DIR}" \
    --grid "${GRID}" \
    --n-trajectories "${N_TRAJ}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    2>&1 | tee "${OUT_DIR}/audit_experiments.log"

echo ""
echo "================================================================================"
echo "GENERATING AUDIT REPORTS & LATEX TABLES"
echo "================================================================================"

python3 experiments/scale/generate_audit_reports.py \
    --input-json "${OUT_DIR}/audit_experiments_results.json" \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/audit_reports.log"

echo ""
echo "================================================================================"
echo "AUDIT SUITE COMPLETE! All deliverables saved in ${OUT_DIR}/"
echo "================================================================================"
