#!/usr/bin/env bash
# ==============================================================================
# Cross-PDE Mathematical Universality Suite (ICLR Submission)
# ==============================================================================
# Evaluates zero-ODE physical manifold recovery across 4 distinct mathematical classes:
#   1. Incompressible Fluid Mechanics:   NS-Gauss (Decaying Vortex Dynamics)
#   2. Forced Fluid Turbulence:           FNS-KF (Stationary Kolmogorov Cascade)
#   3. Stiff Reaction-Diffusion:          ACE (Allen-Cahn Phase Separation)
#   4. Hyperbolic Wave Propagation:       Wave-Gauss (Variable-Speed Acoustics)
#
# Evaluation Sample: N = 50 held-out trajectories per PDE
# Hardware: Delta GPU (CUDA)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
N_TEST_TRAJ="${N_TEST_TRAJ:-50}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-4}"
COARSE_DT="${COARSE_DT:-0.10}"

# 6 distinct physical PDE regimes across fluids, reaction-diffusion, and waves
PDES="NS-Gauss FNS-KF ACE Wave-Gauss NS-SL Wave-Layer"

echo "================================================================================"
echo "CROSS-PDE UNIVERSALITY SUITE: FLUIDS, REACTION-DIFFUSION & WAVES"
echo "================================================================================"
echo "Data Directory:    ${DATA_ROOT}"
echo "PDEs under test:   ${PDES}"
echo "Held-out sample:   ${N_TEST_TRAJ} trajectories per PDE"
echo "Coarse interval:   dt = ${COARSE_DT}s"
echo "Device:            ${DEVICE}"
echo "================================================================================"

mkdir -p results/scale/universality

for PDE in $PDES; do
  DATA_PATH="${DATA_ROOT}/${PDE}.nc"

  if [[ ! -f "$DATA_PATH" ]]; then
    echo ">>> [SKIP] Dataset not found: ${DATA_PATH}."
    continue
  fi

  echo ""
  echo "------------------------------------------------------------------------"
  echo ">>> EVALUATING PDE: ${PDE} (N=${N_TEST_TRAJ})"
  echo "------------------------------------------------------------------------"

  LOG_FILE="results/scale/universality/${PDE}_universality.log"

  python3 experiments/scale/cross_fm_benchmark.py \
    --fm poseidon \
    --fm-size T \
    --pde "${PDE}" \
    --fm-data-path "${DATA_PATH}" \
    --n-test-traj "${N_TEST_TRAJ}" \
    --steps "${STEPS}" \
    --device "${DEVICE}" \
    --coarse-dt "${COARSE_DT}" \
    2>&1 | tee "${LOG_FILE}"

done

echo ""
echo "================================================================================"
echo "UNIVERSALITY SUITE COMPLETE! Results saved in results/scale/universality/"
echo "================================================================================"
