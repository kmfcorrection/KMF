#!/usr/bin/env bash
# ==============================================================================
# Master Runner: End-to-End Evaluation Across All Datasets x All Foundation Models
# ==============================================================================
# Evaluates:
#   Foundation Models: Poseidon (T, B), DPOT (Ti, S), MORPH (Ti, S)
#   PDE Systems:       NS-Gauss, FNS-KF, ACE, Wave-Gauss
#   Sample Size:       N = 50 held-out test trajectories (configurable)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
N_TEST_TRAJ="${N_TEST_TRAJ:-50}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-4}"

# Default matrices (can be overridden via environment variables)
# Note: Poseidon, DPOT, and MORPH are pre-trained on fluid dynamics (NS-Gauss, FNS-KF).
# For scalar transfer experiments, pass PDES="NS-Gauss FNS-KF ACE Wave-Gauss".
FMS=${FMS:-"poseidon dpot morph"}
PDES=${PDES:-"NS-Gauss FNS-KF ACE Wave-Gauss NS-SL Wave-Layer"}

echo "================================================================================"
echo "MASTER RUNNER: CROSS-FOUNDATION MODEL & CROSS-PDE BENCHMARK MATRIX"
echo "================================================================================"
echo "Data Directory:  ${DATA_ROOT}"
echo "Models under test: ${FMS}"
echo "PDEs under test:   ${PDES}"
echo "Test sample size:  ${N_TEST_TRAJ} trajectories per run"
echo "Device:            ${DEVICE}"
echo "================================================================================"

mkdir -p results/scale/cross_benchmark

for FM in $FMS; do
  case "$FM" in
    poseidon) SIZES="T B" ;;
    dpot)     SIZES="Ti S" ;;
    morph)    SIZES="Ti S" ;;
    local)    SIZES="FNO" ;;
    *)        SIZES="default" ;;
  esac

  for SIZE in $SIZES; do
    for PDE in $PDES; do
      DATA_PATH="${DATA_ROOT}/${PDE}.nc"

      if [[ ! -f "$DATA_PATH" ]]; then
        echo ">>> [SKIP] Dataset not found: ${DATA_PATH}. Run prepare_poseidon_iclr_datasets.sh first."
        continue
      fi

      echo ""
      echo "------------------------------------------------------------------------"
      echo ">>> RUNNING: FM=${FM} (${SIZE}) on PDE=${PDE} (N=${N_TEST_TRAJ})"
      echo "------------------------------------------------------------------------"

      LOG_FILE="results/scale/cross_benchmark/${FM}_${SIZE}_${PDE}.log"

      python3 experiments/scale/cross_fm_benchmark.py \
        --fm "${FM}" \
        --fm-size "${SIZE}" \
        --pde "${PDE}" \
        --fm-data-path "${DATA_PATH}" \
        --n-test-traj "${N_TEST_TRAJ}" \
        --steps "${STEPS}" \
        --device "${DEVICE}" \
        --coarse-dt 0.10 \
        2>&1 | tee "${LOG_FILE}"

    done
  done
done

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETED. Results saved in results/scale/cross_benchmark/"
echo "================================================================================"
