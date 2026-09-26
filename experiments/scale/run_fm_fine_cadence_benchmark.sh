#!/usr/bin/env bash
# Runner for Fine-Cadence Super-Resolution Benchmark on Delta GPU
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_NAME="${MODEL_NAME:-poseidon_t}"
DATA_PATH="${DATA_PATH:-data/assembled/NS-Gauss.nc}"
N_TEST_TRAJ="${N_TEST_TRAJ:-8}"
COARSE_DT="${COARSE_DT:-0.10}"
SEED="${SEED:-20260908}"
TAG="${TAG:-fine_cadence_run}"

echo "==================================================================="
echo "Running Fine-Cadence Super-Resolution Benchmark: ${MODEL_NAME}"
echo "Data: ${DATA_PATH}"
echo "Trajectories: ${N_TEST_TRAJ}, Coarse dt: ${COARSE_DT}s"
echo "==================================================================="

FM="${FM:-poseidon}"
FM_SIZE="${FM_SIZE:-T}"

python3 experiments/scale/s4_fm_fine_cadence_benchmark.py \
  --fm "${FM}" \
  --fm-size "${FM_SIZE}" \
  --fm-channels velocity \
  --fm-data-path "${DATA_PATH}" \
  --n-test-traj "${N_TEST_TRAJ}" \
  --coarse-dt "${COARSE_DT}" \
  --seed "${SEED}" \
  --tag "${TAG}" \
  --device "${DEVICE:-cuda}"
