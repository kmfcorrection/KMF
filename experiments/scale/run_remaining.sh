#!/usr/bin/env bash
# ==============================================================================
# RUN REMAINING EXPERIMENTS: DPOT, MORPH, & POLYMATHIC ON ACE & WAVE-GAUSS
# ==============================================================================
# Skips all 130 completed runs and only runs the remaining models on ACE and Wave-Gauss.
# Zero extra disk space needed (uses existing assembled ACE.nc and Wave-Gauss.nc).
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
N_TEST_TRAJ="${N_TEST_TRAJ:-50}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-4}"
CADENCES=${CADENCES:-"0.05 0.10 0.20 0.30 0.40"}
GRID=${GRID:-"128"}

OUT_DIR="results/scale/master_matrix_20260914_073113"
JSON_DIR="${OUT_DIR}/json"
mkdir -p "${OUT_DIR}" "${JSON_DIR}" "results/scale/cross_benchmark"

PDES="ACE Wave-Gauss"
DPOT_SIZES="Ti S"
MORPH_SIZES="Ti S"
POLYMATHIC_FAMILIES="TFNO UNetConvNext FNO"

echo "================================================================================"
echo "RUNNING REMAINING MODELS ON SCALAR PDES: ACE & WAVE-GAUSS"
echo "Models:   DPOT (Ti, S), MORPH (Ti, S), Polymathic (TFNO, UNetConvNext, FNO)"
echo "PDEs:     ${PDES}"
echo "Cadences: ${CADENCES}"
echo "Output:   ${JSON_DIR}/"
echo "================================================================================"

for CADENCE in $CADENCES; do
  for PDE in $PDES; do
    DATA_PATH="${DATA_ROOT}/${PDE}.nc"
    if [[ ! -f "$DATA_PATH" ]]; then
      echo "[SKIP] Dataset not found: ${DATA_PATH}"
      continue
    fi

    # 1. DPOT (Ti, S)
    for SIZE in $DPOT_SIZES; do
      JSON_FILE="${JSON_DIR}/dpot_${SIZE}_${PDE}_dt${CADENCE}_grid${GRID}_results.json"
      if [[ -f "$JSON_FILE" ]]; then
        echo "[EXISTS] Skipping DPOT-${SIZE} on ${PDE} (dt=${CADENCE}s)"
        continue
      fi
      echo ""
      echo ">>> [dt=${CADENCE}s | GRID=${GRID}] DPOT-${SIZE} on ${PDE}..."
      LOG_FILE="${OUT_DIR}/dpot_${SIZE}_${PDE}_dt${CADENCE}_grid${GRID}.log"
      python3 experiments/scale/cross_fm_benchmark.py \
        --fm dpot --fm-size "${SIZE}" --pde "${PDE}" \
        --fm-data-path "${DATA_PATH}" \
        --n-test-traj "${N_TEST_TRAJ}" \
        --steps "${STEPS}" \
        --coarse-dt "${CADENCE}" \
        --grid "${GRID}" \
        --device "${DEVICE}" \
        --out-dir "${JSON_DIR}" \
        2>&1 | tee "${LOG_FILE}" || echo "[WARN] DPOT-${SIZE} failed on ${PDE}"
      cp -u "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || true
    done

    # 2. MORPH (Ti, S)
    for SIZE in $MORPH_SIZES; do
      JSON_FILE="${JSON_DIR}/morph_${SIZE}_${PDE}_dt${CADENCE}_grid${GRID}_results.json"
      if [[ -f "$JSON_FILE" ]]; then
        echo "[EXISTS] Skipping MORPH-${SIZE} on ${PDE} (dt=${CADENCE}s)"
        continue
      fi
      echo ""
      echo ">>> [dt=${CADENCE}s | GRID=${GRID}] MORPH-${SIZE} on ${PDE}..."
      LOG_FILE="${OUT_DIR}/morph_${SIZE}_${PDE}_dt${CADENCE}_grid${GRID}.log"
      python3 experiments/scale/cross_fm_benchmark.py \
        --fm morph --fm-size "${SIZE}" --pde "${PDE}" \
        --fm-data-path "${DATA_PATH}" \
        --n-test-traj "${N_TEST_TRAJ}" \
        --steps "${STEPS}" \
        --coarse-dt "${CADENCE}" \
        --grid "${GRID}" \
        --device "${DEVICE}" \
        --out-dir "${JSON_DIR}" \
        2>&1 | tee "${LOG_FILE}" || echo "[WARN] MORPH-${SIZE} failed on ${PDE}"
      cp -u "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || true
    done

    # 3. Polymathic (TFNO, UNetConvNext, FNO)
    for FAMILY in $POLYMATHIC_FAMILIES; do
      JSON_FILE="${JSON_DIR}/polymathic_${FAMILY}_${PDE}_dt${CADENCE}_grid${GRID}_results.json"
      if [[ -f "$JSON_FILE" ]]; then
        echo "[EXISTS] Skipping Polymathic-${FAMILY} on ${PDE} (dt=${CADENCE}s)"
        continue
      fi
      echo ""
      echo ">>> [dt=${CADENCE}s | GRID=${GRID}] Polymathic ${FAMILY} on ${PDE}..."
      LOG_FILE="${OUT_DIR}/polymathic_${FAMILY}_${PDE}_dt${CADENCE}_grid${GRID}.log"
      python3 experiments/scale/cross_fm_benchmark.py \
        --fm the_well --fm-family "${FAMILY}" --fm-dataset "shear_flow" --pde "${PDE}" \
        --fm-data-path "${DATA_PATH}" \
        --n-test-traj "${N_TEST_TRAJ}" \
        --steps "${STEPS}" \
        --coarse-dt "${CADENCE}" \
        --grid "${GRID}" \
        --device "${DEVICE}" \
        --out-dir "${JSON_DIR}" \
        2>&1 | tee "${LOG_FILE}" || echo "[WARN] Polymathic ${FAMILY} failed on ${PDE}"
      cp -u "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || true
    done

  done
done

echo ""
echo "================================================================================"
echo "GENERATING UPDATED MASTER SUMMARY TABLES..."
echo "================================================================================"
python3 experiments/scale/summarize_master_matrix.py "${JSON_DIR}"
