#!/usr/bin/env bash
# ==============================================================================
# MASTER ICLR EVALUATION RUNNER
# ALL PDES x ALL FMS x ALL TEMPORAL CADENCES x ALL RESOLUTIONS
# ==============================================================================
# PDEs:               NS-Gauss, FNS-KF, ACE, Wave-Gauss, NS-SL, Wave-Layer
# FMs:                Poseidon (T, B, L), DPOT (Ti, S), MORPH (Ti, S), Polymathic (TFNO, UNetConvNext, FNO)
# Temporal Cadences:  0.05s, 0.10s, 0.20s, 0.30s, 0.40s (Both One-Step & Rollout)
# Spatial Grids:      128x128 (default native), 64x64, 256x256
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/assembled}"
N_TEST_TRAJ="${N_TEST_TRAJ:-50}"
N_CAL_TRAJ="${N_CAL_TRAJ:-20}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-4}"
# A KMF state is emitted at every one of these equally spaced positions inside
# each corrected coarse rollout interval.  The bridge is defined continuously;
# dataset timestamps are used only when a dense state can be scored.
DENSE_SUBSTEPS="${DENSE_SUBSTEPS:-2}"
ALPHA_GRID="${ALPHA_GRID:-0 0.01 0.02 0.05 0.10 0.20}"

# All temporal resolutions (cadences) evaluated in one-step and multi-step rollout:
CADENCES=${CADENCES:-"0.05 0.10 0.20 0.30 0.40"}

# Spatial resolution sweep (defaults to native 128x128, can be "64 128 256")
GRIDS=${GRIDS:-"128"}

# PDEs and Foundation Models
ALL_PDES=${ALL_PDES:-"NS-Gauss FNS-KF ACE Wave-Gauss NS-SL Wave-Layer"}
FMS=${FMS:-"poseidon dpot morph polymathic cno"}

POSEIDON_SIZES=${POSEIDON_SIZES:-"T B L"}
DPOT_SIZES=${DPOT_SIZES:-"Ti S"}
MORPH_SIZES=${MORPH_SIZES:-"Ti S"}
POLYMATHIC_FAMILIES=${POLYMATHIC_FAMILIES:-"TFNO UNetConvNext FNO"}
CNO_SOURCE=${CNO_SOURCE:-}
CNO_CHECKPOINT=${CNO_CHECKPOINT:-}
CNO_CONFIG=${CNO_CONFIG:-}
CNO_TIME_SCALE=${CNO_TIME_SCALE:-1.0}
MOTION_SOURCE=${MOTION_SOURCE:-}
MOTION_CHECKPOINT=${MOTION_CHECKPOINT:-}
MOTION_SIZES=${MOTION_SIZES:-"156.9M S"}
# Fixed-step checkpoints do not expose a physical lead-time argument.  A
# one-call endpoint benchmark is therefore only meaningful at the explicitly
# selected target span.  Poseidon continues to sweep every requested cadence.
FIXED_STEP_CADENCES=${FIXED_STEP_CADENCES:-"0.10"}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR="results/scale/master_matrix_${RUN_ID}"
JSON_DIR="${OUT_DIR}/json"
mkdir -p "${OUT_DIR}"
mkdir -p "${JSON_DIR}"
mkdir -p "results/scale/cross_benchmark"

# A missing checkpoint or incompatible released package is a model-level
# condition, not a result for every PDE/cadence cell.  Load each requested
# variant once before the sweep and skip only variants whose declared contract
# cannot be instantiated.  This keeps the matrix logs honest and readable.
declare -A FM_READY
preflight_fm() {
  local key="$1"
  shift
  if [[ -n "${FM_READY[$key]+set}" ]]; then
    [[ "${FM_READY[$key]}" == "1" ]]
    return
  fi
  local log_file="${OUT_DIR}/preflight_${key//[^[:alnum:]_.-]/_}.log"
  echo ">>> Preflight ${key}..."
  if python3 experiments/scale/cross_fm_benchmark.py \
      "$@" --fm-data-path /dev/null --grid 128 --device cpu --preflight-only \
      >"${log_file}" 2>&1; then
    FM_READY["$key"]=1
    echo "    available"
    return 0
  fi
  FM_READY["$key"]=0
  echo "[UNAVAILABLE] ${key}; skipping its matrix cells. See ${log_file}"
  return 1
}

echo "================================================================================"
echo "MASTER ICLR EVALUATION: ALL FMS x ALL PDES x ALL CADENCES (ONE-STEP & ROLLOUT)"
echo "Run ID:               ${RUN_ID}"
echo "================================================================================"
echo "Data Directory:       ${DATA_ROOT}"
echo "Temporal Cadences:    ${CADENCES} (evaluates BOTH 1-Step and Rollout at each dt)"
echo "Spatial Resolutions:  ${GRIDS}"
echo "PDE Systems:          ${ALL_PDES}"
echo "Foundation Models:    ${FMS}"
echo "Fixed-step Cadences:  ${FIXED_STEP_CADENCES}"
echo "CNO availability:     source/config/checkpoint must be supplied explicitly"
echo "MOTION availability:  disabled until its official TensorFlow formatter and loader are validated"
echo "Test Sample Size:     ${N_TEST_TRAJ} trajectories per test"
echo "Calibration Size:     ${N_CAL_TRAJ} trajectories per configuration"
echo "Rollout Steps:        ${STEPS}"
echo "Dense KMF Substeps:   ${DENSE_SUBSTEPS} (interior states per corrected rollout interval)"
echo "Correction Grid:      ${ALPHA_GRID}"
echo "Device:               ${DEVICE}"
echo "Output Logs:          ${OUT_DIR}/"
echo "================================================================================"

for GRID in $GRIDS; do
  for CADENCE in $CADENCES; do
    echo ""
    echo "################################################################################"
    echo ">>> RUNNING BENCHMARK AT TEMPORAL CADENCE dt = ${CADENCE}s | GRID = ${GRID}x${GRID}"
    echo "################################################################################"

    for PDE in $ALL_PDES; do
      # Datasets may be assembled as NetCDF or HDF5.  The previous hard-coded
      # .nc lookup silently skipped NS-SL/Wave-Layer before any FM preflight.
      DATA_PATH=""
      for EXT in nc h5 hdf5; do
        CANDIDATE="${DATA_ROOT}/${PDE}.${EXT}"
        if [[ -f "${CANDIDATE}" ]]; then
          DATA_PATH="${CANDIDATE}"
          break
        fi
      done
      if [[ -z "${DATA_PATH}" ]]; then
        echo ">>> ${PDE} skipped: no ${PDE}.{nc,h5,hdf5} found under ${DATA_ROOT}"
        continue
      fi

      # ------------------------------------------------------------------------
      # 1. Poseidon Suite (T, B, L) across all PDEs
      # ------------------------------------------------------------------------
      if [[ " $FMS " =~ " poseidon " ]]; then
        for SIZE in $POSEIDON_SIZES; do
          preflight_fm "poseidon_${SIZE}" --fm poseidon --fm-size "${SIZE}" || continue
          echo ""
          echo ">>> [dt=${CADENCE}s | GRID=${GRID}] Poseidon-${SIZE} on ${PDE}..."
          LOG_FILE="${OUT_DIR}/poseidon_${SIZE}_${PDE}_dt${CADENCE}_grid${GRID}.log"
          python3 experiments/scale/cross_fm_benchmark.py \
            --fm poseidon --fm-size "${SIZE}" --pde "${PDE}" \
            --fm-data-path "${DATA_PATH}" \
            --n-cal-traj "${N_CAL_TRAJ}" \
            --n-test-traj "${N_TEST_TRAJ}" \
            --steps "${STEPS}" \
            --coarse-dt "${CADENCE}" \
            --dense-substeps "${DENSE_SUBSTEPS}" \
            --alpha-grid ${ALPHA_GRID} \
            --grid "${GRID}" \
            --device "${DEVICE}" \
            --out-dir "${JSON_DIR}" \
            2>&1 | tee "${LOG_FILE}" || echo "[WARN] Poseidon-${SIZE} failed on ${PDE} (dt=${CADENCE}s)"
          cp -u "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || cp "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || true
        done
      fi

      # ------------------------------------------------------------------------
      # 2. Remaining FM adapters on compatible PDE datasets.  DPOT and MORPH
      # checkpoints in this benchmark require two-channel velocity states.
      # Scalar datasets such as ACE and the wave cases are not valid inputs;
      # skip those contract-invalid pairs explicitly rather than fabricating a
      # channel or recording an adapter error as a model failure.
      # ------------------------------------------------------------------------
        # DPOT (Ti, S)
        if [[ " $FMS " =~ " dpot " ]]; then
          if [[ "${PDE}" != "NS-Gauss" && "${PDE}" != "FNS-KF" && "${PDE}" != "NS-SL" ]]; then
            echo ">>> DPOT skipped on ${PDE}: checkpoint contract requires a two-channel velocity state; dataset is not a compatible fluid pair."
          elif [[ " ${FIXED_STEP_CADENCES} " != *" ${CADENCE} "* ]]; then
            echo ">>> DPOT skipped at dt=${CADENCE}s: no exposed lead-time query; configured fixed endpoint span is ${FIXED_STEP_CADENCES}."
          else
          for SIZE in $DPOT_SIZES; do
            preflight_fm "dpot_${SIZE}" --fm dpot --fm-size "${SIZE}" || continue
            echo ""
            echo ">>> [dt=${CADENCE}s | GRID=${GRID}] DPOT-${SIZE} on ${PDE}..."
            LOG_FILE="${OUT_DIR}/dpot_${SIZE}_${PDE}_dt${CADENCE}_grid${GRID}.log"
            python3 experiments/scale/cross_fm_benchmark.py \
              --fm dpot --fm-size "${SIZE}" --pde "${PDE}" \
              --fm-data-path "${DATA_PATH}" \
              --n-cal-traj "${N_CAL_TRAJ}" \
              --n-test-traj "${N_TEST_TRAJ}" \
              --steps "${STEPS}" \
              --coarse-dt "${CADENCE}" \
              --dense-substeps "${DENSE_SUBSTEPS}" \
              --alpha-grid ${ALPHA_GRID} \
              --grid "${GRID}" \
              --device "${DEVICE}" \
              --out-dir "${JSON_DIR}" \
              2>&1 | tee "${LOG_FILE}" || echo "[WARN] DPOT-${SIZE} skipped or failed on ${PDE}"
            cp -u "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || cp "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || true
          done
          fi
        fi

        # MORPH (Ti, S)
        if [[ " $FMS " =~ " morph " ]]; then
          if [[ "${PDE}" != "NS-Gauss" && "${PDE}" != "FNS-KF" && "${PDE}" != "NS-SL" ]]; then
            echo ">>> MORPH skipped on ${PDE}: checkpoint contract requires a two-channel velocity state; dataset is not a compatible fluid pair."
          elif [[ " ${FIXED_STEP_CADENCES} " != *" ${CADENCE} "* ]]; then
            echo ">>> MORPH skipped at dt=${CADENCE}s: no exposed lead-time query; configured fixed endpoint span is ${FIXED_STEP_CADENCES}."
          else
          for SIZE in $MORPH_SIZES; do
            preflight_fm "morph_${SIZE}" --fm morph --fm-size "${SIZE}" || continue
            echo ""
            echo ">>> [dt=${CADENCE}s | GRID=${GRID}] MORPH-${SIZE} on ${PDE}..."
            LOG_FILE="${OUT_DIR}/morph_${SIZE}_${PDE}_dt${CADENCE}_grid${GRID}.log"
            python3 experiments/scale/cross_fm_benchmark.py \
              --fm morph --fm-size "${SIZE}" --pde "${PDE}" \
              --fm-data-path "${DATA_PATH}" \
              --n-cal-traj "${N_CAL_TRAJ}" \
              --n-test-traj "${N_TEST_TRAJ}" \
              --steps "${STEPS}" \
              --coarse-dt "${CADENCE}" \
              --dense-substeps "${DENSE_SUBSTEPS}" \
              --alpha-grid ${ALPHA_GRID} \
              --grid "${GRID}" \
              --device "${DEVICE}" \
              --out-dir "${JSON_DIR}" \
              2>&1 | tee "${LOG_FILE}" || echo "[WARN] MORPH-${SIZE} skipped or failed on ${PDE}"
            cp -u "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || cp "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || true
          done
          fi
        fi

        # Polymathic AI Models (TFNO, UNetConvNext, FNO)
        if [[ " $FMS " =~ " polymathic " ]]; then
          for FAMILY in $POLYMATHIC_FAMILIES; do
            preflight_fm "polymathic_${FAMILY}" --fm the_well --fm-family "${FAMILY}" --fm-dataset "shear_flow" || continue
            echo ""
            echo ">>> [dt=${CADENCE}s | GRID=${GRID}] Polymathic ${FAMILY} on ${PDE}..."
            LOG_FILE="${OUT_DIR}/polymathic_${FAMILY}_${PDE}_dt${CADENCE}_grid${GRID}.log"
            python3 experiments/scale/cross_fm_benchmark.py \
              --fm the_well --fm-family "${FAMILY}" --fm-dataset "shear_flow" --pde "${PDE}" \
              --fm-data-path "${DATA_PATH}" \
              --n-cal-traj "${N_CAL_TRAJ}" \
              --n-test-traj "${N_TEST_TRAJ}" \
              --steps "${STEPS}" \
              --coarse-dt "${CADENCE}" \
              --dense-substeps "${DENSE_SUBSTEPS}" \
              --alpha-grid ${ALPHA_GRID} \
              --grid "${GRID}" \
              --device "${DEVICE}" \
              --out-dir "${JSON_DIR}" \
              2>&1 | tee "${LOG_FILE}" || echo "[WARN] Polymathic ${FAMILY} skipped or failed on ${PDE}"
            cp -u "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || cp "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || true
          done
        fi

        # Time-conditioned CNO-FM.  Its released 5-to-4 field contract can be
        # evaluated on this velocity-only corpus only with explicit pinned
        # density/pressure conditioning.  It is therefore limited to the two
        # fluid systems, and the source/config/checkpoint triplet is mandatory.
        if [[ " $FMS " =~ " cno " ]]; then
          if [[ "${PDE}" != "NS-Gauss" && "${PDE}" != "FNS-KF" ]]; then
            echo ">>> CNO-FM skipped on ${PDE}: no validated velocity-only field contract for this PDE."
          elif [[ -z "${CNO_SOURCE}" || -z "${CNO_CHECKPOINT}" || -z "${CNO_CONFIG}" ]]; then
            echo ">>> CNO-FM skipped: set CNO_SOURCE, CNO_CHECKPOINT, and CNO_CONFIG to the matching released artifacts."
          elif preflight_fm "cno_FM" --fm cno --fm-size FM --fm-checkpoint "${CNO_CHECKPOINT}" --cno-source "${CNO_SOURCE}" --cno-config "${CNO_CONFIG}" --cno-time-scale "${CNO_TIME_SCALE}"; then
            echo ""
            echo ">>> [dt=${CADENCE}s | GRID=${GRID}] CNO-FM on ${PDE}..."
            LOG_FILE="${OUT_DIR}/cno_FM_${PDE}_dt${CADENCE}_grid${GRID}.log"
            python3 experiments/scale/cross_fm_benchmark.py \
              --fm cno --fm-size FM --fm-checkpoint "${CNO_CHECKPOINT}" \
              --cno-source "${CNO_SOURCE}" --cno-config "${CNO_CONFIG}" --cno-time-scale "${CNO_TIME_SCALE}" \
              --pde "${PDE}" --fm-data-path "${DATA_PATH}" \
              --n-cal-traj "${N_CAL_TRAJ}" --n-test-traj "${N_TEST_TRAJ}" \
              --steps "${STEPS}" --coarse-dt "${CADENCE}" \
              --dense-substeps "${DENSE_SUBSTEPS}" --alpha-grid ${ALPHA_GRID} \
              --grid "${GRID}" --device "${DEVICE}" --out-dir "${JSON_DIR}" \
              2>&1 | tee "${LOG_FILE}" || echo "[WARN] CNO-FM failed on ${PDE} (dt=${CADENCE}s)"
            cp -u "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || cp "${JSON_DIR}"/*_results.json results/scale/cross_benchmark/ 2>/dev/null || true
          fi
        fi

        # MOTION remains deliberately disabled.  Do not generate a proxy row
        # until the official TensorFlow eight-slot formatter, normalization, and
        # named-array checkpoint load path are implemented and verified.
        if [[ " $FMS " =~ " motion " ]]; then
          echo ">>> MOTION skipped: disabled pending a validated official adapter; no proxy results are produced."
        fi

    done
  done
done

echo ""
echo "================================================================================"
echo "GENERATING MASTER SUMMARY TABLES FOR RUN ${RUN_ID}..."
echo "================================================================================"
python3 experiments/scale/summarize_master_matrix.py "${JSON_DIR}" || echo "[WARN] Summary generator failed"

echo ""
echo "================================================================================"
echo "MASTER ICLR MATRIX RUN COMPLETE! All results saved in ${OUT_DIR}/"
echo "================================================================================"
