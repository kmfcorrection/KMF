#!/usr/bin/env bash
set -euo pipefail

FMS=${FMS:-poseidon}
POSEIDON_SIZES=${POSEIDON_SIZES:-T}
DPOT_SIZES=${DPOT_SIZES:-Ti}
VARIANTS=${VARIANTS:-1}
K=${K:-64}
CHUNK=${CHUNK:-16}
STRIDE=${STRIDE:-200}
TAG=${TAG:-main}
SAMPLER=${SAMPLER:-none}
DATA_SOURCE=${DATA_SOURCE:-synthetic}
FM_DATA_PATH=${FM_DATA_PATH:-}
HILP_FLOOR_GRID=${HILP_FLOOR_GRID:-1.0}
BETA_GRID=${BETA_GRID:-"0 0.25 0.5 0.75 1"}
# Run one stage at a time to attribute improvements cleanly, or pass all three.
S4_STAGES=${S4_STAGES:-c1_spectral}

DATA_ARGS=(--data-source "$DATA_SOURCE")
if [[ -n "$FM_DATA_PATH" ]]; then DATA_ARGS+=(--fm-data-path "$FM_DATA_PATH"); fi

if [[ "${QUICK:-0}" == "1" ]]; then
  N_CAL_TRAJ=${N_CAL_TRAJ:-1}
  N_TEST_TRAJ=${N_TEST_TRAJ:-1}
  STEPS=${STEPS:-2}
  MAP_STEPS=${MAP_STEPS:-5}
  LAMBDA_GRID=${LAMBDA_GRID:-"0.1 1.0"}
  N_SAMPLES=${N_SAMPLES:-8}
  BURN_IN=${BURN_IN:-8}
else
  N_CAL_TRAJ=${N_CAL_TRAJ:-2}
  N_TEST_TRAJ=${N_TEST_TRAJ:-4}
  STEPS=${STEPS:-4}
  MAP_STEPS=${MAP_STEPS:-30}
  LAMBDA_GRID=${LAMBDA_GRID:-"0.03 0.1 0.3 1.0 3.0 10.0 30.0 100.0"}
  N_SAMPLES=${N_SAMPLES:-64}
  BURN_IN=${BURN_IN:-64}
fi

run_one() {
  local fm=$1 size=$2 channels=$3 lead=$4
  local stage=$5
  local -a STAGE_ARGS
  case "$stage" in
    c1_spectral)
      STAGE_ARGS=(--spectral-whitening --methods raw physics_only isotropic hilp) ;;
    c12_projected)
      STAGE_ARGS=(--spectral-whitening --project-incompressible
                  --methods raw raw_projected physics_only isotropic hilp) ;;
    c123_blend)
      STAGE_ARGS=(--spectral-whitening --project-incompressible
                  --beta-grid $BETA_GRID
                  --methods raw raw_projected physics_only isotropic hilp hilp_blend) ;;
    *) echo "unknown S4 stage: $stage" >&2; exit 2 ;;
  esac
  echo ">>> S4 ${stage}: ${fm}-${size}, lead=${lead}"
  python experiments/scale/s4_fm_physics_correction.py \
    --fm "$fm" --fm-size "$size" --fm-channels "$channels" \
    --pde ns2d_forced --grid 128 --stride "$STRIDE" --lead-steps "$lead" \
    --k "$K" --chunk "$CHUNK" --n-cal-traj "$N_CAL_TRAJ" \
    --n-test-traj "$N_TEST_TRAJ" --steps "$STEPS" --map-steps "$MAP_STEPS" \
    --lambda-grid $LAMBDA_GRID --sampler "$SAMPLER" \
    --hilp-floor-grid $HILP_FLOOR_GRID \
    --n-samples "$N_SAMPLES" --burn-in "$BURN_IN" \
    --tag "lead${lead}_${TAG}_${stage}" "${DATA_ARGS[@]}" "${STAGE_ARGS[@]}"
}

for fm in $FMS; do
  case "$fm" in
    poseidon)
      for size in $POSEIDON_SIZES; do
        for lead in $VARIANTS; do
          for stage in $S4_STAGES; do run_one poseidon "$size" velocity "$lead" "$stage"; done
        done
      done ;;
    dpot)
      echo "FATAL: DPOT S4 is disabled: the released checkpoint has no verified " \
           "physical preprocessing/history contract. Run S1--S3 only." >&2
      exit 2 ;;
    *) echo "unknown FM: $fm" >&2; exit 2 ;;
  esac
done
