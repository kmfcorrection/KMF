#!/usr/bin/env bash
set -euo pipefail

# Run FM-specific Stage 2/3 experiments for the adapters that are currently
# adapters with an AD-valid state-map contract: Poseidon, DPOT, and MORPH.
#
# Environment knobs:
#   FMS="poseidon dpot"
#   POSEIDON_SIZES="T B L"
#   DPOT_SIZES="Ti S M"
#   MORPH_SIZES="Ti S M"
#   VARIANTS="1 2 3"              # interpreted as lead_steps / target horizon
#   RUN_S2=1 RUN_S3=1
#   QUICK=1                       # tiny smoke test
#   CHUNK=16 K=64 STRIDE=200

FMS=${FMS:-"poseidon dpot"}
POSEIDON_SIZES=${POSEIDON_SIZES:-"T B L"}
DPOT_SIZES=${DPOT_SIZES:-"Ti S M"}
MORPH_SIZES=${MORPH_SIZES:-"Ti S M"}
DPOT_CHANNELS=${DPOT_CHANNELS:-velocity}
VARIANTS=${VARIANTS:-"1 2 3"}
RUN_S2=${RUN_S2:-1}
RUN_S3=${RUN_S3:-1}
CHUNK=${CHUNK:-16}
K=${K:-64}
STRIDE=${STRIDE:-200}
PDE=${PDE:-ns2d_forced}
TAG=${TAG:-main}
DATA_SOURCE=${DATA_SOURCE:-synthetic}
FM_DATA_PATH=${FM_DATA_PATH:-}

DATA_ARGS=(--data-source "$DATA_SOURCE")
if [[ -n "$FM_DATA_PATH" ]]; then DATA_ARGS+=(--fm-data-path "$FM_DATA_PATH"); fi

if [[ "${QUICK:-0}" == "1" ]]; then
  N_CAL=${N_CAL:-4}
  N_TEST=${N_TEST:-2}
  S2_STEPS=${S2_STEPS:-3}
  S2_TRAJ=${S2_TRAJ:-3}
  S3_CAL_TRAJ=${S3_CAL_TRAJ:-3}
  S3_TRAJ=${S3_TRAJ:-1}
  S3_STEPS=${S3_STEPS:-2}
  METHODS=${METHODS:-"pushforward identity"}
  N_ITER=${N_ITER:-1}
  N_TAIL=${N_TAIL:-4}
else
  N_CAL=${N_CAL:-32}
  N_TEST=${N_TEST:-16}
  S2_STEPS=${S2_STEPS:-6}
  S2_TRAJ=${S2_TRAJ:-16}
  S3_CAL_TRAJ=${S3_CAL_TRAJ:-16}
  S3_TRAJ=${S3_TRAJ:-4}
  S3_STEPS=${S3_STEPS:-4}
  METHODS=${METHODS:-"pushforward identity"}
  N_ITER=${N_ITER:-2}
  N_TAIL=${N_TAIL:-8}
fi

python - <<'PY'
import importlib.util, sys
missing = [m for m in ("torch", "numpy", "scipy") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("FATAL: missing Python packages: " + ", ".join(missing))
print(">>> interpreter:", sys.executable, flush=True)
PY

run_one() {
  local fm=$1
  local size=$2
  local channels=$3
  local lead=$4
  local label="${fm}-${size}-s${lead}"

  echo
  echo ">>> ${label} -- S2/S3 on ${PDE}, channels=${channels}, k=${K}, stride=${STRIDE}"

  if [[ "$RUN_S2" == "1" ]]; then
    python experiments/scale/s2_fm_calibration.py \
      --fm "$fm" --fm-size "$size" --fm-channels "$channels" \
      --pde "$PDE" --lead-steps "$lead" --grid 128 --stride "$STRIDE" \
      --k "$K" --chunk "$CHUNK" --n-iter "$N_ITER" --n-tail "$N_TAIL" \
      --n-cal "$N_CAL" --n-test "$N_TEST" --steps "$S2_STEPS" \
      --n-traj "$S2_TRAJ" --methods $METHODS --tag "s${lead}_${TAG}"
  fi

  if [[ "$RUN_S3" == "1" ]]; then
    python experiments/scale/s3_fm_rollout_uq.py \
      --fm "$fm" --fm-size "$size" --fm-channels "$channels" \
      --pde "$PDE" --lead-steps "$lead" --grid 128 --stride "$STRIDE" \
      --k "$K" --chunk "$CHUNK" --n-iter "$N_ITER" --n-tail "$N_TAIL" \
      --n-cal "$N_CAL" --steps "$S3_STEPS" --n-cal-traj "$S3_CAL_TRAJ" \
      --n-traj "$S3_TRAJ" --tag "s${lead}_${TAG}" "${DATA_ARGS[@]}"
  fi
}

for fm in $FMS; do
  case "$fm" in
    poseidon)
      for size in $POSEIDON_SIZES; do
        for lead in $VARIANTS; do
          run_one poseidon "$size" velocity "$lead"
        done
      done
      ;;
    dpot)
      for size in $DPOT_SIZES; do
        for lead in $VARIANTS; do
          run_one dpot "$size" "$DPOT_CHANNELS" "$lead"
        done
      done
      ;;
    morph)
      for size in $MORPH_SIZES; do
        for lead in $VARIANTS; do
          # MORPH's adapter is a two-component velocity map. It is intentionally
          # limited to S1--S3 until a released physical trajectory/normalizer
          # contract makes an S4 likelihood meaningful.
          run_one morph "$size" velocity "$lead"
        done
      done
      ;;
    *)
      echo "FATAL: unknown FM '$fm' (supported: poseidon dpot morph)" >&2
      exit 2
      ;;
  esac
done

echo
echo ">>> Done. Results in results/scale/s2_fm and results/scale/s3_fm"
