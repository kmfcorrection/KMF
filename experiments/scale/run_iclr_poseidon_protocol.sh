#!/usr/bin/env bash
# Reproducible ICLR protocol for the currently valid native Poseidon benchmark.
#
# This intentionally does NOT mix DPOT/MORPH transfer experiments into the
# headline benchmark: their released history/normalization and native cadence
# require matched data before they can support a like-for-like main result.
#
# Usage examples:
#   STAGES="gate phase2 phase3 s4" POSEIDON_SIZES="T B L" ./experiments/scale/run_iclr_poseidon_protocol.sh
#   STAGES="s4" POSEIDON_SIZES="B" ./experiments/scale/run_iclr_poseidon_protocol.sh
#
# The split is fixed by trajectory index:
#   [0, N_CAL)                 alpha/scales fit
#   [N_CAL, N_CAL+N_VAL)       lambda/resolution validation
#   [N_CAL+N_VAL, ...)         untouched paper test
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

DATA=${FM_DATA_PATH:?set FM_DATA_PATH to the native Poseidon velocity_16.nc file}
STAGES=${STAGES:-"gate phase2 phase3 s4"}
POSEIDON_SIZES=${POSEIDON_SIZES:-"T B L"}
SEED=${SEED:-20260908}
LEAD=${LEAD:-1}
STEPS=${STEPS:-5}
N_CAL_TRAJ=${N_CAL_TRAJ:-32}
N_VAL_TRAJ=${N_VAL_TRAJ:-16}
N_TEST_TRAJ=${N_TEST_TRAJ:-32}
PHYSICS_RESOLUTION=${PHYSICS_RESOLUTION:-8}
FLOW_CFL=${FLOW_CFL:-0.5}
K=${K:-128}
OVERSAMPLE=${OVERSAMPLE:-16}
CHUNK=${CHUNK:-16}
TRAJECTORY_BATCH=${TRAJECTORY_BATCH:-1}
VERIFY_TRAJECTORY_BATCH=${VERIFY_TRAJECTORY_BATCH:-}
N_TAIL=${N_TAIL:-32}
N_SKETCH=${N_SKETCH:-160}
N_PROBE=${N_PROBE:-32}
LAMBDA_GRID=${LAMBDA_GRID:-"0.03 0.1 0.3 1 3 10 30 100 300"}
TAG_PREFIX=${TAG_PREFIX:-"iclr_poseidon_native128_coarse8"}
# A completed innovation mode is written under its own result directory.
# Keeping this configurable lets a short allocation resume Phase 3 at the
# unfinished modes rather than recomputing its completed controls.
INNOVATIONS=${INNOVATIONS:-"isotropic empirical_lowrank spectral_divfree"}

# This fails early rather than silently overlapping calibration, validation,
# and test trajectories.  Native trajectories remain at 128^2; only Psi uses
# PHYSICS_RESOLUTION^2.
python - "$DATA" "$((N_CAL_TRAJ + N_VAL_TRAJ + N_TEST_TRAJ))" <<'PY'
import sys
import h5py

path, required = sys.argv[1], int(sys.argv[2])
with h5py.File(path, "r") as h:
    key = "velocity" if "velocity" in h else "solution" if "solution" in h else None
    if key is None:
        raise SystemExit(f"{path}: expected 'velocity' or 'solution'")
    shape = h[key].shape
if len(shape) != 5 or shape[2] < 2 or shape[-2:] != (128, 128):
    raise SystemExit(f"{path}: expected (trajectory,time,>=2,128,128), found {shape}")
if shape[0] < required:
    raise SystemExit(f"{path}: need {required} trajectory-disjoint samples, found {shape[0]}")
print(f"ICLR preflight: {path}; dataset shape={shape}; split={required} trajectories")
PY

has_stage () {
  local wanted=$1 item
  for item in $STAGES; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}

if has_stage gate; then
  echo ">>> Phase 1 coarse-physics gate (data-only; no FM)"
  FM_DATA_PATH="$DATA" LEAD="$LEAD" STEPS="$STEPS" \
    N_CAL_TRAJ="$N_CAL_TRAJ" N_TEST_TRAJ="$N_VAL_TRAJ" \
    RESOLUTIONS="8 4" FLOW_CFL="$FLOW_CFL" SEED="$SEED" \
    TAG="${TAG_PREFIX}_physics_gate" \
    ./experiments/scale/run_phase1_physics_controls.sh
fi

for SIZE in $POSEIDON_SIZES; do
  if has_stage phase2; then
    echo ">>> Phase 2 local covariance validation: Poseidon-${SIZE}"
    POSEIDON_SIZE="$SIZE" FM_DATA_PATH="$DATA" LEAD="$LEAD" STEPS=6 \
      N_CAL="$N_CAL_TRAJ" N_TEST="$N_TEST_TRAJ" \
      NOISE_RMS_REL=0.002 TAU_REL=0 K="$K" OVERSAMPLE="$OVERSAMPLE" \
      CHUNK="$CHUNK" N_TAIL="$N_TAIL" SEED="$SEED" \
      TAG="${TAG_PREFIX}_phase2_${SIZE}" \
      ./experiments/scale/run_phase2_injected_s2.sh
  fi

  if has_stage phase3; then
    echo ">>> Phase 3 rollout-UQ diagnostic: Poseidon-${SIZE}"
    POSEIDON_SIZE="$SIZE" FM_DATA_PATH="$DATA" LEAD="$LEAD" STEPS="$STEPS" \
      N_CAL_TRAJ="$((N_CAL_TRAJ + N_VAL_TRAJ))" N_TEST_TRAJ="$N_TEST_TRAJ" \
      INNOVATION_VALIDATION_TRAJ="$N_VAL_TRAJ" INNOVATION_RANK=32 \
      INNOVATION_SHRINK_GRID="0 0.1 0.25 0.5 0.75 1" \
      K="$K" OVERSAMPLE="$OVERSAMPLE" CHUNK="$CHUNK" N_TAIL="$N_TAIL" \
      N_SKETCH="$N_SKETCH" SEED="$SEED" \
      TRAJECTORY_BATCH="$TRAJECTORY_BATCH" \
      VERIFY_TRAJECTORY_BATCH="$VERIFY_TRAJECTORY_BATCH" \
      INNOVATIONS="$INNOVATIONS" \
      TAG="${TAG_PREFIX}_phase3_${SIZE}" \
      ./experiments/scale/run_phase3_structured_s3.sh
  fi

  if has_stage s4; then
    echo ">>> Phase 4 coarse-to-fine posterior correction: Poseidon-${SIZE}"
    POSEIDON_SIZE="$SIZE" DATA_SOURCE=poseidon-native FM_DATA_PATH="$DATA" \
      PHYSICS_LIKELIHOOD=discrete_flow PHYSICS_RESOLUTION="$PHYSICS_RESOLUTION" \
      FLOW_CFL="$FLOW_CFL" LEAD="$LEAD" STEPS="$STEPS" \
      N_CAL_TRAJ="$N_CAL_TRAJ" N_VAL_TRAJ="$N_VAL_TRAJ" N_TEST_TRAJ="$N_TEST_TRAJ" \
      METHODS="physics_only isotropic pushforward" \
      K="$K" OVERSAMPLE="$OVERSAMPLE" CHUNK="$CHUNK" N_TAIL="$N_TAIL" \
      N_SKETCH="$N_SKETCH" N_PROBE="$N_PROBE" LAMBDA_GRID="$LAMBDA_GRID" \
      TRACE_POSTERIOR=1 SEED="$SEED" \
      TAG="${TAG_PREFIX}_phase4_${SIZE}" \
      ./experiments/scale/run_fm_s4_document.sh
  fi
done

echo ">>> ICLR Poseidon protocol completed: stages=[$STAGES], sizes=[$POSEIDON_SIZES]"
