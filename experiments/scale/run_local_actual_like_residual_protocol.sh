#!/usr/bin/env bash
# Fully local, resumable actual-like residual protocol.
#
# Matches NS-Gauss trajectory count (20k) and 21 sparse output frames at dt=0.1,
# while retaining a 64x64 state so the high-accuracy oracle is practical on CPU.
set -euo pipefail

LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/Volumes/ExternalSSD/HILP/data}"
NAME="${NAME:-actual_like_ns2d_forced_n64_dt01_20k}"
DATASET="$LOCAL_DATA_ROOT/$NAME"

LOCAL_DATA_ROOT="$LOCAL_DATA_ROOT" NAME="$NAME" \
N_TRAIN="${N_TRAIN:-16000}" N_VAL="${N_VAL:-2000}" N_TEST="${N_TEST:-2000}" \
SHARD_SIZE="${SHARD_SIZE:-256}" BATCH="${BATCH:-8}" \
SCRATCH_ROOT="${SCRATCH_ROOT:-/private/tmp/hilp_actual_like_ns_staging}" \
COPY_CHUNK_MIB="${COPY_CHUNK_MIB:-1}" \
GRID="${GRID:-64}" SOLVER_DT="${SOLVER_DT:-0.0005}" SNAPSHOT_DT="${SNAPSHOT_DT:-0.1}" \
FRAMES="${FRAMES:-21}" SEED="${SEED:-20260910}" \
bash experiments/scale/run_generate_local_actual_like_ns.sh

LOCAL_DATASET="$DATASET" \
LOCAL_RESULTS="${LOCAL_RESULTS:-/Volumes/ExternalSSD/HILP/results/actual_like_residual_transport_20k}" \
CADENCES="0.1" N_CAL_TRAJ="${N_CAL_TRAJ:-512}" N_TEST_TRAJ="${N_TEST_TRAJ:-512}" \
TRANSITIONS_PER_TRAJ="${TRANSITIONS_PER_TRAJ:-8}" \
bash experiments/scale/run_audit_local_residual_only.sh
