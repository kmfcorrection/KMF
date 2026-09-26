#!/usr/bin/env bash
# Build a small, dense-time, fully local Navier--Stokes fixture on an external SSD.
# It does not download data and it does not contact the cluster.
set -euo pipefail

LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/Volumes/ExternalSSD/HILP/data}"
GRID="${GRID:-64}"
SOLVER_DT="${SOLVER_DT:-0.0005}"
SNAPSHOT_DT="${SNAPSHOT_DT:-0.01}"
FRAMES="${FRAMES:-101}"
N_TRAIN="${N_TRAIN:-32}"
N_VAL="${N_VAL:-8}"
N_TEST="${N_TEST:-8}"
BATCH="${BATCH:-4}"
SEED="${SEED:-20260909}"
NAME="${NAME:-dense_ns2d_forced_n64_dt001}"

python3 experiments/scale/generate_local_dense_ns_residual_dev.py \
  --out-root "$LOCAL_DATA_ROOT" --name "$NAME" \
  --grid "$GRID" --solver-dt "$SOLVER_DT" --snapshot-dt "$SNAPSHOT_DT" \
  --frames "$FRAMES" --train "$N_TRAIN" --val "$N_VAL" --test "$N_TEST" \
  --batch "$BATCH" --seed "$SEED"
