#!/usr/bin/env bash
# Resumable local corpus matching the released NS-Gauss trajectory/frame count.
set -euo pipefail
python3 -u experiments/scale/generate_local_actual_like_ns.py \
  --out-root "${LOCAL_DATA_ROOT:-/Volumes/ExternalSSD/HILP/data}" \
  --name "${NAME:-actual_like_ns2d_forced_n64_dt01_20k}" \
  --grid "${GRID:-64}" --solver-dt "${SOLVER_DT:-0.0005}" \
  --snapshot-dt "${SNAPSHOT_DT:-0.1}" --frames "${FRAMES:-21}" \
  --train "${N_TRAIN:-16000}" --val "${N_VAL:-2000}" --test "${N_TEST:-2000}" \
  --shard-size "${SHARD_SIZE:-256}" --batch "${BATCH:-8}" --seed "${SEED:-20260910}" \
  --scratch-root "${SCRATCH_ROOT:-/private/tmp/hilp_actual_like_ns_staging}" \
  --copy-chunk-mib "${COPY_CHUNK_MIB:-1}"
