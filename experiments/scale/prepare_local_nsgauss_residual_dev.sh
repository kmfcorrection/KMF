#!/usr/bin/env bash
# Make a small raw-cadence NS-Gauss corpus for local Mac residual development.
set -euo pipefail

SOURCE_DATASET=${SOURCE_DATASET:?set SOURCE_DATASET to assembled NS-Gauss.nc}
LOCAL_DATA_ROOT=${LOCAL_DATA_ROOT:-/Volumes/ExternalSSD/HILP/data}
N_TRAIN=${N_TRAIN:-160}
N_VAL=${N_VAL:-32}
N_TEST=${N_TEST:-64}
OUT=${OUT:-"$LOCAL_DATA_ROOT/nsgauss_rawdt005_dev_${N_TRAIN}_${N_VAL}_${N_TEST}.h5"}

python -u experiments/scale/export_nsgauss_residual_dev_subset.py \
  --source "$SOURCE_DATASET" --output "$OUT" \
  --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" \
  --compression gzip --compression-level 1
