#!/usr/bin/env bash
# Download and assemble the official Poseidon downstream datasets needed for
# the ICLR HILP benchmark.  Run on the cluster/login node, not from a GPU job.
#
# Default suite (about 86.5 GB raw; retain at least 200 GB free for assembly):
#   FNS-KF     forced incompressible Navier--Stokes, 2 velocity channels
#   ACE        Allen--Cahn reaction--diffusion, scalar state
#   Wave-Gauss variable-speed wave equation, scalar state plus static c(x)
#
# NS-Gauss is intentionally opt-in: it is substantially larger, but is the
# matched native incompressible benchmark for the frozen Poseidon adapter and
# the only dataset accepted by run_iclr_poseidon_protocol.sh at present.
#
# The assembled NetCDF files contain the publisher-defined contiguous
# train/validation/test partitions.  This script writes their boundaries to a
# manifest; downstream experiment adapters must select only the official test
# partition, never trajectories 0...N by default.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"/scratch/bcqc/${USER}/hilp_poseidon_datasets"}
DATASETS=${DATASETS:-"FNS-KF ACE Wave-Gauss"}
# Raw chunks are the only recoverable source while assembly is in progress.
# They may be removed only after the assembled file passes three actual reads.
DELETE_RAW_AFTER_VERIFY=${DELETE_RAW_AFTER_VERIFY:-0}

mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/assembled"

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD=(huggingface-cli download)
else
  echo "ERROR: need the Hugging Face CLI. In the HILP venv: pip install huggingface_hub" >&2
  exit 2
fi

for DATASET in $DATASETS; do
  case "$DATASET" in
    NS-Gauss) TEST_START=19760; TEST_COUNT=240; EXPECTED="20000,21,3,128,128" ;;
    FNS-KF) TEST_START=19760; TEST_COUNT=240; EXPECTED="20000,21,2,128,128" ;;
    ACE) TEST_START=14760; TEST_COUNT=240; EXPECTED="15000,20,128,128" ;;
    Wave-Gauss) TEST_START=10272; TEST_COUNT=240; EXPECTED="10512,15,128,128" ;;
    *) echo "ERROR: unsupported dataset '$DATASET'" >&2; exit 2 ;;
  esac

  RAW="$DATA_ROOT/raw/$DATASET"
  OUT="$DATA_ROOT/assembled/$DATASET.nc"
  echo ">>> Downloading official camlab-ethz/$DATASET to $RAW"
  # FM checkpoint experiments often export HF_HUB_OFFLINE=1.  Dataset
  # preparation is the intentional exception: override it only for this CLI
  # invocation, leaving the caller's environment otherwise unchanged.
  HF_HUB_OFFLINE=0 "${HF_DOWNLOAD[@]}" "camlab-ethz/$DATASET" \
    --repo-type dataset --local-dir "$RAW"

  if [[ ! -f "$OUT" ]]; then
    if command -v rg >/dev/null 2>&1; then
      ASSEMBLER=$(rg --files "$RAW" | rg '/assemble_data\.py$' | head -n 1 || true)
    else
      # Delta login images need not include ripgrep.  This fallback only
      # locates a file that was just downloaded; it does not alter data.
      ASSEMBLER=$(find "$RAW" -type f -name assemble_data.py -print -quit 2>/dev/null || true)
    fi
    if [[ -z "$ASSEMBLER" ]]; then
      echo "ERROR: no assemble_data.py found in $RAW; do not improvise file concatenation." >&2
      exit 2
    fi
    echo ">>> Assembling $DATASET -> $OUT"
    python "$ASSEMBLER" --input_dir "$RAW" --output_file "$OUT"
  fi

  python - "$OUT" "$DATASET" "$TEST_START" "$TEST_COUNT" "$EXPECTED" <<'PY'
import json
import sys
from pathlib import Path
import h5py

path, name, test_start, test_count, expected = sys.argv[1:]
test_start, test_count = int(test_start), int(test_count)
expected = tuple(map(int, expected.split(',')))
with h5py.File(path, 'r') as h:
    key = 'velocity' if 'velocity' in h else 'solution' if 'solution' in h else None
    if key is None:
        raise SystemExit(f'{path}: neither velocity nor solution exists')
    shape = tuple(h[key].shape)
    # Metadata alone can be readable from a truncated HDF5 file.  Force three
    # real chunk reads before accepting an assembled dataset as valid.
    for i in (0, shape[0] // 2, shape[0] - 1):
        probe = h[key][i, 0]
        if not probe.size:
            raise SystemExit(f'{path}: empty data probe at trajectory {i}')
    static = {'has_wave_speed_c': 'c' in h}
if shape != expected:
    raise SystemExit(f'{path}: {name} expected {expected}, found {shape}')
if test_start + test_count > shape[0]:
    raise SystemExit(f'{path}: declared test split is outside dataset')
record = {
    'dataset': name, 'assembled_path': str(Path(path).resolve()),
    'state_variable': key, 'shape': shape,
    'official_test_start': test_start, 'official_test_count': test_count,
    'official_test_slice': [test_start, test_start + test_count], **static,
}
manifest = Path(path).with_suffix('.manifest.json')
manifest.write_text(json.dumps(record, indent=2) + '\n')
print('verified:', json.dumps(record))
PY

  if [[ "$DELETE_RAW_AFTER_VERIFY" == "1" ]]; then
    echo ">>> Verified $DATASET; removing its raw chunks: $RAW"
    rm -rf "$RAW"
  else
    echo ">>> Raw chunks retained: $RAW (set DELETE_RAW_AFTER_VERIFY=1 to remove after verification)"
  fi
done

echo ">>> Dataset preparation complete under $DATA_ROOT"
