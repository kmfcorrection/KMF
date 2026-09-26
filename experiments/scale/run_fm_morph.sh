#!/usr/bin/env bash
# Set up and run MORPH Ti/S/M through the AD-valid S1--S3 pipeline.
# S4 is intentionally excluded: no MORPH-native physical trajectory,
# normalizer, forcing, or cadence metadata has been registered yet.
set -euo pipefail
cd "$(dirname "$0")/../.."

MORPH_SIZES=${MORPH_SIZES:-"Ti S M"}
MORPH_DIR=${MORPH_PATH:-"$PWD/third_party/morph"}
PYTHON=${PYTHON:-python}
K=${K:-64}
NSTATES=${NSTATES:-8}
CHUNK=${CHUNK:-8}

if ! "$PYTHON" -c "import torch, einops, huggingface_hub" 2>/dev/null; then
  echo "FATAL: activate an environment with torch, einops, and huggingface_hub." >&2
  exit 2
fi

if [[ ! -f "$MORPH_DIR/src/utils/vit_conv_xatt_axialatt2.py" ]]; then
  echo ">>> MORPH source not found at $MORPH_DIR -- cloning"
  git clone --depth 1 https://github.com/lanl/MORPH.git "$MORPH_DIR"
fi
export MORPH_PATH="$MORPH_DIR"

if [[ "${PREFETCH:-0}" == "1" ]]; then
  "$PYTHON" - "$MORPH_SIZES" <<'PY'
import sys
from huggingface_hub import hf_hub_download
names = {
    "Ti": "morph-Ti-FM-max_ar1_ep225.pth",
    "S": "morph-S-FM-max_ar1_ep225.pth",
    "M": "morph-M-FM-max_ar1_ep290_latestbatch.pth",
}
for size in sys.argv[1:]:
    if size not in names:
        raise SystemExit(f"unknown MORPH size {size}; choose {sorted(names)}")
    path = hf_hub_download("mahindrautela/MORPH", names[size], subfolder="models/FM")
    print(f"MORPH-{size} -> {path}", flush=True)
PY
  exit 0
fi

# S1/gate: do not spend S2/S3 time unless the released checkpoint is a
# deterministic, AD-valid physical-state map under the adapter's declared
# causal normalization.  `check_fm` records JVP/finite-difference and adjoint
# checks in results/scale/check_fm/.
for SIZE in $MORPH_SIZES; do
  echo ">>> MORPH-${SIZE}: S1 adapter/geometry gate"
  "$PYTHON" experiments/scale/check_fm.py \
    --fm morph --fm-size "$SIZE" --fm-channels velocity \
    --k "$K" --n-states "$NSTATES" --chunk "$CHUNK" \
    --states fluid --lead-times 1 --tag morph_main
done

if [[ "${GATE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

FMS=morph MORPH_SIZES="$MORPH_SIZES" ./experiments/scale/run_fm_s2s3.sh
