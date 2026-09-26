#!/usr/bin/env bash
set -euo pipefail

extra=()
if [[ -n "${NOISE_SEED_BASE:-}" ]]; then
  extra+=(--noise-seed-base "$NOISE_SEED_BASE")
fi
if [[ -n "${PROBE_SEED_BASE:-}" ]]; then
  extra+=(--probe-seed-base "$PROBE_SEED_BASE")
fi

python -u experiments/scale/phase2_pushforward_audit.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --states "${STATES:-3}" --samples "${SAMPLES:-4}" --steps "${STEPS:-6}" \
  --n-cal "${N_CAL:-3}" --n-test "${N_TEST:-3}" \
  --split "${SPLIT:-cal}" \
  --noise-rms-rel "${NOISE_RMS_REL:-0.02}" \
  --k "${K:-64}" --tau-rel "${TAU_REL:-0}" --oversample "${OVERSAMPLE:-16}" --chunk "${CHUNK:-16}" \
  --n-tail "${N_TAIL:-16}" --tag "${TAG:-phase2_audit}" \
  "${extra[@]}"
