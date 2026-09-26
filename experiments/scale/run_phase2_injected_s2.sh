#!/usr/bin/env bash
set -euo pipefail

python -u experiments/scale/phase2_injected_s2.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels "${FM_CHANNELS:-velocity}" \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --steps "${STEPS:-6}" --lead-steps "${LEAD:-1}" \
  --n-cal "${N_CAL:-24}" --n-test "${N_TEST:-24}" \
  --noise-rms-rel "${NOISE_RMS_REL:-0.02}" --target "${TARGET:-clean_fm}" \
  --k "${K:-64}" --tau-rel "${TAU_REL:-0}" --oversample "${OVERSAMPLE:-16}" --chunk "${CHUNK:-16}" \
  --n-tail "${N_TAIL:-16}" --seed "${SEED:-0}" --tag "${TAG:-phase2}"
