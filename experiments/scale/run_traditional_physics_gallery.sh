#!/usr/bin/env bash
# Compare traditional, non-learned PDE residual approximations on Poseidon data.
set -euo pipefail

python -u experiments/scale/plot_physics_approximations.py \
  --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels velocity \
  --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
  --lead-steps "${LEAD:-1}" --steps "${STEPS:-3}" \
  --n-cal-traj "${N_CAL_TRAJ:-16}" --n-val-traj "${N_VAL_TRAJ:-8}" \
  --physics-resolutions ${PHYSICS_RESOLUTIONS:-"8 16 32 64 128"} \
  --flow-cfl "${FLOW_CFL:-0.5}" \
  --physics-batch "${PHYSICS_BATCH:-8}" \
  --fixed-rk3-substeps ${FIXED_RK3_SUBSTEPS:-"4 8 16 32"} \
  --les-resolutions ${LES_RESOLUTIONS:-"32 64"} \
  --les-cs "${LES_CS:-0.17}" --les-substeps "${LES_SUBSTEPS:-16}" \
  --pod-ranks ${POD_RANKS:-"8 16 32"} --pod-substeps "${POD_SUBSTEPS:-16}" \
  --trajectory "${TRAJECTORY:-0}" --transition "${TRANSITION:-0}" \
  --out-dir "${OUT_DIR:-results/figures/traditional_physics_gallery}" \
  --seed "${SEED:-20260908}"
