#!/usr/bin/env bash
set -euo pipefail

# Phase 3 follows Phase 2.  It compares the scalar Q with (i) unconstrained
# residual PCA and (ii) a centered divergence-free Fourier Q.  The latter
# cannot memorize arbitrary grid patterns from the small calibration corpus.
# Split this explicitly: quoted parameter expansion in a ``for`` list would
# otherwise pass the entire default list as one --innovation argument.
read -r -a INNOVATION_LIST <<< "${INNOVATIONS:-isotropic empirical_lowrank spectral_divfree}"
for INNOVATION in "${INNOVATION_LIST[@]}"; do
  python -u experiments/scale/s3_fm_rollout_uq.py \
    --fm poseidon --fm-size "${POSEIDON_SIZE:-T}" --fm-channels "${FM_CHANNELS:-velocity}" \
    --data-source poseidon-native --fm-data-path "${FM_DATA_PATH:?set FM_DATA_PATH}" \
    --lead-steps "${LEAD:-1}" --steps "${STEPS:-4}" \
    --n-cal-traj "${N_CAL_TRAJ:-8}" --n-traj "${N_TEST_TRAJ:-8}" \
    --k "${K:-64}" --oversample "${OVERSAMPLE:-16}" --chunk "${CHUNK:-16}" \
    --n-tail "${N_TAIL:-16}" --n-sketch "${N_SKETCH:-80}" \
    --trajectory-batch "${TRAJECTORY_BATCH:-1}" \
    ${VERIFY_TRAJECTORY_BATCH:+--verify-trajectory-batch} \
    --innovation "$INNOVATION" --innovation-rank "${INNOVATION_RANK:-16}" \
    --innovation-validation-traj "${INNOVATION_VALIDATION_TRAJ:-0}" \
    --innovation-shrink-grid ${INNOVATION_SHRINK_GRID:-"0 0.25 0.5 0.75 1"} \
    --seed "${SEED:-0}" \
    --complements nystrom isotropic none \
    --tag "${TAG:-phase3}_${INNOVATION}"
done
