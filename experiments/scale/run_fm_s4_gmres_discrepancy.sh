#!/usr/bin/env bash
# Deployable S4: Calibrated Spectral Discrepancy + GMRES Residual Transport.
# Zero forward ODE integration. Pure instantaneous spatial Jacobian inversion.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
SHELLS=${SHELLS:-8}
RIDGE_GRID=${RIDGE_GRID:-"1e-5 1e-4 1e-3 1e-2 1e-1 1.0 10.0"}
GAIN_GRID=${GAIN_GRID:-"0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.5 1.6 1.8 2.0"}
GMRES_ITERS=${GMRES_ITERS:-32}
GMRES_TOL=${GMRES_TOL:-1e-4}
EXTRA_FLAGS=${EXTRA_FLAGS:---project-incompressible}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_gmres_transport}
SEED=${SEED:-20260908}

python -u experiments/scale/s4_fm_gmres_discrepancy.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --shells "$SHELLS" \
  --ridge-grid $RIDGE_GRID \
  --gain-grid $GAIN_GRID \
  --gmres-iters "$GMRES_ITERS" --gmres-tol "$GMRES_TOL" \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG" \
  $EXTRA_FLAGS

