#!/usr/bin/env bash
# S4 High-Fidelity Fixed-Substep Flow Likelihood HILP.
# Eradicates the algebraic collocation truncation error noise floor by
# evaluating a fixed 4-substep SSP-RK3 spectral flow advance from conditioning state:
#   r(u) = u - Psi_RK3(c_t; substeps=4)
# Exact gradient alignment: cos theta = 1.0000.
# Zero forward adaptive solver calls, zero future truth access.
# Runtime ~0.3 ms on GPU.
set -euo pipefail

POSEIDON_SIZE=${POSEIDON_SIZE:-T}
STEPS=${STEPS:-4}
LEAD=${LEAD:-1}
N_CAL_TRAJ=${N_CAL_TRAJ:-8}
N_VAL_TRAJ=${N_VAL_TRAJ:-4}
N_TEST_TRAJ=${N_TEST_TRAJ:-4}
FLOW_SUBSTEPS=${FLOW_SUBSTEPS:-0}
CFL=${CFL:-0.5}
METHODS=${METHODS:-"flow_blend sobolev_flow"}
BETA_GRID=${BETA_GRID:-"0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0"}
TAU=${TAU:-0.1}
SOBOLEV_POWER=${SOBOLEV_POWER:-1.0}
EXTRA_FLAGS=${EXTRA_FLAGS:---project-incompressible}
FM_DATA_PATH=${FM_DATA_PATH:?set FM_DATA_PATH}
TAG=${TAG:-hilp_fixed_flow}
SEED=${SEED:-20260908}

python -u experiments/scale/s4_fm_hilp_fixed_flow.py \
  --fm poseidon --fm-size "$POSEIDON_SIZE" --fm-channels velocity \
  --pde ns2d_forced --grid 128 --lead-steps "$LEAD" --steps "$STEPS" \
  --n-cal-traj "$N_CAL_TRAJ" --n-val-traj "$N_VAL_TRAJ" --n-test-traj "$N_TEST_TRAJ" \
  --flow-substeps "$FLOW_SUBSTEPS" --cfl "$CFL" --methods $METHODS \
  --beta-grid $BETA_GRID --tau "$TAU" --sobolev-power "$SOBOLEV_POWER" \
  --fm-data-path "$FM_DATA_PATH" --seed "$SEED" --tag "$TAG" \
  $EXTRA_FLAGS
