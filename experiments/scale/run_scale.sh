#!/usr/bin/env bash
# Full scaled roadmap on one node.
#
# s0 runs FIRST and is a gate, for the same reason stage 9 was in the 1D suite:
# it validates the approximation on a problem where the exact answer exists. If
# s0 says "lowrank-lossy", a negative result from s1-s4 is unattributable --
# it could be the method failing or the rank-k truncation failing.
#
# Usage:
#   ./experiments/scale/run_scale.sh                       # default PDE, 128x128
#   PDES="kolmogorov" GRID=64 ./experiments/scale/run_scale.sh
#   STAGES="s3" ./experiments/scale/run_scale.sh           # just the rollout stage
#
# Budget at 128x128 (N=16,384), rank 64, on one A100:
#   data generation   ~40 min per PDE (512+128+128 trajectories, float64 solver)
#   training          ~3 h per model; 4 ensemble members + 1 dropout = ~15 h
#   s0                ~10 min (runs at 32x32, where the exact Jacobian exists)
#   s1                ~25 min
#   s2                ~90 min (the baselines dominate: MC-dropout and SWAG are
#                     O(n_samples) forward passes per state)
#   s3                ~60 min
#   s4                ~2 h
set -euo pipefail
cd "$(dirname "$0")/../.."

PDES="${PDES:-ns2d_forced kolmogorov}"
GRID="${GRID:-128}"
ARCH="${ARCH:-fno2d}"
K="${K:-64}"
NTEST="${NTEST:-64}"
NCAL="${NCAL:-256}"
STAGES="${STAGES:-s0 s1 s2 s3 s4}"
EPOCHS="${EPOCHS:-60}"
ENSEMBLE="${ENSEMBLE:-4}"
DEVICE="${DEVICE:-}"
DEV_ARG=""
[ -n "$DEVICE" ] && DEV_ARG="--device $DEVICE"

log() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }
has() { echo " $STAGES " | grep -q " $1 "; }

for pde in $PDES; do
  log "[$pde] data ($GRID x $GRID)"
  python3 -m hipp.scale.data2d --pde "$pde" --n "$GRID" $DEV_ARG

  log "[$pde] training $ARCH ($ENSEMBLE members + dropout + SWAG)"
  python3 -m hipp.scale.train2d --pde "$pde" --arch "$ARCH" --grid "$GRID" \
      --epochs "$EPOCHS" --ensemble "$ENSEMBLE" --dropout 0.1 --swag-epochs 10 \
      $DEV_ARG

  COMMON="--pde $pde --arch $ARCH --grid $GRID --k $K --n-test $NTEST --n-cal $NCAL $DEV_ARG"

  if has s0; then
    log "[$pde] s0 -- low-rank vs exact (GATE)"
    python3 experiments/scale/s0_validate_lowrank.py $COMMON --val-grid 32
  fi
  if has s1; then
    log "[$pde] s1 -- direction test across conditioning regimes"
    python3 experiments/scale/s1_direction_test.py $COMMON
  fi
  if has s2; then
    log "[$pde] s2 -- calibration and quality vs baselines + conformal"
    python3 experiments/scale/s2_calibration.py $COMMON
  fi
  if has s3; then
    log "[$pde] s3 -- rollout propagation (headline)"
    python3 experiments/scale/s3_rollout_uq.py $COMMON --steps 12 --n-traj 16
  fi
  if has s4; then
    log "[$pde] s4 -- scaling ablation"
    python3 experiments/scale/s4_scaling.py $COMMON --axes rank architecture
  fi
done

log "Done. Results in results/scale/<stage>/<pde>/"
