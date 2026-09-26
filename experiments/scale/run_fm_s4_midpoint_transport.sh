#!/usr/bin/env bash
# One-JVP midpoint-transport residual HILP. No flow endpoint is solved.
set -euo pipefail

PHYSICS_LIKELIHOOD=midpoint_transport_1jvp \
PHYSICS_SUBSTEPS=1 \
TRANSPORT_JVP_TERMS=1 \
bash experiments/scale/run_fm_s4_revised.sh
