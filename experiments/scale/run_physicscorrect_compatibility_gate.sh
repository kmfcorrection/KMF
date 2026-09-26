#!/usr/bin/env bash
# This is deliberately a compatibility gate, not a synthetic baseline.
# Set PHYSICSCORRECT_CMD to the exact documented command needed to run the
# public implementation on the same Poseidon state/data contract.
set -euo pipefail

OUT_DIR="${OUT_DIR:-results/audit_experiments/physicscorrect_compatibility}"
mkdir -p "${OUT_DIR}"

if [[ -z "${PHYSICSCORRECT_CMD:-}" ]]; then
  cat > "${OUT_DIR}/README.md" <<'EOF'
# PhysicsCorrect compatibility gate

No empirical PhysicsCorrect number was generated. To make a valid comparison,
provide `PHYSICSCORRECT_CMD` that runs the official implementation or a faithful
documented reimplementation with all of the following fixed:

1. Poseidon velocity-state representation `[u,v]` at 128x128;
2. identical NS-Gauss/FNS-KF trajectory-disjoint calibration/test split;
3. midpoint query at the same 0.10 s endpoint span;
4. known PDE operator access declared explicitly;
5. all optimization/backpropagation/solver time included in latency.

The command must write predictions and wall-clock time into this directory.
EOF
  echo "No PHYSICSCORRECT_CMD supplied; wrote ${OUT_DIR}/README.md. No comparison was fabricated."
  exit 0
fi

echo "Running user-supplied documented PhysicsCorrect command..."
eval "${PHYSICSCORRECT_CMD}" 2>&1 | tee "${OUT_DIR}/physicscorrect.log"
