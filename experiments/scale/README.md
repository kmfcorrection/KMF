# Experiment runners

The scripts in this directory are grouped by their role:

## Core benchmark

- `cross_fm_benchmark.py` runs one frozen-model/PDE/cadence configuration.
- `run_master_iclr_benchmark.sh` enumerates selected model families, PDEs, sizes, and cadences.
- `summarize_master_matrix.py` aggregates JSON outputs into CSV summaries.

## Task 1: temporal refinement

- `run_fm_endpoint_refinement.sh`
- `validate_fm_endpoint_refinement.py`
- `run_real_audit_suite.sh`

These evaluate model-produced endpoint refinement. Dataset intermediate states are used for scoring, not supplied to the candidates as endpoints.

## Task 2: causal rollout correction

- `run_task2_rollout_extension_validation.sh`
- `run_task2_correction_suite.sh`
- `benchmark_task2_correction_suite.py`
- `run_fair_solver_comparison.sh`
- `run_fair_solver_all_pdes.sh`

These evaluate projection, KMF defect relaxation, reduced-state correction baselines, and complementary causal solver controls.

## Closure and diagnostic audits

The remaining `audit_*.py`, `run_*_audit.sh`, and reviewer-closure runners implement the supporting fidelity, calibration, cadence, mismatch, and latency checks. Every runner writes configuration metadata with its numerical output. Set `DATA_ROOT` or pass an explicit data path before running any script.
