# KMF: Physics-Guided Inference-Time Debiasing for PDE Foundation Models

This repository contains the anonymous reproducibility release for Kinematic Manifold Fusion (KMF). KMF operates after a frozen PDE model and provides two capabilities:

1. temporal refinement between successive model endpoints using local spatial PDE evaluations; and
2. causal rollout correction with optional physical projection and defect relaxation.

Foundation-model parameters are never updated by the KMF pipeline. Checkpoints and benchmark data are intentionally not included. The `data/` directory is an empty placeholder for user-downloaded data.

## Repository layout

```text
experiments/scale/       benchmark runners, audits, and report utilities
hipp/scale/              KMF operators, PDE physics, adapters, and metrics
data/                    empty data location; see data/README.md
third_party/             optional external model source checkouts
requirements-scale.txt   core Python dependencies
```

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-scale.txt
python -m pip install huggingface_hub safetensors einops timm
export PYTHONPATH="$PWD:$PYTHONPATH"
```

GPU execution is recommended for the foundation-model benchmarks. The scripts also support CPU execution for adapter checks and small diagnostics.

## Data and checkpoint download

Data and pretrained weights are not redistributed here. Download them from their official project sources and place or cache them locally. The expected assembled benchmark files are:

```text
data/assembled/NS-Gauss.nc
data/assembled/FNS-KF.nc
data/assembled/ACE.nc
data/assembled/Wave-Gauss.nc
```

The benchmark accepts another location through `FM_DATA_PATH` or `DATA_ROOT`.

### Model source repositories

The adapters document the expected public sources and checkpoint identifiers:

```bash
git clone https://github.com/camlab-ethz/poseidon.git third_party/poseidon
git clone https://github.com/HaoZhongkai/DPOT.git third_party/dpot
git clone https://github.com/lanl/MORPH.git third_party/morph
```

Poseidon, DPOT, and MORPH weights are loaded from their public Hugging Face repositories by the adapters when network access is enabled. The The Well/Polymathic adapters use the corresponding public model repositories and require:

```bash
python -m pip install 'the_well[benchmark]'
```

CNO requires its released source, checkpoint, and configuration file. Pass those paths explicitly to the benchmark runner. The CNO configuration helper is `experiments/scale/prepare_cno_config.py`.

### Dataset provenance

The PDE data should be downloaded from the official dataset or model release associated with each benchmark, then converted or assembled into the filenames above. Do not commit downloaded data or model weights to this repository. Before running, verify the files and frame counts:

```bash
mkdir -p data/assembled
find data/assembled -maxdepth 1 -type f -print
```

If a release provides a different filename, either rename the local copy to the expected name or pass its path with `--fm-data-path`.

## Reproduce a single benchmark

```bash
python experiments/scale/cross_fm_benchmark.py \
  --fm poseidon \
  --fm-size B \
  --pde NS-Gauss \
  --coarse-dt 0.10 \
  --fm-data-path data/assembled/NS-Gauss.nc \
  --n-cal-traj 20 \
  --n-test-traj 50 \
  --steps 4 \
  --device cuda \
  --out-dir results/example
```

The protocol uses model-produced endpoints for Task 1. Dataset states are used only for scoring and for calibration/test splits as specified by the runner.

## Run the master benchmark

Set `DATA_ROOT` to the directory containing the assembled data, then select the model families, PDEs, cadences, and sizes to evaluate:

```bash
export DATA_ROOT="$PWD/data/assembled"
export HF_HOME="$PWD/.cache/huggingface"

FMS="poseidon polymathic" \
ALL_PDES="NS-Gauss FNS-KF ACE Wave-Gauss" \
POSEIDON_SIZES="T B L" \
POLYMATHIC_FAMILIES="TFNO FNO UNetConvNext" \
CADENCES="0.05 0.10 0.20" \
FIXED_STEP_CADENCES="0.10" \
GRIDS="128" \
N_CAL_TRAJ=20 \
N_TEST_TRAJ=50 \
STEPS=4 \
bash experiments/scale/run_master_iclr_benchmark.sh
```

The master runner writes JSON outputs and summaries under `results/`. Results are not committed by default.

## Supporting audits

The main supporting runners are:

- `run_fm_endpoint_refinement.sh`: matched endpoint refinement audit;
- `run_real_audit_suite.sh`: projection-matched real-pipeline audit;
- `run_reviewer_closure_suite.sh`: solver, mismatch, calibration, and cadence closure experiments;
- `run_task2_rollout_extension_validation.sh`: active rollout relaxation validation;
- `run_task2_correction_suite.sh`: correction-method and latency comparison;
- `run_fair_solver_comparison.sh`: endpoint-conditioned versus causal solver controls.

Each script accepts an output directory and records its configuration in JSON. Use trajectory-disjoint calibration and test splits when changing sample sizes or seeds.

## Reproducibility and responsibility

This release contains the implementation and experiment runners, not the private cluster paths, downloaded artifacts, or paper result files. Users should record the exact checkpoint revision, dataset revision, hardware, software environment, seed, and command line for any regenerated result.

