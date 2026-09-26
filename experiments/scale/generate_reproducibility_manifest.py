#!/usr/bin/env python3
"""Generate Pinned Reproducibility Manifest for KMF Benchmark and Audit Experiments."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "metadata": {
            "title": "KMF Zero-ODE Kinematic Manifold Fusion - Reproducibility Manifest",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "canonical_benchmark_source": {
            "path": "results/scale/master_matrix_20260914_073113/json/",
            "summary_csv": "results/scale/master_matrix_20260914_073113/json/master_benchmark_summary.csv",
            "summary_csv_sha256": file_sha256(root / "results/scale/master_matrix_20260914_073113/json/master_benchmark_summary.csv") if (root / "results/scale/master_matrix_20260914_073113/json/master_benchmark_summary.csv").exists() else "N/A",
            "total_benchmark_configurations": 200,
            "contract_breakdown": {
                "VERIFIED_PRIMARY_FM": 30,
                "CROSS_PDE_STRESS_TEST": 30,
                "EXPLORATORY_MODEL_ADAPTER": 140
            }
        },
        "model_and_adapter_contracts": {
            "Poseidon": {
                "sizes": ["T", "B", "L"],
                "native_pretraining_domain": "2D Incompressible Navier-Stokes & Kolmogorov turbulence",
                "fluid_contract": "VERIFIED_PRIMARY_FM",
                "cross_pde_contract": "CROSS_PDE_STRESS_TEST"
            },
            "DPOT": {
                "sizes": ["Ti", "S"],
                "adapter_class": "EXPLORATORY_MODEL_ADAPTER",
                "note": "Downstream research adapter with wrapper normalization"
            },
            "MORPH": {
                "sizes": ["Ti", "S"],
                "adapter_class": "EXPLORATORY_MODEL_ADAPTER",
                "note": "Downstream research adapter with wrapper normalization"
            },
            "The_Well_Polymathic": {
                "families": ["TFNO", "UNetConvNext", "FNO"],
                "adapter_class": "EXPLORATORY_MODEL_ADAPTER",
                "note": "Downstream research adapter with wrapper normalization"
            }
        },
        "audit_experiment_scripts": {
            "suite_runner": "experiments/scale/audit_experiments_suite.py",
            "bash_launcher": "experiments/scale/run_audit_suite.sh",
            "report_generator": "experiments/scale/generate_audit_reports.py",
            "master_matrix_summarizer": "experiments/scale/summarize_master_matrix.py"
        },
        "manuscript_artifacts": {
            "latex_source": "iclr2027/iclr2027_conference.tex",
            "latex_sha256": file_sha256(root / "iclr2027/iclr2027_conference.tex") if (root / "iclr2027/iclr2027_conference.tex").exists() else "N/A",
            "overleaf_zip": "iclr2027/iclr2027_overleaf.zip",
            "overleaf_zip_sha256": file_sha256(root / "iclr2027/iclr2027_overleaf.zip") if (root / "iclr2027/iclr2027_overleaf.zip").exists() else "N/A"
        }
    }
    
    out_json = root / "docs/REPRODUCIBILITY_MANIFEST.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump(manifest, fp, indent=2)
        
    # Also write a clean markdown summary
    out_md = root / "docs/REPRODUCIBILITY_MANIFEST.md"
    md_content = f"""# KMF Reproducibility Manifest & Audit Evidence

- **Generated**: {manifest['metadata']['generated_at']}
- **Canonical Result Source**: `{manifest['canonical_benchmark_source']['path']}`
- **Master Summary CSV**: `{manifest['canonical_benchmark_source']['summary_csv']}` (SHA-256: `{manifest['canonical_benchmark_source']['summary_csv_sha256']}`)
- **Total Evaluated Benchmark Configurations**: **200**
  - **VERIFIED_PRIMARY_FM**: 30 configurations (`Poseidon-T/B/L` on `NS-Gauss` and `FNS-KF`)
  - **CROSS_PDE_STRESS_TEST**: 30 configurations (`Poseidon-T/B/L` on `ACE` and `Wave-Gauss`)
  - **EXPLORATORY_MODEL_ADAPTER**: 140 configurations (`DPOT-S/Ti`, `MORPH-S/Ti`, `Polymathic-FNO/TFNO/UNetConvNext` across all 4 PDEs)

## Verification Scripts
- **Audit Experiment Suite**: [`experiments/scale/audit_experiments_suite.py`](../experiments/scale/audit_experiments_suite.py)
- **Bash Cluster Launcher**: [`experiments/scale/run_audit_suite.sh`](../experiments/scale/run_audit_suite.sh)
- **LaTeX & CSV Report Generator**: [`experiments/scale/generate_audit_reports.py`](../experiments/scale/generate_audit_reports.py)
- **Canonical Matrix Summarizer**: [`experiments/scale/summarize_master_matrix.py`](../experiments/scale/summarize_master_matrix.py)

## LaTeX Manuscript Pinned Artifacts
- **Manuscript Source**: [`iclr2027/iclr2027_conference.tex`](../iclr2027/iclr2027_conference.tex) (SHA-256: `{manifest['manuscript_artifacts']['latex_sha256']}`)
- **Overleaf Bundle**: [`iclr2027/iclr2027_overleaf.zip`](../iclr2027/iclr2027_overleaf.zip) (SHA-256: `{manifest['manuscript_artifacts']['overleaf_zip_sha256']}`)
"""
    out_md.write_text(md_content)
    print(f"Generated reproducibility manifest at: {out_json} and {out_md}")


if __name__ == "__main__":
    main()
