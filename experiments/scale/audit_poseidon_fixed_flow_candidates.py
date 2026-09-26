#!/usr/bin/env python3
"""Validation-only screen for deliberately imperfect fixed-RK3 flow defects.

This does not load an FM and never sees held-out trajectories.  It selects
which fixed numerical budget is worth evaluating inside the separate HILP
experiment, without tuning a candidate after looking at held-out RMSE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.fm_eval_common import POSEIDON_NATIVE_RAW_STRIDE, POSEIDON_REFERENCE_DT
from hipp.scale.data2d import PDESpec2D
from hipp.scale.fm_physics import FixedRK3FlowEnergy2D
from hipp.utils import Table, print_header, save_json


def _load(path: Path, n_traj: int, steps: int, offset: int, device: torch.device):
    need = 1 + steps * POSEIDON_NATIVE_RAW_STRIDE
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        ds = h[key]
        data = np.asarray(ds[offset:offset + n_traj, :need:POSEIDON_NATIVE_RAW_STRIDE, :2],
                          dtype=np.float64)
    if data.shape[:2] != (n_traj, steps + 1):
        raise ValueError(f"requested {(n_traj, steps + 1)} native states, received {data.shape}")
    return torch.as_tensor(data, dtype=torch.float64, device=device).reshape(n_traj, steps + 1, -1)


def _score(trajectories: torch.Tensor, spec: PDESpec2D, substeps: int, device: torch.device):
    previous = trajectories[:, :-1].reshape(-1, trajectories.shape[-1])
    truth = trajectories[:, 1:].reshape(-1, trajectories.shape[-1])
    energy = FixedRK3FlowEnergy2D(spec, previous, 2, flow_substeps=substeps,
                                  divergence_weight=0.0, device=device)
    flow_rmse = float((energy.flow_target - truth).square().mean().sqrt())
    persistence_rmse = float((previous - truth).square().mean().sqrt())
    return {"flow_rmse": flow_rmse, "persistence_rmse": persistence_rmse,
            "gain_pct": 100 * (1 - flow_rmse / max(persistence_rmse, 1e-30)),
            "fixed_substeps": int(substeps)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm-data-path", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n-cal-traj", type=int, default=32)
    ap.add_argument("--n-val-traj", type=int, default=16)
    ap.add_argument("--candidate-substeps", nargs="+", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="fixed_flow_screen")
    ap.add_argument("--out-dir", type=Path, default=Path("results/scale/poseidon_fixed_flow_screen"))
    args = ap.parse_args()
    if args.steps < 1 or args.n_cal_traj < 1 or args.n_val_traj < 1:
        raise SystemExit("steps, n-cal-traj, and n-val-traj must all be positive")
    if any(k < 1 for k in args.candidate_substeps):
        raise SystemExit("all candidate substep budgets must be positive")
    device = torch.device(args.device)
    cal = _load(args.fm_data_path, args.n_cal_traj, args.steps, 0, device)
    val = _load(args.fm_data_path, args.n_val_traj, args.steps, args.n_cal_traj, device)
    n = int(round((cal.shape[-1] // 2) ** 0.5))
    spec = PDESpec2D("poseidon_ns_native", L=1.0, n=n, nu=0.0, dt=POSEIDON_REFERENCE_DT,
                     stride=1, forcing="none", warmup=0, ic_peak_k=4.0)
    print_header("Poseidon fixed-RK3 flow-defect candidate screen")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}; "
          f"transitions/trajectory={args.steps}; grid={n}x{n}")
    print("  validation-only candidate screen; no FM and no held-out trajectories loaded")
    rows = []
    table = Table("substeps", "cal flow RMSE", "val flow RMSE", "val persistence", "val gain%")
    for budget in args.candidate_substeps:
        cal_score = _score(cal, spec, budget, device)
        val_score = _score(val, spec, budget, device)
        rows.append({"substeps": budget, "calibration": cal_score, "validation": val_score})
        table.add(budget, cal_score["flow_rmse"], val_score["flow_rmse"],
                  val_score["persistence_rmse"], val_score["gain_pct"])
    print("\n" + str(table))
    print("  Choose HILP_SUBSTEPS before held-out evaluation: the candidate must improve "
          "on persistence but must be assessed alongside the RK3-only runtime/control.")
    path = save_json({"stage": "fixed_rk3_flow_candidate_screen", "held_out_loaded": False,
                      "candidate_results": rows, "steps": args.steps,
                      "calibration_trajectories": args.n_cal_traj,
                      "validation_trajectories": args.n_val_traj},
                     args.out_dir / f"{args.tag}_results.json")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
