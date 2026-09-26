#!/usr/bin/env python3
"""Evaluate KMF temporal refinement using adjacent FM-produced endpoints.

This is deliberately separate from the observation-endpoint benchmark.  For
each trajectory, an observed initial condition seeds an autoregressive FM
rollout at its native 0.10 s cadence.  Each adjacent pair of generated coarse
FM states is then used, without future data, to reconstruct its 0.05 s
midpoint.  Held-out truth is used only to score the completed reconstruction.

The calibration split selects one scalar fusion weight between the KMF bridge
and a direct off-cadence FM query.  Test trajectories never enter that choice.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.cross_fm_benchmark import (
    fluid_hermite_kinematic_spline,
    fm_predict,
    load_dataset_trajectories,
)
from experiments.scale.fm_eval_common import native_poseidon_spec
from hipp.scale.common_scale import load_fm, set_seed
from hipp.scale.data2d import SPECS2D
from hipp.scale.fm_physics import FMPhysicsEnergy2D


def project(op: FMPhysicsEnergy2D, x: torch.Tensor) -> torch.Tensor:
    return op.project_incompressible(x).reshape_as(x)


@torch.no_grad()
def build_fm_endpoint_packages(fm, trajectories: torch.Tensor, op: FMPhysicsEnergy2D,
                               *, native_stride: int, horizon: int, grid: int):
    """Create model-endpoint bridges and direct half-cadence FM candidates."""
    current = trajectories[:, 0].clone()
    bridges, directs, linears, targets = [], [], [], []
    for k in range(horizon):
        # Native model endpoint, spanning two stored 0.05 s frames by default.
        fm.set_lead_time(float(native_stride))
        endpoint = fm_predict(fm, current, current_grid=grid)

        # Direct off-cadence candidate.  This is a baseline, not an endpoint
        # supplied to the physical bridge.
        fm.set_lead_time(float(native_stride) / 2.0)
        direct = project(op, fm_predict(fm, current, current_grid=grid))

        bridge = project(op, fluid_hermite_kinematic_spline(
            op, current, endpoint, 0.05 * native_stride, s=0.5))
        bridges.append(bridge)
        directs.append(direct)
        linears.append(project(op, 0.5 * (current + endpoint)))
        targets.append(trajectories[:, k * native_stride + native_stride // 2])
        current = project(op, endpoint)
    return (torch.stack(bridges, dim=1), torch.stack(directs, dim=1),
            torch.stack(linears, dim=1), torch.stack(targets, dim=1))


def rmse_per_trajectory(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    return (pred.sub(truth).square().mean(dim=(1, 2, 3, 4)).sqrt().cpu().numpy())


def summarize(pred: torch.Tensor, truth: torch.Tensor) -> dict:
    per = rmse_per_trajectory(pred, truth)
    per_step = pred.sub(truth).square().mean(dim=(0, 2, 3, 4)).sqrt().cpu().numpy()
    return {"window_rmse": float(per.mean()), "trajectory_window_rmse": per.tolist(),
            "per_midpoint_rmse": per_step.tolist()}


def fit_fusion_weight(bridge: torch.Tensor, direct: torch.Tensor, truth: torch.Tensor) -> float:
    delta = (direct - bridge).reshape(-1)
    target = (truth - bridge).reshape(-1)
    denom = delta.square().sum()
    if float(denom) <= 1e-12:
        return 0.0
    return float((delta * target).sum().div(denom).clamp(0.0, 1.0))


def evaluate_system(args, fm, name: str, path: Path) -> dict:
    total = args.n_cal + args.n_test
    raw_steps = args.horizon * args.native_stride
    trajectories, _ = load_dataset_trajectories(
        path, fm, total, steps=raw_steps, stride=1, offset=args.offset,
        target_grid=args.grid,
    )
    calibration, test = trajectories[:args.n_cal], trajectories[args.n_cal:]
    if name == "NS-Gauss":
        spec = dataclasses.replace(native_poseidon_spec(args, fm), n=args.grid,
                                   dt=0.05 * args.native_stride)
    else:
        spec = dataclasses.replace(SPECS2D["kolmogorov"], n=args.grid,
                                   dt=0.05 * args.native_stride)
    op = FMPhysicsEnergy2D(spec, trajectories[:1, 0], 2, dt=spec.dt, device=fm.device)

    print(f"\n{'=' * 78}\nFM-endpoint temporal refinement: {name}\n{'=' * 78}")
    print(f"  endpoints: autoregressive FM states at {spec.dt:.2f}s; "
          f"queries: model-endpoint midpoints at {spec.dt / 2:.2f}s")
    print(f"  split: calibration={args.n_cal}, held-out={args.n_test}, horizon={args.horizon}")

    bridge_c, direct_c, linear_c, truth_c = build_fm_endpoint_packages(
        fm, calibration, op, native_stride=args.native_stride, horizon=args.horizon, grid=args.grid)
    weight = fit_fusion_weight(bridge_c, direct_c, truth_c)
    fusion_c = project(op, (1.0 - weight) * bridge_c.reshape(-1, 2, args.grid, args.grid)
                       + weight * direct_c.reshape(-1, 2, args.grid, args.grid)).reshape_as(bridge_c)

    bridge, direct, linear, truth = build_fm_endpoint_packages(
        fm, test, op, native_stride=args.native_stride, horizon=args.horizon, grid=args.grid)
    fusion = project(op, (1.0 - weight) * bridge.reshape(-1, 2, args.grid, args.grid)
                     + weight * direct.reshape(-1, 2, args.grid, args.grid)).reshape_as(bridge)
    methods = {
        "direct_offcadence_fm": direct,
        "fm_endpoint_linear": linear,
        "fm_endpoint_kmf_bridge": bridge,
        "fm_endpoint_kmf_fusion": fusion,
    }
    out = {
        "protocol": {
            "endpoint_source": "autoregressive FM rollout at native cadence",
            "initial_state": "observed initial condition",
            "midpoint_source": "KMF bridge uses only adjacent FM endpoints and PDE RHS evaluations",
            "truth_usage": "calibration selects scalar fusion weight; held-out truth scores completed predictions only",
        },
        "config": {"native_endpoint_dt": spec.dt, "query_dt": spec.dt / 2,
                   "native_stride_raw_frames": args.native_stride, "horizon": args.horizon,
                   "n_cal": args.n_cal, "n_test": args.n_test},
        "selected_fusion_weight": weight,
        "calibration": {name_: summarize(value, truth_c) for name_, value in {
            "fm_endpoint_kmf_bridge": bridge_c, "direct_offcadence_fm": direct_c,
            "fm_endpoint_kmf_fusion": fusion_c}.items()},
        "methods": {name_: summarize(value, truth) for name_, value in methods.items()},
    }
    base = np.asarray(out["methods"]["direct_offcadence_fm"]["trajectory_window_rmse"])
    for name_, value in out["methods"].items():
        per = np.asarray(value["trajectory_window_rmse"])
        value["gain_vs_direct_offcadence_fm_pct"] = float(100.0 * (1.0 - per.mean() / base.mean()))
    print(f"  selected calibration fusion weight={weight:.4f}")
    for name_, value in out["methods"].items():
        print(f"  {name_:28s} RMSE={value['window_rmse']:.6f} "
              f"gain-vs-direct={value['gain_vs_direct_offcadence_fm_pct']:+.2f}%")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fm", default="poseidon", choices=["poseidon"])
    p.add_argument("--fm-size", default="B", choices=["T", "B", "L"])
    p.add_argument("--fm-channels", default="velocity")
    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--native-stride", type=int, default=2)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--n-cal", type=int, default=20)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--offset", type=int, default=19760)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=20260924)
    args = p.parse_args()
    if args.native_stride < 2 or args.native_stride % 2:
        raise ValueError("--native-stride must be an even number of 0.05s raw frames")
    torch.manual_seed(args.seed); np.random.seed(args.seed); set_seed(args.seed)
    fm = load_fm(args, device=torch.device(args.device))
    results = {"metadata": vars(args)}
    for name in ("NS-Gauss", "FNS-KF"):
        path = Path(args.data_root) / f"{name}.nc"
        if path.exists():
            results[name] = evaluate_system(args, fm, name, path)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    result_path = out / "fm_endpoint_refinement.json"
    result_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {result_path}")


if __name__ == "__main__":
    main()
