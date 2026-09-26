#!/usr/bin/env python3
"""Fair endpoint-conditioned versus causal-solver audit on non-NS PDEs.

The two comparison groups are intentionally separate:
  endpoint_conditioned: FM endpoint, linear bridge, Hermite/KMF bridge, fusion
  causal_solver:        RK2/RK4 from the initial state only

KMF uses the observed/model-produced endpoint pair.  RK controls use only the
initial state and the same instantaneous PDE RHS.  Their accuracies and
latencies must therefore not be read as an apples-to-apples ranking.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.cross_fm_benchmark import (  # noqa: E402
    GenericPDEPhysics,
    fm_predict,
    general_kinematic_hermite_spline,
    load_dataset_trajectories,
)
from hipp.scale.common_scale import load_fm, set_seed  # noqa: E402


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).square().mean(dim=(1, 2, 3)).sqrt().mean())


def timed(fn, device: torch.device, warmup: int = 3, trials: int = 10) -> float:
    for _ in range(warmup):
        out = fn()
        if not torch.isfinite(out).all():
            return float("inf")
    sync(device)
    vals = []
    for _ in range(trials):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        if not torch.isfinite(out).all():
            return float("inf")
        vals.append(1000.0 * (time.perf_counter() - t0))
    return float(np.mean(vals))


def project(phys: GenericPDEPhysics, x: torch.Tensor) -> torch.Tensor:
    return phys.project_manifold(x)


def rk_step(phys: GenericPDEPhysics, x: torch.Tensor, h: float,
            order: int, substeps: int) -> torch.Tensor:
    y = x
    hs = h / float(substeps)
    for _ in range(substeps):
        if order == 2:
            k1 = phys.rhs(y)
            k2 = phys.rhs(y + 0.5 * hs * k1)
            y = y + hs * k2
        elif order == 4:
            k1 = phys.rhs(y)
            k2 = phys.rhs(y + 0.5 * hs * k1)
            k3 = phys.rhs(y + 0.5 * hs * k2)
            k4 = phys.rhs(y + hs * k3)
            y = y + hs * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        else:
            raise ValueError(order)
        y = project(phys, y)
    return y


def make_args(args: argparse.Namespace) -> argparse.Namespace:
    # These are the fields consumed by hipp.scale.common_scale.load_fm.
    return argparse.Namespace(
        fm="poseidon", fm_size=args.fm_size, grid=args.grid,
        lead_time=1.0, fm_channels="velocity", fm_checkpoint=None,
        fm_history=None, pde=args.pde, device=args.device,
        arch="fno2d", ckpt=None,
    )


@torch.no_grad()
def run_pde(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    fm = load_fm(make_args(args), device=device)
    path = Path(args.data_root) / f"{args.pde}.nc"
    if not path.exists():
        raise FileNotFoundError(path)

    total = args.n_cal + args.n_test
    traj, c_val = load_dataset_trajectories(
        path, fm, total, steps=2, stride=args.endpoint_stride,
        target_grid=args.grid, prehistory=0)
    if c_val is not None:
        c = torch.as_tensor(c_val, device=device, dtype=torch.float64)
        if c.ndim == 2:
            c = c.unsqueeze(0)
        # GenericPDEPhysics accepts a per-trajectory spatial wave-speed field.
        phys = GenericPDEPhysics(args.pde, n=args.grid, device=device,
                                 dtype=torch.float64)
        phys.c = c[:total]
    else:
        phys = GenericPDEPhysics(args.pde, n=args.grid, device=device,
                                 dtype=torch.float64)

    u0 = traj[:, 0]
    u1 = traj[:, args.endpoint_stride]
    truth = traj[:, 1]
    cal = slice(0, args.n_cal)
    test = slice(args.n_cal, total)
    u0t, u1t, trutht = u0[test], u1[test], truth[test]

    # The Poseidon native endpoint is queried at one native step.  The bridge
    # spans the dataset interval and is evaluated at its midpoint.
    if hasattr(fm, "set_lead_time"):
        fm.set_lead_time(1.0)
    fm_endpoint = project(phys, fm_predict(fm, u0, current_grid=args.grid))
    linear = project(phys, 0.5 * (u0 + fm_endpoint))
    bridge = project(phys, general_kinematic_hermite_spline(
        phys, u0, fm_endpoint, args.span_dt, s=0.5))
    w_num = ((fm_endpoint[cal] - bridge[cal]) * (truth[cal] - bridge[cal])).sum()
    w_den = (fm_endpoint[cal] - bridge[cal]).square().sum().clamp_min(1e-30)
    weight = float(torch.clamp(w_num / w_den, 0.0, 1.0))
    fused = project(phys, bridge + weight * (fm_endpoint - bridge))

    u0_one = u0[test][:1]
    endpoint_one = fm_endpoint[test][:1]
    endpoint_rows = []
    def fused_one() -> torch.Tensor:
        b = project(phys, general_kinematic_hermite_spline(
            phys, u0_one, endpoint_one, args.span_dt, s=0.5))
        return project(phys, b + weight * (endpoint_one - b))

    latency_fns = {
        # The FM forward pass is common to all endpoint-conditioned rows and
        # is reported separately by the main benchmark.  Do not time it here.
        "fm_endpoint": lambda: endpoint_one,
        "linear_bridge": lambda: project(phys, 0.5 * (u0_one + endpoint_one)),
        "kmf_hermite_bridge": lambda: project(
            phys, general_kinematic_hermite_spline(
                phys, u0_one, endpoint_one, args.span_dt, s=0.5)),
        "kmf_calibrated_fusion": fused_one,
    }
    for name, pred in (("fm_endpoint", fm_endpoint), ("linear_bridge", linear),
                       ("kmf_hermite_bridge", bridge),
                       ("kmf_calibrated_fusion", fused)):
        p = pred[test]
        endpoint_rows.append({
            "method": name, "rmse": rmse(p, trutht),
            "online_latency_ms": timed(latency_fns[name], device),
            "information_contract": "u0 plus FM endpoint u1_tilde",
        })

    # Causal controls predict the same midpoint from u0 only.
    solver_rows = []
    for order in (2, 4):
        for substeps in args.solver_substeps:
            pred = rk_step(phys, u0t, args.query_dt, order, substeps)
            solver_rows.append({
                "method": f"rk{order}_{substeps}substeps",
                "rmse": rmse(pred, trutht),
                "online_latency_ms": timed(
                    lambda order=order, substeps=substeps: rk_step(
                        phys, u0t[:1], args.query_dt, order, substeps), device),
                "rhs_evaluations": order * substeps,
                "information_contract": "u0 only",
            })

    # Full-span replay is a solver sanity check, not a midpoint comparison.
    replay = []
    for order in (2, 4):
        for substeps in args.solver_substeps:
            pred = rk_step(phys, u0t, args.span_dt, order, substeps)
            replay.append({
                "method": f"rk{order}_{substeps}substeps",
                "endpoint_rmse": rmse(pred, u1t),
                "finite": bool(torch.isfinite(pred).all()),
            })

    return {
        "pde": args.pde, "fm": "poseidon", "fm_size": args.fm_size,
        "grid": args.grid, "n_cal": args.n_cal, "n_test": args.n_test,
        "endpoint_stride": args.endpoint_stride, "span_dt": args.span_dt,
        "query_dt": args.query_dt, "calibrated_fusion_weight": weight,
        "comparison_groups": {
            "endpoint_conditioned": endpoint_rows,
            "causal_solver": solver_rows,
        },
        "causal_full_span_replay": replay,
        "latency_note": "FM forward pass excluded from endpoint-conditioned online timings; calibration is recorded separately as offline work.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/assembled")
    p.add_argument("--out-dir", default="results/audit_experiments/fair_solver_all_pdes")
    p.add_argument("--pdes", nargs="+", default=["FNS-KF", "ACE", "Wave-Gauss"])
    p.add_argument("--fm-size", default="B")
    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--n-cal", type=int, default=20)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--endpoint-stride", type=int, default=2)
    p.add_argument("--span-dt", type=float, default=0.10)
    p.add_argument("--query-dt", type=float, default=0.05)
    p.add_argument("--solver-substeps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260925)
    args = p.parse_args()
    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for pde in args.pdes:
        print(f"\n=== fair solver audit: {pde} ===", flush=True)
        args.pde = pde
        try:
            result = run_pde(args)
            results.append(result)
            for group, rows in result["comparison_groups"].items():
                print(group, flush=True)
                for row in rows:
                    print(f"  {row['method']}: RMSE={row['rmse']:.8g} latency_ms={row['online_latency_ms']:.3f}", flush=True)
        except Exception as exc:  # preserve other PDE results and report the exact failure
            failures.append({"pde": pde, "error": repr(exc)})
            print(f"[WARN] {pde} failed: {exc}", flush=True)
    payload = {"protocol": "fair_endpoint_conditioned_vs_causal_solver", "results": results, "failures": failures}
    target = out / "fair_solver_all_pdes.json"
    target.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {target}", flush=True)
    if failures:
        print("failures:", json.dumps(failures), flush=True)


if __name__ == "__main__":
    main()
