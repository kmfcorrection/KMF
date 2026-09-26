#!/usr/bin/env python3
"""Fine-Cadence Super-Resolution Benchmark: Poseidon vs. Zero-ODE Kinematic Hermite Splines.

Evaluates continuous-time off-cadence queries t in [0.01s, 0.10s] starting from exact NS-Gauss initial conditions:
1. Generates ground-truth states u_true(t) at fine timestamps t in {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10}s
   using the exact native NS-Gauss spectral flow map (SSP-RK3, AZEBAN spectral viscosity).
2. Evaluates Poseidon raw direct forecast: fm.predict(u0, lead_time = t / 0.05)
3. Evaluates Linear Interpolation between u(0) and u(0.10s): (1-s)*u0 + s*u1
4. Evaluates Hermite Kinematic Spline: 0 forward ODE steps, exact spatial Navier-Stokes acceleration.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    configure_native_poseidon_cadence, native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.utils import Table, print_header, save_json, set_seed


def load_raw_cadence_trajectories(path: str | Path, fm, n_traj: int, steps: int = 2, offset: int = 0):
    """Load initial and boundary snapshots from NS-Gauss."""
    import h5py
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        ds = h[key]
        arr = np.asarray(ds[offset:offset + n_traj, :steps + 1, :2], dtype=np.float32)

    return torch.as_tensor(arr, device=fm.device, dtype=torch.float64)


def hermite_kinematic_spline(op: FMPhysicsEnergy2D, u0: torch.Tensor, u1: torch.Tensor,
                             dt: float, s: float) -> torch.Tensor:
    """Exact Hermite kinematic spline for arbitrary normalized intermediate time s in [0, 1].

    Preserves exact physical values and Navier-Stokes time derivatives at both endpoints
    with strictly 0 forward ODE integration steps.
    """
    w0 = op.to_vorticity(u0)
    w1 = op.to_vorticity(u1)
    f0 = op.rhs_vorticity(w0)
    f1 = op.rhs_vorticity(w1)

    h00 = 1.0 - 3.0 * s**2 + 2.0 * s**3
    h10 = 3.0 * s**2 - 2.0 * s**3
    h01 = dt * (s - 2.0 * s**2 + s**3)
    h11 = dt * (-s**2 + s**3)

    w_interp = h00 * w0 + h10 * w1 + h01 * f0 + h11 * f1
    wh_interp = torch.fft.rfft2(w_interp)
    u, v = op.grid.velocity(wh_interp)
    mean_vel = (1.0 - s) * u0.mean(dim=(-2, -1), keepdim=True) + s * u1.mean(dim=(-2, -1), keepdim=True)
    return torch.stack((u, v), dim=1) + mean_vel


def main():
    parser = base_parser_scale(__doc__.splitlines()[0])
    parser.add_argument("--lead-steps", type=int, default=1, help="Poseidon lead steps (default: 1)")
    parser.add_argument("--n-test-traj", type=int, default=8, help="number of test trajectories")
    parser.add_argument("--offset", type=int, default=0, help="dataset trajectory offset (default: 0)")
    parser.add_argument("--coarse-dt", type=float, default=0.10, help="coarse interval for interpolation")
    parser.add_argument("--fm-data-path", required=True, help="path to NS-Gauss.nc")
    args = parser.parse_args()

    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)

    print_header(f"Fine-Cadence Super-Resolution: {fm.info.name} (128x128)")
    print(f"  test trajectories: {args.n_test_traj}")
    print(f"  coarse interval: dt = {args.coarse_dt:.2f}s (t0=0.00s -> t1={args.coarse_dt:.2f}s)")
    print(f"  integration: STRICTLY ZERO FORWARD ODE SOLVES for Hermite Spline")

    trajs = load_raw_cadence_trajectories(args.fm_data_path, fm, args.n_test_traj, steps=2, offset=args.offset)
    # trajs: (N, 3, 2, 128, 128) where frame 0 is t=0.0s, frame 1 is t=0.05s, frame 2 is t=0.10s

    # Initialize physical operator
    op = FMPhysicsEnergy2D(spec, trajs[0, 0].to(fm.device, torch.float64), 2, dt=args.coarse_dt, device=fm.device)

    # Fine timestamps to evaluate
    fine_timestamps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    
    print("\n" + "=" * 115)
    print(f"BENCHMARK: SUB-CADENCE SUPER-RESOLUTION ACROSS FINE TIMESTAMPS t in [0.01s, {args.coarse_dt:.2f}s]")
    print("=" * 115)
    print("Generating exact fine ground-truth via native spectral flow map & comparing Poseidon vs. Hermite...")

    # Dictionary to store results for plotting
    benchmark_results = {
        "timestamps": fine_timestamps,
        "linear_rmse": [],
        "poseidon_rmse": [],
        "hermite_rmse": [],
        "gain_vs_linear": [],
        "gain_vs_poseidon": [],
        "latency_poseidon_ms": 0.0,
        "latency_hermite_ms": 0.0,
    }

    t_table = Table("Target t", "s = t/dt", "Linear RMSE", "Poseidon FM Raw", "Hermite Spline", "Consensus (Ours)", "Gain vs Lin", "Gain vs FM (Consensus)")

    poseidon_times = []
    hermite_times = []

    for t_target in fine_timestamps:
        s = t_target / args.coarse_dt
        lead_units = float(t_target / 0.05)  # Poseidon lead_time: 1.0 = 0.05s, 2.0 = 0.10s

        lin_errors = []
        fm_errors = []
        herm_errors = []
        cons_errors = []

        w_blend = 0.0 if s <= 0.4 else 0.5 * (math.sin(math.pi * (s - 0.4) / 0.6) ** 2)

        for i in range(args.n_test_traj):
            u0 = trajs[i, 0].to(fm.device, torch.float64).reshape(1, 2, spec.n, spec.n)
            u1 = trajs[i, 2].to(fm.device, torch.float64).reshape(1, 2, spec.n, spec.n)

            # 1. Exact Physical Ground Truth at t_target using native SSP-RK3 flow
            if math.isclose(t_target, 0.05, abs_tol=1e-5):
                u_true = trajs[i, 1].to(fm.device, torch.float64).reshape(1, 2, spec.n, spec.n)
            elif math.isclose(t_target, 0.10, abs_tol=1e-5):
                u_true = trajs[i, 2].to(fm.device, torch.float64).reshape(1, 2, spec.n, spec.n)
            else:
                op_step = FMPhysicsEnergy2D(spec, u0, 2, dt=t_target, device=fm.device)
                u_true = op_step.native_flow_state(cfl=0.5, integrator="ssprk3").reshape(1, 2, spec.n, spec.n)

            # 2. Linear Interpolation
            u_lin = (1.0 - s) * u0 + s * u1
            lin_errors.append(float((u_lin - u_true).square().mean().sqrt()))

            # 3. Poseidon FM Raw Direct Query
            fm.set_lead_time(lead_units)
            t_fm_start = time.time()
            with torch.no_grad():
                u_fm = fm.predict(u0.to(fm.dtype)).double().reshape(1, 2, spec.n, spec.n)
            u_fm = op.project_incompressible(u_fm).reshape(1, 2, spec.n, spec.n)
            poseidon_times.append((time.time() - t_fm_start) * 1000.0)
            fm_errors.append(float((u_fm - u_true).square().mean().sqrt()))

            # 4. Hermite Kinematic Spline (0 ODE steps)
            t_herm_start = time.time()
            u_herm = hermite_kinematic_spline(op, u0, u1, args.coarse_dt, s=s)
            hermite_times.append((time.time() - t_herm_start) * 1000.0)
            herm_errors.append(float((u_herm - u_true).square().mean().sqrt()))

            # 5. Consensus Spline (Physical-Neural Fusion)
            u_cons = (1.0 - w_blend) * u_herm + w_blend * u_fm
            u_cons = op.project_incompressible(u_cons).reshape(1, 2, spec.n, spec.n)
            cons_errors.append(float((u_cons - u_true).square().mean().sqrt()))

        mean_lin = float(np.mean(lin_errors))
        mean_fm = float(np.mean(fm_errors))
        mean_herm = float(np.mean(herm_errors))
        mean_cons = float(np.mean(cons_errors))

        gain_lin = 100.0 * (1.0 - mean_cons / mean_lin) if mean_lin > 1e-9 else 0.0
        gain_fm_cons = 100.0 * (1.0 - mean_cons / mean_fm) if mean_fm > 1e-9 else 0.0

        benchmark_results["linear_rmse"].append(mean_lin)
        benchmark_results["poseidon_rmse"].append(mean_fm)
        benchmark_results["hermite_rmse"].append(mean_herm)
        benchmark_results["consensus_rmse"] = benchmark_results.get("consensus_rmse", [])
        benchmark_results["consensus_rmse"].append(mean_cons)
        benchmark_results["gain_vs_linear"].append(gain_lin)
        benchmark_results["gain_vs_poseidon"].append(gain_fm_cons)

        gain_lin_str = "endpoint" if mean_lin < 1e-9 else f"{gain_lin:+.2f}%"
        gain_fm_str = "endpoint" if math.isclose(s, 1.0, abs_tol=1e-5) else f"{gain_fm_cons:+.2f}%"

        t_table.add(
            f"{t_target:.2f}s",
            f"{s:.2f}",
            f"{mean_lin:.6f}",
            f"{mean_fm:.6f}",
            f"{mean_herm:.6f}",
            f"{mean_cons:.6f}",
            gain_lin_str,
            gain_fm_str,
        )

    print(t_table)


    avg_fm_ms = float(np.mean(poseidon_times))
    avg_herm_ms = float(np.mean(hermite_times))
    benchmark_results["latency_poseidon_ms"] = avg_fm_ms
    benchmark_results["latency_hermite_ms"] = avg_herm_ms

    print(f"\nAverage Execution Latency per Query:")
    print(f"  Poseidon FM Raw:         {avg_fm_ms:.2f} ms")
    print(f"  Hermite Kinematic (Ours): {avg_herm_ms:.2f} ms  ({avg_fm_ms / max(avg_herm_ms, 1e-4):.1f}x speedup)")

    # Save output JSON
    out_dir = Path("results/scale/fine_cadence")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"poseidon_{fm.info.name}_{args.tag}_results.json"
    save_json(benchmark_results, out_file)
    print(f"\nSaved benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
