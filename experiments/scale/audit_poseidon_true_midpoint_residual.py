#!/usr/bin/env python3
"""Check the native PDE residual at true stored temporal midpoint frames.

For raw NS-Gauss frames (u_0,u_1,u_2) at physical times (t,t+.05,t+.1),
compare, on the same 128^2 native spectral RHS,

  endpoint-average: (omega_2-omega_0)-.1 F((omega_0+omega_2)/2)
  true-midpoint:    (omega_2-omega_0)-.1 F(omega_1)

The second expression is diagnostic only: omega_1 is unavailable from a
native-cadence Poseidon endpoint at inference.  It isolates whether temporal
quadrature, rather than spatial resolution or the RHS itself, explains the
nonzero true-transition midpoint defect.  No FM is loaded and no PDE state is
advanced.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.fm_eval_common import POSEIDON_RAW_DT, POSEIDON_REFERENCE_DT
from hipp.scale.data2d import PDESpec2D
from hipp.scale.fm_physics import MidpointTransportEnergy2D
from hipp.utils import print_header, save_json


def _metric(values):
    x = torch.stack(values)
    return {"rms": float(x.square().mean().sqrt()),
            "median_sample_rms": float(x.flatten(1).square().mean(1).sqrt().median()),
            "mean_sample_rms": float(x.flatten(1).square().mean(1).sqrt().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--n-traj", type=int, default=32)
    parser.add_argument("--transitions", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/scale/poseidon_true_midpoint_residual"))
    parser.add_argument("--tag", default="true_midpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.n_traj < 1 or args.transitions < 1:
        raise SystemExit("n-traj and transitions must be positive")

    with h5py.File(args.data, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        ds = h[key]
        need = 2 * args.transitions + 1
        if args.offset + args.n_traj > ds.shape[0] or need > ds.shape[1]:
            raise SystemExit(f"requested {args.n_traj} trajectories / {need} raw frames outside {ds.shape}")
        raw = torch.from_numpy(np.asarray(ds[args.offset:args.offset + args.n_traj, :need, :2],
                                          dtype=np.float64))
    n = raw.shape[-1]
    spec = PDESpec2D("poseidon_ns_native", L=1.0, n=n, nu=0.0,
                     dt=POSEIDON_REFERENCE_DT, stride=1, forcing="none", warmup=0, ic_peak_k=4.0)
    print_header("Poseidon true spatiotemporal-midpoint residual audit")
    print(f"  data: {args.n_traj} trajectories x {args.transitions} native intervals; grid={n}x{n}")
    print(f"  raw data dt={POSEIDON_RAW_DT:g}; native interval dt={POSEIDON_REFERENCE_DT:g}")
    device = torch.device(args.device)
    print(f"  device={device}; no FM, learned model, or PDE endpoint integration is used")

    endpoint_average, true_midpoint, adjacent = [], [], []
    for t in range(args.transitions):
        # Each native interval across trajectories is independent, so one
        # batched spectral RHS call is mathematically identical to the old
        # serial loop and considerably faster on a GPU.
        x0 = raw[:, 2 * t].to(device).reshape(args.n_traj, -1)
        xm = raw[:, 2 * t + 1].to(device).reshape(args.n_traj, -1)
        x1 = raw[:, 2 * t + 2].to(device).reshape(args.n_traj, -1)
        energy = MidpointTransportEnergy2D(spec, x0, 2, dt=POSEIDON_REFERENCE_DT,
                                            substeps=1, divergence_weight=0.0,
                                            device=device)
        with torch.no_grad():
            w0, wm, w1 = (energy.to_vorticity(z) for z in (x0, xm, x1))
            endpoint_average.append((w1 - w0 - POSEIDON_REFERENCE_DT *
                                     energy.rhs_vorticity(.5 * (w0 + w1))).cpu())
            true_midpoint.append((w1 - w0 - POSEIDON_REFERENCE_DT *
                                  energy.rhs_vorticity(wm)).cpu())
            # Adjacent stored-frame midpoint defects at their native raw cadence.
            adjacent.append((wm - w0 - POSEIDON_RAW_DT *
                             energy.rhs_vorticity(.5 * (w0 + wm))).cpu())
            adjacent.append((w1 - wm - POSEIDON_RAW_DT *
                             energy.rhs_vorticity(.5 * (wm + w1))).cpu())

    report = {"endpoint_average": _metric(endpoint_average),
              "true_stored_midpoint": _metric(true_midpoint),
              "adjacent_raw_midpoint": _metric(adjacent)}
    print("\nmethod                    RMS defect   median sample RMS")
    for name, q in report.items():
        print(f"{name:25s} {q['rms']:>11.5g} {q['median_sample_rms']:>18.5g}")
    ratio = report["true_stored_midpoint"]["rms"] / max(report["endpoint_average"]["rms"], 1e-30)
    print(f"\n  true-stored-midpoint / endpoint-average RMS = {ratio:.5g}")
    print("  A ratio near zero means midpoint temporal-state approximation was the main defect;")
    print("  a ratio near one means the remaining mismatch is the discrete AZEBAN time update.")
    payload = {"purpose": "diagnose true midpoint temporal-state versus endpoint-average residual at same native grid/RHS",
               "no_fm": True, "no_pde_endpoint_solver": True, "grid": int(n),
               "raw_dt": POSEIDON_RAW_DT, "native_dt": POSEIDON_REFERENCE_DT,
               "n_trajectories": args.n_traj, "native_transitions_per_trajectory": args.transitions,
               "metrics": report, "true_midpoint_over_endpoint_average": ratio}
    path = save_json(payload, args.out_dir / f"{args.tag}_results.json")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
