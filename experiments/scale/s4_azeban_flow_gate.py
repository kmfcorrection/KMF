#!/usr/bin/env python3
"""Calibration/held-out gate for the discrete AZEBAN flow likelihood.

The current S4 posterior uses a midpoint time-residual.  That residual has a
large floor on the released Poseidon NS-Gauss trajectories, so it may point in
the wrong direction even when the spectral-viscosity filter is correct.  This
script tests the more faithful endpoint residual

    R_flow(x_{t+1}; x_t) = x_{t+1} - Phi_dt^AZEBAN(x_t),

where Phi is SSP-RK3 with the released smooth spectral viscosity and AZEBAN's
published CFL rule (C=0.5).  The reported number of internal steps provides a
direct stability check.  A held-out improvement over persistence is required
before using this likelihood in HILP correction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    POSEIDON_NATIVE_RAW_STRIDE, POSEIDON_REFERENCE_DT, native_poseidon_spec)
from hipp.scale.data2d import PDESpec2D
from hipp.scale.fm_physics import NativeFlowEnergy2D
from hipp.utils import Table, print_header, save_json, set_seed


def _load(path, n_traj, steps, lead, offset, device):
    path = Path(path)
    raw_stride = POSEIDON_NATIVE_RAW_STRIDE * lead
    required = steps * raw_stride + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution" if "solution" in h else None
        if key is None:
            raise KeyError(f"{path} has neither velocity nor solution")
        ds = h[key]
        if ds.ndim != 5 or ds.shape[2] < 2:
            raise ValueError(f"{key} has shape {ds.shape}; expected (sample,time,channel,x,y)")
        if required > ds.shape[1]:
            raise ValueError(f"need {required} snapshots but {path} contains {ds.shape[1]}")
        if offset + n_traj > ds.shape[0]:
            raise ValueError("requested calibration/test trajectories exceed the file")
        data = np.asarray(ds[offset:offset+n_traj, :required:raw_stride, :2], dtype=np.float64)
    return torch.as_tensor(data, device=device, dtype=torch.float64).reshape(n_traj, steps + 1, -1)


def _rms(x):
    return x.reshape(x.shape[0], -1).square().mean(1).sqrt()


def _evaluate(traj, spec, cfl, device):
    """Evaluate the published CFL flow; no configuration is fit here."""
    previous = traj[:, :-1].reshape(-1, traj.shape[-1])
    target = traj[:, 1:].reshape(-1, traj.shape[-1])
    # Every member retains its own adaptive CFL sequence inside the batched
    # integrator; this changes throughput, not the numerical method.
    energy = NativeFlowEnergy2D(spec, previous, channels=2, dt=spec.dt_out,
                                flow_cfl=cfl, device=device)
    pred, inner_steps = energy.flow_target, energy.flow_internal_steps
    if not torch.isfinite(pred).all():
        raise RuntimeError("CFL flow produced a non-finite state")
    state_err = _rms(pred - target)
    persistence_state = _rms(previous - target)
    vort_err = _rms(energy.to_vorticity(pred) - energy.to_vorticity(target))
    persistence_vort = _rms(energy.to_vorticity(previous) - energy.to_vorticity(target))
    mean = lambda values: float(values.mean())
    return {
        "state_flow_rmse": mean(state_err),
        "state_persistence_rmse": mean(persistence_state),
        "vorticity_flow_rmse": mean(vort_err),
        "vorticity_persistence_rmse": mean(persistence_vort),
        "mean_internal_steps": float(inner_steps.double().mean()),
        "max_internal_steps": int(inner_steps.max()),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm-data-path", required=True)
    ap.add_argument("--lead-steps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n-cal-traj", type=int, default=8)
    ap.add_argument("--n-test-traj", type=int, default=8)
    ap.add_argument("--cfl", type=float, default=0.5,
                    help="AZEBAN public configs use C=0.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.lead_steps < 1 or args.steps < 1:
        raise SystemExit("lead-steps and steps must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    # No FM checkpoint is used: this is a pure data/solver-consistency gate.
    spec = PDESpec2D("poseidon_ns_native", L=1.0, n=128, nu=0.0,
                     dt=POSEIDON_REFERENCE_DT * args.lead_steps, stride=1,
                     forcing="none", warmup=0, ic_peak_k=4.0)
    cal = _load(args.fm_data_path, args.n_cal_traj, args.steps, args.lead_steps, 0, device)
    test = _load(args.fm_data_path, args.n_test_traj, args.steps, args.lead_steps,
                 args.n_cal_traj, device)
    print_header("AZEBAN discrete-flow consistency gate")
    print(f"  endpoint interval={spec.dt_out:g}; lead={args.lead_steps}; "
          f"calibration trajectories={args.n_cal_traj}; held-out trajectories={args.n_test_traj}")
    print(f"  integrator=SSP-RK3; exact smooth spectral-viscosity filter; CFL={args.cfl:g}")
    cal_result = _evaluate(cal, spec, args.cfl, device)
    cal_result["state_gain_vs_persistence_pct"] = 100 * (1 - cal_result["state_flow_rmse"] / cal_result["state_persistence_rmse"])
    held = _evaluate(test, spec, args.cfl, device)
    held["state_gain_vs_persistence_pct"] = 100 * (1 - held["state_flow_rmse"] / held["state_persistence_rmse"])
    held["vorticity_gain_vs_persistence_pct"] = 100 * (1 - held["vorticity_flow_rmse"] / held["vorticity_persistence_rmse"])
    table = Table("split", "flow state RMSE", "persistence RMSE", "state gain%", "flow vort RMSE", "vort gain%", "inner steps")
    table.add("calibration", cal_result["state_flow_rmse"], cal_result["state_persistence_rmse"],
              cal_result["state_gain_vs_persistence_pct"], cal_result["vorticity_flow_rmse"],
              100 * (1 - cal_result["vorticity_flow_rmse"] / cal_result["vorticity_persistence_rmse"]),
              f"{cal_result['mean_internal_steps']:.1f}/{cal_result['max_internal_steps']}")
    table.add("held-out", held["state_flow_rmse"], held["state_persistence_rmse"],
              held["state_gain_vs_persistence_pct"], held["vorticity_flow_rmse"],
              held["vorticity_gain_vs_persistence_pct"],
              f"{held['mean_internal_steps']:.1f}/{held['max_internal_steps']}")
    print("\n" + str(table))
    verdict = ("PASS: discrete flow is a stronger physical endpoint model than persistence; "
               "implement a paired midpoint-vs-flow HILP posterior next."
               if held["state_gain_vs_persistence_pct"] > 0 else
               "FAIL: this fixed-step flow is not a stronger endpoint model than persistence; "
               "do not use it for correction yet.")
    print("\n  " + verdict)
    out = {"stage": "s4_azeban_discrete_flow_gate", "integrator": "ssprk3",
           "filter": "AZEBAN SmoothCutoff1D exponent 18, eps=0.05/N, s=1",
           "cfl": args.cfl, "calibration": cal_result, "held_out": held,
           "verdict": verdict}
    suffix = ("_" + args.tag) if args.tag else ""
    path = Path("results/scale/s4_azeban_flow_gate") / f"lead{args.lead_steps}{suffix}" / "results.json"
    save_json(out, path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
