#!/usr/bin/env python3
"""Phase 1: calibration/held-out controls for coarse AZEBAN physics targets.

This gate does not load a foundation model.  It asks whether a coarse-flow
endpoint is useful because it evolves Navier--Stokes, rather than merely because
it low-pass filters and projects the current field.  All controls start from the
same observed current state and are scored against the same future frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import POSEIDON_NATIVE_RAW_STRIDE, POSEIDON_REFERENCE_DT
from hipp.scale.data2d import PDESpec2D
from hipp.scale.fm_physics import CoarseNativeFlowEnergy2D, spectral_resample_state
from hipp.utils import Table, print_header, save_json, set_seed


def _load(path, n_traj, steps, lead, offset, device):
    raw_stride = POSEIDON_NATIVE_RAW_STRIDE * lead
    need = steps * raw_stride + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        ds = h[key]
        if need > ds.shape[1] or offset + n_traj > ds.shape[0]:
            raise ValueError("requested split does not fit in the supplied trajectory file")
        a = np.asarray(ds[offset:offset + n_traj, :need:raw_stride, :2], dtype=np.float64)
    return torch.as_tensor(a, device=device).reshape(n_traj, steps + 1, -1)


def _rms(a):
    return a.reshape(a.shape[0], -1).square().mean(1).sqrt()


def _controls(previous, energy, n):
    """Endpoints sharing the same coarse cutoff, with no PDE evolution."""
    state = previous.reshape(-1, 2, n, n)
    lowpass = spectral_resample_state(
        spectral_resample_state(state, energy.physics_n), n).reshape(-1, 2 * n * n)
    projected = energy.project_incompressible(previous).reshape(-1, 2 * n * n)
    lowpass_projected = energy.project_incompressible(lowpass).reshape(-1, 2 * n * n)
    return {
        "persistence": previous,
        "helmholtz(raw)": projected,
        f"lowpass-{energy.physics_n}": lowpass,
        f"lowpass-{energy.physics_n}+helmholtz": lowpass_projected,
        f"AZEBAN-flow-{energy.physics_n}": energy.flow_target,
    }


def _evaluate(traj, spec, resolution, cfl, device):
    previous = traj[:, :-1].reshape(-1, traj.shape[-1])
    truth = traj[:, 1:].reshape(-1, traj.shape[-1])
    energy = CoarseNativeFlowEnergy2D(spec, previous, 2, resolution,
                                      flow_cfl=cfl, device=device)
    rows = {}
    for name, pred in _controls(previous, energy, spec.n).items():
        rows[name] = {
            "state_rmse": float(_rms(pred - truth).mean()),
            "vorticity_rmse": float(_rms(energy.to_vorticity(pred) - energy.to_vorticity(truth)).mean()),
            "divergence_rms": float(energy.divergence(pred).flatten(1).square().mean(1).sqrt().mean()),
        }
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm-data-path", required=True)
    ap.add_argument("--resolutions", nargs="+", type=int, default=[8, 4])
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n-cal-traj", type=int, default=8)
    ap.add_argument("--n-test-traj", type=int, default=8)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if any(r <= 0 or 128 % r for r in args.resolutions):
        raise SystemExit("every --resolutions value must divide 128")
    set_seed(args.seed)
    device = torch.device(args.device)
    spec = PDESpec2D("poseidon_ns_native", L=1.0, n=128, nu=0.0,
                     dt=POSEIDON_REFERENCE_DT * args.lead_steps, stride=1,
                     forcing="none", warmup=0, ic_peak_k=4.0)
    cal = _load(args.fm_data_path, args.n_cal_traj, args.steps, args.lead_steps, 0, device)
    test = _load(args.fm_data_path, args.n_test_traj, args.steps, args.lead_steps,
                 args.n_cal_traj, device)
    print_header("Phase 1: coarse-physics control gate")
    print("  all targets use the observed current state only; no FM is loaded")
    print(f"  interval={spec.dt_out:g}, CFL={args.cfl:g}, resolutions={args.resolutions}")
    out = {"stage": "phase1_physics_controls", "resolutions": args.resolutions,
           "lead_steps": args.lead_steps, "cfl": args.cfl, "splits": {}}
    for resolution in args.resolutions:
        print(f"\n  coarse resolution {resolution}x{resolution}")
        cal_rows = _evaluate(cal, spec, resolution, args.cfl, device)
        test_rows = _evaluate(test, spec, resolution, args.cfl, device)
        table = Table("control", "cal RMSE", "test RMSE", "test gain vs persistence%", "test div RMS")
        base = test_rows["persistence"]["state_rmse"]
        for name in cal_rows:
            row = test_rows[name]
            table.add(name, cal_rows[name]["state_rmse"], row["state_rmse"],
                      100 * (1 - row["state_rmse"] / base), row["divergence_rms"])
        print(table)
        out["splits"][str(resolution)] = {"calibration": cal_rows, "held_out": test_rows}
    suffix = f"_{args.tag}" if args.tag else ""
    p = Path("results/scale/phase1_physics_controls") / f"lead{args.lead_steps}{suffix}" / "results.json"
    save_json(out, p)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
