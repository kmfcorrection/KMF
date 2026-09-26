#!/usr/bin/env python3
"""Audit a Poseidon midpoint PDE residual against a data-derived strong residual.

For every interior raw frame j, the reference uses a centered finite-difference
estimate of the *physical* time derivative and the native AZEBAN RHS:

  r_ref,j = d omega_j / dt - F_AZEBAN(omega_j).

The sixth-order, seven-point derivative is the reference.  The fourth-order,
five-point derivative is independently reported as a temporal-resolution
consistency check.  The compared inference-time residual is the ordinary
native-interval midpoint residual built only from the two endpoint states:

  r_mid,j = (omega_{j+1}-omega_{j-1}) / (2 dt)
            - F_AZEBAN((omega_{j-1}+omega_{j+1}) / 2).

This is a residual-to-residual audit.  It never compares against forecast
error, never loads an FM, and never advances a PDE state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.fm_eval_common import POSEIDON_RAW_DT
from hipp.scale.data2d import PDESpec2D
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.utils import Table, print_header, save_json


def _relative_and_cosine(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    e, r = estimate.flatten(1), reference.flatten(1)
    rel = (e - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)
    cosine = (e * r).sum(1) / (e.norm(dim=1) * r.norm(dim=1)).clamp_min(1e-30)
    return {
        "relative_error_mean": float(rel.mean()),
        "relative_error_median": float(rel.median()),
        "relative_error_p90": float(torch.quantile(rel, 0.9)),
        "direction_cosine_mean": float(cosine.mean()),
        "estimate_rms": float(estimate.square().mean().sqrt()),
        "reference_rms": float(reference.square().mean().sqrt()),
    }


def _d5(w: torch.Tensor, j: int, h: float) -> torch.Tensor:
    """Fourth-order centered first derivative at raw index j."""
    return (w[:, j - 2] - 8 * w[:, j - 1] + 8 * w[:, j + 1] - w[:, j + 2]) / (12 * h)


def _d7(w: torch.Tensor, j: int, h: float) -> torch.Tensor:
    """Sixth-order centered first derivative at raw index j."""
    return (-w[:, j - 3] + 9 * w[:, j - 2] - 45 * w[:, j - 1]
            + 45 * w[:, j + 1] - 9 * w[:, j + 2] + w[:, j + 3]) / (60 * h)


def _centers(n_raw: int, requested: list[int] | None, max_centers: int | None) -> list[int]:
    if requested:
        centers = requested
    else:
        # Odd raw indices are the actual temporal centers of native 0.1
        # intervals [j-1, j+1].  The 7-point reference needs j +/- 3.
        centers = list(range(3, n_raw - 3, 2))
    if any(j < 3 or j + 3 >= n_raw for j in centers):
        raise ValueError(f"centers must lie in [3, {n_raw - 4}] for the 7-point stencil")
    return centers[:max_centers] if max_centers is not None else centers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm-data-path", required=True, type=Path)
    ap.add_argument("--n-traj", type=int, default=32)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--centers", nargs="*", type=int,
                    help="raw-frame centers; defaults to every usable native-interval center")
    ap.add_argument("--max-centers", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/scale/poseidon_strong_residual_reference"))
    ap.add_argument("--tag", default="strong_residual_reference")
    args = ap.parse_args()
    if args.n_traj < 1:
        raise SystemExit("n-traj must be positive")
    device = torch.device(args.device)
    with h5py.File(args.fm_data_path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        ds = h[key]
        if ds.ndim != 5 or ds.shape[2] < 2:
            raise ValueError(f"expected (trajectory,time,channel,x,y), got {ds.shape}")
        if args.offset < 0 or args.offset + args.n_traj > ds.shape[0]:
            raise ValueError("requested trajectory slice is outside the dataset")
        raw = torch.from_numpy(np.asarray(ds[args.offset:args.offset + args.n_traj, :, :2],
                                          dtype=np.float64)).to(device)
    n_raw, n = int(raw.shape[1]), int(raw.shape[-1])
    centers = _centers(n_raw, args.centers, args.max_centers)
    spec = PDESpec2D("poseidon_ns_native", L=1.0, n=n, nu=0.0,
                     dt=2 * POSEIDON_RAW_DT, stride=1, forcing="none",
                     warmup=0, ic_peak_k=4.0)
    print_header("Poseidon strong PDE-residual reference audit")
    print(f"  split: trajectories={args.n_traj}, raw frames={n_raw}, grid={n}x{n}, device={device}")
    print(f"  raw dt={POSEIDON_RAW_DT:g}; reference centers={centers}")
    print("  reference: 7-point temporal derivative minus native AZEBAN RHS at the true frame")
    print("  candidate: endpoint-only 0.1 midpoint residual; no FM and no PDE endpoint solve")

    all_states = raw.reshape(args.n_traj * n_raw, -1)
    # This object is used only for its spectral curl/RHS operators; the
    # conditioning state has no role in the strong-form residual below.
    operator = FMPhysicsEnergy2D(spec, all_states, 2, divergence_weight=0.0, device=device)
    with torch.no_grad():
        all_w = operator.to_vorticity(all_states).reshape(args.n_traj, n_raw, n, n)

    ref5, ref7, midpoint_endpoint, midpoint_true_state = [], [], [], []
    for j in centers:
        # Constructing the operator with a batch is safe: RHS evaluation is
        # independent across trajectories and uses the raw physical velocity,
        # not Poseidon's internal normalized representation.
        with torch.no_grad():
            wm, w0, wp = all_w[:, j - 1], all_w[:, j], all_w[:, j + 1]
            ref5.append((_d5(all_w, j, POSEIDON_RAW_DT) - operator.rhs_vorticity(w0)).cpu())
            ref7.append((_d7(all_w, j, POSEIDON_RAW_DT) - operator.rhs_vorticity(w0)).cpu())
            midpoint_endpoint.append((((wp - wm) / (2 * POSEIDON_RAW_DT)
                                       - operator.rhs_vorticity(0.5 * (wm + wp))).cpu()))
            midpoint_true_state.append((((wp - wm) / (2 * POSEIDON_RAW_DT)
                                         - operator.rhs_vorticity(w0)).cpu()))

    ref5, ref7 = torch.cat(ref5), torch.cat(ref7)
    midpoint_endpoint, midpoint_true_state = torch.cat(midpoint_endpoint), torch.cat(midpoint_true_state)
    report = {
        "five_vs_seven_reference": _relative_and_cosine(ref5, ref7),
        "endpoint_midpoint_vs_reference": _relative_and_cosine(midpoint_endpoint, ref7),
        "true_state_midpoint_vs_reference": _relative_and_cosine(midpoint_true_state, ref7),
    }
    table = Table("comparison", "rel. mean", "rel. p90", "cosine", "estimate RMS", "reference RMS")
    for name, q in report.items():
        table.add(name, q["relative_error_mean"], q["relative_error_p90"],
                  q["direction_cosine_mean"], q["estimate_rms"], q["reference_rms"])
    print("\n" + str(table))
    consistency = report["five_vs_seven_reference"]
    print("\n  Reference reliability gate: 5-point vs 7-point derivative should agree "
          "(low relative error, cosine near +1).")
    print("  Only if that gate passes can the midpoint-vs-reference rows be interpreted.")
    payload = {
        "purpose": "strong-form PDE residual fidelity; never forecast-error fidelity",
        "future_truth_at_inference": False,
        "reference": "r_ref=d7(omega_true)/dt - F_native(omega_true)",
        "candidate": "r_mid=(omega_{j+1}-omega_{j-1})/(2dt)-F_native((omega_{j-1}+omega_{j+1})/2)",
        "n_trajectories": args.n_traj, "raw_dt": POSEIDON_RAW_DT,
        "centers": centers, "metrics": report,
        "reference_reliability": consistency,
    }
    path = save_json(payload, args.out_dir / f"{args.tag}_results.json")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
