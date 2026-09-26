#!/usr/bin/env python3
"""Diagnose matrix-free GMRES inversion of Poseidon's sparse residual.

For teacher-forced frozen-Poseidon endpoints ``y=FM(x)``, let ``y*`` be the
held-out next state and define the vorticity midpoint defect

    R_x(y) = omega(y)-omega(x)-dt F((omega(x)+omega(y))/2).

At each candidate endpoint the local transport operator is

    M = I - dt/2 J_F((omega(x)+omega(y))/2).

This audit deliberately separates three questions:

* inverse recovery solves ``M z=M e`` for the known error ``e=omega(y)-omega(y*)``;
* truth-centred recovery solves ``M z=R_x(y)-R_x(y*)``;
* deployable recovery solves ``M z=R_x(y)-b_cal``.

Only the final right-hand side is available at inference.  Truth is used in
the first two rows strictly as a held-out diagnostic.  No branch advances a
PDE endpoint or calls AZEBAN/RK flow integration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.fm_eval_common import (configure_native_poseidon_cadence,
                                               load_poseidon_trajectories,
                                               native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm
from hipp.scale.fm_physics import MidpointTransportEnergy2D
from hipp.utils import set_seed


def _gmres(apply, rhs: torch.Tensor, maxiter: int,
           tol: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Unrestarted, re-orthogonalized matrix-free GMRES for one field.

    Kept locally in this GPU-facing audit: it must not depend on the Mac-only
    synthetic-data diagnostics.  A second modified Gram--Schmidt pass is used
    because the midpoint transport operator can be non-normal.
    """
    shape = rhs.shape
    b = rhs.reshape(-1)
    beta = float(b.norm())
    if beta < 1e-30:
        return torch.zeros_like(rhs), {"iterations": 0., "relative_residual": 0.}
    vectors = [b / beta]
    h = torch.zeros((maxiter + 1, maxiter), dtype=rhs.dtype, device=rhs.device)
    target = torch.zeros(maxiter + 1, dtype=rhs.dtype, device=rhs.device)
    target[0] = beta
    best, rel = torch.zeros_like(b), float("inf")
    for j in range(maxiter):
        w = apply(vectors[j].reshape(shape)).reshape(-1)
        for i in range(j + 1):
            h[i, j] = torch.dot(vectors[i], w)
            w = w - h[i, j] * vectors[i]
        for i in range(j + 1):
            correction = torch.dot(vectors[i], w)
            h[i, j] = h[i, j] + correction
            w = w - correction * vectors[i]
        h[j + 1, j] = w.norm()
        if float(h[j + 1, j]) > 1e-30 and j + 1 < maxiter:
            vectors.append(w / h[j + 1, j])
        solution = torch.linalg.lstsq(h[:j + 2, :j + 1], target[:j + 2]).solution
        basis = torch.stack(vectors[:j + 1], dim=1)
        best = basis @ solution
        rel = float((apply(best.reshape(shape)).reshape(-1) - b).norm() / beta)
        if rel <= tol or float(h[j + 1, j]) <= 1e-30:
            return best.reshape(shape), {"iterations": float(j + 1), "relative_residual": rel}
    return best.reshape(shape), {"iterations": float(maxiter), "relative_residual": rel}


def _rel_cos(estimate: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    a, b = estimate.reshape(-1), target.reshape(-1)
    return (float((a - b).norm() / b.norm().clamp_min(1e-30)),
            float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30)))


def _predict(fm, current: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return fm.predict(current.to(fm.device, fm.dtype)).double().reshape(1, -1)


def _defect(energy: MidpointTransportEnergy2D, endpoint: torch.Tensor) -> torch.Tensor:
    return energy.midpoint_endpoint_defect(energy.to_vorticity(endpoint))


def _apply_M(energy: MidpointTransportEnergy2D, endpoint: torch.Tensor,
             vector: torch.Tensor) -> torch.Tensor:
    """Apply I-dt/2 J_F at the raw candidate midpoint, using one RHS JVP."""
    omega = energy.to_vorticity(endpoint)
    midpoint = .5 * (energy.previous_vorticity + omega)
    _, jv = torch.func.jvp(energy.rhs_vorticity, (midpoint,), (vector,))
    return vector - .5 * energy.dt * jv


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _calibration_bias(fm, calibration: torch.Tensor, spec, args) -> torch.Tensor:
    values = []
    for trajectory in calibration:
        for t in range(args.steps):
            energy = MidpointTransportEnergy2D(
                spec, trajectory[t], 2, dt=spec.dt_out, substeps=1,
                divergence_weight=0.0, device=fm.device)
            truth = trajectory[t + 1].reshape(1, -1).to(fm.device, torch.float64)
            with torch.no_grad():
                values.append(_defect(energy, truth)[0])
    return torch.stack(values).mean(0).detach()


def main() -> None:
    parser = base_parser_scale(__doc__.splitlines()[0])
    parser.add_argument("--lead-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--n-cal-traj", type=int, default=32)
    parser.add_argument("--n-test-traj", type=int, default=32)
    parser.add_argument("--gmres-iters", type=int, default=96)
    parser.add_argument("--gmres-tol", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/scale/poseidon_gmres_transport_audit"))
    parser.add_argument("--fm-data-path", required=True)
    args = parser.parse_args()
    if args.fm != "poseidon" or args.fm_channels != "velocity":
        raise SystemExit("this audit is defined for Poseidon velocity on native NS-Gauss")
    if min(args.steps, args.n_cal_traj, args.n_test_traj, args.gmres_iters) < 1:
        raise SystemExit("steps, trajectory counts, and gmres-iters must be positive")

    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    calibration = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj,
                                              args.steps, args.lead_steps, offset=0)
    held_out = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj,
                                           args.steps, args.lead_steps,
                                           offset=args.n_cal_traj)
    print("=" * 78)
    print(f"Poseidon GMRES residual-transport audit: {fm.info.name}")
    print("=" * 78)
    print(f"  split: calibration={args.n_cal_traj}, held-out={args.n_test_traj}; "
          f"teacher-forced transitions/trajectory={args.steps}; GMRES max iterations={args.gmres_iters}")
    print("  M=I-dt/2 J_F(raw midpoint); matrix-free RHS JVPs only; no PDE endpoint solve")
    bias = _calibration_bias(fm, calibration, spec, args)
    print(f"  calibration truth-defect bias RMS={float(bias.square().mean().sqrt()):.5g}")

    rows = {"inverse_recovery": [], "truth_centered": [], "deployable": [], "state": []}
    started = time.time()
    for i, trajectory in enumerate(held_out):
        print(f"  held-out trajectory {i + 1}/{len(held_out)}", flush=True)
        for t in range(args.steps):
            current = trajectory[t]
            truth = trajectory[t + 1].reshape(1, -1).to(fm.device, torch.float64)
            raw = _predict(fm, current)
            energy = MidpointTransportEnergy2D(
                spec, current, 2, dt=spec.dt_out, substeps=1,
                divergence_weight=0.0, device=fm.device)
            with torch.no_grad():
                error = energy.to_vorticity(raw) - energy.to_vorticity(truth)
                r_raw, r_truth = _defect(energy, raw), _defect(energy, truth)
            apply = lambda vector: _apply_M(energy, raw, vector)
            with torch.no_grad():
                transported = error - apply(error)
                a_gain = float(transported.norm() / error.norm().clamp_min(1e-30))
                raw_state_rms = float((raw - truth).square().mean().sqrt() /
                                      truth.square().mean().sqrt().clamp_min(1e-30))
            diagnostics = {
                "inverse_recovery": apply(error),
                "truth_centered": r_raw - r_truth,
                "deployable": r_raw - bias,
            }
            for name, rhs in diagnostics.items():
                estimate, info = _gmres(apply, rhs, args.gmres_iters, args.gmres_tol)
                relative_error, cosine = _rel_cos(estimate, error)
                rows[name].append({"recovered_relative_error": relative_error,
                                   "recovered_cosine": cosine, **info})
            rows["state"].append({"raw_velocity_error_over_truth_rms": raw_state_rms,
                                  "A_on_vorticity_error_gain": a_gain})

    summary = {name: _mean(values) for name, values in rows.items()}
    print("\nmethod             GMRES residual  recovered error  cosine     iterations")
    for name in ("inverse_recovery", "truth_centered", "deployable"):
        q = summary[name]
        print(f"{name:18s} {q['relative_residual']:>12.3e} {q['recovered_relative_error']:>15.4g} "
              f"{q['recovered_cosine']:>+8.4f} {q['iterations']:>13.1f}")
    s = summary["state"]
    print(f"\n  raw velocity error / truth RMS={s['raw_velocity_error_over_truth_rms']:.4g}; "
          f"||A e_omega||/||e_omega||={s['A_on_vorticity_error_gain']:.4g}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "separate GMRES inversion, midpoint-residual linearization, and deployable calibration discrepancy on actual Poseidon outputs",
        "operator": "M=I-dt/2 J_F(raw midpoint), applied matrix-free through known-PDE RHS JVPs",
        "inference_available_row": "deployable",
        "truth_usage": "inverse_recovery and truth_centered are offline diagnostics only",
        "no_pde_endpoint_solver": True,
        "calibration_bias_rms": float(bias.square().mean().sqrt()),
        "n_calibration_trajectories": args.n_cal_traj,
        "n_held_out_trajectories": args.n_test_traj,
        "teacher_forced_transitions_per_trajectory": args.steps,
        "n_held_out_transitions": int(args.n_test_traj * args.steps),
        "seconds": time.time() - started,
        "summary": summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
