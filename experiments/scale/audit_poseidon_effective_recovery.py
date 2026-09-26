#!/usr/bin/env python3
"""Teacher-forced GMRES recovery gate for the effective-discrepancy closure.

This is the necessary post-closure diagnostic:

  e_hat = M_x(y)^(-1) [R_x(y) - b_hat_eff(x,y)],

scored against e=omega(y)-omega(y*) on trajectory-disjoint validation
transitions.  It does not alter a rollout, use a future state at inference, or
advance a PDE endpoint.  Truth is used only to construct calibration targets
and score this diagnostic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.audit_poseidon_effective_discrepancy import (
    ShellSpectralRidge, _apply_M, _defect, _energy, _examples,
    _inference_features, _metrics, _predict)
from experiments.scale.fm_eval_common import (configure_native_poseidon_cadence,
                                               load_poseidon_trajectories,
                                               native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm
from hipp.utils import print_header, save_json, set_seed


def _gmres(apply, rhs, maxiter, tol):
    shape, b = rhs.shape, rhs.reshape(-1)
    beta = float(b.norm())
    if beta < 1e-30:
        return torch.zeros_like(rhs), {"iterations": 0., "relative_residual": 0.}
    vectors = [b / beta]
    h = torch.zeros((maxiter + 1, maxiter), dtype=rhs.dtype, device=rhs.device)
    target = torch.zeros(maxiter + 1, dtype=rhs.dtype, device=rhs.device)
    target[0] = beta
    best, relative = torch.zeros_like(b), float("inf")
    for j in range(maxiter):
        w = apply(vectors[j].reshape(shape)).reshape(-1)
        for i in range(j + 1):
            h[i, j] = torch.dot(vectors[i], w)
            w -= h[i, j] * vectors[i]
        for i in range(j + 1):
            correction = torch.dot(vectors[i], w)
            h[i, j] += correction
            w -= correction * vectors[i]
        h[j + 1, j] = w.norm()
        if float(h[j + 1, j]) > 1e-30 and j + 1 < maxiter:
            vectors.append(w / h[j + 1, j])
        coefficient = torch.linalg.lstsq(h[:j + 2, :j + 1], target[:j + 2]).solution
        best = torch.stack(vectors[:j + 1], dim=1) @ coefficient
        relative = float((apply(best.reshape(shape)).reshape(-1) - b).norm() / beta)
        if relative <= tol or float(h[j + 1, j]) <= 1e-30:
            return best.reshape(shape), {"iterations": float(j + 1), "relative_residual": relative}
    return best.reshape(shape), {"iterations": float(maxiter), "relative_residual": relative}


def _recovery_metrics(fm, trajectories, spec, args, model):
    estimates, errors, residuals, iterations = [], [], [], []
    for i, trajectory in enumerate(trajectories):
        print(f"    validation trajectory {i + 1}/{len(trajectories)}", flush=True)
        for t in range(args.steps):
            current = trajectory[t]
            truth = trajectory[t + 1].reshape(1, -1).to(fm.device, torch.float64)
            energy = _energy(spec, current, fm.state_shape[0], fm.device)
            raw = _predict(fm, current)
            with torch.no_grad():
                features = _inference_features(energy, raw).unsqueeze(0)
                rhs = _defect(energy, raw) - model.predict(features)
                error = energy.to_vorticity(raw) - energy.to_vorticity(truth)
            estimate, info = _gmres(lambda z: _apply_M(energy, raw, z), rhs,
                                    args.gmres_iters, args.gmres_tol)
            estimates.append(estimate.detach().cpu())
            errors.append(error.detach().cpu())
            residuals.append(info["relative_residual"])
            iterations.append(info["iterations"])
    metric = _metrics(torch.cat(estimates), torch.cat(errors))
    metric.update({"gmres_relative_residual": float(np.mean(residuals)),
                   "gmres_iterations": float(np.mean(iterations))})
    return metric


def main():
    parser = base_parser_scale(__doc__.splitlines()[0])
    parser.add_argument("--lead-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--n-cal-traj", type=int, default=32)
    parser.add_argument("--n-val-traj", type=int, default=16)
    parser.add_argument("--shells", type=int, default=8)
    parser.add_argument("--ridge-grid", nargs="+", type=float,
                        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.])
    parser.add_argument("--gmres-iters", type=int, default=192)
    parser.add_argument("--gmres-tol", type=float, default=1e-5)
    parser.add_argument("--fm-data-path", required=True)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/scale/poseidon_effective_recovery_audit"))
    args = parser.parse_args()
    if args.fm != "poseidon" or args.fm_channels != "velocity":
        raise SystemExit("this audit supports Poseidon velocity only")
    if min(args.steps, args.n_cal_traj, args.n_val_traj, args.shells, args.gmres_iters) < 1:
        raise SystemExit("all counts, shells, and gmres-iters must be positive")
    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj, args.steps, args.lead_steps, 0)
    val = load_poseidon_trajectories(args.fm_data_path, fm, args.n_val_traj, args.steps, args.lead_steps, args.n_cal_traj)
    print_header(f"Poseidon effective-discrepancy GMRES recovery gate: {fm.info.name}")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}; transitions/trajectory={args.steps}")
    print("  score: GMRES[M, R(raw)-b_hat_eff] against held-out teacher-forced vorticity error")
    print("  no endpoint PDE solve; future truth is calibration/validation diagnostic only")
    print("\n  building calibration effective-discrepancy examples")
    cal_features, cal_target, _ = _examples(fm, cal, spec, args, "calibration")
    candidates = []
    models = [("constant", None, ShellSpectralRidge.constant(cal_target))]
    models += [("shell_ridge", float(ridge), ShellSpectralRidge.fit(
        cal_features, cal_target, args.shells, ridge, f"shell_effective_ridge{ridge:g}"))
               for ridge in args.ridge_grid]
    print("\n  evaluating teacher-forced recovery on validation")
    for kind, ridge, model in models:
        metric = _recovery_metrics(fm, val, spec, args, model)
        candidates.append((kind, ridge, model, metric))
        label = "constant" if ridge is None else f"ridge={ridge:g}"
        print(f"    {label:12s}: recovery rel.error={metric['relative_error_mean']:.6g}; "
              f"cosine={metric['direction_cosine_mean']:+.4f}; "
              f"GMRES residual={metric['gmres_relative_residual']:.2e}")
    kind, ridge, model, best = min(candidates, key=lambda row: row[3]["relative_error_mean"])
    print(f"\n  selected {model.label}: recovery rel.error={best['relative_error_mean']:.6g}; "
          f"cosine={best['direction_cosine_mean']:+.4f}")
    print("  PASS requires materially sub-unit recovered relative error and a strongly positive cosine.")
    payload = {"purpose": "teacher-forced GMRES recovery gate after effective-discrepancy prediction",
               "operator": "M=I-dt/2 J_F(raw midpoint), matrix-free JVP only",
               "target": "held-out vorticity FM error; diagnostic only",
               "no_pde_endpoint_solver": True,
               "n_calibration_examples": int(cal_target.shape[0]),
               "n_validation_examples": int(args.n_val_traj * args.steps),
               "candidates": [{"kind": k, "ridge": r, "validation_recovery": m}
                              for k, r, _, m in candidates],
               "selected": {"kind": kind, "ridge": ridge, "validation_recovery": best}}
    path = save_json(payload, args.out_dir / f"poseidon_{args.fm_size}_{args.tag}_results.json")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
