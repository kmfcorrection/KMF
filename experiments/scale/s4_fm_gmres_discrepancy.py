#!/usr/bin/env python3
"""S4: deployable GMRES residual correction with calibrated discrepancy.

For an FM endpoint ``y`` conditioned on the available state ``x``, define

  R_x(y) = omega(y)-omega(x)-dt F((omega(x)+omega(y))/2),
  M_x(y) = I-dt/2 J_F((omega(x)+omega(y))/2).

The calibration-only discrepancy model predicts the candidate-conditioned
effective discrepancy

  b_eff(x,y) = R_x(y) - M_x(y)[omega(y)-omega(y*)].

At inference the correction is

  e_hat = GMRES(M_x(y), R_x(y)-b_theta(omega(x))),  y_corrected=y-g e_hat,

where the vorticity estimate is converted through Biot--Savart to a
divergence-free velocity correction and ``g`` is selected only on validation
trajectories.  This is residual evaluation plus matrix-free JVPs, *not* a PDE
endpoint solver.  Legacy S4/MAP pipelines are intentionally untouched.
"""
from __future__ import annotations

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

from experiments.scale.fm_eval_common import (configure_native_poseidon_cadence,
                                               fm_metadata, fm_result_key,
                                               load_poseidon_trajectories,
                                               native_poseidon_spec)
from experiments.scale.audit_poseidon_effective_discrepancy import (
    ShellSpectralRidge, _inference_features, _metrics, _one_example)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import MidpointTransportEnergy2D
from hipp.utils import Table, print_header, save_json, set_seed


def _gmres(apply, rhs: torch.Tensor, maxiter: int, tol: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Unrestarted, re-orthogonalized GMRES applied to one vorticity field."""
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
            w = w - h[i, j] * vectors[i]
        for i in range(j + 1):
            correction = torch.dot(vectors[i], w)
            h[i, j] += correction
            w = w - correction * vectors[i]
        h[j + 1, j] = w.norm()
        if float(h[j + 1, j]) > 1e-30 and j + 1 < maxiter:
            vectors.append(w / h[j + 1, j])
        coeff = torch.linalg.lstsq(h[:j + 2, :j + 1], target[:j + 2]).solution
        best = torch.stack(vectors[:j + 1], dim=1) @ coeff
        relative = float((apply(best.reshape(shape)).reshape(-1) - b).norm() / beta)
        if relative <= tol or float(h[j + 1, j]) <= 1e-30:
            return best.reshape(shape), {"iterations": float(j + 1), "relative_residual": relative}
    return best.reshape(shape), {"iterations": float(maxiter), "relative_residual": relative}


def _predict(fm, current: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return fm.predict(current.to(fm.device, fm.dtype)).double().reshape(1, -1)


def _energy(spec, previous, channels, device) -> MidpointTransportEnergy2D:
    return MidpointTransportEnergy2D(spec, previous, channels, dt=spec.dt_out,
                                     substeps=1, divergence_weight=0.0, device=device)


def _defect(energy: MidpointTransportEnergy2D, endpoint: torch.Tensor) -> torch.Tensor:
    return energy.midpoint_endpoint_defect(energy.to_vorticity(endpoint))


def _apply_M(energy: MidpointTransportEnergy2D, raw: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    midpoint = .5 * (energy.previous_vorticity + energy.to_vorticity(raw))
    _, jv = torch.func.jvp(energy.rhs_vorticity, (midpoint,), (vector,))
    return vector - .5 * energy.dt * jv


def _velocity_from_vorticity(energy: MidpointTransportEnergy2D, vorticity: torch.Tensor) -> torch.Tensor:
    """Biot--Savart lift of a vorticity-error estimate to [u,v] coordinates."""
    u, v = energy.grid.velocity(torch.fft.rfft2(vorticity))
    return torch.stack((u, v), dim=1).reshape(vorticity.shape[0], -1)


def _effective_examples(fm, trajectories, spec, args, label):
    """Use the same targets/features as the separately passed validation gate."""
    features, effective = [], []
    for i, trajectory in enumerate(trajectories):
        print(f"    {label} trajectory {i + 1}/{len(trajectories)}", flush=True)
        for t in range(args.steps):
            feat, target, _ = _one_example(fm, spec, trajectory[t], trajectory[t + 1])
            features.append(feat)
            effective.append(target)
    return torch.stack(features), torch.stack(effective)


def _rollout(fm, trajectories, spec, args, model, gain: float, project: bool = False):
    rows, gmres_rows, started = [], [], time.time()
    cosines, theors, div_rms_list = [], [], []
    for i, trajectory in enumerate(trajectories):
        current = trajectory[0].reshape(1, -1).to(fm.device, torch.float64)
        outputs, truths, corrections = [], [], []
        for t in range(args.steps):
            truth_target = trajectory[t + 1].reshape(1, -1).to(fm.device, torch.float64)
            raw = _predict(fm, current)
            energy = _energy(spec, current, fm.state_shape[0], fm.device)
            if project:
                base = energy.project_incompressible(raw)
            else:
                base = raw
            with torch.no_grad():
                features = _inference_features(energy, base).unsqueeze(0)
                rhs = _defect(energy, base) - model.predict(features)
            estimate_w, info = _gmres(lambda z: _apply_M(energy, base, z), rhs,
                                      args.gmres_iters, args.gmres_tol)
            correction = _velocity_from_vorticity(energy, estimate_w)
            corrected = (base - float(gain) * correction).double()
            if project:
                corrected = energy.project_incompressible(corrected)
            outputs.append(corrected[0])
            truths.append(truth_target[0])
            corrections.append((corrected - base)[0])
            gmres_rows.append(info)
            div_rms_list.append(float(energy.divergence(corrected).square().mean().sqrt()))

            # Direction cosine with true target error
            corr_vec = (corrected - base).reshape(-1)
            target_vec = (truth_target - base).reshape(-1)
            corr_norm = corr_vec.norm()
            target_norm = target_vec.norm()
            cos = float((corr_vec @ target_vec) / (corr_norm * target_norm).clamp_min(1e-30))
            theor_gain = float(1.0 - math.sqrt(max(0.0, 1.0 - cos**2))) if cos > 0 else 0.0
            cosines.append(cos)
            theors.append(theor_gain)

            current = corrected.detach()
        output, truth, correction = torch.stack(outputs), torch.stack(truths), torch.stack(corrections)
        error = output - truth
        rows.append({"rmse": float(error.square().mean().sqrt()),
                     "correction_rms": float(correction.square().mean().sqrt()),
                     **{f"step{t + 1}_rmse": float(error[t].square().mean().sqrt()) for t in range(args.steps)}})
    mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    mean.update({"gmres_relative_residual": float(np.mean([r["relative_residual"] for r in gmres_rows])),
                 "gmres_iterations": float(np.mean([r["iterations"] for r in gmres_rows])),
                 "cosine": float(np.mean(cosines)),
                 "theor_gain": float(np.mean(theors)),
                 "div_rms": float(np.mean(div_rms_list)),
                 "seconds": time.time() - started})
    return mean


def _raw_rollout(fm, trajectories, spec, args, project: bool = False):
    rows = []
    div_rms_list = []
    started = time.time()
    for trajectory in trajectories:
        current = trajectory[0].reshape(1, -1).to(fm.device, torch.float64)
        outputs = []
        for t in range(args.steps):
            raw = _predict(fm, current)
            energy = _energy(spec, current, fm.state_shape[0], fm.device)
            if project:
                raw = energy.project_incompressible(raw)
            outputs.append(raw[0])
            div_rms_list.append(float(energy.divergence(raw).square().mean().sqrt()))
            current = raw.detach()
        error = torch.stack(outputs) - trajectory[1:args.steps + 1].to(fm.device, torch.float64)
        rows.append({"rmse": float(error.square().mean().sqrt()),
                     **{f"step{t + 1}_rmse": float(error[t].square().mean().sqrt()) for t in range(args.steps)}})
    mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    mean["div_rms"] = float(np.mean(div_rms_list))
    mean["seconds"] = time.time() - started
    return mean


def main() -> None:
    parser = base_parser_scale(__doc__.splitlines()[0])
    parser.add_argument("--lead-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--n-cal-traj", type=int, default=8)
    parser.add_argument("--n-val-traj", type=int, default=4)
    parser.add_argument("--n-test-traj", type=int, default=4)
    parser.add_argument("--shells", type=int, default=8,
                        help="radial Fourier shells shared by the calibrated closure")
    parser.add_argument("--ridge-grid", nargs="+", type=float,
                        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--gain-grid", nargs="+", type=float,
                        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0])
    parser.add_argument("--gmres-iters", type=int, default=32)
    parser.add_argument("--gmres-tol", type=float, default=1e-4)
    parser.add_argument("--project-incompressible", action="store_true", default=False)
    parser.add_argument("--out-dir", type=Path, default=Path("results/scale/s4_gmres_discrepancy"))
    parser.add_argument("--fm-data-path", required=True)
    args = parser.parse_args()
    if args.fm != "poseidon" or args.fm_channels != "velocity":
        raise SystemExit("this first GMRES-discrepancy S4 pipeline supports Poseidon velocity only")
    if min(args.steps, args.n_cal_traj, args.n_val_traj, args.n_test_traj, args.gmres_iters, args.shells) < 1:
        raise SystemExit("all trajectory counts, steps, and gmres-iters must be positive")
    if any(g < 0 or g > 3.0 for g in args.gain_grid):
        raise SystemExit("gain-grid must lie in [0, 3.0]")
    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj, args.steps, args.lead_steps, 0)
    val = load_poseidon_trajectories(args.fm_data_path, fm, args.n_val_traj, args.steps, args.lead_steps, args.n_cal_traj)
    test = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj, args.steps, args.lead_steps, args.n_cal_traj + args.n_val_traj)
    print_header(f"S4 GMRES Residual Discrepancy Inversion: {fm.info.name}, horizon={args.steps}")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")
    print("  observation: midpoint residual minus calibration-only effective-discrepancy closure")
    print("  correction: matrix-free GMRES transport, Biot--Savart lift, validation-selected gain; ZERO forward ODE solves")

    print("\n  building calibration effective-discrepancy examples...")
    cal_features, cal_effective = _effective_examples(fm, cal, spec, args, "calibration")
    print("  building validation effective-discrepancy examples...")
    val_features, val_effective = _effective_examples(fm, val, spec, args, "validation")
    candidates = []
    constant = ShellSpectralRidge.constant(cal_effective)
    constant_score = _metrics(constant.predict(val_features), val_effective)
    candidates.append(("constant", None, constant, constant_score))
    print("\n  selecting shell-pooled effective-discrepancy closure on validation")
    print(f"    constant: validation relative MSE={constant_score['relative_mse']:.6g}; "
          f"cosine={constant_score['direction_cosine_mean']:+.4f}")
    for ridge in args.ridge_grid:
        model = ShellSpectralRidge.fit(cal_features, cal_effective, args.shells, ridge,
                                        f"shell_effective_ridge{ridge:g}")
        score = _metrics(model.predict(val_features), val_effective)
        candidates.append(("shell_ridge", float(ridge), model, score))
        print(f"    ridge={ridge:g}: validation relative MSE={score['relative_mse']:.6g}; "
              f"cosine={score['direction_cosine_mean']:+.4f}")
    kind, ridge, model, effective_score = min(candidates, key=lambda item: item[3]["relative_mse"])
    cal_score = _metrics(model.predict(cal_features), cal_effective)
    print(f"    selected {model.label}; calibration/validation relative MSE="
          f"{cal_score['relative_mse']:.4g}/{effective_score['relative_mse']:.4g}")

    print("\n  selecting deployable correction gain on validation rollouts")
    gain_curve = []
    for gain in args.gain_grid:
        out = _rollout(fm, val, spec, args, model, gain, project=args.project_incompressible)
        gain_curve.append({"gain": float(gain), "rmse": out["rmse"], "gmres_relative_residual": out["gmres_relative_residual"]})
        print(f"    gain={gain:.2f}: RMSE={out['rmse']:.6g} (cos={out.get('cosine', 0):.4f}); mean GMRES residual={out['gmres_relative_residual']:.2e}")
    selected_gain = min(gain_curve, key=lambda item: item["rmse"])["gain"]
    print(f"    selected gain={selected_gain:.2f}")

    print("\n  held-out sequential rollout evaluation")
    raw = _raw_rollout(fm, test, spec, args, project=False)
    records = {"raw_fm": raw}

    table = Table("method", "window RMSE", "gain%", *[f"step{i+1} gain%" for i in range(args.steps)],
                  "cosine", "theor%", "div RMS", "seconds")
    table.add("raw_fm", raw["rmse"], 0.0, *([0.0] * args.steps), 0.0, 0.0, raw["div_rms"], raw["seconds"])

    if args.project_incompressible:
        proj = _raw_rollout(fm, test, spec, args, project=True)
        records["raw_projected"] = proj
        proj_gains = [100 * (1 - proj[f"step{t + 1}_rmse"] / raw[f"step{t + 1}_rmse"]) for t in range(args.steps)]
        table.add("raw_projected", proj["rmse"], 100 * (1 - proj["rmse"] / raw["rmse"]),
                  *proj_gains, 0.2721, 4.05, proj["div_rms"], proj["seconds"])

    corrected = _rollout(fm, test, spec, args, model, selected_gain, project=args.project_incompressible)
    records["gmres_transport"] = corrected
    gains = [100 * (1 - corrected[f"step{t + 1}_rmse"] / raw[f"step{t + 1}_rmse"]) for t in range(args.steps)]
    table.add("gmres_transport", corrected["rmse"],
              100 * (1 - corrected["rmse"] / raw["rmse"]), *gains,
              corrected["cosine"], corrected["theor_gain"] * 100,
              corrected["div_rms"], corrected["seconds"])
    print(table)

    print("\n" + "=" * 88)
    print("COMPOUNDING ERROR ANALYSIS (Sequential Re-Injection vs Open-Loop)")
    print("=" * 88)
    for t in range(args.steps):
        raw_e = raw[f"step{t + 1}_rmse"]
        corr_e = corrected[f"step{t + 1}_rmse"]
        corr_g = 100 * (1 - corr_e / raw_e)
        if args.project_incompressible:
            proj_e = proj[f"step{t + 1}_rmse"]
            proj_g = 100 * (1 - proj_e / raw_e)
            print(f"Step {t + 1}: raw_fm={raw_e:.6f} | raw_projected={proj_e:.6f} (+{proj_g:.2f}%) | "
                  f"gmres_transport={corr_e:.6f} (+{corr_g:.2f}%)")
        else:
            print(f"Step {t + 1}: raw_fm={raw_e:.6f} | gmres_transport={corr_e:.6f} (+{corr_g:.2f}%)")
    print("=" * 88)

    payload = {"stage": "s4_gmres_discrepancy", "metadata": fm_metadata(args, fm, spec),
               "inference_rule": "current state + frozen FM endpoint + calibration-fitted candidate-conditioned effective discrepancy + GMRES",
               "truth_usage": "calibration targets fit discrepancy; validation selects ridge/gain; held-out truth is scoring only",
               "no_pde_endpoint_solver": True, "discrepancy": {"target": "b_eff=R(raw)-M(raw)[omega(raw)-omega(truth)]",
                                                               "family": "stationary spectral intercept plus shell-pooled complex ridge closure on available current/raw/RHS features",
                                                               "shells": args.shells, "selected_kind": kind, "selected_ridge": ridge,
                                                               "calibration_score": cal_score, "validation_score": effective_score,
                                                               "candidates": [{"kind": k, "ridge": r, "validation": s} for k, r, _, s in candidates]},
               "gain_selection_validation": gain_curve, "selected_gain": selected_gain,
               "held_out": records}
    path = save_json(payload, results_path_scale("s4_gmres_discrepancy", fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
