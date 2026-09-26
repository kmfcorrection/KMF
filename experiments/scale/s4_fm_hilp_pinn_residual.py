#!/usr/bin/env python3
"""Original sequential HILP with PINN/PiNFDiT-style physics residual energy.

Following Raissi et al. (Physics-Informed Neural Networks) and PiNFDiT (ICLR 2026):
  1. The kinematic residual uses the symplectic Gauss-Legendre midpoint rule (Raissi IRK1):
       r_omega = [omega(c) - omega(c_prev)] / dt - F_NS((omega(c) + omega(c_prev)) / 2)
     Evaluating at the algebraic midpoint preserves Hamiltonian energy and phase for
     nonlinear advection, without the aliasing error of trapezoid or the numerical
     instability of unconditioned tangent transport.
  2. The continuity equation (incompressibility) is explicitly penalized:
       r_div = div(c) = du/dx + dv/dy
     preventing unphysical divergence drift in the velocity field.
  3. The energy is purely:
       E(c) = 0.5 * ||r_omega||^2 / R_cal^2 + 0.5 * beta * ||r_div||^2 / D_cal^2
     with no forward numerical ODE solvers or oracle endpoints.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.fm_eval_common import (POSEIDON_RAW_DT, configure_native_poseidon_cadence,
                                               fm_metadata, fm_result_key, native_poseidon_spec)
from hipp.scale.calibrate_scale import fit_alpha_scale
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.curvature_scale import estimate_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.scale.lowrank import LowRankGaussian
from hipp.scale.posterior_scale import LowRankPhysicsPosterior, force_scale
from hipp.scale.rollout import fit_sigma2, propagate_step
from hipp.utils import Table, print_header, save_json, set_seed

METHODS = ("isotropic", "pushforward", "gauss_newton")


def _load(path: str, n_traj: int, steps: int, offset: int, device) -> torch.Tensor:
    need = 2 * int(steps) + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        data = np.asarray(h[key][offset:offset + n_traj, :need, :2], dtype=np.float32)
    # Retain native raw frames on input; frames 0,2,... are Poseidon's dt=0.1 states.
    return torch.as_tensor(data, device=device).reshape(n_traj, need, -1)


class PINNResidualEnergy(FMPhysicsEnergy2D):
    """Gauss-Legendre midpoint residual + continuity energy (Raissi PINN / PiNFDiT)."""
    def __init__(self, spec, previous, channels, residual_scale, divergence_weight,
                 divergence_scale=1.0, device=None):
        super().__init__(spec, previous, channels, divergence_weight=divergence_weight,
                         vorticity_scale=residual_scale, divergence_scale=divergence_scale,
                         dt=2 * POSEIDON_RAW_DT, device=device)
        self.residual_scale = float(residual_scale)
        self.divergence_scale = float(divergence_scale)

    def residual(self, candidate: torch.Tensor) -> torch.Tensor:
        omega = self.to_vorticity(candidate)
        previous = self.previous_vorticity.expand_as(omega)
        midpoint = 0.5 * (previous + omega)
        # Gauss-Legendre symplectic midpoint rule (Raissi IRK1)
        return (omega - previous) / self.dt - self.rhs_vorticity(midpoint)

    def energy(self, candidate: torch.Tensor) -> torch.Tensor:
        residual = self.residual(candidate)
        # Dimensionless normalized residual energy
        value = 0.5 * residual.flatten(1).square().mean(1) / (self.residual_scale ** 2)
        if self.channels > 1 and self.divergence_weight:
            div2 = self.divergence(candidate).flatten(1).square().mean(1)
            value = value + 0.5 * self.divergence_weight * div2 / (self.divergence_scale ** 2)
        return value

    def rms_residual(self, candidate: torch.Tensor) -> torch.Tensor:
        return self.residual(candidate).flatten(1).square().mean(1).sqrt()


def _center(prior, mean):
    return LowRankGaussian(mean.double(), prior.U, prior.d, prior.tau,
                           alpha=prior.alpha, label=prior.label)


def _prior(method, fm, conditioning, raw, sigma2, args, pf_state, generator):
    """Original HILP local covariance, constructed before entering MAP."""
    if method == "isotropic":
        return LowRankGaussian.isotropic(raw.double(), tau=1.0, label="isotropic"), None
    if method == "gauss_newton":
        estimate = estimate_scale(fm, conditioning, method="gn", k=args.k,
                                  tau_rel=args.tau_rel, oversample=args.oversample,
                                  n_iter=args.n_iter, chunk=args.chunk,
                                  n_tail=args.n_tail, generator=generator)
        return estimate.prior, None
    if pf_state is None:
        return LowRankGaussian.isotropic(raw.double(), tau=sigma2,
                                         label="pushforward-step1"), None
    propagated, diag = propagate_step(fm, conditioning, pf_state, sigma2,
                                      k=args.k, n_sketch=args.n_sketch,
                                      n_probe=args.n_probe, chunk=args.chunk,
                                      generator=generator)
    return propagated, diag


def _calibration_scales(calibration, spec, steps, divergence_weight, device) -> tuple[float, float]:
    res_values, div_values = [], []
    for trajectory in calibration:
        for step in range(steps):
            previous, truth = trajectory[2 * step], trajectory[2 * step + 2]
            energy = PINNResidualEnergy(spec, previous, 2, 1.0, divergence_weight, 1.0, device)
            with torch.no_grad():
                res_values.append(energy.residual(truth.double()))
                if divergence_weight:
                    div_values.append(energy.divergence(truth.double()))
    r_scale = float(torch.cat(res_values).square().mean().sqrt().clamp(min=1e-30))
    d_scale = float(torch.cat(div_values).square().mean().sqrt().clamp(min=1e-30)) if div_values else 1.0
    return r_scale, d_scale


def _raw_calibration_summaries(args, fm, calibration, spec, sigma2, methods):
    """HILP alpha fitting on calibration-only raw autoregressive rollouts."""
    summaries = {method: [] for method in methods}
    for method in methods:
        print(f"    building {method} calibration priors ({len(calibration)} trajectories x {args.steps} steps)")
        for i, trajectory in enumerate(calibration):
            conditioning, pf_state = trajectory[0].to(fm.device, fm.dtype), None
            for step in range(args.steps):
                raw = fm.predict(conditioning).detach().double().reshape(-1)
                generator = torch.Generator(device="cpu").manual_seed(
                    args.seed + 90_001 * i + 1_009 * step)
                prior, _ = _prior(method, fm, conditioning, raw, sigma2, args, pf_state, generator)
                truth = trajectory[2 * step + 2].double()
                summaries[method].append(prior.summarize(truth))
                conditioning = raw.to(fm.dtype)  # raw calibration rollout, per HILP protocol
                if method == "pushforward":
                    pf_state = _center(prior, raw)
    return summaries


def _map(prior, energy, lambda_rel, args):
    """Smooth float64 MAP: the FM and all Jacobian work are absent here."""
    ref = force_scale(prior, energy)
    posterior = LowRankPhysicsPosterior(prior, energy, float(lambda_rel) * ref)
    corrected, info = posterior.map_estimate(n_steps=args.map_steps, lr=args.map_lr)
    info.update({"lambda_rel": float(lambda_rel), "lambda": float(lambda_rel * ref),
                 "lambda_reference": float(ref), "prior_rank": prior.k})
    return corrected, info


def _metric(corrected, raw, truth, energy):
    correction, error = corrected - raw, truth - raw
    cosine = float((correction.reshape(-1) @ error.reshape(-1)) /
                   (correction.norm() * error.norm()).clamp_min(1e-30))
    return {"rmse": float((corrected - truth).square().mean().sqrt()),
            "raw_rmse": float((raw - truth).square().mean().sqrt()),
            "correction_rms": float(correction.square().mean().sqrt()),
            "correction_error_cosine": cosine,
            "residual_rms": float(energy.rms_residual(corrected)[0]),
            "raw_residual_rms": float(energy.rms_residual(raw)[0]),
            "function_evals": float("nan")}


def _aggregate(rows):
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _raw_rollout(fm, trajectories, steps):
    per_step = [[] for _ in range(steps)]
    for trajectory in trajectories:
        state = trajectory[0].to(fm.device, fm.dtype)
        for step in range(steps):
            raw = fm.predict(state).detach().double().reshape(-1)
            truth = trajectory[2 * step + 2].double()
            per_step[step].append(float((raw - truth).square().mean().sqrt()))
            state = raw.to(fm.dtype)
    return [float(np.mean(items)) for items in per_step]


def _rollout(args, fm, trajectories, spec, sigma2, residual_scale, divergence_scale,
             method, alpha, lambda_rel):
    per_step, diagnostics = [[] for _ in range(args.steps)], []
    start = time.time()
    for i, trajectory in enumerate(trajectories):
        conditioning, pf_state = trajectory[0].to(fm.device, fm.dtype), None
        for step in range(args.steps):
            # Original HILP sequence: predictor and geometry first, MAP second.
            raw = fm.predict(conditioning).detach().double().reshape(-1)
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + 90_001 * i + 1_009 * step)
            # Pushforward calibration scales every innovation in the Lyapunov
            # recursion.  Passing alpha*q at each propagation is equivalent to
            # scaling the complete covariance, but avoids accidentally leaving
            # the newly injected qI term unscaled at later rollout steps.
            injected = alpha * sigma2 if method == "pushforward" else sigma2
            base, propagation = _prior(method, fm, conditioning, raw, injected,
                                       args, pf_state, generator)
            prior = base if method == "pushforward" else base.rescaled(alpha)
            energy = PINNResidualEnergy(spec, conditioning, 2, residual_scale,
                                        args.divergence_weight, divergence_scale, fm.device)
            corrected, info = _map(prior, energy, lambda_rel, args)
            if args.project_incompressible:
                corrected = energy.project_incompressible(corrected)
            truth = trajectory[2 * step + 2].double()
            metrics = _metric(corrected, raw, truth, energy)
            metrics["function_evals"] = float(info["function_evals"])
            per_step[step].append(metrics)
            diagnostics.append({"trajectory": i, "step": step + 1, **info,
                                "raw_residual_rms": metrics["raw_residual_rms"],
                                "map_residual_rms": metrics["residual_rms"]})
            # Freeze MAP endpoint.  The next FM/Jacobian evaluation is conditional on it,
            # but it never becomes a differentiable parent of the MAP problem above.
            conditioning = corrected.to(fm.dtype)
            if method == "pushforward":
                pf_state = _center(prior, corrected)
    return [_aggregate(rows) for rows in per_step], diagnostics, time.time() - start


def _score(rows):
    return float(np.mean([row["rmse"] for row in rows]))


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=8)
    ap.add_argument("--n-val-traj", type=int, default=4)
    ap.add_argument("--n-test-traj", type=int, default=8)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0, .01, .03, .1, .3, 1, 3, 10, 30, 100])
    ap.add_argument("--map-steps", type=int, default=30)
    ap.add_argument("--map-lr", type=float, default=.5)
    ap.add_argument("--divergence-weight", type=float, default=1.0,
                    help="weight for incompressibility / continuity residual (default: 1.0)")
    ap.add_argument("--project-incompressible", action="store_true", default=False,
                    help="Helmholtz-project MAP endpoint onto div(u)=0")
    ap.add_argument("--n-sketch", type=int, default=None)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--n-tail", type=int, default=16)
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.lead_steps != 1:
        raise SystemExit("this native Poseidon experiment requires --lead-steps=1")
    set_seed(args.seed)
    args.fm, args.fm_channels = "poseidon", "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = _load(args.fm_data_path, args.n_cal_traj, args.steps, 0, fm.device)
    val = _load(args.fm_data_path, args.n_val_traj, args.steps, args.n_cal_traj, fm.device)
    test = _load(args.fm_data_path, args.n_test_traj, args.steps,
                 args.n_cal_traj + args.n_val_traj, fm.device)
    xs = [cal[i, 2 * step] for i in range(len(cal)) for step in range(args.steps)]
    ys = [cal[i, 2 * step + 2] for i in range(len(cal)) for step in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]
    residual_scale, divergence_scale = _calibration_scales(
        cal, spec, args.steps, args.divergence_weight, fm.device)

    print_header(f"S4 original sequential HILP + PINN residual: {fm.info.name}, k={args.k}")
    print("  HILP update: sequential local Gaussian MAP; FM and Jacobian probes are outside L-BFGS")
    print("  likelihood: Gauss-Legendre symplectic midpoint residual (Raissi IRK1) + continuity energy")
    print("  autoregression: freeze MAP endpoint, then recompute FM mean/local geometry at that endpoint")
    print(f"  q={sigma2:.5g}; R_cal^1/2={residual_scale:.5g}; D_cal^1/2={divergence_scale:.5g}; div_weight={args.divergence_weight:g}")
    print(f"  split: calibration={len(cal)}, validation={len(val)}, held-out={len(test)}")

    print("\n  calibrating original HILP prior scales on raw calibration rollouts")
    summaries = _raw_calibration_summaries(args, fm, cal, spec, sigma2, args.methods)
    alphas, alpha_diag = {}, {}
    for method in args.methods:
        alphas[method], alpha_diag[method] = fit_alpha_scale(
            summaries[method], objective=args.alpha_objective, return_diagnostics=True, seed=args.seed)
        geo_share = alpha_diag[method]['geometry_share']
        geo_str = "n/a (d<0)" if math.isnan(geo_share) else f"{geo_share:.4g}"
        print(f"    {method:14s} alpha={alphas[method]:.6g}; "
              f"geometry-share={geo_str}")

    curves, selected = {}, {}
    for method in args.methods:
        print(f"\n  selecting lambda for {method} on trajectory-disjoint validation rollouts")
        curve = []
        for lam in args.lambda_grid:
            rows, _, seconds = _rollout(args, fm, val, spec, sigma2, residual_scale,
                                        divergence_scale, method, alphas[method], lam)
            item = {"lambda_rel": lam, "rmse": _score(rows), "steps": rows, "seconds": seconds}
            curve.append(item)
            print(f"    lambda/ref={lam:g}: RMSE={item['rmse']:.6g}", flush=True)
        selected[method] = min(curve, key=lambda item: item["rmse"])["lambda_rel"]
        curves[method] = curve
        print(f"    selected {selected[method]:g} x reference")

    print("\n  held-out sequential HILP evaluation")
    raw_steps = _raw_rollout(fm, test, args.steps)
    raw_rmse = float(np.mean(raw_steps))
    table = Table("method", "RMSE", "gain%", *[f"step{i + 1} gain%" for i in range(args.steps)],
                  "corr RMS", "corr/error cosine", "resid RMS", "LBFGS evals", "seconds")
    table.add("raw", raw_rmse, 0., *([0.] * args.steps), 0., 0., float("nan"), 0., 0.)
    held, diagnostics = {"raw": {"mean_rollout_rmse": raw_rmse, "steps": raw_steps}}, {}
    for method in args.methods:
        rows, diag, seconds = _rollout(args, fm, test, spec, sigma2, residual_scale,
                                       divergence_scale, method, alphas[method], selected[method])
        score = _score(rows)
        gains = [100 * (1 - row["rmse"] / raw) for row, raw in zip(rows, raw_steps)]
        table.add(method, score, 100 * (1 - score / raw_rmse), *gains,
                  float(np.mean([row["correction_rms"] for row in rows])),
                  float(np.mean([row["correction_error_cosine"] for row in rows])),
                  float(np.mean([row["residual_rms"] for row in rows])),
                  float(np.mean([row["function_evals"] for row in rows])), seconds)
        held[method] = {"mean_rollout_rmse": score, "steps": rows, "seconds": seconds}
        diagnostics[method] = diag
    print(table)
    output = {"stage": "s4_original_hilp_pinn_residual", "metadata": fm_metadata(args, fm, spec),
              "objective": {"prior": "sequential local Gaussian HILP",
                            "physics": "symplectic_midpoint_irk1_plus_continuity",
                            "residual_scale": residual_scale,
                            "divergence_scale": divergence_scale,
                            "divergence_weight": args.divergence_weight,
                            "project_incompressible": args.project_incompressible,
                            "y0": "fixed", "fm_inside_map": False,
                            "no_flow_solver_or_oracle": True},
              "sigma2": sigma2, "alpha": alphas, "alpha_diagnostics": alpha_diag,
              "validation": {"curves": curves, "selected_lambda_rel": selected},
              "held_out": held, "diagnostics": diagnostics}
    path = save_json(output, results_path_scale("s4_original_hilp_pinn_residual",
                                                 fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
