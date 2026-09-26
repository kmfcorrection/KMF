#!/usr/bin/env python3
"""Whole-timescale joint trajectory HILP for frozen Poseidon rollouts.

Inspired by PiNFDiT (Cao et al., ICLR 2026), this program optimizes the
entire future forecast window z = (z_1, ..., z_T) simultaneously under:

    E(z) = 1/2 ||r_omega(z)||^2 / R_cal^2 + beta/2 ||r_div(z)||^2 / D_cal^2
    prior(z) ~ N(z_raw, Sigma_joint)

Supported residual modes:
  - centered (default, PiNFDiT style):
        r_n = (omega_{n+1} - omega_{n-1}) / (2 dt) - F(omega_n),  n=1,...,T-1
  - collocation (Gauss-Legendre symplectic midpoint on all intervals):
        r_n = (omega_n - omega_{n-1}) / dt - F((omega_{n-1} + omega_n)/2), n=1,...,T

The initial condition x_0 is fixed. Ground truth future frames are NEVER
accessed during prior construction, residual evaluation, lambda selection,
or MAP optimization.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (configure_native_poseidon_cadence,
    fm_metadata, fm_result_key, load_poseidon_trajectories, native_poseidon_spec)
from hipp.scale.calibrate_scale import fit_alpha_scale
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.scale.jacobian import jacobian_ops
from hipp.scale.lowrank import LowRankGaussian, merge_lowrank, randn
from hipp.scale.posterior_scale import LowRankPhysicsPosterior, force_scale
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("isotropic", "joint_pushforward")
RESIDUAL_MODES = ("centered", "collocation")
RESIDUAL_SPACES = ("vorticity", "velocity")


class WholeWindowResidualEnergy:
    """Whole-timescale differentiable NS residual over a future state window.

    Given fixed initial state x_0 and candidate z = (z_1, ..., z_T), computes:
      1. Vorticity or velocity residual under either centered differences (PiNFDiT)
         or Gauss-Legendre midpoint collocation.
      2. Velocity divergence r_div = grad . u across all frames.
    """
    def __init__(self, spec, x0, *, steps, dt, residual_scale, divergence_scale,
                 divergence_weight, mode="centered", residual_space="velocity", device=None):
        if steps < 2:
            raise ValueError("whole window residual requires --steps >= 2")
        self.spec, self.steps, self.dt = spec, int(steps), float(dt)
        self.N, self.device = 2 * spec.N, device
        self.residual_scale = float(residual_scale)
        self.divergence_scale = float(divergence_scale)
        self.divergence_weight = float(divergence_weight)
        self.mode = mode
        self.residual_space = residual_space
        if self.residual_scale <= 0 or self.divergence_scale <= 0:
            raise ValueError("residual scales must be positive")
        self.operator = FMPhysicsEnergy2D(spec, x0, 2, divergence_weight=0.0,
                                          dt=dt, device=device)
        self.x0 = x0.detach().to(device, torch.float64).reshape(1, self.N)

    def _window(self, z):
        z = z.to(self.device, torch.float64).reshape(-1, self.steps, self.N)
        if z.shape[0] != 1:
            raise ValueError("joint-window posterior solves one rollout at a time")
        return z

    def residual_fields(self, z):
        future = self._window(z)[0]
        states = torch.cat((self.x0, future), dim=0)  # shape (steps + 1, N)
        vort = self.operator.to_vorticity(states)      # shape (steps + 1, H, W)

        if self.mode == "centered":
            # PiNFDiT 2nd-order centered difference on interior states n = 1, ..., T-1
            r_w = ((vort[2:] - vort[:-2]) / (2.0 * self.dt) -
                   self.operator._native_rhs(vort[1:-1]))
        elif self.mode == "collocation":
            # Gauss-Legendre midpoint rule on all intervals [n-1, n] for n = 1, ..., T
            midpoints = 0.5 * (vort[:-1] + vort[1:])
            r_w = ((vort[1:] - vort[:-1]) / self.dt -
                   self.operator._native_rhs(midpoints))
        else:
            raise ValueError(f"unknown residual mode: {self.mode}")

        if self.residual_space == "velocity":
            rh = torch.fft.rfft2(r_w)
            ru, rv = self.operator.grid.velocity(rh)
            return torch.stack((ru, rv), dim=1)
        return r_w

    def energy(self, z):
        future = self._window(z)[0]
        r = self.residual_fields(future)
        residual_sq = r.square().mean()
        div_sq = self.operator.divergence(future).square().mean()
        e = 0.5 * residual_sq / (self.residual_scale ** 2)
        if self.divergence_weight:
            e = e + 0.5 * self.divergence_weight * div_sq / (self.divergence_scale ** 2)
        return e.reshape(1)

    def grad(self, z):
        x = z.detach().to(self.device, torch.float64).reshape(1, -1).requires_grad_(True)
        (g,) = torch.autograd.grad(self.energy(x).sum(), x)
        return g

    def rms_residual(self, z):
        return self.residual_fields(z).square().mean().sqrt().reshape(1)

    def rms_divergence(self, z):
        future = self._window(z)[0]
        return self.operator.divergence(future).square().mean().sqrt().reshape(1)

    def project_incompressible(self, z):
        """Helmholtz-project every forecast frame onto div(u)=0 in Fourier space."""
        future = self._window(z)[0]
        projected = []
        for t in range(self.steps):
            frame = self.operator.project_incompressible(future[t:t+1])
            projected.append(frame.reshape(self.N))
        return torch.cat(projected, dim=0).reshape(1, -1)


def _raw_window(fm, x0, steps):
    """Frozen autoregressive mean trajectory across all steps."""
    states, c = [], x0.to(fm.device, fm.dtype)
    for _ in range(steps):
        c = fm.predict(c).detach()
        states.append(c.double().reshape(-1))
    return torch.stack(states)


def _joint_pushforward_prior(fm, x0, raw, sigma2, args, generator):
    """Tangent-ensemble covariance of the complete future trajectory.

    Propagates perturbations autoregressively:
        delta z_1 = e_1
        delta z_{t+1} = J_t delta z_t + e_{t+1},  e_t ~ N(0, sigma^2 I)
    Stacking across time yields the low-rank factor of the joint covariance.
    """
    m = int(args.tangent_samples or (args.k + args.oversample))
    if m < 2:
        raise ValueError("--tangent-samples must be at least 2")
    n = raw.shape[1]
    delta = math.sqrt(max(float(sigma2), 1e-30)) * randn(
        (n, m), fm.dtype, fm.device, generator)
    factors = [delta.double()]
    calls = 0
    for t in range(1, args.steps):
        ops = jacobian_ops(fm.flat_fn(), raw[t - 1].to(fm.device, fm.dtype))
        innovation = math.sqrt(max(float(sigma2), 1e-30)) * randn(
            (n, m), fm.dtype, fm.device, generator)
        delta = ops.mm(delta, chunk=args.chunk) + innovation
        calls += ops.n_calls[0]
        factors.append(delta.double())
    A = torch.cat(factors, dim=0) / math.sqrt(m)
    trace_sample = float(A.square().sum())
    floor = max(float(args.joint_floor_rel) * trace_sample / A.shape[0], 1e-30)
    joint_rank = min(int(args.joint_rank or args.k), A.shape[1])
    U, d, tau = merge_lowrank([A], k=joint_rank, tau=floor)
    prior = LowRankGaussian(raw.reshape(-1).double(), U, d, tau,
                            label="joint_pushforward")
    return prior, {"tangent_samples": m, "joint_rank": prior.k,
                   "joint_floor": float(tau), "joint_sample_trace": trace_sample,
                   "joint_jvp_calls": calls}


def _prior(method, fm, x0, raw, sigma2, args, generator):
    mean = raw.reshape(-1).double()
    if method == "isotropic":
        pf, info = _joint_pushforward_prior(fm, x0, raw, sigma2, args, generator)
        return LowRankGaussian.isotropic(mean, tau=pf.trace() / pf.N,
                                         label="joint_isotropic"), info
    if method == "joint_pushforward":
        return _joint_pushforward_prior(fm, x0, raw, sigma2, args, generator)
    raise ValueError(method)


def _calibration_scales(fm, cal, spec, sigma2, args):
    residuals, divergences = [], []
    for i in range(cal.shape[0]):
        raw = _raw_window(fm, cal[i, 0], args.steps)
        e = WholeWindowResidualEnergy(spec, cal[i, 0], steps=args.steps,
                                      dt=args.lead_steps * spec.dt_out,
                                      residual_scale=1.0, divergence_scale=1.0,
                                      divergence_weight=args.divergence_weight,
                                      mode=args.residual_mode,
                                      residual_space=args.residual_space,
                                      device=fm.device)
        residuals.append(float(e.rms_residual(raw)))
        divergences.append(float(e.rms_divergence(raw)))
    return {"residual_scale": max(float(np.median(residuals)), 1e-30),
            "divergence_scale": max(float(np.median(divergences)), 1e-30),
            "calibration_residual_median": float(np.median(residuals)),
            "calibration_divergence_median": float(np.median(divergences))}


def _temporal_residual_floor_audit(fm, cal, spec, scales, args):
    """Audit likelihood discriminability: truth residual vs raw FM residual."""
    truth_rms, raw_rms = [], []
    for i in range(cal.shape[0]):
        energy = WholeWindowResidualEnergy(
            spec, cal[i, 0], steps=args.steps,
            dt=args.lead_steps * spec.dt_out,
            residual_scale=scales["residual_scale"],
            divergence_scale=scales["divergence_scale"],
            divergence_weight=args.divergence_weight,
            mode=args.residual_mode,
            residual_space=args.residual_space,
            device=fm.device)
        truth = cal[i, 1:args.steps + 1].to(fm.device, torch.float64)
        raw = _raw_window(fm, cal[i, 0], args.steps)
        truth_rms.append(float(energy.rms_residual(truth)))
        raw_rms.append(float(energy.rms_residual(raw)))
    truth_med, raw_med = float(np.median(truth_rms)), float(np.median(raw_rms))
    return {"truth_residual_median": truth_med,
            "raw_residual_median": raw_med,
            "truth_over_raw": truth_med / max(raw_med, 1e-30),
            "truth_wins_fraction": float(np.mean(np.asarray(truth_rms) < np.asarray(raw_rms)))}


def _prepare(method, fm, trajectories, sigma2, scales, spec, args, reference=None):
    packages = []
    for i in range(trajectories.shape[0]):
        print(f"    {method}: trajectory {i + 1}/{trajectories.shape[0]}", flush=True)
        if method == "isotropic" and reference is not None:
            raw, energy = reference[i]["raw"], reference[i]["energy"]
            pf = reference[i]["prior"]
            prior = LowRankGaussian.isotropic(raw.reshape(-1).double(),
                                            tau=pf.trace() / pf.N,
                                            label="joint_isotropic")
            info = {"matched_pushforward_trace": pf.trace(),
                    "tangent_samples": reference[i]["prior_info"]["tangent_samples"]}
        else:
            raw = _raw_window(fm, trajectories[i, 0], args.steps)
            gen = torch.Generator(device="cpu").manual_seed(args.seed + 10_009 * i)
            prior, info = _prior(method, fm, trajectories[i, 0], raw, sigma2, args, gen)
            energy = WholeWindowResidualEnergy(
                spec, trajectories[i, 0], steps=args.steps,
                dt=args.lead_steps * spec.dt_out,
                residual_scale=scales["residual_scale"],
                divergence_scale=scales["divergence_scale"],
                divergence_weight=args.divergence_weight,
                mode=args.residual_mode,
                residual_space=args.residual_space,
                device=fm.device)
        packages.append({"raw": raw, "prior": prior, "energy": energy,
                         "truth": trajectories[i, 1:args.steps + 1].to(fm.device, torch.float64),
                         "prior_info": info})
    return packages


def _summaries(packages):
    return [p["prior"].summarize(p["truth"].reshape(-1)) for p in packages]


def _evaluate(method, alpha, lam_rel, packages, args, collect_trace=False, project=None):
    if project is None:
        project = bool(args.project_incompressible)
    rmses, per_step, diagnostics = [], [[] for _ in range(args.steps)], []
    cosines, resid_rms, div_rms = [], [], []
    started = time.time()
    for i, package in enumerate(packages):
        raw, pinfo = package["raw"], package["prior_info"]
        prior, energy, truth = package["prior"].rescaled(alpha), package["energy"], package["truth"]
        ref = force_scale(prior, energy)
        post = LowRankPhysicsPosterior(prior, energy, lam_rel * ref)
        corrected, info = post.map_estimate(n_steps=args.map_steps, lr=args.map_lr)
        if project:
            corrected = energy.project_incompressible(corrected)
        corrected_window = corrected.reshape(args.steps, -1)
        raw_err = (raw - truth).square().mean().sqrt()
        corr_err = (corrected_window - truth).square().mean().sqrt()
        rmses.append(float(corr_err))
        for t in range(args.steps):
            per_step[t].append(float((corrected_window[t] - truth[t]).square().mean().sqrt()))

        # Diagnostics: alignment, residual, divergence
        corr_vec = (corrected_window - raw).reshape(-1)
        err_vec = (truth - raw).reshape(-1)
        cos = float((corr_vec @ err_vec) / (corr_vec.norm() * err_vec.norm()).clamp_min(1e-30))
        cosines.append(cos)
        resid_rms.append(float(energy.rms_residual(corrected)[0]))
        div_rms.append(float(energy.rms_divergence(corrected)[0]))

        if collect_trace:
            g = energy.grad(prior.mean).reshape(-1)
            frac = ((prior.U.T @ g).square().sum() / g.square().sum().clamp(min=1e-300)
                    if prior.k else g.new_zeros(()))
            diagnostics.append({"trajectory": i, "raw_rmse": float(raw_err),
                                "corrected_rmse": float(corr_err),
                                "cosine": cos,
                                "grad_in_joint_rank": float(frac),
                                "correction_rms": float((corrected - prior.mean).square().mean().sqrt()),
                                "residual_rms": float(energy.rms_residual(corrected)[0]),
                                "divergence_rms": float(energy.rms_divergence(corrected)[0]),
                                "lambda": lam_rel * ref, "lambda_ref": ref,
                                **pinfo, **info})
    return {"rmse": float(np.mean(rmses)),
            "step_rmse": [float(np.mean(x)) for x in per_step],
            "cosine": float(np.mean(cosines)),
            "residual_rms": float(np.mean(resid_rms)),
            "divergence_rms": float(np.mean(div_rms)),
            "seconds": time.time() - started, "diagnostics": diagnostics}


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=8)
    ap.add_argument("--n-val-traj", type=int, default=4)
    ap.add_argument("--n-test-traj", type=int, default=8)
    ap.add_argument("--residual-mode", choices=RESIDUAL_MODES, default="centered",
                    help="residual formulation: 'centered' (PiNFDiT) or 'collocation' (midpoint)")
    ap.add_argument("--residual-space", choices=RESIDUAL_SPACES, default="velocity",
                    help="residual metric space: 'velocity' (Biot-Savart velocity defect, matching evaluation RMSE) or 'vorticity'")
    ap.add_argument("--joint-rank", type=int, default=None,
                    help="rank of complete stacked trajectory covariance; default --k")
    ap.add_argument("--tangent-samples", type=int, default=None,
                    help="independent tangent trajectories; default k+oversample")
    ap.add_argument("--joint-floor-rel", type=float, default=1e-3,
                    help="explicit isotropic ridge as fraction of sampled mean variance")
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--map-steps", type=int, default=30)
    ap.add_argument("--map-lr", type=float, default=0.5)
    ap.add_argument("--divergence-weight", type=float, default=1.0)
    ap.add_argument("--project-incompressible", action="store_true",
                    help="apply exact Fourier Helmholtz projection to each frame after MAP")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()

    if args.steps < 2 or args.n_val_traj < 1:
        raise SystemExit("whole-timescale joint HILP needs --steps >= 2 and --n-val-traj >= 1")
    set_seed(args.seed)
    args.fm, args.fm_size, args.fm_channels = "poseidon", args.fm_size, "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj,
                                     args.steps, args.lead_steps, 0)
    val = load_poseidon_trajectories(args.fm_data_path, fm, args.n_val_traj,
                                     args.steps, args.lead_steps, args.n_cal_traj)
    test = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj,
                                      args.steps, args.lead_steps,
                                      args.n_cal_traj + args.n_val_traj)

    print_header(f"S4 whole-timescale joint HILP ({args.residual_mode}/{args.residual_space}): {fm.info.name}, horizon={args.steps}, k={args.k}")
    print(f"  likelihood: {args.residual_mode} temporal NS residual in {args.residual_space} space + continuity energy")
    print("  prior: trace-matched isotropic control vs. tangent-ensemble joint pushforward covariance")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")

    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]
    scales = _calibration_scales(fm, cal, spec, sigma2, args)
    print(f"  fitted one-step sigma²={sigma2:.5e}; residual scale={scales['residual_scale']:.4g}; "
          f"divergence scale={scales['divergence_scale']:.4g}; div_weight={args.divergence_weight:g}")

    floor = _temporal_residual_floor_audit(fm, cal, spec, scales, args)
    print("  temporal-residual floor audit (calibration only): "
          f"truth={floor['truth_residual_median']:.4g}, raw={floor['raw_residual_median']:.4g}, "
          f"truth/raw={floor['truth_over_raw']:.3f}, truth wins={100 * floor['truth_wins_fraction']:.1f}%")

    alphas, summaries_meta, prepared = {}, {}, {}
    print("\n  calibrating joint prior scales")
    method_order = (["joint_pushforward"] if "joint_pushforward" in args.methods else []) + \
                   [m for m in args.methods if m != "joint_pushforward"]
    for method in method_order:
        packages = _prepare(method, fm, cal, sigma2, scales, spec, args,
                            reference=prepared.get(("calibration", "joint_pushforward")))
        prepared[("calibration", method)] = packages
        summaries = _summaries(packages)
        alpha, diag = fit_alpha_scale(summaries, return_diagnostics=True, seed=args.seed)
        alphas[method], summaries_meta[method] = alpha, {
            "fit": diag, "prior_meta": [p["prior_info"] for p in packages]}
        print(f"    {method:18s} alpha={alpha:.4g}; error-in-joint-rank="
              f"{diag.get('error_energy_in_subspace', float('nan')):.4g}")

    selected, selection = {}, {}
    for method in method_order:
        print(f"\n  selecting lambda for {method} on validation windows")
        packages = _prepare(method, fm, val, sigma2, scales, spec, args,
                            reference=prepared.get(("validation", "joint_pushforward")))
        prepared[("validation", method)] = packages
        curve = []
        for lam in args.lambda_grid:
            out = _evaluate(method, alphas[method], lam, packages, args)
            curve.append({"lambda_rel": lam, "rmse": out["rmse"]})
            print(f"    lambda/ref={lam:g}: mean window RMSE={out['rmse']:.6g}", flush=True)
        best = min(curve, key=lambda x: x["rmse"])
        selected[method], selection[method] = best["lambda_rel"], curve
        print(f"    selected {best['lambda_rel']:g} x reference (RMSE={best['rmse']:.6g})")

    print("\n  held-out joint-window evaluation")
    table = Table("method", "window RMSE", "gain%", *[f"step{i+1} gain%" for i in range(args.steps)],
                  "cosine", "resid RMS", "div RMS", "seconds")
    records = {}
    test_packages = {}
    for method in method_order:
        test_packages[method] = _prepare(
            method, fm, test, sigma2, scales, spec, args,
            reference=test_packages.get("joint_pushforward"))
        prepared[("held_out", method)] = test_packages[method]

    ref_method = "joint_pushforward" if "joint_pushforward" in test_packages else method_order[0]
    raw_unprojected = _evaluate(ref_method, 1.0, 0.0, test_packages[ref_method], args, project=False)
    records["raw_unprojected"] = raw_unprojected
    table.add("raw_fm", raw_unprojected["rmse"], 0.0, *([0.0] * args.steps), 0.0,
              raw_unprojected["residual_rms"], raw_unprojected["divergence_rms"], raw_unprojected["seconds"])

    if args.project_incompressible:
        raw_projected = _evaluate(ref_method, 1.0, 0.0, test_packages[ref_method], args, project=True)
        records["raw_projected"] = raw_projected
        proj_gains = [100 * (1 - raw_projected["step_rmse"][t] / raw_unprojected["step_rmse"][t]) for t in range(args.steps)]
        table.add("raw_projected", raw_projected["rmse"], 100 * (1 - raw_projected["rmse"] / raw_unprojected["rmse"]),
                  *proj_gains, raw_projected["cosine"], raw_projected["residual_rms"],
                  raw_projected["divergence_rms"], raw_projected["seconds"])

    base_raw = raw_unprojected

    for method in args.methods:
        out = _evaluate(method, alphas[method], selected[method], test_packages[method],
                        args, collect_trace=True)
        records[method] = out
        gains = [100 * (1 - out["step_rmse"][t] / base_raw["step_rmse"][t]) for t in range(args.steps)]
        table.add(method, out["rmse"], 100 * (1 - out["rmse"] / base_raw["rmse"]), *gains,
                  out["cosine"], out["residual_rms"], out["divergence_rms"], out["seconds"])
        if out["diagnostics"]:
            print(f"  {method}: mean physics-gradient energy in joint rank="
                  f"{np.mean([d['grad_in_joint_rank'] for d in out['diagnostics']]):.3e}")
    print(table)

    out = {"stage": "s4_fm_hilp_joint_window",
           "metadata": fm_metadata(args, fm, spec),
           "residual_mode": args.residual_mode,
           "sigma2": sigma2, "physics_scales": scales,
           "temporal_residual_floor_calibration": floor, "alphas": alphas,
           "calibration": summaries_meta, "lambda_selection": selection,
           "selected_lambda": selected, "held_out": records,
           "project_incompressible": args.project_incompressible,
           "validity": {"future_truth_used_by_method": False,
                        "residual_mode": args.residual_mode,
                        "temporal_residual": "centered interior finite difference" if args.residual_mode == "centered" else "midpoint collocation",
                        "spatial_operator": "published AZEBAN pseudospectral RHS",
                        "prior": "Monte-Carlo tangent joint rollout covariance"}}
    path = save_json(out, results_path_scale("s4_joint_window", fm_result_key(args),
                                             "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
