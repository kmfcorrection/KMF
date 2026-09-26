#!/usr/bin/env python3
"""Whole-timescale joint HILP with 4th-order symplectic Hermite-Simpson collocation.

Addresses the finite-cadence discretization floor by upgrading temporal collocation
from 2nd-order (O(dt^2) ~ 12% error) to 4th-order (O(dt^4) ~ 0.01% error):

For each interval [t_n, t_{n+1}], the cubic Hermite state at midpoint is:
    omega_mid = (omega_n + omega_{n+1})/2 + dt/8 * (F(omega_n) - F(omega_{n+1}))

and the 4th-order Simpson collocation defect is:
    r_HS = (omega_{n+1} - omega_n)/dt - 1/6 * (F(omega_n) + 4 F(omega_mid) + F(omega_{n+1}))

Zero ODE integration, zero forward marching. Pure algebraic defect ||r_HS||^2 = E.
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
from hipp.scale.posterior_scale import LowRankPhysicsPosterior, force_scale, covariance_matvec
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("isotropic", "joint_pushforward")


class IRK4WindowResidualEnergy:
    """Joint trajectory residual energy with 4th-order Hermite-Simpson collocation."""
    def __init__(self, spec, x0, *, steps, dt, order=4, bias=None, residual_space="vorticity",
                 residual_scale=1.0, divergence_scale=1.0, divergence_weight=10.0, device=None):
        if steps < 2:
            raise ValueError("joint residual requires --steps >= 2")
        self.spec, self.steps, self.dt = spec, int(steps), float(dt)
        self.order = int(order)
        self.residual_space = str(residual_space)
        self.N, self.device = 2 * spec.N, device
        self.residual_scale = float(residual_scale)
        self.divergence_scale = float(divergence_scale)
        self.divergence_weight = float(divergence_weight)
        if bias is not None:
            self.bias = bias.detach().to(self.device, torch.float64)
        else:
            self.bias = None
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

    def raw_residual_fields(self, z):
        future = self._window(z)[0]
        states = torch.cat((self.x0, future), dim=0)  # (steps + 1, N)
        vort = self.operator.to_vorticity(states)      # (steps + 1, H, W)

        w_left, w_right = vort[:-1], vort[1:]          # each (steps, H, W)

        if self.order == 2:
            # 2nd-order symplectic Gauss-Legendre midpoint rule
            midpoints = 0.5 * (w_left + w_right)
            r_w = (w_right - w_left) / self.dt - self.operator._native_rhs(midpoints)

        elif self.order == 4:
            # 4th-order symplectic Hermite-Simpson collocation
            f_left = self.operator._native_rhs(w_left)
            f_right = self.operator._native_rhs(w_right)
            # Cubic Hermite bridge at midpoint
            w_mid = 0.5 * (w_left + w_right) + 0.125 * self.dt * (f_left - f_right)
            f_mid = self.operator._native_rhs(w_mid)
            # Simpson quadrature defect
            r_w = ((w_right - w_left) / self.dt -
                   (1.0 / 6.0) * (f_left + 4.0 * f_mid + f_right))
        else:
            raise ValueError(f"unsupported collocation order: {self.order}")

        if self.residual_space == "velocity":
            rh = torch.fft.rfft2(r_w)
            ru, rv = self.operator.grid.velocity(rh)
            return torch.stack((ru, rv), dim=1)  # (steps, 2, H, W)
        return r_w

    def residual_fields(self, z):
        r = self.raw_residual_fields(z)
        if self.bias is not None:
            r = r - self.bias
        return r

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

    def rms_raw_residual(self, z):
        return self.raw_residual_fields(z).square().mean().sqrt().reshape(1)

    def rms_divergence(self, z):
        future = self._window(z)[0]
        return self.operator.divergence(future).square().mean().sqrt().reshape(1)

    def project_incompressible(self, z):
        future = self._window(z)[0]
        projected = []
        for t in range(self.steps):
            frame = self.operator.project_incompressible(future[t:t+1])
            projected.append(frame.reshape(self.N))
        return torch.cat(projected, dim=0).reshape(1, -1)


def _raw_window(fm, x0, steps):
    states, c = [], x0.to(fm.device, fm.dtype)
    for _ in range(steps):
        c = fm.predict(c).detach()
        states.append(c.double().reshape(-1))
    return torch.stack(states)


def _joint_pushforward_prior(fm, x0, raw, sigma2, args, generator):
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


def _calibrate_irk_scales(fm, cal, spec, sigma2, args):
    truth_defects, raw_defects, divergences = [], [], []
    for i in range(cal.shape[0]):
        e_uncentered = IRK4WindowResidualEnergy(
            spec, cal[i, 0], steps=args.steps,
            dt=args.lead_steps * spec.dt_out,
            order=args.order, bias=None,
            residual_space=args.residual_space,
            residual_scale=1.0,
            divergence_scale=1.0, divergence_weight=args.divergence_weight,
            device=fm.device)
        truth = cal[i, 1:args.steps + 1].to(fm.device, torch.float64)
        raw = _raw_window(fm, cal[i, 0], args.steps)
        r_truth = e_uncentered.raw_residual_fields(truth)
        r_raw = e_uncentered.raw_residual_fields(raw)
        truth_defects.append(r_truth)
        raw_defects.append(r_raw)
        divergences.append(float(e_uncentered.rms_divergence(raw)))

    truth_all = torch.stack(truth_defects)  # (N_cal, steps, ...)
    raw_all = torch.stack(raw_defects)      # (N_cal, steps, ...)

    if args.residual_bias == "calibration_mean":
        bias = truth_all.mean(dim=0)
    else:
        bias = torch.zeros_like(truth_all[0])

    centered_truth = truth_all - bias[None, ...]
    centered_raw = raw_all - bias[None, ...]

    residual_scale = float(centered_truth.square().mean().sqrt().clamp_min(1e-30))
    divergence_scale = max(float(np.median(divergences)), 1e-30)

    truth_rms = float(centered_truth.square().mean().sqrt())
    raw_rms = float(centered_raw.square().mean().sqrt())
    spatial_dims = tuple(range(1, truth_all.ndim))
    per_traj_truth = centered_truth.square().mean(dim=spatial_dims).sqrt()
    per_traj_raw = centered_raw.square().mean(dim=spatial_dims).sqrt()
    truth_wins = float((per_traj_truth < per_traj_raw).float().mean())

    audit = {"truth_residual_rms": truth_rms,
             "raw_residual_rms": raw_rms,
             "raw_truth_defect_rms": float(truth_all.square().mean().sqrt()),
             "bias_rms": float(bias.square().mean().sqrt()),
             "truth_over_raw": truth_rms / max(raw_rms, 1e-30),
             "truth_wins_fraction": truth_wins}

    return bias, residual_scale, divergence_scale, audit


def _prepare(method, fm, trajectories, sigma2, bias, residual_scale, divergence_scale,
             spec, args, reference=None):
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
            energy = IRK4WindowResidualEnergy(
                spec, trajectories[i, 0], steps=args.steps,
                dt=args.lead_steps * spec.dt_out,
                order=args.order,
                bias=bias,
                residual_space=args.residual_space,
                residual_scale=residual_scale,
                divergence_scale=divergence_scale,
                divergence_weight=args.divergence_weight,
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

        corr_vec = (corrected_window - raw).reshape(-1)
        err_vec = (truth - raw).reshape(-1)
        cos = float((corr_vec @ err_vec) / (corr_vec.norm() * err_vec.norm()).clamp_min(1e-30))
        cosines.append(cos)
        resid_rms.append(float(energy.rms_residual(corrected)[0]))
        div_rms.append(float(energy.rms_divergence(corrected)[0]))

        if collect_trace:
            g = energy.grad(prior.mean).reshape(-1)
            e_vec = (truth.reshape(-1) - raw.reshape(-1))
            e_norm = float(e_vec.norm().clamp_min(1e-30))
            g_norm = float(g.norm().clamp_min(1e-30))

            if prior.k:
                proj_e = prior.U.T @ e_vec
                e_in_U = float(proj_e.square().sum() / (e_norm ** 2))
                max_gain_U = 100.0 * (1.0 - math.sqrt(max(0.0, 1.0 - e_in_U)))

                proj_g = prior.U.T @ g
                g_in_U = float(proj_g.square().sum() / (g_norm ** 2))

                delta_z = (corrected - prior.mean).reshape(-1)
                delta_norm = float(delta_z.norm())
                proj_delta = prior.U.T @ delta_z
                delta_in_U = float(proj_delta.square().sum() / max(delta_norm ** 2, 1e-30))

                w = prior.d / (prior.d + prior.tau)
                maha_subspace = float((((1.0 - w) * (proj_delta ** 2)).sum() / prior.tau) / prior.alpha)
                delta_perp_norm2 = max(float(delta_z.square().sum() - proj_delta.square().sum()), 0.0)
                maha_perp = float((delta_perp_norm2 / prior.tau) / prior.alpha)
            else:
                e_in_U = 1.0
                max_gain_U = 100.0
                g_in_U = 1.0
                delta_in_U = 1.0
                delta_z = (corrected - prior.mean).reshape(-1)
                delta_norm = float(delta_z.norm())
                maha_subspace = 0.0
                maha_perp = float(0.5 * prior.mahalanobis_sq(corrected))

            cos_g_e = float((-g @ e_vec) / (g_norm * e_norm))
            sg = covariance_matvec(prior, g).reshape(-1)
            cos_sg_e = float((-sg @ e_vec) / (sg.norm().clamp_min(1e-30) * e_norm))

            diag_item = {
                "trajectory": i,
                "raw_rmse": float(raw_err),
                "corrected_rmse": float(corr_err),
                "cosine": cos,
                "error_energy_in_rank": e_in_U,
                "max_theoretical_gain_in_rank_pct": max_gain_U,
                "grad_in_joint_rank": g_in_U,
                "cos_phys_grad_error": cos_g_e,
                "cos_precond_grad_error": cos_sg_e,
                "step_norm": delta_norm,
                "step_over_error": delta_norm / e_norm,
                "step_energy_in_rank": delta_in_U,
                "maha_subspace": maha_subspace,
                "maha_perp": maha_perp,
                "correction_rms": float((corrected - prior.mean).square().mean().sqrt()),
                "residual_rms": float(energy.rms_residual(corrected)[0]),
                "divergence_rms": float(energy.rms_divergence(corrected)[0]),
                "lambda": lam_rel * ref,
                "lambda_ref": ref,
                **pinfo, **info,
            }
            diagnostics.append(diag_item)
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
    ap.add_argument("--n-test-traj", type=int, default=4)
    ap.add_argument("--order", type=int, choices=(2, 4), default=4,
                    help="collocation order: 4 (Hermite-Simpson O(dt^4)) or 2 (midpoint O(dt^2))")
    ap.add_argument("--residual-space", choices=("vorticity", "velocity"), default="vorticity",
                    help="residual representation: 'vorticity' (H1 velocity norm) or 'velocity' (H-1 streamfunction norm)")
    ap.add_argument("--residual-bias", choices=("zero", "calibration_mean"), default="calibration_mean",
                    help="discretization bias target: 'calibration_mean' or 'zero'")
    ap.add_argument("--joint-rank", type=int, default=None,
                    help="rank of complete stacked trajectory covariance; default --k")
    ap.add_argument("--tangent-samples", type=int, default=None,
                    help="independent tangent trajectories; default k+oversample")
    ap.add_argument("--joint-floor-rel", type=float, default=1e-3,
                    help="explicit isotropic ridge as fraction of sampled mean variance")
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=["joint_pushforward"])
    ap.add_argument("--map-steps", type=int, default=40)
    ap.add_argument("--map-lr", type=float, default=0.5)
    ap.add_argument("--divergence-weight", type=float, default=10.0)
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

    print_header(f"S4 whole-timescale joint HILP (IRK order {args.order}, {args.residual_space}): {fm.info.name}, horizon={args.steps}, k={args.k}")
    print(f"  likelihood: order-{args.order} symplectic collocation in {args.residual_space} space (bias={args.residual_bias}) + continuity energy")
    print("  prior: trace-matched isotropic control vs. tangent-ensemble joint pushforward covariance")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")

    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]

    bias, residual_scale, divergence_scale, audit = _calibrate_irk_scales(
        fm, cal, spec, sigma2, args)

    print(f"  fitted one-step sigma²={sigma2:.5e}; raw truth defect RMS={audit['raw_truth_defect_rms']:.4g}; "
          f"bias RMS={audit['bias_rms']:.4g}; residual scale={residual_scale:.4g}; div scale={divergence_scale:.4g}")
    print("  temporal-residual floor audit (calibration only): "
          f"truth={audit['truth_residual_rms']:.4g}, raw={audit['raw_residual_rms']:.4g}, "
          f"truth/raw={audit['truth_over_raw']:.3f}, truth wins={100 * audit['truth_wins_fraction']:.1f}%")

    alphas, summaries_meta, prepared = {}, {}, {}
    print("\n  calibrating joint prior scales")
    method_order = (["joint_pushforward"] if "joint_pushforward" in args.methods else []) + \
                   [m for m in args.methods if m != "joint_pushforward"]
    for method in method_order:
        packages = _prepare(method, fm, cal, sigma2, bias, residual_scale, divergence_scale,
                            spec, args, reference=prepared.get(("calibration", "joint_pushforward")))
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
        packages = _prepare(method, fm, val, sigma2, bias, residual_scale, divergence_scale,
                            spec, args, reference=prepared.get(("validation", "joint_pushforward")))
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
            method, fm, test, sigma2, bias, residual_scale, divergence_scale,
            spec, args, reference=test_packages.get("joint_pushforward"))
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
    print(table)

    for method in args.methods:
        d_list = records[method].get("diagnostics", [])
        if d_list:
            mean_e_in_U = float(np.mean([d["error_energy_in_rank"] for d in d_list]))
            mean_max_gain = float(np.mean([d["max_theoretical_gain_in_rank_pct"] for d in d_list]))
            mean_g_in_U = float(np.mean([d["grad_in_joint_rank"] for d in d_list]))
            mean_cos_g = float(np.mean([d["cos_phys_grad_error"] for d in d_list]))
            mean_cos_sg = float(np.mean([d["cos_precond_grad_error"] for d in d_list]))
            mean_step_ratio = float(np.mean([d["step_over_error"] for d in d_list]))
            mean_delta_in_U = float(np.mean([d["step_energy_in_rank"] for d in d_list]))
            mean_maha_sub = float(np.mean([d["maha_subspace"] for d in d_list]))
            mean_maha_perp = float(np.mean([d["maha_perp"] for d in d_list]))

            print("\n" + "=" * 78)
            print(f"DIAGNOSTIC AUDIT: Bottleneck Analysis ({method})")
            print("=" * 78)
            print(f"1. SUBSPACE CAPABILITY:")
            print(f"   True error energy inside rank-{args.k} tangent subspace U: {100*mean_e_in_U:.2f}%")
            print(f"   Theoretical MAXIMUM gain possible inside U:          {mean_max_gain:.2f}%")
            print(f"   Actual observed gain over raw_fm:                    {100*(1 - records[method]['rmse']/base_raw['rmse']):.2f}%")
            if args.project_incompressible:
                print(f"   Gain from Helmholtz divergence projection alone:     {100*(1 - records['raw_projected']['rmse']/base_raw['rmse']):.2f}%")
                print(f"   Net MAP contribution beyond Helmholtz projection:    {100*(records['raw_projected']['rmse'] - records[method]['rmse'])/base_raw['rmse']:.2f}%")
            print(f"\n2. GRADIENT & STEP ALIGNMENT:")
            print(f"   Physics gradient energy in U:                        {100*mean_g_in_U:.4f}%")
            print(f"   Raw physics gradient alignment with true error:      cos = {mean_cos_g:+.4f}")
            print(f"   Preconditioned step alignment with true error:        cos = {mean_cos_sg:+.4f}")
            print(f"   Actual MAP correction step alignment:                cos = {records[method]['cosine']:+.4f}")
            print(f"   Relative step size ||delta z|| / ||true error||:     {100*mean_step_ratio:.2f}%")
            print(f"   Step energy confined inside U:                       {100*mean_delta_in_U:.2f}%")
            print(f"\n3. MAHALANOBIS PENALTY BREAKDOWN:")
            print(f"   Penalty from moving inside U (subspace):             {mean_maha_sub:.2f}")
            print(f"   Penalty from moving outside U (orthogonal):          {mean_maha_perp:.2f}")
            print("=" * 78 + "\n")

    out = {"stage": "s4_fm_hilp_irk4_residual",
           "metadata": fm_metadata(args, fm, spec),
           "order": args.order,
           "residual_space": args.residual_space,
           "residual_bias": args.residual_bias,
           "sigma2": sigma2,
           "raw_truth_defect_rms": audit["raw_truth_defect_rms"],
           "bias_rms": audit["bias_rms"],
           "residual_scale": residual_scale,
           "divergence_scale": divergence_scale,
           "temporal_residual_floor_calibration": audit,
           "alphas": alphas,
           "calibration": summaries_meta,
           "lambda_selection": selection,
           "selected_lambda": selected,
           "held_out": records,
           "project_incompressible": args.project_incompressible}
    path = save_json(out, results_path_scale("s4_joint_window", fm_result_key(args),
                                             "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
