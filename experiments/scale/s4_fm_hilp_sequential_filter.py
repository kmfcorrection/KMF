#!/usr/bin/env python3
"""Sequential Autoregressive Filtering HILP with 4th-order Hermite-Simpson collocation.

Breaks the static whole-window error-compounding ceiling by applying closed-loop
physics-informed MAP correction and Helmholtz incompressibility projection at
EACH autoregressive step, re-injecting the cleaned state back into the foundation
model for subsequent predictions:

    c_0 = x_0
    for t = 0 ... T-1:
        u_{t+1}^{raw} = FoundationModel(c_t)
        u_{t+1}^{MAP} = argmin_u [ 0.5 (u - u^{raw})^T Sigma^{-1} (u - u^{raw}) + lambda E_{IRK4}(u; c_t) ]
        c_{t+1} = P_{div-free}(u_{t+1}^{MAP})    <-- Re-injected conditioning state

Strictly zero forward ODE marching, zero numerical solver calls.
Pure algebraic Hermite-Simpson defect on each transition.
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
from hipp.scale.lowrank import LowRankGaussian
from hipp.scale.posterior_scale import (LowRankPhysicsPosterior,
                                         covariance_matvec, force_scale)
from hipp.scale.rollout import fit_sigma2, propagate_step
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("isotropic", "pushforward")


def _center(prior, mean):
    return LowRankGaussian(mean.reshape(-1).double(), prior.U, prior.d, prior.tau,
                           alpha=prior.alpha, label=prior.label)


class SingleStepIRK4Energy:
    """Single-step 4th-order symplectic Hermite-Simpson collocation energy."""
    def __init__(self, spec, x_prev, *, dt, order=4, bias=None,
                 residual_scale=1.0, divergence_scale=1.0, divergence_weight=10.0,
                 device=None):
        self.spec = spec
        self.dt = float(dt)
        self.order = int(order)
        self.N = 2 * spec.N
        self.device = device
        self.residual_scale = float(residual_scale)
        self.divergence_scale = float(divergence_scale)
        self.divergence_weight = float(divergence_weight)
        if bias is not None:
            self.bias = bias.detach().to(self.device, torch.float64)
        else:
            self.bias = None
        if self.residual_scale <= 0 or self.divergence_scale <= 0:
            raise ValueError("residual scales must be positive")
        self.operator = FMPhysicsEnergy2D(spec, x_prev, 2, divergence_weight=0.0,
                                          dt=dt, device=device)
        self.x_prev = x_prev.detach().to(device, torch.float64).reshape(1, 2, spec.n, spec.n)
        self.w_prev = self.operator.to_vorticity(self.x_prev)  # (1, H, W)
        self.f_prev = self.operator._native_rhs(self.w_prev)   # (1, H, W)

    def raw_residual_fields(self, x_cand):
        cand = x_cand.to(self.device, torch.float64).reshape(-1, 2, self.spec.n, self.spec.n)
        w_cand = self.operator.to_vorticity(cand)
        f_cand = self.operator._native_rhs(w_cand)

        if self.order == 2:
            w_mid = 0.5 * (self.w_prev + w_cand)
            f_mid = self.operator._native_rhs(w_mid)
            return (w_cand - self.w_prev) / self.dt - f_mid

        elif self.order == 4:
            # Hermite cubic midpoint
            w_mid = 0.5 * (self.w_prev + w_cand) + 0.125 * self.dt * (self.f_prev - f_cand)
            f_mid = self.operator._native_rhs(w_mid)
            return ((w_cand - self.w_prev) / self.dt -
                    (1.0 / 6.0) * (self.f_prev + 4.0 * f_mid + f_cand))
        else:
            raise ValueError(f"unsupported collocation order: {self.order}")

    def residual_fields(self, x_cand):
        r = self.raw_residual_fields(x_cand)
        if self.bias is not None:
            r = r - self.bias
        return r

    def energy(self, x_cand):
        cand = x_cand.to(self.device, torch.float64).reshape(-1, 2, self.spec.n, self.spec.n)
        r = self.residual_fields(cand)
        residual_sq = r.square().mean()
        div_sq = self.operator.divergence(cand).square().mean()
        e = 0.5 * residual_sq / (self.residual_scale ** 2)
        if self.divergence_weight:
            e = e + 0.5 * self.divergence_weight * div_sq / (self.divergence_scale ** 2)
        return e.reshape(1)

    def grad(self, x_cand):
        x = x_cand.detach().to(self.device, torch.float64).reshape(1, -1).requires_grad_(True)
        (g,) = torch.autograd.grad(self.energy(x).sum(), x)
        return g

    def rms_residual(self, x_cand):
        return self.residual_fields(x_cand).square().mean().sqrt().reshape(1)

    def rms_raw_residual(self, x_cand):
        return self.raw_residual_fields(x_cand).square().mean().sqrt().reshape(1)

    def rms_divergence(self, x_cand):
        cand = x_cand.to(self.device, torch.float64).reshape(-1, 2, self.spec.n, self.spec.n)
        return self.operator.divergence(cand).square().mean().sqrt().reshape(1)

    def project_incompressible(self, x_cand):
        cand = x_cand.to(self.device, torch.float64).reshape(-1, 2, self.spec.n, self.spec.n)
        return self.operator.project_incompressible(cand).reshape(1, -1)


def _calibrate_single_step_scales(fm, cal, spec, sigma2, args):
    """Calibrate single-step spatial bias and residual scales across all transitions."""
    dt = args.lead_steps * spec.dt_out
    truth_defects, raw_defects, divergences = [], [], []

    for i in range(cal.shape[0]):
        for t in range(args.steps):
            x_prev = cal[i, t].to(fm.device, torch.float64)
            x_truth = cal[i, t + 1].to(fm.device, torch.float64)
            e_uncentered = SingleStepIRK4Energy(
                spec, x_prev, dt=dt, order=args.order, bias=None,
                residual_scale=1.0, divergence_scale=1.0,
                divergence_weight=args.divergence_weight, device=fm.device)
            r_truth = e_uncentered.raw_residual_fields(x_truth)
            with torch.no_grad():
                u_raw = fm.predict(x_prev.to(fm.dtype)).double()
            r_raw = e_uncentered.raw_residual_fields(u_raw)

            truth_defects.append(r_truth)
            raw_defects.append(r_raw)
            divergences.append(float(e_uncentered.rms_divergence(u_raw)))

    truth_all = torch.stack(truth_defects)  # (N_cal * steps, 1, H, W)
    raw_all = torch.stack(raw_defects)

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
    per_trans_truth = centered_truth.square().mean(dim=(1, 2, 3)).sqrt()
    per_trans_raw = centered_raw.square().mean(dim=(1, 2, 3)).sqrt()
    truth_wins = float((per_trans_truth < per_trans_raw).float().mean())

    audit = {
        "truth_residual_rms": truth_rms,
        "raw_residual_rms": raw_rms,
        "raw_truth_defect_rms": float(truth_all.square().mean().sqrt()),
        "bias_rms": float(bias.square().mean().sqrt()),
        "truth_over_raw": truth_rms / max(raw_rms, 1e-30),
        "truth_wins_fraction": truth_wins,
        "n_transitions": int(truth_all.shape[0]),
    }
    return bias, residual_scale, divergence_scale, audit


def _calibrate_method_alpha(method, fm, cal, spec, sigma2, bias, residual_scale,
                           divergence_scale, args):
    """Fit alpha scale parameter on single-step transitions."""
    dt = args.lead_steps * spec.dt_out
    summaries = []
    for i in range(cal.shape[0]):
        c = cal[i, 0].to(fm.device, fm.dtype)
        state_prior = None
        for t in range(args.steps):
            raw_mean = fm.predict(c).double()
            gen = torch.Generator(device="cpu").manual_seed(args.seed + 100_003 * i + 1_009 * t)
            if method == "isotropic" or state_prior is None:
                prior = LowRankGaussian.isotropic(raw_mean, tau=sigma2, label="isotropic")
            else:
                prior, _ = propagate_step(
                    fm, c, state_prior, sigma2, k=args.k, chunk=args.chunk,
                    complement="nystrom", generator=gen)

            truth_next = cal[i, t + 1].to(fm.device, torch.float64).reshape(-1)
            summaries.append(prior.summarize(truth_next))

            c = cal[i, t + 1].to(fm.device, fm.dtype)
            state_prior = _center(prior, c.double())

    alpha, diag = fit_alpha_scale(summaries, return_diagnostics=True, seed=args.seed)
    return alpha, diag


def _run_sequential_rollout(fm, trajectories, spec, sigma2, bias, residual_scale,
                            divergence_scale, method, alpha, lam_rel, args,
                            collect_trace=False):
    """Execute closed-loop autoregressive rollouts for a given method."""
    dt = args.lead_steps * spec.dt_out
    n_traj = trajectories.shape[0]
    all_traj_preds = []   # will hold (n_traj, steps, N)
    step_rmses = [[] for _ in range(args.steps)]
    cosines, resid_rms_list, div_rms_list = [], [], []
    diagnostics = []
    started = time.time()

    for i in range(n_traj):
        c = trajectories[i, 0].to(fm.device, fm.dtype)
        state_prior = None
        traj_states = []

        for t in range(args.steps):
            truth_target = trajectories[i, t + 1].to(fm.device, torch.float64).reshape(1, -1)
            raw_mean = fm.predict(c).double().reshape(1, -1)
            energy = SingleStepIRK4Energy(
                spec, c, dt=dt, order=args.order, bias=bias,
                residual_scale=residual_scale, divergence_scale=divergence_scale,
                divergence_weight=args.divergence_weight, device=fm.device)

            if method == "raw_fm":
                corrected = raw_mean
                info = {}
            elif method == "raw_projected":
                corrected = energy.project_incompressible(raw_mean)
                info = {}
            elif method in ("isotropic", "pushforward"):
                gen = torch.Generator(device="cpu").manual_seed(args.seed + 100_003 * i + 1_009 * t)
                if method == "isotropic" or state_prior is None:
                    prior = LowRankGaussian.isotropic(raw_mean, tau=sigma2, label="isotropic")
                    pinfo = {"prior_type": "isotropic", "k": 0}
                else:
                    prior, pinfo = propagate_step(
                        fm, c, state_prior, sigma2, k=args.k, chunk=args.chunk,
                        complement="nystrom", generator=gen)

                prior = prior.rescaled(alpha)
                ref = force_scale(prior, energy)
                post = LowRankPhysicsPosterior(prior, energy, lam_rel * ref)
                corrected, opt_info = post.map_estimate(n_steps=args.map_steps, lr=args.map_lr)
                if args.project_incompressible:
                    corrected = energy.project_incompressible(corrected)
                info = {**pinfo, **opt_info, "lambda_rel": lam_rel, "lambda_ref": ref}
            else:
                raise ValueError(f"unknown method: {method}")

            corr_flat = corrected.reshape(1, -1)
            err_flat = (corr_flat - truth_target).reshape(-1)
            step_rmse = float(err_flat.square().mean().sqrt())
            step_rmses[t].append(step_rmse)

            corr_vec = (corr_flat - raw_mean).reshape(-1)
            target_vec = (truth_target - raw_mean).reshape(-1)
            cos = float((corr_vec @ target_vec) / (corr_vec.norm() * target_vec.norm()).clamp_min(1e-30))
            cosines.append(cos)
            resid_rms_list.append(float(energy.rms_residual(corr_flat)[0]))
            div_rms_list.append(float(energy.rms_divergence(corr_flat)[0]))

            traj_states.append(corr_flat)

            # Sequential closed-loop re-injection: the corrected state is the input for next step!
            c = corr_flat.to(fm.dtype)
            if method in ("isotropic", "pushforward") and 'prior' in locals():
                state_prior = _center(prior, corr_flat)

            if collect_trace and method not in ("raw_fm", "raw_projected"):
                diagnostics.append({
                    "trajectory": i, "step": t + 1,
                    "raw_step_rmse": float((raw_mean - truth_target).square().mean().sqrt()),
                    "corrected_step_rmse": step_rmse,
                    "cosine": cos,
                    "resid_rms": float(energy.rms_residual(corr_flat)[0]),
                    "div_rms": float(energy.rms_divergence(corr_flat)[0]),
                    **info
                })

        all_traj_preds.append(torch.cat(traj_states, dim=0))  # (steps, N)

    elapsed = time.time() - started
    all_preds = torch.stack(all_traj_preds)  # (n_traj, steps, N)
    truth_all = trajectories[:, 1:args.steps + 1].to(fm.device, torch.float64)
    window_rmse = float((all_preds - truth_all).square().mean().sqrt())

    return {
        "rmse": window_rmse,
        "step_rmse": [float(np.mean(x)) for x in step_rmses],
        "cosine": float(np.mean(cosines)),
        "residual_rms": float(np.mean(resid_rms_list)),
        "divergence_rms": float(np.mean(div_rms_list)),
        "seconds": elapsed,
        "diagnostics": diagnostics,
    }


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=8)
    ap.add_argument("--n-val-traj", type=int, default=4)
    ap.add_argument("--n-test-traj", type=int, default=4)
    ap.add_argument("--order", type=int, choices=(2, 4), default=4,
                    help="collocation order: 4 (Hermite-Simpson O(dt^4)) or 2 (midpoint O(dt^2))")
    ap.add_argument("--residual-bias", choices=("zero", "calibration_mean"), default="calibration_mean",
                    help="discretization bias target: 'calibration_mean' or 'zero'")
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=["isotropic"])
    ap.add_argument("--map-steps", type=int, default=40)
    ap.add_argument("--map-lr", type=float, default=0.5)
    ap.add_argument("--divergence-weight", type=float, default=10.0)
    ap.add_argument("--project-incompressible", action="store_true", default=True,
                    help="apply exact Fourier Helmholtz projection to each frame after MAP before re-injection")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()

    if args.steps < 2 or args.n_val_traj < 1:
        raise SystemExit("sequential filter needs --steps >= 2 and --n-val-traj >= 1")
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

    print_header(f"S4 Sequential Autoregressive Filter HILP (IRK order {args.order}): {fm.info.name}, horizon={args.steps}")
    print(f"  likelihood: step-by-step order-{args.order} symplectic collocation (bias={args.residual_bias}) + continuity")
    print(f"  re-injection: corrected divergence-free state fed back into {fm.info.name} at each step")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")

    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]

    bias, residual_scale, divergence_scale, audit = _calibrate_single_step_scales(
        fm, cal, spec, sigma2, args)

    print(f"  fitted one-step sigma²={sigma2:.5e}; raw truth defect RMS={audit['raw_truth_defect_rms']:.4g}; "
          f"bias RMS={audit['bias_rms']:.4g}; residual scale={residual_scale:.4g}; div scale={divergence_scale:.4g}")
    print(f"  temporal-residual floor audit ({audit['n_transitions']} transitions): "
          f"truth={audit['truth_residual_rms']:.4g}, raw={audit['raw_residual_rms']:.4g}, "
          f"truth/raw={audit['truth_over_raw']:.3f}, truth wins={100 * audit['truth_wins_fraction']:.1f}%")

    alphas = {}
    print("\n  calibrating single-step prior scales")
    for method in args.methods:
        alpha, diag = _calibrate_method_alpha(method, fm, cal, spec, sigma2, bias,
                                              residual_scale, divergence_scale, args)
        alphas[method] = alpha
        print(f"    {method:15s} alpha={alpha:.4g}")

    selected, selection = {}, {}
    for method in args.methods:
        print(f"\n  selecting lambda for {method} on validation rollouts")
        curve = []
        for lam in args.lambda_grid:
            out = _run_sequential_rollout(fm, val, spec, sigma2, bias, residual_scale,
                                          divergence_scale, method, alphas[method], lam, args)
            curve.append({"lambda_rel": lam, "rmse": out["rmse"]})
            print(f"    lambda/ref={lam:g}: window RMSE={out['rmse']:.6g}", flush=True)
        best = min(curve, key=lambda x: x["rmse"])
        selected[method], selection[method] = best["lambda_rel"], curve
        print(f"    selected {best['lambda_rel']:g} x reference (RMSE={best['rmse']:.6g})")

    print("\n  held-out sequential rollout evaluation")
    table = Table("method", "window RMSE", "gain%", *[f"step{i+1} gain%" for i in range(args.steps)],
                  "cosine", "resid RMS", "div RMS", "seconds")
    records = {}

    # 1. Baseline Open-Loop Poseidon Rollout
    raw_res = _run_sequential_rollout(fm, test, spec, sigma2, bias, residual_scale,
                                      divergence_scale, "raw_fm", 1.0, 0.0, args)
    records["raw_fm"] = raw_res
    table.add("raw_fm", raw_res["rmse"], 0.0, *([0.0] * args.steps), 0.0,
              raw_res["residual_rms"], raw_res["divergence_rms"], raw_res["seconds"])

    # 2. Baseline Closed-Loop Re-injected Helmholtz Projection
    proj_res = _run_sequential_rollout(fm, test, spec, sigma2, bias, residual_scale,
                                       divergence_scale, "raw_projected", 1.0, 0.0, args)
    records["raw_projected"] = proj_res
    proj_gains = [100 * (1 - proj_res["step_rmse"][t] / raw_res["step_rmse"][t]) for t in range(args.steps)]
    table.add("raw_projected", proj_res["rmse"], 100 * (1 - proj_res["rmse"] / raw_res["rmse"]),
              *proj_gains, proj_res["cosine"], proj_res["residual_rms"],
              proj_res["divergence_rms"], proj_res["seconds"])

    # 3. Closed-Loop Sequential Filtering
    for method in args.methods:
        out = _run_sequential_rollout(fm, test, spec, sigma2, bias, residual_scale,
                                      divergence_scale, method, alphas[method],
                                      selected[method], args, collect_trace=True)
        records[method] = out
        gains = [100 * (1 - out["step_rmse"][t] / raw_res["step_rmse"][t]) for t in range(args.steps)]
        table.add(method, out["rmse"], 100 * (1 - out["rmse"] / raw_res["rmse"]), *gains,
                  out["cosine"], out["residual_rms"], out["divergence_rms"], out["seconds"])

    print(table)

    # Detailed Compounding Error Growth Analysis
    print("\n" + "=" * 78)
    print("COMPOUNDING ERROR ANALYSIS (Sequential Re-Injection vs Open-Loop)")
    print("=" * 78)
    for t in range(args.steps):
        raw_e = raw_res["step_rmse"][t]
        proj_e = proj_res["step_rmse"][t]
        line = f"Step {t+1}: raw_fm={raw_e:.6f} | raw_projected={proj_e:.6f} ({100*(1-proj_e/raw_e):+.2f}%)"
        for m in args.methods:
            m_e = records[m]["step_rmse"][t]
            line += f" | {m}={m_e:.6f} ({100*(1-m_e/raw_e):+.2f}%)"
        print(line)
    print("=" * 78 + "\n")

    out_data = {
        "stage": "s4_fm_hilp_sequential_filter",
        "metadata": fm_metadata(args, fm, spec),
        "order": args.order,
        "residual_bias": args.residual_bias,
        "sigma2": sigma2,
        "raw_truth_defect_rms": audit["raw_truth_defect_rms"],
        "bias_rms": audit["bias_rms"],
        "residual_scale": residual_scale,
        "divergence_scale": divergence_scale,
        "temporal_residual_floor_calibration": audit,
        "alphas": alphas,
        "lambda_selection": selection,
        "selected_lambda": selected,
        "held_out": records,
        "project_incompressible": args.project_incompressible,
    }
    path = save_json(out_data, results_path_scale("s4_sequential_filter", fm_result_key(args),
                                                  "results.json", args.tag))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
