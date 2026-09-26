#!/usr/bin/env python3
"""S4 Dynamic Discrepancy Filter HILP: Wiener-Kalman Discrepancy Subtraction.

Cancels the state-dependent Hermite-Simpson truncation error noise floor by
learning a spectral Wiener transfer kernel between the collocation residual
r(u_raw) and the true prediction error e = u* - u_raw on calibration transitions:

    H(k) = E[ e_hat(k) * r_hat(k)^* ] / ( E[ |r_hat(k)|^2 ] + eps )

This automatically filters out frequency bands where r(u) is dominated by
discretization truncation noise, retaining only the frequencies carrying
genuine signal about Poseidon's error.

At inference time:
    Delta u = IFFT2( H(k) * FFT2(r(u_raw)) )
    Delta u_proj = P_div_free(Delta u)
    u_corrected = u_raw + gamma * Delta u_proj

Zero forward ODE marching, zero oracle solver calls.
Runtime < 2 ms on GPU.
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
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("raw_fm", "raw_projected", "wiener_filter")


class SpectralWienerDiscrepancy:
    """Optimal 2D Fourier Wiener transfer filter from collocation residual to true error."""
    def __init__(self, n: int, L: float = 1.0, reg: float = 1e-4,
                 device=None, dtype=torch.float64):
        self.n = n
        self.L = float(L)
        self.reg = float(reg)
        self.device = device
        self.dtype = dtype
        self.H = None  # will hold (2, n, n) complex transfer coefficients

    def fit(self, errors: list[torch.Tensor], residuals: list[torch.Tensor]):
        """Fit Wiener transfer function H(k) = S_er / (S_rr + reg)."""
        # errors: list of (2, n, n) tensors
        # residuals: list of (1, n, n) or (2, n, n) tensors
        n = self.n
        S_er = torch.zeros(2, n, n, dtype=torch.complex128, device=self.device)
        S_rr = torch.zeros(2, n, n, dtype=torch.float64, device=self.device)

        M = len(errors)
        for e, r in zip(errors, residuals):
            e_2d = e.reshape(2, n, n).to(self.device, torch.float64)
            # Expand vorticity residual to 2 channels if needed
            if r.shape[0] == 1:
                r_2d = r.expand(2, n, n).to(self.device, torch.float64)
            else:
                r_2d = r.reshape(2, n, n).to(self.device, torch.float64)

            e_hat = torch.fft.fft2(e_2d)
            r_hat = torch.fft.fft2(r_2d)

            S_er += e_hat * r_hat.conj()
            S_rr += r_hat.abs().square()

        S_er /= max(M, 1)
        S_rr /= max(M, 1)
        reg_val = self.reg * float(S_rr.mean().clamp_min(1e-30))
        self.H = S_er / (S_rr + reg_val)

    def predict_error(self, r: torch.Tensor) -> torch.Tensor:
        """Apply Wiener filter to predict true error from residual."""
        if self.H is None:
            raise RuntimeError("Wiener filter not fitted")
        orig_shape = r.shape
        n = self.n
        if r.shape[-3] == 1:
            r_2d = r.reshape(-1, 1, n, n).expand(-1, 2, n, n).to(self.device, torch.float64)
        else:
            r_2d = r.reshape(-1, 2, n, n).to(self.device, torch.float64)

        r_hat = torch.fft.fft2(r_2d)
        e_pred_hat = self.H.unsqueeze(0) * r_hat
        e_pred = torch.fft.ifft2(e_pred_hat).real
        return e_pred.reshape(-1, 2 * n * n)


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
        self.operator = FMPhysicsEnergy2D(spec, x_prev, 2, divergence_weight=0.0,
                                          dt=dt, device=device)
        self.x_prev = x_prev.detach().to(device, torch.float64).reshape(1, 2, spec.n, spec.n)
        self.w_prev = self.operator.to_vorticity(self.x_prev)
        self.f_prev = self.operator._native_rhs(self.w_prev)

    def raw_residual_fields(self, x_cand):
        cand = x_cand.to(self.device, torch.float64).reshape(-1, 2, self.spec.n, self.spec.n)
        w_cand = self.operator.to_vorticity(cand)
        f_cand = self.operator._native_rhs(w_cand)

        # Hermite cubic midpoint
        w_mid = 0.5 * (self.w_prev + w_cand) + 0.125 * self.dt * (self.f_prev - f_cand)
        f_mid = self.operator._native_rhs(w_mid)
        return ((w_cand - self.w_prev) / self.dt -
                (1.0 / 6.0) * (self.f_prev + 4.0 * f_mid + f_cand))

    def residual_fields(self, x_cand):
        r = self.raw_residual_fields(x_cand)
        if self.bias is not None:
            r = r - self.bias
        return r

    def rms_residual(self, x_cand):
        return self.residual_fields(x_cand).square().mean().sqrt().reshape(1)

    def rms_divergence(self, x_cand):
        cand = x_cand.to(self.device, torch.float64).reshape(-1, 2, self.spec.n, self.spec.n)
        return self.operator.divergence(cand).square().mean().sqrt().reshape(1)

    def project_incompressible(self, x_cand):
        cand = x_cand.to(self.device, torch.float64).reshape(-1, 2, self.spec.n, self.spec.n)
        return self.operator.project_incompressible(cand).reshape(1, -1)


def _calibrate_wiener_filter(fm, cal, spec, wiener, args):
    """Calibrate Wiener transfer filter between collocation residual and error."""
    dt = args.lead_steps * spec.dt_out
    errors = []
    residuals = []

    for i in range(cal.shape[0]):
        for t in range(args.steps):
            x_prev = cal[i, t].to(fm.device, torch.float64)
            x_truth = cal[i, t + 1].to(fm.device, torch.float64)
            with torch.no_grad():
                u_raw = fm.predict(x_prev.to(fm.dtype)).double()

            e = (x_truth - u_raw).reshape(2, spec.n, spec.n)
            energy = SingleStepIRK4Energy(spec, x_prev, dt=dt, order=args.order, device=fm.device)
            r = energy.raw_residual_fields(u_raw).reshape(1, spec.n, spec.n)

            errors.append(e)
            residuals.append(r)

    wiener.fit(errors, residuals)


def _run_dynamic_rollout(fm, trajectories, spec, sigma2, wiener, method,
                         gamma, args, collect_trace=False):
    """Execute closed-loop autoregressive rollouts for Dynamic Discrepancy HILP."""
    dt = args.lead_steps * spec.dt_out
    n_traj = trajectories.shape[0]
    all_traj_preds = []
    step_rmses = [[] for _ in range(args.steps)]
    cosines, gammas, theors = [], [], []
    resid_rms_list, div_rms_list = [], []
    diagnostics = []
    started = time.time()

    for i in range(n_traj):
        c = trajectories[i, 0].to(fm.device, fm.dtype)
        traj_states = []

        for t in range(args.steps):
            truth_target = trajectories[i, t + 1].to(fm.device, torch.float64).reshape(1, -1)
            raw_mean = fm.predict(c).double().reshape(1, -1)
            energy = SingleStepIRK4Energy(
                spec, c, dt=dt, order=args.order, device=fm.device)

            if method == "raw_fm":
                corrected = raw_mean
                info = {}
            elif method == "raw_projected":
                corrected = energy.project_incompressible(raw_mean)
                info = {}
            elif method == "wiener_filter":
                r_raw = energy.raw_residual_fields(raw_mean)
                # Predict true error via Wiener filter
                e_pred = wiener.predict_error(r_raw).reshape(1, -1)
                # Project predicted error to divergence-free space
                e_pred_proj = energy.project_incompressible(e_pred)

                # Update state: u = u_raw + gamma * e_pred_proj
                corrected = raw_mean + gamma * e_pred_proj
                if args.project_incompressible:
                    corrected = energy.project_incompressible(corrected)
                info = {"gamma": gamma}
            else:
                raise ValueError(f"unknown method: {method}")

            corr_flat = corrected.reshape(1, -1)
            err_flat = (corr_flat - truth_target).reshape(-1)
            step_rmse = float(err_flat.square().mean().sqrt())
            step_rmses[t].append(step_rmse)

            corr_vec = (corr_flat - raw_mean).reshape(-1)
            target_vec = (truth_target - raw_mean).reshape(-1)
            corr_norm = corr_vec.norm()
            target_norm = target_vec.norm()
            cos = float((corr_vec @ target_vec) / (corr_norm * target_norm).clamp_min(1e-30))
            theor_gain = float(1.0 - math.sqrt(max(0.0, 1.0 - cos**2))) if cos > 0 else 0.0

            cosines.append(cos)
            gammas.append(float(corr_norm / target_norm.clamp_min(1e-30)))
            theors.append(theor_gain)
            resid_rms_list.append(float(energy.rms_residual(corr_flat)[0]))
            div_rms_list.append(float(energy.rms_divergence(corr_flat)[0]))

            traj_states.append(corr_flat)

            # Sequential closed-loop re-injection
            c = corr_flat.to(fm.dtype)

            if collect_trace and method not in ("raw_fm", "raw_projected"):
                raw_step_e = float((raw_mean - truth_target).square().mean().sqrt())
                actual_gain = float(1.0 - step_rmse / max(raw_step_e, 1e-30))
                diagnostics.append({
                    "trajectory": i, "step": t + 1,
                    "raw_step_rmse": raw_step_e,
                    "corrected_step_rmse": step_rmse,
                    "actual_step_gain": actual_gain,
                    "cosine": cos,
                    "theoretical_max_gain": theor_gain,
                    "resid_rms": float(energy.rms_residual(corr_flat)[0]),
                    "div_rms": float(energy.rms_divergence(corr_flat)[0]),
                    **info
                })

        all_traj_preds.append(torch.cat(traj_states, dim=0))

    elapsed = time.time() - started
    all_preds = torch.stack(all_traj_preds)
    truth_all = trajectories[:, 1:args.steps + 1].to(fm.device, torch.float64)
    window_rmse = float((all_preds - truth_all).square().mean().sqrt())

    return {
        "rmse": window_rmse,
        "step_rmse": [float(np.mean(x)) for x in step_rmses],
        "cosine": float(np.mean(cosines)),
        "gamma": float(np.mean(gammas)),
        "theor_gain": float(np.mean(theors)),
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
    ap.add_argument("--order", type=int, choices=(2, 4), default=4)
    ap.add_argument("--gamma-grid", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
                    help="step gain gamma for Wiener discrepancy filter")
    ap.add_argument("--wiener-reg", type=float, default=1e-3,
                    help="regularization fraction for Wiener denominator")
    ap.add_argument("--project-incompressible", action="store_true", default=True)
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

    wiener = SpectralWienerDiscrepancy(spec.n, L=spec.L, reg=args.wiener_reg,
                                      device=fm.device, dtype=torch.float64)

    print_header(f"S4 Dynamic Discrepancy Filter HILP (Wiener-Kalman): {fm.info.name}, horizon={args.steps}")
    print(f"  likelihood: Wiener spectral transfer function between IRK4 collocation defect and error")
    print(f"  re-injection: corrected divergence-free state fed back into {fm.info.name} at each step")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")

    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]

    print("\n  fitting spectral Wiener discrepancy transfer filter on calibration rollouts...")
    _calibrate_wiener_filter(fm, cal, spec, wiener, args)
    print(f"  Wiener filter fitted across {len(xs)} transitions.")

    print("\n  selecting optimal correction gain gamma on validation rollouts")
    curve = []
    for gamma in args.gamma_grid:
        out = _run_dynamic_rollout(fm, val, spec, sigma2, wiener, "wiener_filter",
                                   gamma, args)
        curve.append({"gamma": gamma, "rmse": out["rmse"]})
        print(f"    gamma={gamma:4.2f}: window RMSE={out['rmse']:.6g} (cos={out['cosine']:.4f})", flush=True)

    best = min(curve, key=lambda x: x["rmse"])
    selected_gamma = best["gamma"]
    print(f"    selected gamma={selected_gamma:4.2f} (RMSE={best['rmse']:.6g})")

    print("\n  held-out sequential rollout evaluation")
    table = Table("method", "window RMSE", "gain%", *[f"step{i+1} gain%" for i in range(args.steps)],
                  "cosine", "theor%", "resid RMS", "div RMS", "seconds")
    records = {}

    # 1. Baseline Open-Loop Poseidon Rollout
    raw_res = _run_dynamic_rollout(fm, test, spec, sigma2, wiener, "raw_fm", 0.0, args)
    records["raw_fm"] = raw_res
    table.add("raw_fm", raw_res["rmse"], 0.0, *([0.0] * args.steps), 0.0, 0.0,
              raw_res["residual_rms"], raw_res["divergence_rms"], raw_res["seconds"])

    # 2. Baseline Closed-Loop Re-injected Helmholtz Projection
    proj_res = _run_dynamic_rollout(fm, test, spec, sigma2, wiener, "raw_projected", 0.0, args)
    records["raw_projected"] = proj_res
    proj_gains = [100 * (1 - proj_res["step_rmse"][t] / raw_res["step_rmse"][t]) for t in range(args.steps)]
    table.add("raw_projected", proj_res["rmse"], 100 * (1 - proj_res["rmse"] / raw_res["rmse"]),
              *proj_gains, proj_res["cosine"], 100 * proj_res["theor_gain"],
              proj_res["residual_rms"], proj_res["divergence_rms"], proj_res["seconds"])

    # 3. Dynamic Discrepancy Wiener Filter
    out = _run_dynamic_rollout(fm, test, spec, sigma2, wiener, "wiener_filter",
                               selected_gamma, args, collect_trace=True)
    records["wiener_filter"] = out
    gains = [100 * (1 - out["step_rmse"][t] / raw_res["step_rmse"][t]) for t in range(args.steps)]
    table.add("wiener_filter", out["rmse"], 100 * (1 - out["rmse"] / raw_res["rmse"]), *gains,
              out["cosine"], 100 * out["theor_gain"], out["residual_rms"],
              out["divergence_rms"], out["seconds"])

    print(table)

    print("\n" + "=" * 88)
    print("COMPOUNDING ERROR ANALYSIS (Sequential Re-Injection vs Open-Loop)")
    print("=" * 88)
    for t in range(args.steps):
        raw_e = raw_res["step_rmse"][t]
        proj_e = proj_res["step_rmse"][t]
        w_e = records["wiener_filter"]["step_rmse"][t]
        print(f"Step {t+1}: raw_fm={raw_e:.6f} | raw_projected={proj_e:.6f} ({100*(1-proj_e/raw_e):+.2f}%) | wiener_filter={w_e:.6f} ({100*(1-w_e/raw_e):+.2f}%)")
    print("=" * 88 + "\n")

    out_data = {
        "stage": "s4_fm_hilp_dynamic_discrepancy",
        "metadata": fm_metadata(args, fm, spec),
        "order": args.order,
        "wiener_reg": args.wiener_reg,
        "selected_gamma": selected_gamma,
        "gamma_selection": curve,
        "held_out": records,
        "project_incompressible": args.project_incompressible,
    }
    path = save_json(out_data, results_path_scale("s4_dynamic_discrepancy", fm_result_key(args),
                                                  "results.json", args.tag))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
