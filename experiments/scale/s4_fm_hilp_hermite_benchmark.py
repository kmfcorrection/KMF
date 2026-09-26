#!/usr/bin/env python3
"""S4 Hermite Kinematic Interpolation and 4th-Order HILP Benchmark on GPU.

Replicates the local kinematic interpolation breakthrough on native Poseidon NS-Gauss (128x128):
1. Benchmark 1: Midpoint Kinematic Interpolation Benchmark (0 ODE Steps):
   Evaluates predicting the intermediate physical state at t = 0.05s between t = 0.0s and t = 0.10s
   against the exact raw intermediate frame in NS-Gauss.
   Compares:
     - Linear Midpoint: 0.5 * (u0 + u1)
     - Poseidon Native Half-Step: fm.predict(u0, lead_time=1.0)
     - Hermite Kinematic Midpoint: 0.5 * (u0 + u1) + (dt / 8) * (F(u0) - F(u1))

2. Benchmark 2: HILP Physical Residual Noise Floor & Poseidon Error Alignment:
   Compares the standard midpoint collocation residual vs Hermite-Simpson 4th-order residual.

3. Benchmark 3: Autoregressive Rollouts with Calibrated Corrections (0 Forward ODE Steps):
   - raw_fm (Baseline Poseidon)
   - raw_projected (Helmholtz-Leray incompressibility projection)
   - hermite_normalized_grad (Unit-normalized 4th-order Hermite physical gradient step scaled by sigma_err)
   - spectral_mmse_deconv (Full 2D Fourier MMSE deconvolution transfer function across 128x65 modes)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    POSEIDON_NATIVE_RAW_STRIDE, POSEIDON_RAW_DT, configure_native_poseidon_cadence,
    fm_metadata, fm_result_key, native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.utils import Table, print_header, save_json, set_seed


def load_raw_cadence_trajectories(path: str | Path, fm, n_traj: int, steps: int, offset: int = 0):
    """Load raw snapshots including intermediate half-steps (stride 1)."""
    import h5py
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    need_frames = 2 * steps + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        ds = h[key]
        if ds.shape[1] < need_frames:
            raise ValueError(f"need {need_frames} frames, found {ds.shape[1]}")
        arr = np.asarray(ds[offset:offset + n_traj, :need_frames, :2], dtype=np.float32)

    return torch.as_tensor(arr, device=fm.device, dtype=torch.float64)  # (N, need_frames, 2, 128, 128)


def hermite_kinematic_midpoint(op: FMPhysicsEnergy2D, u0: torch.Tensor, u1: torch.Tensor,
                               dt: float) -> torch.Tensor:
    """Exact Hermite kinematic midpoint: 0 forward ODE steps, O(dt^4) streamline curvature."""
    return hermite_kinematic_spline(op, u0, u1, dt, s=0.5)


def hermite_kinematic_spline(op: FMPhysicsEnergy2D, u0: torch.Tensor, u1: torch.Tensor,
                             dt: float, s: float = 0.5) -> torch.Tensor:
    """Exact Hermite kinematic spline for arbitrary normalized intermediate time s in [0, 1].

    Preserves exact physical values and Navier-Stokes time derivatives at both endpoints
    with strictly 0 forward ODE integration steps.
    """
    w0 = op.to_vorticity(u0)
    w1 = op.to_vorticity(u1)
    f0 = op.rhs_vorticity(w0)
    f1 = op.rhs_vorticity(w1)

    h00 = 1.0 - 3.0 * s**2 + 2.0 * s**3
    h10 = 3.0 * s**2 - 2.0 * s**3
    h01 = dt * (s - 2.0 * s**2 + s**3)
    h11 = dt * (-s**2 + s**3)

    w_interp = h00 * w0 + h10 * w1 + h01 * f0 + h11 * f1
    wh_interp = torch.fft.rfft2(w_interp)
    u, v = op.grid.velocity(wh_interp)
    mean_vel = (1.0 - s) * u0.mean(dim=(-2, -1), keepdim=True) + s * u1.mean(dim=(-2, -1), keepdim=True)
    return torch.stack((u, v), dim=1) + mean_vel


def hermite_simpson_residual(op: FMPhysicsEnergy2D, u0: torch.Tensor, u1: torch.Tensor,
                             dt: float) -> torch.Tensor:
    """4th-order Hermite-Simpson physical residual in vorticity space (0 forward ODE steps)."""
    w0 = op.to_vorticity(u0)
    w1 = op.to_vorticity(u1)
    f0 = op.rhs_vorticity(w0)
    f1 = op.rhs_vorticity(w1)

    w_mid = 0.5 * (w0 + w1) + (dt / 8.0) * (f0 - f1)
    f_mid = op.rhs_vorticity(w_mid)
    return (w1 - w0) / dt - (1.0 / 6.0) * (f0 + 4.0 * f_mid + f1)


def standard_midpoint_residual(op: FMPhysicsEnergy2D, u0: torch.Tensor, u1: torch.Tensor,
                               dt: float) -> torch.Tensor:
    """Standard linear midpoint collocation residual (O(dt^2) truncation error)."""
    w0 = op.to_vorticity(u0)
    w1 = op.to_vorticity(u1)
    w_mid = 0.5 * (w0 + w1)
    f_mid = op.rhs_vorticity(w_mid)
    return (w1 - w0) / dt - f_mid


def lift_vorticity_to_velocity(op: FMPhysicsEnergy2D, w: torch.Tensor, spec, c_mean: torch.Tensor) -> torch.Tensor:
    """Lift vorticity into divergence-free velocity with DC momentum conservation."""
    wh = torch.fft.rfft2(w.reshape(-1, spec.n, spec.n))
    psih = wh * op.grid.inv_k2
    u = torch.fft.irfft2(1j * op.grid.ky * psih, s=(spec.n, spec.n))
    v = torch.fft.irfft2(-1j * op.grid.kx * psih, s=(spec.n, spec.n))
    vel = torch.stack((u, v), dim=1)
    return vel + c_mean


def _run_rollout(fm, trajectories, spec, method, param, args,
                 sigma_err: float = 0.0068, mmse_kernel: torch.Tensor | None = None,
                 best_mmse_gamma: float = 0.40):
    """Closed-loop autoregressive rollout with Hermite-Simpson & Spectral MMSE (0 ODE STEPS)."""
    dt = args.lead_steps * spec.dt_out
    n_traj = trajectories.shape[0]
    step_rmses = [[] for _ in range(args.steps)]
    cosines, theors, div_rms_list = [], [], []
    energy_errs, enstrophy_errs = [], []
    started = time.time()

    op = FMPhysicsEnergy2D(spec, trajectories[0, 0].to(fm.device, torch.float64), 2, dt=dt, device=fm.device)
    k_max = spec.n // 2
    smooth_filter = torch.exp(-36.0 * (op.grid.k2.sqrt() / k_max).pow(36))

    for i in range(n_traj):
        c = trajectories[i, 0].to(fm.device, fm.dtype).reshape(1, 2, spec.n, spec.n)

        for t in range(args.steps):
            target_idx = 2 * (t + 1)
            truth_target = trajectories[i, target_idx].to(fm.device, torch.float64).reshape(1, 2, spec.n, spec.n)

            fm.set_lead_time(2.0)
            with torch.no_grad():
                raw_mean = fm.predict(c.to(fm.dtype)).double().reshape(1, 2, spec.n, spec.n)

            raw_proj = op.project_incompressible(raw_mean).reshape(1, 2, spec.n, spec.n)
            c_mean = c.mean(dim=(-2, -1), keepdim=True)

            if method == "raw_fm":
                corrected = raw_mean
            elif method == "raw_projected":
                corrected = raw_proj
            elif method == "spectral_mmse_deconv":
                gamma = float(param)
                if gamma == 0.0 or mmse_kernel is None:
                    corrected = raw_proj
                else:
                    # Full 2D Fourier MMSE deconvolution (0 ODE steps, < 1ms)
                    w_proj = op.to_vorticity(raw_proj)
                    wh_proj = torch.fft.rfft2(w_proj)
                    H_gamma = (1.0 - gamma) + gamma * mmse_kernel
                    wh_corr = H_gamma * wh_proj
                    w_corr = torch.fft.irfft2(wh_corr, s=(spec.n, spec.n))
                    corrected = lift_vorticity_to_velocity(op, w_corr, spec, c_mean)
                    corrected = op.project_incompressible(corrected).reshape(1, 2, spec.n, spec.n)
            elif method == "hermite_simpson_quad":
                alpha = float(param)
                if alpha == 0.0:
                    corrected = raw_proj
                else:
                    w0 = op.to_vorticity(c)
                    w1 = op.to_vorticity(raw_proj)
                    f0 = op.rhs_vorticity(w0)
                    f1 = op.rhs_vorticity(w1)

                    # Hermite kinematic midpoint (0 ODE steps)
                    w_mid = 0.5 * (w0 + w1) + (dt / 8.0) * (f0 - f1)
                    f_mid = op.rhs_vorticity(w_mid)

                    # 4th-order Simpson reconstruction
                    w_simp = w0 + (dt / 6.0) * (f0 + 4.0 * f_mid + f1)
                    wh_diff = torch.fft.rfft2(w_simp - w1)
                    w_diff_filt = torch.fft.irfft2(wh_diff * smooth_filter, s=(spec.n, spec.n))

                    w_corr = w1 + alpha * w_diff_filt
                    corrected = lift_vorticity_to_velocity(op, w_corr, spec, c_mean)
                    corrected = op.project_incompressible(corrected).reshape(1, 2, spec.n, spec.n)
            elif method in ("dual_cadence_hs", "hybrid_hs_mmse"):
                alpha = float(param)
                if alpha == 0.0 and method == "dual_cadence_hs":
                    corrected = raw_proj
                else:
                    w0 = op.to_vorticity(c)
                    w1 = op.to_vorticity(raw_proj)
                    f0 = op.rhs_vorticity(w0)
                    f1 = op.rhs_vorticity(w1)

                    # Poseidon native half-step prediction (lead_time=1.0, 0.05s)
                    fm.set_lead_time(1.0)
                    with torch.no_grad():
                        raw_half = fm.predict(c.to(fm.dtype)).double().reshape(1, 2, spec.n, spec.n)
                    raw_half_proj = op.project_incompressible(raw_half).reshape(1, 2, spec.n, spec.n)
                    w_mid_fm = op.to_vorticity(raw_half_proj)

                    # Consensus between Hermite kinematic curvature and Poseidon neural half-step
                    w_mid_herm = 0.5 * (w0 + w1) + (dt / 8.0) * (f0 - f1)
                    w_mid = 0.5 * (w_mid_herm + w_mid_fm)
                    f_mid = op.rhs_vorticity(w_mid)

                    # 4th-order Simpson physical quadrature
                    w_simp = w0 + (dt / 6.0) * (f0 + 4.0 * f_mid + f1)
                    wh_diff = torch.fft.rfft2(w_simp - w1)
                    w_diff_filt = torch.fft.irfft2(wh_diff * smooth_filter, s=(spec.n, spec.n))
                    w_corr = w1 + alpha * w_diff_filt

                    if method == "hybrid_hs_mmse" and mmse_kernel is not None:
                        wh_c = torch.fft.rfft2(w_corr)
                        H_gamma = (1.0 - best_mmse_gamma) + best_mmse_gamma * mmse_kernel
                        w_corr = torch.fft.irfft2(H_gamma * wh_c, s=(spec.n, spec.n))

                    corrected = lift_vorticity_to_velocity(op, w_corr, spec, c_mean)
                    corrected = op.project_incompressible(corrected).reshape(1, 2, spec.n, spec.n)
            else:
                raise ValueError(f"Unknown method {method}")

            corrected = corrected.reshape(1, 2, spec.n, spec.n)

            # Error and alignment diagnostics
            err = (corrected - truth_target).reshape(-1)
            step_rmse = float(err.square().mean().sqrt())
            step_rmses[t].append(step_rmse)

            corr_vec = (corrected - raw_proj).reshape(-1)
            target_vec = (truth_target - raw_proj).reshape(-1)
            corr_norm = corr_vec.norm()
            target_norm = target_vec.norm()
            cos = float((corr_vec @ target_vec) / (corr_norm * target_norm).clamp_min(1e-30))
            theor_gain = float(1.0 - math.sqrt(max(0.0, 1.0 - cos**2))) if cos > 0 else 0.0
            cosines.append(cos)
            theors.append(theor_gain)

            div_rms = float(op.divergence(corrected).square().mean().sqrt())
            div_rms_list.append(div_rms)

            E_corr = float(0.5 * corrected.square().mean())
            E_true = float(0.5 * truth_target.square().mean())
            energy_errs.append(abs(E_corr - E_true) / max(E_true, 1e-12))

            w_corr = op.to_vorticity(corrected)
            w_true = op.to_vorticity(truth_target)
            Ens_corr = float(0.5 * w_corr.square().mean())
            Ens_true = float(0.5 * w_true.square().mean())
            enstrophy_errs.append(abs(Ens_corr - Ens_true) / max(Ens_true, 1e-12))

            # Autoregressive state update
            c = corrected.detach()

    window_rmse = float(np.mean([np.mean(step_rmses[t]) for t in range(args.steps)]))
    mean_step_rmses = [float(np.mean(step_rmses[t])) for t in range(args.steps)]

    return {
        "rmse": window_rmse,
        "step_rmse": mean_step_rmses,
        "cosine": float(np.mean(cosines)),
        "theor_gain": float(np.mean(theors)),
        "substeps": 0.0,
        "div_rms": float(np.mean(div_rms_list)),
        "energy_rel_err": float(np.mean(energy_errs)),
        "enstrophy_rel_err": float(np.mean(enstrophy_errs)),
        "seconds": time.time() - started,
    }


def main():
    parser = base_parser_scale(__doc__.splitlines()[0])
    parser.add_argument("--lead-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--n-cal-traj", type=int, default=16)
    parser.add_argument("--n-val-traj", type=int, default=4)
    parser.add_argument("--n-test-traj", type=int, default=4)
    parser.add_argument("--alpha-grid", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
                        help="sweep grid for Hermite-Simpson blending weight alpha")
    parser.add_argument("--mmse-gamma-grid", nargs="+", type=float,
                        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        help="sweep grid for MMSE deconvolution gain gamma")
    parser.add_argument("--fm-data-path", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)

    total_traj = args.n_cal_traj + args.n_val_traj + args.n_test_traj
    all_trajs = load_raw_cadence_trajectories(args.fm_data_path, fm, total_traj, args.steps, offset=0)

    cal = all_trajs[:args.n_cal_traj]
    val = all_trajs[args.n_cal_traj:args.n_cal_traj + args.n_val_traj]
    test = all_trajs[args.n_cal_traj + args.n_val_traj:]

    print_header(f"S4 Hermite & Spectral MMSE Benchmark: {fm.info.name} (128x128)")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")
    print(f"  integration: STRICTLY ZERO FORWARD ODE SOLVES across all methods")

    op = FMPhysicsEnergy2D(spec, test[0, 0].to(fm.device, torch.float64), 2, dt=0.1, device=fm.device)
    dt_transition = 0.10

    # =========================================================================
    # BENCHMARK 1: Multi-Cadence Kinematic Super-Resolution vs Poseidon
    # =========================================================================
    print("\n" + "=" * 115)
    print("BENCHMARK 1: MULTI-CADENCE KINEMATIC RECONSTRUCTION vs POSEIDON (OFF-TRAINING CADENCE GENERALIZATION)")
    print("=" * 115)
    print("  Testing whether kinematic streamline curvature generalizes across multiple time horizons and off-cadence queries")

    # (start_frame, end_frame, target_frame, dt_span)
    test_cases = [
        (0, 2, 1, 0.10),  # dt=0.10, midpoint t=0.05s (s=0.50)
        (0, 4, 1, 0.20),  # dt=0.20, quarter t=0.05s (s=0.25)
        (0, 4, 2, 0.20),  # dt=0.20, midpoint t=0.10s (s=0.50)
        (0, 4, 3, 0.20),  # dt=0.20, three-quarter t=0.15s (s=0.75)
        (0, 6, 3, 0.30),  # dt=0.30, midpoint t=0.15s (s=0.50)
        (0, 8, 4, 0.40),  # dt=0.40, midpoint t=0.20s (s=0.50)
    ]

    t_interp = Table("Span [t0 -> t1]", "dt", "Target t", "s", "Linear RMSE", "Poseidon FM RMSE", "Hermite Kinematic", "Gain vs Lin", "Gain vs FM")

    for start_f, end_f, target_f, dt_span in test_cases:
        s = (target_f - start_f) / (end_f - start_f)
        lead_units = float(target_f - start_f)
        t_target = target_f * 0.05

        lin_rmses, herm_rmses, fm_rmses = [], [], []

        for i in range(test.shape[0]):
            u0 = test[i, start_f].to(fm.device, torch.float64).reshape(1, 2, spec.n, spec.n)
            u1 = test[i, end_f].to(fm.device, torch.float64).reshape(1, 2, spec.n, spec.n)
            u_true = test[i, target_f].to(fm.device, torch.float64).reshape(1, 2, spec.n, spec.n)

            # 1. Linear interpolation
            u_lin = (1.0 - s) * u0 + s * u1
            lin_rmses.append(float((u_lin - u_true).square().mean().sqrt()))

            # 2. Hermite kinematic spline (0 ODE steps)
            u_herm = hermite_kinematic_spline(op, u0, u1, dt_span, s=s)
            herm_rmses.append(float((u_herm - u_true).square().mean().sqrt()))

            # 3. Poseidon direct prediction
            fm.set_lead_time(lead_units)
            with torch.no_grad():
                u_fm = fm.predict(u0.to(fm.dtype)).double().reshape(1, 2, spec.n, spec.n)
            u_fm = op.project_incompressible(u_fm).reshape(1, 2, spec.n, spec.n)
            fm_rmses.append(float((u_fm - u_true).square().mean().sqrt()))

        mean_lin = float(np.mean(lin_rmses))
        mean_herm = float(np.mean(herm_rmses))
        mean_fm = float(np.mean(fm_rmses))

        gain_lin = 100.0 * (1.0 - mean_herm / mean_lin)
        gain_fm = 100.0 * (1.0 - mean_herm / mean_fm)

        t_interp.add(
            f"[{start_f*0.05:.2f}s -> {end_f*0.05:.2f}s]",
            f"{dt_span:.2f}s",
            f"{t_target:.2f}s",
            f"{s:.2f}",
            f"{mean_lin:.6f}",
            f"{mean_fm:.6f}",
            f"{mean_herm:.6f}",
            f"{gain_lin:+.2f}%",
            f"{gain_fm:+.2f}%",
        )

    print(t_interp)

    # =========================================================================
    # CALIBRATION: Estimate Error Scale and Fit 2D Spectral MMSE Kernel
    # =========================================================================
    print("\n  [CALIBRATION] Estimating Poseidon error scale and fitting 2D Spectral MMSE Kernel...")
    cal_errs = []
    cross_sum, auto_sum = None, None
    M = args.n_cal_traj * args.steps

    for i in range(args.n_cal_traj):
        c = cal[i, 0].to(fm.device, fm.dtype).reshape(1, 2, spec.n, spec.n)
        for t in range(args.steps):
            target_idx = 2 * (t + 1)
            truth_target = cal[i, target_idx].to(fm.device, torch.float64).reshape(1, 2, spec.n, spec.n)

            fm.set_lead_time(2.0)
            with torch.no_grad():
                raw_mean = fm.predict(c.to(fm.dtype)).double().reshape(1, 2, spec.n, spec.n)
            raw_proj = op.project_incompressible(raw_mean).reshape(1, 2, spec.n, spec.n)

            err_norm = float((raw_proj - truth_target).norm())
            cal_errs.append(err_norm)

            # Accumulate spectra for MMSE transfer function
            w_p = op.to_vorticity(raw_proj)
            w_t = op.to_vorticity(truth_target)
            wh_p = torch.fft.rfft2(w_p)
            wh_t = torch.fft.rfft2(w_t)

            cross = wh_t * torch.conj(wh_p)
            auto = wh_p.abs() ** 2

            if cross_sum is None:
                cross_sum = cross
                auto_sum = auto
            else:
                cross_sum = cross_sum + cross
                auto_sum = auto_sum + auto

            c = raw_proj.detach()

    sigma_err = float(np.mean(cal_errs)) / math.sqrt(2 * spec.n * spec.n)
    print(f"    calibrated Poseidon error scale sigma_err = {sigma_err:.6f}")

    cross_mean = cross_sum / M
    auto_mean = auto_sum / M
    reg = 1e-3 * auto_mean.mean().clamp_min(1e-12)
    mmse_kernel = cross_mean / (auto_mean + reg)
    print(f"    fitted 2D Spectral MMSE Kernel ({spec.n}x{spec.n//2 + 1} complex modes).")

    # =========================================================================
    # BENCHMARK 3: Autoregressive Rollouts on Held-Out Test Set (0 ODE Steps)
    # =========================================================================
    print("\n" + "=" * 105)
    print("BENCHMARK 3: AUTOREGRESSIVE HILP ROLLOUTS ON HELD-OUT TEST TRAJECTORIES (0 ODE STEPS)")
    print("=" * 105)

    # 1. Sweep gamma for Spectral MMSE deconvolution
    print("  [VALIDATION] Sweeping Spectral MMSE deconvolution gain gamma...")
    val_mmse = []
    for gamma in args.mmse_gamma_grid:
        out = _run_rollout(fm, val, spec, "spectral_mmse_deconv", gamma, args, mmse_kernel=mmse_kernel)
        val_mmse.append((gamma, out["rmse"], out["cosine"]))
        print(f"    gamma={gamma:.2f}: window RMSE={out['rmse']:.8g} (cos={out['cosine']:+.4f})")
    best_gamma_mmse, best_val_rmse_m, best_cos_m = min(val_mmse, key=lambda x: x[1])
    print(f"    >>> SELECTED MMSE gamma={best_gamma_mmse:.2f} (val RMSE={best_val_rmse_m:.8g}, cos={best_cos_m:+.4f})")

    # 2. Sweep alpha for Direct Hermite-Simpson Quadrature
    print("\n  [VALIDATION] Sweeping Direct Hermite-Simpson blend weight alpha...")
    val_hs = []
    for alpha in args.alpha_grid:
        out = _run_rollout(fm, val, spec, "hermite_simpson_quad", alpha, args)
        val_hs.append((alpha, out["rmse"], out["cosine"]))
        print(f"    alpha={alpha:.2f}: window RMSE={out['rmse']:.8g} (cos={out['cosine']:+.4f})")
    best_alpha_hs, best_val_rmse_hs, best_cos_hs = min(val_hs, key=lambda x: x[1])
    print(f"    >>> SELECTED Direct HS alpha={best_alpha_hs:.2f} (val RMSE={best_val_rmse_hs:.8g})")

    # 3. Sweep alpha for Dual-Cadence Consensus Hermite-Simpson
    print("\n  [VALIDATION] Sweeping Dual-Cadence Consensus Hermite-Simpson blend weight alpha...")
    val_dc = []
    for alpha in args.alpha_grid:
        out = _run_rollout(fm, val, spec, "dual_cadence_hs", alpha, args)
        val_dc.append((alpha, out["rmse"], out["cosine"]))
        print(f"    alpha={alpha:.2f}: window RMSE={out['rmse']:.8g} (cos={out['cosine']:+.4f})")
    best_alpha_dc, best_val_rmse_dc, best_cos_dc = min(val_dc, key=lambda x: x[1])
    print(f"    >>> SELECTED Dual-Cadence HS alpha={best_alpha_dc:.2f} (val RMSE={best_val_rmse_dc:.8g})")

    records = {}
    raw_res = _run_rollout(fm, test, spec, "raw_fm", 0.0, args)
    records["raw_fm"] = raw_res

    proj_res = _run_rollout(fm, test, spec, "raw_projected", 0.0, args)
    records["raw_projected"] = proj_res

    mmse_res = _run_rollout(fm, test, spec, "spectral_mmse_deconv", best_gamma_mmse, args, mmse_kernel=mmse_kernel)
    records["spectral_mmse_deconv"] = mmse_res

    hs_res = _run_rollout(fm, test, spec, "hermite_simpson_quad", best_alpha_hs, args)
    records["hermite_simpson_quad"] = hs_res

    dc_res = _run_rollout(fm, test, spec, "dual_cadence_hs", best_alpha_dc, args)
    records["dual_cadence_hs"] = dc_res

    hyb_res = _run_rollout(fm, test, spec, "hybrid_hs_mmse", best_alpha_dc, args,
                           mmse_kernel=mmse_kernel, best_mmse_gamma=best_gamma_mmse)
    records["hybrid_hs_mmse"] = hyb_res

    raw_rmse = raw_res["rmse"]
    table_rollout = Table("method", "window RMSE", "gain%", *[f"step{i+1} gain%" for i in range(args.steps)],
                          "substeps", "cosine", "div RMS", "E err%", "Ens err%", "seconds")

    for name, res in [("raw_fm", raw_res),
                      ("raw_projected", proj_res),
                      ("spectral_mmse_deconv", mmse_res),
                      ("hermite_simpson_quad", hs_res),
                      ("dual_cadence_hs", dc_res),
                      ("hybrid_hs_mmse", hyb_res)]:
        gains = [100 * (1 - res["step_rmse"][t] / raw_res["step_rmse"][t]) for t in range(args.steps)]
        tot_gain = 100 * (1 - res["rmse"] / raw_rmse)
        table_rollout.add(name, res["rmse"], tot_gain, *gains,
                          f"{res['substeps']:.1f}", f"{res['cosine']:+.4f}",
                          f"{res['div_rms']:.1e}", f"{res['energy_rel_err']*100:.2f}%",
                          f"{res['enstrophy_rel_err']*100:.2f}%", f"{res['seconds']:.2f}s")

    print("\n" + str(table_rollout))

    payload = {
        "stage": "s4_hermite_mmse_benchmark",
        "metadata": fm_metadata(args, fm, spec),
        "selected_gamma_mmse": best_gamma_mmse,
        "selected_alpha_hs": best_alpha_hs,
        "selected_alpha_dc": best_alpha_dc,
        "held_out": records,
    }
    path = save_json(payload, results_path_scale("s4_hermite_mmse", fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
