#!/usr/bin/env python3
"""Local Temporal Interpolation Benchmark for 2D Navier-Stokes Vorticity.

Given snapshots at t0 and t2 = t0 + 0.2 (spanning Delta t = 0.2), we evaluate
methods for predicting the intermediate snapshot at t1 = t0 + 0.1 (ZERO forward ODE solves)
and compare against the ground truth frame at t1.

Candidate Interpolation Methods:
1. Linear (Straight-Line) Interpolation:
       w_linear = 0.5 * (w0 + w2)
2. Hermite Cubic Kinematic Interpolant:
       Uses instantaneous PDE derivatives dw/dt = F(w) at endpoints:
       w_hermite = 0.5 * (w0 + w2) + (dt / 8) * (F(w0) - F(w2))
3. Hermite Quintic Kinematic Interpolant:
       Uses acceleration d2w/dt2 via directional Jacobian-vector products (JVP):
       w_quintic = w_hermite + (dt^2 / 384) * (d2w_0 - d2w_2)
4. Spectral Phase Interpolation:
       Interpolates Fourier log-amplitudes and unwrapped phases separately.
5. Semi-Lagrangian Characteristic Transport:
       Advects w0 forward by dt/2 and w2 backward by dt/2 along the mean velocity streamline:
       w_mid = 0.5 * (Advect(w0, +u_mid, dt/2) + Advect(w2, -u_mid, dt/2))
6. Viscous-Corrected Hermite:
       Applies exact viscous decay factor exp(-nu * k^2 * dt/2) to Hermite midpoint.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hipp.scale.data2d import PDESpec2D, SpectralGrid2D
from hipp.utils import Table, print_header


def compute_rhs(grid: SpectralGrid2D, w: torch.Tensor) -> torch.Tensor:
    """Evaluate instantaneous physical RHS: dw/dt = -(u . grad)w + nu * Lap(w) + forcing."""
    wh = torch.fft.rfft2(w)
    rhs_h = grid.rhs_hat(wh)
    if grid.forcing is not None:
        rhs_h = rhs_h + torch.fft.rfft2(grid.forcing)
    return torch.fft.irfft2(rhs_h, s=(grid.spec.n, grid.spec.n))


def compute_acceleration(grid: SpectralGrid2D, w: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """Compute 2nd time derivative d2w/dt2 = (dF/dw) * f0 via forward-mode autodiff JVP."""
    w_var = w.detach().clone().requires_grad_(True)
    wh = torch.fft.rfft2(w_var)
    rhs_h = grid.rhs_hat(wh)
    if grid.forcing is not None:
        rhs_h = rhs_h + torch.fft.rfft2(grid.forcing)
    f_var = torch.fft.irfft2(rhs_h, s=(grid.spec.n, grid.spec.n))

    # JVP: directional derivative in direction f0
    _, jvp = torch.autograd.functional.jvp(
        lambda x: compute_rhs(grid, x), (w,), (f0,)
    )
    return jvp


def semi_lagrangian_advection(grid: SpectralGrid2D, w: torch.Tensor, u: torch.Tensor,
                              v: torch.Tensor, dt: float) -> torch.Tensor:
    """Backtrace particles along steady velocity (u, v) over time dt and sample via grid_sample."""
    n = grid.spec.n
    L = grid.spec.L

    # Normalized mesh coordinates in [-1, 1] for torch.nn.functional.grid_sample
    x_idx = torch.linspace(-1, 1 - 2/n, n, device=w.device, dtype=w.dtype)
    grid_y, grid_x = torch.meshgrid(x_idx, x_idx, indexing="ij")

    # Shift by displacement d = u * dt, periodic on [-1, 1)
    disp_x = (u * dt) / (L / 2)
    disp_y = (v * dt) / (L / 2)

    sample_x = (grid_x - disp_x + 1) % 2 - 1
    sample_y = (grid_y - disp_y + 1) % 2 - 1

    sampling_grid = torch.stack((sample_x, sample_y), dim=-1).unsqueeze(0)  # (1, n, n, 2)
    w_sampled = F.grid_sample(w.unsqueeze(0).unsqueeze(0), sampling_grid,
                               mode="bicubic", padding_mode="border", align_corners=False)
    return w_sampled.squeeze(0).squeeze(0)


def spectral_phase_interp(w0: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Interpolate log-amplitudes and phases in 2D Fourier space."""
    w0_h = torch.fft.rfft2(w0)
    w2_h = torch.fft.rfft2(w2)

    amp0 = torch.abs(w0_h)
    amp2 = torch.abs(w2_h)
    phase0 = torch.angle(w0_h)
    phase2 = torch.angle(w2_h)

    amp_mid = 0.5 * (amp0 + amp2)

    # Angular difference with 2pi wrap-around
    dphase = (phase2 - phase0 + math.pi) % (2 * math.pi) - math.pi
    phase_mid = phase0 + 0.5 * dphase

    wh_mid = amp_mid * torch.exp(1j * phase_mid)
    return torch.fft.irfft2(wh_mid, s=(w0.shape[-2], w0.shape[-1]))


def evaluate_interpolation_methods(data_path: str, n_trajectories: int = 128):
    data = np.load(data_path)
    n_traj = min(n_trajectories, data.shape[0])

    spec = PDESpec2D(
        name="local_ns2d_forced_actual_like",
        L=2 * math.pi,
        n=32,
        nu=0.0001,
        dt=0.0005,
        stride=200,
        drag=0.0,
        forcing="li",
        forcing_amp=0.1,
        forcing_k=4,
        warmup=0,
        ic_peak_k=4.0
    )
    grid = SpectralGrid2D(spec, device="cpu", dtype=torch.float64)

    dt = 0.2  # interval between t0 and t2

    methods = [
        "linear_midpoint",
        "hermite_cubic",
        "hermite_quintic",
        "viscous_hermite",
        "semi_lagrangian",
        "spectral_phase",
    ]

    results = {m: {"rmse": [], "rel_err": [], "enstrophy_err": [], "time_ms": []} for m in methods}

    print_header(f"Evaluating Interpolation Methods on {n_traj} Trajectories (Delta t = {dt}s -> {dt/2}s)")
    print(f"  dataset: {data_path}")
    print(f"  grid: {spec.n}x{spec.n}, nu={spec.nu}, forcing={spec.forcing}")
    print(f"  input: Frame 0 (t=0.0) and Frame 2 (t=0.2) -> target: Frame 1 (t=0.1)")

    for i in range(n_traj):
        w0 = torch.from_numpy(data[i, 0]).to(torch.float64)
        w1_true = torch.from_numpy(data[i, 1]).to(torch.float64)
        w2 = torch.from_numpy(data[i, 2]).to(torch.float64)

        norm_true = w1_true.norm().clamp_min(1e-12)
        ens_true = 0.5 * w1_true.square().mean()

        # 1. Linear Midpoint
        t_start = time.perf_counter()
        w_lin = 0.5 * (w0 + w2)
        results["linear_midpoint"]["time_ms"].append((time.perf_counter() - t_start) * 1000)

        # 2. Hermite Cubic
        t_start = time.perf_counter()
        f0 = compute_rhs(grid, w0)
        f2 = compute_rhs(grid, w2)
        w_hermite = 0.5 * (w0 + w2) + (dt / 8.0) * (f0 - f2)
        results["hermite_cubic"]["time_ms"].append((time.perf_counter() - t_start) * 1000)

        # 3. Hermite Quintic
        t_start = time.perf_counter()
        acc0 = compute_acceleration(grid, w0, f0)
        acc2 = compute_acceleration(grid, w2, f2)
        w_quintic = w_hermite + ((dt ** 2) / 384.0) * (acc0 - acc2)
        results["hermite_quintic"]["time_ms"].append((time.perf_counter() - t_start) * 1000)

        # 4. Viscous-Corrected Hermite
        t_start = time.perf_counter()
        decay = torch.exp(-spec.nu * (dt / 2.0) * grid.k2)
        wh_hermite = torch.fft.rfft2(w_hermite)
        w_visc = torch.fft.irfft2(decay * wh_hermite, s=(spec.n, spec.n))
        results["viscous_hermite"]["time_ms"].append((time.perf_counter() - t_start) * 1000)

        # 5. Semi-Lagrangian
        t_start = time.perf_counter()
        wh_mid = torch.fft.rfft2(w_hermite)
        u_mid, v_mid = grid.velocity(wh_mid)
        w_fwd = semi_lagrangian_advection(grid, w0, u_mid, v_mid, dt / 2.0)
        w_bwd = semi_lagrangian_advection(grid, w2, -u_mid, -v_mid, dt / 2.0)
        w_sl = 0.5 * (w_fwd + w_bwd)
        results["semi_lagrangian"]["time_ms"].append((time.perf_counter() - t_start) * 1000)

        # 6. Spectral Phase
        t_start = time.perf_counter()
        w_phase = spectral_phase_interp(w0, w2)
        results["spectral_phase"]["time_ms"].append((time.perf_counter() - t_start) * 1000)

        preds = {
            "linear_midpoint": w_lin,
            "hermite_cubic": w_hermite,
            "hermite_quintic": w_quintic,
            "viscous_hermite": w_visc,
            "semi_lagrangian": w_sl,
            "spectral_phase": w_phase,
        }

        for m, pred in preds.items():
            diff = pred - w1_true
            results[m]["rmse"].append(float(diff.square().mean().sqrt()))
            results[m]["rel_err"].append(float(diff.norm() / norm_true))
            ens_pred = 0.5 * pred.square().mean()
            results[m]["enstrophy_err"].append(float(abs(ens_pred - ens_true) / ens_true))

    lin_rmse = np.mean(results["linear_midpoint"]["rmse"])

    table = Table("Interpolation Method", "RMSE", "Rel Error %", "Gain vs Linear %", "Enstrophy Err %", "Time (ms)")
    for m in methods:
        m_rmse = float(np.mean(results[m]["rmse"]))
        m_rel = float(np.mean(results[m]["rel_err"])) * 100
        m_gain = 100.0 * (1.0 - m_rmse / lin_rmse)
        m_ens = float(np.mean(results[m]["enstrophy_err"])) * 100
        m_time = float(np.mean(results[m]["time_ms"]))
        table.add(m, f"{m_rmse:.6f}", f"{m_rel:.2f}%", f"{m_gain:+.2f}%", f"{m_ens:.2f}%", f"{m_time:.2f}ms")

    print(table)


if __name__ == "__main__":
    evaluate_interpolation_methods(
        "/Volumes/ExternalSSD/HILP/data/actual_like_ns2d_forced_n32_dt01_poc_2k/test/000000.npy",
        n_trajectories=128
    )
