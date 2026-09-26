#!/usr/bin/env python3
"""Fine-Time Interpolation and HILP Residual Benchmark on Densely Sampled Navier-Stokes.

Uses /Volumes/ExternalSSD/HILP/data/dense_ns2d_forced_n64_dt001 (dt = 0.01s, 101 frames, 64x64).

Evaluates:
Part 1: Temporal Interpolation Scaling Across Cadences Delta t in [0.02, 0.04, 0.06, 0.08, 0.10].
        Compares Linear Midpoint vs Hermite Kinematic Midpoint against exact ground truth.
Part 2: HILP Physical Residual Fidelity & Noise Floor:
        Compares standard midpoint collocation residual vs Hermite-Simpson 4th-order residual:
        - Truncation noise floor on true states: ||r(w*)||
        - Directional alignment with forecast error: cos(r, e)
        - Theoretical reachability gain: 1 - sqrt(1 - cos^2)
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hipp.scale.data2d import PDESpec2D, SpectralGrid2D
from hipp.utils import Table, print_header


def compute_rhs(grid: SpectralGrid2D, w: torch.Tensor) -> torch.Tensor:
    """Instantaneous physical RHS: dw/dt = -(u . grad)w + nu * Lap(w) + forcing."""
    wh = torch.fft.rfft2(w)
    rhs_h = grid.rhs_hat(wh)
    if grid.forcing is not None:
        rhs_h = rhs_h + torch.fft.rfft2(grid.forcing)
    return torch.fft.irfft2(rhs_h, s=(grid.spec.n, grid.spec.n))


def hermite_midpoint(grid: SpectralGrid2D, w0: torch.Tensor, w1: torch.Tensor, dt: float) -> torch.Tensor:
    """Exact Hermite cubic kinematic midpoint: 0 forward ODE steps, O(dt^4) accuracy."""
    f0 = compute_rhs(grid, w0)
    f1 = compute_rhs(grid, w1)
    return 0.5 * (w0 + w1) + (dt / 8.0) * (f0 - f1)


def simpson_hermite_residual(grid: SpectralGrid2D, w0: torch.Tensor, w1: torch.Tensor, dt: float) -> torch.Tensor:
    """4th-order Hermite-Simpson algebraic collocation residual (0 forward ODE steps)."""
    f0 = compute_rhs(grid, w0)
    f1 = compute_rhs(grid, w1)
    w_mid = 0.5 * (w0 + w1) + (dt / 8.0) * (f0 - f1)
    f_mid = compute_rhs(grid, w_mid)
    return (w1 - w0) / dt - (1.0 / 6.0) * (f0 + 4.0 * f_mid + f1)


def standard_midpoint_residual(grid: SpectralGrid2D, w0: torch.Tensor, w1: torch.Tensor, dt: float) -> torch.Tensor:
    """Standard midpoint collocation residual (O(dt^2) truncation error)."""
    w_mid = 0.5 * (w0 + w1)
    f_mid = compute_rhs(grid, w_mid)
    return (w1 - w0) / dt - f_mid


def main():
    data_path = Path("/Volumes/ExternalSSD/HILP/data/dense_ns2d_forced_n64_dt001/test.npy")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    data = np.load(data_path)  # (8, 101, 64, 64)
    n_traj, n_frames, n_grid, _ = data.shape

    spec = PDESpec2D(
        name="local_ns2d_forced_dense",
        L=2 * math.pi,
        n=64,
        nu=0.0001,
        dt=0.0005,
        stride=20,
        drag=0.0,
        forcing="li",
        forcing_amp=0.1,
        forcing_k=4,
        warmup=0,
        ic_peak_k=4.0
    )
    grid = SpectralGrid2D(spec, device="cpu", dtype=torch.float64)

    stored_dt = 0.01  # each frame is 0.01s

    print_header(f"Finer-t Benchmark: Interpolation & HILP Residuals on Dense NS ({n_grid}x{n_grid}, dt={stored_dt}s)")
    print(f"  test set: {n_traj} trajectories, {n_frames} frames ({n_frames * stored_dt:.1f}s total horizon)")
    print(f"  integration: STRICTLY ZERO FORWARD ODE SOLVES across all methods")

    # =========================================================================
    # PART 1: Interpolation Across Finer Temporal Cadences
    # =========================================================================
    print("\n" + "=" * 95)
    print("PART 1: TEMPORAL INTERPOLATION SCALING ACROSS CADENCES (ZERO ODE STEPS)")
    print("=" * 95)

    cadence_strides = [2, 4, 6, 8, 10]  # dt = 0.02, 0.04, 0.06, 0.08, 0.10

    table_interp = Table("dt (s)", "Linear RMSE", "Hermite RMSE", "Gain %",
                         "Hermite RelErr", "Enstrophy Err %", "Time (ms)")

    for stride in cadence_strides:
        dt = stride * stored_dt
        half_stride = stride // 2
        half_dt = dt / 2.0

        lin_rmses, herm_rmses = [], []
        herm_rel_errs, enstrophy_errs = [], []
        times = []

        for i in range(n_traj):
            # Sample all disjoint intervals across the trajectory
            for start in range(0, n_frames - stride, stride):
                w0 = torch.from_numpy(data[i, start]).to(torch.float64)
                w_mid_true = torch.from_numpy(data[i, start + half_stride]).to(torch.float64)
                w1 = torch.from_numpy(data[i, start + stride]).to(torch.float64)

                norm_true = w_mid_true.norm().clamp_min(1e-12)
                ens_true = 0.5 * w_mid_true.square().mean()

                # Linear
                w_lin = 0.5 * (w0 + w1)

                # Hermite Kinematic
                t0 = time.perf_counter()
                w_herm = hermite_midpoint(grid, w0, w1, dt)
                times.append((time.perf_counter() - t0) * 1000)

                lin_rmses.append(float((w_lin - w_mid_true).square().mean().sqrt()))
                diff_herm = w_herm - w_mid_true
                herm_rmses.append(float(diff_herm.square().mean().sqrt()))
                herm_rel_errs.append(float(diff_herm.norm() / norm_true))

                ens_herm = 0.5 * w_herm.square().mean()
                enstrophy_errs.append(float(abs(ens_herm - ens_true) / ens_true))

        mean_lin = float(np.mean(lin_rmses))
        mean_herm = float(np.mean(herm_rmses))
        gain = 100.0 * (1.0 - mean_herm / mean_lin)
        mean_rel = float(np.mean(herm_rel_errs)) * 100.0
        mean_ens = float(np.mean(enstrophy_errs)) * 100.0
        mean_time = float(np.mean(times))

        table_interp.add(f"{dt:.2f}s", f"{mean_lin:.6f}", f"{mean_herm:.6f}",
                         f"{gain:+.2f}%", f"{mean_rel:.4f}%", f"{mean_ens:.4f}%", f"{mean_time:.2f}ms")

    print(table_interp)

    # =========================================================================
    # PART 2: HILP Residual Fidelity & Noise Floor Audit
    # =========================================================================
    print("\n" + "=" * 95)
    print("PART 2: HILP PHYSICAL RESIDUAL NOISE FLOOR & FORECAST ERROR ALIGNMENT")
    print("=" * 95)

    table_hilp = Table("dt (s)", "Midpoint Noise Floor", "Hermite Noise Floor", "Noise Reduction",
                       "Midpoint cos", "Hermite cos", "Hermite Reach Gain%")

    for stride in cadence_strides:
        dt = stride * stored_dt

        mid_noise_floors, herm_noise_floors = [], []
        mid_cosines, herm_cosines = [], []
        herm_reach_gains = []

        for i in range(n_traj):
            for start in range(0, n_frames - stride, stride):
                w0 = torch.from_numpy(data[i, start]).to(torch.float64)
                w1_true = torch.from_numpy(data[i, start + stride]).to(torch.float64)

                # 1. Evaluate Residual on TRUE trajectory (measures discretization noise floor)
                r_mid_true = standard_midpoint_residual(grid, w0, w1_true, dt)
                r_herm_true = simpson_hermite_residual(grid, w0, w1_true, dt)

                mid_noise_floors.append(float(r_mid_true.square().mean().sqrt()))
                herm_noise_floors.append(float(r_herm_true.square().mean().sqrt()))

                # 2. Simulate realistic Neural Operator forecast error:
                # Spectral blurring + 5% high-frequency phase drift
                wh_true = torch.fft.rfft2(w1_true)
                k2 = grid.k2
                blur = torch.exp(-0.0003 * k2)
                phase_noise = 0.05 * torch.randn_like(k2)
                wh_cand = wh_true * blur * torch.exp(1j * phase_noise)
                w1_cand = torch.fft.irfft2(wh_cand, s=(spec.n, spec.n))

                # Ground truth forecast error vector
                error_vec = (w1_cand - w1_true).reshape(-1)
                err_norm = error_vec.norm().clamp_min(1e-12)

                # Residuals evaluated on candidate prediction
                r_mid_cand = standard_midpoint_residual(grid, w0, w1_cand, dt).reshape(-1)
                r_herm_cand = simpson_hermite_residual(grid, w0, w1_cand, dt).reshape(-1)

                cos_mid = float((r_mid_cand @ error_vec) / (r_mid_cand.norm() * err_norm).clamp_min(1e-12))
                cos_herm = float((r_herm_cand @ error_vec) / (r_herm_cand.norm() * err_norm).clamp_min(1e-12))

                mid_cosines.append(cos_mid)
                herm_cosines.append(cos_herm)

                reach_gain = 100.0 * (1.0 - math.sqrt(max(0.0, 1.0 - cos_herm**2))) if cos_herm > 0 else 0.0
                herm_reach_gains.append(reach_gain)

        mean_mid_noise = float(np.mean(mid_noise_floors))
        mean_herm_noise = float(np.mean(herm_noise_floors))
        noise_red = 100.0 * (1.0 - mean_herm_noise / mean_mid_noise)

        mean_mid_cos = float(np.mean(mid_cosines))
        mean_herm_cos = float(np.mean(herm_cosines))
        mean_reach_gain = float(np.mean(herm_reach_gains))

        table_hilp.add(f"{dt:.2f}s", f"{mean_mid_noise:.6f}", f"{mean_herm_noise:.6f}",
                       f"{noise_red:+.2f}%", f"{mean_mid_cos:+.4f}", f"{mean_herm_cos:+.4f}",
                       f"+{mean_reach_gain:.2f}%")

    print(table_hilp)


if __name__ == "__main__":
    main()
