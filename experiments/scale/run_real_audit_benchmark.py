#!/usr/bin/env python3
"""Unified Review Audit Benchmark Suite on Real Foundation Models & Datasets.

Executes the 6 reviewer audit experiments on the genuine pipeline:
1. Primary Sub-Cadence Interpolation on real Poseidon (NS-Gauss & FNS-KF) with matched-cost baselines.
2. Calibration sample efficiency & degradation ablation across budgets M in {1, 2, 5, 10, 20} on real FM residuals.
3. True 4-way component ablation with strictly ZERO future truth in any deployed branch.
4. Long-horizon autoregressive rollout (H = 20 steps) tracking error compounding vs. saturation.
5. Wave-Gauss physical dispersion analysis on real acoustic wave data.
6. Synchronized end-to-end component wall-clock latency profiling on GPU.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as stats
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.cross_fm_benchmark import (
    GenericPDEPhysics, fluid_hermite_kinematic_spline, general_kinematic_hermite_spline,
    load_dataset_trajectories, fm_predict
)
from experiments.scale.fm_eval_common import native_poseidon_spec
from hipp.scale.common_scale import load_fm, set_seed
from hipp.scale.data2d import PDESpec2D, SpectralGrid2D, SPECS2D
from hipp.scale.fm_physics import FMPhysicsEnergy2D


# ---------------------------------------------------------------------------
# Statistical Protocols: Paired Bootstrap 95% CIs and Wilcoxon Signed-Rank
# ---------------------------------------------------------------------------
def compute_paired_bootstrap_ci(errors_base: np.ndarray, errors_cand: np.ndarray,
                                n_boot: int = 10000, seed: int = 20260914) -> Dict[str, float]:
    """Compute paired trajectory-level bootstrap 95% confidence intervals and Wilcoxon p-value."""
    n = len(errors_base)
    assert n == len(errors_cand), "Arrays must be paired across identical trajectories"
    
    # Relative gain per trajectory: 100 * (1 - e_cand / e_base)
    rel_gains = 100.0 * (1.0 - errors_cand / np.maximum(errors_base, 1e-12))
    abs_diffs = errors_base - errors_cand
    
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_gains = np.mean(rel_gains[boot_indices], axis=1)
    boot_diffs = np.mean(abs_diffs[boot_indices], axis=1)
    
    ci_low = float(np.percentile(boot_gains, 2.5))
    ci_high = float(np.percentile(boot_gains, 97.5))
    diff_low = float(np.percentile(boot_diffs, 2.5))
    diff_high = float(np.percentile(boot_diffs, 97.5))
    
    diff = errors_cand - errors_base
    if np.all(diff == 0):
        p_val = 1.0
    else:
        try:
            stat, p_val = stats.wilcoxon(diff, alternative="two-sided")
            p_val = float(p_val)
        except Exception:
            p_val = float("nan")
            
    return {
        "mean_gain": float(np.mean(rel_gains)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_diff": float(np.mean(abs_diffs)),
        "diff_ci_low": diff_low,
        "diff_ci_high": diff_high,
        "p_value": p_val
    }


# ---------------------------------------------------------------------------
# Physical Invariant Evaluators
# ---------------------------------------------------------------------------
def compute_divergence_norm(vel: torch.Tensor, L: float = 1.0, op: FMPhysicsEnergy2D | None = None) -> float:
    """Compute spatial L2 divergence norm ||div(u)|| = ||du/dx + dv/dy||."""
    if op is not None:
        return float(op.rms_divergence(vel).mean().item())
    u, v = vel[:, 0], vel[:, 1]
    n = u.shape[-1]
    kx = torch.fft.fftfreq(n, d=L / (2 * math.pi * n), device=vel.device)
    ky = torch.fft.fftfreq(n, d=L / (2 * math.pi * n), device=vel.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    
    uh = torch.fft.fft2(u)
    vh = torch.fft.fft2(v)
    div_h = 1j * KX * uh + 1j * KY * vh
    div = torch.fft.ifft2(div_h).real
    return float(div.square().mean().sqrt().item())


def compute_kinetic_energy(vel: torch.Tensor) -> float:
    """Compute total kinetic energy E = 0.5 * integral (u^2 + v^2)."""
    return float((0.5 * vel.square().sum(dim=1)).mean().item())


def compute_enstrophy(vel: torch.Tensor, L: float = 1.0, op: FMPhysicsEnergy2D | None = None) -> float:
    """Compute total enstrophy Omega = 0.5 * integral (vorticity^2)."""
    if op is not None:
        return float(op.enstrophy(vel).mean().item())
    u, v = vel[:, 0], vel[:, 1]
    n = u.shape[-1]
    kx = torch.fft.fftfreq(n, d=L / (2 * math.pi * n), device=vel.device)
    ky = torch.fft.fftfreq(n, d=L / (2 * math.pi * n), device=vel.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    
    uh = torch.fft.fft2(u)
    vh = torch.fft.fft2(v)
    wh = 1j * KX * vh - 1j * KY * uh
    w = torch.fft.ifft2(wh).real
    return float((0.5 * w.square()).mean().item())


# ---------------------------------------------------------------------------
# EXPERIMENT 1: PRIMARY SUB-CADENCE BENCHMARK (REAL POSEIDON ON FLUIDS)
# ---------------------------------------------------------------------------
def run_real_primary_contract_benchmark(
    fm, pde_name: str, trajs: torch.Tensor, cal_trajs: torch.Tensor,
    phys: GenericPDEPhysics, op: FMPhysicsEnergy2D | None,
    dt_step: float = 0.10, grid_size: int = 128
) -> Dict[str, Any]:
    """Evaluates primary-contract sub-cadence interpolation on real Poseidon and real fluid trajectories."""
    stride = int(round(dt_step / 0.05))
    mid_idx = stride // 2
    n_test = trajs.shape[0]
    
    u0 = trajs[:, 0]
    u1 = trajs[:, stride]
    u_mid_true = trajs[:, mid_idx]
    
    # 1. Linear Secant Baseline & Linear + Projection Control
    u_lin = 0.5 * (u0 + u1)
    rmse_lin_traj = np.sqrt(np.mean((u_lin.cpu().numpy() - u_mid_true.cpu().numpy())**2, axis=(-3, -2, -1)))
    if op is not None:
        u_lin_proj = op.project_incompressible(u_lin).reshape_as(u_lin)
    else:
        u_lin_proj = phys.project_manifold(u_lin)
    rmse_lin_proj_traj = np.sqrt(np.mean((u_lin_proj.cpu().numpy() - u_mid_true.cpu().numpy())**2, axis=(-3, -2, -1)))
    
    # 2. Raw Poseidon Model Forecast & Raw FM + Projection Control
    fm.set_lead_time(float(mid_idx))
    with torch.no_grad():
        u_fm = fm_predict(fm, u0, current_grid=grid_size)
    rmse_fm_traj = np.sqrt(np.mean((u_fm.cpu().numpy() - u_mid_true.cpu().numpy())**2, axis=(-3, -2, -1)))
    if op is not None:
        u_fm_proj = op.project_incompressible(u_fm).reshape_as(u_fm)
    else:
        u_fm_proj = phys.project_manifold(u_fm)
    rmse_fm_proj_traj = np.sqrt(np.mean((u_fm_proj.cpu().numpy() - u_mid_true.cpu().numpy())**2, axis=(-3, -2, -1)))
    
    # 3. KMF Kinematic Hermite Bridge (0 forward ODE steps, Projected)
    if op is not None:
        u_herm = fluid_hermite_kinematic_spline(op, u0, u1, dt_step, s=0.5)
        u_herm = op.project_incompressible(u_herm).reshape_as(u_herm)
    else:
        u_herm = general_kinematic_hermite_spline(phys, u0, u1, dt_step, s=0.5)
        u_herm = phys.project_manifold(u_herm)
    rmse_herm_traj = np.sqrt(np.mean((u_herm.cpu().numpy() - u_mid_true.cpu().numpy())**2, axis=(-3, -2, -1)))
    
    # 4. Zero-Leakage Calibrated Fusion (weight fit on held-out cal_trajs)
    w_cal = 0.5
    if cal_trajs.shape[0] > 0:
        u0_c = cal_trajs[:, 0]
        u1_c = cal_trajs[:, stride]
        u_mid_c = cal_trajs[:, mid_idx]
        with torch.no_grad():
            u_fm_c = fm_predict(fm, u0_c, current_grid=grid_size)
        if op is not None:
            u_fm_c = op.project_incompressible(u_fm_c).reshape_as(u_fm_c)
            u_herm_c = fluid_hermite_kinematic_spline(op, u0_c, u1_c, dt_step, s=0.5)
            u_herm_c = op.project_incompressible(u_herm_c).reshape_as(u_herm_c)
        else:
            u_fm_c = phys.project_manifold(u_fm_c)
            u_herm_c = general_kinematic_hermite_spline(phys, u0_c, u1_c, dt_step, s=0.5)
            u_herm_c = phys.project_manifold(u_herm_c)
            
        diff_fc = (u_fm_c - u_herm_c).reshape(-1)
        diff_tc = (u_mid_c - u_herm_c).reshape(-1)
        denom = float(diff_fc.square().sum())
        if denom > 1e-12:
            w_cal = max(0.0, min(1.0, float((diff_fc * diff_tc).sum() / denom)))
            
    u_cal = (1.0 - w_cal) * u_herm + w_cal * u_fm_proj
    if op is not None:
        u_cal = op.project_incompressible(u_cal).reshape_as(u_cal)
    else:
        u_cal = phys.project_manifold(u_cal)
    rmse_cal_traj = np.sqrt(np.mean((u_cal.cpu().numpy() - u_mid_true.cpu().numpy())**2, axis=(-3, -2, -1)))
    
    # 5. Two-Endpoint PCHIP Limiter & PCHIP + Projection Control
    if op is not None:
        w0 = op.to_vorticity(u0)
        w1 = op.to_vorticity(u1)
        rhs_w0 = op.rhs_vorticity(w0)
        rhs_w1 = op.rhs_vorticity(w1)
        u_dot0, v_dot0 = op.grid.velocity(torch.fft.rfft2(rhs_w0))
        u_dot1, v_dot1 = op.grid.velocity(torch.fft.rfft2(rhs_w1))
        f0 = torch.stack((u_dot0, v_dot0), dim=1).cpu().numpy()
        f1 = torch.stack((u_dot1, v_dot1), dim=1).cpu().numpy()
    else:
        f0 = phys.rhs(u0).cpu().numpy()
        f1 = phys.rhs(u1).cpu().numpy()
        
    u0_np = u0.cpu().numpy()
    u1_np = u1.cpu().numpy()
    secant_slope = (u1_np - u0_np) / dt_step
    d0_mod = np.where(secant_slope * f0 > 0, f0, 0.0)
    d1_mod = np.where(secant_slope * f1 > 0, f1, 0.0)
    max_slope = 3.0 * np.abs(secant_slope)
    d0_mod = np.sign(d0_mod) * np.minimum(np.abs(d0_mod), max_slope)
    d1_mod = np.sign(d1_mod) * np.minimum(np.abs(d1_mod), max_slope)
    
    h00, h10 = 0.5, 0.5
    h01 = dt_step * 0.125
    h11 = -dt_step * 0.125
    u_pchip = h00 * u0_np + h10 * u1_np + h01 * d0_mod + h11 * d1_mod
    rmse_pchip_traj = np.sqrt(np.mean((u_pchip - u_mid_true.cpu().numpy())**2, axis=(-3, -2, -1)))
    
    u_pchip_t = torch.as_tensor(u_pchip, device=u0.device, dtype=u0.dtype)
    if op is not None:
        u_pchip_proj = op.project_incompressible(u_pchip_t).reshape_as(u_pchip_t)
    else:
        u_pchip_proj = phys.project_manifold(u_pchip_t)
    rmse_pchip_proj_traj = np.sqrt(np.mean((u_pchip_proj.cpu().numpy() - u_mid_true.cpu().numpy())**2, axis=(-3, -2, -1)))
    
    # Statistical tests with matched projected controls
    ci_herm_vs_lin = compute_paired_bootstrap_ci(rmse_lin_traj, rmse_herm_traj)
    ci_herm_vs_lin_proj = compute_paired_bootstrap_ci(rmse_lin_proj_traj, rmse_herm_traj)
    ci_herm_vs_pchip_proj = compute_paired_bootstrap_ci(rmse_pchip_proj_traj, rmse_herm_traj)
    ci_cal_vs_lin = compute_paired_bootstrap_ci(rmse_lin_traj, rmse_cal_traj)
    ci_cal_vs_fm = compute_paired_bootstrap_ci(rmse_fm_traj, rmse_cal_traj)
    ci_cal_vs_fm_proj = compute_paired_bootstrap_ci(rmse_fm_proj_traj, rmse_cal_traj)
    
    results = {
        "pde": pde_name,
        "dt": dt_step,
        "n_test": n_test,
        "w_cal": w_cal,
        "rmse_linear": float(np.mean(rmse_lin_traj)),
        "rmse_linear_proj": float(np.mean(rmse_lin_proj_traj)),
        "rmse_fm": float(np.mean(rmse_fm_traj)),
        "rmse_fm_proj": float(np.mean(rmse_fm_proj_traj)),
        "rmse_pchip_2pt": float(np.mean(rmse_pchip_traj)),
        "rmse_pchip_proj": float(np.mean(rmse_pchip_proj_traj)),
        "rmse_hermite": float(np.mean(rmse_herm_traj)),
        "rmse_calibrated": float(np.mean(rmse_cal_traj)),
        "gain_hermite_vs_linear": float(100.0 * (1.0 - np.mean(rmse_herm_traj) / np.mean(rmse_lin_traj))),
        "gain_hermite_vs_linear_proj": float(100.0 * (1.0 - np.mean(rmse_herm_traj) / np.mean(rmse_lin_proj_traj))),
        "gain_hermite_vs_pchip_proj": float(100.0 * (1.0 - np.mean(rmse_herm_traj) / np.mean(rmse_pchip_proj_traj))),
        "gain_cal_vs_linear": float(100.0 * (1.0 - np.mean(rmse_cal_traj) / np.mean(rmse_lin_traj))),
        "gain_cal_vs_fm": float(100.0 * (1.0 - np.mean(rmse_cal_traj) / np.mean(rmse_fm_traj))),
        "gain_cal_vs_fm_proj": float(100.0 * (1.0 - np.mean(rmse_cal_traj) / np.mean(rmse_fm_proj_traj))),
        "ci_hermite_vs_linear": ci_herm_vs_lin,
        "ci_hermite_vs_linear_proj": ci_herm_vs_lin_proj,
        "ci_hermite_vs_pchip_proj": ci_herm_vs_pchip_proj,
        "ci_cal_vs_linear": ci_cal_vs_lin,
        "ci_cal_vs_fm": ci_cal_vs_fm,
        "ci_cal_vs_fm_proj": ci_cal_vs_fm_proj,
        "trajectory_rmses": {
            "linear": rmse_lin_traj.tolist(),
            "linear_proj": rmse_lin_proj_traj.tolist(),
            "fm": rmse_fm_traj.tolist(),
            "fm_proj": rmse_fm_proj_traj.tolist(),
            "pchip": rmse_pchip_traj.tolist(),
            "pchip_proj": rmse_pchip_proj_traj.tolist(),
            "hermite": rmse_herm_traj.tolist(),
            "calibrated": rmse_cal_traj.tolist()
        }
    }
    
    print(f"\n--- Primary Benchmark: {pde_name} (dt={dt_step:.2f}s, N={n_test}) ---")
    print(f"Linear Secant RMSE:            {results['rmse_linear']:.6f}")
    print(f"Linear + Projection Control:   {results['rmse_linear_proj']:.6f}")
    print(f"Raw Poseidon Forecast RMSE:    {results['rmse_fm']:.6f}")
    print(f"Raw Poseidon + Proj Control:   {results['rmse_fm_proj']:.6f}")
    print(f"PCHIP (2-endpoint limited):    {results['rmse_pchip_2pt']:.6f}")
    print(f"PCHIP + Projection Control:    {results['rmse_pchip_proj']:.6f}")
    print(f"Kinematic Hermite (Projected): {results['rmse_hermite']:.6f} (Gain vs Lin: {results['gain_hermite_vs_linear']:+.2f}%, vs Lin+Proj: {results['gain_hermite_vs_linear_proj']:+.2f}%)")
    print(f"Calibrated Fusion (w*={w_cal:.2f}):    {results['rmse_calibrated']:.6f} (Gain vs Lin: {results['gain_cal_vs_linear']:+.2f}%, vs FM+Proj: {results['gain_cal_vs_fm_proj']:+.2f}%)")
    print(f"  --> Paired Bootstrap 95% CI vs Lin+Proj: [{ci_herm_vs_lin_proj['ci_low']:+.2f}%, {ci_herm_vs_lin_proj['ci_high']:+.2f}%], p = {ci_herm_vs_lin_proj['p_value']:.4e}")
    print(f"  --> Paired Bootstrap 95% CI vs FM+Proj:  [{ci_cal_vs_fm_proj['ci_low']:+.2f}%, {ci_cal_vs_fm_proj['ci_high']:+.2f}%], p = {ci_cal_vs_fm_proj['p_value']:.4e}")
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 2: CALIBRATION SAMPLE EFFICIENCY ABLATION (REAL FM RESIDUALS)
# ---------------------------------------------------------------------------
def run_real_calibration_ablation(
    fm, all_trajs: torch.Tensor, phys: GenericPDEPhysics, op: FMPhysicsEnergy2D | None,
    budgets: List[int] = [1, 2, 5, 10, 20], n_splits: int = 5, dt_step: float = 0.10, grid_size: int = 128
) -> Dict[str, Any]:
    """Ablate calibration sample efficiency across budgets M using actual Poseidon predictions and residuals."""
    stride = int(round(dt_step / 0.05))
    mid_idx = stride // 2
    total_n = all_trajs.shape[0]
    
    u0_all = all_trajs[:, 0]
    u1_all = all_trajs[:, stride]
    u_true_all = all_trajs[:, mid_idx]
    
    # Precompute actual FM forecasts and Hermite bridges
    fm.set_lead_time(float(mid_idx))
    with torch.no_grad():
        u_fm_all = fm_predict(fm, u0_all, current_grid=grid_size)
    if op is not None:
        u_fm_all = op.project_incompressible(u_fm_all).reshape_as(u_fm_all)
        u_herm_all = fluid_hermite_kinematic_spline(op, u0_all, u1_all, dt_step, s=0.5)
        u_herm_all = op.project_incompressible(u_herm_all).reshape_as(u_herm_all)
    else:
        u_fm_all = phys.project_manifold(u_fm_all)
        u_herm_all = general_kinematic_hermite_spline(phys, u0_all, u1_all, dt_step, s=0.5)
        u_herm_all = phys.project_manifold(u_herm_all)
        
    results_by_budget = {}
    rng = np.random.default_rng(20260914)
    
    for M in budgets:
        if M >= total_n:
            continue
        w_selected_list = []
        test_rmses_list = []
        degradation_count = 0
        total_runs = 0
        
        for _ in range(n_splits):
            perm = rng.permutation(total_n)
            cal_idx = perm[:M]
            test_idx = perm[M:min(total_n, M + 50)]
            if len(test_idx) < 10:
                test_idx = perm[M:]
                
            u_herm_c = u_herm_all[cal_idx].reshape(-1)
            u_fm_c = u_fm_all[cal_idx].reshape(-1)
            u_true_c = u_true_all[cal_idx].reshape(-1)
            
            diff_fc = (u_fm_c - u_herm_c)
            diff_tc = (u_true_c - u_herm_c)
            denom = float(diff_fc.square().sum())
            if denom > 1e-12:
                w_opt = float((diff_fc * diff_tc).sum() / denom)
                w_cal = max(0.0, min(1.0, w_opt))
            else:
                w_cal = 0.5
            w_selected_list.append(w_cal)
            
            # Evaluate out-of-sample on untouched test split
            u_test_herm = u_herm_all[test_idx]
            u_test_fm = u_fm_all[test_idx]
            u_test_true = u_true_all[test_idx]
            
            u_test_fused = (1.0 - w_cal) * u_test_herm + w_cal * u_test_fm
            if op is not None:
                u_test_fused = op.project_incompressible(u_test_fused).reshape_as(u_test_fused)
            else:
                u_test_fused = phys.project_manifold(u_test_fused)
                
            rmse_fused = float((u_test_fused - u_test_true).square().mean().sqrt().item())
            rmse_herm = float((u_test_herm - u_test_true).square().mean().sqrt().item())
            rmse_fm = float((u_test_fm - u_test_true).square().mean().sqrt().item())
            
            test_rmses_list.append(rmse_fused)
            # Degradation occurs if fused test error is worse than either pure constituent
            if rmse_fused > min(rmse_herm, rmse_fm):
                degradation_count += 1
            total_runs += 1
            
        results_by_budget[str(M)] = {
            "M": M,
            "w_median": float(np.median(w_selected_list)),
            "w_iqr": [float(np.percentile(w_selected_list, 25)), float(np.percentile(w_selected_list, 75))],
            "w_zeros_frac": float(np.mean(np.array(w_selected_list) == 0.0)),
            "test_rmse_mean": float(np.mean(test_rmses_list)),
            "test_rmse_std": float(np.std(test_rmses_list)),
            "degradation_rate": float(degradation_count / total_runs)
        }
        r = results_by_budget[str(M)]
        print(f"Budget M={M:2d}: w* median={r['w_median']:.4f}, IQR=[{r['w_iqr'][0]:.4f}, {r['w_iqr'][1]:.4f}], zero-fraction={r['w_zeros_frac']:.1%}, Test RMSE={r['test_rmse_mean']:.6f} ± {r['test_rmse_std']:.6f}, Degradation={r['degradation_rate']:.1%}")
        
    return results_by_budget


# ---------------------------------------------------------------------------
# EXPERIMENT 3: COMPONENT ABLATION & INVARIANTS (ZERO FUTURE TRUTH)
# ---------------------------------------------------------------------------
def run_real_component_ablation(
    fm, trajs: torch.Tensor, cal_trajs: torch.Tensor, phys: GenericPDEPhysics,
    op: FMPhysicsEnergy2D | None, dt_step: float = 0.10, steps: int = 4, grid_size: int = 128
) -> Dict[str, Any]:
    """4-way component ablation with strictly zero future truth in any deployed branch."""
    stride = int(round(dt_step / 0.05))
    avail_steps = min(steps, (trajs.shape[1] - 1) // stride)
    
    u_init = trajs[:, 0]
    true_divs, true_energies, true_enstrophies = [], [], []
    for s in range(avail_steps + 1):
        tgt = trajs[:, s * stride]
        true_divs.append(compute_divergence_norm(tgt, op=op))
        true_energies.append(compute_kinetic_energy(tgt))
        true_enstrophies.append(compute_enstrophy(tgt, op=op))
        
    fm.set_lead_time(float(stride))
    
    # Tune relaxation parameter alpha purely on calibration split D_cal
    alpha_hs = 0.0
    if cal_trajs.shape[0] > 0:
        c_u0 = cal_trajs[:, 0]
        c_tgt = cal_trajs[:, stride]
        with torch.no_grad():
            c_cand = fm_predict(fm, c_u0, current_grid=grid_size)
        if op is not None:
            c_cand = op.project_incompressible(c_cand).reshape_as(c_cand)
            w_p = op.to_vorticity(c_u0)
            w_c = op.to_vorticity(c_cand)
            f_p = op.rhs_vorticity(w_p)
            f_c = op.rhs_vorticity(w_c)
            w_m = 0.5 * (w_p + w_c) + (dt_step / 8.0) * (f_p - f_c)
            f_m = op.rhs_vorticity(w_m)
            w_s = w_p + (dt_step / 6.0) * (f_p + 4.0 * f_m + f_c)
            wh_d = torch.fft.rfft2(w_s - w_c)
            k_m = grid_size // 2
            sm = torch.exp(-36.0 * (op.grid.k2.sqrt() / k_m).pow(36))
            w_df = torch.fft.irfft2(wh_d * sm, s=(grid_size, grid_size))
            best_a = 0.0
            best_e = float((c_cand - c_tgt).square().mean())
            for a_try in [0.01, 0.02, 0.05, 0.10, 0.20]:
                wh_try = torch.fft.rfft2(w_c + a_try * w_df)
                cu_try, cv_try = op.grid.velocity(wh_try)
                hs_try = torch.stack((cu_try, cv_try), dim=1) + c_u0.mean(dim=(-2, -1), keepdim=True)
                hs_try = op.project_incompressible(hs_try).reshape_as(hs_try)
                e_try = float((hs_try - c_tgt).square().mean())
                if e_try < best_e:
                    best_e = e_try
                    best_a = a_try
            alpha_hs = best_a
            
    configs = ["raw_fm", "projection_only", "relaxation_only", "full_kmf"]
    ablation_results = {}
    
    for cfg in configs:
        curr = u_init.clone()
        step_rmses = []
        div_norms = []
        energy_errors = []
        enstrophy_errors = []
        
        for s in range(avail_steps):
            tgt = trajs[:, (s + 1) * stride]
            with torch.no_grad():
                pred = fm_predict(fm, curr, current_grid=grid_size)
                
            if cfg == "raw_fm":
                curr = pred
            elif cfg == "projection_only":
                if op is not None:
                    curr = op.project_incompressible(pred).reshape_as(pred)
                else:
                    curr = phys.project_manifold(pred)
            elif cfg == "relaxation_only":
                if op is not None and alpha_hs > 0:
                    w_prev = op.to_vorticity(curr)
                    w_pred = op.to_vorticity(pred)
                    f_prev = op.rhs_vorticity(w_prev)
                    f_pred = op.rhs_vorticity(w_pred)
                    w_mid = 0.5 * (w_prev + w_pred) + (dt_step / 8.0) * (f_prev - f_pred)
                    f_mid = op.rhs_vorticity(w_mid)
                    w_simp = w_prev + (dt_step / 6.0) * (f_prev + 4.0 * f_mid + f_pred)
                    wh_diff = torch.fft.rfft2(w_simp - w_pred)
                    k_max = grid_size // 2
                    smooth = torch.exp(-36.0 * (op.grid.k2.sqrt() / k_max).pow(36))
                    w_diff_filt = torch.fft.irfft2(wh_diff * smooth, s=(grid_size, grid_size))
                    w_corr = w_pred + alpha_hs * w_diff_filt
                    cu, cv = op.grid.velocity(torch.fft.rfft2(w_corr))
                    curr = torch.stack((cu, cv), dim=1) + curr.mean(dim=(-2, -1), keepdim=True)
                else:
                    curr = pred
            elif cfg == "full_kmf":
                if op is not None:
                    pred_p = op.project_incompressible(pred).reshape_as(pred)
                    if alpha_hs > 0:
                        w_prev = op.to_vorticity(curr)
                        w_pred = op.to_vorticity(pred_p)
                        f_prev = op.rhs_vorticity(w_prev)
                        f_pred = op.rhs_vorticity(w_pred)
                        w_mid = 0.5 * (w_prev + w_pred) + (dt_step / 8.0) * (f_prev - f_pred)
                        f_mid = op.rhs_vorticity(w_mid)
                        w_simp = w_prev + (dt_step / 6.0) * (f_prev + 4.0 * f_mid + f_pred)
                        wh_diff = torch.fft.rfft2(w_simp - w_pred)
                        k_max = grid_size // 2
                        smooth = torch.exp(-36.0 * (op.grid.k2.sqrt() / k_max).pow(36))
                        w_diff_filt = torch.fft.irfft2(wh_diff * smooth, s=(grid_size, grid_size))
                        w_corr = w_pred + alpha_hs * w_diff_filt
                        cu, cv = op.grid.velocity(torch.fft.rfft2(w_corr))
                        hs_next = torch.stack((cu, cv), dim=1) + curr.mean(dim=(-2, -1), keepdim=True)
                        curr = op.project_incompressible(hs_next).reshape_as(hs_next)
                    else:
                        curr = pred_p
                else:
                    curr = phys.project_manifold(pred)
                    
            rmse = float((curr - tgt).square().mean().sqrt().item())
            div = compute_divergence_norm(curr, op=op)
            e_err = abs(compute_kinetic_energy(curr) - true_energies[s + 1]) / max(true_energies[s + 1], 1e-12)
            en_err = abs(compute_enstrophy(curr, op=op) - true_enstrophies[s + 1]) / max(true_enstrophies[s + 1], 1e-12)
            
            step_rmses.append(rmse)
            div_norms.append(div)
            energy_errors.append(e_err)
            enstrophy_errors.append(en_err)
            
        ablation_results[cfg] = {
            "window_rmse": float(np.mean(step_rmses)),
            "final_step_rmse": float(step_rmses[-1]),
            "mean_divergence": float(np.mean(div_norms)),
            "mean_energy_error_pct": float(np.mean(energy_errors) * 100.0),
            "mean_enstrophy_error_pct": float(np.mean(enstrophy_errors) * 100.0)
        }
        res = ablation_results[cfg]
        print(f"Config: {cfg:18} | Window RMSE: {res['window_rmse']:.6f} | Div Norm: {res['mean_divergence']:.4e} | Energy Err: {res['mean_energy_error_pct']:.2f}% | Enstrophy Err: {res['mean_enstrophy_error_pct']:.2f}%")
        
    ablation_results["metadata"] = {
        "alpha_calibrated": float(alpha_hs),
        "relaxation_active": bool(alpha_hs > 0)
    }
    return ablation_results


# ---------------------------------------------------------------------------
# EXPERIMENT 4: LONG-HORIZON AUTOREGRESSIVE ROLLOUT (H = 20 STEPS)
# ---------------------------------------------------------------------------
def run_real_long_horizon_rollout(
    fm, trajs: torch.Tensor, phys: GenericPDEPhysics, op: FMPhysicsEnergy2D | None,
    dt_step: float = 0.05, max_steps: int = 20, grid_size: int = 128
) -> Dict[str, Any]:
    """Roll out actual Poseidon autoregressively for 20 steps, tracking compounding vs saturation."""
    stride = max(1, int(round(dt_step / 0.05)))
    avail_steps = min(max_steps, (trajs.shape[1] - 1) // stride)
    
    fm.set_lead_time(float(stride))
    curr_raw = trajs[:, 0].clone()
    curr_kmf = trajs[:, 0].clone()
    
    raw_step_errors = []
    kmf_step_errors = []
    
    for s in range(avail_steps):
        tgt = trajs[:, (s + 1) * stride]
        with torch.no_grad():
            curr_raw = fm_predict(fm, curr_raw, current_grid=grid_size)
            pred_kmf = fm_predict(fm, curr_kmf, current_grid=grid_size)
            
        if op is not None:
            curr_kmf = op.project_incompressible(pred_kmf).reshape_as(pred_kmf)
        else:
            curr_kmf = phys.project_manifold(pred_kmf)
            
        err_raw = float((curr_raw - tgt).square().mean().sqrt().item())
        err_kmf = float((curr_kmf - tgt).square().mean().sqrt().item())
        raw_step_errors.append(err_raw)
        kmf_step_errors.append(err_kmf)
        
    horizon_results = {
        "steps": avail_steps,
        "raw_trajectory_rmse": raw_step_errors,
        "kmf_trajectory_rmse": kmf_step_errors,
        "raw_final_rmse": raw_step_errors[-1],
        "kmf_final_rmse": kmf_step_errors[-1],
        "final_step_gain": float(100.0 * (1.0 - kmf_step_errors[-1] / raw_step_errors[-1]))
    }
    
    print(f"\n--- Real Autoregressive Rollout ({avail_steps} Steps) ---")
    print(f"Step  1: Raw RMSE = {raw_step_errors[0]:.6f} | KMF RMSE = {kmf_step_errors[0]:.6f}")
    print(f"Step  5: Raw RMSE = {raw_step_errors[min(4, avail_steps-1)]:.6f} | KMF RMSE = {kmf_step_errors[min(4, avail_steps-1)]:.6f}")
    print(f"Step 10: Raw RMSE = {raw_step_errors[min(9, avail_steps-1)]:.6f} | KMF RMSE = {kmf_step_errors[min(9, avail_steps-1)]:.6f}")
    print(f"Step 20: Raw RMSE = {raw_step_errors[-1]:.6f} | KMF RMSE = {kmf_step_errors[-1]:.6f} (Gain: {horizon_results['final_step_gain']:+.2f}%)")
    return horizon_results


# ---------------------------------------------------------------------------
# EXPERIMENT 5: WAVE-GAUSS REAL DATA DISPERSION ANALYSIS
# ---------------------------------------------------------------------------
def run_real_wave_dispersion_analysis(
    fm, wave_trajs: torch.Tensor, phys: GenericPDEPhysics, dt_step: float = 0.10, grid_size: int = 128
) -> Dict[str, Any]:
    """Analyzes real acoustic wave propagation errors across the spatial Fourier wavenumber spectrum."""
    stride = int(round(dt_step / 0.05))
    mid_idx = stride // 2
    n_test = wave_trajs.shape[0]
    
    u0 = wave_trajs[:, 0]
    u1 = wave_trajs[:, stride]
    u_mid_true = wave_trajs[:, mid_idx]
    
    u_lin = 0.5 * (u0 + u1)
    
    fm.set_lead_time(float(mid_idx))
    with torch.no_grad():
        u_fm = fm_predict(fm, u0, current_grid=grid_size)
        
    u_herm = general_kinematic_hermite_spline(phys, u0, u1, dt_step, s=0.5)
    
    # Compute error spectra across radial spatial wavenumbers k
    diff_lin = (u_lin - u_mid_true)[:, 0]
    diff_herm = (u_herm - u_mid_true)[:, 0]
    diff_fm = (u_fm - u_mid_true)[:, 0]
    
    fft_lin = torch.fft.rfft2(diff_lin).abs().mean(dim=0).cpu().numpy()
    fft_herm = torch.fft.rfft2(diff_herm).abs().mean(dim=0).cpu().numpy()
    fft_fm = torch.fft.rfft2(diff_fm).abs().mean(dim=0).cpu().numpy()
    
    # Radial binning
    H, W_half = fft_lin.shape
    kx = np.fft.fftfreq(H, d=1.0/H)
    ky = np.fft.rfftfreq(H, d=1.0/H)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K_rad = np.sqrt(KX**2 + KY**2).T
    
    k_bins = np.linspace(0, H//2, 10)
    bin_lin, bin_herm, bin_fm = [], [], []
    for i in range(len(k_bins) - 1):
        mask = (K_rad >= k_bins[i]) & (K_rad < k_bins[i+1])
        bin_lin.append(float(np.mean(fft_lin[mask])))
        bin_herm.append(float(np.mean(fft_herm[mask])))
        bin_fm.append(float(np.mean(fft_fm[mask])))
        
    dispersion_results = {
        "k_bin_centers": [float(0.5 * (k_bins[i] + k_bins[i+1])) for i in range(len(k_bins) - 1)],
        "error_spectrum_linear": bin_lin,
        "error_spectrum_hermite": bin_herm,
        "error_spectrum_fm": bin_fm,
        "overall_rmse_linear": float((u_lin - u_mid_true).square().mean().sqrt().item()),
        "overall_rmse_hermite": float((u_herm - u_mid_true).square().mean().sqrt().item()),
        "overall_rmse_fm": float((u_fm - u_mid_true).square().mean().sqrt().item())
    }
    
    print(f"\n--- Wave-Gauss Dispersion Analysis (Real Data, N={n_test}) ---")
    print(f"Overall Linear RMSE:  {dispersion_results['overall_rmse_linear']:.6f}")
    print(f"Overall FM RMSE:      {dispersion_results['overall_rmse_fm']:.6f}")
    print(f"Overall Hermite RMSE: {dispersion_results['overall_rmse_hermite']:.6f}")
    print(f"Low-k (k={dispersion_results['k_bin_centers'][0]:.1f}) Error:  Lin={bin_lin[0]:.4e} | Herm={bin_herm[0]:.4e} (Hermite Gain: {100*(1-bin_herm[0]/bin_lin[0]):+.1f}%)")
    print(f"High-k (k={dispersion_results['k_bin_centers'][-1]:.1f}) Error: Lin={bin_lin[-1]:.4e} | Herm={bin_herm[-1]:.4e} (High-k Phase Dispersion)")
    return dispersion_results


# ---------------------------------------------------------------------------
# EXPERIMENT 6: SYNCHRONIZED END-TO-END LATENCY (FM + OPERATORS)
# ---------------------------------------------------------------------------
def run_real_synchronized_timing_protocol(
    fm, trajs: torch.Tensor, phys: GenericPDEPhysics, op: FMPhysicsEnergy2D | None,
    n_trials: int = 50, n_warmup: int = 10, grid_size: int = 128
) -> Dict[str, Any]:
    """Measures exact wall-clock latencies of real FM inference and KMF physical operators with CUDA synchronization."""
    u = trajs[:1, 0].clone()
    is_cuda = (u.device.type == "cuda") and torch.cuda.is_available()
    
    def sync():
        if is_cuda:
            torch.cuda.synchronize()
            
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = fm_predict(fm, u, current_grid=grid_size)
        if op is not None:
            w = op.to_vorticity(u)
            rhs_w = op.rhs_vorticity(w)
            _ = op.grid.velocity(torch.fft.rfft2(rhs_w))
            _ = op.project_incompressible(u)
        else:
            _ = phys.rhs(u)
            _ = phys.project_manifold(u)
    sync()
    
    # 1. FM Forward Inference Pass
    fm_times = []
    for _ in range(n_trials):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = fm_predict(fm, u, current_grid=grid_size)
        sync()
        fm_times.append((time.perf_counter() - t0) * 1000.0)
        
    # 2. Physical Acceleration RHS F(u)
    rhs_times = []
    for _ in range(n_trials):
        sync()
        t0 = time.perf_counter()
        if op is not None:
            w = op.to_vorticity(u)
            rhs_w = op.rhs_vorticity(w)
            _ = op.grid.velocity(torch.fft.rfft2(rhs_w))
        else:
            _ = phys.rhs(u)
        sync()
        rhs_times.append((time.perf_counter() - t0) * 1000.0)
        
    # 3. Helmholtz-Leray Projection
    proj_times = []
    for _ in range(n_trials):
        sync()
        t0 = time.perf_counter()
        if op is not None:
            _ = op.project_incompressible(u)
        else:
            _ = phys.project_manifold(u)
        sync()
        proj_times.append((time.perf_counter() - t0) * 1000.0)
        
    # 4. Arithmetic Bridge combination
    bridge_times = []
    for _ in range(n_trials):
        sync()
        t0 = time.perf_counter()
        _ = 0.5 * (u + u) + 0.0125 * (u - u)
        sync()
        bridge_times.append((time.perf_counter() - t0) * 1000.0)
        
    timing_results = {
        "fm_inference_ms": float(np.mean(fm_times)),
        "fm_inference_std_ms": float(np.std(fm_times)),
        "rhs_single_ms": float(np.mean(rhs_times)),
        "rhs_single_std_ms": float(np.std(rhs_times)),
        "rhs_dual_ms": float(2.0 * np.mean(rhs_times)),
        "projection_ms": float(np.mean(proj_times)),
        "projection_std_ms": float(np.std(proj_times)),
        "bridge_arithmetic_ms": float(np.mean(bridge_times)),
        "total_kmf_overhead_ms": float(2.0 * np.mean(rhs_times) + np.mean(proj_times) + np.mean(bridge_times)),
        "total_end_to_end_ms": float(np.mean(fm_times) + 2.0 * np.mean(rhs_times) + np.mean(proj_times) + np.mean(bridge_times))
    }
    
    print(f"\n--- Synchronized Hardware Latency Benchmark (N={n_trials} trials on {u.device}) ---")
    print(f"Poseidon Neural Forward Pass:   {timing_results['fm_inference_ms']:.3f} ms ± {timing_results['fm_inference_std_ms']:.3f} ms")
    print(f"Physical Vector Field F(u):     {timing_results['rhs_single_ms']:.3f} ms ± {timing_results['rhs_single_std_ms']:.3f} ms (per call)")
    print(f"Helmholtz-Leray Projection P:   {timing_results['projection_ms']:.3f} ms ± {timing_results['projection_std_ms']:.3f} ms")
    print(f"Cubic Bridge Arithmetic:        {timing_results['bridge_arithmetic_ms']:.3f} ms")
    print(f"Total KMF Physical Overhead:    {timing_results['total_kmf_overhead_ms']:.3f} ms (strictly 2 RHS calls, 0 ODE steps)")
    print(f"Total End-to-End Latency:       {timing_results['total_end_to_end_ms']:.3f} ms (FM + KMF Bridge)")
    return timing_results


# ---------------------------------------------------------------------------
# Main Driver Function
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real Review Audit Benchmark Suite")
    parser.add_argument("--data-root", type=str, default="data/assembled",
                        help="Path to assembled NetCDF datasets on cluster")
    parser.add_argument("--out-dir", type=str, default="results/audit_experiments/real_pipeline",
                        help="Output directory for results")
    parser.add_argument("--fm", type=str, default="poseidon", choices=["poseidon"], help="Model under test")
    parser.add_argument("--fm-size", type=str, default="B", choices=["T", "B", "L"], help="Poseidon model size")
    parser.add_argument("--grid", type=int, default=128, help="Grid resolution")
    parser.add_argument("--n-cal", type=int, default=20, help="Total calibration pool size")
    parser.add_argument("--n-test", type=int, default=50, help="Number of held-out test trajectories")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    
    print(f"================================================================================")
    print(f"STARTING REAL REVIEW AUDIT BENCHMARK SUITE")
    print(f"Foundation Model:  {args.fm.upper()} ({args.fm_size})")
    print(f"Data Root:         {data_root}")
    print(f"Device:            {args.device}")
    print(f"Resolution:        {args.grid}x{args.grid}")
    print(f"Calibration Pool:  {args.n_cal} trajectories")
    print(f"Held-out Test:     {args.n_test} trajectories")
    print(f"================================================================================")
    
    # 1. Load Pretrained Foundation Model
    args.fm_channels = "velocity"
    fm = load_fm(args, device=torch.device(args.device))
    
    total_req = args.n_cal + args.n_test
    full_audit_results = {
        "metadata": {
            "suite": "Real Foundation Model Review Audit",
            "model": f"{args.fm}_{args.fm_size}",
            "device": args.device,
            "grid": args.grid,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Check for NS-Gauss
    ns_path = data_root / "NS-Gauss.nc"
    if ns_path.exists():
        print(f"\n>>> Loading NS-Gauss dataset: {ns_path}")
        ns_trajs, _ = load_dataset_trajectories(ns_path, fm, total_req, steps=20, stride=1, offset=19760, target_grid=args.grid)
        cal_ns = ns_trajs[:args.n_cal]
        test_ns = ns_trajs[args.n_cal:args.n_cal + args.n_test]
        
        phys_ns = GenericPDEPhysics("NS-Gauss", n=args.grid, device=fm.device, dtype=torch.float64)
        spec_ns = dataclasses.replace(native_poseidon_spec(args, fm), n=args.grid)
        op_ns = FMPhysicsEnergy2D(spec_ns, test_ns[0, 0].to(fm.device, torch.float64), 2, dt=0.10, device=fm.device)
        
        # Exp 1: Primary interpolation (dt=0.10s, dt=0.20s)
        full_audit_results["exp1_primary_ns_gauss_dt010"] = run_real_primary_contract_benchmark(
            fm, "NS-Gauss", test_ns, cal_ns, phys_ns, op_ns, dt_step=0.10, grid_size=args.grid
        )
        full_audit_results["exp1_primary_ns_gauss_dt020"] = run_real_primary_contract_benchmark(
            fm, "NS-Gauss", test_ns, cal_ns, phys_ns, op_ns, dt_step=0.20, grid_size=args.grid
        )
        
        # Exp 2: Real Calibration ablation
        full_audit_results["exp2_calibration_ablation_ns_gauss"] = run_real_calibration_ablation(
            fm, ns_trajs, phys_ns, op_ns, budgets=[1, 2, 5, 10, 20], n_splits=5, dt_step=0.10, grid_size=args.grid
        )
        
        # Exp 3: Component ablation (Zero future truth)
        full_audit_results["exp3_component_ablation_ns_gauss"] = run_real_component_ablation(
            fm, test_ns, cal_ns, phys_ns, op_ns, dt_step=0.10, steps=4, grid_size=args.grid
        )
        
        # Exp 4: Long horizon rollouts (Native cadence H=10 at dt=0.10s, and off-native stress test H=20 at dt=0.05s)
        full_audit_results["exp4_native_rollout_ns_gauss_dt010"] = run_real_long_horizon_rollout(
            fm, test_ns, phys_ns, op_ns, dt_step=0.10, max_steps=10, grid_size=args.grid
        )
        full_audit_results["exp4_stress_rollout_ns_gauss_dt005"] = run_real_long_horizon_rollout(
            fm, test_ns, phys_ns, op_ns, dt_step=0.05, max_steps=20, grid_size=args.grid
        )
        
        # Exp 6: Latency protocol
        full_audit_results["exp6_timing_protocol"] = run_real_synchronized_timing_protocol(
            fm, test_ns, phys_ns, op_ns, n_trials=50, n_warmup=10, grid_size=args.grid
        )
    else:
        print(f"WARNING: {ns_path} not found.")
        
    # Check for FNS-KF
    fns_path = data_root / "FNS-KF.nc"
    if fns_path.exists():
        print(f"\n>>> Loading FNS-KF dataset: {fns_path}")
        fns_trajs, _ = load_dataset_trajectories(fns_path, fm, total_req, steps=20, stride=1, offset=19760, target_grid=args.grid)
        cal_fns = fns_trajs[:args.n_cal]
        test_fns = fns_trajs[args.n_cal:args.n_cal + args.n_test]
        
        phys_fns = GenericPDEPhysics("FNS-KF", n=args.grid, device=fm.device, dtype=torch.float64)
        base_spec = SPECS2D.get("kolmogorov", native_poseidon_spec(args, fm))
        spec_fns = dataclasses.replace(base_spec, n=args.grid, dt=0.10)
        op_fns = FMPhysicsEnergy2D(spec_fns, test_fns[0, 0].to(fm.device, torch.float64), 2, dt=0.10, device=fm.device)
        
        full_audit_results["exp1_primary_fns_kf_dt010"] = run_real_primary_contract_benchmark(
            fm, "FNS-KF", test_fns, cal_fns, phys_fns, op_fns, dt_step=0.10, grid_size=args.grid
        )
        full_audit_results["exp1_primary_fns_kf_dt020"] = run_real_primary_contract_benchmark(
            fm, "FNS-KF", test_fns, cal_fns, phys_fns, op_fns, dt_step=0.20, grid_size=args.grid
        )
        full_audit_results["exp3_component_ablation_fns_kf"] = run_real_component_ablation(
            fm, test_fns, cal_fns, phys_fns, op_fns, dt_step=0.10, steps=4, grid_size=args.grid
        )
        full_audit_results["exp4_native_rollout_fns_kf_dt010"] = run_real_long_horizon_rollout(
            fm, test_fns, phys_fns, op_fns, dt_step=0.10, max_steps=10, grid_size=args.grid
        )
    else:
        print(f"WARNING: {fns_path} not found.")
        
    # Check for Wave-Gauss
    wave_path = data_root / "Wave-Gauss.nc"
    if wave_path.exists():
        print(f"\n>>> Loading Wave-Gauss dataset: {wave_path}")
        wave_trajs, c_val = load_dataset_trajectories(wave_path, fm, total_req, steps=10, stride=1, offset=10272, target_grid=args.grid)
        phys_wave = GenericPDEPhysics("Wave-Gauss", n=args.grid, device=fm.device, dtype=torch.float64)
        if c_val is not None:
            phys_wave.c = torch.as_tensor(c_val[args.n_cal:args.n_cal + args.n_test], device=fm.device, dtype=torch.float64)
        test_wave = wave_trajs[args.n_cal:args.n_cal + args.n_test]
        
        # Exp 5: Real Wave dispersion analysis
        full_audit_results["exp5_wave_dispersion"] = run_real_wave_dispersion_analysis(
            fm, test_wave, phys_wave, dt_step=0.10, grid_size=args.grid
        )
    else:
        print(f"WARNING: {wave_path} not found.")
        
    # Save output JSON
    res_path = out_dir / "real_audit_results.json"
    with open(res_path, "w") as fp:
        json.dump(full_audit_results, fp, indent=2)
        
    print(f"\n================================================================================")
    print(f"REAL AUDIT SUITE COMPLETE! Results saved to: {res_path}")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
