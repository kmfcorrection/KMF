#!/usr/bin/env python3
"""Does the endpoint PDE-energy gradient point toward the AZEBAN correction?

For the same predecessor state p, compare the FM endpoint x=F(p) with the
native adaptive AZEBAN endpoint x*=Phi_dt(p).  The physically comparable
quantities are both in vorticity space:

  d_omega = curl(x - x*)
  r_h = [curl(x)-curl(p)]/dt - .5 [F_NS(curl(p))+F_NS(curl(x))].

The deployed physics update is not the bare residual r_h.  It is the energy
gradient grad_x E_r.  This audit compares it with the oracle error-energy
gradient grad_x E_x, where E_r=||r_h||^2 and E_x=||x-x*||^2:

  grad_x E_r = 2 J_r(x)^T r_h(x),   grad_x E_x = 2(x-x*).

For a useful correction, these gradients must align.  The calibrated MAP
gradient is also reported; it differs only by a positive scalar R_cal^{-1}/2.
The bare vorticity comparison is retained only as a secondary operator audit.
The solver endpoint is an oracle diagnostic only; it is never deployed here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.fm_eval_common import configure_native_poseidon_cadence, fm_metadata, fm_result_key, native_poseidon_spec
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.utils import Table, print_header, save_json, set_seed


def _load(path, n_traj, steps, offset, device):
    need = 2 * int(steps) + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        a = np.asarray(h[key][offset:offset + n_traj, :need, :2], dtype=np.float32)
    return torch.as_tensor(a, device=device).reshape(n_traj, need, -1)


def _cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _trapezoid_residual(op, previous, endpoint):
    wp = op.to_vorticity(previous.reshape(1, -1))[0]
    wx = op.to_vorticity(endpoint.reshape(1, -1))[0]
    return (wx - wp) / op.dt - .5 * (op.rhs_vorticity(wp) + op.rhs_vorticity(wx))


def _velocity_from_vorticity(op, w):
    u, v = op.grid.velocity(torch.fft.rfft2(w.reshape(1, op.spec.n, op.spec.n)))
    return torch.stack((u, v), dim=1).reshape(-1)


def _one(fm, spec, previous, cfl, residual_scale):
    previous = previous.detach().to(fm.device, torch.float64).reshape(-1)
    op = FMPhysicsEnergy2D(spec, previous, 2, divergence_weight=0.0, dt=spec.dt_out,
                           device=fm.device)
    raw = fm.predict(previous.to(fm.dtype)).detach().double().reshape(-1)
    oracle = op.native_flow_state(cfl=cfl).detach().double().reshape(-1)
    candidate = raw.detach().clone().requires_grad_(True)
    residual = _trapezoid_residual(op, previous, candidate)
    # Primary comparison requested by this audit:
    # E_r=||r||^2 versus E_x=||x-x*||^2.  The calibrated deployed energy is
    # retained separately because it has the same direction but another norm.
    residual_energy = residual.square().sum()
    residual_energy_gradient = torch.autograd.grad(residual_energy, candidate)[0].detach()
    calibrated_gradient = residual_energy_gradient / (2 * residual_scale ** 2)
    residual = residual.detach()
    defect_w = op.to_vorticity((raw - oracle).reshape(1, -1))[0]
    # Curl cannot see the spatially uniform velocity mode.  Compare the
    # divergence-free velocity components reconstructed from each vorticity.
    defect_vel = _velocity_from_vorticity(op, defect_w)
    residual_vel = _velocity_from_vorticity(op, op.dt * residual)
    return raw, {
        "residual": residual,
        "defect_w": defect_w,
        "defect_vel": defect_vel,
        "residual_vel": residual_vel,
        "residual_energy_gradient": residual_energy_gradient,
        "calibrated_energy_gradient": calibrated_gradient,
        "endpoint_defect": raw - oracle,
        "oracle_error_gradient": 2 * (raw - oracle),
        "raw_oracle_rms": float((raw - oracle).square().mean().sqrt()),
        "residual_rms": float(residual.square().mean().sqrt()),
        "residual_energy_gradient_rms": float(residual_energy_gradient.square().mean().sqrt()),
        "calibrated_gradient_rms": float(calibrated_gradient.square().mean().sqrt()),
    }


def _summarize(rows):
    residual = torch.cat([r["residual"].reshape(-1) for r in rows])
    defect_w = torch.cat([r["defect_w"].reshape(-1) for r in rows])
    residual_vel = torch.cat([r["residual_vel"].reshape(-1) for r in rows])
    defect_vel = torch.cat([r["defect_vel"].reshape(-1) for r in rows])
    residual_gradient = torch.cat([r["residual_energy_gradient"].reshape(-1) for r in rows])
    calibrated_gradient = torch.cat([r["calibrated_energy_gradient"].reshape(-1) for r in rows])
    endpoint_defect = torch.cat([r["endpoint_defect"].reshape(-1) for r in rows])
    oracle_gradient = torch.cat([r["oracle_error_gradient"].reshape(-1) for r in rows])
    oracle_gradient_norm = oracle_gradient.norm().clamp_min(1e-30)
    return {
        # Primary energy-gradient comparison: d||r||^2 against d||x-x*||^2.
        "residual_oracle_energy_gradient_cosine": _cos(residual_gradient, oracle_gradient),
        "residual_oracle_energy_gradient_norm_ratio": float(residual_gradient.norm() / oracle_gradient_norm),
        # Exact deployed MAP physical-gradient direction and scale.
        "calibrated_oracle_energy_gradient_cosine": _cos(calibrated_gradient, oracle_gradient),
        "calibrated_oracle_energy_gradient_norm_ratio": float(calibrated_gradient.norm() / oracle_gradient_norm),
        "vorticity_cosine": _cos(residual, defect_w),
        "vorticity_transport_relative_error": float((residual_vel - defect_vel).norm() / defect_vel.norm().clamp_min(1e-30)),
        "lifted_velocity_cosine": _cos(residual_vel, defect_vel),
        "raw_oracle_rms": float(np.mean([r["raw_oracle_rms"] for r in rows])),
        "residual_rms": float(np.mean([r["residual_rms"] for r in rows])),
        "residual_energy_gradient_rms": float(np.mean([r["residual_energy_gradient_rms"] for r in rows])),
        "calibrated_gradient_rms": float(np.mean([r["calibrated_gradient_rms"] for r in rows])),
    }


def _spectral_shell_summary(rows, n, edges):
    """Resolve the gradient comparison in periodic Stokes/Fourier eigenspaces.

    Filtering is performed in rFFT space and transformed back before computing
    inner products, so the reported norms and cosines use the ordinary real
    velocity-space metric (including the correct Hermitian multiplicities).
    """
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("--spectral-bands must contain at least two strictly increasing edges")
    ky = torch.fft.fftfreq(n, d=1.0 / n, device=rows[0]["endpoint_defect"].device)
    kx = torch.fft.rfftfreq(n, d=1.0 / n, device=ky.device)
    radius = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    raw_grad = torch.stack([r["residual_energy_gradient"].reshape(2, n, n) for r in rows])
    phys_grad = torch.stack([r["calibrated_energy_gradient"].reshape(2, n, n) for r in rows])
    oracle_grad = torch.stack([r["oracle_error_gradient"].reshape(2, n, n) for r in rows])
    raw_hat = torch.fft.rfft2(raw_grad)
    phys_hat = torch.fft.rfft2(phys_grad)
    oracle_hat = torch.fft.rfft2(oracle_grad)
    total_oracle = oracle_grad.square().sum().clamp_min(1e-30)
    total_raw = raw_grad.square().sum().clamp_min(1e-30)
    total_phys = phys_grad.square().sum().clamp_min(1e-30)
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        # The final edge is inclusive, so an edge >= Nyquist captures all
        # remaining modes without a silent omission.
        mask = (radius >= low) & ((radius < high) if high < float(radius.max()) else (radius <= high))
        def band(xhat):
            return torch.fft.irfft2(xhat * mask, s=(n, n))
        g_raw, g_phys, g_oracle = band(raw_hat), band(phys_hat), band(oracle_hat)
        result.append({
            "k_low": float(low), "k_high": float(high),
            "modes": int(mask.sum().item()),
            "cos_dEr_dEx": _cos(g_raw, g_oracle),
            "cos_dEphys_dEx": _cos(g_phys, g_oracle),
            "oracle_energy_share": float(g_oracle.square().sum() / total_oracle),
            "dEr_energy_share": float(g_raw.square().sum() / total_raw),
            "dEphys_energy_share": float(g_phys.square().sum() / total_phys),
            "dEr_dEx_norm_ratio": float(g_raw.norm() / g_oracle.norm().clamp_min(1e-30)),
            "dEphys_dEx_norm_ratio": float(g_phys.norm() / g_oracle.norm().clamp_min(1e-30)),
        })
    return result


def _calibration_residual_scale(fine, spec, steps, device):
    """R_cal^{1/2} from true calibration endpoint trajectories only."""
    values = []
    for trajectory in fine:
        for t in range(steps):
            previous, endpoint = trajectory[2 * t], trajectory[2 * t + 2]
            op = FMPhysicsEnergy2D(spec, previous, 2, divergence_weight=0.0,
                                   dt=spec.dt_out, device=device)
            with torch.no_grad():
                values.append(_trapezoid_residual(op, previous, endpoint))
    return float(torch.cat(values).square().mean().sqrt().clamp(min=1e-30))


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=8)
    ap.add_argument("--calibration-offset", type=int, default=0)
    ap.add_argument("--n-traj", type=int, default=16)
    ap.add_argument("--split-offset", type=int, default=0)
    ap.add_argument("--flow-cfl", type=float, default=.5)
    ap.add_argument("--spectral-bands", type=float, nargs="+",
                    default=[0, 2, 4, 8, 16, 32, 64, 128],
                    help="Fourier-radius shell edges for the dE_r versus dE_x audit.")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.lead_steps != 1:
        raise SystemExit("this audit requires --lead-steps=1")
    set_seed(args.seed)
    args.fm, args.fm_channels = "poseidon", "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    calibration = _load(args.fm_data_path, args.n_cal_traj, args.steps,
                        args.calibration_offset, fm.device)
    fine = _load(args.fm_data_path, args.n_traj, args.steps, args.split_offset, fm.device)
    residual_scale = _calibration_residual_scale(calibration, spec, args.steps, fm.device)

    print_header(f"Poseidon endpoint-residual / AZEBAN-flow alignment audit: {fm.info.name}")
    print(f"  transitions: {len(fine)} trajectories x {args.steps}; adaptive AZEBAN CFL={args.flow_cfl:g}")
    print(f"  physics energy: E=0.5||r_trapezoid||^2/R_cal; R_cal^1/2={residual_scale:.5g} "
          f"from {len(calibration)} calibration trajectories")
    print("  primary comparison: d||r_trapezoid||^2/dx versus d||FM endpoint - AZEBAN endpoint||^2/dx")
    print("  positive cosine means the physics descent points toward AZEBAN correction x* - x")

    teacher_rows, autoreg_rows = [], []
    for i, trajectory in enumerate(fine):
        print(f"  trajectory {i + 1}/{len(fine)}", flush=True)
        auto_previous = trajectory[0]
        for t in range(args.steps):
            # Isolates each FM one-step map at an observed data state.
            _, teacher = _one(fm, spec, trajectory[2 * t], args.flow_cfl, residual_scale)
            teacher_rows.append(teacher)
            # Reproduces the actual autoregressive FM conditioning path.
            auto_raw, auto = _one(fm, spec, auto_previous, args.flow_cfl, residual_scale)
            autoreg_rows.append(auto)
            auto_previous = auto_raw

    summaries = {"teacher_forced": _summarize(teacher_rows),
                 "autoregressive": _summarize(autoreg_rows)}
    spectral = {
        "teacher_forced": _spectral_shell_summary(teacher_rows, spec.n, args.spectral_bands),
        "autoregressive": _spectral_shell_summary(autoreg_rows, spec.n, args.spectral_bands),
    }
    table = Table("conditioning", "cos(dEr,dEx)", "|dEr|/|dEx|", "cos(dEphys,dEx)", "|dEphys|/|dEx|", "FM-oracle RMS", "|dEr| RMS", "|dEphys| RMS")
    for name in ("teacher_forced", "autoregressive"):
        s = summaries[name]
        table.add(name, s["residual_oracle_energy_gradient_cosine"],
                  s["residual_oracle_energy_gradient_norm_ratio"],
                  s["calibrated_oracle_energy_gradient_cosine"],
                  s["calibrated_oracle_energy_gradient_norm_ratio"],
                  s["raw_oracle_rms"], s["residual_energy_gradient_rms"], s["calibrated_gradient_rms"])
    print("\n  oracle-direction comparison")
    print(table)
    print("\n  Fourier/Stokes-eigenmode decomposition of the energy gradients")
    print("  Shares are fractions of total gradient energy.  High-k alignment is useful only if"
          " that band also has non-negligible oracle-error energy.")
    for name in ("teacher_forced", "autoregressive"):
        shell_table = Table(f"{name} k-band", "modes", "cos(dEr,dEx)", "cos(dEphys,dEx)",
                            "dEx share", "dEr share", "|dEphys|/|dEx|")
        for item in spectral[name]:
            shell_table.add(f"[{item['k_low']:g},{item['k_high']:g})", item["modes"],
                            item["cos_dEr_dEx"], item["cos_dEphys_dEx"],
                            item["oracle_energy_share"], item["dEr_energy_share"],
                            item["dEphys_dEx_norm_ratio"])
        print(shell_table)
    print("\n  Pass signal: strongly positive cos(dEr,dEx).  A negative or near-zero value means "
          "the residual-energy update cannot point toward the AZEBAN correction at this cadence.")
    output = {"stage": "poseidon_residual_oracle_alignment", "metadata": fm_metadata(args, fm, spec),
              "residual": "endpoint trapezoid", "residual_scale": residual_scale,
              "oracle": "native adaptive AZEBAN flow", "results": summaries,
              "spectral_bands": list(args.spectral_bands), "spectral_results": spectral}
    path = save_json(output, results_path_scale("poseidon_residual_oracle_alignment", fm_result_key(args),
                                                 "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
