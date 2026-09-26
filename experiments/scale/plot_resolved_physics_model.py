#!/usr/bin/env python3
"""Visualize a native Poseidon transition and the resolved-physics likelihood.

This is a diagnostic/communication tool.  It uses precisely the same spectral
restriction and coarse AZEBAN endpoint used by
``s4_resolved_physics_assimilation.py``.  In particular, the coarse target is
*not* presented as a full-resolution ground-truth next state: it is only an
observation in the modes represented by the chosen physics grid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    configure_native_poseidon_cadence, load_poseidon_trajectories,
    native_poseidon_spec,
)
from hipp.scale.common_scale import base_parser_scale, load_fm
from hipp.scale.fm_physics import ResolvedCoarseFlowEnergy2D
from hipp.utils import set_seed


def _sym_limits(*fields):
    value = max(float(x.detach().abs().amax()) for x in fields)
    return max(value, 1e-12)


def _image(ax, field, title, *, lim=None, cmap="RdBu_r"):
    arr = field.detach().float().cpu().numpy()
    if lim is None:
        lim = _sym_limits(field)
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _vorticity_velocity(state):
    """Curl of a ``(1, 2, n, n)`` velocity state on the unit periodic box."""
    n = state.shape[-1]
    wave = 2.0 * np.pi * torch.fft.fftfreq(n, d=1.0 / n,
                                             device=state.device, dtype=state.dtype)
    kx, ky = torch.meshgrid(wave, wave, indexing="ij")
    uh = torch.fft.fft2(state[:, 0])
    vh = torch.fft.fft2(state[:, 1])
    return torch.fft.ifft2(1j * kx * vh - 1j * ky * uh).real[0]


def _make_energy(spec, previous, channels, args, bias=None, covariance=None, device=None):
    return ResolvedCoarseFlowEnergy2D(
        spec, previous, channels, physics_n=args.physics_resolution,
        flow_cfl=args.flow_cfl, dt=max(1, args.lead_steps) * spec.dt_out,
        divergence_weight=0.0, discrepancy_bias=bias,
        discrepancy_covariance=covariance, device=device,
    )


def _fit_discrepancy(calibration, spec, channels, args, device):
    """Fit the displayed residual floor from calibration transitions only."""
    samples = []
    for trajectory in calibration:
        for t in range(args.steps):
            energy = _make_energy(spec, trajectory[t], channels, args, device=device)
            samples.append((energy.restrict(trajectory[t + 1]) - energy.low_flow_target).squeeze(0))
    d = torch.stack(samples).double()
    bias = d.mean(0)
    centered = d - bias
    # The display needs the same diagonal metric used by the default S4 run.
    variance = centered.square().mean(0)
    scale = variance.mean().clamp(min=1e-30)
    covariance = torch.diag(variance + args.ridge_rel * scale)
    return bias, covariance, {
        "samples": int(d.shape[0]),
        "bias_rms": float(bias.square().mean().sqrt()),
        "centered_rms": float(centered.square().mean().sqrt()),
        "variance_mean": float(variance.mean()),
    }


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=3,
                    help="transitions available per plotted trajectory")
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=16)
    ap.add_argument("--n-val-traj", type=int, default=8)
    ap.add_argument("--physics-resolution", type=int, default=8)
    ap.add_argument("--flow-cfl", type=float, default=0.5)
    ap.add_argument("--trajectory", type=int, default=0,
                    help="held-out trajectory index to draw")
    ap.add_argument("--transition", type=int, default=0,
                    help="transition index within the held-out trajectory")
    ap.add_argument("--out-dir", default="results/figures/resolved_physics")
    ap.add_argument("--ridge-rel", type=float, default=1e-4)
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.fm != "poseidon":
        raise SystemExit("this visualizer is currently defined for native Poseidon velocity data")
    if not 0 < args.physics_resolution < 128 or 128 % args.physics_resolution:
        raise SystemExit("--physics-resolution must divide 128 and lie below 128")
    if not 0 <= args.transition < args.steps:
        raise SystemExit("--transition must be in [0, --steps)")

    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj,
                                     args.steps, args.lead_steps, offset=0)
    held_out_offset = args.n_cal_traj + args.n_val_traj + args.trajectory
    example = load_poseidon_trajectories(args.fm_data_path, fm, 1, args.steps,
                                         args.lead_steps, offset=held_out_offset)[0]
    bias, covariance, discrepancy = _fit_discrepancy(
        cal, spec, fm.state_shape[0], args, fm.device)

    current = example[args.transition]
    truth = example[args.transition + 1]
    with torch.no_grad():
        raw = fm.predict(current).double()
    energy = _make_energy(spec, current, fm.state_shape[0], args, bias, covariance, fm.device)

    # Fine-grid fields: what the frozen FM actually forecasts.
    omega_current = energy.to_vorticity(current)[0]
    omega_truth = energy.to_vorticity(truth)[0]
    omega_raw = energy.to_vorticity(raw)[0]
    omega_error = omega_raw - omega_truth

    # Coarse/resolved fields: the only quantities appearing in the likelihood.
    truth_low = energy.restrict(truth)
    raw_low = energy.restrict(raw)
    target_low = energy.observation_target
    low_truth_omega = _vorticity_velocity(
        truth_low.reshape(1, 2, args.physics_resolution, args.physics_resolution))
    low_raw_omega = _vorticity_velocity(
        raw_low.reshape(1, 2, args.physics_resolution, args.physics_resolution))
    low_target_omega = _vorticity_velocity(
        target_low.reshape(1, 2, args.physics_resolution, args.physics_resolution))
    low_truth_residual = low_truth_omega - low_target_omega
    low_raw_residual = low_raw_omega - low_target_omega

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fine_lim = _sym_limits(omega_current, omega_truth, omega_raw)
    error_lim = _sym_limits(omega_error)
    low_lim = _sym_limits(low_truth_omega, low_raw_omega, low_target_omega)
    low_residual_lim = _sym_limits(low_truth_residual, low_raw_residual)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    panels = [
        (omega_current, r"observed $\omega_t$", fine_lim),
        (omega_truth, r"actual data: $\omega_{t+\Delta t}$", fine_lim),
        (omega_raw, r"frozen FM: $\hat\omega_{t+\Delta t}$", fine_lim),
        (omega_error, r"FM error: $\hat\omega-\omega^{true}$", error_lim),
        (low_truth_residual, r"truth coarse discrepancy: $H\omega^{true}-H\Psi_h-b$", low_residual_lim),
        (low_raw_residual, r"modelled discrepancy: $H\hat\omega-H\Psi_h-b$", low_residual_lim),
    ]
    for ax, (field, label, lim) in zip(axes.flat, panels):
        _image(ax, field, label, lim=lim)
    fig.suptitle(
        f"Fine forecast versus calibrated resolved physics (Poseidon, h={args.physics_resolution})",
        fontsize=13, y=1.02)
    _save(fig, out / "01_fine_forecast_and_residual.png")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    panels = [
        (low_target_omega, r"physics target: $H\Psi_h(x_t)+b$"),
        (low_truth_omega, r"actual resolved next state: $H x_{t+\Delta t}^{true}$"),
        (low_raw_omega, r"FM resolved forecast: $H\hat x_{t+\Delta t}$"),
    ]
    for ax, (field, label) in zip(axes.flat, panels):
        _image(ax, field, label, lim=low_lim)
    fig.suptitle(
        "What the resolved likelihood compares — all panels are on the coarse physics grid",
        fontsize=12, y=1.03)
    _save(fig, out / "02_resolved_observation_space.png")

    summary = {
        "description": "Calibration-aware resolved-mode likelihood visualization",
        "trajectory_is_held_out": True,
        "trajectory_index_within_held_out_split": args.trajectory,
        "transition": args.transition,
        "physics_resolution": args.physics_resolution,
        "physics_dt": max(1, args.lead_steps) * spec.dt_out,
        "calibration_discrepancy": discrepancy,
        "coarse_target_rms": float(target_low.square().mean().sqrt()),
        "truth_coarse_discrepancy_rms": float((truth_low - target_low).square().mean().sqrt()),
        "fm_coarse_discrepancy_rms": float((raw_low - target_low).square().mean().sqrt()),
        "fm_full_state_rmse": float((raw - truth).square().mean().sqrt()),
        "interpretation": (
            "Only Hx is modelled by the coarse physics likelihood. Fine modes outside H remain governed by the FM prior; "
            "the lifted coarse endpoint is intentionally not plotted as a proposed full fine-grid replacement."
        ),
    }
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {path}")
    print("\nRead the plots as follows:")
    print("  01: top row is the actual high-resolution forecasting problem; bottom-right is the residual HILP models.")
    print("  02: the physics target and candidate states are compared only after spectral restriction to h x h.")


if __name__ == "__main__":
    main()
