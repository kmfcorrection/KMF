#!/usr/bin/env python3
"""Compare strong, weak, and discrete-flow PDE diagnostics on one FM transition.

The output is deliberately a *diagnostic gallery*, not an S4 benchmark.  It
answers a prerequisite question: does a candidate physics quantity distinguish
the actual next state from the frozen-FM forecast, after allowing for the
nonzero discretization floor observed on calibration transitions?
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
from hipp.scale.fm_physics import (
    CoarseNativeFlowEnergy2D, FMPhysicsEnergy2D, NativeFlowEnergy2D,
    spectral_resample_state,
)
from hipp.scale.traditional_residuals import (
    finite_difference_midpoint_residual, finite_volume_midpoint_residual,
    galerkin_residual, pseudospectral_midpoint_residual, fixed_step_azeban_endpoint,
    smagorinsky_les_endpoint, fit_pod_vorticity_basis, pod_galerkin_endpoint,
)
from hipp.utils import set_seed


def _vorticity(state):
    """Vorticity of `(B,2,n,n)` unit-box velocity fields."""
    n = state.shape[-1]
    q = 2 * np.pi * torch.fft.fftfreq(n, d=1 / n, device=state.device, dtype=state.dtype)
    kx, ky = torch.meshgrid(q, q, indexing="ij")
    return torch.fft.ifft2(1j * kx * torch.fft.fft2(state[:, 1]) -
                           1j * ky * torch.fft.fft2(state[:, 0])).real


def _lowpass_scalar(field, n):
    """Spectrally restrict and lift a scalar field: P_n r on the fine grid."""
    fine_n = field.shape[-1]
    if n == fine_n:
        return field
    coarse = spectral_resample_state(field[:, None], n)
    return spectral_resample_state(coarse, fine_n)[:, 0]


def _limits(*xs):
    return max(max(float(x.abs().amax()) for x in xs), 1e-12)


def _show(ax, x, title, vmax):
    im = ax.imshow(x.detach().float().cpu().numpy(), origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _write(fig, target):
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {target}")


def _midpoint_residual(spec, previous, candidate, device):
    return pseudospectral_midpoint_residual(
        spec, previous, candidate, dt=spec.dt_out, device=device)


def _traditional_residuals(spec, previous, candidate, device):
    """All non-learned strong/weak residual candidates on one transition."""
    n = spec.n
    # ``previous`` may contain PHYSICS_BATCH independent transitions.  Preserve
    # that leading dimension: every traditional operator below is vectorized.
    x0 = previous.double().reshape(-1, 2, n, n)
    y = candidate.double().reshape(-1, 2, n, n)
    spectral = _midpoint_residual(spec, previous, candidate, device)
    out = {"pseudospectral": spectral}
    for order in (2, 4, 6):
        out[f"fd-{order}"] = finite_difference_midpoint_residual(
            x0, y, L=spec.L, dt=spec.dt_out, order=order)
    out["finite-volume"] = finite_volume_midpoint_residual(
        x0, y, L=spec.L, dt=spec.dt_out)
    return out


def _flow_difference(spec, previous, candidate, h, cfl, device):
    """Difference in *solver-resolved* velocity modes, never a lifted field loss."""
    if h == spec.n:
        energy = NativeFlowEnergy2D(spec, previous, 2, divergence_weight=0.0,
                                    flow_cfl=cfl, dt=spec.dt_out, device=device)
        return (candidate.double() - energy.flow_target).reshape(-1, 2, h, h)
    energy = CoarseNativeFlowEnergy2D(spec, previous, 2, physics_n=h,
                                      divergence_weight=0.0, flow_cfl=cfl,
                                      dt=spec.dt_out, device=device)
    candidate_low = spectral_resample_state(candidate.reshape(-1, 2, spec.n, spec.n), h)
    return candidate_low - energy.low_flow_target.reshape(-1, 2, h, h)


def _surrogate_differences(spec, previous, candidate, device, fixed_steps, les_resolutions,
                           les_cs, les_substeps, pod_model, pod_substeps):
    """Endpoint discrepancies for additional classical surrogate solvers."""
    out = {}
    candidate_state = candidate.double().reshape(-1, 2, spec.n, spec.n)
    for steps in fixed_steps:
        endpoint = fixed_step_azeban_endpoint(spec, previous, dt=spec.dt_out,
                                              substeps=steps, device=device)
        out[f"fixed-rk3-{steps}"] = _vorticity(
            candidate_state - endpoint.reshape(-1, 2, spec.n, spec.n))
    for h in les_resolutions:
        endpoint = smagorinsky_les_endpoint(spec, previous, coarse_n=h, dt=spec.dt_out,
                                            cs=les_cs, substeps=les_substeps, device=device)
        candidate_low = spectral_resample_state(candidate_state, h)
        out[f"les-smag-{h}"] = _vorticity(candidate_low - endpoint)
    if pod_model is not None:
        mean, bases = pod_model
        for rank, basis in bases.items():
            endpoint = pod_galerkin_endpoint(spec, previous, dt=spec.dt_out, mean=mean,
                                              basis=basis, substeps=pod_substeps, device=device)
            out[f"pod-galerkin-{rank}"] = _vorticity(candidate_state - endpoint)
    return out


def _calibration_floor(cal, spec, resolutions, cfl, device, batch_size, *, fixed_steps,
                       les_resolutions, les_cs, les_substeps, pod_model, pod_substeps):
    """Fit per-pixel calibration floors in batches, without retaining all fields."""
    previous_all = cal[:, :-1].reshape(-1, cal.shape[-1]).to(device)
    truth_all = cal[:, 1:].reshape(-1, cal.shape[-1]).to(device)
    # Each entry holds [count, sum(field), sum(field^2)].  This is exactly the
    # same mean/standard deviation as stack(...).std(), but uses O(n^2) rather
    # than O(number_of_transitions * n^2) memory.
    stats = {}
    for start in range(0, len(previous_all), batch_size):
        stop = min(start + batch_size, len(previous_all))
        previous, truth = previous_all[start:stop], truth_all[start:stop]
        print(f"  calibration transitions {start + 1}-{stop}/{len(previous_all)}")
        residuals = _traditional_residuals(spec, previous, truth, device)
        fields = dict(residuals)
        for h in resolutions:
            fields[f"galerkin-{h}"] = galerkin_residual(residuals["pseudospectral"], h)
            fields[f"flow-{h}"] = _vorticity(_flow_difference(
                spec, previous, truth, h, cfl, device))
        fields.update(_surrogate_differences(
            spec, previous, truth, device, fixed_steps, les_resolutions,
            les_cs, les_substeps, pod_model, pod_substeps))
        for name, field in fields.items():
            field = field.double()
            total = stats.get(name)
            if total is None:
                stats[name] = [field.shape[0], field.sum(0), field.square().sum(0)]
            else:
                total[0] += field.shape[0]
                total[1] += field.sum(0)
                total[2] += field.square().sum(0)
    out = {}
    for name, (count, total, squared_total) in stats.items():
        mean = total / count
        variance = ((squared_total - count * mean.square()) / max(count - 1, 1)).clamp(min=1e-24)
        out[name] = {"mean": mean, "std": variance.sqrt(),
                     "truth_rms": float((squared_total / count).mean().sqrt())}
    return out


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=16)
    ap.add_argument("--n-val-traj", type=int, default=8)
    ap.add_argument("--physics-resolutions", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--flow-cfl", type=float, default=0.5)
    ap.add_argument("--physics-batch", type=int, default=8,
                    help="independent transitions evaluated together on the GPU")
    ap.add_argument("--fixed-rk3-substeps", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--les-resolutions", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--les-cs", type=float, default=0.17)
    ap.add_argument("--les-substeps", type=int, default=16)
    ap.add_argument("--pod-ranks", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--pod-substeps", type=int, default=16)
    ap.add_argument("--trajectory", type=int, default=0)
    ap.add_argument("--transition", type=int, default=0)
    ap.add_argument("--out-dir", default="results/figures/physics_approximations")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.fm != "poseidon":
        raise SystemExit("the gallery is currently validated for Poseidon native velocity transitions")
    if any(h <= 0 or h > 128 or 128 % h for h in args.physics_resolutions):
        raise SystemExit("physics resolutions must divide 128")
    if any(h <= 0 or h >= 128 or 128 % h for h in args.les_resolutions):
        raise SystemExit("LES resolutions must divide 128 and be below 128")
    if not 0 <= args.transition < args.steps:
        raise SystemExit("transition must lie in [0, steps)")

    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    if args.lead_steps != 1:
        spec = type(spec)(**{**spec.__dict__, "dt": spec.dt * args.lead_steps})
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj,
                                     args.steps, args.lead_steps, offset=0)
    example = load_poseidon_trajectories(args.fm_data_path, fm, 1, args.steps,
                                         args.lead_steps,
                                         offset=args.n_cal_traj + args.n_val_traj + args.trajectory)[0]
    pod_model = None
    if args.pod_ranks:
        print("building calibration-only POD bases ...")
        snapshots = cal[:, :-1].reshape(-1, cal.shape[-1])
        max_rank = max(args.pod_ranks)
        mean, full_basis = fit_pod_vorticity_basis(spec, snapshots, max_rank, fm.device)
        pod_model = (mean, {rank: full_basis[:, :rank] for rank in args.pod_ranks})
    print("fitting calibration-only residual floors ...")
    floors = _calibration_floor(cal, spec, args.physics_resolutions, args.flow_cfl,
                                fm.device, args.physics_batch,
                                fixed_steps=args.fixed_rk3_substeps,
                                les_resolutions=args.les_resolutions, les_cs=args.les_cs,
                                les_substeps=args.les_substeps, pod_model=pod_model,
                                pod_substeps=args.pod_substeps)

    previous = example[args.transition]
    truth = example[args.transition + 1]
    with torch.no_grad():
        raw = fm.predict(previous).double()
    truth_residuals = {k: v[0] for k, v in _traditional_residuals(
        spec, previous, truth, fm.device).items()}
    raw_residuals = {k: v[0] for k, v in _traditional_residuals(
        spec, previous, raw, fm.device).items()}
    truth_surrogates = {k: v[0] for k, v in _surrogate_differences(
        spec, previous, truth, fm.device, args.fixed_rk3_substeps,
        args.les_resolutions, args.les_cs, args.les_substeps, pod_model,
        args.pod_substeps).items()}
    raw_surrogates = {k: v[0] for k, v in _surrogate_differences(
        spec, previous, raw, fm.device, args.fixed_rk3_substeps,
        args.les_resolutions, args.les_cs, args.les_substeps, pod_model,
        args.pod_substeps).items()}

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # 1. Independent conventional strong residuals, followed by the same
    # pseudospectral residual in increasingly conservative Galerkin spaces.
    rows = ["pseudospectral", "fd-2", "fd-4", "fd-6", "finite-volume"] + [
        f"galerkin-{h}" for h in args.physics_resolutions]
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 2.7 * len(rows)))
    scores = {}
    for r, name in enumerate(rows):
        if name.startswith("galerkin-"):
            h = int(name.split("-")[1])
            rt = galerkin_residual(truth_residuals["pseudospectral"][None], h)[0]
            rr = galerkin_residual(raw_residuals["pseudospectral"][None], h)[0]
        else:
            rt, rr = truth_residuals[name], raw_residuals[name]
        f = floors[name]
        zt, zr = (rt - f["mean"]) / f["std"], (rr - f["mean"]) / f["std"]
        lim = _limits(zt, zr)
        _show(axes[r, 0], zt, f"{name}: truth standardized residual", lim)
        _show(axes[r, 1], zr, f"{name}: FM standardized residual", lim)
        _show(axes[r, 2], zr - zt, f"{name}: FM minus truth", lim)
        scores[name] = {"truth_standardized_rms": float(zt.square().mean().sqrt()),
                        "fm_standardized_rms": float(zr.square().mean().sqrt())}
    fig.suptitle("Traditional PDE residual approximations: numerical schemes and Galerkin test spaces", y=1.01)
    _write(fig, out / "01_traditional_and_weak_residuals.png")

    # 2. Endpoint differences in the actual grid of each AZEBAN solver.
    rows = ([f"flow-{h}" for h in args.physics_resolutions] +
            list(truth_surrogates))
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 2.7 * len(rows)))
    for r, name in enumerate(rows):
        if name.startswith("flow-"):
            h = int(name.split("-")[1])
            print(f"integrating AZEBAN endpoint at {h}x{h} for display ...")
            dt = _vorticity(_flow_difference(spec, previous, truth, h, args.flow_cfl, fm.device))[0]
            dr = _vorticity(_flow_difference(spec, previous, raw, h, args.flow_cfl, fm.device))[0]
        else:
            dt, dr = truth_surrogates[name], raw_surrogates[name]
        f = floors[name]
        zt, zr = (dt - f["mean"]) / f["std"], (dr - f["mean"]) / f["std"]
        lim = _limits(zt, zr)
        _show(axes[r, 0], zt, f"AZEBAN {h}²: truth endpoint discrepancy", lim)
        _show(axes[r, 1], zr, f"AZEBAN {h}²: FM endpoint discrepancy", lim)
        _show(axes[r, 2], zr - zt, f"AZEBAN {h}²: FM minus truth", lim)
        scores[name] = {"truth_standardized_rms": float(zt.square().mean().sqrt()),
                        "fm_standardized_rms": float(zr.square().mean().sqrt())}
    fig.suptitle("Classical endpoint surrogates — each is evaluated only in its own resolved modes", y=1.01)
    _write(fig, out / "02_classical_endpoint_surrogates.png")

    # 3. A compact, directly comparable ranking.  FM > truth is the required
    # discrimination condition; it alone does not prove a usable correction direction.
    labels = list(scores)
    truth_scores = [scores[x]["truth_standardized_rms"] for x in labels]
    fm_scores = [scores[x]["fm_standardized_rms"] for x in labels]
    pos = np.arange(len(labels)); width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.1), 4.5))
    ax.bar(pos - width / 2, truth_scores, width, label="actual transition")
    ax.bar(pos + width / 2, fm_scores, width, label="frozen FM forecast")
    ax.axhline(1, color="black", lw=0.8, ls="--", label="one calibrated residual std")
    ax.set_xticks(pos, labels, rotation=30, ha="right")
    ax.set_ylabel("RMS of calibration-standardized residual")
    ax.set_title("A useful physics diagnostic should score the FM forecast above truth")
    ax.legend()
    _write(fig, out / "03_discrimination_summary.png")

    payload = {
        "description": "Held-out diagnostic comparison; calibration statistics use fit trajectories only.",
        "physics_resolutions": args.physics_resolutions,
        "trajectory_index": args.trajectory, "transition": args.transition,
        "scores": scores,
        "floor_truth_rms": {k: v["truth_rms"] for k, v in floors.items()},
        "notes": {
            "pseudospectral": "Dealiased pseudospectral midpoint defect with AZEBAN's smooth spectral-viscosity filter.",
            "fd": "Periodic centered finite-difference strong residuals, orders 2/4/6, with a conventional local viscosity closure.",
            "finite_volume": "Conservative periodic flux-form residual, independent of the centered finite-difference discretization.",
            "galerkin": "Pseudospectral residual projected to low Fourier test modes, avoiding unreliable high-frequency constraints.",
            "flow": "AZEBAN endpoint discrepancy in the solver's own resolved modes. h=128 is an oracle-style consistency control, not a paper likelihood.",
            "fixed_rk3": "Same full-grid AZEBAN equation with a prescribed low SSP-RK3 step count; a pure speed/accuracy tradeoff.",
            "les_smag": "Closed coarse large-eddy simulation with a classical Smagorinsky eddy-viscosity closure.",
            "pod_galerkin": "Calibration-only POD--Galerkin physics ROM. Its nonlinear pseudospectral RHS is not hyper-reduced, so it is an accuracy control rather than a DEIM speed claim.",
        },
    }
    target = out / "summary.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
