#!/usr/bin/env python3
"""Held-out fidelity audit for residual-only midpoint transport on Poseidon.

For the actual frozen Poseidon forecast y=F_FM(x), assess whether the cheap
one-JVP observable estimates the *vorticity forecast error*

    e_omega = curl(y) - curl(x_true_next).

The proposed observable consumes only x and y.  Next-frame truth is used only
offline to fit the calibration bias and to score held-out fidelity.  No method
in this file calls an AZEBAN/RK flow map or advances a PDE state.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (configure_native_poseidon_cadence,
                                               load_poseidon_trajectories,
                                               native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm
from hipp.scale.fm_physics import MidpointTransportEnergy2D
from hipp.utils import set_seed


def _metric(proxy: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p, t = proxy.flatten(1), target.flatten(1)
    rel = (p - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-30)
    cos = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-30)
    return {"relative_error_mean": float(rel.mean()),
            "relative_error_p90": float(torch.quantile(rel, 0.9)),
            "direction_cosine_mean": float(cos.mean()),
            "proxy_rms": float(proxy.square().mean().sqrt()),
            "target_rms": float(target.square().mean().sqrt())}


def _transport(energy: MidpointTransportEnergy2D, candidate: torch.Tensor,
               bias: torch.Tensor, terms: int) -> torch.Tensor:
    """Finite Neumann transport of an endpoint midpoint defect.

    ``terms=1`` is the proposed method.  Higher terms are audit-only controls;
    they are intentionally not selectable in the S4 likelihood.
    """
    w = energy.to_vorticity(candidate)
    midpoint = 0.5 * (energy.previous_vorticity + w)
    value = energy.midpoint_endpoint_defect(w) - bias
    total = value
    for _ in range(terms):
        _, tangent = torch.func.jvp(energy.rhs_vorticity, (midpoint,), (value,))
        value = 0.5 * energy.dt * tangent
        total = total + value
    return total


def _predict_one(fm, current: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return fm.predict(current.to(fm.device, fm.dtype)).double().reshape(1, -1)


def _calibration_bias(fm, calibration: torch.Tensor, spec, args) -> torch.Tensor:
    defects = []
    for i in range(calibration.shape[0]):
        for t in range(args.steps):
            previous = calibration[i, t]
            truth = calibration[i, t + 1].reshape(1, -1)
            energy = MidpointTransportEnergy2D(
                spec, previous, 2, dt=spec.dt_out, substeps=1,
                divergence_weight=0.0, device=fm.device)
            with torch.no_grad():
                defects.append(energy.midpoint_endpoint_defect(
                    energy.to_vorticity(truth))[0])
    return torch.stack(defects).mean(0).detach()


def main() -> None:
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3,
                    help="teacher-forced transitions per trajectory")
    ap.add_argument("--n-cal-traj", type=int, default=32)
    ap.add_argument("--n-test-traj", type=int, default=32)
    ap.add_argument("--terms", nargs="+", type=int, default=[0, 1, 2, 4])
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/scale/poseidon_midpoint_transport_fidelity"))
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.fm != "poseidon" or args.fm_channels != "velocity":
        raise SystemExit("this audit is defined for Poseidon velocity on native NS-Gauss")
    if args.steps < 1 or args.n_cal_traj < 1 or args.n_test_traj < 1:
        raise SystemExit("steps and trajectory counts must be positive")
    if any(t < 0 for t in args.terms):
        raise SystemExit("transport terms must be nonnegative")
    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    if args.lead_steps != 1:
        spec = type(spec)(**{**spec.__dict__, "dt": spec.dt * args.lead_steps})
    calibration = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj,
                                              args.steps, args.lead_steps, offset=0)
    held_out = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj,
                                           args.steps, args.lead_steps,
                                           offset=args.n_cal_traj)
    print("=" * 78)
    print(f"Poseidon midpoint-transport residual-fidelity audit: {fm.info.name}")
    print("=" * 78)
    print(f"  split: calibration={args.n_cal_traj}, held-out={args.n_test_traj}; "
          f"teacher-forced transitions/trajectory={args.steps}; dt={spec.dt_out:g}")
    print("  proposed proxy: endpoint midpoint defect + exactly one known-PDE RHS JVP")
    print("  no PDE endpoint solver is called; held-out next-frame truth is evaluation-only")

    bias = _calibration_bias(fm, calibration, spec, args)
    print(f"  calibration midpoint-defect bias RMS={float(bias.square().mean().sqrt()):.5g}")
    buckets = {f"midpoint_transport_{t}JVP": [] for t in args.terms}
    targets, truth_rms, raw_rms = [], [], []
    for i in range(held_out.shape[0]):
        print(f"  held-out trajectory {i + 1}/{held_out.shape[0]}", flush=True)
        for t in range(args.steps):
            current = held_out[i, t]
            truth = held_out[i, t + 1].reshape(1, -1).to(fm.device, torch.float64)
            raw = _predict_one(fm, current)
            energy = MidpointTransportEnergy2D(
                spec, current, 2, dt=spec.dt_out, substeps=1,
                divergence_weight=0.0, device=fm.device)
            with torch.no_grad():
                target = energy.to_vorticity(raw) - energy.to_vorticity(truth)
                truth_rms.append(float(_transport(energy, truth, bias, 1).square().mean().sqrt()))
                raw_rms.append(float(_transport(energy, raw, bias, 1).square().mean().sqrt()))
            # Forward-mode JVP is intentionally outside no_grad: this checks
            # the exact differentiable algebra used by the S4 likelihood.
            for terms in args.terms:
                buckets[f"midpoint_transport_{terms}JVP"].append(
                    _transport(energy, raw, bias, terms).detach().cpu())
            targets.append(target.detach().cpu())

    target = torch.cat(targets)
    summary = {name: _metric(torch.cat(values), target) for name, values in buckets.items()}
    ordered = sorted(summary, key=lambda name: summary[name]["relative_error_mean"])
    gate = {"truth_rms_median": float(np.median(truth_rms)),
            "raw_rms_median": float(np.median(raw_rms)),
            "truth_over_raw": float(np.median(truth_rms) / max(np.median(raw_rms), 1e-30)),
            "truth_wins_fraction": float(np.mean(np.asarray(truth_rms) < np.asarray(raw_rms)))}
    print("\nmethod                      rel.error   p90      direction cosine")
    for name in ordered:
        q = summary[name]
        print(f"{name:27s} {q['relative_error_mean']:>9.4g} {q['relative_error_p90']:>8.4g} "
              f"{q['direction_cosine_mean']:>18.5f}")
    print("\n  residual discrimination gate: "
          f"truth/raw={gate['truth_over_raw']:.4g}; truth wins={100 * gate['truth_wins_fraction']:.1f}%")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ordered
    axes[0].bar(labels, [summary[k]["relative_error_mean"] for k in labels])
    axes[0].set_title("Proxy error vs actual FM vorticity error")
    axes[0].set_ylabel("relative vector error (lower is better)")
    axes[1].bar(labels, [summary[k]["direction_cosine_mean"] for k in labels])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Proxy direction agreement")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fidelity.png", dpi=180)
    payload = {
        "purpose": "held-out residual-only fidelity against actual Poseidon FM errors",
        "inference_rule": "proxy uses only current state and frozen FM endpoint; truth is calibration/evaluation only",
        "target": "vorticity forecast error curl(FM(x_t)) - curl(x_{t+1})",
        "physical_term": "no flow integration; endpoint midpoint defect plus finite-Neumann PDE-RHS JVP transport",
        "calibration_bias_rms": float(bias.square().mean().sqrt()),
        "calibration_trajectories": args.n_cal_traj,
        "held_out_trajectories": args.n_test_traj,
        "teacher_forced_transitions_per_trajectory": args.steps,
        "n_held_out_transitions": int(target.shape[0]),
        "residual_discrimination_gate": gate,
        "metrics": summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'fidelity.png'}")


if __name__ == "__main__":
    main()
