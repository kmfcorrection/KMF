#!/usr/bin/env python3
"""Validation gate for candidate-conditioned effective residual discrepancy.

For frozen teacher-forced Poseidon endpoint y=FM(x), held-out truth y*, and
the known midpoint transport operator M=I-dt/2 J_F(midpoint(x,y)), define

    b_eff(x,y) = R_x(y) - M[omega(y)-omega(y*)].

Only calibration trajectories construct this target.  At inference a model may
consume x, y, and known PDE RHS evaluations, but never y*.  This script tests
whether that target is predictable before any new S4 correction is attempted.

The candidate model is deliberately restricted: a fixed Fourier intercept plus
complex ridge coefficients shared by radial wavenumber shells.  It is a
calibrated discrete-physics closure, not a neural output corrector and not a
PDE endpoint solve.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.fm_eval_common import (configure_native_poseidon_cadence,
                                               load_poseidon_trajectories,
                                               native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm
from hipp.scale.fm_physics import MidpointTransportEnergy2D
from hipp.utils import print_header, save_json, set_seed


FEATURE_NAMES = ("omega_prev", "omega_raw", "rhs_prev", "rhs_raw", "increment_raw")


def _energy(spec, previous, channels, device):
    return MidpointTransportEnergy2D(spec, previous, channels, dt=spec.dt_out,
                                     substeps=1, divergence_weight=0.0, device=device)


def _predict(fm, current):
    with torch.no_grad():
        return fm.predict(current.to(fm.device, fm.dtype)).double().reshape(1, -1)


def _defect(energy, endpoint):
    return energy.midpoint_endpoint_defect(energy.to_vorticity(endpoint))


def _apply_M(energy, raw, vector):
    midpoint = .5 * (energy.previous_vorticity + energy.to_vorticity(raw))
    _, jv = torch.func.jvp(energy.rhs_vorticity, (midpoint,), (vector,))
    return vector - .5 * energy.dt * jv


def _inference_features(energy, raw):
    """The exact five feature fields permitted at inference time."""
    with torch.no_grad():
        w_prev = energy.previous_vorticity
        w_raw = energy.to_vorticity(raw)
        return torch.stack((w_prev[0], w_raw[0], energy.rhs_vorticity(w_prev)[0],
                            energy.rhs_vorticity(w_raw)[0], (w_raw - w_prev)[0]))


def _one_example(fm, spec, current, truth):
    """Build calibration target and inference-available spectral features."""
    energy = _energy(spec, current, fm.state_shape[0], fm.device)
    raw = _predict(fm, current)
    truth = truth.reshape(1, -1).to(fm.device, torch.float64)
    with torch.no_grad():
        w_prev = energy.previous_vorticity
        w_raw = energy.to_vorticity(raw)
        w_truth = energy.to_vorticity(truth)
        r_raw = _defect(energy, raw)
        error = w_raw - w_truth
    transported_error = _apply_M(energy, raw, error).detach()
    effective = (r_raw - transported_error).detach()
    features = _inference_features(energy, raw)
    return features.cpu(), effective[0].cpu(), r_raw[0].cpu()


def _examples(fm, trajectories, spec, args, label):
    features, effective, raw_defects = [], [], []
    for i, trajectory in enumerate(trajectories):
        print(f"    {label} trajectory {i + 1}/{len(trajectories)}", flush=True)
        for t in range(args.steps):
            feat, target, raw = _one_example(fm, spec, trajectory[t], trajectory[t + 1])
            features.append(feat)
            effective.append(target)
            raw_defects.append(raw)
    return torch.stack(features), torch.stack(effective), torch.stack(raw_defects)


class ShellSpectralRidge:
    """Intercept-per-mode + standardized affine features pooled by |k| shell."""
    def __init__(self, intercept, feature_mean, scales, coefficients, shell, label):
        self.intercept, self.feature_mean = intercept, feature_mean
        self.scales, self.coefficients, self.shell, self.label = scales, coefficients, shell, label

    @classmethod
    def fit(cls, features: torch.Tensor, target: torch.Tensor, shells: int,
            ridge_rel: float, label: str):
        # features [S,F,n,n], target [S,n,n]; all computations use rFFT modes.
        fh = torch.fft.rfft2(features)
        th = torch.fft.rfft2(target)
        feature_mean, intercept = fh.mean(0), th.mean(0)
        centered = fh - feature_mean.unsqueeze(0)
        n, nr = th.shape[-2:]
        kx = torch.fft.fftfreq(n, dtype=torch.float64).view(-1, 1)
        ky = torch.fft.rfftfreq(n, dtype=torch.float64).view(1, -1)
        radius = (kx.square() + ky.square()).sqrt()
        shell = torch.clamp((radius / radius.max().clamp_min(1e-30) * shells).long(), max=shells - 1)
        f = features.shape[1]
        scales = torch.ones(shells, f, dtype=torch.float64)
        coefficients = torch.zeros(shells, f, dtype=torch.complex128)
        for s in range(shells):
            mask = shell == s
            # Rows jointly pool all calibration samples and all modes in shell.
            x = centered[:, :, mask].permute(0, 2, 1).reshape(-1, f).to(torch.complex128)
            y = (th - intercept.unsqueeze(0))[:, mask].reshape(-1).to(torch.complex128)
            scale = x.abs().square().mean(0).sqrt().clamp_min(1e-20)
            z = x / scale
            gram = z.conj().T @ z
            ridge = float(ridge_rel) * torch.eye(f, dtype=torch.complex128) * max(float(torch.trace(gram).real) / f, 1e-20)
            coefficients[s] = torch.linalg.solve(gram + ridge, z.conj().T @ y) / scale
            scales[s] = scale.real
        return cls(intercept, feature_mean, scales, coefficients, shell, label)

    @classmethod
    def constant(cls, target: torch.Tensor):
        n, nr = target.shape[-2], target.shape[-1] // 2 + 1
        intercept = torch.fft.rfft2(target).mean(0)
        return cls(intercept, torch.zeros((len(FEATURE_NAMES), n, nr), dtype=torch.complex128),
                   torch.ones((1, len(FEATURE_NAMES))), torch.zeros((1, len(FEATURE_NAMES)), dtype=torch.complex128),
                   torch.zeros((n, nr), dtype=torch.long), "constant")

    def predict(self, features: torch.Tensor):
        fh = torch.fft.rfft2(features)
        z = fh - self.feature_mean.to(fh.device).unsqueeze(0)
        output = self.intercept.to(fh.device).unsqueeze(0).expand(features.shape[0], -1, -1).clone()
        for s in range(self.coefficients.shape[0]):
            mask = self.shell.to(fh.device) == s
            if not bool(mask.any()):
                continue
            coeff = self.coefficients[s].to(fh.device)
            output[:, mask] += (z[:, :, mask] * coeff[None, :, None]).sum(1)
        return torch.fft.irfft2(output, s=features.shape[-2:])


def _metrics(prediction, target):
    p, t = prediction.flatten(1), target.flatten(1)
    relative = (p - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-30)
    cosine = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-30)
    mse = (p - t).square().mean() / t.square().mean().clamp_min(1e-30)
    return {"relative_mse": float(mse), "relative_error_mean": float(relative.mean()),
            "direction_cosine_mean": float(cosine.mean())}


def main():
    parser = base_parser_scale(__doc__.splitlines()[0])
    parser.add_argument("--lead-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--n-cal-traj", type=int, default=32)
    parser.add_argument("--n-val-traj", type=int, default=16)
    parser.add_argument("--shells", type=int, default=8)
    parser.add_argument("--ridge-grid", nargs="+", type=float,
                        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--fm-data-path", required=True)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/scale/poseidon_effective_discrepancy_audit"))
    args = parser.parse_args()
    if args.fm != "poseidon" or args.fm_channels != "velocity":
        raise SystemExit("this audit currently supports Poseidon velocity only")
    if min(args.steps, args.n_cal_traj, args.n_val_traj, args.shells) < 1:
        raise SystemExit("steps, trajectory counts, and shells must be positive")
    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj, args.steps, args.lead_steps, 0)
    val = load_poseidon_trajectories(args.fm_data_path, fm, args.n_val_traj, args.steps, args.lead_steps, args.n_cal_traj)
    print_header(f"Poseidon effective-discrepancy validation gate: {fm.info.name}")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}; transitions/trajectory={args.steps}")
    print("  target: b_eff=R(raw)-M(raw)[omega(raw)-omega(truth)]; truth is calibration/validation only")
    print("  predictors: current/raw vorticity, their known RHS values, and raw increment; no PDE endpoint solve")
    print("\n  building calibration examples")
    cal_f, cal_b, _ = _examples(fm, cal, spec, args, "calibration")
    print("\n  building validation examples")
    val_f, val_b, _ = _examples(fm, val, spec, args, "validation")
    candidates = []
    constant = ShellSpectralRidge.constant(cal_b)
    score = _metrics(constant.predict(val_f), val_b)
    candidates.append(("constant", None, constant, score))
    print(f"\n  constant baseline: validation relative MSE={score['relative_mse']:.6g}; cosine={score['direction_cosine_mean']:+.4f}")
    print("  selecting shell-pooled spectral ridge on validation")
    for ridge in args.ridge_grid:
        model = ShellSpectralRidge.fit(cal_f, cal_b, args.shells, ridge, f"shell_ridge{ridge:g}")
        score = _metrics(model.predict(val_f), val_b)
        candidates.append(("shell_ridge", float(ridge), model, score))
        print(f"    ridge={ridge:g}: relative MSE={score['relative_mse']:.6g}; cosine={score['direction_cosine_mean']:+.4f}")
    kind, ridge, model, best = min(candidates, key=lambda row: row[3]["relative_mse"])
    cal_score = _metrics(model.predict(cal_f), cal_b)
    print(f"\n  selected {model.label}: calibration/validation relative MSE="
          f"{cal_score['relative_mse']:.6g}/{best['relative_mse']:.6g}")
    print("  PASS requires validation relative MSE materially below 1 and positive directional agreement.")
    payload = {"purpose": "validation-only gate for deployable candidate-conditioned effective discrepancy",
               "target": "b_eff=R_x(raw)-M_x(raw)[omega(raw)-omega(truth)]",
               "inference_features": list(FEATURE_NAMES), "no_pde_endpoint_solver": True,
               "truth_usage": "calibration targets and validation model selection only",
               "n_calibration_examples": int(cal_b.shape[0]), "n_validation_examples": int(val_b.shape[0]),
               "shells": args.shells, "constant_validation": candidates[0][3],
               "candidates": [{"kind": k, "ridge": r, "validation": s} for k, r, _, s in candidates],
               "selected": {"kind": kind, "ridge": ridge, "calibration": cal_score, "validation": best}}
    path = save_json(payload, args.out_dir / f"poseidon_{args.fm_size}_{args.tag}_results.json")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
