#!/usr/bin/env python3
"""Endpoint-only HILP with a PDE residual on actual FM output times.

The physics likelihood contains no latent temporal state.  Starting from the
known input x_0, every state used by the residual is an actual autoregressive
FM output (initially) or its current MAP-corrected value (during inference):

  J(y) = prior(y) + lambda/2 || r_h(x_0, y_1, ..., y_T) ||^2 / R_cal.

The primary target is r_h=0.  Calibration estimates only the scalar residual
unit R_cal; it never subtracts a residual mean.  This is the endpoint-output
analogue of a physics score evaluated directly on a generated time series.
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

from experiments.scale.fm_eval_common import (POSEIDON_RAW_DT, configure_native_poseidon_cadence,
                                               fm_metadata, fm_result_key, native_poseidon_spec)
from experiments.scale.s4_fm_revised_hilp import InnovationPrior, IsotropicPrior, _raw_window_and_ops
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed

METHODS = ("isotropic", "block_innovation")
SCHEMES = ("backward_euler", "trapezoid", "centered")


def _load(path: str, n_traj: int, steps: int, offset: int, device) -> torch.Tensor:
    need = 2 * steps + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        a = np.asarray(h[key][offset:offset + n_traj, :need, :2], dtype=np.float32)
    return torch.as_tensor(a, device=device).reshape(n_traj, need, -1)


class ActualOutputResidual:
    """Discrete vorticity residual using only the output-time state sequence."""
    def __init__(self, spec, x0, steps, scale, scheme, divergence_weight, device):
        self.steps, self.dt, self.scheme = int(steps), 2 * POSEIDON_RAW_DT, scheme
        self.operator = FMPhysicsEnergy2D(spec, x0, 2, divergence_weight=0.0,
                                          dt=self.dt, device=device)
        self.x0 = x0.detach().double().reshape(1, -1).to(device)
        self.N = self.x0.shape[1]
        self.residual_dof = spec.n * spec.n
        self.scale = float(scale)
        self.divergence_weight = float(divergence_weight)
        if scheme == "centered" and self.steps < 2:
            raise ValueError("centered residual needs at least two future endpoints")

    def residuals(self, endpoints: torch.Tensor) -> torch.Tensor:
        ys = endpoints.reshape(self.steps, self.N)
        states = torch.cat((self.x0, ys), dim=0)
        w = self.operator.to_vorticity(states)
        rhs = self.operator.rhs_vorticity(w)
        if self.scheme == "backward_euler":
            return (w[1:] - w[:-1]) / self.dt - rhs[1:]
        if self.scheme == "trapezoid":
            return (w[1:] - w[:-1]) / self.dt - .5 * (rhs[1:] + rhs[:-1])
        # Central derivative at each interior actual output time.  It uses
        # states (y_{t-1}, y_t, y_{t+1}) only; no invented midpoint exists.
        return (w[2:] - w[:-2]) / (2 * self.dt) - rhs[1:-1]

    def value(self, endpoints: torch.Tensor) -> tuple[torch.Tensor, dict]:
        r = self.residuals(endpoints)
        # Exact scalar-R norm, summed over residual time nodes and vorticity
        # grid degrees of freedom.  This matches the endpoint prior's sum form.
        physical = .5 * r.square().sum() / (self.scale ** 2)
        div = self.operator.divergence(endpoints.reshape(self.steps, self.N))
        divergence = .5 * self.divergence_weight * div.square().sum()
        return physical + divergence, {"physical": physical, "divergence": divergence, "residual": r}


def _raw_package(fm, fine, spec, sigma2, steps, scale, args):
    packages = []
    for i, trajectory in enumerate(fine):
        print(f"    preparing trajectory {i + 1}/{len(fine)}", flush=True)
        raw, ops = _raw_window_and_ops(fm, trajectory[0], steps)
        residual = ActualOutputResidual(spec, trajectory[0], steps, scale, args.residual_scheme,
                                        args.divergence_weight, fm.device)
        packages.append({"raw": raw, "truth": trajectory[2::2].double(), "ops": ops,
                         "residual": residual})
    return packages


def _fit_residual_scale(fine, spec, args, device) -> float:
    values = []
    for trajectory in fine:
        op = ActualOutputResidual(spec, trajectory[0], args.steps, 1.0, args.residual_scheme,
                                  0.0, device)
        with torch.no_grad():
            values.append(op.residuals(trajectory[2::2].double()))
    return float(torch.cat(values).square().mean().sqrt().clamp(min=1e-30))


def _prior(package, method, alpha, sigma2):
    if method == "isotropic":
        return IsotropicPrior(package["raw"], alpha=alpha)
    return InnovationPrior(package["raw"], package["ops"], sigma2, alpha=alpha)


def _fit_alpha(packages, method, sigma2):
    values = []
    for p in packages:
        if method == "isotropic":
            values.append(float((p["truth"] - p["raw"]).square().mean()))
        else:
            values.append(InnovationPrior(p["raw"], p["ops"], sigma2).normalized_innovation_mse(p["truth"]))
    return max(float(np.mean(values)), 1e-30)


def _map(package, method, alpha, sigma2, lam, args):
    prior = _prior(package, method, alpha, sigma2)
    y = prior.mean.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([y], lr=args.map_lr, max_iter=args.map_steps, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                            tolerance_change=1e-12)
    calls = [0]

    def closure():
        opt.zero_grad(set_to_none=True)
        prior_value, prior_grad = prior.value_grad(y)
        _, terms = package["residual"].value(y)
        loss = y.new_tensor(prior_value) + lam * (terms["physical"] + terms["divergence"])
        loss.backward()
        with torch.no_grad():
            y.grad.add_(prior_grad)
        calls[0] += 1
        return loss

    opt.step(closure)
    with torch.no_grad():
        _, info = package["residual"].value(y)
    return y.detach(), {"function_evals": calls[0],
                        "residual_rms": float(info["residual"].square().mean().sqrt()),
                        "physical": float(info["physical"])}


def _evaluate(packages, method, alpha, sigma2, lam, args):
    rows = []
    for p in packages:
        y, info = _map(p, method, alpha, sigma2, lam, args)
        raw, truth = p["raw"], p["truth"]
        corr, error = y - raw, truth - raw
        cosine = float((corr.reshape(-1) @ error.reshape(-1)) /
                       (corr.norm() * error.norm()).clamp_min(1e-30))
        rows.append((float((y - truth).square().mean().sqrt()),
                     float((raw - truth).square().mean().sqrt()),
                     float(corr.square().mean().sqrt()), cosine,
                     info["residual_rms"], info["physical"], info["function_evals"]))
    a = np.asarray(rows)
    return {"rmse": float(a[:, 0].mean()), "raw_rmse": float(a[:, 1].mean()),
            "correction_rms": float(a[:, 2].mean()), "correction_error_cosine": float(a[:, 3].mean()),
            "residual_rms": float(a[:, 4].mean()), "physical": float(a[:, 5].mean()),
            "function_evals": float(a[:, 6].mean())}


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=16)
    ap.add_argument("--n-val-traj", type=int, default=8)
    ap.add_argument("--n-test-traj", type=int, default=16)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--residual-scheme", choices=SCHEMES, default="trapezoid")
    ap.add_argument("--lambda-grid", nargs="+", type=float, default=[0, 1e-5, 1e-4, 1e-3, 1e-2, .1, 1])
    ap.add_argument("--map-steps", type=int, default=50)
    ap.add_argument("--map-lr", type=float, default=.5)
    ap.add_argument("--divergence-weight", type=float, default=0.0)
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.lead_steps != 1:
        raise SystemExit("actual-output residual requires --lead-steps=1")
    set_seed(args.seed)
    args.fm, args.fm_channels = "poseidon", "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = _load(args.fm_data_path, args.n_cal_traj, args.steps, 0, fm.device)
    val = _load(args.fm_data_path, args.n_val_traj, args.steps, args.n_cal_traj, fm.device)
    test = _load(args.fm_data_path, args.n_test_traj, args.steps,
                 args.n_cal_traj + args.n_val_traj, fm.device)
    xs = [cal[i, 2*t] for i in range(len(cal)) for t in range(args.steps)]
    ys = [cal[i, 2*t + 2] for i in range(len(cal)) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]
    scale = _fit_residual_scale(cal, spec, args, fm.device)

    print_header(f"S4 actual-output PDE residual HILP: {fm.info.name}, horizon={args.steps}")
    print(f"  physics: {args.residual_scheme}; all residual states are actual FM/corrected endpoints")
    print("  target: literal r_PDE=0; no latent midpoint, bridge, bias subtraction, or endpoint flow solve")
    print(f"  split: calibration={len(cal)}, validation={len(val)}, held-out={len(test)}")
    print(f"  calibration: q={sigma2:.5g}; residual scale={scale:.5g}")
    print("\n  building calibration packages")
    cal_p = _raw_package(fm, cal, spec, sigma2, args.steps, scale, args)
    print("\n  building validation packages")
    val_p = _raw_package(fm, val, spec, sigma2, args.steps, scale, args)
    alphas = {m: _fit_alpha(cal_p, m, sigma2) for m in args.methods}
    for m, alpha in alphas.items():
        print(f"    {m:18s} alpha={alpha:.5g}")
    selected, curves = {}, {}
    for method in args.methods:
        print(f"\n  selecting lambda for {method} on validation")
        curve = []
        for lam in args.lambda_grid:
            out = _evaluate(val_p, method, alphas[method], sigma2, lam, args)
            curve.append({"lambda": lam, **out})
            print(f"    lambda={lam:g}: endpoint RMSE={out['rmse']:.6g}; "
                  f"corr/error cosine={out['correction_error_cosine']:+.4f}", flush=True)
        selected[method] = min(curve, key=lambda row: row["rmse"])["lambda"]
        curves[method] = curve
        print(f"    selected lambda={selected[method]:g}")
    print("\n  building held-out packages")
    test_p = _raw_package(fm, test, spec, sigma2, args.steps, scale, args)
    raw = _evaluate(test_p, "isotropic", 1.0, sigma2, 0., args)
    table = Table("method", "RMSE", "gain%", "corr RMS", "corr/error cosine", "resid RMS", "LBFGS evals")
    table.add("raw", raw["rmse"], 0., 0., 0., raw["residual_rms"], 0.)
    held = {"raw": raw}
    for method in args.methods:
        out = _evaluate(test_p, method, alphas[method], sigma2, selected[method], args)
        held[method] = out
        table.add(method, out["rmse"], 100 * (1 - out["rmse"] / raw["rmse"]),
                  out["correction_rms"], out["correction_error_cosine"],
                  out["residual_rms"], out["function_evals"])
    print("\n  held-out actual-output correction evaluation")
    print(table)
    output = {"stage": "s4_actual_output_residual", "metadata": fm_metadata(args, fm, spec),
              "physics": {"scheme": args.residual_scheme, "target": "zero", "scale": scale},
              "calibration": {"sigma2": sigma2, "alphas": alphas},
              "validation": {"curves": curves, "selected_lambda": selected}, "held_out": held}
    path = save_json(output, results_path_scale("s4_actual_output_residual", fm_result_key(args),
                                                 "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
