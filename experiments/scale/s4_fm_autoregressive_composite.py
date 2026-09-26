#!/usr/bin/env python3
"""Clean S4: nonlinear autoregressive FM prior plus zero-rooted PDE residual.

This implementation is intentionally independent of the earlier S4 paths.
For each window, the incoming state x_0 is fixed.  The only optimization
variables are future endpoints y_1,...,y_H.  The HILP objective is

  E(y) = E_FM(y) + lambda * E_PDE(y),

  E_FM(block) = 1/(2 alpha q) sum_j || y_j - F(y_{j-1}) ||^2,
  E_PDE        = 1/(2 R_cal) sum_j || R_h(y_{j-1}, y_j) ||^2,

with y_0=x_0 exactly.  Crucially, F(y_{j-1}) is recomputed inside every
optimizer closure, so a correction at one step affects all later FM means.
No frozen Jacobians, residual bias, latent midpoint, bridge penalty, oracle
endpoint, flow integration, or learned discrepancy model is used.

``isotropic`` is the trace-unstructured control: it uses the same raw window
but a diagonal endpoint fidelity term.  ``block_innovation`` is the nonlinear
autoregressive innovation likelihood above.
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
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed

METHODS = ("isotropic", "block_innovation")
SCHEMES = ("backward_euler", "trapezoid", "centered")


def _load(path: str, n_traj: int, steps: int, offset: int, device) -> torch.Tensor:
    need = 2 * int(steps) + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        data = np.asarray(h[key][offset:offset + n_traj, :need, :2], dtype=np.float32)
    return torch.as_tensor(data, device=device).reshape(n_traj, need, -1)


def _raw_rollout(fm, x0: torch.Tensor, steps: int) -> torch.Tensor:
    """Raw autoregressive mean used only as initialization/control centre."""
    states, current = [], x0.to(fm.device, fm.dtype).reshape(-1)
    for _ in range(int(steps)):
        current = fm.predict(current).detach().reshape(-1)
        states.append(current.double())
    return torch.stack(states)


class EndpointResidual:
    """R_h evaluated only on the true optimization states x0,y1,...,yH."""
    def __init__(self, spec, x0, steps, scale, scheme, device):
        self.steps = int(steps)
        self.dt = 2 * POSEIDON_RAW_DT
        self.scheme = scheme
        self.x0 = x0.detach().to(device=device, dtype=torch.float64).reshape(1, -1)
        self.nstate = self.x0.shape[-1]
        self.scale = float(scale)
        self.op = FMPhysicsEnergy2D(spec, self.x0[0], 2, divergence_weight=0.0,
                                    dt=self.dt, device=device)
        if scheme == "centered" and self.steps < 2:
            raise ValueError("centered residual requires at least two future endpoints")

    def residuals(self, endpoints: torch.Tensor) -> torch.Tensor:
        y = endpoints.reshape(self.steps, self.nstate)
        states = torch.cat((self.x0, y), dim=0)
        omega = self.op.to_vorticity(states)
        rhs = self.op.rhs_vorticity(omega)
        if self.scheme == "backward_euler":
            return (omega[1:] - omega[:-1]) / self.dt - rhs[1:]
        if self.scheme == "trapezoid":
            return (omega[1:] - omega[:-1]) / self.dt - .5 * (rhs[1:] + rhs[:-1])
        return (omega[2:] - omega[:-2]) / (2 * self.dt) - rhs[1:-1]

    def value(self, endpoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.residuals(endpoints)
        return .5 * residual.square().sum() / (self.scale ** 2), residual


class CompositeWindow:
    """Exact nonlinear window objective; all FM dependencies remain in graph."""
    def __init__(self, fm, x0, raw, residual, q, alpha, method):
        self.fm, self.method = fm, method
        self.x0 = x0.detach().to(fm.device, fm.dtype).reshape(-1)
        self.raw = raw.detach().double()
        self.residual = residual
        self.q, self.alpha = float(q), float(alpha)
        if self.q <= 0 or self.alpha <= 0:
            raise ValueError("q and alpha must be positive")

    def nonlinear_innovations(self, endpoints: torch.Tensor) -> torch.Tensor:
        """y_j-F(y_{j-1}), with a differentiable re-evaluation of F at every j."""
        y = endpoints.reshape_as(self.raw)
        previous = self.x0
        innovation = []
        fn = self.fm.flat_fn()
        for j in range(y.shape[0]):
            mean = fn(previous).reshape(-1)
            innovation.append(y[j] - mean.double())
            # This is deliberately y[j], not a stored raw output.
            previous = y[j].to(self.fm.dtype)
        return torch.stack(innovation)

    def prior(self, endpoints: torch.Tensor) -> torch.Tensor:
        if self.method == "isotropic":
            return .5 * (endpoints.reshape_as(self.raw) - self.raw).square().sum() / self.alpha
        innovation = self.nonlinear_innovations(endpoints)
        return .5 * innovation.square().sum() / (self.alpha * self.q)

    def objective(self, endpoints: torch.Tensor, lam: float) -> tuple[torch.Tensor, dict]:
        prior = self.prior(endpoints)
        physics, residual = self.residual.value(endpoints)
        return prior + float(lam) * physics, {"prior": prior, "physics": physics, "residual": residual}


def _residual_scale(trajectories, spec, args, device) -> float:
    values = []
    for trajectory in trajectories:
        evaluator = EndpointResidual(spec, trajectory[0], args.steps, 1.0,
                                     args.residual_scheme, device)
        with torch.no_grad():
            values.append(evaluator.residuals(trajectory[2::2].double()))
    return float(torch.cat(values).square().mean().sqrt().clamp(min=1e-30))


def _fit_alphas(fm, packages, q):
    iso, innovation = [], []
    with torch.no_grad():
        fn = fm.flat_fn()
        for package in packages:
            truth, raw = package["truth"], package["raw"]
            iso.append(float((truth - raw).square().mean()))
            previous, errors = package["x0"].to(fm.dtype), []
            for target in truth:
                errors.append((target - fn(previous).double()).square().mean())
                previous = target.to(fm.dtype)
            innovation.append(float(torch.stack(errors).mean() / q))
    return {"isotropic": max(float(np.mean(iso)), 1e-30),
            "block_innovation": max(float(np.mean(innovation)), 1e-30)}


def _packages(fm, trajectories, spec, q, scale, args, label):
    result = []
    for i, trajectory in enumerate(trajectories):
        print(f"    preparing {label} trajectory {i + 1}/{len(trajectories)}", flush=True)
        x0 = trajectory[0]
        result.append({"x0": x0.detach(), "raw": _raw_rollout(fm, x0, args.steps),
                       "truth": trajectory[2::2].double(),
                       "residual": EndpointResidual(spec, x0, args.steps, scale,
                                                    args.residual_scheme, fm.device)})
    return result


def _map(package, fm, method, alpha, q, lam, args):
    objective = CompositeWindow(fm, package["x0"], package["raw"], package["residual"],
                                q, alpha, method)
    y = package["raw"].detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([y], lr=args.map_lr, max_iter=args.map_steps,
                                  history_size=args.lbfgs_history, line_search_fn="strong_wolfe",
                                  tolerance_grad=args.map_grad_tol,
                                  tolerance_change=args.map_change_tol)
    evaluations = [0]

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective.objective(y, lam)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite composite objective")
        loss.backward()
        evaluations[0] += 1
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        _, info = objective.objective(y, lam)
        residual_rms = float(info["residual"].square().mean().sqrt())
        prior_value, physical = float(info["prior"]), float(info["physics"])
    return y.detach(), {"function_evals": evaluations[0], "residual_rms": residual_rms,
                        "prior": prior_value, "physics": physical}


def _evaluate(packages, fm, method, alpha, q, lam, args):
    rows = []
    for package in packages:
        corrected, info = _map(package, fm, method, alpha, q, lam, args)
        raw, truth = package["raw"], package["truth"]
        correction, error = corrected - raw, truth - raw
        cosine = float((correction.reshape(-1) @ error.reshape(-1)) /
                       (correction.norm() * error.norm()).clamp_min(1e-30))
        step_rmse = (corrected - truth).square().mean(dim=1).sqrt().detach().cpu().numpy()
        rows.append({"rmse": float((corrected - truth).square().mean().sqrt()),
                     "raw_rmse": float((raw - truth).square().mean().sqrt()),
                     "correction_rms": float(correction.square().mean().sqrt()),
                     "correction_error_cosine": cosine, "step_rmse": step_rmse,
                     **info})
    return {"rmse": float(np.mean([r["rmse"] for r in rows])),
            "raw_rmse": float(np.mean([r["raw_rmse"] for r in rows])),
            "correction_rms": float(np.mean([r["correction_rms"] for r in rows])),
            "correction_error_cosine": float(np.mean([r["correction_error_cosine"] for r in rows])),
            "residual_rms": float(np.mean([r["residual_rms"] for r in rows])),
            "prior": float(np.mean([r["prior"] for r in rows])),
            "physics": float(np.mean([r["physics"] for r in rows])),
            "function_evals": float(np.mean([r["function_evals"] for r in rows])),
            "step_rmse": np.mean(np.stack([r["step_rmse"] for r in rows]), axis=0).tolist()}


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=16)
    ap.add_argument("--n-val-traj", type=int, default=8)
    ap.add_argument("--n-test-traj", type=int, default=16)
    ap.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    ap.add_argument("--residual-scheme", choices=SCHEMES, default="trapezoid")
    ap.add_argument("--lambda-grid", type=float, nargs="+",
                    default=[0, 1e-5, 1e-4, 1e-3, 1e-2, .1, 1.])
    ap.add_argument("--map-steps", type=int, default=40)
    ap.add_argument("--map-lr", type=float, default=.5)
    ap.add_argument("--lbfgs-history", type=int, default=20)
    ap.add_argument("--map-grad-tol", type=float, default=1e-8)
    ap.add_argument("--map-change-tol", type=float, default=1e-11)
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.lead_steps != 1:
        raise SystemExit("this implementation requires Poseidon's native --lead-steps=1")
    set_seed(args.seed)
    args.fm, args.fm_channels = "poseidon", "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = _load(args.fm_data_path, args.n_cal_traj, args.steps, 0, fm.device)
    val = _load(args.fm_data_path, args.n_val_traj, args.steps, args.n_cal_traj, fm.device)
    test = _load(args.fm_data_path, args.n_test_traj, args.steps,
                 args.n_cal_traj + args.n_val_traj, fm.device)
    xs = [cal[i, 2 * t] for i in range(len(cal)) for t in range(args.steps)]
    ys = [cal[i, 2 * t + 2] for i in range(len(cal)) for t in range(args.steps)]
    q = fit_sigma2(fm, xs, ys)["sigma2"]
    scale = _residual_scale(cal, spec, args, fm.device)

    print_header(f"S4 nonlinear autoregressive composite HILP: {fm.info.name}, horizon={args.steps}")
    print("  objective: endpoint FM fidelity + lambda * literal zero-rooted PDE residual energy")
    print("  y0 is fixed to the supplied input; every F(y_{j-1}) is recomputed inside each MAP closure")
    print("  exclusions: no frozen Jacobian prior, residual bias, latent midpoint, bridge, discrepancy model, or flow solve")
    print(f"  residual: {args.residual_scheme}; q={q:.5g}; R_cal^1/2={scale:.5g}")
    print(f"  split: calibration={len(cal)}, validation={len(val)}, held-out={len(test)}")

    print("\n  building calibration packages")
    cal_packages = _packages(fm, cal, spec, q, scale, args, "calibration")
    alphas = _fit_alphas(fm, cal_packages, q)
    for method in args.methods:
        print(f"    {method:18s} alpha={alphas[method]:.6g}")
    print("\n  building validation packages")
    val_packages = _packages(fm, val, spec, q, scale, args, "validation")
    curves, selected = {}, {}
    for method in args.methods:
        print(f"\n  selecting lambda for {method} on validation")
        curve = []
        for lam in args.lambda_grid:
            out = _evaluate(val_packages, fm, method, alphas[method], q, lam, args)
            curve.append({"lambda": lam, **out})
            print(f"    lambda={lam:g}: RMSE={out['rmse']:.6g}; corr/error cosine="
                  f"{out['correction_error_cosine']:+.4f}; mean LBFGS evals={out['function_evals']:.1f}", flush=True)
        selected[method] = min(curve, key=lambda item: item["rmse"])["lambda"]
        curves[method] = curve
        print(f"    selected lambda={selected[method]:g}")

    print("\n  building held-out packages")
    test_packages = _packages(fm, test, spec, q, scale, args, "held-out")
    raw = _evaluate(test_packages, fm, "isotropic", 1., q, 0., args)
    table = Table("method", "RMSE", "gain%",
                  *[f"step{j + 1} gain%" for j in range(args.steps)],
                  "corr RMS", "corr/error cosine", "resid RMS", "LBFGS evals")
    table.add("raw", raw["rmse"], 0., *([0.] * args.steps), 0., 0., raw["residual_rms"], 0.)
    held = {"raw": raw}
    for method in args.methods:
        out = _evaluate(test_packages, fm, method, alphas[method], q, selected[method], args)
        held[method] = out
        gains = [100 * (1 - corrected / baseline) for corrected, baseline in zip(out["step_rmse"], raw["step_rmse"])]
        table.add(method, out["rmse"], 100 * (1 - out["rmse"] / raw["rmse"]), *gains,
                  out["correction_rms"], out["correction_error_cosine"], out["residual_rms"], out["function_evals"])
    print("\n  held-out nonlinear autoregressive composite evaluation")
    print(table)
    payload = {"stage": "s4_nonlinear_autoregressive_composite", "metadata": fm_metadata(args, fm, spec),
               "objective": {"x0": "hard fixed input", "fm": "nonlinear recomputed F(y[j-1])",
                             "physics": f"zero-rooted {args.residual_scheme} residual", "residual_scale": scale,
                             "no_bias_or_latent_or_flow_solver": True},
               "calibration": {"q": q, "alphas": alphas},
               "validation": {"curves": curves, "selected_lambda": selected}, "held_out": held}
    path = save_json(payload, results_path_scale("s4_nonlinear_autoregressive_composite",
                                                 fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
