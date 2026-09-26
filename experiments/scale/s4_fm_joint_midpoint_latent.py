#!/usr/bin/env python3
"""Joint endpoint/midpoint MAP HILP with the original strong PDE residual.

For each native Poseidon interval, optimize endpoint y_t and latent midpoint
z_t under

  prior(y) + ||z-H(y_{t-1},y_t)||^2/(2 s_z^2)
  + lambda/2 || [curl(y_t)-curl(y_{t-1})]/dt - F(curl(z_t)) ||^2/R_cal.

H is a PDE-Hermite algebraic bridge, not a numerical flow step.  Calibration
truth estimates only R_cal, bridge variance, and prior scale.  Future
truth is never accessed during validation or held-out correction.
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


def _load(path: str, n_traj: int, steps: int, offset: int, device) -> torch.Tensor:
    need = 2 * steps + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        a = np.asarray(h[key][offset:offset + n_traj, :need, :2], dtype=np.float32)
    return torch.as_tensor(a, device=device).reshape(n_traj, need, -1)


def _velocity_rhs(operator: FMPhysicsEnergy2D, state: torch.Tensor) -> torch.Tensor:
    flat = state.reshape(-1, operator.N).to(operator.device, torch.float64)
    rw = operator.rhs_vorticity(operator.to_vorticity(flat))
    du, dv = operator.grid.velocity(torch.fft.rfft2(rw))
    return torch.stack((du, dv), dim=1).reshape_as(flat)


def _hermite(operator: FMPhysicsEnergy2D, left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    # Cubic endpoint Hermite bridge at a=1/2.  RHS evaluations only.
    return .5 * (left + right) + .125 * dt * (_velocity_rhs(operator, left) - _velocity_rhs(operator, right))


class JointStrongResidual:
    def __init__(self, spec, x0, steps, bias, scale, bridge_variance, divergence_weight, device):
        self.steps, self.dt = int(steps), 2 * POSEIDON_RAW_DT
        self.operator = FMPhysicsEnergy2D(spec, x0, 2, divergence_weight=0.0, dt=self.dt, device=device)
        self.x0 = x0.detach().double().reshape(1, -1).to(device)
        self.N = self.x0.shape[1]
        # r_PDE is one scalar vorticity field per time interval, whereas the
        # endpoint state has two velocity components.  Keep these dimensions
        # distinct when turning a mean-square residual into its squared norm.
        self.residual_dof = spec.n * spec.n
        self.bias = bias.detach().to(device, torch.float64)
        self.scale = float(scale)
        self.bridge_variance = float(bridge_variance)
        self.divergence_weight = float(divergence_weight)

    def bridge(self, endpoints: torch.Tensor) -> torch.Tensor:
        ys = endpoints.reshape(self.steps, self.N)
        states = torch.cat((self.x0, ys), dim=0)
        return _hermite(self.operator, states[:-1], states[1:], self.dt)

    def residuals(self, endpoints: torch.Tensor, midpoints: torch.Tensor) -> torch.Tensor:
        ys, zs = endpoints.reshape(self.steps, self.N), midpoints.reshape(self.steps, self.N)
        states = torch.cat((self.x0, ys), dim=0)
        w = self.operator.to_vorticity(states)
        wz = self.operator.to_vorticity(zs)
        return (w[1:] - w[:-1]) / self.dt - self.operator.rhs_vorticity(wz) - self.bias

    def value(self, endpoints: torch.Tensor, midpoints: torch.Tensor) -> tuple[torch.Tensor, dict]:
        z = midpoints.reshape(self.steps, self.N)
        target = self.bridge(endpoints)
        bridge = .5 * (z - target).square().sum() / self.bridge_variance
        r = self.residuals(endpoints, z)
        # ||r_PDE||^2 / R_cal with R_cal = scale^2 I in vorticity space.
        physical = .5 * self.residual_dof * r.square().mean() / (self.scale ** 2)
        ys = endpoints.reshape(self.steps, self.N)
        divergence = .5 * self.N * self.divergence_weight * self.operator.divergence(ys).square().mean()
        return bridge + physical + divergence, {"bridge": bridge, "physical": physical,
                                                 "divergence": divergence, "residual": r}


def _raw_package(fm, fine, spec, sigma2, steps, bias, scale, bridge_var, args):
    packages = []
    for i, truth in enumerate(fine):
        print(f"    preparing trajectory {i + 1}/{len(fine)}", flush=True)
        raw, ops = _raw_window_and_ops(fm, truth[0], steps)
        residual = JointStrongResidual(spec, truth[0], steps, bias, scale, bridge_var,
                                       args.divergence_weight, fm.device)
        packages.append({"raw": raw, "mid_truth": truth[1::2].double(),
                         "truth": truth[2::2].double(), "ops": ops, "residual": residual})
    return packages


def _fit_scales(fm, fine, spec, args):
    truth_defects, bridge_errors = [], []
    for trajectory in fine:
        x0 = trajectory[0]
        op = FMPhysicsEnergy2D(spec, x0, 2, divergence_weight=0.0, dt=2 * POSEIDON_RAW_DT, device=fm.device)
        states = trajectory[::2].double()
        mids = trajectory[1::2].double()
        with torch.no_grad():
            w = op.to_vorticity(states)
            wz = op.to_vorticity(mids)
            truth_defects.append((w[1:] - w[:-1]) / (2 * POSEIDON_RAW_DT) - op.rhs_vorticity(wz))
            bridge_errors.append(mids - _hermite(op, states[:-1], states[1:], 2 * POSEIDON_RAW_DT))
    defects = torch.cat(truth_defects)
    calibration_mean = defects.mean(0)
    # The primary HILP likelihood is rooted at r_PDE=0.  Calibration supplies
    # a unit scale only; it must not shift the root to an average residual.
    # The old centred variant remains available strictly as a diagnostic.
    bias = calibration_mean if args.residual_bias == "calibration_mean" else torch.zeros_like(calibration_mean)
    scale = float((defects - bias).square().mean().sqrt().clamp(min=1e-30))
    bridge_var = float(torch.cat(bridge_errors).square().mean().clamp(min=1e-30))
    return bias.detach(), scale, bridge_var


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
    z = package["residual"].bridge(y).detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([y, z], lr=args.map_lr, max_iter=args.map_steps,
                            history_size=10, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-8, tolerance_change=1e-11)
    calls = [0]
    def closure():
        opt.zero_grad(set_to_none=True)
        prior_value, prior_grad = prior.value_grad(y)
        # The midpoint bridge is part of the prior, not part of the physics
        # likelihood.  In particular it must remain active at lambda=0.
        _, terms = package["residual"].value(y, z)
        loss = (y.new_tensor(prior_value) + terms["bridge"] +
                lam * (terms["physical"] + terms["divergence"]))
        loss.backward()
        with torch.no_grad():
            y.grad.add_(prior_grad)
        calls[0] += 1
        return loss
    opt.step(closure)
    with torch.no_grad():
        _, info = package["residual"].value(y, z)
    return y.detach(), z.detach(), {"function_evals": calls[0],
                                     "residual_rms": float(info["residual"].square().mean().sqrt().detach()),
                                     "bridge_rms": float((z - package["residual"].bridge(y)).square().mean().sqrt().detach())}


def _evaluate(packages, method, alpha, sigma2, lam, args):
    rows = []
    for p in packages:
        y, z, info = _map(p, method, alpha, sigma2, lam, args)
        truth, raw = p["truth"], p["raw"]
        rows.append((float((y - truth).square().mean().sqrt()),
                     float((raw - truth).square().mean().sqrt()),
                     float((y - raw).square().mean().sqrt()), info["residual_rms"],
                     info["bridge_rms"], info["function_evals"]))
    a = np.asarray(rows)
    return {"rmse": float(a[:, 0].mean()), "raw_rmse": float(a[:, 1].mean()),
            "correction_rms": float(a[:, 2].mean()), "residual_rms": float(a[:, 3].mean()),
            "bridge_deviation_rms": float(a[:, 4].mean()), "function_evals": float(a[:, 5].mean())}


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=16)
    ap.add_argument("--n-val-traj", type=int, default=8)
    ap.add_argument("--n-test-traj", type=int, default=16)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--lambda-grid", nargs="+", type=float, default=[0, 1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--map-steps", type=int, default=30)
    ap.add_argument("--map-lr", type=float, default=.5)
    ap.add_argument("--divergence-weight", type=float, default=0.0)
    ap.add_argument("--residual-bias", choices=("zero", "calibration_mean"), default="zero",
                    help="Residual target. zero is the primary PDE residual; calibration_mean is diagnostic only.")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.lead_steps != 1 or args.steps < 1:
        raise SystemExit("joint midpoint MAP requires lead-steps=1 and steps >= 1")
    set_seed(args.seed)
    args.fm, args.fm_channels = "poseidon", "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = _load(args.fm_data_path, args.n_cal_traj, args.steps, 0, fm.device)
    val = _load(args.fm_data_path, args.n_val_traj, args.steps, args.n_cal_traj, fm.device)
    test = _load(args.fm_data_path, args.n_test_traj, args.steps, args.n_cal_traj + args.n_val_traj, fm.device)
    xs = [cal[i, 2*t] for i in range(len(cal)) for t in range(args.steps)]
    ys = [cal[i, 2*t+2] for i in range(len(cal)) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]
    bias, scale, bridge_var = _fit_scales(fm, cal, spec, args)
    print_header(f"S4 joint latent-midpoint residual HILP: {fm.info.name}, horizon={args.steps}")
    print("  physics: full-resolution original residual (omega_end-omega_start)/0.1 - F(omega_mid)")
    print("  inference variables: corrected endpoints plus latent midpoints; no flow endpoint solve")
    print(f"  calibration: q={sigma2:.5g}; residual target={args.residual_bias}; "
          f"residual scale={scale:.5g}; bridge variance={bridge_var:.5g}")
    print("\n  building calibration packages")
    cal_p = _raw_package(fm, cal, spec, sigma2, args.steps, bias, scale, bridge_var, args)
    print("\n  building validation packages")
    val_p = _raw_package(fm, val, spec, sigma2, args.steps, bias, scale, bridge_var, args)
    alphas = {m: _fit_alpha(cal_p, m, sigma2) for m in args.methods}
    for m, alpha in alphas.items(): print(f"    {m:18s} alpha={alpha:.5g}")
    selected, curves = {}, {}
    for m in args.methods:
        print(f"\n  selecting lambda for {m} on validation")
        curve = []
        for lam in args.lambda_grid:
            result = _evaluate(val_p, m, alphas[m], sigma2, lam, args)
            curve.append({"lambda": lam, **result})
            print(f"    lambda={lam:g}: endpoint RMSE={result['rmse']:.6g}", flush=True)
        selected[m] = min(curve, key=lambda x: x["rmse"])["lambda"]
        curves[m] = curve
        print(f"    selected lambda={selected[m]:g}")
    print("\n  building held-out packages")
    test_p = _raw_package(fm, test, spec, sigma2, args.steps, bias, scale, bridge_var, args)
    raw = _evaluate(test_p, "isotropic", 1.0, sigma2, 0.0, args)
    table = Table("method", "RMSE", "gain%", "corr RMS", "resid RMS", "bridge dev", "LBFGS evals")
    table.add("raw", raw["rmse"], 0.0, 0.0, raw["residual_rms"], 0.0, 0.0)
    held = {"raw": raw}
    for m in args.methods:
        result = _evaluate(test_p, m, alphas[m], sigma2, selected[m], args)
        held[m] = result
        table.add(m, result["rmse"], 100*(1-result["rmse"]/raw["rmse"]), result["correction_rms"],
                  result["residual_rms"], result["bridge_deviation_rms"], result["function_evals"])
    print("\n  held-out joint latent-midpoint evaluation")
    print(table)
    output = {"stage": "s4_joint_latent_midpoint", "metadata": fm_metadata(args, fm, spec),
              "physics": "strong full-resolution midpoint residual with optimized latent midpoint",
              "inference": "truth-free; endpoints and midpoints jointly MAP optimized",
              "calibration": {"sigma2": sigma2, "residual_target": args.residual_bias,
                              "residual_bias_rms": float(bias.square().mean().sqrt()),
                              "residual_scale": scale, "bridge_variance": bridge_var, "alphas": alphas},
              "validation": {"curves": curves, "selected_lambda": selected}, "held_out": held}
    path = save_json(output, results_path_scale("s4_joint_latent_midpoint", fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
