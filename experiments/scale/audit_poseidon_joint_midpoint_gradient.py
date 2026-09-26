#!/usr/bin/env python3
"""Teacher-forced direction gate for joint latent-midpoint residual HILP.

This diagnostic does not change S4.  It asks whether the *deployable*
full-resolution residual likelihood used by ``s4_fm_joint_midpoint_latent``
has a local endpoint-update direction that points toward the held-out
forecast error.  Future truth is used only after the direction is computed.

For a raw endpoint window mu and its error e=x_truth-mu, we report the
prior-preconditioned descent direction

    d = - Sigma_prior grad_y E_phys(mu, H(mu)),

and the actual joint-MAP correction for each requested lambda.  A positive
cosine and substantial reachable fraction (cosine squared) are necessary
conditions for this observation/prior pair to improve endpoint RMSE.
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
from experiments.scale.s4_fm_joint_midpoint_latent import (_fit_alpha, _fit_scales, _map,
                                                            _prior, _raw_package, _load)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _covariance_action(prior, gradient: torch.Tensor) -> torch.Tensor:
    """Apply the covariance corresponding exactly to the S4 endpoint prior."""
    g = gradient.reshape(prior.T, prior.N).double()
    if prior.label == "isotropic":
        return prior.alpha * g

    # Sigma = alpha*q*B^{-1}B^{-T} for the scalar-innovation prior used by
    # this experiment.  First solve B^T w=g, then B v=alpha*q*w.
    w = [None] * prior.T
    w[-1] = g[-1]
    for t in range(prior.T - 2, -1, -1):
        w[t] = g[t] + prior.ops[t].rmatvec(w[t + 1].to(prior.ops[t].dtype)).double()
    v = [prior.alpha * prior.q * w[0]]
    for t in range(prior.T - 1):
        jv = prior.ops[t].matvec(v[-1].to(prior.ops[t].dtype)).double()
        v.append(prior.alpha * prior.q * w[t + 1] + jv)
    return torch.stack(v)


def _physics_descent(package, method: str, alpha: float, sigma2: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Current MAP's initial endpoint direction; latent midpoint is fixed at H(mu)."""
    prior = _prior(package, method, alpha, sigma2)
    y = prior.mean.detach().clone().requires_grad_(True)
    # In the deployed joint objective z is an independent variable.  At its
    # initialization z=H(mu), the bridge gradient is zero, so this is exactly
    # the initial endpoint gradient of the actual optimization problem.
    z = package["residual"].bridge(y).detach()
    _, terms = package["residual"].value(y, z)
    terms["physical"].backward()
    return -_covariance_action(prior, y.grad.detach()), prior.mean


def _term_ledger(package, method: str, alpha: float, sigma2: float, args,
                 lam: float | None = None) -> dict:
    """Objective scales and gradient scales at raw or after the deployed MAP."""
    prior = _prior(package, method, alpha, sigma2)
    if lam is None:
        y = prior.mean.detach().clone().requires_grad_(True)
        z = package["residual"].bridge(y).detach().clone().requires_grad_(True)
    else:
        y0, z0, _ = _map(package, method, alpha, sigma2, lam, args)
        y, z = y0.requires_grad_(True), z0.requires_grad_(True)
    prior_value, _ = prior.value_grad(y)
    _, terms = package["residual"].value(y, z)
    # Separate gradients expose whether the residual acts chiefly through the
    # endpoint or through the unconstrained latent midpoint.
    gy_phys, gz_phys = torch.autograd.grad(terms["physical"], (y, z), retain_graph=True)
    gy_bridge, gz_bridge = torch.autograd.grad(terms["bridge"], (y, z), retain_graph=True)
    return {
        "prior": float(prior_value),
        "bridge": float(terms["bridge"].detach()),
        "physical": float(terms["physical"].detach()),
        "weighted_physical": float((0.0 if lam is None else lam) * terms["physical"].detach()),
        "residual_rms": float(terms["residual"].square().mean().sqrt().detach()),
        "endpoint_phys_grad_rms": float(gy_phys.square().mean().sqrt().detach()),
        "midpoint_phys_grad_rms": float(gz_phys.square().mean().sqrt().detach()),
        "endpoint_bridge_grad_rms": float(gy_bridge.square().mean().sqrt().detach()),
        "midpoint_bridge_grad_rms": float(gz_bridge.square().mean().sqrt().detach()),
    }


def _truth_residual_ledger(package) -> dict:
    """Diagnostic only: strong residual evaluated on stored future truth/midpoints."""
    with torch.no_grad():
        _, terms = package["residual"].value(package["truth"], package["mid_truth"])
    return {"truth_residual_rms": float(terms["residual"].square().mean().sqrt()),
            "truth_physical": float(terms["physical"])}


def _direction_summary(directions: list[torch.Tensor], errors: list[torch.Tensor]) -> dict:
    d, e = torch.cat([x.reshape(-1) for x in directions]), torch.cat([x.reshape(-1) for x in errors])
    dot, d2, e2 = float(d @ e), float(d.square().sum()), float(e.square().sum())
    cosine = dot / max((d2 * e2) ** .5, 1e-30)
    # Best scalar gain is a truth-only diagnostic of the correction subspace.
    gain = max(dot, 0.0) / max(d2, 1e-30)
    reachable = max(dot, 0.0) ** 2 / max(d2 * e2, 1e-30)
    return {"cosine": cosine, "reachable_fraction": reachable,
            "oracle_scalar_gain": gain, "direction_rms": (d2 / d.numel()) ** .5}


def _map_summary(packages, method, alpha, sigma2, lam, args) -> dict:
    corrections, errors, rmses = [], [], []
    for p in packages:
        y, _, _ = _map(p, method, alpha, sigma2, lam, args)
        correction = y - p["raw"]
        error = p["truth"] - p["raw"]
        corrections.append(correction)
        errors.append(error)
        rmses.append(float((y - p["truth"]).square().mean().sqrt()))
    s = _direction_summary(corrections, errors)
    s["rmse"] = float(np.mean(rmses))
    return s


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=32)
    ap.add_argument("--n-val-traj", type=int, default=8,
                    help="Reserved trajectory-disjoint validation count; never used by this diagnostic.")
    ap.add_argument("--n-test-traj", type=int, default=16)
    ap.add_argument("--lambda-grid", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, .1, 1.])
    ap.add_argument("--map-steps", type=int, default=30)
    ap.add_argument("--map-lr", type=float, default=.5)
    ap.add_argument("--divergence-weight", type=float, default=0.)
    ap.add_argument("--residual-bias", choices=("zero", "calibration_mean"), default="zero")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.lead_steps != 1:
        raise SystemExit("this diagnostic requires --lead-steps=1")
    set_seed(args.seed)
    args.fm, args.fm_channels = "poseidon", "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = _load(args.fm_data_path, args.n_cal_traj, args.steps, 0, fm.device)
    # Preserve a disjoint validation partition even though this audit does not
    # score it.  This makes the diagnostic's test result comparable with S4.
    test_offset = args.n_cal_traj + args.n_val_traj
    test = _load(args.fm_data_path, args.n_test_traj, args.steps, test_offset, fm.device)
    xs = [cal[i, 2 * t] for i in range(len(cal)) for t in range(args.steps)]
    ys = [cal[i, 2 * t + 2] for i in range(len(cal)) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]
    bias, scale, bridge_var = _fit_scales(fm, cal, spec, args)

    print_header(f"Poseidon joint latent-midpoint gradient-alignment gate: {fm.info.name}")
    print(f"  split: calibration={len(cal)}, reserved-validation={args.n_val_traj}, "
          f"held-out={len(test)}, horizon={args.steps}")
    print("  direction: -Sigma_prior grad_y E_phys(raw, H(raw)); truth is diagnostic only")
    print(f"  physics: original full-resolution midpoint PDE residual; residual target={args.residual_bias}; "
          "no endpoint flow solve")
    print("\n  building calibration packages")
    cal_p = _raw_package(fm, cal, spec, sigma2, args.steps, bias, scale, bridge_var, args)
    alphas = {m: _fit_alpha(cal_p, m, sigma2) for m in ("isotropic", "block_innovation")}
    print(f"  calibrated scales: isotropic={alphas['isotropic']:.5g}, "
          f"block_innovation={alphas['block_innovation']:.5g}")
    print("\n  building held-out packages")
    test_p = _raw_package(fm, test, spec, sigma2, args.steps, bias, scale, bridge_var, args)

    local, output = {}, {"stage": "joint_midpoint_gradient_alignment", "metadata": fm_metadata(args, fm, spec),
                         "calibration": {"sigma2": sigma2, "residual_scale": scale,
                                         "bridge_variance": bridge_var, "alphas": alphas}, "methods": {}}
    for method in ("isotropic", "block_innovation"):
        directions, errors = [], []
        for p in test_p:
            d, raw = _physics_descent(p, method, alphas[method], sigma2)
            directions.append(d)
            errors.append(p["truth"] - raw)
        local[method] = _direction_summary(directions, errors)

    table = Table("method", "local cosine", "reachable%", "oracle scalar", "direction RMS")
    for method in ("isotropic", "block_innovation"):
        s = local[method]
        table.add(method, s["cosine"], 100 * s["reachable_fraction"], s["oracle_scalar_gain"], s["direction_rms"])
    print("\n  local likelihood-direction diagnostic")
    print(table)

    # Aggregate only like-for-like quantities across held-out packages.  This
    # is a diagnostic ledger; truth entries are never used in correction.
    truth_ledger = _truth_residual_ledger(test_p[0])
    raw_ledgers = {m: _term_ledger(test_p[0], m, alphas[m], sigma2, args) for m in ("isotropic", "block_innovation")}
    print("\n  held-out objective-scale ledger (first window; truth row is diagnostic only)")
    ledger_table = Table("state/method", "prior", "bridge", "physical", "resid RMS", "|gy phys|", "|gz phys|")
    ledger_table.add("truth diagnostic", 0., 0., truth_ledger["truth_physical"], truth_ledger["truth_residual_rms"], 0., 0.)
    for method in ("isotropic", "block_innovation"):
        s = raw_ledgers[method]
        ledger_table.add(f"raw/{method}", s["prior"], s["bridge"], s["physical"], s["residual_rms"],
                         s["endpoint_phys_grad_rms"], s["midpoint_phys_grad_rms"])
    print(ledger_table)
    output["objective_scale_ledger"] = {"truth_diagnostic": truth_ledger, "raw": raw_ledgers}

    for method in ("isotropic", "block_innovation"):
        print(f"\n  actual joint-MAP correction alignment: {method}")
        curve = []
        for lam in args.lambda_grid:
            s = _map_summary(test_p, method, alphas[method], sigma2, lam, args)
            ledger = _term_ledger(test_p[0], method, alphas[method], sigma2, args, lam)
            s["first_window_ledger"] = ledger
            curve.append({"lambda": lam, **s})
            print(f"    lambda={lam:g}: RMSE={s['rmse']:.6g}; corr/error cosine={s['cosine']:+.4f}; "
                  f"reachable={100*s['reachable_fraction']:.2f}%; prior={ledger['prior']:.3g}; "
                  f"bridge={ledger['bridge']:.3g}; lambda*phys={ledger['weighted_physical']:.3g}", flush=True)
        output["methods"][method] = {"local": local[method], "map_curve": curve}
    print("\n  Interpretation: positive local cosine is necessary; a small reachable fraction means "
          "the current residual observation cannot materially reduce endpoint RMSE under that prior.")
    path = save_json(output, results_path_scale("poseidon_joint_midpoint_gradient_alignment",
                                                  fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
