"""Shared experiment scaffolding: argument parsing, model/data loading,
calibration-set construction. Every stage script starts by calling
`setup(...)` so the stages are directly comparable.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .calibrate import fit_alpha
from .curvature import ESTIMATORS, estimate
from .data import SPECS, build_dataset, pairs
from .train import load_model
from .utils import get_device, set_seed


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--pde", default="burgers", choices=list(SPECS))
    ap.add_argument("--n-test", type=int, default=64, help="test states to evaluate")
    ap.add_argument("--n-cal", type=int, default=192,
                    help="states used to fit alpha. Its MLE is a mean of a "
                         "heavy-tailed quantity, so too few states makes it noisy.")
    ap.add_argument("--n-fit", type=int, default=512,
                    help="states used to fit baseline statistics (residual "
                         "covariance, Laplace). Must exceed N or the fitted "
                         "covariance is rank-deficient and its inverse is junk.")
    ap.add_argument("--tau-rel", type=float, default=1e-2, help="relative damping")
    ap.add_argument("--n-cal-baseline", type=int, default=64,
                    help="calibration states for the sampling-based baselines. "
                         "They cost O(n_samples) or O(N) network passes per "
                         "state, so they get a smaller split than the curvature "
                         "estimators; alpha is one scalar and converges quickly.")
    ap.add_argument("--alpha-objective", default="mle", choices=["mle", "median"],
                    help="'mle' is the closed-form Gaussian MLE (a mean, and so "
                         "sensitive to the heavy tail of per-state error energy); "
                         "'median' matches median D^2 to the chi^2_N median.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-plots", action="store_true")
    return ap


def subsample(x: torch.Tensor, y: torch.Tensor, n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:min(n, x.shape[0])]
    return x[idx], y[idx]


def setup(args, need_ensemble: bool = False, need_dropout: bool = False):
    """Load data + frozen model(s) and carve out calibration / test states."""
    set_seed(args.seed)
    device = get_device(args.device)
    ds = build_dataset(args.pde)
    spec = ds["spec"]
    model = load_model(args.pde, member=0, device=device)

    x_cal, y_cal = pairs(ds["val"], device=device)
    x_test, y_test = pairs(ds["test"], device=device)
    n_fit = getattr(args, "n_fit", 512)
    x_fit, y_fit = subsample(x_cal, y_cal, n_fit, seed=args.seed + 2)
    x_cal, y_cal = subsample(x_cal, y_cal, args.n_cal, seed=args.seed)
    x_test, y_test = subsample(x_test, y_test, args.n_test, seed=args.seed + 1)

    ctx = {"model": model, "spec": spec, "device": device, "pde": args.pde,
           "x_cal": x_cal, "y_cal": y_cal, "x_test": x_test, "y_test": y_test,
           "x_fit": x_fit, "y_fit": y_fit, "dataset": ds}
    if need_ensemble:
        members = []
        for m in range(8):
            try:
                members.append(load_model(args.pde, member=m, device=device))
            except FileNotFoundError:
                break
        ctx["ensemble"] = members
    if need_dropout:
        try:
            ctx["model_do"] = load_model(args.pde, member=0, dropout=0.1, device=device)
        except FileNotFoundError:
            ctx["model_do"] = None
    return ctx


def jacobian_cache(model, xs: torch.Tensor) -> list:
    """Exact Jacobians for every state, computed once and shared across
    estimators (they are by far the dominant cost)."""
    from .curvature import exact_jacobian
    return [exact_jacobian(model, xs[i]) for i in range(xs.shape[0])]


def curvature_priors(model, xs: torch.Tensor, method: str, tau_rel: float,
                     jac: list | None = None, **kw) -> list:
    """Unit-scale priors (alpha = 1) for every state in `xs`."""
    out = []
    for i in range(xs.shape[0]):
        est = estimate(model, xs[i], method=method, tau_rel=tau_rel,
                       J_exact=(jac[i] if jac is not None else None), **kw)
        out.append(est.prior(alpha=1.0))
    return out


def calibrated_curvature_priors(model, x_cal, y_cal, x_test, method: str,
                                tau_rel: float, **kw):
    """Fit alpha on the calibration split, apply it to the test split."""
    cal = curvature_priors(model, x_cal, method, tau_rel, **kw)
    alpha = fit_alpha(cal, y_cal)
    test = curvature_priors(model, x_test, method, tau_rel, **kw)
    return [p.rescaled(alpha) for p in test], alpha


def calibrate_baseline(build_fn, x_cal, y_cal, x_test, objective: str = "mle"):
    """Same one-parameter calibration, applied to a baseline."""
    cal = [build_fn(x_cal[i]) for i in range(x_cal.shape[0])]
    unit = [p.rescaled(1.0) for p in cal]
    alpha = fit_alpha(unit, y_cal, objective=objective)
    test = [build_fn(x_test[i]).rescaled(alpha) for i in range(x_test.shape[0])]
    return test, alpha


DEFAULT_ESTIMATORS = ["gn", "pushforward", "dircov", "diag_gn",
                      "lowrank_pushforward", "identity"]


def estimator_choices() -> list[str]:
    return list(ESTIMATORS)


def build_method_zoo(args, ctx, estimators=None, baselines=None,
                     verbose: bool = True) -> dict:
    """Build calibrated priors on the test split for every method under test.

    Returns {name: {"priors": [...], "alpha": float, "kind": "curvature"|"baseline"}}.
    Every entry has been through the identical one-parameter calibration, so
    differences between entries are differences in covariance *shape* only.
    """
    from .baselines import BASELINES, build_baseline_fn, fit_laplace, fit_residual_stats
    from .train import load_model

    estimators = DEFAULT_ESTIMATORS if estimators is None else estimators
    baselines = BASELINES if baselines is None else baselines
    model = ctx["model"]
    x_cal, y_cal, x_test = ctx["x_cal"], ctx["y_cal"], ctx["x_test"]
    zoo = {}

    if verbose:
        print(f"  precomputing exact Jacobians "
              f"({x_cal.shape[0]} cal + {x_test.shape[0]} test states) ...", flush=True)
    jac_cal = jacobian_cache(model, x_cal)
    jac_test = jacobian_cache(model, x_test)

    for meth in estimators:
        if verbose:
            print(f"  curvature: {meth} ...", end="", flush=True)
        cal = curvature_priors(model, x_cal, meth, args.tau_rel, jac=jac_cal)
        alpha, diag = fit_alpha(cal, y_cal, return_diagnostics=True,
                                objective=getattr(args, "alpha_objective", "mle"))
        priors = [p.rescaled(alpha) for p in
                  curvature_priors(model, x_test, meth, args.tau_rel, jac=jac_test)]
        if verbose:
            print(f" alpha={alpha:.4g} (bootstrap rel.SE {diag['rel_se']:.2f}, "
                  f"tail {diag['tail_ratio']:.2f})", flush=True)
        zoo[meth] = {"priors": priors, "alpha": alpha, "kind": "curvature",
                     "alpha_diag": diag}

    if not baselines:
        return zoo

    # Baseline statistics are fitted on the larger x_fit split: a 128x128
    # residual covariance estimated from 64 samples is rank-deficient, and its
    # inverse is dominated by the shrinkage floor rather than by the data.
    x_fit, y_fit = ctx.get("x_fit", x_cal), ctx.get("y_fit", y_cal)
    if verbose:
        print(f"  (baseline statistics fitted on {x_fit.shape[0]} states, "
              f"alpha on {x_cal.shape[0]})")
    bctx = {"model": model, "res_stats": fit_residual_stats(model, x_fit, y_fit)}
    if "laplace_lastlayer" in baselines:
        bctx["laplace"] = fit_laplace(model, x_fit[:16], y_fit[:16])
    if "deep_ensemble" in baselines:
        bctx["ensemble"] = ctx.get("ensemble") or []
        if len(bctx["ensemble"]) < 2:
            baselines = [b for b in baselines if b != "deep_ensemble"]
    if "mc_dropout" in baselines:
        bctx["model_do"] = ctx.get("model_do")
        if bctx["model_do"] is None:
            baselines = [b for b in baselines if b != "mc_dropout"]
    if "swag" in baselines:
        try:
            _, payload = load_model(args.pde, member=0, device=ctx["device"],
                                    with_payload=True)
            if "swag" in payload:
                bctx["swag"] = payload["swag"]
            else:
                baselines = [b for b in baselines if b != "swag"]
        except FileNotFoundError:
            baselines = [b for b in baselines if b != "swag"]

    # MC-dropout / SWAG / Laplace cost O(n_samples) or O(N) network passes per
    # state, so calibrating them on the full x_cal split dominates runtime for
    # no benefit -- alpha is a single scalar.
    n_cb = min(getattr(args, "n_cal_baseline", 64), x_cal.shape[0])
    x_cb, y_cb = x_cal[:n_cb], y_cal[:n_cb]
    for name in baselines:
        if verbose:
            print(f"  baseline:  {name} ...", end="", flush=True)
        fn = build_baseline_fn(name, bctx)
        cheap = name in ("isotropic", "diagonal_fitted", "fixed_global")
        xs, ys = (x_cal, y_cal) if cheap else (x_cb, y_cb)
        priors, alpha = calibrate_baseline(fn, xs, ys, x_test,
                                           getattr(args, "alpha_objective", "mle"))
        if verbose:
            print(f" alpha={alpha:.4g} (on {xs.shape[0]} states)", flush=True)
        zoo[name] = {"priors": priors, "alpha": alpha, "kind": "baseline"}
    return zoo
