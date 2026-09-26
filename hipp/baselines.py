"""Uncertainty baselines, all returned as GaussianPrior objects so that the
same calibration and the same metrics apply to every method.

The important one is `fixed_global`: a single state-independent covariance
fitted from calibration residuals. If a curvature estimate cannot beat it, the
Jacobian is contributing no per-sample information (docs/method.md section 6).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .curvature import damp, predict
from .models import enable_dropout
from .priors import GaussianPrior


def _cov_from_samples(samples: torch.Tensor, shrink: float = 1e-2) -> torch.Tensor:
    """Shrunk sample covariance. Necessary: with T << N the raw sample
    covariance is singular and its inverse is meaningless."""
    S = samples.to(torch.float64)
    S = S - S.mean(dim=0, keepdim=True)
    C = (S.T @ S) / max(S.shape[0] - 1, 1)
    return damp(C, shrink)


# ---------------------------------------------------------------------------
# State-independent baselines
# ---------------------------------------------------------------------------

def isotropic_prior(model, c: torch.Tensor, **kw) -> GaussianPrior:
    mean = predict(model, c).to(torch.float64)
    return GaussianPrior.isotropic(mean, label="isotropic")


def fit_residual_stats(model, xs: torch.Tensor, ys: torch.Tensor,
                       shrink: float = 1e-2) -> dict:
    """Per-position variance and full covariance of calibration residuals."""
    with torch.no_grad():
        preds = model(xs)
    res = (ys - preds).to(torch.float64)
    return {
        "var": res.var(dim=0).clamp(min=1e-30),
        "cov": damp((res.T @ res) / max(res.shape[0] - 1, 1), shrink),
        "residuals": res,
    }


def diagonal_prior(model, c: torch.Tensor, stats: dict, **kw) -> GaussianPrior:
    mean = predict(model, c).to(torch.float64)
    return GaussianPrior.diagonal(mean, stats["var"], label="diagonal_fitted")


def fixed_global_prior(model, c: torch.Tensor, stats: dict, **kw) -> GaussianPrior:
    mean = predict(model, c).to(torch.float64)
    return GaussianPrior.from_covariance(mean, stats["cov"], label="fixed_global")


# ---------------------------------------------------------------------------
# Sampling-based baselines
# ---------------------------------------------------------------------------

def mc_dropout_prior(model_do, c: torch.Tensor, n_samples: int = 64,
                     shrink: float = 1e-2, **kw) -> GaussianPrior:
    enable_dropout(model_do)
    with torch.no_grad():
        samples = torch.stack([model_do(c.reshape(-1)) for _ in range(n_samples)])
    model_do.eval()
    mean = samples.mean(0).to(torch.float64)
    return GaussianPrior.from_covariance(mean, _cov_from_samples(samples, shrink),
                                         label="mc_dropout")


def ensemble_prior(models: list, c: torch.Tensor, shrink: float = 1e-2, **kw) -> GaussianPrior:
    with torch.no_grad():
        samples = torch.stack([m(c.reshape(-1)) for m in models])
    mean = samples.mean(0).to(torch.float64)
    return GaussianPrior.from_covariance(mean, _cov_from_samples(samples, shrink),
                                         label="deep_ensemble")


def swag_prior(model, swag: dict, c: torch.Tensor, n_samples: int = 32,
               scale: float = 0.5, shrink: float = 1e-2, **kw) -> GaussianPrior:
    """SWAG: sample weights from the low-rank + diagonal SGD-iterate posterior,
    push each through the network, take the sample covariance of the outputs."""
    device = c.device
    mean_w = swag["mean"].to(device)
    var_w = swag["var"].to(device)
    dev = swag["dev"].to(device)                       # (K, P)
    K = dev.shape[0]
    backup = parameters_to_vector(model.parameters()).detach().clone()
    outs = []
    with torch.no_grad():
        for _ in range(n_samples):
            z1 = torch.randn_like(mean_w)
            z2 = torch.randn(K, device=device, dtype=mean_w.dtype)
            w = mean_w + scale * (var_w.sqrt() * z1 / np.sqrt(2)
                                  + (dev.T @ z2) / np.sqrt(2 * max(K - 1, 1)))
            vector_to_parameters(w, model.parameters())
            outs.append(model(c.reshape(-1)).clone())
        vector_to_parameters(backup, model.parameters())
    samples = torch.stack(outs)
    mean = samples.mean(0).to(torch.float64)
    return GaussianPrior.from_covariance(mean, _cov_from_samples(samples, shrink),
                                         label="swag")


# ---------------------------------------------------------------------------
# Last-layer Laplace
# ---------------------------------------------------------------------------

def _last_layer_params(model) -> list[torch.nn.Parameter]:
    return list(model.proj[-1].parameters())


def fit_laplace(model, xs: torch.Tensor, ys: torch.Tensor,
                prior_prec: float = 1.0) -> dict:
    """Diagonal GGN over the final linear layer (last-layer Laplace).

    Full-network Laplace on an FNO means a 10^5-column Jacobian per sample; the
    last-layer variant is the standard tractable approximation and is what we
    report. Documented as such rather than labelled plain "Laplace".
    """
    params = _last_layer_params(model)
    for p in params:
        p.requires_grad_(True)
    P = sum(p.numel() for p in params)
    ggn = torch.zeros(P, dtype=torch.float64, device=xs.device)
    for i in range(xs.shape[0]):
        pred = model(xs[i])
        for j in range(pred.numel()):
            grads = torch.autograd.grad(pred[j], params, retain_graph=(j < pred.numel() - 1))
            g = torch.cat([g.reshape(-1) for g in grads]).to(torch.float64)
            ggn += g ** 2
    resid_var = float(((ys - model(xs)).detach() ** 2).mean())
    post_prec = ggn / max(resid_var, 1e-12) + prior_prec
    for p in params:
        p.requires_grad_(False)
    return {"post_var": (1.0 / post_prec), "resid_var": resid_var}


def laplace_prior(model, c: torch.Tensor, lap: dict, shrink: float = 1e-2,
                  **kw) -> GaussianPrior:
    """Linearized predictive: Sigma = J_w diag(post_var) J_w^T + resid_var I."""
    params = _last_layer_params(model)
    for p in params:
        p.requires_grad_(True)
    pred = model(c.reshape(-1))
    rows = []
    for j in range(pred.numel()):
        grads = torch.autograd.grad(pred[j], params, retain_graph=(j < pred.numel() - 1))
        rows.append(torch.cat([g.reshape(-1) for g in grads]).to(torch.float64))
    Jw = torch.stack(rows)                                   # (N, P)
    for p in params:
        p.requires_grad_(False)
    Sigma = (Jw * lap["post_var"]) @ Jw.T
    Sigma = Sigma + lap["resid_var"] * torch.eye(Sigma.shape[0], dtype=Sigma.dtype,
                                                 device=Sigma.device)
    return GaussianPrior.from_covariance(pred.detach().to(torch.float64),
                                         damp(Sigma, shrink), label="laplace_lastlayer")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES = ["isotropic", "diagonal_fitted", "fixed_global", "mc_dropout",
             "deep_ensemble", "swag", "laplace_lastlayer"]


def build_baseline_fn(name: str, ctx: dict):
    """Return f(c) -> GaussianPrior for the named baseline.

    `ctx` supplies whatever the baseline needs: fitted residual stats, the
    dropout model, ensemble members, SWAG statistics, Laplace statistics.
    """
    if name == "isotropic":
        return lambda c: isotropic_prior(ctx["model"], c)
    if name == "diagonal_fitted":
        return lambda c: diagonal_prior(ctx["model"], c, ctx["res_stats"])
    if name == "fixed_global":
        return lambda c: fixed_global_prior(ctx["model"], c, ctx["res_stats"])
    if name == "mc_dropout":
        return lambda c: mc_dropout_prior(ctx["model_do"], c, ctx.get("n_mc", 64))
    if name == "deep_ensemble":
        return lambda c: ensemble_prior(ctx["ensemble"], c)
    if name == "swag":
        return lambda c: swag_prior(ctx["model"], ctx["swag"], c, ctx.get("n_swag", 32))
    if name == "laplace_lastlayer":
        return lambda c: laplace_prior(ctx["model"], c, ctx["laplace"])
    raise ValueError(f"unknown baseline {name!r}")
