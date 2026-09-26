"""One-parameter scale calibration.

Curvature surrogates determine the *shape* of the predictive covariance but not
its magnitude. For Sigma = alpha H^{-1} the Gaussian MLE for alpha over a
calibration set has a closed form:

    alpha* = (1/M) sum_j  (1/N) (c_j - chat_j)^T H_j (c_j - chat_j)

Every baseline receives the same treatment so that comparisons isolate the
covariance *shape*, which is the only thing the method claims to provide.
"""
from __future__ import annotations

import numpy as np
import torch

from .priors import GaussianPrior


def fit_alpha(priors: list[GaussianPrior], targets: torch.Tensor,
              return_diagnostics: bool = False, objective: str = "mle"):
    """Fit the global scale alpha given unit-scale priors.

    `objective="mle"` is the closed-form Gaussian maximum likelihood estimate,
    a *mean* of per-state quadratic forms.

    `objective="median"` instead matches the median of D^2 to the median of
    chi^2_N. This is not the MLE and should be labelled as such wherever it is
    used -- but the per-state quadratic forms are strongly heavy-tailed in
    practice (empirically mean/median ~ 2 with a maximum ~60x the median, see
    docs/method.md 2.3), which makes the MLE both noisy across calibration
    draws and systematically too large for the typical state. Reporting both
    separates "the covariance shape is wrong" from "the scale was set by three
    outlier states".
    """
    if len(priors) == 0:
        raise ValueError("no calibration samples")
    quads = np.array([float(p.quad(y.reshape(1, -1))[0]) / p.N
                      for p, y in zip(priors, targets)])
    if objective == "mle":
        a = max(float(np.mean(quads)), 1e-30)
    elif objective == "median":
        from scipy.stats import chi2
        N = priors[0].N
        a = max(float(np.median(quads) * N / chi2.ppf(0.5, df=N)), 1e-30)
    else:
        raise ValueError(objective)
    if not return_diagnostics:
        return a
    rng = np.random.default_rng(0)
    stat = np.mean if objective == "mle" else np.median
    boot = np.array([stat(rng.choice(quads, size=len(quads), replace=True))
                     for _ in range(500)])
    boot = boot * (a / max(float(stat(quads)), 1e-30))
    return a, {
        "alpha": a,
        "objective": objective,
        "n": len(quads),
        "rel_se": float(np.std(boot) / max(a, 1e-30)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "tail_ratio": float(np.mean(quads) / max(np.median(quads), 1e-30)),
    }


def fit_alpha_grid(priors: list[GaussianPrior], targets: torch.Tensor,
                   objective: str = "nll", n_grid: int = 61,
                   span: float = 3.0) -> tuple[float, dict]:
    """Numerically optimize alpha on a log grid around the closed-form MLE.

    `objective="nll"` reproduces the closed form (a useful self-check);
    `objective="coverage"` instead targets nominal 90% coverage, which can
    differ substantially when the residual distribution is heavy-tailed.
    """
    from scipy.stats import chi2

    a0 = fit_alpha(priors, targets)
    grid = a0 * np.logspace(-span, span, n_grid)
    N = priors[0].N
    quads = np.array([float(p.quad(y.reshape(1, -1))[0]) for p, y in zip(priors, targets)])
    # logdet of the *shape* H, independent of whatever alpha the prior carries
    logdet_shape = np.array(
        [float(2.0 * torch.log(torch.diagonal(p.chol_H)).sum()) for p in priors])

    best, best_val, curve = a0, np.inf, []
    thresh90 = chi2.ppf(0.90, df=N)
    for a in grid:
        if objective == "nll":
            val = float(np.mean(0.5 * (quads / a - logdet_shape + N * np.log(a)
                                       + N * np.log(2 * np.pi))))
        elif objective == "coverage":
            cov = float(np.mean(quads / a <= thresh90))
            val = abs(cov - 0.90)
        else:
            raise ValueError(objective)
        curve.append((float(a), val))
        if val < best_val:
            best, best_val = float(a), val
    return best, {"curve": curve, "closed_form": a0, "objective": objective}


def calibrated_prior(prior: GaussianPrior, alpha: float) -> GaussianPrior:
    return prior.rescaled(alpha)
