"""One-parameter scale calibration, streaming.

Identical estimator to `hipp/calibrate.py` -- the closed-form Gaussian MLE

    alpha* = (1/M) sum_j (1/N) r_j^T H_j r_j

and the robust median-matching alternative -- but consuming `QuadSummary`
objects rather than materialized priors. That is the whole point: a rank-64
basis at N = 16,384 is 8 MB, so holding 2,000 calibration priors is 16 GB, while
holding 2,000 summaries is a few megabytes.

The heavy-tail warning from the 1D study applies with more force here, not less.
alpha* is a mean of per-state quadratic forms, and at N = 16,384 those forms are
sums over far more dimensions, so their *relative* spread narrows -- but the
tail that matters is across states (some states are genuinely harder), and that
does not average away with N. Both objectives are still reported with a
bootstrap standard error.

There is one new failure mode worth naming. When the retained rank k is small
relative to N, the quadratic form is dominated by the isotropic floor:

    r^T S^{-1} r  ~=  ||r||^2 / tau  -  (subspace correction)

so alpha ends up fitting tau, and the curvature contributes a correction of
order (k/N) * (captured error fraction). `alpha_decomposition` reports that
split explicitly, because a method whose alpha is 99% floor is an isotropic
Gaussian wearing a hat, and no downstream metric will reveal that on its own.
"""
from __future__ import annotations

import math

import numpy as np

from .lowrank import QuadSummary


def fit_alpha_scale(summaries: list[QuadSummary], objective: str = "mle",
                    return_diagnostics: bool = False, n_boot: int = 500,
                    seed: int = 0):
    """Fit the single scale alpha from streamed per-state summaries."""
    if not summaries:
        raise ValueError("no calibration summaries")
    N = summaries[0].N
    quads = np.array([s.quad() / s.N for s in summaries], dtype=np.float64)
    if objective == "mle":
        alpha = max(float(np.mean(quads)), 1e-300)
    elif objective == "median":
        from scipy.stats import chi2
        alpha = max(float(np.median(quads) * N / chi2.ppf(0.5, df=N)), 1e-300)
    else:
        raise ValueError(f"objective must be 'mle' or 'median', got {objective!r}")
    if not return_diagnostics:
        return alpha

    rng = np.random.default_rng(seed)
    stat = np.mean if objective == "mle" else np.median
    boot = np.array([stat(rng.choice(quads, size=quads.size, replace=True))
                     for _ in range(n_boot)])
    boot *= alpha / max(float(stat(quads)), 1e-300)
    diag = {
        "alpha": alpha, "objective": objective, "n": int(quads.size), "N": N,
        "rel_se": float(np.std(boot) / max(alpha, 1e-300)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "tail_ratio": float(np.mean(quads) / max(np.median(quads), 1e-300)),
        "max_over_median": float(np.max(quads) / max(np.median(quads), 1e-300)),
    }
    diag.update(alpha_decomposition(summaries))
    return alpha, diag


def alpha_decomposition(summaries: list[QuadSummary]) -> dict:
    """How much of the quadratic form is the isotropic floor versus the geometry.

    quad = (||r||^2 - sum_i w_i p_i^2) / tau  with  w_i = d_i/(d_i+tau).

    `floor_share` is the first term's share. Near 1 means the fitted alpha is
    essentially ||r||^2/(N tau), i.e. the estimator has degenerated to isotropic
    and any apparent win over the isotropic baseline is coming from tau, which
    alpha then cancels. This is the single most useful number for deciding
    whether a rank-k result is real.
    """
    # The positive "floor minus low-rank correction" interpretation below is
    # specific to a pushforward covariance (d >= 0).  For Gauss--Newton,
    # d < 0 by construction; reporting a negative geometry share there is not
    # an inference failure, merely an invalid diagnostic algebra.
    if any(np.any(s.d < 0) for s in summaries):
        caps = [float((s.proj ** 2).sum()) / max(s.r_sq, 1e-300)
                for s in summaries]
        return {
            "floor_share": float("nan"),
            "geometry_share": float("nan"),
            "error_energy_in_subspace": float(np.mean(caps)),
            "decomposition_note": "not defined for signed Gauss--Newton offsets",
        }
    floor, corr, cap = [], [], []
    for s in summaries:
        w = s.d / (s.d + s.tau)
        base = s.r_sq / s.tau
        correction = float((w * s.proj ** 2).sum()) / s.tau
        floor.append(base)
        corr.append(correction)
        cap.append(float((s.proj ** 2).sum()) / max(s.r_sq, 1e-300))
    floor = np.array(floor)
    corr = np.array(corr)
    denom = np.maximum(floor, 1e-300)
    return {
        "floor_share": float(np.mean(1.0 - corr / denom)),
        "geometry_share": float(np.mean(corr / denom)),
        "error_energy_in_subspace": float(np.mean(cap)),
    }


def fit_alpha_coverage(summaries: list[QuadSummary], level: float = 0.90,
                       n_grid: int = 81, span: float = 3.0) -> tuple[float, dict]:
    """Fit alpha to hit nominal ellipsoid coverage instead of maximizing likelihood.

    Kept because the two disagree exactly when the residual distribution is
    heavier-tailed than Gaussian, and the size of the disagreement is a
    measurement of that. Reported alongside, never substituted silently.
    """
    from scipy.stats import chi2
    N = summaries[0].N
    q = np.array([s.quad() for s in summaries], dtype=np.float64)
    a0 = fit_alpha_scale(summaries)
    thresh = chi2.ppf(level, df=N)
    grid = a0 * np.logspace(-span, span, n_grid)
    curve = [(float(a), abs(float(np.mean(q / a <= thresh)) - level)) for a in grid]
    best = min(curve, key=lambda t: t[1])[0]
    return best, {"curve": curve, "mle": a0, "level": level,
                  "ratio_to_mle": best / max(a0, 1e-300)}


def alpha_report(diag: dict) -> str:
    """One-line human summary for stage logs."""
    return (f"alpha={diag['alpha']:.4g} "
            f"(rel.SE {diag['rel_se']:.2f}, tail {diag['tail_ratio']:.2f}, "
            f"floor share {diag.get('floor_share', float('nan')):.3f}, "
            f"err in subspace {diag.get('error_energy_in_subspace', float('nan')):.3f})")


def expected_maha(N: int) -> float:
    """E[D^2] under a correct Gaussian. Trivial, but stated once so that no stage
    re-derives it and the chi^2_N reference is unambiguous."""
    return float(N)


def maha_z(maha_sq: np.ndarray, N: int) -> np.ndarray:
    """(D^2 - N) / sqrt(2N): the standardized deviation from chi^2_N.

    At N = 16,384 a chi^2_N goodness-of-fit test rejects on deviations far too
    small to matter, because the test's power grows with N while the effect size
    does not. This z-score is what should be reported instead -- it says *how
    far off* the calibration is in units the reader can interpret, rather than
    producing a p-value of 0 for every method including the correct one.
    """
    return (np.asarray(maha_sq, dtype=np.float64) - N) / math.sqrt(2.0 * N)
