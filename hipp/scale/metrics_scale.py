"""Uncertainty metrics that stay meaningful at N ~ 10^4.

Most of `hipp/metrics.py` transfers unchanged in *form* but not in
*interpretation*, and two of its tests stop working outright. What is here:

Kept as-is
    marginal NLL, CRPS, ECE, sharpness, interval score -- all per-pixel, so
    they are dimension-agnostic and directly comparable to the 1D numbers.

Replaced
    the chi^2_N goodness-of-fit test. At N = 16,384 it has enough power to
    reject any method on a deviation of no practical size, so a table of
    p-values is all zeros and says nothing. `chi2_report` returns the
    standardized z instead, and adds the split below.

New, and the reason this file exists
    `subspace_split_test`. A rank-k covariance makes two separable claims: that
    the geometry inside span(U) is right, and that the isotropic floor covering
    the other N-k directions is right. Those fail independently and for
    different reasons, and at k/N ~ 0.004 the aggregate statistic is almost
    entirely the floor -- so a method can look perfectly calibrated while its
    curvature contributes nothing. Splitting D^2 into D_k^2 ~ chi^2_k and
    D_perp^2 ~ chi^2_{N-k} separates them, and both are computable from the
    streamed summary alone.

    `pooled_direction_test_lowrank`. Stage 1's primary statistic, restricted to
    the k retained ranks. The oracle ceiling from `hipp/metrics.py` still
    applies and must still be used -- with fewer ranks the ceiling is *lower*,
    so transferring the 1D threshold would be exactly the mistake documented in
    docs/method.md 2.4.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .lowrank import LowRankGaussian, QuadSummary


# ---------------------------------------------------------------------------
# Joint (whole-state) calibration
# ---------------------------------------------------------------------------

def maha_from_summaries(summaries: list[QuadSummary], alpha: float) -> np.ndarray:
    return np.array([s.maha_sq(alpha) for s in summaries], dtype=np.float64)


def chi2_report(maha_sq: np.ndarray, N: int) -> dict:
    """Standardized calibration of the joint Mahalanobis distance.

    `z_mean` is the headline: 0 is perfect, positive means under-dispersed
    (the predicted covariance is too small), negative means over-dispersed.
    The KS p-value is still computed but should be read as a shape diagnostic,
    not a pass/fail -- see the module docstring.
    """
    from scipy.stats import chi2, kstest
    m = np.asarray(maha_sq, dtype=np.float64)
    z = (m - N) / math.sqrt(2.0 * N)
    ks = kstest(m, lambda v: chi2.cdf(v, df=N))
    return {
        "N": N, "n_states": int(m.size),
        "mean_maha_over_N": float(m.mean() / N),
        "median_maha_over_N": float(np.median(m) / N),
        "z_mean": float(z.mean()), "z_median": float(np.median(z)),
        "z_std": float(z.std()),
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "tail_ratio": float(m.mean() / max(np.median(m), 1e-300)),
    }


def subspace_split_test(summaries: list[QuadSummary], alpha: float) -> dict:
    """Split D^2 into the retained subspace and its complement.

    In-subspace:   D_k^2   = sum_i p_i^2 / (alpha (d_i + tau))       ~ chi^2_k
    Complement:    D_perp^2 = (||r||^2 - ||p||^2) / (alpha tau)      ~ chi^2_{N-k}

    Both are exact given the summary. The two z-scores are the diagnostic:

      z_sub ~ 0, z_perp ~ 0    the covariance is right in both parts
      z_sub > 0, z_perp ~ 0    the geometry under-predicts variance where it
                               claims to know something -- the interesting failure
      z_sub ~ 0, z_perp > 0    error lives outside the retained subspace, i.e.
                               rank k is too small or the error is bias-dominated
      both far from 0          the scale alpha is compensating across two parts
                               that need different scales, which is the strongest
                               evidence that a single global alpha is inadequate
    """
    dk, dp, ks_, np_ = [], [], None, None
    for s in summaries:
        eig = s.d + s.tau
        d_sub = float((s.proj ** 2 / np.maximum(eig, 1e-300)).sum()) / alpha
        perp = max(s.r_sq - float((s.proj ** 2).sum()), 0.0)
        d_perp = perp / (alpha * s.tau)
        dk.append(d_sub)
        dp.append(d_perp)
        ks_, np_ = s.k, s.N - s.k
    dk, dp = np.array(dk), np.array(dp)
    return {
        "k": ks_, "n_perp": np_,
        "sub": {"mean_over_dof": float(dk.mean() / max(ks_, 1)),
                "z_mean": float(((dk - ks_) / math.sqrt(2.0 * max(ks_, 1))).mean())},
        "perp": {"mean_over_dof": float(dp.mean() / max(np_, 1)),
                 "z_mean": float(((dp - np_) / math.sqrt(2.0 * max(np_, 1))).mean())},
        "share_of_maha_in_subspace": float(dk.mean() / max(dk.mean() + dp.mean(), 1e-300)),
    }


def ellipsoid_coverage(maha_sq: np.ndarray, N: int,
                       levels=(0.9, 0.95, 0.99)) -> dict:
    from scipy.stats import chi2
    m = np.asarray(maha_sq, dtype=np.float64)
    return {f"cov_{int(100*l)}": float(np.mean(m <= chi2.ppf(l, df=N)))
            for l in levels}


# ---------------------------------------------------------------------------
# Marginal (per-pixel) quality
# ---------------------------------------------------------------------------

def marginal_report(prior: LowRankGaussian, y: torch.Tensor) -> dict:
    """Per-pixel NLL / CRPS / ECE / sharpness / coverage for one state.

    Marginal rather than joint on purpose: these are the numbers a practitioner
    reads off an error bar, they are comparable across resolutions, and unlike
    the joint NLL they are not dominated by the log-determinant of a
    16,384-dimensional floor.
    """
    mu = prior.mean.detach().double().cpu().numpy()
    sd = prior.marginal_std().detach().double().cpu().numpy()
    yy = y.detach().double().reshape(-1).cpu().numpy()
    return {
        "nll": gaussian_nll_marginal(mu, sd, yy),
        "crps": crps_gaussian(mu, sd, yy),
        "ece": expected_calibration_error(mu, sd, yy),
        "sharpness": float(np.mean(sd)),
        "rmse": float(np.sqrt(np.mean((yy - mu) ** 2))),
        "interval_score_90": interval_score(mu, sd, yy, alpha=0.1),
        "cov_90_marginal": float(np.mean(np.abs(yy - mu) <= 1.6448536 * sd)),
        "cov_95_marginal": float(np.mean(np.abs(yy - mu) <= 1.959964 * sd)),
    }


def gaussian_nll_marginal(mu, sd, y) -> float:
    sd = np.maximum(sd, 1e-30)
    return float(np.mean(0.5 * (((y - mu) / sd) ** 2 + 2 * np.log(sd)
                                + math.log(2 * math.pi))))


def crps_gaussian(mu, sd, y) -> float:
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-30)
    z = (y - mu) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1)
                               + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))))


def calibration_curve(mu, sd, y, n_levels: int = 21):
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-30)
    z = np.abs((y - mu) / sd)
    nominal = np.linspace(0.0, 1.0, n_levels)
    empirical = np.array([float(np.mean(z <= norm.ppf(0.5 + p / 2))) if p > 0 else 0.0
                          for p in nominal])
    return nominal, empirical


def expected_calibration_error(mu, sd, y, n_levels: int = 21) -> float:
    nominal, empirical = calibration_curve(mu, sd, y, n_levels)
    return float(np.mean(np.abs(nominal - empirical)))


def interval_score(mu, sd, y, alpha: float = 0.1) -> float:
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-30)
    z = norm.ppf(1 - alpha / 2)
    lo, hi = mu - z * sd, mu + z * sd
    return float(np.mean((hi - lo)
                         + 2 / alpha * (lo - y) * (y < lo)
                         + 2 / alpha * (y - hi) * (y > hi)))


def aggregate(reports: list[dict]) -> dict:
    if not reports:
        return {}
    keys = reports[0].keys()
    out = {}
    for kk in keys:
        v = np.array([r[kk] for r in reports], dtype=np.float64)
        out[kk] = float(v.mean())
        out[kk + "_se"] = float(v.std() / max(math.sqrt(v.size), 1))
    return out


# ---------------------------------------------------------------------------
# Stage 1's direction test, restricted to the retained ranks
# ---------------------------------------------------------------------------

def pooled_direction_test_lowrank(svals_list, proj_list):
    """Spearman(s_j^2, mean_j |u_j^T e|^2) pooled across states, per singular rank.

    Pooling before correlating is what raises the achievable correlation above
    the half-normal noise floor (docs/method.md 2.4). Ranks are comparable
    across states because both are sorted, and with a rank-k basis only the top
    k ranks exist -- which lowers the oracle ceiling relative to the 1D study
    and must be recomputed, not carried over.
    """
    from scipy.stats import spearmanr
    S = np.stack([np.asarray(s, dtype=np.float64) for s in svals_list])   # (M, k)
    P = np.stack([np.asarray(p, dtype=np.float64) for p in proj_list])    # (M, k)
    s_mean = (S ** 2).mean(axis=0)
    e_mean = (P ** 2).mean(axis=0)
    good = np.isfinite(s_mean) & np.isfinite(e_mean)
    if good.sum() < 3:
        return float("nan"), float("nan")
    r = spearmanr(s_mean[good], e_mean[good])
    return float(r.statistic), float(r.pvalue)


def oracle_direction_ceiling_lowrank(svals: np.ndarray, pooled_over: int = 1,
                                     n_draws: int = 400, seed: int = 0) -> dict:
    """Achievable Spearman when the pushforward model holds *exactly*.

    Simulates u_i^T e = s_i z_i with z ~ N(0,1), pools `pooled_over` states the
    same way the real statistic does, and reports the distribution of the
    resulting rank correlation. Any observed rho must be read as a fraction of
    this; an observed value above the 95th percentile is a confound, not a win.
    """
    rng = np.random.default_rng(seed)
    from scipy.stats import spearmanr
    s = np.asarray(svals, dtype=np.float64)
    s2 = s ** 2
    out = []
    for _ in range(n_draws):
        z = rng.standard_normal((max(pooled_over, 1), s.size))
        e = (s2 * z ** 2).mean(axis=0)
        out.append(spearmanr(s2, e).statistic)
    out = np.array([v for v in out if np.isfinite(v)])
    return {"median": float(np.median(out)),
            "p05": float(np.percentile(out, 5)),
            "p95": float(np.percentile(out, 95)),
            "n_ranks": int(s.size), "pooled_over": int(pooled_over)}


def explained_error_fraction(U: torch.Tensor, err: torch.Tensor, k: int) -> float:
    """Share of squared error inside span(U[:, :k]).

    Divided by k/N this is the bias diagnostic: ~1 means the error is isotropic
    with respect to J, i.e. model bias, which no Jacobian construction can
    capture (docs/method.md 9). At scale k/N is tiny, so the ratio has a large
    dynamic range and is more informative than it was in 1D.
    """
    e = err.detach().double().reshape(-1)
    p = U[:, :k].double().T @ e
    return float((p ** 2).sum() / (e @ e).clamp(min=1e-300))


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    g = np.isfinite(a) & np.isfinite(b)
    if g.sum() < 3:
        return float("nan"), float("nan")
    r = spearmanr(a[g], b[g])
    return float(r.statistic), float(r.pvalue)


def pearson(a, b):
    from scipy.stats import pearsonr
    a, b = np.asarray(a, float), np.asarray(b, float)
    g = np.isfinite(a) & np.isfinite(b)
    if g.sum() < 3:
        return float("nan"), float("nan")
    r = pearsonr(a[g], b[g])
    return float(r.statistic), float(r.pvalue)
