"""Uncertainty-quality metrics (Stages 2-3) and correlation tools (Stage 1)."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import stats

from .priors import GaussianPrior

SQRT2 = math.sqrt(2.0)
SQRT_PI = math.sqrt(math.pi)


# ---------------------------------------------------------------------------
# Joint (full-covariance) diagnostics
# ---------------------------------------------------------------------------

def mahalanobis_sq(prior: GaussianPrior, y: torch.Tensor) -> float:
    return float(prior.mahalanobis_sq(y.reshape(1, -1))[0])


def joint_nll(prior: GaussianPrior, y: torch.Tensor, per_dim: bool = True) -> float:
    lp = float(prior.log_prob(y.reshape(1, -1))[0])
    return -lp / prior.N if per_dim else -lp


def ellipsoid_coverage(maha_sq: np.ndarray, N: int, levels=(0.9, 0.95, 0.99)) -> dict:
    """Empirical coverage of the nominal chi^2_N confidence ellipsoids."""
    out = {}
    for lv in levels:
        out[f"coverage@{lv:g}"] = float(np.mean(maha_sq <= stats.chi2.ppf(lv, df=N)))
    return out


def maha_chi2_test(maha_sq: np.ndarray, N: int) -> dict:
    """Is the Mahalanobis distribution consistent with chi^2_N?

    Under a correct Gaussian, D^2 ~ chi^2_N exactly. The KS statistic on the
    chi^2 CDF is the sharpest single check of Stage 2's hypothesis.
    """
    u = stats.chi2.cdf(maha_sq, df=N)
    ks = stats.kstest(u, "uniform")
    return {
        "maha_mean": float(np.mean(maha_sq)), "maha_expected": float(N),
        "maha_ratio": float(np.mean(maha_sq) / N),
        "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
    }


# ---------------------------------------------------------------------------
# Marginal (per-coordinate) diagnostics
# ---------------------------------------------------------------------------

def gaussian_nll_marginal(mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> float:
    z = (y - mu) / sd
    return float(np.mean(0.5 * z ** 2 + np.log(sd) + 0.5 * np.log(2 * np.pi)))


def crps_gaussian(mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> float:
    """Closed-form CRPS for a Gaussian predictive marginal."""
    z = (y - mu) / sd
    return float(np.mean(sd * (z * (2 * stats.norm.cdf(z) - 1)
                               + 2 * stats.norm.pdf(z) - 1 / SQRT_PI)))


def calibration_curve(mu: np.ndarray, sd: np.ndarray, y: np.ndarray,
                      n_levels: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Nominal vs empirical coverage of central credible intervals."""
    nominal = np.linspace(0.0, 1.0, n_levels)
    z = np.abs((y - mu) / sd)
    empirical = np.array([np.mean(z <= stats.norm.ppf(0.5 + p / 2)) if p > 0 else 0.0
                          for p in nominal])
    return nominal, empirical


def expected_calibration_error(mu, sd, y, n_levels: int = 21) -> float:
    nom, emp = calibration_curve(mu, sd, y, n_levels)
    return float(np.mean(np.abs(nom - emp)))


def sharpness(sd: np.ndarray) -> float:
    """Mean predictive std. Meaningless alone -- read next to calibration."""
    return float(np.mean(sd))


def interval_score(mu, sd, y, alpha: float = 0.1) -> float:
    """Proper scoring rule for the central (1-alpha) interval."""
    zq = stats.norm.ppf(1 - alpha / 2)
    lo, hi = mu - zq * sd, mu + zq * sd
    width = hi - lo
    pen = (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return float(np.mean(width + pen))


def marginal_report(prior: GaussianPrior, y: torch.Tensor) -> dict:
    mu = prior.mean.cpu().numpy()
    sd = prior.marginal_std().cpu().numpy()
    yy = y.reshape(-1).to(torch.float64).cpu().numpy()
    return {
        "nll_marginal": gaussian_nll_marginal(mu, sd, yy),
        "crps": crps_gaussian(mu, sd, yy),
        "ece": expected_calibration_error(mu, sd, yy),
        "sharpness": sharpness(sd),
        "interval_score_90": interval_score(mu, sd, yy, 0.1),
        "rmse": float(np.sqrt(np.mean((mu - yy) ** 2))),
        "mae": float(np.mean(np.abs(mu - yy))),
    }


def aggregate(reports: list[dict]) -> dict:
    keys = reports[0].keys()
    return {k: float(np.mean([r[k] for r in reports])) for k in keys}


# ---------------------------------------------------------------------------
# Stage 1 correlation tools
# ---------------------------------------------------------------------------

def spearman(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r = stats.spearmanr(a[ok], b[ok])
    return float(r.statistic), float(r.pvalue)


def pearson(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r = stats.pearsonr(a[ok], b[ok])
    return float(r.statistic), float(r.pvalue)


def directional_errors(basis: torch.Tensor, err: torch.Tensor) -> np.ndarray:
    """Project the error vector onto an orthonormal basis (columns)."""
    return (err.reshape(1, -1).to(torch.float64) @ basis.to(torch.float64)
            ).reshape(-1).abs().cpu().numpy()


def oracle_direction_ceiling(svals: np.ndarray, n_draws: int = 200,
                             pooled_over: int = 0, seed: int = 0) -> dict:
    """Largest per-direction Spearman achievable when the pushforward model is
    EXACTLY true and there is no model bias at all.

    Under e = J delta with delta ~ N(0, I), the projection onto the i-th left
    singular vector is u_i^T e = s_i z_i with z_i ~ N(0,1). So |u_i^T e| carries
    multiplicative half-normal noise (CV ~ 0.76) on top of the systematic s_i
    trend. When cond(J) is small -- and a residual model has J = I + dNet/dc, so
    it always is -- that noise swamps the trend and a *single-sample* rank
    correlation is bounded far below 1.

    This ceiling must be computed before any observed rho is called a pass or a
    failure: a threshold above the ceiling is unreachable by construction, and
    an observed value above the ceiling indicates a confound rather than a
    success. `pooled_over > 0` returns the ceiling for the pooled statistic,
    which averages |u_i^T e|^2 over that many states first and is far stronger.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(svals, float)
    out = []
    for _ in range(n_draws):
        if pooled_over > 0:
            acc = np.zeros_like(s)
            for _ in range(pooled_over):
                acc += (s * rng.standard_normal(len(s))) ** 2
            r, _ = spearman(s ** 2, acc / pooled_over)
        else:
            r, _ = spearman(s, np.abs(s * rng.standard_normal(len(s))))
        out.append(r)
    out = np.array(out)
    return {"median": float(np.median(out)),
            "p05": float(np.percentile(out, 5)),
            "p95": float(np.percentile(out, 95))}


def pooled_direction_test(svals_list, proj_list) -> tuple[float, float]:
    """Per-rank pooled test: correlate mean_j s_j^2 against mean_j |u_j^T e|^2,
    averaging over states before correlating.

    Singular ranks are comparable across states (both are sorted), so pooling
    is legitimate and removes the half-normal noise that cripples the
    per-sample version.
    """
    S = np.mean(np.stack([np.asarray(s, float) ** 2 for s in svals_list]), axis=0)
    E = np.mean(np.stack([np.asarray(p, float) ** 2 for p in proj_list]), axis=0)
    return spearman(S, E)


def explained_error_fraction(U: torch.Tensor, err: torch.Tensor, k: int) -> float:
    """Fraction of squared error inside span of the leading k columns of U.

    Compared against k/N (the isotropic expectation), this is the direct test
    for bias-dominated error described in docs/method.md section 9.
    """
    e = err.reshape(-1).to(torch.float64)
    proj = U[:, :k].to(torch.float64).T @ e
    denom = float((e ** 2).sum())
    return float((proj ** 2).sum() / denom) if denom > 0 else float("nan")
