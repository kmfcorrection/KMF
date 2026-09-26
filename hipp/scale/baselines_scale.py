"""Baselines at scale. Every one produces a `LowRankGaussian`, so all of them go
through the identical one-parameter calibration and differ only in covariance
*shape* -- the same discipline as `hipp/baselines.py`.

Which 1D baselines survive, and what replaces the ones that do not:

  isotropic         unchanged.
  diagonal_fitted   unchanged in spirit; the per-pixel residual variance is a
                    dense N-vector, which is fine, but it has no low-rank
                    structure, so it is carried the same way `diag_gn` is:
                    top-k deviations from a median floor.
  fixed_global      **this is the one that matters** and it does not survive
                    literally. A full N x N residual covariance at N = 16,384
                    needs >N samples to be non-singular and 2 GB to store. It is
                    replaced by `fixed_global_lowrank`: rank-k PCA of the
                    calibration residuals plus an isotropic floor from the
                    discarded variance -- which is *exactly the same
                    representation* as the curvature estimators, so the
                    comparison is like-for-like rather than being confounded by
                    one side being truncated. This is the sharpest null
                    hypothesis: it is state-independent, so if HILP does not beat
                    it, the Jacobian is contributing no per-state information.
  deep_ensemble     unchanged in form; the K member predictions span a rank-(K-1)
                    subspace, which is natively low-rank. Note K << k typically,
                    so the ensemble is *rank-starved* relative to the curvature
                    estimators -- worth stating rather than presenting as a fair
                    fight at equal rank.
  mc_dropout        unchanged; T stochastic passes give a rank-min(T, k) sample
                    covariance.
  swag              unchanged in form, but the linearization J_theta Sigma_theta
                    J_theta^T is now done matrix-free by sampling weights and
                    differencing predictions, since J_theta is (N x P) with
                    P ~ 10^7.
  laplace_lastlayer dropped at scale. It needs the last layer's N x P_last
                    Jacobian; for a width-64 FNO head that is 16,384 x 8,320,
                    which is affordable, but the resulting covariance is
                    dominated by whichever pixels the head happens to weight
                    and it was already the weakest baseline in 1D. Kept out
                    rather than shipped as a strawman.

  conformal         **new, and not optional.** Split conformal prediction gives
                    distribution-free marginal coverage from the calibration
                    residuals alone, needs no model geometry, and is the
                    standard thing a reviewer will ask why you did not compare
                    against. It is not a Gaussian, so it is reported on coverage
                    and interval width rather than NLL -- comparing it on NLL
                    would be comparing it on a metric it does not optimize.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .lowrank import LowRankGaussian

BASELINES_SCALE = ["isotropic", "diagonal_fitted", "fixed_global_lowrank",
                   "deep_ensemble", "mc_dropout", "swag"]


# ---------------------------------------------------------------------------
# Residual statistics (streaming, one pass)
# ---------------------------------------------------------------------------

def fit_residual_stats(fm, xs, ys, k: int = 64, device=None,
                       max_states: int | None = None, split: float = 0.5) -> dict:
    """Mean, per-pixel variance and a *cross-validated* rank-k PCA of the residuals.

    The cross-validation is not a refinement; without it this baseline is
    unusable at scale, and the failure is silent.

    With M residual states in N dimensions, the sample covariance has at most M
    non-zero eigenvalues. The 1D code handled that by demanding M > N (the
    `--n-fit` flag, README "Known limitations"), which at N = 128 meant 512
    states. At N = 16,384 it would mean more residual states than the test set
    has pixels -- not merely expensive, but more data than the experiment has.
    Estimating the isotropic floor in-sample then gives

        tau = (total variance - captured variance) / (N - k)  ->  0

    because the *sample* eigenvalues outside rank k are exactly zero even though
    the true ones are not. A zero floor makes the complement infinitely
    confident, the Mahalanobis distance diverges, and the fitted alpha explodes
    to absorb it -- which looks like a spectacularly bad baseline rather than an
    unidentifiable one.

    The fix is a split: the basis U comes from one half of the states, and both
    the retained eigenvalues and the floor are measured on the *other* half,
    where the directions in U are not fitted to the data being measured. The
    floor is then the genuine out-of-sample variance orthogonal to U:

        tau = E_heldout ||(I - U U^T) r||^2 / (N - k)

    which is well-posed for any M and is what the state-independent covariance
    actually is.
    """
    n = len(xs) if max_states is None else min(max_states, len(xs))
    if n < 4:
        raise ValueError(f"fixed-global baseline needs at least 4 fit states, got {n}")
    R = torch.empty(n, fm.N, dtype=torch.float32, device=device)
    for i in range(n):
        x = xs[i].reshape(-1).to(fm.device, fm.dtype)
        pred = fm.predict(x)
        R[i] = (ys[i].reshape(-1).to(pred.device, pred.dtype) - pred).to(R.dtype).to(device)
    mean = R.mean(dim=0)
    # Second moment about f(c), *not* about the residual mean. Every method here
    # claims N(f(c), Sigma), so the covariance that claim needs is E[r r^T] with
    # r = y - f(c). Centering first would silently exclude the model's
    # systematic bias from the covariance while leaving it in the error, which
    # shows up as the complement being under-dispersed by exactly the bias
    # energy -- and since bias is state-independent, letting the
    # state-independent baseline absorb it is the fair comparison, not a
    # concession.
    Rc = R.double()
    var = Rc.pow(2).mean(dim=0)

    n_a = max(int(split * n), 2)
    A, B = Rc[:n_a], Rc[n_a:]
    kk = max(min(k, n_a - 1), 1)
    U, _, _ = torch.svd_lowrank(A.T, q=min(kk + 10, min(A.shape)), niter=4)
    U = U[:, :kk].contiguous()

    proj = B @ U                                     # (n_b, kk), held out
    perp = B.pow(2).sum(dim=1) - proj.pow(2).sum(dim=1)
    tau = float(perp.mean() / max(fm.N - kk, 1))
    if not (tau > 0):
        # Only reachable if the held-out residuals lie exactly in span(U), which
        # means the split was degenerate rather than that the floor is zero.
        tau = float(var.mean()) / max(fm.N, 1)
    d = proj.pow(2).mean(dim=0) - tau                # d + tau = held-out variance
    return {"mean": mean.double(), "var": var, "U": U, "d": d, "tau": tau,
            "n": n, "n_basis": n_a, "n_eval": n - n_a,
            "total_var": float(var.sum()), "k": kk,
            "captured_frac": float((d + tau).sum() / max(float(var.sum()), 1e-300))}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def isotropic_prior(fm, c, **kw) -> LowRankGaussian:
    return LowRankGaussian.isotropic(fm.predict(c.reshape(-1)).double(), tau=1.0,
                                     label="isotropic")


def diagonal_prior(fm, c, stats: dict, k: int = 64, **kw) -> LowRankGaussian:
    """Per-pixel variance, carried as top-k deviations from a median floor."""
    mean = fm.predict(c.reshape(-1)).double()
    var = stats["var"].to(mean.device)
    floor = float(var.median().clamp(min=1e-30))
    idx = torch.topk((var - floor).abs(), min(k, var.numel())).indices
    U = torch.zeros(var.numel(), idx.numel(), dtype=mean.dtype, device=mean.device)
    U[idx, torch.arange(idx.numel(), device=mean.device)] = 1.0
    return LowRankGaussian(mean, U, (var[idx] - floor), floor, label="diagonal_fitted")


def fixed_global_lowrank_prior(fm, c, stats: dict, **kw) -> LowRankGaussian:
    """State-independent rank-k residual covariance. The primary null hypothesis."""
    mean = fm.predict(c.reshape(-1)).double()
    return LowRankGaussian(mean, stats["U"].to(mean.device).double(),
                           stats["d"].to(mean.device).double(), stats["tau"],
                           label="fixed_global_lowrank")


def ensemble_prior(members, c, k: int = 64, floor_rel: float = 1e-3,
                   **kw) -> LowRankGaussian:
    """Sample covariance over K independently trained models.

    Rank is at most K-1. With K = 4 that is rank 3 against a rank-64 curvature
    estimator; the floor keeps it well-posed but the rank gap is real and is
    reported rather than papered over.
    """
    preds = torch.stack([m.predict(c.reshape(-1)).double() for m in members])
    mean = preds.mean(dim=0)
    D = (preds - mean) / math.sqrt(max(preds.shape[0] - 1, 1))
    U, S, _ = torch.linalg.svd(D.T, full_matrices=False)
    kk = min(k, S.numel())
    d = S[:kk] ** 2
    tau = max(floor_rel * float(d.mean().clamp(min=1e-30)), 1e-30)
    return LowRankGaussian(mean, U[:, :kk].contiguous(), d, tau, label="deep_ensemble")


def mc_dropout_prior(fm_do, c, n_samples: int = 32, k: int = 64,
                     floor_rel: float = 1e-3, **kw) -> LowRankGaussian:
    from .models2d import enable_dropout
    enable_dropout(fm_do.module)
    with torch.no_grad():
        preds = torch.stack([fm_do.predict(c.reshape(-1)).double()
                             for _ in range(n_samples)])
    fm_do.module.eval()
    mean = preds.mean(dim=0)
    D = (preds - mean) / math.sqrt(max(n_samples - 1, 1))
    U, S, _ = torch.linalg.svd(D.T, full_matrices=False)
    kk = min(k, S.numel())
    d = S[:kk] ** 2
    tau = max(floor_rel * float(d.mean().clamp(min=1e-30)), 1e-30)
    return LowRankGaussian(mean, U[:, :kk].contiguous(), d, tau, label="mc_dropout")


def swag_prior(fm, swag: dict, c, n_samples: int = 24, k: int = 64,
               scale: float = 0.5, floor_rel: float = 1e-3, **kw) -> LowRankGaussian:
    """SWAG, linearized to output space by sampling weights and differencing.

    The dense form J_theta Sigma_theta J_theta^T is unavailable (J_theta is
    N x P with P ~ 10^7). Sampling weights from the SWAG posterior and taking
    the empirical covariance of the resulting predictions is the same object to
    first order and costs n_samples forward passes.
    """
    module = fm.module
    base = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    mean_w = swag["mean"].to(base.device)
    var_w = swag["var"].to(base.device)
    dev = swag["dev"].to(base.device)
    preds = []
    try:
        for _ in range(n_samples):
            z1 = torch.randn_like(mean_w)
            z2 = torch.randn(dev.shape[0], device=base.device)
            w = mean_w + scale * (var_w.sqrt() * z1
                                  + (dev.T @ z2) / math.sqrt(2 * max(dev.shape[0] - 1, 1)))
            _load_flat(module, w)
            preds.append(fm.predict(c.reshape(-1)).double())
    finally:
        _load_flat(module, base)
    preds = torch.stack(preds)
    mean = preds.mean(dim=0)
    D = (preds - mean) / math.sqrt(max(n_samples - 1, 1))
    U, S, _ = torch.linalg.svd(D.T, full_matrices=False)
    kk = min(k, S.numel())
    d = S[:kk] ** 2
    tau = max(floor_rel * float(d.mean().clamp(min=1e-30)), 1e-30)
    return LowRankGaussian(mean, U[:, :kk].contiguous(), d, tau, label="swag")


def _load_flat(module, flat: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in module.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def build_baseline_fn(name: str, ctx: dict):
    """Return `c -> LowRankGaussian` for the named baseline."""
    fm, k = ctx["fm"], ctx.get("k", 64)
    if name == "isotropic":
        return lambda c: isotropic_prior(fm, c)
    if name == "diagonal_fitted":
        return lambda c: diagonal_prior(fm, c, ctx["res_stats"], k=k)
    if name == "fixed_global_lowrank":
        return lambda c: fixed_global_lowrank_prior(fm, c, ctx["res_stats"])
    if name == "deep_ensemble":
        return lambda c: ensemble_prior(ctx["ensemble"], c, k=k)
    if name == "mc_dropout":
        return lambda c: mc_dropout_prior(ctx["fm_do"], c,
                                          n_samples=ctx.get("n_dropout", 32), k=k)
    if name == "swag":
        return lambda c: swag_prior(fm, ctx["swag"], c,
                                    n_samples=ctx.get("n_swag", 24), k=k)
    raise ValueError(f"unknown baseline {name!r}; choices: {BASELINES_SCALE}")


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------

class SplitConformal:
    """Distribution-free per-pixel prediction intervals.

    Two variants, both standard:

      absolute     score = |y - yhat|, one global quantile. Guarantees marginal
                   coverage but produces a constant-width band, so it cannot
                   express any state dependence at all -- which makes it the
                   right floor for "does per-state information help".

      normalized   score = |y - yhat| / sigma(x), with sigma from any of the
                   Gaussian methods above. This is where conformal and HILP
                   *combine* rather than compete: the curvature supplies the
                   shape, conformal supplies a coverage guarantee the Gaussian
                   assumption does not provide. Reporting the normalized variant
                   on top of the pushforward covariance is the strongest form of
                   the method's claim.

    Coverage holds marginally over the exchangeable calibration+test draw. Pixels
    within a state are *not* exchangeable, so this is a guarantee about the
    pooled pixel population, not per-state simultaneous coverage; stating it any
    more strongly would be wrong.
    """

    def __init__(self, alpha: float = 0.1, normalized: bool = False):
        self.alpha = float(alpha)
        self.normalized = bool(normalized)
        self.q: float | None = None
        self.n_cal = 0

    def fit(self, residuals: np.ndarray, sigmas: np.ndarray | None = None
            ) -> "SplitConformal":
        r = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
        if self.normalized:
            if sigmas is None:
                raise ValueError("normalized conformal needs sigmas")
            r = r / np.maximum(np.asarray(sigmas, dtype=np.float64).reshape(-1), 1e-30)
        n = r.size
        # Finite-sample corrected quantile level: ceil((n+1)(1-alpha))/n.
        level = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q = float(np.quantile(r, level, method="higher"))
        self.n_cal = n
        return self

    def interval(self, mu: np.ndarray, sigma: np.ndarray | None = None):
        if self.q is None:
            raise RuntimeError("call fit() first")
        half = self.q * (np.asarray(sigma) if self.normalized else 1.0)
        return mu - half, mu + half

    def report(self, mu: np.ndarray, y: np.ndarray,
               sigma: np.ndarray | None = None) -> dict:
        lo, hi = self.interval(mu, sigma)
        cov = float(np.mean((y >= lo) & (y <= hi)))
        width = float(np.mean(hi - lo))
        return {"target_coverage": 1 - self.alpha, "coverage": cov,
                "mean_width": width, "q": self.q, "n_cal": self.n_cal,
                "normalized": self.normalized,
                # Interval score at the same level, so conformal and the
                # Gaussian methods can be compared on one proper scoring rule.
                "interval_score": float(np.mean(
                    (hi - lo)
                    + 2 / self.alpha * (lo - y) * (y < lo)
                    + 2 / self.alpha * (y - hi) * (y > hi)))}
