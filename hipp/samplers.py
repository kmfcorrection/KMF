"""MCMC samplers and convergence diagnostics.

All samplers take an object exposing `log_prob(x)` and `grad_log_prob(x)` for
x of shape (1, N), which both HessianPhysicsPosterior and GaussianPrior-backed
targets satisfy.

The mass matrix for HMC defaults to the curvature estimate H itself. This is
worth stating plainly: even if H turns out to be a poor *prior*, it is a
perfectly good *preconditioner*, and that is a separate (easier) claim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .priors import safe_cholesky


@dataclass
class Chain:
    samples: torch.Tensor            # (n_keep, N)
    accept_rate: float
    step_size: float
    n_grad_evals: int
    diverged: int = 0
    info: dict = field(default_factory=dict)

    def mean(self) -> torch.Tensor:
        return self.samples.mean(0)

    def cov(self) -> torch.Tensor:
        S = self.samples - self.samples.mean(0, keepdim=True)
        return (S.T @ S) / max(S.shape[0] - 1, 1)


class DualAveraging:
    """Nesterov dual averaging for step-size adaptation (Hoffman & Gelman)."""

    def __init__(self, target: float, init_step: float, gamma: float = 0.05,
                 t0: float = 10.0, kappa: float = 0.75):
        self.mu = math.log(10 * init_step)
        self.target, self.gamma, self.t0, self.kappa = target, gamma, t0, kappa
        self.log_eps_bar, self.h_bar, self.t = 0.0, 0.0, 0

    def update(self, accept_prob: float) -> float:
        self.t += 1
        eta = 1.0 / (self.t + self.t0)
        self.h_bar = (1 - eta) * self.h_bar + eta * (self.target - accept_prob)
        log_eps = self.mu - math.sqrt(self.t) / self.gamma * self.h_bar
        w = self.t ** (-self.kappa)
        self.log_eps_bar = w * log_eps + (1 - w) * self.log_eps_bar
        return math.exp(log_eps)

    def final(self) -> float:
        return math.exp(self.log_eps_bar)


def _as_row(x: torch.Tensor, N: int) -> torch.Tensor:
    return x.reshape(1, N)


def geometric_step_size(precision: torch.Tensor, safety: float = 0.5) -> float:
    """A step size matched to the *stiffest* direction of the target.

    Langevin stability is governed by the largest eigenvalue of the precision:
    the step must resolve the smallest standard deviation, sigma_min =
    lambda_max(H)^-1/2. A fixed absolute step (say 1e-2) is meaningless here,
    because a calibrated prior precision routinely reaches 1e9 -- ULA simply
    diverges. Returns safety * sigma_min.
    """
    ev = torch.linalg.eigvalsh(0.5 * (precision + precision.T))
    lam_max = float(ev[-1].clamp(min=1e-300))
    return safety * lam_max ** -0.5


# ---------------------------------------------------------------------------
# Langevin
# ---------------------------------------------------------------------------

def langevin(target, x0: torch.Tensor, n_samples: int = 2000, burn_in: int = 500,
             step_size: float = 1e-3, metropolis: bool = True, thin: int = 1,
             adapt: bool = True, generator=None) -> Chain:
    """MALA when `metropolis=True`, unadjusted Langevin (ULA) otherwise."""
    N = x0.numel()
    x = _as_row(x0.clone().to(torch.float64), N)
    lp = target.log_prob(x)[0]
    g = target.grad_log_prob(x)
    eps = step_size
    da = DualAveraging(0.574, eps) if (adapt and metropolis) else None
    kept, n_acc, n_grad, n_div = [], 0, 1, 0
    total = burn_in + n_samples
    for it in range(total):
        noise = torch.randn(1, N, dtype=x.dtype, device=x.device, generator=generator)
        prop = x + 0.5 * eps ** 2 * g + eps * noise
        lp_p = target.log_prob(prop)[0]
        g_p = target.grad_log_prob(prop)
        n_grad += 1
        if metropolis:
            fwd = -((prop - x - 0.5 * eps ** 2 * g) ** 2).sum() / (2 * eps ** 2)
            bwd = -((x - prop - 0.5 * eps ** 2 * g_p) ** 2).sum() / (2 * eps ** 2)
            log_ratio = float(lp_p - lp + bwd - fwd)
            a = min(1.0, math.exp(min(log_ratio, 0.0))) if np.isfinite(log_ratio) else 0.0
            if torch.rand(1, device=x.device, generator=generator).item() < a:
                x, lp, g = prop, lp_p, g_p
                n_acc += 1
            if da is not None and it < burn_in:
                eps = da.update(a)
            elif da is not None and it == burn_in:
                eps = da.final()
        else:
            # ULA is unadjusted, so nothing stops it from walking off to
            # infinity when the step size is too large for the local geometry.
            # Detect that and report it rather than returning the blown-up
            # state as if it were a sample.
            if not torch.isfinite(lp_p) or float(lp_p) < float(lp) - 1e8:
                n_div += 1
                break
            x, lp, g = prop, lp_p, g_p
            n_acc += 1
        if it >= burn_in and (it - burn_in) % thin == 0:
            kept.append(x.clone().reshape(-1))
    return Chain(torch.stack(kept) if kept else x.reshape(1, -1),
                 n_acc / max(total, 1), eps, n_grad, diverged=n_div,
                 info={"algorithm": "mala" if metropolis else "ula"})


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

def hmc(target, x0: torch.Tensor, n_samples: int = 1000, burn_in: int = 500,
        step_size: float = 0.1, n_leapfrog: int = 20, mass: torch.Tensor | None = None,
        adapt: bool = True, thin: int = 1, jitter: bool = True,
        generator=None) -> Chain:
    """Hamiltonian Monte Carlo with an optional dense mass matrix.

    `mass` is the metric M in K(p) = 0.5 p^T M^{-1} p; setting M = H makes the
    Gaussian part of the target isotropic in the transformed coordinates.
    """
    N = x0.numel()
    dtype, device = torch.float64, x0.device
    x = _as_row(x0.clone().to(dtype), N)

    if mass is not None:
        M = mass.to(dtype)
        L = safe_cholesky(M)                    # M = L L^T
        eye = torch.eye(N, dtype=dtype, device=device)
        Minv = torch.cholesky_solve(eye, L)

        def sample_p():
            z = torch.randn(1, N, dtype=dtype, device=device, generator=generator)
            return z @ L.T                       # p ~ N(0, M)

        def kinetic(p):
            return 0.5 * float((p @ Minv * p).sum())

        def dK(p):
            return p @ Minv
    else:
        def sample_p():
            return torch.randn(1, N, dtype=dtype, device=device, generator=generator)

        def kinetic(p):
            return 0.5 * float((p ** 2).sum())

        def dK(p):
            return p

    eps = step_size
    da = DualAveraging(0.8, eps) if adapt else None
    kept, n_acc, n_grad, n_div = [], 0, 0, 0
    total = burn_in + n_samples

    for it in range(total):
        p = sample_p()
        x_new, p_new = x.clone(), p.clone()
        h0 = -float(target.log_prob(x)[0]) + kinetic(p)
        step = eps * (0.9 + 0.2 * torch.rand(
            1, device=device, generator=generator).item()) if jitter else eps

        g = target.grad_log_prob(x_new)
        n_grad += 1
        p_new = p_new + 0.5 * step * g
        ok = True
        for l in range(n_leapfrog):
            x_new = x_new + step * dK(p_new)
            g = target.grad_log_prob(x_new)
            n_grad += 1
            if not torch.isfinite(g).all():
                ok = False
                break
            p_new = p_new + (step if l < n_leapfrog - 1 else 0.5 * step) * g

        if ok:
            lp_new = target.log_prob(x_new)[0]
            h1 = -float(lp_new) + kinetic(p_new)
            dH = h0 - h1
            a = min(1.0, math.exp(min(dH, 0.0))) if np.isfinite(dH) else 0.0
            if abs(dH) > 1000:
                n_div += 1
        else:
            a, n_div = 0.0, n_div + 1

        if torch.rand(1, device=device, generator=generator).item() < a:
            x = x_new
            n_acc += 1
        if da is not None and it < burn_in:
            eps = da.update(a)
        elif da is not None and it == burn_in:
            eps = da.final()
        if it >= burn_in and (it - burn_in) % thin == 0:
            kept.append(x.clone().reshape(-1))

    return Chain(torch.stack(kept) if kept else x.reshape(1, -1),
                 n_acc / max(total, 1), eps, n_grad, diverged=n_div,
                 info={"algorithm": "hmc", "n_leapfrog": n_leapfrog,
                       "mass": "dense" if mass is not None else "identity"})


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def effective_sample_size(chain: torch.Tensor) -> np.ndarray:
    """Per-coordinate ESS via the initial-positive-sequence estimator."""
    x = chain.detach().cpu().numpy().astype(np.float64)
    n, d = x.shape
    ess = np.empty(d)
    for j in range(d):
        v = x[:, j] - x[:, j].mean()
        var = np.dot(v, v) / n
        if var <= 0 or not np.isfinite(var):
            ess[j] = float(n)
            continue
        f = np.fft.rfft(v, n=2 * n)
        acf = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n].real
        acf /= acf[0]
        s, t = 0.0, 1
        while t + 1 < n:
            pair = acf[t] + acf[t + 1]
            if pair <= 0:
                break
            s += pair
            t += 2
        ess[j] = n / max(1.0 + 2.0 * s, 1e-12)
    return np.clip(ess, 1.0, float(n))


def r_hat(chains: list[torch.Tensor]) -> np.ndarray:
    """Split-free Gelman-Rubin R-hat across chains, per coordinate."""
    X = np.stack([c.detach().cpu().numpy() for c in chains])   # (m, n, d)
    m, n, _ = X.shape
    if m < 2:
        return np.full(X.shape[-1], np.nan)
    means, varis = X.mean(axis=1), X.var(axis=1, ddof=1)
    B = n * means.var(axis=0, ddof=1)
    W = varis.mean(axis=0)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(np.maximum(var_hat, 0) / np.maximum(W, 1e-30))


def run_multichain(target, x0s: list[torch.Tensor], sampler: str = "hmc",
                   **kw) -> tuple[list[Chain], dict]:
    fn = {"hmc": hmc, "mala": lambda *a, **k: langevin(*a, metropolis=True, **k),
          "ula": lambda *a, **k: langevin(*a, metropolis=False, **k)}[sampler]
    chains = [fn(target, x0, **kw) for x0 in x0s]
    ess = np.mean([effective_sample_size(c.samples) for c in chains], axis=0)
    diag = {
        "accept_rate": float(np.mean([c.accept_rate for c in chains])),
        "step_size": float(np.mean([c.step_size for c in chains])),
        "ess_mean": float(np.mean(ess)), "ess_min": float(np.min(ess)),
        "grad_evals": int(np.sum([c.n_grad_evals for c in chains])),
        "divergences": int(np.sum([c.diverged for c in chains])),
        "rhat_max": float(np.nanmax(r_hat([c.samples for c in chains]))),
    }
    return chains, diag
