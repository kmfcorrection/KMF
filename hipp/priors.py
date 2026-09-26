"""Gaussian objects: N(mean, alpha * H^{-1}).

The scale `alpha` is carried separately from the shape `H` throughout, because
the whole method determines the *shape* of the covariance and leaves the scale
to a one-parameter post-hoc calibration (see docs/method.md section 2.3).
"""
from __future__ import annotations

import math

import torch

JITTER = 1e-10


def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def safe_cholesky(A: torch.Tensor, max_tries: int = 8) -> torch.Tensor:
    """Cholesky with escalating jitter; falls back to an eigen-clip."""
    A = _sym(A)
    scale = torch.diagonal(A, dim1=-2, dim2=-1).mean().clamp(min=1e-30)
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    jit = JITTER
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(A + jit * scale * eye)
        except Exception:
            jit *= 10
    evals, evecs = torch.linalg.eigh(A)
    evals = evals.clamp(min=1e-8 * float(evals.max().clamp(min=1e-30)))
    return torch.linalg.cholesky(_sym(evecs @ torch.diag(evals) @ evecs.T))


class GaussianPrior:
    """Dense Gaussian in N dimensions, parameterized by a precision *shape*.

    Sigma = alpha * H^{-1}.  Everything is computed in float64.
    """

    def __init__(self, mean: torch.Tensor, H: torch.Tensor, alpha: float = 1.0,
                 label: str = ""):
        self.mean = mean.detach().to(torch.float64).reshape(-1)
        self.H = _sym(H.detach().to(torch.float64))
        self.alpha = float(alpha)
        self.label = label
        self.N = self.mean.numel()
        self._chol = None
        self._eig = None

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_covariance(cls, mean, Sigma, alpha: float = 1.0, label: str = ""):
        Sigma = _sym(Sigma.detach().to(torch.float64))
        L = safe_cholesky(Sigma)
        eye = torch.eye(Sigma.shape[-1], dtype=Sigma.dtype, device=Sigma.device)
        H = torch.cholesky_solve(eye, L)
        return cls(mean, H, alpha=alpha, label=label)

    @classmethod
    def isotropic(cls, mean, alpha: float = 1.0, label: str = "isotropic"):
        eye = torch.eye(mean.numel(), dtype=torch.float64, device=mean.device)
        return cls(mean, eye, alpha=alpha, label=label)

    @classmethod
    def diagonal(cls, mean, var: torch.Tensor, alpha: float = 1.0, label: str = "diagonal"):
        var = var.detach().to(torch.float64).reshape(-1).clamp(min=1e-30)
        return cls(mean, torch.diag(1.0 / var), alpha=alpha, label=label)

    # ---- cached factorizations ----------------------------------------
    @property
    def chol_H(self) -> torch.Tensor:
        if self._chol is None:
            self._chol = safe_cholesky(self.H)
        return self._chol

    def eig(self):
        """Eigendecomposition of the *shape* H (ascending eigenvalues)."""
        if self._eig is None:
            self._eig = torch.linalg.eigh(self.H)
        return self._eig

    # ---- quantities ----------------------------------------------------
    def precision(self) -> torch.Tensor:
        return self.H / self.alpha

    def covariance(self) -> torch.Tensor:
        eye = torch.eye(self.N, dtype=self.H.dtype, device=self.H.device)
        return self.alpha * torch.cholesky_solve(eye, self.chol_H)

    def marginal_var(self) -> torch.Tensor:
        return torch.diagonal(self.covariance()).clamp(min=1e-30)

    def marginal_std(self) -> torch.Tensor:
        return self.marginal_var().sqrt()

    def logdet_cov(self) -> float:
        ld_H = 2.0 * torch.log(torch.diagonal(self.chol_H)).sum()
        return float(self.N * math.log(self.alpha) - ld_H)

    def quad(self, x: torch.Tensor) -> torch.Tensor:
        """(x - mean)^T H (x - mean), unscaled by alpha. Batched over rows."""
        d = (x.to(torch.float64).reshape(-1, self.N) - self.mean)
        return torch.einsum("bi,ij,bj->b", d, self.H, d)

    def mahalanobis_sq(self, x: torch.Tensor) -> torch.Tensor:
        return self.quad(x) / self.alpha

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (self.mahalanobis_sq(x) + self.logdet_cov()
                       + self.N * math.log(2 * math.pi))

    def sample(self, n: int = 1, generator=None) -> torch.Tensor:
        z = torch.randn(n, self.N, dtype=self.H.dtype, device=self.H.device,
                        generator=generator)
        # Sigma = alpha H^{-1} = alpha (LL^T)^{-1}  =>  x = mean + sqrt(alpha) L^{-T} z
        w = torch.linalg.solve_triangular(self.chol_H.T, z.T, upper=True).T
        return self.mean + math.sqrt(self.alpha) * w

    def grad_logprob(self, x: torch.Tensor) -> torch.Tensor:
        d = x.to(torch.float64).reshape(-1, self.N) - self.mean
        return -(d @ self.H) / self.alpha

    def rescaled(self, alpha: float) -> "GaussianPrior":
        return GaussianPrior(self.mean, self.H, alpha=alpha, label=self.label)

    def condition_number(self) -> float:
        ev = self.eig()[0].clamp(min=1e-300)
        return float(ev[-1] / ev[0])

    def __repr__(self):
        return f"GaussianPrior(N={self.N}, alpha={self.alpha:.4g}, label={self.label!r})"
