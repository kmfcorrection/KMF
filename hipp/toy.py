"""Stage 9: low-dimensional problems with a computable exact posterior.

Purpose: isolate the quality of the Laplace/Gauss-Newton *approximation* from
every foundation-model confound. In 2D the posterior can be computed by
quadrature on a grid, so KL, Wasserstein and coverage are exact rather than
estimated.

Four problems, ordered by how badly they should break the approximation:
  linear   -> posterior exactly Gaussian; KL must be ~0 (correctness check)
  mild     -> weak nonlinearity; the regime the method assumes
  banana   -> strongly curved ridge; Laplace keeps the mode, loses the shape
  bimodal  -> two modes; a single Gaussian cannot represent it, by construction
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ToyProblem:
    name: str
    theta_true: np.ndarray
    y: np.ndarray
    noise_std: float
    prior_std: float

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ---- densities ----------------------------------------------------
    def log_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        g = self.forward(theta)
        y = torch.as_tensor(self.y, dtype=theta.dtype, device=theta.device)
        return -0.5 * ((g - y) ** 2).sum(-1) / self.noise_std ** 2

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        return -0.5 * (theta ** 2).sum(-1) / self.prior_std ** 2

    def log_posterior(self, theta: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood(theta) + self.log_prior(theta)

    # ---- geometry ------------------------------------------------------
    def jacobian(self, theta: torch.Tensor) -> torch.Tensor:
        th = theta.reshape(-1).clone().detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(lambda t: self.forward(t.reshape(1, -1))
                                                  .reshape(-1), th)

    def gauss_newton(self, theta: torch.Tensor) -> torch.Tensor:
        """H_GN = J^T J / sigma^2 + prior precision -- the toy analogue of the
        method's H = J^T J (no second-derivative term)."""
        J = self.jacobian(theta)
        d = theta.numel()
        eye = torch.eye(d, dtype=J.dtype, device=J.device)
        return J.T @ J / self.noise_std ** 2 + eye / self.prior_std ** 2

    def exact_hessian(self, theta: torch.Tensor) -> torch.Tensor:
        """Full Hessian of the negative log posterior (includes the term the
        Gauss-Newton approximation drops)."""
        th = theta.reshape(1, -1).clone().detach().requires_grad_(True)
        return torch.autograd.functional.hessian(
            lambda t: -self.log_posterior(t.reshape(1, -1)).sum(), th
        ).reshape(theta.numel(), theta.numel())

    def map_estimate(self, x0=None, n_restarts: int = 8, seed: int = 0) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        best, best_val = None, np.inf
        starts = [np.zeros(2)] if x0 is None else [np.asarray(x0)]
        starts += [rng.normal(0, self.prior_std, 2) for _ in range(n_restarts)]
        for s in starts:
            th = torch.tensor(s, dtype=torch.float64).reshape(1, -1).requires_grad_(True)
            opt = torch.optim.LBFGS([th], max_iter=300, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = -self.log_posterior(th).sum()
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                v = float(-self.log_posterior(th).sum())
            if np.isfinite(v) and v < best_val:
                best_val, best = v, th.detach().reshape(-1).clone()
        return best


class LinearToy(ToyProblem):
    def __init__(self, **kw):
        A = np.array([[1.0, 0.5], [0.2, -1.0], [0.7, 0.3]])
        self.A = A
        theta_true = np.array([0.8, -0.4])
        noise_std, prior_std = kw.get("noise_std", 0.3), kw.get("prior_std", 2.0)
        rng = np.random.default_rng(kw.get("seed", 0))
        y = A @ theta_true + noise_std * rng.standard_normal(3)
        super().__init__("linear", theta_true, y, noise_std, prior_std)

    def forward(self, theta):
        A = torch.as_tensor(self.A, dtype=theta.dtype, device=theta.device)
        return theta.reshape(-1, 2) @ A.T


class MildToy(ToyProblem):
    def __init__(self, **kw):
        self.A = np.array([[1.0, 0.5], [0.2, -1.0], [0.7, 0.3]])
        self.eta = kw.get("eta", 0.25)
        theta_true = np.array([0.8, -0.4])
        noise_std, prior_std = kw.get("noise_std", 0.3), kw.get("prior_std", 2.0)
        rng = np.random.default_rng(kw.get("seed", 0))
        y = (self.A @ theta_true
             + self.eta * theta_true[0] ** 2 * np.array([1.0, 0.0, 0.5])
             + noise_std * rng.standard_normal(3))
        super().__init__("mild", theta_true, y, noise_std, prior_std)

    def forward(self, theta):
        th = theta.reshape(-1, 2)
        A = torch.as_tensor(self.A, dtype=th.dtype, device=th.device)
        b = torch.as_tensor([1.0, 0.0, 0.5], dtype=th.dtype, device=th.device)
        return th @ A.T + self.eta * (th[:, 0:1] ** 2) * b


class BananaToy(ToyProblem):
    def __init__(self, **kw):
        theta_true = np.array([0.5, 0.6])
        noise_std, prior_std = kw.get("noise_std", 0.4), kw.get("prior_std", 3.0)
        rng = np.random.default_rng(kw.get("seed", 0))
        y = np.array([theta_true[0] + theta_true[1] ** 2]) + noise_std * rng.standard_normal(1)
        super().__init__("banana", theta_true, y, noise_std, prior_std)

    def forward(self, theta):
        th = theta.reshape(-1, 2)
        return (th[:, 0:1] + th[:, 1:2] ** 2)


class BimodalToy(ToyProblem):
    def __init__(self, **kw):
        theta_true = np.array([1.2, 0.0])
        noise_std, prior_std = kw.get("noise_std", 0.3), kw.get("prior_std", 2.0)
        rng = np.random.default_rng(kw.get("seed", 0))
        y = np.array([theta_true[0] ** 2 + 0.5 * theta_true[1]]) + noise_std * rng.standard_normal(1)
        super().__init__("bimodal", theta_true, y, noise_std, prior_std)

    def forward(self, theta):
        th = theta.reshape(-1, 2)
        return th[:, 0:1] ** 2 + 0.5 * th[:, 1:2]


PROBLEMS = {"linear": LinearToy, "mild": MildToy, "banana": BananaToy, "bimodal": BimodalToy}


# ---------------------------------------------------------------------------
# Exact posterior by quadrature
# ---------------------------------------------------------------------------

def grid_posterior(problem: ToyProblem, lim: float = 4.0, n: int = 401):
    g = torch.linspace(-lim, lim, n, dtype=torch.float64)
    T1, T2 = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([T1.reshape(-1), T2.reshape(-1)], dim=-1)
    with torch.no_grad():
        lp = problem.log_posterior(pts)
    lp = lp - lp.max()
    p = torch.exp(lp).reshape(n, n)
    dA = float((g[1] - g[0]) ** 2)
    p = p / (p.sum() * dA)
    return {"grid": g, "T1": T1, "T2": T2, "pdf": p, "dA": dA, "pts": pts}


def gaussian_pdf_on_grid(mean: torch.Tensor, cov: torch.Tensor, grid: dict) -> torch.Tensor:
    d = grid["pts"] - mean.reshape(1, -1)
    L = torch.linalg.cholesky(0.5 * (cov + cov.T))
    sol = torch.linalg.solve_triangular(L, d.T, upper=False)
    q = torch.exp(-0.5 * (sol ** 2).sum(0)) / (2 * np.pi * torch.linalg.det(L))
    n = grid["grid"].numel()
    return q.reshape(n, n)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, dA: float) -> float:
    """KL(p || q) by quadrature."""
    p_ = p.clamp(min=1e-300)
    q_ = q.clamp(min=1e-300)
    return float((p_ * (torch.log(p_) - torch.log(q_))).sum() * dA)


def grid_moments(p: torch.Tensor, grid: dict):
    pts = grid["pts"]
    w = (p.reshape(-1) * grid["dA"])
    w = w / w.sum()
    mean = (w[:, None] * pts).sum(0)
    d = pts - mean
    cov = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    return mean, cov


def wasserstein2_gaussian(m1, C1, m2, C2) -> float:
    """Exact W2 between two Gaussians (Bures metric)."""
    from scipy.linalg import sqrtm
    C1n, C2n = C1.cpu().numpy(), C2.cpu().numpy()
    s = sqrtm(C1n)
    cross = sqrtm(s @ C2n @ s)
    if np.iscomplexobj(cross):
        cross = cross.real
    bures = np.trace(C1n) + np.trace(C2n) - 2 * np.trace(cross)
    dm = float(((m1 - m2) ** 2).sum())
    return float(np.sqrt(max(dm + bures, 0.0)))


def credible_coverage(p_exact: torch.Tensor, mean, cov, grid: dict,
                      levels=(0.5, 0.9, 0.95)) -> dict:
    """How much exact-posterior mass falls inside the Gaussian's nominal
    credible ellipses? The direct calibration statement for Stage 9."""
    from scipy.stats import chi2
    d = grid["pts"] - mean.reshape(1, -1)
    prec = torch.linalg.inv(0.5 * (cov + cov.T))
    m2 = torch.einsum("bi,ij,bj->b", d, prec, d)
    w = (p_exact.reshape(-1) * grid["dA"])
    w = w / w.sum()
    return {f"coverage@{lv:g}": float((w * (m2 <= chi2.ppf(lv, df=2))).sum())
            for lv in levels}
