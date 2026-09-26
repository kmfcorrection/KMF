"""The physics-constrained posterior

    pi(c) ~ exp( -0.5 (c - chat)^T H (c - chat) / alpha  -  lambda E_phys(c) )

plus its MAP estimate and, where the PDE is linear, its exact Gaussian form
(which is the correctness reference the samplers in Stage 7 must reproduce).
"""
from __future__ import annotations

import torch

from .physics import PhysicsEnergy
from .priors import GaussianPrior, safe_cholesky


class HessianPhysicsPosterior:
    def __init__(self, prior: GaussianPrior, energy: PhysicsEnergy | None = None,
                 lam: float = 0.0):
        self.prior = prior
        self.energy = energy
        self.lam = float(lam)
        self.N = prior.N
        self.dtype = prior.H.dtype
        self.device = prior.H.device

    # ---- density ------------------------------------------------------
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.N).to(self.dtype)
        lp = -0.5 * self.prior.mahalanobis_sq(x)
        if self.energy is not None and self.lam != 0.0:
            lp = lp - self.lam * self.energy(x)
        return lp

    def grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.N).to(self.dtype)
        g = self.prior.grad_logprob(x)
        if self.energy is not None and self.lam != 0.0:
            g = g - self.lam * self.energy.grad(x)
        return g

    def log_prob_and_grad(self, x: torch.Tensor):
        return self.log_prob(x), self.grad_log_prob(x)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x)

    # ---- point estimates ----------------------------------------------
    def map_estimate(self, x0: torch.Tensor | None = None, n_steps: int = 200,
                     lr: float = 0.5) -> torch.Tensor:
        x = (self.prior.mean.clone() if x0 is None else x0.reshape(-1).clone())
        x = x.to(self.dtype).requires_grad_(True)
        opt = torch.optim.LBFGS([x], lr=lr, max_iter=n_steps, history_size=20,
                                tolerance_grad=1e-12, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self.potential(x).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return x.detach()

    # ---- exact Gaussian reference (linear PDE) -------------------------
    def linearized_gaussian(self, x0: torch.Tensor | None = None) -> GaussianPrior:
        """Laplace approximation of the posterior around x0 (default: MAP).

        For a linear PDE the residual is affine, so this is the *exact*
        posterior and any correct sampler must match it.
        """
        if self.energy is None or self.lam == 0.0:
            return self.prior
        x0 = self.map_estimate() if x0 is None else x0.reshape(-1)
        x0 = x0.to(self.dtype)
        # Gauss-Newton on the residual: H_post = H_prior/alpha + lam * A^T A / N
        A = torch.autograd.functional.jacobian(
            lambda z: self.energy.residual(z).reshape(-1), x0.clone().requires_grad_(True)
        ).to(self.dtype)
        H_post = self.prior.precision() + self.lam * (A.T @ A) / self.N
        return GaussianPrior(x0, H_post, alpha=1.0, label="posterior_laplace")

    def posterior_sample_reference(self, n: int = 512) -> torch.Tensor:
        return self.linearized_gaussian().sample(n)


def reference_lambda(prior: GaussianPrior, energy: PhysicsEnergy) -> float:
    """The lambda at which the physics term and the prior contribute equal
    curvature.

    Absolute lambda values are meaningless across problems: the calibrated
    prior precision H/alpha spans many orders of magnitude depending on how
    accurate the model is (alpha ~ 1e-9 for a well-fit linear PDE), so a fixed
    grid like [1e-4 ... 1e2] either does nothing or swamps the prior. Sweeping
    lambda in units of this reference makes the trade-off comparable across
    PDEs and across models.

    Matches Gauss-Newton traces:  tr(H/alpha)  ==  lambda * tr(A^T A)/N,
    where A is the Jacobian of the residual.
    """
    x0 = prior.mean.clone().to(prior.H.dtype).requires_grad_(True)
    A = torch.autograd.functional.jacobian(
        lambda z: energy.residual(z).reshape(-1), x0).to(prior.H.dtype)
    phys_tr = float((A * A).sum()) / prior.N
    prior_tr = float(torch.diagonal(prior.precision()).sum())
    return prior_tr / max(phys_tr, 1e-300)


def build_posterior(prior: GaussianPrior, spec, c_t: torch.Tensor, lam: float,
                    substeps: int = 1) -> HessianPhysicsPosterior:
    energy = PhysicsEnergy(spec, c_t, substeps=substeps, device=prior.H.device)
    return HessianPhysicsPosterior(prior, energy, lam)
