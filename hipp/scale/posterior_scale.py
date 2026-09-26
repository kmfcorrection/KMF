"""Physics posterior backed by a scalable low-rank Gaussian prior."""
from __future__ import annotations

import math
import torch


class LowRankPhysicsPosterior:
    def __init__(self, prior, energy, lam: float):
        self.prior, self.energy, self.lam = prior, energy, float(lam)
        self.N, self.dtype, self.device = prior.N, prior.mean.dtype, prior.mean.device

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z = x.reshape(-1, self.N).to(self.device, self.dtype)
        return -0.5 * self.prior.mahalanobis_sq(z) - self.lam * self.energy.energy(z)

    def grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z = x.reshape(-1, self.N).to(self.device, self.dtype)
        return self.prior.grad_log_prob(z) - self.lam * self.energy.grad(z)

    def map_estimate(self, n_steps: int = 40, lr: float = 0.5) -> tuple[torch.Tensor, dict]:
        x = self.prior.mean.clone().requires_grad_(True)
        initial = float((-self.log_prob(x)).detach())
        opt = torch.optim.LBFGS([x], lr=lr, max_iter=n_steps, history_size=10,
                                tolerance_grad=1e-9, tolerance_change=1e-11,
                                line_search_fn="strong_wolfe")
        calls = [0]
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = -self.log_prob(x).sum()
            loss.backward()
            calls[0] += 1
            return loss
        opt.step(closure)
        final = float((-self.log_prob(x)).detach())
        return x.detach(), {"objective_initial": initial, "objective_final": final,
                            "objective_decrease": initial - final,
                            "function_evals": calls[0],
                            "finite": bool(torch.isfinite(x).all())}


def covariance_matvec(prior, v: torch.Tensor) -> torch.Tensor:
    v = v.reshape(-1, prior.N).to(prior.mean)
    out = prior.tau * v
    if prior.k:
        out = out + (v @ prior.U * prior.d) @ prior.U.T
    return prior.alpha * out


def force_scale(prior, energy) -> float:
    """Lambda giving an approximately one-sigma natural-gradient displacement."""
    g = energy.grad(prior.mean).reshape(-1)
    sg = covariance_matvec(prior, g).reshape(-1)
    norm2 = float((g * sg).sum().clamp(min=1e-300))
    return 1.0 / math.sqrt(norm2)
