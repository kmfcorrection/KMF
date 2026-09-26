"""Differentiable PDE residual energies E_phys(c).

The residual compares a candidate next state c against the known current state
c_t through a midpoint (Crank-Nicolson) discretization, so it is second-order
accurate in dt and does not systematically penalize the true solution:

    R(c) = (c - c_t)/dt - rhs((c + c_t)/2),      E_phys = 0.5 ||R||^2 / N

Spatial derivatives are spectral (periodic domain), so E_phys is smooth and its
gradient is cheap -- which matters because HMC needs it at every leapfrog step.
"""
from __future__ import annotations

import torch

from .data import PDESpec


def _dx(u: torch.Tensor, k: torch.Tensor, order: int) -> torch.Tensor:
    uh = torch.fft.rfft(u, dim=-1)
    mult = (1j * k) ** order
    if order % 2 == 0:
        mult = mult.real.to(uh.dtype)
    out = torch.fft.irfft(uh * mult, n=u.shape[-1], dim=-1)
    return out


def rhs(spec: PDESpec, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Right-hand side of u_t = rhs(u)."""
    if spec.name == "advdiff":
        return -spec.c_adv * _dx(u, k, 1) + spec.nu * _dx(u, k, 2)
    if spec.name == "burgers":
        return -u * _dx(u, k, 1) + spec.nu * _dx(u, k, 2)
    if spec.name == "ks":
        return -u * _dx(u, k, 1) - _dx(u, k, 2) - _dx(u, k, 4)
    raise ValueError(spec.name)


class PhysicsEnergy:
    """E_phys(c) for a fixed conditioning state c_t.

    Instantiate once per test sample, then call repeatedly inside a sampler.
    """

    def __init__(self, spec: PDESpec, c_t: torch.Tensor, dt: float | None = None,
                 substeps: int = 1, device=None, dtype=torch.float64):
        self.spec = spec
        self.dt = float(dt if dt is not None else spec.dt_out)
        self.substeps = substeps
        self.c_t = c_t.detach().to(dtype).reshape(-1)
        device = device or self.c_t.device
        self.k = torch.fft.rfftfreq(spec.N, d=spec.L / spec.N, device=device,
                                    dtype=dtype) * 2 * torch.pi
        self.N = spec.N

    def residual(self, c: torch.Tensor) -> torch.Tensor:
        """Pointwise residual field. Batched over leading dim."""
        c = c.reshape(-1, self.N)
        ct = self.c_t.expand_as(c)
        if self.substeps == 1:
            mid = 0.5 * (c + ct)
            return (c - ct) / self.dt - rhs(self.spec, mid, self.k)
        # Multi-substep variant: linear interpolation between c_t and c, which
        # is a better residual when dt_out is large relative to the PDE's
        # fastest timescale (the KS case).
        res = 0.0
        h = self.dt / self.substeps
        for s in range(self.substeps):
            a = (s + 0.5) / self.substeps
            mid = (1 - a) * ct + a * c
            res = res + ((c - ct) / self.dt - rhs(self.spec, mid, self.k)) ** 2
        return (res / self.substeps).sqrt() * torch.sign(c - ct + 1e-30)

    def energy(self, c: torch.Tensor) -> torch.Tensor:
        r = self.residual(c)
        return 0.5 * (r ** 2).sum(dim=-1) / self.N

    def __call__(self, c: torch.Tensor) -> torch.Tensor:
        return self.energy(c)

    def grad(self, c: torch.Tensor) -> torch.Tensor:
        c = c.detach().reshape(-1, self.N).requires_grad_(True)
        e = self.energy(c).sum()
        (g,) = torch.autograd.grad(e, c)
        return g

    def energy_and_grad(self, c: torch.Tensor):
        c = c.detach().reshape(-1, self.N).requires_grad_(True)
        e = self.energy(c)
        (g,) = torch.autograd.grad(e.sum(), c)
        return e.detach(), g

    def rms_residual(self, c: torch.Tensor) -> torch.Tensor:
        return self.residual(c).pow(2).mean(dim=-1).sqrt()
