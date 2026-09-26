"""Differentiable 2D physics residual energy, the scaled counterpart of `hipp/physics.py`.

    R(w) = (w - w_t)/dt - rhs((w + w_t)/2),      E_phys = 0.5 ||R||^2 / N

Midpoint (Crank-Nicolson) in time so the residual is second-order accurate and
does not systematically penalize the true solution -- the same construction as
the 1D code, for the same reason.

What changes at scale is cost, not form. E_phys and its gradient are two rfft2 /
irfft2 pairs, i.e. O(N log N), which is cheaper than a single forward pass of
the surrogate. That matters: HMC needs the gradient at every leapfrog step, so
if the physics term were the bottleneck the posterior would be unusable at
128x128 regardless of how well the covariance scaled.

`grad` uses autograd through the spectral operations rather than a hand-derived
adjoint. The FFT is linear and torch differentiates it exactly, so there is no
accuracy loss, and it keeps the residual and its gradient from drifting apart
when the specification changes.
"""
from __future__ import annotations

import torch

from .data2d import PDESpec2D, SpectralGrid2D


class PhysicsEnergy2D:
    """E_phys(w) for a fixed conditioning state w_t on a 2D periodic grid.

    Instantiate once per test state, then call repeatedly inside a sampler or
    optimizer. The grid (wavenumbers, dealias mask, forcing field) is built once
    and shared, which is the difference between this being negligible and being
    the dominant cost of a rollout study.
    """

    def __init__(self, spec: PDESpec2D, w_t: torch.Tensor, dt: float | None = None,
                 substeps: int = 1, dtype=torch.float64, device=None):
        self.spec = spec
        self.dt = float(dt if dt is not None else spec.dt_out)
        self.substeps = int(substeps)
        self.n = spec.n
        self.N = spec.N
        device = device or w_t.device
        self.grid = SpectralGrid2D(spec, device=device, dtype=dtype)
        self.w_t = w_t.detach().to(dtype).reshape(self.n, self.n)
        self.dtype = dtype

    # ---- residual -------------------------------------------------------
    def _rhs(self, w: torch.Tensor) -> torch.Tensor:
        """rhs of w_t = rhs(w), physical space, batched over leading dim."""
        wh = torch.fft.rfft2(w)
        return torch.fft.irfft2(self.grid.rhs_hat(wh), s=(self.n, self.n))

    def residual(self, w: torch.Tensor) -> torch.Tensor:
        """Pointwise residual field, (B, n, n). Accepts flat or gridded input."""
        w = w.to(self.dtype).reshape(-1, self.n, self.n)
        wt = self.w_t.unsqueeze(0).expand_as(w)
        if self.substeps == 1:
            return (w - wt) / self.dt - self._rhs(0.5 * (w + wt))
        # Several midpoints along the straight line from w_t to w. Worth using
        # when dt_out is long relative to the flow's fastest timescale, which is
        # the usual situation for a foundation model trained on coarse output
        # cadence -- a single midpoint then under-resolves the advection.
        acc = 0.0
        for s in range(self.substeps):
            a = (s + 0.5) / self.substeps
            mid = (1 - a) * wt + a * w
            acc = acc + ((w - wt) / self.dt - self._rhs(mid)) ** 2
        return (acc / self.substeps).sqrt()

    def energy(self, w: torch.Tensor) -> torch.Tensor:
        r = self.residual(w)
        return 0.5 * r.flatten(1).pow(2).sum(dim=1) / self.N

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return self.energy(w)

    def grad(self, w: torch.Tensor) -> torch.Tensor:
        """dE/dw, flattened to (B, N) to match the Gaussian's convention."""
        x = w.detach().to(self.dtype).reshape(-1, self.N).requires_grad_(True)
        e = self.energy(x).sum()
        (g,) = torch.autograd.grad(e, x)
        return g

    def energy_and_grad(self, w: torch.Tensor):
        x = w.detach().to(self.dtype).reshape(-1, self.N).requires_grad_(True)
        e = self.energy(x)
        (g,) = torch.autograd.grad(e.sum(), x)
        return e.detach(), g

    def rms_residual(self, w: torch.Tensor) -> torch.Tensor:
        return self.residual(w).flatten(1).pow(2).mean(dim=1).sqrt()

    # ---- physical diagnostics ------------------------------------------
    def enstrophy(self, w: torch.Tensor) -> torch.Tensor:
        return 0.5 * w.reshape(-1, self.N).pow(2).mean(dim=1)

    def energy_spectrum(self, w: torch.Tensor, n_bins: int = 32) -> torch.Tensor:
        """Isotropically binned kinetic-energy spectrum E(k).

        The check that a sampled or perturbed field is still physically
        admissible: a Gaussian perturbation drawn from a covariance whose
        geometry is wrong shows up here as energy piled at the grid scale, which
        is far more legible than any scalar residual.
        """
        w = w.to(self.dtype).reshape(-1, self.n, self.n)
        wh = torch.fft.rfft2(w)
        e = (wh.abs() ** 2) * self.grid.inv_k2            # |psi_hat|^2 * k^2 = |w_hat|^2/k^2
        kk = self.grid.k2.sqrt()
        kmax = float(kk.max())
        idx = (kk / kmax * (n_bins - 1)).round().long().clamp(0, n_bins - 1)
        out = torch.zeros(w.shape[0], n_bins, dtype=self.dtype, device=w.device)
        out.scatter_add_(1, idx.reshape(1, -1).expand(w.shape[0], -1),
                         e.reshape(w.shape[0], -1))
        return out


def spectral_band_energy(w: torch.Tensor, n: int, lo: float, hi: float) -> torch.Tensor:
    """Fraction of enstrophy in the wavenumber band [lo, hi) * k_max.

    Used by the geometry stage: the claim that leading curvature directions
    match the physically most-amplified modes becomes, in 2D, a statement about
    which band the leading eigenvectors occupy.
    """
    w = w.reshape(-1, n, n)
    wh = torch.fft.rfft2(w)
    kx = torch.fft.fftfreq(n, device=w.device).view(-1, 1) * n
    ky = torch.fft.rfftfreq(n, device=w.device).view(1, -1) * n
    kk = (kx ** 2 + ky ** 2).sqrt()
    kmax = float(kk.max())
    mask = ((kk >= lo * kmax) & (kk < hi * kmax)).to(wh.real.dtype)
    tot = (wh.abs() ** 2).flatten(1).sum(1).clamp(min=1e-30)
    return ((wh.abs() ** 2) * mask).flatten(1).sum(1) / tot
