"""Traditional, non-learned Navier--Stokes residual discretizations.

All operators assume a full periodic velocity field on ``[0,L)^2``.  They use
only the observed current field and a candidate next field; periodic wrapping
is a stated boundary condition, not extrapolated data.  These routines are
intended for residual-selection diagnostics before a likelihood is chosen.
"""
from __future__ import annotations

import math

import torch

from .data2d import PDESpec2D
from .fm_physics import FMPhysicsEnergy2D, spectral_resample_state


_D1 = {
    2: ((-1, -0.5), (1, 0.5)),
    4: ((-2, 1 / 12), (-1, -2 / 3), (1, 2 / 3), (2, -1 / 12)),
    6: ((-3, -1 / 60), (-2, 9 / 60), (-1, -45 / 60),
        (1, 45 / 60), (2, -9 / 60), (3, 1 / 60)),
}
_D2 = {
    2: ((-1, 1.0), (0, -2.0), (1, 1.0)),
    4: ((-2, -1 / 12), (-1, 4 / 3), (0, -5 / 2),
        (1, 4 / 3), (2, -1 / 12)),
    6: ((-3, 1 / 90), (-2, -3 / 20), (-1, 3 / 2), (0, -49 / 18),
        (1, 3 / 2), (2, -3 / 20), (3, 1 / 90)),
}


def _check_state(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-3] != 2 or x.shape[-2] != x.shape[-1]:
        raise ValueError("expected (..., 2, n, n) velocity fields")
    return x.to(dtype=torch.float64)


def periodic_derivative(field: torch.Tensor, *, axis: int, dx: float, order: int) -> torch.Tensor:
    """Periodic centered first derivative of order 2, 4, or 6."""
    if order not in _D1:
        raise ValueError("order must be 2, 4, or 6")
    dim = -2 if axis == 0 else -1
    out = torch.zeros_like(field)
    for offset, coefficient in _D1[order]:
        # roll(-offset) evaluates f[i + offset].
        out = out + coefficient * torch.roll(field, shifts=-offset, dims=dim)
    return out / dx


def periodic_laplacian(field: torch.Tensor, *, dx: float, order: int) -> torch.Tensor:
    """Periodic centered Laplacian of order 2, 4, or 6."""
    if order not in _D2:
        raise ValueError("order must be 2, 4, or 6")
    out = torch.zeros_like(field)
    for dim in (-2, -1):
        for offset, coefficient in _D2[order]:
            out = out + coefficient * torch.roll(field, shifts=-offset, dims=dim)
    return out / (dx * dx)


def vorticity_fd(state: torch.Tensor, *, L: float, order: int) -> torch.Tensor:
    """Finite-difference curl, ``omega = dv/dx - du/dy``."""
    state = _check_state(state)
    dx = L / state.shape[-1]
    u, v = state[..., 0, :, :], state[..., 1, :, :]
    return (periodic_derivative(v, axis=0, dx=dx, order=order) -
            periodic_derivative(u, axis=1, dx=dx, order=order))


def _spectral_viscosity_fd(omega: torch.Tensor, *, L: float, strength: float) -> torch.Tensor:
    """Traditional FD closure for AZEBAN's high-mode spectral viscosity.

    It deliberately uses a standard Laplacian rather than claiming to recreate
    AZEBAN's nonlocal smooth filter.  This is a conventional-model mismatch
    control, useful for asking how much the exact filter matters.
    """
    dx = L / omega.shape[-1]
    return strength * periodic_laplacian(omega, dx=dx, order=4)


def finite_difference_midpoint_residual(
        previous: torch.Tensor, candidate: torch.Tensor, *, L: float, dt: float,
        order: int, viscosity_strength: float | None = None) -> torch.Tensor:
    """Strong-form centered-FD vorticity residual on a periodic grid.

    ``viscosity_strength=None`` uses AZEBAN's nominal ``0.05/n`` closure.  It
    is intentionally a *traditional approximation* rather than the exact
    AZEBAN spectral filter, which is supplied by the pseudospectral method.
    """
    previous, candidate = _check_state(previous), _check_state(candidate)
    n, dx = previous.shape[-1], L / previous.shape[-1]
    eps = 0.05 / n if viscosity_strength is None else float(viscosity_strength)
    w0, w1 = vorticity_fd(previous, L=L, order=order), vorticity_fd(candidate, L=L, order=order)
    mid_state, wmid = 0.5 * (previous + candidate), 0.5 * (w0 + w1)
    u, v = mid_state[..., 0, :, :], mid_state[..., 1, :, :]
    adv = u * periodic_derivative(wmid, axis=0, dx=dx, order=order)
    adv = adv + v * periodic_derivative(wmid, axis=1, dx=dx, order=order)
    return (w1 - w0) / dt + adv - _spectral_viscosity_fd(wmid, L=L, strength=eps)


def finite_volume_midpoint_residual(
        previous: torch.Tensor, candidate: torch.Tensor, *, L: float, dt: float,
        viscosity_strength: float | None = None) -> torch.Tensor:
    """Conservative second-order finite-volume vorticity residual.

    The advective term is ``div(u omega)`` with centered periodic face fluxes.
    For incompressible flow it equals ``u . grad(omega)`` but has a distinct
    flux discretization, making it an independent numerical control.
    """
    previous, candidate = _check_state(previous), _check_state(candidate)
    n, dx = previous.shape[-1], L / previous.shape[-1]
    eps = 0.05 / n if viscosity_strength is None else float(viscosity_strength)
    w0, w1 = vorticity_fd(previous, L=L, order=2), vorticity_fd(candidate, L=L, order=2)
    mid, wmid = 0.5 * (previous + candidate), 0.5 * (w0 + w1)
    u, v = mid[..., 0, :, :], mid[..., 1, :, :]
    # Flux at i+1/2, then F_{i+1/2}-F_{i-1/2}; axes x/y are -2/-1.
    fx_plus = 0.25 * (u + torch.roll(u, -1, -2)) * (wmid + torch.roll(wmid, -1, -2))
    fy_plus = 0.25 * (v + torch.roll(v, -1, -1)) * (wmid + torch.roll(wmid, -1, -1))
    divergence = ((fx_plus - torch.roll(fx_plus, 1, -2)) +
                  (fy_plus - torch.roll(fy_plus, 1, -1))) / dx
    return (w1 - w0) / dt + divergence - _spectral_viscosity_fd(wmid, L=L, strength=eps)


def pseudospectral_midpoint_residual(
        spec: PDESpec2D, previous: torch.Tensor, candidate: torch.Tensor,
        *, dt: float | None = None, device=None) -> torch.Tensor:
    """Dealiased AZEBAN-filtered pseudospectral midpoint residual."""
    energy = FMPhysicsEnergy2D(spec, previous, 2, divergence_weight=0.0,
                               dt=dt if dt is not None else spec.dt_out, device=device)
    return energy.residual_fields(energy.to_vorticity(candidate)).mean(0)


def galerkin_residual(residual: torch.Tensor, modes: int) -> torch.Tensor:
    """Project a residual field to a low Fourier/Galerkin test space and lift it.

    The returned field is in fine-grid coordinates only for visualization.  A
    likelihood should operate on the retained spectral coefficients directly.
    """
    if residual.shape[-1] != residual.shape[-2]:
        raise ValueError("Galerkin projection requires square fields")
    n = residual.shape[-1]
    if not 0 < int(modes) <= n or n % int(modes):
        raise ValueError("modes must divide the grid resolution")
    if modes == n:
        return residual
    # Residual fields have shape (..., n, n).  Insert an explicit channel
    # dimension immediately before the two spatial axes: (..., 1, n, n).
    # ``[..., None, :, :]`` is not equivalent here; with a batched field it
    # inserts the channel between the spatial axes.
    low = spectral_resample_state(residual.unsqueeze(-3), modes)
    return spectral_resample_state(low, n).squeeze(-3)


def fixed_step_azeban_endpoint(spec: PDESpec2D, previous: torch.Tensor, *,
                               dt: float, substeps: int, device=None) -> torch.Tensor:
    """AZEBAN RHS with a prescribed number of SSP-RK3 steps.

    This is a genuinely classical speed/accuracy control.  Unlike the adaptive
    reference implementation, every transition has exactly ``substeps`` RHS
    evaluations; no data fitting or learned closure is involved.
    """
    energy = FMPhysicsEnergy2D(spec, previous, 2, divergence_weight=0.0,
                               dt=dt, device=device)
    return energy.native_flow_state(substeps=int(substeps))


def _smagorinsky_rhs(energy: FMPhysicsEnergy2D, w: torch.Tensor, cs: float) -> torch.Tensor:
    """AZEBAN RHS plus a standard local Smagorinsky subgrid closure."""
    wh = torch.fft.rfft2(w)
    u, v = energy.grid.velocity(wh)
    ux = torch.fft.irfft2(1j * energy.grid.kx * torch.fft.rfft2(u), s=(energy.n, energy.n))
    uy = torch.fft.irfft2(1j * energy.grid.ky * torch.fft.rfft2(u), s=(energy.n, energy.n))
    vx = torch.fft.irfft2(1j * energy.grid.kx * torch.fft.rfft2(v), s=(energy.n, energy.n))
    vy = torch.fft.irfft2(1j * energy.grid.ky * torch.fft.rfft2(v), s=(energy.n, energy.n))
    strain = torch.sqrt((2 * ux.square() + 2 * vy.square() + (uy + vx).square()).clamp(min=0))
    delta = energy.spec.L / energy.n
    nu_t = (float(cs) * delta) ** 2 * strain
    wx = torch.fft.irfft2(1j * energy.grid.kx * wh, s=(energy.n, energy.n))
    wy = torch.fft.irfft2(1j * energy.grid.ky * wh, s=(energy.n, energy.n))
    closure = torch.fft.irfft2(
        1j * energy.grid.kx * torch.fft.rfft2(nu_t * wx) +
        1j * energy.grid.ky * torch.fft.rfft2(nu_t * wy), s=(energy.n, energy.n))
    return energy._native_rhs(w) + closure


def smagorinsky_les_endpoint(spec: PDESpec2D, previous: torch.Tensor, *,
                              coarse_n: int, dt: float, cs: float = 0.17,
                              substeps: int = 16, device=None) -> torch.Tensor:
    """Closed coarse periodic LES endpoint, returned in its own coarse modes.

    The LES has a conventional Smagorinsky eddy-viscosity closure; unlike the
    old coarse AZEBAN control, it does not simply discard unresolved modes.
    """
    if not 0 < int(coarse_n) < spec.n or spec.n % int(coarse_n):
        raise ValueError("coarse_n must divide, and be below, the fine grid")
    coarse_spec = PDESpec2D(**{**spec.__dict__, "n": int(coarse_n)})
    state = spectral_resample_state(previous.reshape(-1, 2, spec.n, spec.n), coarse_n)
    flat = state.reshape(state.shape[0], -1)
    energy = FMPhysicsEnergy2D(coarse_spec, flat, 2, divergence_weight=0.0,
                               dt=dt, device=device)
    w = energy.previous_vorticity.clone()
    h = dt / int(substeps)
    for _ in range(int(substeps)):
        w1 = w + h * _smagorinsky_rhs(energy, w, cs)
        w2 = 0.75 * w + 0.25 * (w1 + h * _smagorinsky_rhs(energy, w1, cs))
        w = (w / 3) + (2 / 3) * (w2 + h * _smagorinsky_rhs(energy, w2, cs))
    uh = torch.fft.rfft2(w)
    u, v = energy.grid.velocity(uh)
    u = u + energy.previous[:, 0].mean(dim=(-2, -1), keepdim=True)
    v = v + energy.previous[:, 1].mean(dim=(-2, -1), keepdim=True)
    return torch.stack((u, v), dim=1)


def fit_pod_vorticity_basis(spec: PDESpec2D, states: torch.Tensor, rank: int,
                            device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Classical POD basis from calibration velocity snapshots only."""
    energy = FMPhysicsEnergy2D(spec, states, 2, divergence_weight=0.0,
                               dt=spec.dt_out, device=device)
    snapshots = energy.previous_vorticity.reshape(len(states), -1).double()
    mean = snapshots.mean(0)
    centered = snapshots - mean
    max_rank = min(int(rank), centered.shape[0], centered.shape[1])
    # The right singular vectors are the standard method-of-snapshots POD modes.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return mean, vh[:max_rank].T.contiguous()


def pod_galerkin_endpoint(spec: PDESpec2D, previous: torch.Tensor, *, dt: float,
                          mean: torch.Tensor, basis: torch.Tensor,
                          substeps: int = 16, device=None) -> torch.Tensor:
    """Physics-derived POD--Galerkin vorticity ROM endpoint.

    The online nonlinear RHS is still evaluated pseudospectrally before
    projection.  Therefore this is an accuracy/closure control, *not* a DEIM
    speed claim.  DEIM for a nonlocal FFT Biot--Savart operator requires a
    dedicated hyper-reduction construction and must not be faked by fitting
    future states.
    """
    energy = FMPhysicsEnergy2D(spec, previous, 2, divergence_weight=0.0,
                               dt=dt, device=device)
    u_basis = basis.to(energy.device, energy.dtype)
    # ``previous`` may be a single flattened state `(N,)`; the canonical batch
    # size is the leading dimension after FMPhysicsEnergy2D has reshaped it.
    batch = energy.previous.shape[0]
    w0 = energy.previous_vorticity.reshape(batch, -1)
    mean = mean.to(energy.device, energy.dtype)
    a = (w0 - mean) @ u_basis

    def rhs_reduced(coefficients):
        w = (mean + coefficients @ u_basis.T).reshape(-1, energy.n, energy.n)
        return energy._native_rhs(w).reshape(len(coefficients), -1) @ u_basis

    h = dt / int(substeps)
    for _ in range(int(substeps)):
        a1 = a + h * rhs_reduced(a)
        a2 = 0.75 * a + 0.25 * (a1 + h * rhs_reduced(a1))
        a = a / 3 + (2 / 3) * (a2 + h * rhs_reduced(a2))
    w = (mean + a @ u_basis.T).reshape(-1, energy.n, energy.n)
    uh = torch.fft.rfft2(w)
    u, v = energy.grid.velocity(uh)
    u = u + energy.previous[:, 0].mean(dim=(-2, -1), keepdim=True)
    v = v + energy.previous[:, 1].mean(dim=(-2, -1), keepdim=True)
    return torch.stack((u, v), dim=1)
