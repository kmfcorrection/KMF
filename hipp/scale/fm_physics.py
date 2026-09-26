"""Differentiable Navier--Stokes diagnostics in an FM's state coordinates."""
from __future__ import annotations

import math
import torch

from .data2d import PDESpec2D, SpectralGrid2D
from .physics2d import PhysicsEnergy2D


def spectral_resample_state(x: torch.Tensor, n_out: int) -> torch.Tensor:
    """Band-limited periodic resampling of ``(..., C, n, n)`` velocity fields.

    Cropping/padding centred complex Fourier coefficients, rather than using
    interpolation, preserves the resolved periodic modes exactly.  The scale
    factor compensates for PyTorch's inverse-FFT normalization.
    """
    z = x.reshape(*x.shape[:-2], x.shape[-2], x.shape[-1])
    n_in = int(z.shape[-1])
    if z.shape[-2] != n_in:
        raise ValueError("spectral resampling requires square fields")
    if n_out == n_in:
        return z.clone()
    h = torch.fft.fftshift(torch.fft.fft2(z), dim=(-2, -1))
    if n_out < n_in:
        start = (n_in - n_out) // 2
        h = h[..., start:start + n_out, start:start + n_out]
    else:
        pad = n_out - n_in
        lo, hi = pad // 2, pad - pad // 2
        h = torch.nn.functional.pad(h, (lo, hi, lo, hi))
    h = torch.fft.ifftshift(h, dim=(-2, -1))
    return torch.fft.ifft2(h, s=(n_out, n_out)).real * (n_out / n_in) ** 2


class FMPhysicsEnergy2D:
    """Vorticity-equation energy for [u,v] or [rho,u,v,p] FM states.

    The curl maps velocity to vorticity differentiably.  A divergence penalty
    prevents a correction that lowers the vorticity residual by leaving the
    incompressible state manifold.
    """

    def __init__(self, spec: PDESpec2D, previous: torch.Tensor,
                 channels: int, divergence_weight: float = 1.0,
                 vorticity_scale: float = 1.0,
                 divergence_scale: float = 1.0,
                 spectral_whitener: torch.Tensor | None = None,
                 spectral_residual_mean: torch.Tensor | None = None,
                 substeps: int = 1, dt: float | None = None,
                 dtype=torch.float64, device=None):
        if channels not in (1, 2, 4):
            raise ValueError("physics correction supports 1, 2, or 4 channels")
        self.spec, self.channels = spec, int(channels)
        self.n, self.N = spec.n, self.channels * spec.N
        self.dtype = dtype
        self.device = device or previous.device
        self.divergence_weight = float(divergence_weight)
        self.vorticity_scale = float(vorticity_scale)
        self.divergence_scale = float(divergence_scale)
        if self.vorticity_scale <= 0 or self.divergence_scale <= 0:
            raise ValueError("physics residual scales must be positive")
        self.substeps = int(substeps)
        self.dt = float(dt if dt is not None else spec.dt_out)
        self.poseidon_spectral_viscosity = spec.name == "poseidon_ns_native"
        self.grid = SpectralGrid2D(spec, device=self.device, dtype=dtype)
        if spectral_whitener is not None:
            expected = (spec.n, spec.n // 2 + 1)
            if tuple(spectral_whitener.shape) != expected:
                raise ValueError(f"spectral whitener must have shape {expected}")
            self.spectral_whitener = spectral_whitener.to(self.device, dtype)
        else:
            self.spectral_whitener = None
        if spectral_residual_mean is not None:
            expected = (spec.n, spec.n // 2 + 1)
            if tuple(spectral_residual_mean.shape) != expected:
                raise ValueError(f"spectral residual mean must have shape {expected}")
            self.spectral_residual_mean = spectral_residual_mean.to(self.device)
        else:
            self.spectral_residual_mean = None
        # Keeping a leading batch dimension lets independent physical flow maps
        # share GPU FFT kernels.  A batch of one is exactly the legacy path.
        prev = previous.detach().to(self.device, dtype).reshape(
            -1, channels, spec.n, spec.n)
        self.previous = prev
        self.previous_vorticity = self.to_vorticity(prev)
        if self.poseidon_spectral_viscosity and prev.shape[0] > 1:
            # The native branch supplies its own residual/flow operators.  The
            # generic midpoint helper accepts one conditioning field only.
            self.vorticity_energy = None
        elif prev.shape[0] == 1:
            self.vorticity_energy = PhysicsEnergy2D(
                spec, self.previous_vorticity[0],
                dt=dt if dt is not None else spec.dt_out,
                substeps=substeps, dtype=dtype, device=self.device)
        else:
            raise ValueError("batched conditioning states are currently supported only for native flow")

    def _native_rhs(self, w: torch.Tensor) -> torch.Tensor:
        """AZEBAN's published smooth spectral-viscosity RHS.

        AZEBAN uses s=1, epsilon_N=0.05/N, m_N=2*pi*sqrt(N)/L, and
        Q(k)=1-exp(-(abs(k)/m_N)^18).  This reproduces its *filter* exactly.
        The time residual below remains a midpoint diagnostic, rather than a
        bitwise reproduction of AZEBAN's full time integrator.
        """
        wh = torch.fft.rfft2(w)
        adv = self.grid.advection(wh)
        mode = self.grid.k2.sqrt()
        cutoff = 2 * math.pi * math.sqrt(self.n) / self.spec.L
        q = 1.0 - torch.exp(-(mode / cutoff).pow(18))
        eps = 0.05 / self.n
        rhs_h = adv - eps * self.grid.k2 * q * wh
        return torch.fft.irfft2(rhs_h, s=(self.n, self.n))

    def rhs_vorticity(self, w: torch.Tensor) -> torch.Tensor:
        """Evaluate the known vorticity PDE RHS without advancing time."""
        if self.poseidon_spectral_viscosity:
            return self._native_rhs(w)
        wh = torch.fft.rfft2(w)
        return torch.fft.irfft2(self.grid.rhs_hat(wh), s=(self.n, self.n))

    def native_flow_vorticity(self, w: torch.Tensor, *, substeps: int | None = None,
                               cfl: float = 0.5, max_steps: int = 20_000,
                               return_steps: bool = False,
                               integrator: str = "ssprk3") -> torch.Tensor | tuple[torch.Tensor, int]:
        """Differentiate through a discrete AZEBAN-style vorticity flow map.

        The released NS-Gauss generator uses AZEBAN's smooth spectral
        viscosity.  Its public integrator implementation contains SSP-RK3;
        the stages below reproduce that scheme exactly for a *fixed* step size.
        AZEBAN's public configurations use a CFL number of 0.5.  With
        ``substeps=None`` (the default), the code follows its state-dependent
        rule ``h = min(remaining, C / (N * max|u|))``.  A prescribed fixed
        ``substeps`` remains available solely for numerical-convergence tests.

        This differs materially from ``residual_fields``: it compares a
        candidate endpoint with ``Phi_dt(previous)`` rather than asking a
        straight line between the endpoints to satisfy a midpoint ODE.
        """
        if not self.poseidon_spectral_viscosity:
            raise ValueError("native discrete flow is only defined for poseidon_ns_native")
        if integrator != "ssprk3":
            raise ValueError(f"unsupported native integrator {integrator!r}")
        if substeps is not None and int(substeps) < 1:
            raise ValueError("substeps must be positive")
        if cfl <= 0:
            raise ValueError("cfl must be positive")
        z = w.to(self.device, self.dtype).reshape(-1, self.n, self.n)
        n_batch = z.shape[0]
        steps = torch.zeros(n_batch, dtype=torch.int64, device=z.device)
        remaining = torch.full((n_batch,), self.dt, dtype=z.dtype, device=z.device)
        while bool((remaining > 1e-15).any()):
            if substeps is None:
                uh = torch.fft.rfft2(z)
                u, v = self.grid.velocity(uh)
                # AZEBAN's IncompressibleEuler::dt(C) is C/(N*u_max).
                # `max_norm` is the maximum physical velocity magnitude.
                speed = torch.sqrt(u.square() + v.square()).amax(dim=(-2, -1))
                h = torch.minimum(remaining, cfl / (self.n * speed.clamp(min=1e-30)))
            else:
                h = self.dt / int(substeps)
                if int(steps.max()) >= int(substeps):
                    break
                h = torch.full_like(remaining, h)
            active = remaining > 1e-15
            h = torch.where(active, h, torch.zeros_like(h))
            h_field = h[:, None, None]
            # SSP-RK3 exactly as AZEBAN's `SSP_RK3::integrate` for prescribed h.
            z1 = z + h_field * self._native_rhs(z)
            z2 = 0.75 * z + 0.25 * (z1 + h_field * self._native_rhs(z1))
            candidate = (1.0 / 3.0) * z + (2.0 / 3.0) * (
                z2 + h_field * self._native_rhs(z2))
            z = torch.where(active[:, None, None], candidate, z)
            remaining -= h
            steps += active.to(steps.dtype)
            if int(steps.max()) > max_steps:
                raise RuntimeError(f"native CFL integrator exceeded max_steps={max_steps}")
        step_info = int(steps[0]) if n_batch == 1 else steps
        return (z, step_info) if return_steps else z

    def native_flow_state(self, *, substeps: int | None = None,
                          cfl: float = 0.5, return_steps: bool = False,
                          integrator: str = "ssprk3") -> torch.Tensor | tuple[torch.Tensor, int]:
        """Return ``Phi_dt(previous)`` in the original velocity-state space.

        Vorticity does not encode spatially constant velocity.  On the torus
        that mode is conserved by this unforced model, so it is restored from
        the conditioning state rather than silently discarded.
        """
        if self.channels != 2:
            raise ValueError("native discrete flow gate currently requires [u,v]")
        result = self.native_flow_vorticity(
            self.previous_vorticity, substeps=substeps, cfl=cfl,
            return_steps=return_steps, integrator=integrator)
        w_next, steps = result if return_steps else (result, None)
        uh = torch.fft.rfft2(w_next)
        u, v = self.grid.velocity(uh)
        u = u + self.previous[:, 0].mean(dim=(-2, -1), keepdim=True)
        v = v + self.previous[:, 1].mean(dim=(-2, -1), keepdim=True)
        state = torch.stack((u, v), dim=1).reshape(-1, self.N)
        return (state, steps) if return_steps else state

    def _native_residual_sq(self, w: torch.Tensor) -> torch.Tensor:
        return self.residual_fields(w).pow(2).mean(0)

    def residual_fields(self, w: torch.Tensor) -> torch.Tensor:
        """Per-substep residual fields, shape (substeps, batch, n, n)."""
        wt = self.previous_vorticity.expand_as(w)
        residuals = []
        for s in range(self.substeps):
            a = (s + 0.5) / self.substeps
            mid = (1 - a) * wt + a * w
            r = (w - wt) / self.dt - self._native_rhs(mid)
            residuals.append(r)
        return torch.stack(residuals)

    def _residual_energy_sq(self, w: torch.Tensor) -> torch.Tensor:
        """Mean squared residual after an optional calibration spectral metric."""
        if self.poseidon_spectral_viscosity:
            residuals = self.residual_fields(w)
        else:
            residuals = self.vorticity_energy.residual(w).unsqueeze(0)
        if self.spectral_whitener is not None:
            rh = torch.fft.rfft2(residuals)
            if self.spectral_residual_mean is not None:
                rh = rh - self.spectral_residual_mean
            residuals = torch.fft.irfft2(
                rh * self.spectral_whitener, s=(self.n, self.n))
        return residuals.pow(2).flatten(2).mean(2).mean(0)

    def _field(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device, self.dtype).reshape(-1, self.channels, self.n, self.n)

    def velocity(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._field(x)
        if self.channels == 1:
            wh = torch.fft.rfft2(z[:, 0])
            return self.grid.velocity(wh)
        if self.channels == 2:
            return z[:, 0], z[:, 1]
        return z[:, 1], z[:, 2]

    def to_vorticity(self, x: torch.Tensor) -> torch.Tensor:
        z = self._field(x)
        if self.channels == 1:
            return z[:, 0]
        u, v = self.velocity(z)
        uh, vh = torch.fft.rfft2(u), torch.fft.rfft2(v)
        wh = 1j * self.grid.kx * vh - 1j * self.grid.ky * uh
        return torch.fft.irfft2(wh, s=(self.n, self.n))

    def divergence(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels == 1:
            return self._field(x)[:, 0] * 0.0
        u, v = self.velocity(x)
        dh = 1j * self.grid.kx * torch.fft.rfft2(u)
        dh = dh + 1j * self.grid.ky * torch.fft.rfft2(v)
        return torch.fft.irfft2(dh, s=(self.n, self.n))

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        w = self.to_vorticity(x)
        residual_sq = self._residual_energy_sq(w)
        # Each component is standardized by a scale estimated on calibration
        # data.  This prevents a quantity with units of omega/time from being
        # added directly to one with units of velocity/length.
        e = 0.5 * residual_sq / (self.vorticity_scale ** 2)
        if self.channels > 1 and self.divergence_weight:
            div2 = self.divergence(x).flatten(1).pow(2).mean(1)
            e = e + (0.5 * self.divergence_weight * div2 /
                     (self.divergence_scale ** 2))
        return e

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        z = x.detach().to(self.device, self.dtype).reshape(-1, self.N).requires_grad_(True)
        (g,) = torch.autograd.grad(self.energy(z).sum(), z)
        return g

    def rms_residual(self, x: torch.Tensor) -> torch.Tensor:
        w = self.to_vorticity(x)
        if self.poseidon_spectral_viscosity:
            return self._native_residual_sq(w).flatten(1).mean(1).sqrt()
        return self.vorticity_energy.rms_residual(w)

    def rms_divergence(self, x: torch.Tensor) -> torch.Tensor:
        return self.divergence(x).flatten(1).pow(2).mean(1).sqrt()

    def project_incompressible(self, x: torch.Tensor) -> torch.Tensor:
        """Helmholtz-project velocity channels onto div(u)=0 exactly in Fourier space."""
        if self.channels == 1:
            return x.detach().to(self.device, self.dtype).reshape(-1, self.N)
        z = self._field(x).clone()
        iu, iv = (0, 1) if self.channels == 2 else (1, 2)
        uh, vh = torch.fft.rfft2(z[:, iu]), torch.fft.rfft2(z[:, iv])
        kdot = self.grid.kx * uh + self.grid.ky * vh
        uh = uh - self.grid.kx * self.grid.inv_k2 * kdot
        vh = vh - self.grid.ky * self.grid.inv_k2 * kdot
        # Nyquist coefficients cannot carry a real-valued derivative on an
        # even grid.  Removing them makes the rFFT representation exactly
        # compatible with the Helmholtz constraint after inverse FFT.
        uh[:, self.n // 2, :] = 0
        uh[:, :, self.n // 2] = 0
        vh[:, self.n // 2, :] = 0
        vh[:, :, self.n // 2] = 0
        z[:, iu] = torch.fft.irfft2(uh, s=(self.n, self.n))
        z[:, iv] = torch.fft.irfft2(vh, s=(self.n, self.n))
        return z.reshape(-1, self.N)

    def enstrophy(self, x: torch.Tensor) -> torch.Tensor:
        w = self.to_vorticity(x)
        return 0.5 * w.flatten(1).pow(2).mean(1)


class MidpointTransportEnergy2D(FMPhysicsEnergy2D):
    """One-JVP, residual-only proxy for a sparse endpoint defect.

    It uses the endpoint increment defect
    ``r(y)=y-x-dt F((x+y)/2)`` and calibration-only bias ``b``.  The returned
    observation is ``(I + dt/2 J_F(m))(r-b)``.  The JVP is of the known PDE
    RHS at the candidate midpoint: no AZEBAN/RK flow endpoint is evaluated.
    """

    def __init__(self, *args, transport_bias: torch.Tensor | None = None,
                 transport_jvp_terms: int = 1, **kwargs):
        if int(kwargs.get("substeps", 1)) != 1:
            raise ValueError("midpoint_transport_1jvp requires physics-substeps=1")
        if int(transport_jvp_terms) != 1:
            raise ValueError("midpoint_transport_1jvp is fixed to exactly one PDE-RHS JVP")
        super().__init__(*args, **kwargs)
        if transport_bias is None:
            bias = torch.zeros((self.n, self.n), device=self.device, dtype=self.dtype)
        else:
            bias = transport_bias.detach().to(self.device, self.dtype)
            if tuple(bias.shape) != (self.n, self.n):
                raise ValueError(f"transport bias must be {(self.n, self.n)}, got {tuple(bias.shape)}")
        self.transport_bias = bias

    def midpoint_endpoint_defect(self, w: torch.Tensor) -> torch.Tensor:
        """Uncentered midpoint defect in endpoint-increment units."""
        previous = self.previous_vorticity.expand_as(w)
        midpoint = 0.5 * (previous + w)
        return w - previous - self.dt * self.rhs_vorticity(midpoint)

    def residual_fields(self, w: torch.Tensor) -> torch.Tensor:
        previous = self.previous_vorticity.expand_as(w)
        midpoint = 0.5 * (previous + w)
        centered = self.midpoint_endpoint_defect(w) - self.transport_bias
        # One forward-mode JVP of the known PDE RHS. This remains
        # differentiable through candidate w for the MAP objective.
        _, transported_term = torch.func.jvp(
            self.rhs_vorticity, (midpoint,), (centered,))
        transported = centered + 0.5 * self.dt * transported_term
        return transported.unsqueeze(0)

    def rms_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Unwhitened transported-defect RMS for calibration gates/reporting."""
        w = self.to_vorticity(x)
        return self.residual_fields(w).flatten(2).pow(2).mean(2).mean(0).sqrt()


class NativeFlowEnergy2D(FMPhysicsEnergy2D):
    """Endpoint likelihood from a CFL-stable AZEBAN discrete flow map.

    For a fixed conditioning state ``x_t``, the physical prediction
    ``Phi_dt(x_t)`` is computed once and the likelihood is simply a quadratic
    mismatch in the candidate next state.  This is both the appropriate
    discrete-time likelihood for the released data and much cheaper inside a
    MAP solver than differentiating through the roughly 100--200 internal CFL
    stages on every optimizer evaluation.

    The flow itself is *not* differentiated with respect to ``previous`` in
    this class.  That is intentional: its adaptive time-step decisions are
    non-smooth and the posterior conditions on a fixed preceding state.
    """

    def __init__(self, spec: PDESpec2D, previous: torch.Tensor, channels: int,
                 divergence_weight: float = 1.0, vorticity_scale: float = 1.0,
                 divergence_scale: float = 1.0, flow_cfl: float = 0.5,
                 dt: float | None = None, dtype=torch.float64, device=None):
        super().__init__(spec, previous, channels,
                         divergence_weight=divergence_weight,
                         vorticity_scale=vorticity_scale,
                         divergence_scale=divergence_scale,
                         dt=dt, dtype=dtype, device=device)
        if channels != 2:
            raise ValueError("NativeFlowEnergy2D currently requires [u,v] states")
        with torch.no_grad():
            target, steps = self.native_flow_state(cfl=flow_cfl, return_steps=True)
        self.flow_target = target.detach()
        self.flow_cfl = float(flow_cfl)
        self.flow_internal_steps = (int(steps) if isinstance(steps, int)
                                    else steps.detach())

    def endpoint_residual(self, x: torch.Tensor) -> torch.Tensor:
        z = x.to(self.device, self.dtype).reshape(-1, self.N)
        return z - self.flow_target

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        residual_sq = self.endpoint_residual(x).square().mean(1)
        e = 0.5 * residual_sq / (self.vorticity_scale ** 2)
        if self.divergence_weight:
            div2 = self.divergence(x).flatten(1).square().mean(1)
            e = e + (0.5 * self.divergence_weight * div2 /
                     (self.divergence_scale ** 2))
        return e

    def rms_residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.endpoint_residual(x).square().mean(1).sqrt()

    def member(self, index: int) -> "NativeFlowEnergy2D":
        """A zero-copy single-trajectory view of a batched flow likelihood.

        The flow target has already been integrated.  Sharing the grid and
        target storage is safe because every diagnostic and MAP solve treats
        the energy as immutable.
        """
        if not 0 <= int(index) < self.previous.shape[0]:
            raise IndexError(index)
        view = object.__new__(NativeFlowEnergy2D)
        view.__dict__ = self.__dict__.copy()
        view.previous = self.previous[index:index + 1]
        view.previous_vorticity = self.previous_vorticity[index:index + 1]
        view.flow_target = self.flow_target[index:index + 1]
        if isinstance(self.flow_internal_steps, torch.Tensor):
            view.flow_internal_steps = int(self.flow_internal_steps[index])
        return view


class FixedRK3FlowEnergy2D(NativeFlowEnergy2D):
    """Endpoint likelihood from a fixed-budget full-grid SSP-RK3 flow.

    This differs from :class:`NativeFlowEnergy2D`, which uses AZEBAN's
    adaptive CFL controller.  Here exactly ``flow_substeps`` SSP-RK3 stages
    advance the known PDE.  It is an approximate numerical PDE solve, used
    only to form the inference-time defect
    ``x_next - Psi_RK3(x_previous)``; the future truth is never read while
    constructing the endpoint.
    """

    def __init__(self, spec: PDESpec2D, previous: torch.Tensor, channels: int,
                 divergence_weight: float = 1.0, vorticity_scale: float = 1.0,
                 divergence_scale: float = 1.0, flow_substeps: int = 64,
                 dt: float | None = None, dtype=torch.float64, device=None):
        # Avoid NativeFlowEnergy2D.__init__, which would first run adaptive flow.
        FMPhysicsEnergy2D.__init__(
            self, spec, previous, channels,
            divergence_weight=divergence_weight,
            vorticity_scale=vorticity_scale,
            divergence_scale=divergence_scale,
            dt=dt, dtype=dtype, device=device)
        if channels != 2:
            raise ValueError("FixedRK3FlowEnergy2D currently requires [u,v] states")
        if int(flow_substeps) <= 0:
            raise ValueError("flow_substeps must be positive")
        with torch.no_grad():
            target, steps = self.native_flow_state(
                substeps=int(flow_substeps), return_steps=True)
        self.flow_target = target.detach()
        self.flow_substeps = int(flow_substeps)
        self.flow_internal_steps = (int(steps) if isinstance(steps, int)
                                    else steps.detach())

    def member(self, index: int) -> "FixedRK3FlowEnergy2D":
        if not 0 <= int(index) < self.previous.shape[0]:
            raise IndexError(index)
        view = object.__new__(FixedRK3FlowEnergy2D)
        view.__dict__ = self.__dict__.copy()
        view.previous = self.previous[index:index + 1]
        view.previous_vorticity = self.previous_vorticity[index:index + 1]
        view.flow_target = self.flow_target[index:index + 1]
        if isinstance(self.flow_internal_steps, torch.Tensor):
            view.flow_internal_steps = int(self.flow_internal_steps[index])
        return view


class CoarseNativeFlowEnergy2D(FMPhysicsEnergy2D):
    """A fixed, lower-resolution AZEBAN endpoint likelihood on a fine state.

    The candidate and divergence penalty remain on the target grid, but the
    endpoint is formed by restrict -> CFL-stable AZEBAN flow -> prolong.  This
    is a pre-specified imperfect-physics likelihood, not an oracle flow.
    """
    def __init__(self, spec: PDESpec2D, previous: torch.Tensor, channels: int,
                 physics_n: int, divergence_weight: float = 1.0,
                 vorticity_scale: float = 1.0, divergence_scale: float = 1.0,
                 flow_cfl: float = 0.5, dt: float | None = None,
                 dtype=torch.float64, device=None):
        if not 0 < int(physics_n) < int(spec.n):
            raise ValueError("physics_n must be strictly below target-grid resolution")
        super().__init__(spec, previous, channels, divergence_weight,
                         vorticity_scale, divergence_scale, dt=dt,
                         dtype=dtype, device=device)
        if channels != 2:
            raise ValueError("coarse native flow requires [u,v] states")
        coarse_spec = PDESpec2D(**{**spec.__dict__, "n": int(physics_n)})
        prev_coarse = spectral_resample_state(self.previous, int(physics_n)).reshape(-1, 2 * int(physics_n) ** 2)
        with torch.no_grad():
            coarse = NativeFlowEnergy2D(coarse_spec, prev_coarse, 2,
                                         divergence_weight=divergence_weight,
                                         vorticity_scale=1.0, divergence_scale=1.0,
                                         flow_cfl=flow_cfl, dt=self.dt,
                                         dtype=dtype, device=self.device)
            low_target = coarse.flow_target.reshape(-1, 2, int(physics_n), int(physics_n))
            # Keep the solver's actual resolved endpoint.  The lifted target is
            # useful for the historical full-field endpoint control, whereas a
            # calibrated coarse-physics likelihood must never treat its
            # unresolved fine modes as observations.
            self.low_flow_target = low_target.reshape(-1, 2 * int(physics_n) ** 2).detach()
            lifted = spectral_resample_state(low_target, spec.n).reshape(-1, self.N)
            # Explicitly restore the fine-grid incompressibility constraint;
            # rFFT Nyquist conventions otherwise leave a tiny grid-dependent
            # divergence after restriction/prolongation.
            self.flow_target = self.project_incompressible(lifted).detach()
            self.flow_internal_steps = coarse.flow_internal_steps
        self.physics_n, self.flow_cfl = int(physics_n), float(flow_cfl)

    def endpoint_residual(self, x):
        return x.to(self.device, self.dtype).reshape(-1, self.N) - self.flow_target

    def energy(self, x):
        e = 0.5 * self.endpoint_residual(x).square().mean(1) / self.vorticity_scale ** 2
        if self.divergence_weight:
            e = e + 0.5 * self.divergence_weight * self.divergence(x).flatten(1).square().mean(1) / self.divergence_scale ** 2
        return e

    def rms_residual(self, x):
        return self.endpoint_residual(x).square().mean(1).sqrt()

    def member(self, index: int):
        view = object.__new__(CoarseNativeFlowEnergy2D)
        view.__dict__ = self.__dict__.copy()
        view.previous = self.previous[index:index + 1]
        view.previous_vorticity = self.previous_vorticity[index:index + 1]
        view.flow_target = self.flow_target[index:index + 1]
        view.low_flow_target = self.low_flow_target[index:index + 1]
        if isinstance(self.flow_internal_steps, torch.Tensor):
            view.flow_internal_steps = int(self.flow_internal_steps[index])
        return view


class ResolvedCoarseFlowEnergy2D(CoarseNativeFlowEnergy2D):
    """Calibration-aware likelihood for modes resolved by a coarse flow only.

    For a coarse solver grid ``h``, let ``H`` be spectral restriction from the
    fine state to its ``2*h*h`` velocity values.  This energy is

        1/2 (H x - (Phi_h(H x_t) + b))^T R^-1
            (H x - (Phi_h(H x_t) + b)),

    where ``b`` and ``R`` are fitted on calibration transitions.  Crucially it
    contains no penalty on ``(I-H^T H)x``: fine modes the coarse solver cannot
    represent remain governed by the FM prior.
    """
    _operator_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def __init__(self, *args, discrepancy_bias: torch.Tensor | None = None,
                 discrepancy_covariance: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.low_flow_target.shape[1]
        if discrepancy_bias is None:
            discrepancy_bias = torch.zeros(m, dtype=self.dtype, device=self.device)
        if discrepancy_covariance is None:
            discrepancy_covariance = torch.eye(m, dtype=self.dtype, device=self.device)
        self.discrepancy_bias = discrepancy_bias.to(self.device, self.dtype).reshape(1, m)
        self.discrepancy_covariance = discrepancy_covariance.to(self.device, self.dtype)
        if self.discrepancy_covariance.shape != (m, m):
            raise ValueError(f"coarse discrepancy covariance must have shape {(m, m)}")
        self.discrepancy_covariance = 0.5 * (
            self.discrepancy_covariance + self.discrepancy_covariance.T)
        self.R_chol = torch.linalg.cholesky(self.discrepancy_covariance)
        self.observation_target = self.low_flow_target + self.discrepancy_bias

    @property
    def observation_dim(self) -> int:
        return int(self.low_flow_target.shape[1])

    def restrict(self, x: torch.Tensor) -> torch.Tensor:
        """The linear resolved-mode observation operator H."""
        z = x.to(self.device, self.dtype).reshape(-1, self.channels, self.n, self.n)
        return spectral_resample_state(z, self.physics_n).reshape(z.shape[0], -1)

    def _operator_key(self):
        return (self.n, self.physics_n, self.channels, str(self.device), self.dtype)

    def restriction_adjoint_and_gram(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rows of H^T and H H^T, exactly for this FFT implementation.

        The FFT crop/pad convention has even-grid Nyquist details, so deriving
        this as an assumed scalar multiple of prolongation is unsafe.  We build
        the small operator once by autodifferentiating the actual restriction.
        """
        key = self._operator_key()
        cached = self._operator_cache.get(key)
        if cached is not None:
            return cached
        m = self.observation_dim
        basis = torch.eye(m, dtype=self.dtype, device=self.device)
        fine = torch.zeros(m, self.N, dtype=self.dtype, device=self.device,
                           requires_grad=True)
        restricted = self.restrict(fine)
        (ht_rows,) = torch.autograd.grad(restricted, fine, grad_outputs=basis)
        ht_rows = ht_rows.detach()
        hht = self.restrict(ht_rows).T.detach()
        hht = 0.5 * (hht + hht.T)
        self._operator_cache[key] = (ht_rows, hht)
        return ht_rows, hht

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.restrict(x) - self.observation_target

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        r = self.residual(x)
        q = torch.cholesky_solve(r.T, self.R_chol).T
        return 0.5 * (r * q).sum(dim=1)

    def rms_residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x).square().mean(dim=1).sqrt()

    def member(self, index: int):
        view = super().member(index)
        view.__class__ = ResolvedCoarseFlowEnergy2D
        view.discrepancy_bias = self.discrepancy_bias
        view.discrepancy_covariance = self.discrepancy_covariance
        view.R_chol = self.R_chol
        view.observation_target = self.observation_target[index:index + 1]
        return view
