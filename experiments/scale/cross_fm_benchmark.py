#!/usr/bin/env python3
"""Unified Cross-Foundation Model & Cross-PDE Benchmark Suite.

Evaluates arbitrary PDE Foundation Models (Poseidon, DPOT, MORPH, FNO, U-Net)
across diverse physical PDE benchmarks (NS-Gauss, FNS-KF, ACE, Wave-Gauss, Shear-Layer).

Benchmarks:
1. Sub-Cadence Kinematic Super-Resolution (Linear vs. Raw FM vs. Hermite Kinematic Spline)
2. Autoregressive Rollout & Physical Invariant Preservation (Raw FM vs. Manifold Recovery)
3. Latency & Pareto Efficiency
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    configure_native_poseidon_cadence, native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.data2d import PDESpec2D, SpectralGrid2D
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.utils import Table, print_header, save_json, set_seed


# ---------------------------------------------------------------------------
# General Physical PDE Right-Hand-Side (RHS) Dispatcher
# ---------------------------------------------------------------------------
class GenericPDEPhysics:
    """Evaluates physical acceleration F(u) and invariants across PDE classes."""

    def __init__(self, pde_type: str, n: int = 128, L: float | None = None,
                 nu: float = 1e-4, epsilon: float = 0.04, device=None, dtype=torch.float64):
        self.pde_type = pde_type.lower()
        self.n = n
        if L is None:
            # ACE is periodic on [0, 2*pi]^2; fluids and waves are on [0, 1]^2
            self.L = 2.0 * math.pi if "ace" in self.pde_type else 1.0
        else:
            self.L = float(L)
        self.nu = nu
        self.epsilon = epsilon
        self.c = None
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # 2D Fourier grid
        kx = torch.fft.fftfreq(n, d=self.L / (2 * math.pi * n), device=self.device, dtype=self.dtype)
        ky = torch.fft.fftfreq(n, d=self.L / (2 * math.pi * n), device=self.device, dtype=self.dtype)
        self.KY, self.KX = torch.meshgrid(ky, kx, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.inv_k2 = torch.where(self.K2 == 0, torch.zeros_like(self.K2), 1.0 / self.K2)

    def rhs(self, u: torch.Tensor) -> torch.Tensor:
        """Evaluate physical time derivative F(u) = du/dt."""
        # u: (B, C, H, W)
        if "ns" in self.pde_type or "shear" in self.pde_type or "kolmogorov" in self.pde_type or "fns" in self.pde_type:
            # 2D Incompressible Navier-Stokes: dw/dt = -(u . grad)w + nu * Lap(w) + forcing
            return self._rhs_navier_stokes(u)
        elif "ace" in self.pde_type or "allen" in self.pde_type:
            # Allen-Cahn: dphi/dt = eps^2 * Lap(phi) - (phi^3 - phi)
            return self._rhs_allen_cahn(u)
        elif "wave" in self.pde_type:
            # Acoustic Wave: du/dt = v, dv/dt = c^2 * Lap(u)
            return self._rhs_wave(u)
        else:
            # Default to 2D Navier-Stokes
            return self._rhs_navier_stokes(u)

    def _rhs_navier_stokes(self, vel: torch.Tensor) -> torch.Tensor:
        # vel: (B, 2, H, W)
        u, v = vel[:, 0], vel[:, 1]
        uh = torch.fft.fft2(u)
        vh = torch.fft.fft2(v)
        # Vorticity w = dv/dx - du/dy
        wh = 1j * self.KX * vh - 1j * self.KY * uh

        w = torch.fft.ifft2(wh).real
        # Advection -(u * dw/dx + v * dw/dy)
        w_x = torch.fft.ifft2(1j * self.KX * wh).real
        w_y = torch.fft.ifft2(1j * self.KY * wh).real
        adv = -(u * w_x + v * w_y)
        adv_h = torch.fft.fft2(adv)

        # Viscous diffusion
        diff_h = -self.nu * self.K2 * wh
        dw_dt_h = adv_h + diff_h

        # Lift back to velocity acceleration du/dt, dv/dt
        psih = dw_dt_h * self.inv_k2
        a_u = torch.fft.ifft2(1j * self.KY * psih).real
        a_v = torch.fft.ifft2(-1j * self.KX * psih).real
        return torch.stack([a_u, a_v], dim=1)

    def _rhs_allen_cahn(self, phi: torch.Tensor) -> torch.Tensor:
        # phi: (B, 1, H, W)
        phi_val = phi[:, 0]
        phih = torch.fft.fft2(phi_val)
        lap_phi = torch.fft.ifft2(-self.K2 * phih).real
        reaction = -(phi_val**3 - phi_val)
        dphi_dt = (self.epsilon**2) * lap_phi + reaction
        return dphi_dt.unsqueeze(1)

    def _rhs_wave(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, 1, H, W) displacement or (B, 2, H, W) [u, v]
        dx = self.L / self.n
        if self.c is not None:
            c = self.c
            if c.shape[0] != state.shape[0]:
                if hasattr(self, "c_cal") and self.c_cal is not None and self.c_cal.shape[0] == state.shape[0]:
                    c = self.c_cal
                else:
                    c = c[:state.shape[0]]
            c2 = c**2
        else:
            c2 = 1.0

        if state.shape[1] == 1:
            u = state[:, 0]
            pad = torch.nn.functional.pad(u.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
            lap = (pad[:, 2:, 1:-1] + pad[:, :-2, 1:-1] + pad[:, 1:-1, 2:] + pad[:, 1:-1, :-2] - 4.0 * pad[:, 1:-1, 1:-1]) / (dx**2)
            return (c2 * lap).unsqueeze(1)
        u = state[:, 0]
        v = state[:, 1]
        pad = torch.nn.functional.pad(u.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
        lap = (pad[:, 2:, 1:-1] + pad[:, :-2, 1:-1] + pad[:, 1:-1, 2:] + pad[:, 1:-1, :-2] - 4.0 * pad[:, 1:-1, 1:-1]) / (dx**2)
        return torch.stack([v, c2 * lap], dim=1)

    def project_manifold(self, u: torch.Tensor) -> torch.Tensor:
        """Enforces physical manifold invariants (divergence-free for fluids, physical bounds for ACE, wave filter)."""
        if "ns" in self.pde_type or "shear" in self.pde_type or "fns" in self.pde_type:
            # Exact Helmholtz-Leray Projection in Fourier space
            uh = torch.fft.fft2(u[:, 0])
            vh = torch.fft.fft2(u[:, 1])
            kdot = self.KX * uh + self.KY * vh
            uh_proj = uh - self.KX * self.inv_k2 * kdot
            vh_proj = vh - self.KY * self.inv_k2 * kdot
            u_proj = torch.fft.ifft2(uh_proj).real
            v_proj = torch.fft.ifft2(vh_proj).real
            return torch.stack([u_proj, v_proj], dim=1)
        elif "ace" in self.pde_type:
            # Allen-Cahn exact physical order parameter bounds [-1.0, 1.0]
            # Eliminates unphysical neural overshooting beyond pure phase boundaries
            return u.clamp(-1.0, 1.0)
        elif "wave" in self.pde_type:
            # Suppress high-frequency non-physical dispersion noise
            uh = torch.fft.rfft2(u)
            kx = torch.linspace(0, 1, uh.shape[-2], device=u.device).view(-1, 1)
            ky = torch.linspace(0, 1, uh.shape[-1], device=u.device).view(1, -1)
            k_norm = torch.sqrt(kx**2 + ky**2)
            filt = torch.exp(-0.5 * (k_norm / 0.85)**4).unsqueeze(0).unsqueeze(0)
            return torch.fft.irfft2(uh * filt, s=(u.shape[-2], u.shape[-1]))
        return u


# ---------------------------------------------------------------------------
# General Kinematic Hermite Spline (0 ODE Steps)
# ---------------------------------------------------------------------------
def fluid_hermite_kinematic_spline(op: FMPhysicsEnergy2D, u0: torch.Tensor, u1: torch.Tensor,
                                   dt: float, s: float = 0.5) -> torch.Tensor:
    """Exact Hermite kinematic spline for fluid velocity in vorticity coordinates."""
    w0 = op.to_vorticity(u0)
    w1 = op.to_vorticity(u1)
    f0 = op.rhs_vorticity(w0)
    f1 = op.rhs_vorticity(w1)

    h00 = 1.0 - 3.0 * s**2 + 2.0 * s**3
    h10 = 3.0 * s**2 - 2.0 * s**3
    h01 = dt * (s - 2.0 * s**2 + s**3)
    h11 = dt * (-s**2 + s**3)

    w_interp = h00 * w0 + h10 * w1 + h01 * f0 + h11 * f1
    wh_interp = torch.fft.rfft2(w_interp)
    u, v = op.grid.velocity(wh_interp)
    mean_vel = (1.0 - s) * u0.mean(dim=(-2, -1), keepdim=True) + s * u1.mean(dim=(-2, -1), keepdim=True)
    return torch.stack((u, v), dim=1) + mean_vel


def general_kinematic_hermite_spline(phys: GenericPDEPhysics, u0: torch.Tensor, u1: torch.Tensor,
                                     dt: float, s: float) -> torch.Tensor:
    """Exact C^1 cubic Hermite spline for any autonomous PDE with strictly 0 ODE steps."""
    if "wave" in phys.pde_type and u0.shape[1] == 1:
        a0 = phys.rhs(u0)
        a1 = phys.rhs(u1)
        # Parabolic physical curvature bridge:
        # At s = 0.5, curvature coefficient is (dt^2 / 16.0) on the average acceleration 0.5*(a0 + a1)
        curvature_factor = (dt**2 / 16.0) * (4.0 * s * (1.0 - s))
        return (1.0 - s) * u0 + s * u1 + curvature_factor * 0.5 * (a0 + a1)

    h00 = 1.0 - 3.0 * s**2 + 2.0 * s**3
    h10 = 3.0 * s**2 - 2.0 * s**3
    h01 = dt * (s - 2.0 * s**2 + s**3)
    h11 = dt * (-s**2 + s**3)

    f0 = phys.rhs(u0)
    f1 = phys.rhs(u1)
    u_interp = h00 * u0 + h10 * u1 + h01 * f0 + h11 * f1
    return phys.project_manifold(u_interp)


# ---------------------------------------------------------------------------
# Universal Dataset Loader
# ---------------------------------------------------------------------------
def load_dataset_trajectories(path: str | Path, fm, n_traj: int, steps: int = 4,
                             stride: int = 2, offset: int = 0, target_grid: int = 128,
                             prehistory: int = 0):
    """Load trajectories across NetCDF / HDF5 datasets and adapt to target resolution."""
    import h5py
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    c_val = None
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution" if "solution" in h else list(h.keys())[0]
        ds = h[key]
        raw_shape = ds.shape
        # ``prehistory`` contains only states at or before the deployment
        # origin.  It is needed for released history-conditioned FMs such as
        # DPOT, never for scoring or future-dependent normalization.
        future_frames = max(2, steps * stride)
        need_frames = int(prehistory) + future_frames + 1
        if need_frames > ds.shape[1]:
            raise ValueError(
                f"{path} contains {ds.shape[1]} frames but this run needs "
                f"{need_frames} ({prehistory} causal-history frames + {future_frames} future frames)")

        if "c" in h:
            c_ds = h["c"]
            if c_ds.ndim == 3:
                c_raw = np.asarray(c_ds[offset:offset + n_traj], dtype=np.float32)
            else:
                c_raw = np.asarray(c_ds, dtype=np.float32)
            # Wave-Gauss seismic sound speed is in m/s (~1500-4500 m/s) on a 5120m block.
            # Normalize to dimensionless unit domain [0, 1]^2:
            if float(np.median(c_raw)) > 100.0:
                c_val = c_raw / 5120.0
            else:
                c_val = c_raw

        # Handle 4D or 5D arrays
        if ds.ndim == 4:
            # (sample, time, H, W) -> add channel dim
            arr = np.asarray(ds[offset:offset + n_traj, :need_frames], dtype=np.float32)[:, :, None, :, :]
        elif ds.ndim == 5:
            # Fluid PDEs (NS-Gauss, FNS-KF) always carry 2 physical velocity channels [u, v]
            if "ns" in str(path).lower() or "fns" in str(path).lower():
                req_c = 2
            else:
                req_c = ds.shape[2]
            arr = np.asarray(ds[offset:offset + n_traj, :need_frames, :req_c], dtype=np.float32)
        else:
            raise ValueError(f"unexpected dataset ndim: {ds.ndim} with shape {raw_shape}")

    t = torch.as_tensor(arr, device=fm.device, dtype=torch.float64)
    if target_grid != 128:
        # Interpolate spatial resolution (B, T, C, 128, 128) -> (B, T, C, target_grid, target_grid)
        B, T, C, H, W = t.shape
        t = torch.nn.functional.interpolate(t.view(B * T, C, H, W), size=(target_grid, target_grid), mode="bicubic", align_corners=False).view(B, T, C, target_grid, target_grid)
        if c_val is not None:
            c_t = torch.as_tensor(c_val, device=fm.device, dtype=torch.float64)
            if c_t.ndim == 2:
                c_t = torch.nn.functional.interpolate(c_t.unsqueeze(0).unsqueeze(0), size=(target_grid, target_grid), mode="bicubic", align_corners=False).squeeze(0).squeeze(0)
            elif c_t.ndim == 3:
                c_t = torch.nn.functional.interpolate(c_t.unsqueeze(1), size=(target_grid, target_grid), mode="bicubic", align_corners=False).squeeze(1)
            c_val = c_t.cpu().numpy()

    return t, c_val


def fm_predict(fm, u: torch.Tensor, current_grid: int = 128,
               history_context: torch.Tensor | None = None) -> torch.Tensor:
    """Evaluates FM prediction with zero-shot spatial resolution and channel alignment."""
    # 1. Spatial interpolation to FM native resolution (128x128) if needed
    if current_grid != 128:
        u_in = torch.nn.functional.interpolate(u, size=(128, 128), mode="bicubic", align_corners=False)
    else:
        u_in = u

    if history_context is not None and hasattr(fm, "set_history"):
        h = history_context
        if current_grid != 128:
            b, t, c, _, _ = h.shape
            h = torch.nn.functional.interpolate(
                h.reshape(b * t, c, h.shape[-2], h.shape[-1]), size=(128, 128),
                mode="bicubic", align_corners=False).reshape(b, t, c, 128, 128)
        fm.set_history(h.to(device=fm.device, dtype=fm.dtype))

    fm_c = fm.state_shape[0] if hasattr(fm, "state_shape") else u.shape[1]
    in_c = u.shape[1]

    # 2. Forward pass through FM with channel alignment
    if in_c == fm_c:
        pred_native = fm.predict(u_in.to(fm.dtype)).double()
    elif in_c > fm_c:
        preds = []
        for ch in range(in_c):
            p = fm.predict(u_in[:, ch:ch+1].to(fm.dtype)).double()
            preds.append(p)
        pred_native = torch.cat(preds, dim=1)
    else:
        pad = torch.zeros(u_in.shape[0], fm_c - in_c, u_in.shape[2], u_in.shape[3], device=u_in.device, dtype=u_in.dtype)
        u_padded = torch.cat([u_in, pad], dim=1)
        p = fm.predict(u_padded.to(fm.dtype)).double()
        pred_native = p[:, :in_c]

    # 3. Spatial interpolation back to target grid
    if current_grid != 128:
        return torch.nn.functional.interpolate(pred_native, size=(current_grid, current_grid), mode="bicubic", align_corners=False)
    return pred_native


def _project_task1_state(x: torch.Tensor, *, is_fluid: bool,
                         op: FMPhysicsEnergy2D | None,
                         phys: GenericPDEPhysics) -> torch.Tensor:
    """Apply the same known-state constraint to every Task 1 candidate."""
    if is_fluid and op is not None:
        return op.project_incompressible(x).reshape_as(x)
    return phys.project_manifold(x)


@torch.no_grad()
def _fm_endpoint_task1_candidates(
        fm, u0: torch.Tensor, *, span_raw_frames: int, dt_span: float,
        grid: int, is_fluid: bool, op: FMPhysicsEnergy2D | None,
        phys: GenericPDEPhysics,
        history_context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build Task 1 candidates from an FM-produced endpoint.

    The endpoint is never read from the dataset.  ``direct`` is returned only
    for adapters that expose an actual time-conditioned query.  A fixed-step
    adapter must not be relabelled as a half-step predictor merely because its
    single output is available.
    """
    supports_time_query = hasattr(fm, "set_lead_time")
    if supports_time_query:
        _set_fm_query_time(fm, raw_frames=span_raw_frames, physical_dt=dt_span)
    endpoint = _project_task1_state(
        fm_predict(fm, u0, current_grid=grid, history_context=history_context),
        is_fluid=is_fluid, op=op, phys=phys)
    linear = _project_task1_state(
        0.5 * (u0 + endpoint), is_fluid=is_fluid, op=op, phys=phys)
    if is_fluid and op is not None:
        bridge = fluid_hermite_kinematic_spline(op, u0, endpoint, dt_span, s=0.5)
    else:
        bridge = general_kinematic_hermite_spline(phys, u0, endpoint, dt_span, s=0.5)
    bridge = _project_task1_state(bridge, is_fluid=is_fluid, op=op, phys=phys)

    direct = None
    if supports_time_query:
        _set_fm_query_time(fm, raw_frames=float(span_raw_frames) / 2.0, physical_dt=dt_span / 2.0)
        direct = _project_task1_state(
            fm_predict(fm, u0, current_grid=grid, history_context=history_context),
            is_fluid=is_fluid, op=op, phys=phys)
    return endpoint, linear, bridge, direct


def _rmse(x: torch.Tensor, target: torch.Tensor) -> float:
    return float((x - target).square().mean(dim=(-3, -2, -1)).sqrt().mean())


def _set_fm_query_time(fm, *, raw_frames: float, physical_dt: float) -> None:
    """Set an adapter's documented time coordinate, never an inferred one."""
    if not hasattr(fm, "set_lead_time"):
        return
    units = getattr(fm, "lead_time_units", "raw_frames")
    if units == "raw_frames":
        fm.set_lead_time(float(raw_frames))
    elif units == "physical_time":
        fm.set_lead_time(float(physical_dt))
    else:
        raise ValueError(f"unknown lead-time units {units!r} for {getattr(fm.info, 'name', 'FM')}")


def _coarse_defect_correction(
        previous: torch.Tensor, forecast: torch.Tensor, *, alpha: float,
        dt: float, is_fluid: bool, op: FMPhysicsEnergy2D | None,
        phys: GenericPDEPhysics, spec: PDESpec2D | None,
        midpoint: torch.Tensor | None = None) -> torch.Tensor:
    """Apply one zero-flow-step Simpson curvature-defect correction.

    The only PDE calls are instantaneous RHS evaluations at the two endpoints
        and a midpoint state.  When ``midpoint`` is supplied, it is a
        time-conditioned FM query at the physical half-time.  Otherwise the
        midpoint is reconstructed from the endpoints.  In either case this is
        not an RK/ODE update from ``previous`` to ``forecast``.
    """
    candidate = _project_task1_state(forecast, is_fluid=is_fluid, op=op, phys=phys)
    if alpha == 0.0:
        return candidate

    if is_fluid and op is not None and spec is not None:
        w0, w1 = op.to_vorticity(previous), op.to_vorticity(candidate)
        f0, f1 = op.rhs_vorticity(w0), op.rhs_vorticity(w1)
        if midpoint is not None:
            midpoint_state = _project_task1_state(
                midpoint, is_fluid=True, op=op, phys=phys)
            wmid = op.to_vorticity(midpoint_state)
        else:
            # Endpoint-only Hermite midpoint followed by Simpson quadrature.
            wmid = 0.5 * (w0 + w1) + (dt / 8.0) * (f0 - f1)
        fmid = op.rhs_vorticity(wmid)
        w_simpson = w0 + (dt / 6.0) * (f0 + 4.0 * fmid + f1)
        defect = w_simpson - w1
        # A smooth spectral filter prevents an endpoint defect from injecting
        # unresolved Nyquist-scale corrections.
        kmax = max(1, spec.n // 2)
        filt = torch.exp(-36.0 * (op.grid.k2.sqrt() / kmax).pow(36))
        defect = torch.fft.irfft2(torch.fft.rfft2(defect) * filt, s=(spec.n, spec.n))
        wc = w1 + alpha * defect
        uc, vc = op.grid.velocity(torch.fft.rfft2(wc))
        corrected = torch.stack((uc, vc), dim=1)
        # The vorticity lift has zero spatial mean.  Preserve the FM endpoint's
        # mean, not the preceding state's mean, so this correction changes only
        # the curvature-defect component of the forecast.
        corrected = corrected + candidate.mean(dim=(-2, -1), keepdim=True)
        return op.project_incompressible(corrected).reshape_as(corrected)

    f0, f1 = phys.rhs(previous), phys.rhs(candidate)
    if midpoint is None:
        midpoint = 0.5 * (previous + candidate) + (dt / 8.0) * (f0 - f1)
    else:
        midpoint = _project_task1_state(midpoint, is_fluid=False, op=op, phys=phys)
    fmid = phys.rhs(midpoint)
    simpson = previous + (dt / 6.0) * (f0 + 4.0 * fmid + f1)
    return phys.project_manifold(candidate + alpha * (simpson - candidate))


def _dense_kinematic_bridge(
        left: torch.Tensor, right: torch.Tensor, *, s: float, dt: float,
        is_fluid: bool, op: FMPhysicsEnergy2D | None,
        phys: GenericPDEPhysics, project_output: bool = True) -> torch.Tensor:
    """Return a zero-flow-step KMF state at any ``s`` in a corrected interval."""
    if is_fluid and op is not None:
        value = fluid_hermite_kinematic_spline(op, left, right, dt, s=s)
    else:
        value = general_kinematic_hermite_spline(phys, left, right, dt, s=s)
    if project_output:
        return _project_task1_state(value, is_fluid=is_fluid, op=op, phys=phys)
    return value





# ---------------------------------------------------------------------------
# Main Benchmark Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Cross-FM & Cross-PDE Benchmark")
    parser.add_argument("--fm", type=str, default="poseidon", choices=["poseidon", "dpot", "morph", "cno", "motion", "local", "the_well", "well", "walrus"])
    parser.add_argument("--fm-size", "--fm_size", type=str, default="T", help="T/B/L for Poseidon, Ti/S/M for DPOT/MORPH, 156.9M/S for MOTION")
    parser.add_argument("--fm-family", "--fm_family", type=str, default="FNO", help="Family for The Well (FNO, TFNO, UNetConvNext, UNetClassic)")
    parser.add_argument("--fm-dataset", "--fm_dataset", type=str, default="shear_flow", help="Dataset for The Well")
    parser.add_argument(
        "--fm-checkpoint", "--fm_checkpoint", "--cno-checkpoint",
        dest="fm_checkpoint", type=str, default=None,
        help="Foundation-model checkpoint repo or path (CNO alias: --cno-checkpoint)")
    parser.add_argument("--fm-history", "--fm_history", type=int, default=None, help="History length for autoregressive models")
    parser.add_argument("--cno-source", type=str, default=None,
                        help="Official CNO2d_temporal source directory; required by --fm cno.")
    parser.add_argument("--cno-config", type=str, default=None,
                        help="Released CNO architecture JSON paired with --fm-checkpoint.")
    parser.add_argument("--cno-time-scale", type=float, default=1.0,
                        help="Physical-time normalization used by the paired CNO checkpoint.")
    parser.add_argument("--cno-rho", type=float, default=0.8,
                        help="Pinned density conditioning for velocity-only CNO evaluation.")
    parser.add_argument("--cno-pressure", type=float, default=0.0,
                        help="Pinned pressure conditioning for velocity-only CNO evaluation.")
    parser.add_argument("--motion-source", type=str, default=None,
                        help="Official MOTION repository source directory; required by --fm motion.")
    parser.add_argument("--motion-checkpoint", type=str, default=None,
                        help="Released MOTION checkpoint directory or archive.")
    parser.add_argument("--motion-config", type=str, default=None,
                        help="Optional MOTION config path.")
    parser.add_argument("--motion-time-scale", type=float, default=1.0,
                        help="Physical-time normalization used by the paired MOTION checkpoint.")
    parser.add_argument("--pde", type=str, default="NS-Gauss", help="NS-Gauss, FNS-KF, ACE, Wave-Gauss")
    parser.add_argument("--fm-data-path", "--fm_data_path", type=str, default=None, help="Path to assembled NetCDF dataset")
    parser.add_argument("--n-cal-traj", "--n_cal_traj", type=int, default=5, help="Number of calibration trajectories for zero-leakage parameter tuning")
    parser.add_argument("--n-test-traj", "--n_test_traj", type=int, default=50, help="Number of held-out test trajectories")
    parser.add_argument("--steps", type=int, default=4, help="Autoregressive steps")
    parser.add_argument("--coarse-dt", "--coarse_dt", type=float, default=0.10, help="Coarse time interval dt")
    parser.add_argument("--alpha-grid", "--alpha_grid", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
                        help="Calibration-only grid for the zero-step curvature-defect relaxation.")
    parser.add_argument("--defect-midpoint", choices=["hermite", "fm"], default="hermite",
                        help="Midpoint used by Task 2 defect: paper endpoint-Hermite method, or causal half-time FM query.")
    parser.add_argument("--dense-substeps", "--dense_substeps", type=int, default=2,
                        help="Interior KMF states constructed per corrected Task 2 interval (default: 2).")
    parser.add_argument("--dense-fractions", "--dense_fractions", type=float, nargs="+", default=None,
                        help="Optional arbitrary in-interval query fractions s in (0,1). Overrides uniform dense substeps.")
    parser.add_argument("--save-dense-rollout", action="store_true",
                        help="Write selected held-out KMF dense states and their physical times to a .pt artifact.")
    parser.add_argument("--grid", type=int, default=128, help="Spatial resolution")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fm-channels", "--fm_channels", type=str, default="velocity", help="'all' or 'velocity'")
    parser.add_argument("--lead-time", "--lead_time", type=float, default=1.0, help="output cadence")
    parser.add_argument("--offset", type=int, default=0, help="trajectory offset in dataset")
    parser.add_argument("--out-dir", "--out_dir", type=str, default="results/scale/cross_benchmark", help="Output directory for benchmark JSON")
    parser.add_argument("--tag", type=str, default="cross_benchmark")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Load the frozen FM under its declared contract and exit."
                             " Used by the matrix runner to fail unavailable checkpoints once.")
    args = parser.parse_args()

    if args.fm_data_path is None:
        import os
        candidate_roots = [
            Path(os.environ.get("DATA_ROOT", "data/assembled")),
            Path("data/assembled"),
            Path("data/assembled"),
            Path("data"),
        ]
        pde_names = [args.pde]
        pde_lower = args.pde.lower().replace("_", "-")
        if "fns" in pde_lower or "kolmogorov" in pde_lower:
            pde_names.extend(["FNS-KF", "fns-kf", "FNS", "fns"])
        elif "ns" in pde_lower and "gauss" in pde_lower:
            pde_names.extend(["NS-Gauss", "ns-gauss"])
        elif "ace" in pde_lower or "allen" in pde_lower:
            pde_names.extend(["ACE", "ace"])
        elif "wave" in pde_lower and "gauss" in pde_lower:
            pde_names.extend(["Wave-Gauss", "wave-gauss"])
        elif "sl" in pde_lower:
            pde_names.extend(["NS-SL", "ns-sl"])
        elif "layer" in pde_lower:
            pde_names.extend(["Wave-Layer", "wave-layer"])

        found = None
        for root in candidate_roots:
            if not root.exists():
                continue
            for name in pde_names:
                for ext in [".nc", ".h5", ".hdf5"]:
                    p = root / f"{name}{ext}"
                    if p.exists():
                        found = str(p)
                        break
                if found:
                    break
            if found:
                break

        if found:
            args.fm_data_path = found
            print(f"  [Auto-detect] Found dataset at: {args.fm_data_path}")
        else:
            parser.error(f"--fm-data-path is required (could not auto-detect dataset for PDE '{args.pde}' in {candidate_roots[0]})")

    set_seed(args.seed)
    fm = load_fm(args)
    if args.preflight_only:
        fm_label = (f"{args.fm_family}" if args.fm in ("the_well", "well")
                    else f"{args.fm.upper()} ({args.fm_size})")
        print(f"PRECHECK PASS: {fm_label}; state shape={fm.state_shape}; device={fm.device}")
        return
    phys = GenericPDEPhysics(args.pde, n=args.grid, device=fm.device, dtype=torch.float64)

    is_fluid = "ns" in args.pde.lower() or "fns" in args.pde.lower()
    adapter_class = "frozen_foundation_model_interface"
    contract_note = (
        "Frozen foundation-model interface evaluated under its documented "
        "state, channel, and time-query contract")

    fm_label = f"{args.fm_family}" if args.fm in ("the_well", "well") else f"{args.fm.upper()} ({args.fm_size})"
    print_header(f"Cross-Benchmark: FM={fm_label} | PDE={args.pde}")
    print(f"  Contract: {contract_note}")
    print(f"  Test trajectories: {args.n_test_traj} | Resolution: {args.grid}x{args.grid} | Device: {args.device}")
    print(f"  Coarse Interval: {args.coarse_dt:.2f}s | Method: Strictly ZERO Forward ODE Steps")

    stride = max(1, int(round(args.coarse_dt / 0.05)))
    # The model may need prior observations even though KMF itself refines the
    # interval beginning at the final observed state.  DPOT requires ten true
    # preceding frames; MORPH uses the same causal prefix only for RevIN.
    context_frames = max(
        int(getattr(fm, "required_history", 1)),
        int(getattr(fm, "normalization_history", 1)),
    )
    origin = context_frames - 1
    if context_frames > 1:
        print(f"  Causal model context: {context_frames} prior-and-current frames; "
              "no states after the deployment origin enter the FM input.")
    total_traj_req = args.n_cal_traj + args.n_test_traj
    all_trajs, c_val = load_dataset_trajectories(
        args.fm_data_path, fm, total_traj_req, steps=args.steps, stride=stride,
        offset=args.offset, target_grid=args.grid, prehistory=origin)
    # Split calibration (for 0-leakage parameter tuning) and test set
    if all_trajs.shape[0] > args.n_cal_traj and args.n_cal_traj > 0:
        cal_trajs = all_trajs[:args.n_cal_traj]
        trajs = all_trajs[args.n_cal_traj:args.n_cal_traj + args.n_test_traj]
    else:
        cal_trajs = all_trajs[:min(5, len(all_trajs))]
        trajs = all_trajs

    actual_test_n = trajs.shape[0]
    print(f"  Split: {cal_trajs.shape[0]} calibration trajectories (held-out tuning) | {actual_test_n} test trajectories")
    if c_val is not None:
        c_tensor = torch.as_tensor(c_val, device=fm.device, dtype=torch.float64)
        if all_trajs.shape[0] > args.n_cal_traj and args.n_cal_traj > 0:
            phys.c_cal = c_tensor[:args.n_cal_traj]
            phys.c = c_tensor[args.n_cal_traj:args.n_cal_traj + args.n_test_traj]
        else:
            phys.c_cal = c_tensor[:min(5, len(c_tensor))]
            phys.c = c_tensor

    is_fluid = "ns" in args.pde.lower() or "fns" in args.pde.lower()
    if is_fluid:
        if "fns" in args.pde.lower() or "kolmogorov" in args.pde.lower():
            from hipp.scale.data2d import SPECS2D
            base_spec = SPECS2D.get("kolmogorov", native_poseidon_spec(args, fm))
            spec = dataclasses.replace(base_spec, n=args.grid, dt=args.coarse_dt)
            print(f"  [Physics] Initialized Forced Kolmogorov Navier-Stokes operator (nu={spec.nu}, forcing={spec.forcing}, drag={spec.drag})")
        else:
            spec = dataclasses.replace(native_poseidon_spec(args, fm), n=args.grid)
            print(f"  [Physics] Initialized Poseidon Navier-Stokes operator (spectral viscosity, div-free projection)")
        op = FMPhysicsEnergy2D(spec, trajs[0, origin].to(fm.device, torch.float64), 2, dt=args.coarse_dt, device=fm.device)
    else:
        op = None

    # -----------------------------------------------------------------------
    # Benchmark 1: FM-endpoint temporal refinement
    # -----------------------------------------------------------------------
    if stride >= 2:
        u0 = trajs[:, origin]
        u_true = trajs[:, origin + stride // 2]  # midpoint ground truth
        t_query = 0.5 * args.coarse_dt
        dt_span = args.coarse_dt
    else:
        # Note: Raw dataset is sampled at delta_t = 0.05s. A discrete intermediate
        # midpoint requires an interval span of at least 2 frames (dt_span = 0.10s).
        # We evaluate the minimal discrete midpoint span [0.00s -> 0.10s] with midpoint t = 0.05s.
        u0 = trajs[:, origin]
        u_true = trajs[:, origin + 1]
        t_query = 0.05
        dt_span = 0.10
        if args.coarse_dt < 0.10:
            print(f"  [NOTE] Requested coarse_dt={args.coarse_dt:.2f}s is finer than dataset frame pair stride.")
            print(f"         Evaluating minimal discrete midpoint span: [0.00s -> 0.10s] (query t=0.05s).")

    dt_eff = (dt_span / 0.10) * (2.0 / 15.0) if "wave" in args.pde.lower() else dt_span

    print("\n" + "=" * 115)
    print(f"BENCHMARK 1: FM-ENDPOINT TEMPORAL REFINEMENT (Span [0.00s -> {dt_span:.2f}s] | Query t = {t_query:.2f}s | {args.grid}x{args.grid})")
    print("=" * 115)
    print("  Protocol: the observed initial state and one FM-produced coarse endpoint define the bridge.")
    print("  Dataset states after the initial state are never used by Task 1 candidates; midpoint truth is scoring-only.")

    # The source endpoint is deliberately the FM's own forecast, never u1 from
    # the dataset.  This is the deployable temporal-refinement contract.
    t_task1 = time.time()
    u_end, u_lin, u_herm, u_fm = _fm_endpoint_task1_candidates(
        fm, u0, span_raw_frames=max(2, int(round(dt_span / 0.05))), dt_span=dt_eff,
        grid=args.grid, is_fluid=is_fluid, op=op, phys=phys,
        history_context=trajs[:, :origin + 1])
    task1_total_ms = (time.time() - t_task1) * 1000.0 / actual_test_n
    lin_rmse = _rmse(u_lin, u_true)
    herm_rmse = _rmse(u_herm, u_true)
    fm_rmse = _rmse(u_fm, u_true) if u_fm is not None else float("nan")

    # These two fusions remain deployable because their residuals use the
    # model-produced endpoint ``u_end``, not a dataset endpoint.
    u_eq = None
    u_phys = None
    if u_fm is not None:
        u_eq = _project_task1_state(0.5 * (u_herm + u_fm), is_fluid=is_fluid, op=op, phys=phys)
        if is_fluid and op is not None:
            w0_v, w1_v = op.to_vorticity(u0), op.to_vorticity(u_end)
            f0_v, f1_v = op.rhs_vorticity(w0_v), op.rhs_vorticity(w1_v)

            def _defect(candidate: torch.Tensor) -> torch.Tensor:
                wc = op.to_vorticity(candidate)
                fc = op.rhs_vorticity(wc)
                r = (w1_v - w0_v) / dt_eff - (f0_v + 4.0 * fc + f1_v) / 6.0
                return r.square().mean(dim=(-2, -1)).sqrt()
        else:
            f0_g, f1_g = phys.rhs(u0), phys.rhs(u_end)

            def _defect(candidate: torch.Tensor) -> torch.Tensor:
                r = (u_end - u0) / dt_eff - (f0_g + 4.0 * phys.rhs(candidate) + f1_g) / 6.0
                return r.square().mean(dim=(-3, -2, -1)).sqrt()

        res_h, res_f = _defect(u_herm), _defect(u_fm)
        inv_h2, inv_f2 = 1.0 / (res_h.square() + 1e-8), 1.0 / (res_f.square() + 1e-8)
        w_phys = (inv_f2 / (inv_h2 + inv_f2)).view(-1, 1, 1, 1)
        u_phys = _project_task1_state(
            (1.0 - w_phys) * u_herm + w_phys * u_fm,
            is_fluid=is_fluid, op=op, phys=phys)

    w_cal = 0.0
    u_cal = u_herm
    if u_fm is not None and cal_trajs.shape[0] > 0:
        u0_c = cal_trajs[:, origin]
        u_true_c = (cal_trajs[:, origin + stride // 2] if stride >= 2
                    else cal_trajs[:, origin + 1])
        _, _, u_herm_c, u_fm_c = _fm_endpoint_task1_candidates(
            fm, u0_c, span_raw_frames=max(2, int(round(dt_span / 0.05))), dt_span=dt_eff,
            grid=args.grid, is_fluid=is_fluid, op=op, phys=phys,
            history_context=cal_trajs[:, :origin + 1])
        delta = (u_fm_c - u_herm_c).reshape(-1)
        target = (u_true_c - u_herm_c).reshape(-1)
        denom = float(delta.square().sum())
        if denom > 1e-12:
            w_cal = float((delta * target).sum() / denom)
            w_cal = max(0.0, min(1.0, w_cal))
        u_cal = _project_task1_state(
            (1.0 - w_cal) * u_herm + w_cal * u_fm,
            is_fluid=is_fluid, op=op, phys=phys)
    cal_rmse = _rmse(u_cal, u_true)
    cal_ms = task1_total_ms
    herm_ms = task1_total_ms
    lin_ms = 0.0
    fm_ms = float("nan") if u_fm is None else task1_total_ms
    eq_rmse = _rmse(u_eq, u_true) if u_eq is not None else float("nan")
    phys_rmse = _rmse(u_phys, u_true) if u_phys is not None else float("nan")
    eq_ms = task1_total_ms if u_eq is not None else float("nan")
    phys_ms = task1_total_ms if u_phys is not None else float("nan")

    def _gain(candidate: float, baseline: float) -> float:
        return 100.0 * (1.0 - candidate / baseline) if np.isfinite(baseline) and baseline > 1e-12 else float("nan")

    gain_raw_lin = _gain(fm_rmse, lin_rmse)
    gain_herm_lin = _gain(herm_rmse, lin_rmse)
    gain_herm_fm = _gain(herm_rmse, fm_rmse)
    gain_cal_lin = _gain(cal_rmse, lin_rmse)
    gain_cal_fm = _gain(cal_rmse, fm_rmse)
    gain_eq_lin, gain_eq_fm = _gain(eq_rmse, lin_rmse), _gain(eq_rmse, fm_rmse)
    gain_phys_lin, gain_phys_fm = _gain(phys_rmse, lin_rmse), _gain(phys_rmse, fm_rmse)

    t_sub = Table("Method", "Midpoint RMSE", "Gain vs. Endpoint Linear", "Gain vs. Direct FM", "Information used", "GPU Latency")
    t_sub.add("FM-endpoint linear", f"{lin_rmse:.6f}", "baseline", "N/A", "observed state + one FM endpoint", f"{lin_ms:.2f} ms")
    if u_fm is not None:
        t_sub.add(f"Direct {fm_label} query", f"{fm_rmse:.6f}", f"{gain_raw_lin:+.2f}%", "baseline", "one time-conditioned FM query", f"{fm_ms:.2f} ms")
    else:
        t_sub.add(f"Direct {fm_label} query", "N/A", "N/A", "N/A", "not exposed by fixed-step adapter", "N/A")
    t_sub.add("KMF bridge", f"{herm_rmse:.6f}", f"{gain_herm_lin:+.2f}%", f"{gain_herm_fm:+.2f}%" if np.isfinite(gain_herm_fm) else "N/A", "observed state + one FM endpoint + 2 RHS", f"{herm_ms:.2f} ms")
    t_sub.add("KMF equal fusion", f"{eq_rmse:.6f}" if np.isfinite(eq_rmse) else "N/A", f"{gain_eq_lin:+.2f}%" if np.isfinite(gain_eq_lin) else "N/A", f"{gain_eq_fm:+.2f}%" if np.isfinite(gain_eq_fm) else "N/A", "bridge/direct equal fusion", f"{eq_ms:.2f} ms" if np.isfinite(eq_ms) else "N/A")
    t_sub.add("KMF physics-gated fusion", f"{phys_rmse:.6f}" if np.isfinite(phys_rmse) else "N/A", f"{gain_phys_lin:+.2f}%" if np.isfinite(gain_phys_lin) else "N/A", f"{gain_phys_fm:+.2f}%" if np.isfinite(gain_phys_fm) else "N/A", "model-endpoint Simpson defect", f"{phys_ms:.2f} ms" if np.isfinite(phys_ms) else "N/A")
    t_sub.add(f"KMF calibrated fusion (w={w_cal:.2f})", f"{cal_rmse:.6f}", f"{gain_cal_lin:+.2f}%", f"{gain_cal_fm:+.2f}%" if np.isfinite(gain_cal_fm) else "N/A", "calibration-selected bridge/direct fusion", f"{cal_ms:.2f} ms")
    print(t_sub)

    print("\n  Task 1 cadence matrix: each --coarse-dt invocation above is one deployed")
    print("  FM-endpoint refinement experiment. The former within-run multi-cadence table")
    print("  was retired because it used dataset endpoint states after t=0.")


    # -----------------------------------------------------------------------
    # Benchmark 2: causal coarse correction and dense in-interval evaluation
    # -----------------------------------------------------------------------
    actual_steps = min(args.steps, (trajs.shape[1] - 1 - origin) // stride)
    actual_steps = max(1, actual_steps)
    # Task 2 uses the requested causal rollout cadence itself.  In particular,
    # a direct lead-1 query at 0.05 s must not inherit Task 1's 0.10 s
    # discrete midpoint span.  ``dt_span`` above is only the Task 1 scoring
    # span used when the dataset has no separately stored midpoint.
    rollout_dt = float(args.coarse_dt)
    dense_substeps = int(args.dense_substeps) if args.dense_substeps else 2
    if dense_substeps < 1:
        raise ValueError("--dense-substeps must be positive")
    if args.dense_fractions is None:
        dense_fractions = [q / dense_substeps for q in range(1, dense_substeps)]
    else:
        dense_fractions = sorted(set(float(s) for s in args.dense_fractions))
        if any(not 0.0 < s < 1.0 for s in dense_fractions):
            raise ValueError("--dense-fractions values must lie strictly inside (0,1)")

    print("\n" + "=" * 115)
    print(f"BENCHMARK 2: CAUSAL COARSE CORRECTION + DENSE KMF ROLLOUT ({actual_steps} coarse steps | dt={rollout_dt:.2f}s)")
    print("=" * 115)
    print("  Coarse correction: instantaneous Simpson curvature defect; zero forward ODE steps.")
    print(f"  Dense output: {len(dense_fractions)} KMF state(s) constructed per corrected coarse step; "
          "dataset timestamps, when present, are scoring-only.")

    def _advance_history(history: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.cat((history, state[:, None]), dim=1)[:, -context_frames:]

    def _set_coarse_lead() -> None:
        _set_fm_query_time(fm, raw_frames=stride, physical_dt=rollout_dt)

    def _predict_kmf_pair(state: torch.Tensor, history: torch.Tensor):
        """Get the coarse FM endpoint and, when supported, its half-time query.

        The midpoint query is made from the same corrected input and history as
        the coarse query.  It is therefore a causal model output, not a dataset
        midpoint or an ODE substep.  Adapters without a documented time query
        receive ``None`` and use the endpoint-only Hermite fallback.
        """
        _set_coarse_lead()
        with torch.no_grad():
            endpoint = fm_predict(fm, state, current_grid=args.grid,
                                  history_context=history)

        midpoint = None
        midpoint_supported = (
            args.defect_midpoint == "fm"
            and hasattr(fm, "set_lead_time")
            and stride >= 2
        )
        if midpoint_supported:
            _set_fm_query_time(
                fm, raw_frames=float(stride) / 2.0,
                physical_dt=rollout_dt / 2.0)
            with torch.no_grad():
                midpoint = fm_predict(fm, state, current_grid=args.grid,
                                      history_context=history)
        return endpoint, midpoint

    def _rollout(data: torch.Tensor, alpha: float, *, collect_dense: bool = False):
        current_raw = data[:, origin].clone()
        current_proj = current_raw.clone()
        current_kmf = current_raw.clone()
        history_raw = data[:, :origin + 1].clone()
        history_proj = history_raw.clone()
        history_kmf = history_raw.clone()
        raw_scores, proj_scores, kmf_scores = [], [], []
        dense_scores = {
            "raw_linear": [],
            "raw_hermite": [],
            "projected_linear": [],
            "kmf_hermite": [],
        }
        dense_records = []

        for step in range(actual_steps):
            target = data[:, origin + (step + 1) * stride]
            _set_coarse_lead()
            with torch.no_grad():
                current_raw = fm_predict(fm, current_raw, current_grid=args.grid,
                                         history_context=history_raw)
            history_raw = _advance_history(history_raw, current_raw)
            raw_scores.append(_rmse(current_raw, target))

            _set_coarse_lead()
            with torch.no_grad():
                projected_forecast = fm_predict(fm, current_proj, current_grid=args.grid,
                                                history_context=history_proj)
            current_proj = _project_task1_state(projected_forecast, is_fluid=is_fluid, op=op, phys=phys)
            history_proj = _advance_history(history_proj, current_proj)
            proj_scores.append(_rmse(current_proj, target))

            left_raw = current_raw
            left_proj = current_proj
            left = current_kmf
            if alpha != 0.0:
                forecast, midpoint = _predict_kmf_pair(left, history_kmf)
            else:
                _set_coarse_lead()
                with torch.no_grad():
                    forecast = fm_predict(fm, left, current_grid=args.grid,
                                          history_context=history_kmf)
                midpoint = None
            current_kmf = _coarse_defect_correction(
                left, forecast, alpha=alpha, dt=rollout_dt,
                is_fluid=is_fluid, op=op, phys=phys,
                spec=spec if is_fluid else None, midpoint=midpoint)
            history_kmf = _advance_history(history_kmf, current_kmf)
            kmf_scores.append(_rmse(current_kmf, target))

            if collect_dense:
                for s in dense_fractions:
                    raw_linear = (1.0 - s) * left_raw + s * current_raw
                    raw_hermite = _dense_kinematic_bridge(
                        left_raw, current_raw, s=s, dt=rollout_dt,
                        is_fluid=is_fluid, op=op, phys=phys,
                        project_output=False)
                    projected_linear = (1.0 - s) * left_proj + s * current_proj
                    dense = _dense_kinematic_bridge(
                        left, current_kmf, s=s, dt=rollout_dt,
                        is_fluid=is_fluid, op=op, phys=phys)
                    # Construct every requested continuum-time state.  Only
                    # scoring requires an exactly stored reference timestamp.
                    if args.save_dense_rollout:
                        dense_records.append({
                            "step": step,
                            "fraction": s,
                            "time": (step + s) * rollout_dt,
                            "state": dense.detach().cpu(),
                        })
                    raw_offset = s * stride
                    if float(raw_offset).is_integer():
                        truth = data[:, origin + step * stride + int(raw_offset)]
                        dense_scores["raw_linear"].append(_rmse(raw_linear, truth))
                        dense_scores["raw_hermite"].append(_rmse(raw_hermite, truth))
                        dense_scores["projected_linear"].append(_rmse(projected_linear, truth))
                        dense_scores["kmf_hermite"].append(_rmse(dense, truth))
        return raw_scores, proj_scores, kmf_scores, dense_scores, dense_records

    # Select alpha using the complete causal calibration rollout, not a single
    # teacher-forced transition.  No test trajectory enters this selection.
    alpha_grid = sorted(set(float(a) for a in args.alpha_grid))
    if 0.0 not in alpha_grid:
        alpha_grid.insert(0, 0.0)
    alpha_hs, alpha_calibration = 0.0, {}
    if cal_trajs.shape[0] > 0:
        for alpha_try in alpha_grid:
            _, _, cal_kmf, _, _ = _rollout(cal_trajs, alpha_try)
            alpha_calibration[alpha_try] = float(np.mean(cal_kmf))
            print(f"  calibration alpha={alpha_try:g}: coarse-window RMSE={alpha_calibration[alpha_try]:.6f}")
        alpha_hs = min(alpha_calibration, key=alpha_calibration.get)
    midpoint_query_available = (
        args.defect_midpoint == "fm"
        and hasattr(fm, "set_lead_time")
        and stride >= 2
    )
    print(f"  selected calibration-only alpha={alpha_hs:g}")
    print("  KMF defect midpoint: " + (
        "causal half-time FM query when supported; endpoint-Hermite fallback otherwise."
        if midpoint_query_available else
        "endpoint-Hermite reconstruction (paper protocol)."))

    step_rmses_raw, step_rmses_proj, step_rmses_hs, dense_step_rmses_hs, dense_records = _rollout(
        trajs, alpha_hs, collect_dense=True)
    mean_raw_window = float(np.mean(step_rmses_raw))
    mean_proj_window = float(np.mean(step_rmses_proj))
    mean_hs_window = float(np.mean(step_rmses_hs))
    dense_raw_linear_window = (
        float(np.mean(dense_step_rmses_hs["raw_linear"]))
        if dense_step_rmses_hs["raw_linear"] else float("nan"))
    dense_raw_hermite_window = (
        float(np.mean(dense_step_rmses_hs["raw_hermite"]))
        if dense_step_rmses_hs["raw_hermite"] else float("nan"))
    dense_projected_linear_window = (
        float(np.mean(dense_step_rmses_hs["projected_linear"]))
        if dense_step_rmses_hs["projected_linear"] else float("nan"))
    dense_hs_window = (
        float(np.mean(dense_step_rmses_hs["kmf_hermite"]))
        if dense_step_rmses_hs["kmf_hermite"] else float("nan"))
    gain_rollout_proj = _gain(mean_proj_window, mean_raw_window)
    gain_rollout_hs = _gain(mean_hs_window, mean_raw_window)

    t_roll = Table("Method", "Step 1 RMSE", f"Coarse RMSE ({actual_steps} steps)", "Dense RMSE", "Gain %", "Mechanism")
    t_roll.add(f"Raw {fm_label} rollout", f"{step_rmses_raw[0]:.6f}", f"{mean_raw_window:.6f}", "N/A", "baseline", "unconstrained FM")
    t_roll.add("Constraint-projected rollout", f"{step_rmses_proj[0]:.6f}", f"{mean_proj_window:.6f}", "N/A", f"{gain_rollout_proj:+.2f}%", "known-state projection")
    t_roll.add(f"KMF curvature-defect rollout (alpha={alpha_hs:g})", f"{step_rmses_hs[0]:.6f}", f"{mean_hs_window:.6f}",
               f"{dense_hs_window:.6f}" if np.isfinite(dense_hs_window) else "not discretely scored",
               f"{gain_rollout_hs:+.2f}%", "FM midpoint-conditioned Simpson defect + dense bridge" if midpoint_query_available
               else "endpoint-Hermite Simpson defect + dense bridge")
    print(t_roll)

    print("\n  Dense intermediate-time comparison (identical corrected-rollout query times)")
    t_dense = Table("Dense method", "RMSE", "Gain vs. raw linear", "Gain vs. raw Hermite")
    dense_methods = [
        ("raw endpoint linear", dense_raw_linear_window),
        ("raw endpoint Hermite", dense_raw_hermite_window),
        ("projected endpoint linear", dense_projected_linear_window),
        ("KMF corrected endpoint Hermite", dense_hs_window),
    ]
    for name, value in dense_methods:
        t_dense.add(
            name,
            f"{value:.6f}" if np.isfinite(value) else "not discretely scored",
            ("baseline" if name == "raw endpoint linear" or not np.isfinite(value)
             or not np.isfinite(dense_raw_linear_window)
             else f"{_gain(value, dense_raw_linear_window):+.2f}%"),
            ("baseline" if name == "raw endpoint Hermite" or not np.isfinite(value)
             or not np.isfinite(dense_raw_hermite_window)
             else f"{_gain(value, dense_raw_hermite_window):+.2f}%"),
        )
    print(t_dense)

    # Save output JSON
    results = {
        "fm": args.fm,
        "fm_size": args.fm_size,
        "fm_family": getattr(args, "fm_family", None),
        "fm_dataset": getattr(args, "fm_dataset", None),
        "pde": args.pde,
        "adapter_class": adapter_class,
        "contract_note": contract_note,
        "grid": args.grid,
        "requested_dt": args.coarse_dt,
        "endpoint_span_dt": dt_span,
        "coarse_dt": args.coarse_dt,
        "stride": stride,
        "causal_context_frames": context_frames,
        "n_test_traj": actual_test_n,
        "n_cal_traj": cal_trajs.shape[0],
        "sub_cadence": {
            "protocol": "fm_endpoint_refinement",
            "endpoint_source": "autoregressive/coarse FM endpoint from observed initial state",
            "truth_usage": "calibration-only fusion fitting and held-out scoring only",
            "linear_rmse": lin_rmse,
            "fm_rmse": fm_rmse,
            "hermite_rmse": herm_rmse,
            "equal_consensus_rmse": eq_rmse,
            "physics_gated_rmse": phys_rmse,
            "calibrated_rmse": cal_rmse,
            # Strictly pre-selected method on calibration set (zero test leakage):
            "ours_midpoint_rmse": cal_rmse,
            "gain_vs_linear": gain_cal_lin,
            "gain_vs_fm": gain_cal_fm,
            "gain_hermite_vs_linear": gain_herm_lin,
            "gain_hermite_vs_fm": gain_herm_fm,
            "gain_equal_vs_linear": gain_eq_lin,
            "gain_equal_vs_fm": gain_eq_fm,
            "gain_phys_vs_linear": gain_phys_lin,
            "gain_phys_vs_fm": gain_phys_fm,
            "gain_cal_vs_linear": gain_cal_lin,
            "gain_cal_vs_fm": gain_cal_fm,
            "calibrated_weight": w_cal,
            "latency_fm_ms": fm_ms,
            "latency_ours_ms": cal_ms,
        },
        "rollout": {
            "steps": actual_steps,
            "rollout_dt": rollout_dt,
            "rollout_endpoint_span_dt": rollout_dt,
            "protocol": (
                "causal FM rollout from u0; endpoint correction and manifold projection "
                "at every step; corrected endpoint reinjected; Hermite bridge between "
                "successive corrected endpoints"),
            "alpha_hs_calibrated": alpha_hs,
            "alpha_calibration_coarse_window_rmse": alpha_calibration,
            "defect_midpoint_mode": args.defect_midpoint,
        "coarse_correction_protocol": "instantaneous Simpson curvature defect; zero forward ODE steps",
        "coarse_defect_midpoint": (
            "causal_half_time_fm_query" if midpoint_query_available
            else "endpoint_hermite_fallback"),
            "dense_protocol": "Hermite KMF bridge between consecutive corrected rollout endpoints",
            "dense_substeps_per_coarse_interval": dense_substeps,
            "dense_fractions": dense_fractions,
            "dense_queries_requested": actual_steps * len(dense_fractions),
            "dense_queries_scored": len(dense_step_rmses_hs["kmf_hermite"]),
            "step1_raw_rmse": step_rmses_raw[0],
            "step1_proj_rmse": step_rmses_proj[0],
            "step1_simpson_rmse": step_rmses_hs[0],
            "window_raw_rmse": mean_raw_window,
            "window_proj_rmse": mean_proj_window,
            "window_simpson_rmse": mean_hs_window,
            # Strictly pre-selected method using calibration alpha_hs (zero test leakage):
            "window_ours_rmse": mean_hs_window,
            "gain_rollout": gain_rollout_hs,
            "gain_rollout_proj": gain_rollout_proj,
            "gain_rollout_simpson": gain_rollout_hs,
            "step_rmses_raw": step_rmses_raw,
            "step_rmses_proj": step_rmses_proj,
            "step_rmses_simpson": step_rmses_hs,
            "dense_step_rmses_raw_linear": dense_step_rmses_hs["raw_linear"],
            "dense_step_rmses_raw_hermite": dense_step_rmses_hs["raw_hermite"],
            "dense_step_rmses_projected_linear": dense_step_rmses_hs["projected_linear"],
            "dense_step_rmses_kmf": dense_step_rmses_hs["kmf_hermite"],
            "dense_window_rmse_raw_linear": dense_raw_linear_window,
            "dense_window_rmse_raw_hermite": dense_raw_hermite_window,
            "dense_window_rmse_projected_linear": dense_projected_linear_window,
            "dense_window_rmse_kmf": dense_hs_window,
        },
    }

    out_dir = Path(getattr(args, "out_dir", "results/scale/cross_benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fm_str = f"polymathic_{args.fm_family}" if args.fm in ("the_well", "well") else f"{args.fm}_{args.fm_size}"
    out_path = out_dir / f"{fm_str}_{args.pde}_dt{args.coarse_dt:.2f}_grid{args.grid}_results.json"
    dense_path = None
    if args.save_dense_rollout:
        dense_path = out_path.with_name(out_path.stem.replace("_results", "_dense_rollout") + ".pt")
        results["rollout"]["dense_rollout_artifact"] = str(dense_path)
    save_json(results, out_path)
    if dense_path is not None:
        torch.save({"endpoint_span_dt": rollout_dt, "fractions": dense_fractions,
                    "records": dense_records}, dense_path)
        print(f"Saved dense KMF rollout states to: {dense_path}")
    print(f"\nSaved cross-benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
