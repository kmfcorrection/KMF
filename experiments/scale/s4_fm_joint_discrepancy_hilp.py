#!/usr/bin/env python3
"""S4 joint innovation--residual discrepancy HILP.

This is a separate implementation of the coarse-residual modification in
``docs/hilp_revised_math_body.tex``.  It retains HILP's block innovation
operator, but does not assume that a coarse residual's discrepancy is
independent of the FM innovation:

    xi = B e,
    z = -phi(r_h(mu)) = G xi + eta,
    Cov[(xi, eta)] = [[Q, S], [S^T, R]].

For centered variables the block correction is

    delta = B^{-1} (Q G^T + S)
            (G Q G^T + G S + S^T G^T + R)^{-1} z.

The residual feature map phi is deliberately small and fixed: real/imaginary
low Fourier coefficients of the vorticity residual at each output time.  Truth
is used only on calibration trajectories.  The joint covariance is estimated
as a convex mixture of a structured positive target and the empirical joint
covariance of (xi, eta), so it stays positive semidefinite without clipping S.
Validation selects the mixture weight and a global safe correction gain;
held-out trajectories are untouched until final scoring.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (configure_native_poseidon_cadence,
    fm_metadata, fm_result_key, load_poseidon_trajectories, native_poseidon_spec)
from experiments.scale.s4_fm_revised_hilp import InnovationPrior
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.data2d import SpectralGrid2D
from hipp.scale.fm_physics import FMPhysicsEnergy2D, FixedRK3FlowEnergy2D
from hipp.scale.jacobian import jacobian_ops
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("raw", "isotropic_s0", "block_s0", "block_joint", "rk3_flow_only")


def _modes(n: int, count: int) -> list[tuple[int, int]]:
    """Fixed non-DC Fourier probes, ordered from coarse to fine."""
    candidates = [(ky, kx) for ky in range(0, min(8, n // 2))
                  for kx in range(0, min(8, n // 2 + 1)) if ky or kx]
    candidates.sort(key=lambda q: (q[0] ** 2 + q[1] ** 2, q[0], q[1]))
    return candidates[:count]


class CoarseResidualFeatures:
    """Differentiable, fixed low-dimensional features of a coarse PDE defect."""
    def __init__(self, spec, x0, raw, modes, device):
        self.T, self.N = raw.shape
        self.n, self.device = spec.n, device
        self.modes = list(modes)
        # The operator supplies spectral derivatives and the AZEBAN RHS.  Its
        # stored previous field is not used below: residual intervals are
        # formed directly from the full candidate window so A includes every
        # endpoint's influence on its adjacent temporal defects.
        self.operator = FMPhysicsEnergy2D(spec, x0, 2, divergence_weight=0.0,
                                          dt=spec.dt_out, device=device)
        self.x0 = x0.detach().to(device, torch.float64).reshape(1, self.N)

    @property
    def dimension(self):
        return 2 * len(self.modes) * self.T

    def __call__(self, candidate):
        x = candidate.reshape(self.T, self.N).to(self.device, torch.float64)
        states = torch.cat((self.x0, x), dim=0)
        vort = self.operator.to_vorticity(states)
        midpoint = 0.5 * (vort[:-1] + vort[1:])
        residuals = (vort[1:] - vort[:-1]) / self.operator.dt - self.operator._native_rhs(midpoint)
        values = []
        for residual in residuals:
            # One actual coarse time interval; no flow solve and no invented
            # temporal node is used by this residual.
            rh = torch.fft.rfft2(residual) / (self.n * self.n)
            for ky, kx in self.modes:
                c = rh[ky, kx]
                values.extend((c.real, c.imag))
        return torch.stack(values)


class FixedRK3FlowDefectFeatures:
    """Fourier features of x_{t+1}-Psi_RK3(x_t^FM).

    Psi_RK3 is intentionally a fixed-budget, imperfect full-grid physical
    map.  Each target is computed once from the observed initial state or the
    preceding raw FM output.  Thus inference never reads future truth and no
    PDE solve is nested inside the posterior update.  Calibration estimates
    the remaining flow-defect discrepancy eta.
    """
    def __init__(self, spec, x0, raw, modes, flow_substeps, device):
        self.T, self.N = raw.shape
        self.n, self.device = spec.n, device
        self.modes = list(modes)
        targets, previous = [], x0.detach().to(device, torch.float64).reshape(-1)
        for t in range(self.T):
            energy = FixedRK3FlowEnergy2D(spec, previous, 2, divergence_weight=0.0,
                                           flow_substeps=flow_substeps, dt=spec.dt_out,
                                           device=device)
            targets.append(energy.flow_target.detach().reshape(-1))
            previous = raw[t].detach()
        self.targets = torch.stack(targets)
        self.projector = FMPhysicsEnergy2D(spec, x0, 2, divergence_weight=0.0,
                                           dt=spec.dt_out, device=device)

    @property
    def dimension(self):
        return 2 * len(self.modes) * self.T

    def __call__(self, candidate):
        defect = candidate.reshape(self.T, self.N).to(self.device, torch.float64) - self.targets
        vort = self.projector.to_vorticity(defect)
        values = []
        for field in vort:
            dh = torch.fft.rfft2(field) / (self.n * self.n)
            for ky, kx in self.modes:
                c = dh[ky, kx]
                values.extend((c.real, c.imag))
        return torch.stack(values)


class JointFixedRK3FlowDefectFeatures:
    """Fourier features of the jointly conditioned defect x_{t+1}-Psi(x_t).

    Unlike ``FixedRK3FlowDefectFeatures``, every interval conditions the
    physical map on the candidate predecessor.  Hence the truth residual is
    evaluated on consecutive truth states and is directly comparable to the
    raw-FM residual.  Psi uses a fixed, deliberately limited RK3 budget and
    is differentiated only while forming the local observation Jacobian.
    """
    def __init__(self, spec, x0, raw, modes, flow_substeps, device):
        self.T, self.N = raw.shape
        self.n, self.device = spec.n, device
        self.modes, self.flow_substeps = list(modes), int(flow_substeps)
        self.x0 = x0.detach().to(device, torch.float64).reshape(-1)
        # This supplies immutable spectral operators. Its detached conditioning
        # field is never used by _flow; candidate states remain differentiable.
        self.operator = FMPhysicsEnergy2D(spec, self.x0, 2, divergence_weight=0.0,
                                          dt=spec.dt_out, device=device)

    @property
    def dimension(self):
        return 2 * len(self.modes) * self.T

    def _flow(self, previous):
        state = previous.reshape(1, 2, self.n, self.n).to(self.device, torch.float64)
        u, v = state[:, 0], state[:, 1]
        wh = 1j * self.operator.grid.kx * torch.fft.rfft2(v) - 1j * self.operator.grid.ky * torch.fft.rfft2(u)
        w = torch.fft.irfft2(wh, s=(self.n, self.n))
        w_next = self.operator.native_flow_vorticity(w, substeps=self.flow_substeps)
        uh = torch.fft.rfft2(w_next)
        next_u, next_v = self.operator.grid.velocity(uh)
        # The zero velocity mode is conserved by this periodic, unforced flow.
        next_u = next_u + u.mean(dim=(-2, -1), keepdim=True)
        next_v = next_v + v.mean(dim=(-2, -1), keepdim=True)
        return torch.stack((next_u, next_v), dim=1).reshape(-1)

    def __call__(self, candidate):
        states = torch.cat((self.x0.reshape(1, self.N),
                            candidate.reshape(self.T, self.N).to(self.device, torch.float64)), dim=0)
        defects = torch.stack([states[t + 1] - self._flow(states[t]) for t in range(self.T)])
        vort = self.operator.to_vorticity(defects)
        values = []
        for field in vort:
            dh = torch.fft.rfft2(field) / (self.n * self.n)
            for ky, kx in self.modes:
                c = dh[ky, kx]
                values.extend((c.real, c.imag))
        return torch.stack(values)


class StructuredInnovationSubspace:
    """Fixed divergence-free Fourier coordinates for the innovation window.

    The cross covariance is estimable only after restricting xi to this small,
    physically defined subspace.  For every selected spatial Fourier mode we
    use its real and imaginary divergence-free velocity fields, and repeat
    those coordinates independently at each output time.
    """
    def __init__(self, spec, steps, modes, device):
        grid = SpectralGrid2D(spec, device=device, dtype=torch.float64)
        fields = []
        for ky, kx in modes:
            for phase in (1.0, 1.0j):
                wh = torch.zeros((spec.n, spec.n // 2 + 1), dtype=torch.complex128, device=device)
                wh[ky, kx] = phase
                u, v = grid.velocity(wh)
                fields.append(torch.stack((u, v)).reshape(-1))
        # QR removes normalization and small finite-grid nonorthogonalities.
        spatial, _ = torch.linalg.qr(torch.stack(fields, dim=1), mode="reduced")
        self.spatial = spatial
        self.T = int(steps)
        self.N = int(spatial.shape[0])
        self.basis = torch.block_diag(*([spatial] * self.T)).contiguous()

    @property
    def dimension(self):
        return int(self.basis.shape[1])

    def project(self, xi):
        return self.basis.T @ xi.reshape(-1).double()


def _solve_B(prior: InnovationPrior, xi: torch.Tensor) -> torch.Tensor:
    """Matrix-free e=B^{-1}xi using the local JVP chain."""
    z = xi.reshape(prior.T, prior.N).double()
    out = [z[0]]
    for t, op in enumerate(prior.ops):
        out.append(z[t + 1] + op.matvec(out[-1].to(op.dtype)).double())
    return torch.stack(out)


def _solve_BT(prior: InnovationPrior, rhs: torch.Tensor) -> torch.Tensor:
    """Matrix-free y=B^{-T}rhs using VJPs, solved backward in time."""
    w = rhs.reshape(prior.T, prior.N).double()
    out = [None] * prior.T
    out[-1] = w[-1]
    for t in range(prior.T - 2, -1, -1):
        out[t] = w[t] + prior.ops[t].rmatvec(out[t + 1].to(prior.ops[t].dtype)).double()
    return torch.stack(out)


def _raw_window_and_ops_timed(fm, x0, steps):
    """Raw autoregressive path with separate FM and local-Jacobian timings."""
    raw, ops = [], []
    forward_seconds = jacobian_seconds = 0.0
    current = x0.to(fm.device, fm.dtype)
    for t in range(steps):
        started = time.perf_counter()
        nxt = fm.predict(current).detach().reshape(-1)
        forward_seconds += time.perf_counter() - started
        raw.append(nxt.double())
        if t + 1 < steps:
            started = time.perf_counter()
            ops.append(jacobian_ops(fm.flat_fn(), nxt))
            jacobian_seconds += time.perf_counter() - started
        current = nxt
    return torch.stack(raw), ops, {"fm_forward_seconds": forward_seconds,
                                   "fm_jacobian_setup_seconds": jacobian_seconds}


def _feature_jacobians(prior: InnovationPrior, feature: CoarseResidualFeatures):
    """Return A^T, G^T, and the *positive* residual feature r(mu).

    The observation used by the conditional-Gaussian update is z=-r(mu),
    because r(mu + e) ~= r(mu) + A e and a physical trajectory has small
    residual.  Keeping the derivative of r positive while negating only the
    """
    started = time.perf_counter()
    candidate = prior.mean.detach().clone().requires_grad_(True)
    residual = feature(candidate)
    feature_seconds = time.perf_counter() - started
    cols_a, cols_g = [], []
    started = time.perf_counter()
    for k in range(residual.numel()):
        (grad,) = torch.autograd.grad(residual[k], candidate, retain_graph=True)
        a_col = grad.detach().reshape(prior.T, prior.N)
        cols_a.append(a_col.reshape(-1))
        cols_g.append(_solve_BT(prior, a_col).reshape(-1))
    return (torch.stack(cols_a, dim=1), torch.stack(cols_g, dim=1), residual.detach(),
            {"physics_feature_seconds": feature_seconds,
             "physics_vjp_seconds": time.perf_counter() - started})


def _package(fm, trajectories, spec, modes, steps, innovation_subspace, args):
    packages = []
    timings = {"fm_forward_seconds": 0.0, "fm_jacobian_setup_seconds": 0.0,
               "physics_feature_seconds": 0.0, "physics_vjp_seconds": 0.0,
               "flow_only_seconds": 0.0}
    for i, truth in enumerate(trajectories):
        print(f"    preparing trajectory {i + 1}/{len(trajectories)}", flush=True)
        if not torch.isfinite(truth).all():
            raise RuntimeError(f"non-finite dataset state in trajectory {i + 1}; aborting before calibration")
        raw, ops, raw_timing = _raw_window_and_ops_timed(fm, truth[0], steps)
        if not torch.isfinite(raw).all():
            raise RuntimeError(f"non-finite FM rollout in trajectory {i + 1}; aborting before calibration")
        for key, value in raw_timing.items():
            timings[key] += value
        prior = InnovationPrior(raw, ops, q=1.0)
        if args.observation == "midpoint_fourier":
            feature = CoarseResidualFeatures(spec, truth[0], raw, modes, fm.device)
        elif args.observation == "fixed_rk3_flow_defect":
            feature = FixedRK3FlowDefectFeatures(spec, truth[0], raw, modes,
                                                  args.flow_defect_substeps, fm.device)
        else:
            feature = JointFixedRK3FlowDefectFeatures(spec, truth[0], raw, modes,
                                                       args.flow_defect_substeps, fm.device)
        at, gt, residual, physics_timing = _feature_jacobians(prior, feature)
        if not torch.isfinite(residual).all() or not torch.isfinite(at).all() or not torch.isfinite(gt).all():
            bad = torch.nonzero(~torch.isfinite(residual), as_tuple=False).flatten().tolist()
            interval = (bad[0] // (2 * len(modes)) + 1) if bad else "unknown"
            raise RuntimeError(
                "non-finite physical observation/Jacobian in trajectory "
                f"{i + 1}, interval {interval}, with fixed RK3 substeps="
                f"{args.flow_defect_substeps}. This fixed-step budget is not "
                "stable on this split; do not fit a covariance from this run. "
                "Increase the pre-specified fixed substep budget and rerun the "
                "calibration gate.")
        for key, value in physics_timing.items():
            timings[key] += value
        # With r(raw + e) ~= r(raw) + A e and r(truth) near zero,
        # z=-r(raw) satisfies z ~= A e = G xi.  The old runner accidentally
        # used +r(raw), which sends the posterior in the wrong direction.
        z = -residual
        error = truth[1:].double() - raw
        xi = prior.apply_B(error)
        # z = G xi + eta under the first-order coarse-residual model.
        eta = z - gt.T @ xi.reshape(-1)
        flow_only = None
        if args.observation == "joint_fixed_rk3_flow_defect":
            # Required solver-strength control: recursively advance exactly
            # the same fixed-budget map from the known initial state.  This is
            # never used to form an HILP correction or select hyperparameters.
            with torch.no_grad():
                started = time.perf_counter()
                current, path = truth[0].detach().to(fm.device, torch.float64).reshape(-1), []
                for _ in range(steps):
                    current = feature._flow(current).detach()
                    path.append(current)
                flow_only = torch.stack(path)
                timings["flow_only_seconds"] += time.perf_counter() - started
        packages.append({"raw": raw, "truth": truth[1:].double(), "prior": prior,
                         "at": at, "gt": gt, "z": z, "xi": xi.reshape(-1), "xi_window": xi,
                         "eta": eta, "feature": feature,
                         "u": innovation_subspace.project(xi), "flow_only": flow_only})
    return packages, timings


def _covariance(x: torch.Tensor, ridge_rel: float, shrink: float):
    """Centered covariance with diagonal shrinkage and a positive ridge."""
    if x.shape[0] < 2:
        raise ValueError("need at least two calibration trajectories")
    c = x - x.mean(0, keepdim=True)
    full = c.T @ c / (x.shape[0] - 1)
    diagonal = torch.diag(torch.diag(full))
    out = (1.0 - shrink) * full + shrink * diagonal
    scale = float(torch.diag(out).mean().clamp(min=1e-30))
    return out + ridge_rel * scale * torch.eye(out.shape[0], dtype=out.dtype, device=out.device)


def _fit_calibration(packages, innovation_subspace, r_shrink, r_ridge_rel):
    xi = torch.stack([p["xi"] for p in packages])
    xi_window = torch.stack([p["xi_window"] for p in packages])
    eta = torch.stack([p["eta"] for p in packages])
    errors = torch.stack([(p["truth"] - p["raw"]).reshape(-1) for p in packages])
    xiw = xi_window - xi_window.mean(0, keepdim=True)
    # Q = Q_time kron I_space.  This is the largest innovation covariance
    # identifiable from finite trajectories without fitting a dense D x D
    # matrix; unlike qI it retains calibrated temporal innovation correlation.
    q_time = torch.einsum("mtn,msn->ts", xiw, xiw) / max((len(packages) - 1) * xiw.shape[-1], 1)
    q_scale = float(torch.diag(q_time).mean().clamp(min=1e-30))
    q_time = q_time + r_ridge_rel * q_scale * torch.eye(q_time.shape[0], dtype=q_time.dtype, device=q_time.device)
    alpha_iso = float((errors - errors.mean(0, keepdim=True)).square().mean().clamp(min=1e-30))
    # The empirical joint covariance is fitted only in fixed Fourier
    # innovation coordinates u=U^T xi.  Its number of cross parameters is
    # p*m, rather than D*m, and U is fixed before looking at held-out data.
    u = torch.stack([p["u"] for p in packages])
    uc, ec = u - u.mean(0, keepdim=True), eta - eta.mean(0, keepdim=True)
    u_cols = uc.T.contiguous()
    e_rows = ec.contiguous()
    q_u_emp = u_cols @ u_cols.T / (len(packages) - 1)
    s_u_emp = u_cols @ e_rows / (len(packages) - 1)
    r_emp = ec.T @ ec / (len(packages) - 1)
    # The empirical joint covariance is PSD but R_emp is rank deficient when
    # feature dimension exceeds trajectory count.  The same small positive
    # ridge used by the target makes each observation covariance solvable.
    r_emp_scale = float(torch.diag(r_emp).mean().clamp(min=1e-30))
    r_emp = r_emp + r_ridge_rel * r_emp_scale * torch.eye(r_emp.shape[0], dtype=r_emp.dtype, device=r_emp.device)
    r_target = _covariance(eta, r_ridge_rel, r_shrink)
    # We assume a conditionally unbiased FM innovation, E[xi]=0.  The coarse
    # residual approximation can have a nonzero feature-space intercept,
    # E[eta].  Using mean(z) here is wrong because G varies by trajectory;
    # use the discrepancy intercept explicitly in every local observation.
    basis = innovation_subspace.basis
    q_u0 = basis.T @ _q_apply(q_time, basis, innovation_subspace.T, innovation_subspace.N)
    q_u0 = 0.5 * (q_u0 + q_u0.T)
    return {"Q_time": q_time, "alpha_iso": alpha_iso, "basis": basis,
            "Q_u0": q_u0, "Q_u_emp": q_u_emp, "S_u_emp": s_u_emp,
            "R_emp": r_emp, "R_target": r_target,
            "n_calibration": len(packages), "eta_mean": eta.mean(0)}


def _q_apply(q_time, x, t, n):
    """Apply Q_time kron I_space to a D-vector or D-by-feature matrix."""
    tail = x.shape[1:]
    y = x.reshape(t, n, *tail)
    return torch.einsum("ts,sn...->tn...", q_time, y).reshape_as(x)


def _joint_terms(package, fitted, rho):
    """Return QG^T, S, R from a PSD joint-covariance mixture.

    Write xi=Uu+xi_perp, with U the fixed divergence-free Fourier basis.  The
    complement keeps Q_0=Q_time kron I.  We mix the joint covariance of
    (u,eta), rather than fitting an unidentifiable D-by-feature cross term:
    Sigma_rho=(1-rho) diag(Q_u0,R_0)+rho Cov_cal[(u,eta)].  This is PSD for
    every rho in [0,1]; lifting through U preserves that property.
    """
    t, n = package["prior"].T, package["prior"].N
    gt = package["gt"]
    q0gt = _q_apply(fitted["Q_time"], gt, t, n)
    if rho == 0.0:
        return q0gt, torch.zeros_like(gt), fitted["R_target"]
    basis, q_u0 = fitted["basis"], fitted["Q_u0"]
    u_gt = basis.T @ gt
    # QG^T = Q_0G^T + U (Q_u,rho-Q_u0) U^T G^T.
    q_u = (1.0 - rho) * q_u0 + rho * fitted["Q_u_emp"]
    qgt = q0gt + basis @ ((q_u - q_u0) @ u_gt)
    s = basis @ (rho * fitted["S_u_emp"])
    r = (1.0 - rho) * fitted["R_target"] + rho * fitted["R_emp"]
    return qgt, s, r


def _stable_solve(v, rhs):
    v = 0.5 * (v + v.T)
    scale = float(torch.diag(v).mean().clamp(min=1e-30))
    for multiplier in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return torch.linalg.solve(v + multiplier * scale * torch.eye(v.shape[0], dtype=v.dtype, device=v.device), rhs)
        except torch.linalg.LinAlgError:
            continue
    raise RuntimeError("residual-feature covariance is not numerically positive definite")


def _information_metrics(g, qgt, r):
    """Residual signal-to-discrepancy diagnostics for a local observation."""
    signal = g @ qgt
    signal = 0.5 * (signal + signal.T)
    trace_ratio = float(torch.diag(signal).sum() / torch.diag(r).sum().clamp(min=1e-30))
    chol = torch.linalg.cholesky(0.5 * (r + r.T))
    white = torch.linalg.solve_triangular(chol, signal, upper=False)
    white = torch.linalg.solve_triangular(chol, white.T, upper=False).T
    return trace_ratio, float(torch.linalg.eigvalsh(0.5 * (white + white.T))[-1])


def _correction(package, fitted, method, rho, gain):
    if method == "raw":
        return package["raw"], {"rho_effective": 0.0, "rho_cap": 1.0,
                                 "signal_to_discrepancy": 0.0, "max_information_eigenvalue": 0.0}
    z = package["z"] - fitted["eta_mean"]
    at, gt = package["at"], package["gt"]
    if method == "isotropic_s0":
        alpha = fitted["alpha_iso"]
        r = fitted["R_target"]
        cross = alpha * at
        v = alpha * (at.T @ at) + r
        delta = cross @ _stable_solve(v, z)
        signal = alpha * (at.T @ at)
        ratio, maximum = _information_metrics(torch.eye(signal.shape[0], dtype=signal.dtype, device=signal.device), signal, r)
        return package["raw"] + gain * delta.reshape_as(package["raw"]), {"rho_effective": 0.0, "rho_cap": 1.0,
                                                                              "signal_to_discrepancy": ratio, "max_information_eigenvalue": maximum}
    if method == "block_s0":
        rho = 0.0
    qgt, s, r = _joint_terms(package, fitted, rho)
    g = gt.T
    cross = qgt + s
    v = g @ qgt + g @ s + s.T @ gt + r
    xi_hat = cross @ _stable_solve(v, z)
    delta = _solve_B(package["prior"], xi_hat.reshape(package["prior"].T, package["prior"].N))
    ratio, maximum = _information_metrics(g, qgt, r)
    return package["raw"] + gain * delta, {"rho_effective": float(rho), "rho_cap": 1.0,
                                             "signal_to_discrepancy": ratio, "max_information_eigenvalue": maximum}


def _evaluate(packages, fitted, method, rho, gain):
    rows, infos = [], []
    correction_seconds = 0.0
    for p in packages:
        if method == "rk3_flow_only":
            if p["flow_only"] is None:
                raise ValueError("rk3_flow_only requires joint_fixed_rk3_flow_defect")
            corrected = p["flow_only"]
            info = {"rho_effective": 0.0, "rho_cap": 1.0,
                    "signal_to_discrepancy": float("nan"), "max_information_eigenvalue": float("nan")}
        else:
            started = time.perf_counter()
            corrected, info = _correction(p, fitted, method, rho, gain)
            correction_seconds += time.perf_counter() - started
        error, delta = p["truth"] - p["raw"], corrected - p["raw"]
        rows.append((float((corrected - p["truth"]).square().mean().sqrt()),
                     float((p["raw"] - p["truth"]).square().mean().sqrt()),
                     float(delta.square().mean().sqrt()),
                     float((delta * error).sum() / (delta.norm() * error.norm()).clamp(min=1e-30))))
        infos.append(info)
    a = np.asarray(rows)
    return {"rmse": float(a[:, 0].mean()), "raw_rmse": float(a[:, 1].mean()),
            "correction_rms": float(a[:, 2].mean()), "correction_error_cosine": float(a[:, 3].mean()),
            "rho_effective": float(np.mean([x["rho_effective"] for x in infos])),
            "rho_cap": float(np.mean([x["rho_cap"] for x in infos])),
            "signal_to_discrepancy": float(np.mean([x["signal_to_discrepancy"] for x in infos])),
            "max_information_eigenvalue": float(np.mean([x["max_information_eigenvalue"] for x in infos])),
            "correction_seconds": correction_seconds}


def _blend_evaluate(packages, blend):
    """Calibration-selected convex FM/RK3 baseline, not an HILP correction."""
    if any(p["flow_only"] is None for p in packages):
        raise ValueError("rk3 blend requires joint_fixed_rk3_flow_defect")
    rows = []
    for p in packages:
        prediction = (1.0 - blend) * p["raw"] + blend * p["flow_only"]
        rows.append(float((prediction - p["truth"]).square().mean().sqrt()))
    return {"rmse": float(np.mean(rows)), "blend": float(blend)}


def _correction_subspace(package, fitted, method, rho):
    """Columns spanning all linear corrections reachable from the observation.

    This diagnostic intentionally uses no truth.  Truth enters only afterwards
    to project the held-out forecast error onto this fixed local subspace.
    """
    if method == "isotropic_s0":
        return fitted["alpha_iso"] * package["at"]
    if method == "block_s0":
        rho = 0.0
    qgt, s, _ = _joint_terms(package, fitted, rho)
    cross = qgt + s
    columns = [_solve_B(package["prior"], cross[:, j]).reshape(-1)
               for j in range(cross.shape[1])]
    return torch.stack(columns, dim=1)


def _reachable_error(packages, fitted, method, rho):
    """Truth-only diagnostic fraction of error in the linear correction span."""
    fractions, ranks = [], []
    for p in packages:
        subspace = _correction_subspace(p, fitted, method, rho)
        q, r = torch.linalg.qr(subspace, mode="reduced")
        diagonal = torch.abs(torch.diag(r))
        threshold = float(diagonal.max().clamp(min=1e-30)) * 1e-8
        rank = int((diagonal > threshold).sum())
        error = (p["truth"] - p["raw"]).reshape(-1)
        if rank:
            projected = q[:, :rank] @ (q[:, :rank].T @ error)
            fractions.append(float(projected.square().sum() / error.square().sum().clamp(min=1e-30)))
        else:
            fractions.append(0.0)
        ranks.append(rank)
    return {"mean_fraction": float(np.mean(fractions)),
            "median_fraction": float(np.median(fractions)),
            "mean_rank": float(np.mean(ranks))}


def _merge_timings(*timings):
    output = {}
    for timing in timings:
        for key, value in timing.items():
            output[key] = output.get(key, 0.0) + float(value)
    return output


def _feature_gate(packages):
    truth, raw = [], []
    for p in packages:
        truth_feature, raw_feature = p["feature"](p["truth"]), p["feature"](p["raw"])
        if not torch.isfinite(truth_feature).all() or not torch.isfinite(raw_feature).all():
            raise RuntimeError(
                "non-finite truth/raw physical feature during the calibration gate. "
                "The chosen fixed RK3 observation is numerically unstable on at "
                "least one calibration state; increase its fixed substep budget.")
        truth.append(float(truth_feature.norm()))
        raw.append(float(raw_feature.norm()))
    return float(np.median(truth)), float(np.median(raw)), float(np.mean(np.asarray(truth) < np.asarray(raw)))


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=32)
    ap.add_argument("--n-val-traj", type=int, default=16)
    ap.add_argument("--n-test-traj", type=int, default=32)
    ap.add_argument("--feature-modes", type=int, default=4)
    ap.add_argument("--observation", choices=["midpoint_fourier", "fixed_rk3_flow_defect", "joint_fixed_rk3_flow_defect"],
                    default="midpoint_fourier")
    ap.add_argument("--flow-defect-substeps", type=int, default=32,
                    help="fixed SSP-RK3 stages for the imperfect flow-defect observation")
    ap.add_argument("--r-shrink", type=float, default=0.5)
    ap.add_argument("--r-ridge-rel", type=float, default=1e-3)
    ap.add_argument("--cross-shrink-grid", nargs="+", type=float,
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--gain-grid", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--blend-grid", nargs="+", type=float,
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                    help="validation-only weights for the FM/RK3 control")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.steps < 2:
        raise SystemExit("joint discrepancy HILP requires at least two forecast steps")
    if not 0 <= args.r_shrink <= 1 or args.r_ridge_rel < 0:
        raise SystemExit("R shrinkage must lie in [0,1] and its ridge must be nonnegative")
    if any(not 0 <= rho <= 1 for rho in args.cross_shrink_grid):
        raise SystemExit("cross-shrink grid values must lie in [0,1]")
    if any(not 0 <= blend <= 1 for blend in args.blend_grid):
        raise SystemExit("blend grid values must lie in [0,1]")
    if args.flow_defect_substeps < 1:
        raise SystemExit("flow-defect-substeps must be positive")
    set_seed(args.seed)
    args.fm, args.fm_channels = "poseidon", "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    modes = _modes(spec.n, args.feature_modes)
    total = args.n_cal_traj + args.n_val_traj + args.n_test_traj
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj, args.steps, args.lead_steps, 0)
    val = load_poseidon_trajectories(args.fm_data_path, fm, args.n_val_traj, args.steps, args.lead_steps, args.n_cal_traj)
    test = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj, args.steps, args.lead_steps,
                                      args.n_cal_traj + args.n_val_traj)

    print_header(f"S4 joint innovation--residual discrepancy HILP: {fm.info.name}, horizon={args.steps}")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")
    print(f"  residual representation: {len(modes)} fixed Fourier modes x real/imag x {args.steps} times = {2 * len(modes) * args.steps} features")
    if args.observation == "midpoint_fourier":
        print(f"  observation=midpoint residual; modes={modes}; physical interval={spec.dt_out:g}")
    elif args.observation == "fixed_rk3_flow_defect":
        print(f"  observation=fixed RK3 flow defect with {args.flow_defect_substeps} stages; modes={modes}; physical interval={spec.dt_out:g}")
    else:
        print(f"  observation=joint fixed RK3 flow defect with {args.flow_defect_substeps} stages; modes={modes}; physical interval={spec.dt_out:g}")
    print("  no flow endpoint is substituted into the forecast and no future truth is read at inference")

    print("\n  building calibration packages")
    innovation_subspace = StructuredInnovationSubspace(spec, args.steps, modes, fm.device)
    cal_p, cal_timing = _package(fm, cal, spec, modes, args.steps, innovation_subspace, args)
    gate_truth, gate_raw, wins = _feature_gate(cal_p)
    fitted = _fit_calibration(cal_p, innovation_subspace, args.r_shrink, args.r_ridge_rel)
    print(f"  coarse feature gate (calibration): truth={gate_truth:.4g}, raw={gate_raw:.4g}, truth/raw={gate_truth/max(gate_raw, 1e-30):.3f}, truth wins={100*wins:.1f}%")
    print(f"  fitted Q=Q_time kron I: diag={torch.diag(fitted['Q_time']).detach().cpu().numpy().round(7).tolist()}; isotropic alpha={fitted['alpha_iso']:.4g}; R shrink={args.r_shrink:g}; eta-intercept RMS={fitted['eta_mean'].square().mean().sqrt():.4g}")
    print(f"  structured cross covariance: {innovation_subspace.dimension} divergence-free innovation coordinates x {2 * len(modes) * args.steps} residual features")

    print("\n  building validation packages")
    val_p, val_timing = _package(fm, val, spec, modes, args.steps, innovation_subspace, args)
    selected, curves = {}, {}
    for method in ("isotropic_s0", "block_s0", "block_joint"):
        candidates = []
        rhos = args.cross_shrink_grid if method == "block_joint" else [0.0]
        print(f"\n  selecting gain for {method} on validation")
        for rho in rhos:
            for gain in args.gain_grid:
                out = _evaluate(val_p, fitted, method, rho, gain)
                candidates.append({"rho_requested": rho, "gain": gain, **out})
                print(f"    rho={rho:g}, gain={gain:g}: RMSE={out['rmse']:.6g}, corr-RMS={out['correction_rms']:.4g}, corr/error cosine={out['correction_error_cosine']:+.4f}, signal/R={out['signal_to_discrepancy']:.3g}, max-info-eig={out['max_information_eigenvalue']:.3g}")
        selected[method] = min(candidates, key=lambda d: d["rmse"])
        curves[method] = candidates
        best = selected[method]
        print(f"    selected rho={best['rho_requested']:g}, gain={best['gain']:g}")

    blend_choice = None
    if args.observation == "joint_fixed_rk3_flow_defect":
        blend_curve = []
        print("\n  selecting FM/RK3 blend on validation (non-HILP control)")
        for blend in args.blend_grid:
            out = _blend_evaluate(val_p, blend)
            blend_curve.append(out)
            print(f"    RK3 weight={blend:g}: RMSE={out['rmse']:.6g}")
        blend_choice = min(blend_curve, key=lambda d: d["rmse"])
        print(f"    selected RK3 weight={blend_choice['blend']:g}")

    print("\n  building held-out packages")
    test_p, test_timing = _package(fm, test, spec, modes, args.steps, innovation_subspace, args)
    raw = _evaluate(test_p, fitted, "raw", 0.0, 0.0)
    table = Table("method", "RMSE", "gain%", "corr RMS", "corr/error cosine", "reachable error%", "requested rho", "effective rho")
    table.add("raw", raw["rmse"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    held = {"raw": raw}
    reachability = {}
    for method in ("isotropic_s0", "block_s0", "block_joint"):
        choice = selected[method]
        out = _evaluate(test_p, fitted, method, choice["rho_requested"], choice["gain"])
        reach = _reachable_error(test_p, fitted, method, choice["rho_requested"])
        reachability[method] = reach
        held[method] = out
        table.add(method, out["rmse"], 100 * (1 - out["rmse"] / raw["rmse"]),
                  out["correction_rms"], out["correction_error_cosine"],
                  100 * reach["mean_fraction"], choice["rho_requested"], out["rho_effective"])
    if args.observation == "joint_fixed_rk3_flow_defect":
        out = _evaluate(test_p, fitted, "rk3_flow_only", 0.0, 0.0)
        held["rk3_flow_only"] = out
        table.add("rk3_flow_only", out["rmse"], 100 * (1 - out["rmse"] / raw["rmse"]),
                  0.0, 0.0, float("nan"), 0.0, 0.0)
        blend = _blend_evaluate(test_p, blend_choice["blend"])
        held["fm_rk3_blend"] = blend
        table.add(f"fm_rk3_blend ({blend_choice['blend']:g})", blend["rmse"],
                  100 * (1 - blend["rmse"] / raw["rmse"]), 0.0, 0.0,
                  float("nan"), 0.0, 0.0)
    print("\n  held-out joint correction evaluation")
    print(table)

    runtime = {"calibration_packages": cal_timing, "validation_packages": val_timing,
               "held_out_packages": test_timing,
               "held_out_selected_correction_seconds": {
                   method: held[method]["correction_seconds"]
                   for method in ("isotropic_s0", "block_s0", "block_joint")}}
    print("\n  held-out runtime (seconds; package construction excludes validation sweeps)")
    runtime_table = Table("component", "seconds")
    for key, value in test_timing.items():
        runtime_table.add(key.replace("_seconds", ""), value)
    for method, value in runtime["held_out_selected_correction_seconds"].items():
        runtime_table.add(f"{method} correction", value)
    print(runtime_table)
    print("  reachable-error diagnostic (truth is used only after held-out correction):")
    for method, result in reachability.items():
        print(f"    {method}: mean={100 * result['mean_fraction']:.2f}%, median={100 * result['median_fraction']:.2f}%, mean rank={result['mean_rank']:.1f}")

    payload = {"stage": "s4_joint_innovation_residual_discrepancy", "metadata": fm_metadata(args, fm, spec),
               "split_total": total, "feature_modes": modes, "feature_dimension": 2 * len(modes) * args.steps,
               "observation": {"name": args.observation, "flow_defect_substeps": args.flow_defect_substeps},
               "calibration": {"Q_time": fitted["Q_time"].detach().cpu().numpy().tolist(), "alpha_isotropic": fitted["alpha_iso"],
                               "r_shrink": args.r_shrink, "r_ridge_rel": args.r_ridge_rel,
                               "joint_covariance": "Q_time kron I on the complement; (1-rho) diag(Q_u0, R_target) + rho Cov_calibration[(U^T xi, eta)] in fixed divergence-free Fourier coordinates",
                               "structured_innovation_dimension": innovation_subspace.dimension,
                               "feature_gate": {"truth": gate_truth, "raw": gate_raw, "wins": wins}},
               "validation": {"curves": curves, "selected": selected,
                              "fm_rk3_blend_curve": blend_curve if blend_choice is not None else None,
                              "selected_fm_rk3_blend": blend_choice},
               "held_out": held, "reachable_error": reachability, "runtime_seconds": runtime,
               "validity": {"formula": "z=-r(mu)-E[eta]; delta=B^-1(QG^T+S)(GQG^T+GS+S^TG^T+R)^-1 z",
                            "future_truth_used_at_inference": False,
                            "s0_ablation": "block_s0 sets S=0",
                            "cross_term": "S is estimated on calibration trajectories only",
                            "reachable_error": "truth-only diagnostic projection of held-out FM error onto range[B^-1(QG^T+S)]",
                            "rk3_controls": "RK3-only and FM/RK3 blend are reported as solver-strength controls, never used by HILP"}}
    path = save_json(payload, results_path_scale("s4_joint_discrepancy", fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
