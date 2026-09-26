#!/usr/bin/env python3
"""Revised S4 HILP: innovation-precision physics correction for frozen Poseidon.

This implements the primary construction in docs/hilp_revised_math.tex.
For a raw autoregressive window mu=(mu_1,...,mu_T), define errors relative to
that window and the linearized innovation operator

    (B delta)_1 = delta_1,
    (B delta)_{t+1} = delta_{t+1} - J_t delta_t.

With Q_t=q I, the reference precision is P^{-1}=B^T Q^{-1} B.  No dense
Jacobian, dense covariance, sampled covariance, or output-space inverse-J^T J
is formed.  The physics term is a separately selected differentiable residual.

This script deliberately keeps the legacy S4 implementations untouched.
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
from experiments.scale.s4_fm_physics_correction import (
    _calibration_physics_scales, _truth_floor_audit,
    calibrate_midpoint_transport_bias, make_physics_energy)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.jacobian import jacobian_ops
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("isotropic", "block_innovation")


def _finite_mean(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _raw_window_and_ops(fm, x0, steps):
    """Raw rollout and the local maps J_t used by the innovation precision."""
    raw, ops = [], []
    c = x0.to(fm.device, fm.dtype)
    for t in range(steps):
        nxt = fm.predict(c).detach().reshape(-1)
        raw.append(nxt.double())
        if t + 1 < steps:
            ops.append(jacobian_ops(fm.flat_fn(), nxt))
        c = nxt
    return torch.stack(raw), ops


class InnovationCovariance:
    """Low-rank-plus-isotropic covariance with matrix-free precision actions.

    The covariance is trace-matched to ``q I``.  ``rho=0`` is exactly the
    existing scalar-innovation model; ``rho=1`` uses the calibration-estimated
    innovation spectrum plus its isotropic tail.
    """
    def __init__(self, q, vectors=None, eigenvalues=None, rho=0.0):
        self.q = float(q)
        self.vectors = vectors
        self.eigenvalues = eigenvalues
        self.rho = float(rho)
        if self.q <= 0 or not 0.0 <= self.rho <= 1.0:
            raise ValueError("q must be positive and rho must lie in [0, 1]")
        if vectors is None or eigenvalues is None or self.rho == 0.0:
            self.vectors, self.eigenvalues = None, None
            self.base = self.q
            self.coefficients = None
            self.rank = 0
            return
        self.rank = int(eigenvalues.numel())
        n = vectors.shape[0]
        # A rank-r empirical covariance is completed by a scalar tail so its
        # trace equals N*q.  This prevents a larger rank from simply granting
        # the structured model more total prior variance than qI.
        tail = max((n * self.q - float(eigenvalues.sum())) / max(n - self.rank, 1),
                   self.q * 1e-8)
        eig = torch.clamp(eigenvalues, min=tail)
        self.base = (1.0 - self.rho) * self.q + self.rho * tail
        self.coefficients = self.rho * (eig - tail)

    def precision(self, z):
        if self.rank == 0:
            return z / self.base
        flat = z.reshape(-1, z.shape[-1])
        u = self.vectors.to(flat.device, flat.dtype)
        proj = flat @ u
        factors = self.coefficients.to(flat.device, flat.dtype) / (
            self.base * (self.base + self.coefficients.to(flat.device, flat.dtype)))
        return (flat / self.base - (proj * factors) @ u.T).reshape_as(z)

    def logdet(self, n):
        if self.rank == 0:
            return n * math.log(self.base)
        return (n - self.rank) * math.log(self.base) + float(torch.log(
            self.base + self.coefficients).sum())


class InnovationPrior:
    """Matrix-free N(mu, alpha q B^{-1} B^{-T}) quadratic prior.

    Its value and gradient are exact for the frozen local linearization.  The
    gradient is B^T B delta/(alpha q), evaluated with JVP/VJP actions.  This is
    the precision action required by the revised specification.
    """
    def __init__(self, mean, ops, q, alpha=1.0, label="block_innovation", covariance=None):
        self.mean = mean.detach().double()
        self.ops = list(ops)
        self.T, self.N = self.mean.shape
        self.D = self.T * self.N
        self.q = float(q)
        self.covariance = covariance or InnovationCovariance(self.q)
        self.alpha = float(alpha)
        self.label = label
        if len(self.ops) != self.T - 1:
            raise ValueError("need one local Jacobian for each transition after step one")
        if self.q <= 0 or self.alpha <= 0:
            raise ValueError("innovation variance and alpha must be positive")

    def rescaled(self, alpha):
        return InnovationPrior(self.mean, self.ops, self.q, alpha=alpha, label=self.label,
                               covariance=self.covariance)

    def apply_B(self, delta):
        z = delta.reshape(self.T, self.N).double()
        out = [z[0]]
        for t, op in enumerate(self.ops):
            jz = op.matvec(z[t].to(op.dtype)).double()
            out.append(z[t + 1] - jz)
        return torch.stack(out)

    def apply_BT(self, w):
        z = w.reshape(self.T, self.N).double()
        out = [z[t].clone() for t in range(self.T)]
        for t, op in enumerate(self.ops):
            out[t] -= op.rmatvec(z[t + 1].to(op.dtype)).double()
        return torch.stack(out)

    def value_grad(self, candidate):
        delta = candidate.detach().double().reshape(self.T, self.N) - self.mean
        bdelta = self.apply_B(delta)
        precision_bdelta = self.covariance.precision(bdelta) / self.alpha
        value = 0.5 * float((bdelta * precision_bdelta).sum())
        grad = self.apply_BT(precision_bdelta)
        return value, grad

    def innovation_mse(self, truth):
        error = truth.double().reshape(self.T, self.N) - self.mean
        return float(self.apply_B(error).square().mean())

    def normalized_innovation_mse(self, truth):
        error = truth.double().reshape(self.T, self.N) - self.mean
        z = self.apply_B(error)
        return float((z * self.covariance.precision(z)).mean())


class IsotropicPrior:
    """Trace-free reference with one independently fitted scalar scale."""
    def __init__(self, mean, alpha=1.0):
        self.mean = mean.detach().double()
        self.T, self.N = self.mean.shape
        self.D = self.T * self.N
        self.alpha = float(alpha)
        self.label = "isotropic"

    def rescaled(self, alpha):
        return IsotropicPrior(self.mean, alpha=alpha)

    def value_grad(self, candidate):
        delta = candidate.detach().double().reshape(self.T, self.N) - self.mean
        value = 0.5 * float(delta.square().sum() / self.alpha)
        return value, delta / self.alpha


class WindowResidual:
    """Sum of fixed-conditioning one-step physical residuals over a raw window.

    Each residual E_t(z_t; mu_{t-1}) is evaluated using information available at
    inference.  Its previous state is fixed to the raw FM path, so the prior
    supplies the cross-time coupling and the physical term remains a genuine
    residual evaluation rather than an inner PDE solve.
    """
    def __init__(self, spec, x0, raw, channels, args, scales, device):
        previous = x0.to(device, torch.float64).reshape(-1)
        self.energies = []
        for t in range(raw.shape[0]):
            self.energies.append(make_physics_energy(spec, previous, channels, args,
                                                     device,
                                                     vorticity_scale=scales["vorticity_scale"],
                                                     divergence_scale=scales["divergence_scale"],
                                                     transport_bias=scales.get("transport_bias")))
            previous = raw[t].detach()
        self.T, self.N = raw.shape

    def value(self, candidate):
        x = candidate.reshape(self.T, self.N)
        # Existing energy classes use per-gridpoint means.  Multiply by N so
        # lambda weights the squared discretized residual norm in Eq. (9).
        return sum(e.energy(x[t:t + 1]).sum() * self.N for t, e in enumerate(self.energies))

    def rmses(self, candidate):
        x = candidate.detach().reshape(self.T, self.N)
        return [float(e.rms_residual(x[t:t + 1])[0]) for t, e in enumerate(self.energies)]


def _map(prior, residual, lam, args):
    """Minimize the revised objective with its exact matrix-free prior gradient."""
    x = prior.mean.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=args.map_lr, max_iter=args.map_steps,
                            history_size=10, tolerance_grad=args.map_grad_tol,
                            tolerance_change=1e-11, line_search_fn="strong_wolfe")
    calls = [0]
    initial_prior, _ = prior.value_grad(x)
    initial_phys = float(residual.value(x).detach())

    def closure():
        opt.zero_grad(set_to_none=True)
        prior_value, prior_grad = prior.value_grad(x)
        physical = residual.value(x)
        # The scalar prior term is detached because its exact gradient is
        # supplied by B^T B. This avoids differentiating through JVP operators.
        loss = lam * physical + x.new_tensor(prior_value)
        loss.backward()
        with torch.no_grad():
            x.grad.add_(prior_grad)
        calls[0] += 1
        return loss

    opt.step(closure)
    final_prior, final_grad = prior.value_grad(x)
    final_phys = float(residual.value(x).detach())
    grad_rms = float((final_grad + lam * torch.autograd.grad(
        residual.value(x), x, retain_graph=False)[0]).square().mean().sqrt())
    return x.detach(), {"function_evals": calls[0], "prior_initial": initial_prior,
                         "prior_final": final_prior, "physics_initial": initial_phys,
                         "physics_final": final_phys, "objective_initial": initial_prior + lam * initial_phys,
                         "objective_final": final_prior + lam * final_phys,
                         "gradient_rms_final": grad_rms,
                         "finite": bool(torch.isfinite(x).all())}


def _make_package(fm, trajectories, i, raw, ops, spec, sigma2, args, scales):
    """Construct one package once its raw window and local Jacobians exist."""
    prior = InnovationPrior(raw, ops, sigma2)
    residual = WindowResidual(spec, trajectories[i, 0], raw, fm.state_shape[0], args,
                              scales, fm.device)
    truth = trajectories[i, 1:args.steps + 1].to(fm.device, torch.float64)
    # Keep public method names as dictionary keys.  In particular, the
    # evaluation loop indexes this package by ``block_innovation`` rather
    # than the internal shorthand ``block``.
    return {"raw": raw, "block_innovation": prior, "isotropic": IsotropicPrior(raw),
            "residual": residual, "truth": truth}


def _package(fm, trajectories, spec, sigma2, args, scales):
    """Build packages; only raw FM forwards may be state-batched.

    Jacobian operators stay per state, preserving the exact same local
    linearization and MAP objective as the serial path.
    """
    out, n = [], trajectories.shape[0]
    batch = int(args.package_batch)
    batch_safe = bool(getattr(fm.module, "_hipp_batch_safe", False))
    if batch <= 1 or not batch_safe:
        if batch > 1 and not batch_safe:
            print("    package batching unavailable for this adapter; using serial FM forwards")
        for i in range(n):
            print(f"    preparing trajectory {i + 1}/{n}", flush=True)
            raw, ops = _raw_window_and_ops(fm, trajectories[i, 0], args.steps)
            out.append(_make_package(fm, trajectories, i, raw, ops, spec, sigma2, args, scales))
        return out

    for first in range(0, n, batch):
        last = min(first + batch, n)
        print(f"    preparing trajectories {first + 1}-{last}/{n} (batched FM forward)", flush=True)
        current = trajectories[first:last, 0].to(fm.device, fm.dtype).reshape(last - first, -1)
        windows = []
        for _ in range(args.steps):
            current = fm.predict(current).detach().reshape(last - first, -1)
            windows.append(current.double())
        raw_windows = torch.stack(windows, dim=1)
        if args.verify_package_batch and first == 0:
            serial = torch.stack([_raw_window_and_ops(fm, trajectories[i, 0], args.steps)[0]
                                  for i in range(first, last)])
            max_abs = float((raw_windows - serial).abs().max())
            torch.testing.assert_close(raw_windows, serial, rtol=5e-5, atol=5e-6)
            print(f"    package-batch equivalence passed: raw rollout max abs={max_abs:.2e}")
        for local, i in enumerate(range(first, last)):
            raw = raw_windows[local]
            # The stored mean is float64 for stable prior/physics arithmetic,
            # while the frozen FM and its JVP/VJP trace retain their native
            # float32 parameter dtype.
            ops = [jacobian_ops(fm.flat_fn(), raw[t].to(fm.dtype))
                   for t in range(args.steps - 1)]
            out.append(_make_package(fm, trajectories, i, raw, ops, spec, sigma2, args, scales))
    return out


def _fit_alphas(packages):
    block_q, iso_q = [], []
    for p in packages:
        error = p["truth"] - p["raw"]
        block_q.append(p["block_innovation"].normalized_innovation_mse(p["truth"]))
        iso_q.append(float(error.square().mean()))
    return {"block_innovation": max(float(np.mean(block_q)), 1e-30),
            "isotropic": max(float(np.mean(iso_q)), 1e-30)}, {
                "block_innovation": {"n": len(block_q), "mean_normalized_innovation_mse": float(np.mean(block_q)),
                                       "median_normalized_innovation_mse": float(np.median(block_q))},
                "isotropic": {"n": len(iso_q), "mean_error_mse": float(np.mean(iso_q)),
                              "median_error_mse": float(np.median(iso_q))}}


def _innovation_samples(packages):
    """Calibration innovations B(x_truth-mu), with no future test information."""
    samples = []
    for p in packages:
        prior = p["block_innovation"]
        error = p["truth"] - p["raw"]
        samples.append(prior.apply_B(error).detach())
    return torch.cat(samples, dim=0)


def _structured_covariance(packages, q, rank, rho):
    """Fit an uncentered second-moment shape and trace-match it to qI.

    The prior remains centered on the FM rollout.  Therefore this deliberately
    estimates E[eta eta^T], rather than subtracting a calibration mean and
    silently introducing a learned mean correction at lambda=0.
    """
    samples = _innovation_samples(packages)
    m, n = samples.shape
    rank = min(int(rank), m, n)
    if rank < 1 or rho == 0.0:
        return InnovationCovariance(q)
    # SVD of M x N innovations only; no N x N covariance is formed.
    _, s, vh = torch.linalg.svd(samples / math.sqrt(m), full_matrices=False)
    eig = s[:rank].square()
    # Trace-match empirical shape to the scalar calibration scale q.
    trace = float(samples.square().mean())
    if not math.isfinite(trace) or trace <= 0:
        raise RuntimeError("non-finite calibration innovation second moment")
    eig = eig * (q / trace)
    return InnovationCovariance(q, vh[:rank].T.detach(), eig.detach(), rho=rho)


def _set_innovation_covariance(packages, covariance):
    for p in packages:
        old = p["block_innovation"]
        p["block_innovation"] = InnovationPrior(old.mean, old.ops, old.q,
                                                  label=old.label, covariance=covariance)


def _innovation_validation_nll(packages, covariance):
    """Per-coordinate predictive innovation NLL, used solely to select rho."""
    values = []
    for p in packages:
        old = p["block_innovation"]
        z = old.apply_B(p["truth"] - p["raw"])
        quadratic = float((z * covariance.precision(z)).sum())
        values.append(0.5 * (quadratic + z.shape[0] * covariance.logdet(z.shape[-1])) / z.numel())
    return _finite_mean(values)


def _evaluate(method, alpha, lam, packages, args, diagnostics=False):
    rows, traces = [], []
    started = time.time()
    for p in packages:
        prior = p[method].rescaled(alpha)
        corrected, info = _map(prior, p["residual"], lam, args)
        raw, truth = p["raw"], p["truth"]
        raw_err = raw - truth
        corr_err = corrected - truth
        correction = corrected - raw
        row = {"rmse": float(corr_err.square().mean().sqrt()),
               "raw_rmse": float(raw_err.square().mean().sqrt()),
               "correction_rms": float(correction.square().mean().sqrt()),
               "correction_error_cosine": float((correction.reshape(-1) @ (-raw_err).reshape(-1)) /
                   (correction.norm() * raw_err.norm()).clamp(min=1e-300)),
               "physics_rms_raw": float(np.mean(p["residual"].rmses(raw))),
               "physics_rms_corrected": float(np.mean(p["residual"].rmses(corrected))),
               "innovation_mse_raw": prior.innovation_mse(raw) if method == "block_innovation" else 0.0,
               **info}
        for t in range(args.steps):
            row[f"step{t + 1}_rmse"] = float(corr_err[t].square().mean().sqrt())
        rows.append(row)
        if diagnostics:
            traces.append(row)
    keys = rows[0]
    return {k: _finite_mean([r[k] for r in rows]) for k in keys} | {
        "seconds": time.time() - started, "rows": traces if diagnostics else []}


def _isotropic_lambda_reference(packages, alpha):
    """Report a common direct-lambda scale; it does not alter selection.

    At one isotropic prior standard deviation, the prior-gradient RMS is
    1/sqrt(alpha).  This returns the lambda whose physics-gradient RMS at the
    raw window matches that quantity.  It is only a sweep-centering aid:
    validation still chooses the actual shared lambda.
    """
    references = []
    for p in packages:
        x = p["raw"].detach().clone().requires_grad_(True)
        grad = torch.autograd.grad(p["residual"].value(x), x)[0]
        grad_rms = float(grad.square().mean().sqrt())
        if math.isfinite(grad_rms) and grad_rms > 0:
            references.append(1.0 / (math.sqrt(alpha) * grad_rms))
    return float(np.median(references)) if references else float("nan")


def _first_step_residual_gate(fm, trajectories, spec, args, scales):
    """Comparable truth/raw residual check, both conditioned on observed x_0."""
    truth_rms, raw_rms = [], []
    for i in range(trajectories.shape[0]):
        raw = fm.predict(trajectories[i, 0].to(fm.device, fm.dtype)).double().reshape(1, -1)
        energy = make_physics_energy(spec, trajectories[i, 0], fm.state_shape[0], args,
                                     fm.device, vorticity_scale=scales["vorticity_scale"],
                                     divergence_scale=scales["divergence_scale"],
                                     transport_bias=scales.get("transport_bias"))
        truth = trajectories[i, 1].to(fm.device, torch.float64).reshape(1, -1)
        truth_rms.append(float(energy.rms_residual(truth)[0]))
        raw_rms.append(float(energy.rms_residual(raw)[0]))
    truth, raw = np.asarray(truth_rms), np.asarray(raw_rms)
    return {"truth_median": float(np.median(truth)), "raw_median": float(np.median(raw)),
            "truth_over_raw": float(np.median(truth) / max(np.median(raw), 1e-30)),
            "truth_wins_fraction": float(np.mean(truth < raw))}


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=16)
    ap.add_argument("--n-val-traj", type=int, default=8)
    ap.add_argument("--n-test-traj", type=int, default=16)
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--innovation-covariance", choices=["isotropic", "structured_lowrank"],
                    default="isotropic", help="optional calibration-only Q shape for block innovation")
    ap.add_argument("--innovation-rank", type=int, default=32)
    ap.add_argument("--innovation-shrink-grid", nargs="+", type=float,
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--package-batch", type=int, default=1,
                    help="batch only independent raw FM rollout forwards; Jacobians remain exact per state")
    ap.add_argument("--verify-package-batch", action="store_true",
                    help="compare the first batched raw rollout against the serial path")
    ap.add_argument("--map-steps", type=int, default=30)
    ap.add_argument("--map-lr", type=float, default=0.5)
    ap.add_argument("--map-grad-tol", type=float, default=1e-8)
    ap.add_argument("--physics-substeps", type=int, default=2)
    ap.add_argument("--physics-likelihood", choices=["midpoint_residual", "midpoint", "midpoint_transport_1jvp", "discrete_flow", "fixed_rk3_flow"],
                    default="midpoint_residual")
    ap.add_argument("--transport-jvp-terms", type=int, default=1,
                    help="must be 1: exactly one known-PDE RHS JVP for midpoint transport")
    ap.add_argument("--flow-cfl", type=float, default=0.5)
    ap.add_argument("--fixed-rk3-substeps", type=int, default=64)
    ap.add_argument("--physics-resolution", type=int, default=128)
    ap.add_argument("--divergence-weight", type=float, default=1.0)
    ap.add_argument("--allow-residual-gate-fail", action="store_true",
                    help="run despite calibration truth/raw residual discrimination failure")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.steps < 2 or args.n_val_traj < 1:
        raise SystemExit("revised S4 needs --steps >= 2 and a trajectory-disjoint validation split")
    if args.physics_likelihood == "midpoint":
        args.physics_likelihood = "midpoint_residual"
    if args.physics_likelihood == "midpoint_transport_1jvp":
        if args.physics_substeps != 1:
            raise SystemExit("midpoint_transport_1jvp requires --physics-substeps=1")
        if args.transport_jvp_terms != 1:
            raise SystemExit("midpoint_transport_1jvp is fixed to exactly one PDE-RHS JVP")
    if args.physics_resolution != 128:
        raise SystemExit("this first revised implementation uses physical residuals on the FM grid; set --physics-resolution=128")
    set_seed(args.seed)
    args.fm, args.fm_channels = "poseidon", "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj, args.steps, args.lead_steps, 0)
    val = load_poseidon_trajectories(args.fm_data_path, fm, args.n_val_traj, args.steps, args.lead_steps, args.n_cal_traj)
    test = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj, args.steps, args.lead_steps, args.n_cal_traj + args.n_val_traj)
    print_header(f"Revised S4 innovation-precision HILP: {fm.info.name}, horizon={args.steps}")
    print("  prior: exact matrix-free block innovation precision; no sampled rollout covariance")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")
    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]
    transport_bias, transport_meta = calibrate_midpoint_transport_bias(
        args, cal, spec, fm.state_shape[0], fm.device)
    floor = _truth_floor_audit(cal, spec, fm.state_shape[0], args, fm.device,
                               transport_bias=transport_bias)
    scales = _calibration_physics_scales(args, fm, cal, spec, floor,
                                         transport_bias=transport_bias)
    if transport_bias is not None:
        scales["transport_bias"] = transport_bias
    gate = _first_step_residual_gate(fm, cal, spec, args, scales)
    print(f"  fitted scalar innovation scale q={sigma2:.5e}; physical residual={args.physics_likelihood}")
    if transport_meta["enabled"]:
        print("  midpoint transport: exactly one known-PDE RHS JVP per candidate; "
              f"calibration bias RMS={transport_meta['bias_rms']:.4g}; "
              f"centered truth RMS={transport_meta['centered_truth_rms']:.4g}")
    print("  comparable first-step residual gate (calibration): "
          f"truth={gate['truth_median']:.4g}, raw={gate['raw_median']:.4g}, "
          f"truth/raw={gate['truth_over_raw']:.3f}, truth wins={100*gate['truth_wins_fraction']:.1f}%")
    if gate["truth_wins_fraction"] < 0.5 and not args.allow_residual_gate_fail:
        raise SystemExit("residual gate failed: truth does not beat the raw FM on calibration. "
                         "Use --allow-residual-gate-fail only for a diagnostic run.")

    print("\n  building calibration packages")
    cal_packages = _package(fm, cal, spec, sigma2, args, scales)
    print("\n  building validation packages")
    val_packages = _package(fm, val, spec, sigma2, args, scales)
    structured_meta = {"mode": args.innovation_covariance, "rank": 0, "selected_rho": 0.0}
    if args.innovation_covariance == "structured_lowrank":
        print("\n  selecting calibration-only innovation covariance shrinkage by validation NLL")
        candidates = []
        for rho in args.innovation_shrink_grid:
            covariance = _structured_covariance(cal_packages, sigma2, args.innovation_rank, rho)
            nll = _innovation_validation_nll(val_packages, covariance)
            candidates.append((float(rho), covariance, nll))
            print(f"    rho={rho:g}: validation innovation NLL/dim={nll:.6g}")
        rho, covariance, nll = min(candidates, key=lambda x: x[2])
        _set_innovation_covariance(cal_packages, covariance)
        _set_innovation_covariance(val_packages, covariance)
        structured_meta = {"mode": args.innovation_covariance, "rank": covariance.rank,
                           "selected_rho": rho, "validation_nll_per_dim": nll,
                           "candidates": [{"rho": r, "nll_per_dim": v} for r, _, v in candidates]}
        print(f"    selected rho={rho:g}, rank={covariance.rank} by validation innovation NLL")

    print("\n  fitting independently calibrated prior scales")
    alphas, alpha_meta = _fit_alphas(cal_packages)
    for method in args.methods:
        print(f"    {method:18s} alpha={alphas[method]:.5g}; "
              f"calibration statistic={alpha_meta[method]}")
    lambda_reference = _isotropic_lambda_reference(cal_packages, alphas["isotropic"])
    print(f"    common isotropic one-prior-RMS lambda reference={lambda_reference:.4g} "
          "(sweep guide only; validation selects the direct lambda)")
    selected, curves = {}, {}
    for method in args.methods:
        print(f"\n  selecting direct lambda for {method} on validation")
        cells = []
        for lam in args.lambda_grid:
            out = _evaluate(method, alphas[method], lam, val_packages, args)
            cells.append({"lambda": lam, "rmse": out["rmse"], "seconds": out["seconds"]})
            print(f"    lambda={lam:g}: RMSE={out['rmse']:.6g}", flush=True)
        best = min(cells, key=lambda x: x["rmse"])
        selected[method], curves[method] = best["lambda"], cells
        print(f"    selected lambda={best['lambda']:g}")
    print("\n  building held-out packages")
    test_packages = _package(fm, test, spec, sigma2, args, scales)
    if args.innovation_covariance == "structured_lowrank":
        _set_innovation_covariance(test_packages, covariance)
    raw = _evaluate("isotropic", 1.0, 0.0, test_packages, args)
    table = Table("method", "RMSE", "gain%", *[f"step{i+1} gain%" for i in range(args.steps)],
                  "corr-error cosine", "physics RMS ratio")
    table.add("raw", raw["rmse"], 0.0, *([0.0] * args.steps), 0.0, 1.0)
    held = {"raw": raw}
    for method in args.methods:
        out = _evaluate(method, alphas[method], selected[method], test_packages, args, diagnostics=True)
        held[method] = out
        gains = [100 * (1 - out[f"step{t+1}_rmse"] / raw[f"step{t+1}_rmse"]) for t in range(args.steps)]
        table.add(method, out["rmse"], 100 * (1 - out["rmse"] / raw["rmse"]), *gains,
                  out["correction_error_cosine"], out["physics_rms_corrected"] / max(out["physics_rms_raw"], 1e-30))
    print("\n  held-out evaluation")
    print(table)
    payload = {"stage": "s4_revised_innovation_precision", "metadata": fm_metadata(args, fm, spec),
               "sigma2": sigma2, "physics_scales": scales, "residual_gate_calibration": gate,
               "residual_floor_calibration": floor, "alphas": alphas, "alpha_metadata": alpha_meta,
               "innovation_covariance": structured_meta,
               "lambda_reference_isotropic": lambda_reference,
               "lambda_selection_validation": curves, "selected_lambda": selected, "held_out": held,
               "validity": {"prior": "P^-1=B^T Q^-1 B with Q selected without held-out trajectories", "q_source": "calibration teacher-forced one-step MSE",
                            "lambda": "shared direct residual weight; validation includes lambda=0",
                            "future_truth_used_by_method": False,
                            "physical_term": "fixed-conditioning residual over raw FM window",
                            "legacy_lowrank_output_GN_used": False}}
    path = save_json(payload, results_path_scale("s4_revised", fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
