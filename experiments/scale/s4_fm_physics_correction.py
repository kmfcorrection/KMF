#!/usr/bin/env python3
"""FM S4 -- does HILP post-hoc correction produce better PDE solutions?"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    POSEIDON_NATIVE_RAW_STRIDE, configure_fm_cadence,
    configure_native_poseidon_cadence, fm_metadata,
    fm_result_key, generate_fm_trajectories, load_poseidon_trajectories,
    native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import (CoarseNativeFlowEnergy2D, FMPhysicsEnergy2D,
                                   FixedRK3FlowEnergy2D, MidpointTransportEnergy2D,
                                   NativeFlowEnergy2D)
from hipp.scale.lowrank import LowRankGaussian
from hipp.scale.posterior_scale import (LowRankPhysicsPosterior,
                                         covariance_matvec, force_scale)
from hipp.scale.rollout import fit_sigma2, propagate_step
from hipp.samplers import effective_sample_size, hmc, langevin
from hipp.utils import Table, print_header, save_json, set_seed


def _center(prior, mean):
    return LowRankGaussian(mean.double(), prior.U, prior.d, prior.tau,
                           alpha=prior.alpha, label=prior.label)


def _isotropize(prior, mean, label="isotropic"):
    return LowRankGaussian.isotropic(mean.double(), tau=prior.trace() / prior.N,
                                     label=label)


def _scale_complement(prior, mean, scale):
    """Shrink only covariance outside span(U), preserving retained eigenvalues.

    If lambda_i = alpha * (d_i + tau), replacing tau by rho*tau and
    d_i by d_i + (1-rho)*tau leaves every retained lambda_i unchanged.  A
    strictly positive floor is required for a proper Gaussian posterior.
    """
    rho = float(scale)
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"HILP complement-floor scale must be in (0, 1], got {rho}")
    if prior.k == 0 or rho == 1.0:
        return _center(prior, mean)
    tau = rho * prior.tau
    d = prior.d + (1.0 - rho) * prior.tau
    return LowRankGaussian(mean.double(), prior.U, d, tau,
                           alpha=prior.alpha,
                           label=f"{prior.label}-floor{rho:g}")


def _blend_hilp_isotropic(prior, mean, beta):
    """Convex covariance blend: beta=1 is HILP, beta=0 is isotropic."""
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"covariance blend beta must lie in [0, 1], got {beta}")
    iso_var = prior.trace() / prior.N
    # alpha*(U diag(d) U' + tau I) mixed with iso_var I can be represented
    # exactly as a LowRankGaussian with alpha=1.
    return LowRankGaussian(mean.double(), prior.U, beta * prior.alpha * prior.d,
                           beta * prior.alpha * prior.tau + (1 - beta) * iso_var,
                           alpha=1.0, label=f"hilp-blend{beta:g}")


def _physics_solve(mean, energy, n_steps, lr):
    x = mean.detach().double().clone().requires_grad_(True)
    initial = float(energy.energy(x)[0].detach())
    opt = torch.optim.LBFGS([x], lr=lr, max_iter=n_steps, history_size=10,
                            tolerance_grad=1e-9, tolerance_change=1e-11,
                            line_search_fn="strong_wolfe")
    calls = [0]
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = energy.energy(x).sum()
        loss.backward()
        calls[0] += 1
        return loss
    opt.step(closure)
    grad_rms = float(energy.grad(x).pow(2).mean().sqrt())
    solution = x.detach()
    return solution, {"energy_initial": initial,
                        "energy_final": float(energy.energy(x)[0].detach()),
                        "physics_grad_rms_final": grad_rms,
                        "physics_correction_rms": float(
                            (solution - mean.detach().double()).square().mean().sqrt()),
                        "function_evals": calls[0],
                        "finite": bool(torch.isfinite(x).all())}


def _cosine(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-300))


def _one_metrics(candidate, raw_mean, truth, energy, truth_energy, prior=None,
                 same_conditioning_state: bool = False):
    c, r, y = candidate.double(), raw_mean.double(), truth.double()
    desired = y - r
    correction = c - r
    pde = float(energy.rms_residual(c)[0])
    raw_pde = float(energy.rms_residual(r)[0])
    truth_pde = float(truth_energy.rms_residual(y)[0])
    floor_gap = raw_pde - truth_pde
    out = {
        "rmse": float((c - y).pow(2).mean().sqrt()),
        "relative_l2": float((c - y).norm() / y.norm().clamp(min=1e-30)),
        "pde_residual": pde,
        "raw_pde_residual": raw_pde,
        "truth_pde_residual": truth_pde,
        "pde_residual_over_truth_floor": pde / max(truth_pde, 1e-30),
        "pde_excess_over_truth_floor": pde - truth_pde,
        # This comparison is valid only at rollout step 1, where raw/corrected
        # predictions and truth all condition on the observed x_0. Later every
        # autoregressive method conditions on its own previous prediction, so a
        # true-frame residual is no longer a common attainable floor.
        "first_step_pde_progress_to_truth_floor": (
            (raw_pde - pde) / floor_gap
            if same_conditioning_state and floor_gap > 1e-30 else float("nan")),
        "divergence": float(energy.rms_divergence(c)[0]),
        "truth_divergence": float(truth_energy.rms_divergence(y)[0]),
        "enstrophy_error": float((energy.enstrophy(c)[0] -
                                  energy.enstrophy(y)[0]).abs()),
        "correction_rms": float((c - r).pow(2).mean().sqrt()),
        "correction_error_cosine": _cosine(correction, desired),
    }
    if prior is not None:
        out["prior_maha_over_N"] = float(prior.mahalanobis_sq(c) / prior.N)
        g = energy.grad(r).reshape(-1)
        ng = -covariance_matvec(prior, g).reshape(-1)
        # An oracle may use truth only as a diagnostic, never as a correction:
        # it tells us the maximum first-order error reduction available along
        # the HILP-preconditioned physics direction.  If this is small, the
        # physics loss is the issue; if large but MAP fails, it is optimization/
        # prior scaling.
        h2 = (ng * ng).sum().clamp(min=1e-300)
        e2 = (desired * desired).sum().clamp(min=1e-300)
        h_dot_e = ng @ desired
        eta_oracle = float((h_dot_e / h2).clamp(min=0))
        tangent = r.reshape(-1) + eta_oracle * ng
        out.update({
            "negative_physics_grad_error_cosine": _cosine(-g, desired),
            "preconditioned_grad_error_cosine": _cosine(ng, desired),
            "tangent_oracle_step": eta_oracle,
            "tangent_oracle_rmse_gain": float(
                1 - (tangent - y.reshape(-1)).pow(2).mean().sqrt() /
                (desired.pow(2).mean().sqrt().clamp(min=1e-300))),
            "tangent_oracle_mse_fraction": float(
                (h_dot_e.clamp(min=0).pow(2) / (h2 * e2))),
            "prior_variance_per_dim": prior.trace() / prior.N,
            "prior_floor_variance": prior.alpha * prior.tau,
            "prior_rank": prior.k,
        })
        if prior.k:
            err_frac = float((prior.U.T @ desired).pow(2).sum() /
                             desired.pow(2).sum().clamp(min=1e-300))
            grad_frac = float((prior.U.T @ g).pow(2).sum() /
                              g.pow(2).sum().clamp(min=1e-300))
            eig = prior.alpha * (prior.d + prior.tau)
            out.update({
                "error_energy_in_prior_rank": err_frac,
                "physics_grad_energy_in_prior_rank": grad_frac,
                "error_rank_enrichment": err_frac / (prior.k / prior.N),
                "physics_grad_rank_enrichment": grad_frac / (prior.k / prior.N),
                "prior_retained_condition": float(eig.max() / eig.min().clamp(min=1e-300)),
            })
        else:
            out.update({k: float("nan") for k in (
                "error_energy_in_prior_rank", "physics_grad_energy_in_prior_rank",
                "error_rank_enrichment", "physics_grad_rank_enrichment",
                "prior_retained_condition")})
    return out


def _finite_mean(values):
    """Mean finite values without NumPy's misleading empty-slice warning."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _aggregate(records):
    keys = records[0].keys()
    return {k: _finite_mean([r[k] for r in records]) for k in keys}


def make_physics_energy(spec, previous, channels, args, device,
                        divergence_weight=None, vorticity_scale=1.0,
                        divergence_scale=1.0, spectral_whitener=None,
                        spectral_residual_mean=None, transport_bias=None):
    """Create the requested physics likelihood with a common diagnostics API."""
    kw = dict(divergence_weight=(args.divergence_weight if divergence_weight is None
                                 else divergence_weight),
              vorticity_scale=vorticity_scale, divergence_scale=divergence_scale,
              dt=max(1, args.lead_steps) * spec.dt_out, device=device)
    likelihood = getattr(args, "physics_likelihood", "midpoint_residual")
    if likelihood == "midpoint_transport_1jvp":
        return MidpointTransportEnergy2D(
            spec, previous, channels,
            transport_bias=transport_bias,
            transport_jvp_terms=int(getattr(args, "transport_jvp_terms", 1)),
            spectral_whitener=spectral_whitener,
            spectral_residual_mean=spectral_residual_mean, **kw)
    if likelihood == "fixed_rk3_flow":
        if int(getattr(args, "physics_resolution", spec.n)) != int(spec.n):
            raise ValueError("fixed_rk3_flow requires --physics-resolution equal to target resolution")
        return FixedRK3FlowEnergy2D(
            spec, previous, channels,
            flow_substeps=int(getattr(args, "fixed_rk3_substeps", 64)), **kw)
    if likelihood == "discrete_flow":
        physics_n = int(getattr(args, "physics_resolution", spec.n))
        if physics_n < spec.n:
            return CoarseNativeFlowEnergy2D(
                spec, previous, channels, physics_n=physics_n,
                flow_cfl=args.flow_cfl, **kw)
        return NativeFlowEnergy2D(spec, previous, channels,
                                  flow_cfl=args.flow_cfl, **kw)
    return FMPhysicsEnergy2D(
        spec, previous, channels, substeps=args.physics_substeps,
        spectral_whitener=spectral_whitener,
        spectral_residual_mean=spectral_residual_mean, **kw)


def calibrate_midpoint_transport_bias(args, calibration, spec, channels, device):
    """Fit only the observation bias from calibration truth transitions.

    This nuisance term is fixed before validation and held-out evaluation. It
    does not predict a future state and cannot use future truth at inference.
    """
    if getattr(args, "physics_likelihood", None) != "midpoint_transport_1jvp":
        return None, {"enabled": False}
    defects = []
    with torch.no_grad():
        for i in range(calibration.shape[0]):
            for t in range(args.steps):
                previous = calibration[i, t].to(device, torch.float64)
                truth = calibration[i, t + 1].to(device, torch.float64)
                energy = MidpointTransportEnergy2D(
                    spec, previous, channels, divergence_weight=0.0,
                    substeps=1, dt=max(1, args.lead_steps) * spec.dt_out,
                    device=device)
                defects.append(energy.midpoint_endpoint_defect(
                    energy.to_vorticity(truth))[0])
    stacked = torch.stack(defects)
    bias = stacked.mean(0).detach()
    centered = stacked - bias
    return bias, {
        "enabled": True,
        "n_calibration_transitions": int(stacked.shape[0]),
        "bias_rms": float(bias.square().mean().sqrt()),
        "centered_truth_rms": float(centered.square().mean().sqrt()),
        "definition": "mean calibration midpoint endpoint defect in vorticity units",
    }


def _truth_floor_audit(truth, spec, channels, args, device, transport_bias=None):
    """Measure the residual floor induced by cadence/operator approximation.

    A finite-difference residual need not vanish on a true pair.  Persistence
    (x_{t+1}=x_t) provides a model-free reference: the physics score is useful
    only if true next frames score materially better than persistence.
    """
    true_residuals, persistence_residuals, true_divergences = [], [], []
    for i in range(truth.shape[0]):
        for t in range(args.steps):
            prev = truth[i, t].to(device)
            nxt = truth[i, t + 1].to(device)
            energy = make_physics_energy(spec, prev, channels, args, device,
                                         transport_bias=transport_bias)
            true_residuals.append(float(energy.rms_residual(nxt)[0]))
            persistence_residuals.append(float(energy.rms_residual(prev)[0]))
            true_divergences.append(float(energy.rms_divergence(nxt)[0]))

    true_arr = np.asarray(true_residuals)
    persist_arr = np.asarray(persistence_residuals)
    ratio = true_arr / np.maximum(persist_arr, 1e-30)
    return {
        "n_pairs": int(true_arr.size),
        "truth_residual_mean": float(true_arr.mean()),
        "truth_residual_median": float(np.median(true_arr)),
        "truth_residual_p90": float(np.quantile(true_arr, 0.9)),
        "persistence_residual_mean": float(persist_arr.mean()),
        "truth_over_persistence_mean": float(ratio.mean()),
        "truth_divergence_mean": float(np.mean(true_divergences)),
        "truth_beats_persistence_fraction": float(np.mean(true_arr < persist_arr)),
    }


def _calibration_physics_scales(args, fm, calibration, spec, floor_audit,
                                transport_bias=None):
    """Construct dimensionless physics scales without touching held-out data.

    The vorticity scale is the observed discrete-residual floor on true
    calibration transitions.  Divergence is an equality constraint whose true
    scale is effectively numerical zero, so its usable scale is the RMS
    divergence of *uncorrected calibration forecasts*.  Consequently both
    standardized defects are O(1) at the calibration forecast and the relative
    divergence weight remains interpretable.
    """
    raw_divergences = []
    for i in range(calibration.shape[0]):
        for t in range(args.steps):
            prev = calibration[i, t].to(fm.device, fm.dtype)
            raw = fm.predict(prev).double()
            unscaled = make_physics_energy(
                spec, prev, fm.state_shape[0], args, fm.device,
                divergence_weight=0.0, transport_bias=transport_bias)
            raw_divergences.append(float(unscaled.rms_divergence(raw)[0]))
    vortex = max(float(floor_audit["truth_residual_median"]), 1e-12)
    div = max(float(np.median(raw_divergences)), 1e-12)
    return {
        "mode": "calibration_standardized",
        "vorticity_scale": vortex,
        "vorticity_scale_source": "median true-transition residual on calibration",
        "divergence_scale": div,
        "divergence_scale_source": "median raw-forecast divergence on calibration",
        "raw_forecast_divergence_mean": float(np.mean(raw_divergences)),
        "n_calibration_forecasts": len(raw_divergences),
    }


def _calibration_spectral_whitener(args, fm, calibration, spec):
    """Estimate a regularized, radially pooled Fourier residual likelihood.

    The midpoint residual has a nonzero, solver-discretization-dependent mean
    even on true transitions.  We therefore model R_hat - E[R_hat], and pool
    its variance in radial bins; estimating a separate variance for each of
    8k Fourier coefficients from a dozen trajectories is not defensible.
    """
    if not args.spectral_whitening:
        return None, None, None
    dt_phys = max(1, args.lead_steps) * spec.dt_out
    samples = []
    for i in range(calibration.shape[0]):
        for t in range(args.steps):
            prev, nxt = calibration[i, t].to(fm.device), calibration[i, t + 1].to(fm.device)
            energy = FMPhysicsEnergy2D(
                spec, prev, fm.state_shape[0], divergence_weight=0.0,
                substeps=args.physics_substeps, dt=dt_phys, device=fm.device)
            rh = torch.fft.rfft2(energy.residual_fields(energy.to_vorticity(nxt)))
            samples.append(rh.reshape(-1, spec.n, spec.n // 2 + 1))
    samples = torch.cat(samples, dim=0)
    residual_mean = samples.mean(0)
    power = (samples - residual_mean).abs().pow(2).mean(0)
    grid = FMPhysicsEnergy2D(
        spec, calibration[0, 0].to(fm.device), fm.state_shape[0],
        divergence_weight=0.0, dt=dt_phys, device=fm.device).grid
    radius = grid.k2.sqrt()
    n_bins = 24
    bin_id = (radius / radius.max().clamp(min=1e-30) * (n_bins - 1)).long()
    global_power = power.mean().clamp(min=1e-30)
    radial_power = torch.empty_like(power)
    for b in range(n_bins):
        mask = bin_id == b
        # 10% global shrinkage guards both finite-sample variance and the
        # unresolved Nyquist end of a real FFT.
        pooled = power[mask].mean() if mask.any() else global_power
        radial_power[mask] = 0.9 * pooled + 0.1 * global_power
    whitener = radial_power.rsqrt()

    transformed_rms = []
    for i in range(calibration.shape[0]):
        for t in range(args.steps):
            prev, nxt = calibration[i, t].to(fm.device), calibration[i, t + 1].to(fm.device)
            energy = FMPhysicsEnergy2D(
                spec, prev, fm.state_shape[0], divergence_weight=0.0,
                spectral_whitener=whitener, spectral_residual_mean=residual_mean,
                substeps=args.physics_substeps,
                dt=dt_phys, device=fm.device)
            w = energy.to_vorticity(nxt)
            transformed_rms.append(float(energy._residual_energy_sq(w)[0].sqrt()))
    scale = max(float(np.median(transformed_rms)), 1e-12)
    meta = {"enabled": True, "centered_on_calibration_residual_mean": True,
            "radial_bins": n_bins, "global_variance_shrinkage": 0.1,
            "transformed_residual_scale": scale,
            "n_calibration_residual_fields": int(samples.shape[0])}
    return whitener, residual_mean, meta


def run_rollouts(args, fm, truth, spec, sigma2, method, lambda_rel,
                 hilp_floor_scale=1.0,
                 covariance_beta=1.0,
                 physics_scales=None,
                 capture_target=False):
    all_steps, diagnostics = [[] for _ in range(args.steps)], []
    captured = None
    dt_phys = max(1, args.lead_steps) * spec.dt_out
    physics_scales = physics_scales or {"vorticity_scale": 1.0,
                                        "divergence_scale": 1.0}
    spectral_whitener = physics_scales.get("spectral_whitener")
    spectral_residual_mean = physics_scales.get("spectral_residual_mean")
    started = time.time()
    for i in range(truth.shape[0]):
        c = truth[i, 0].to(fm.device, fm.dtype)
        state_prior = None
        for t in range(args.steps):
            raw_mean = fm.predict(c).double()
            energy = FMPhysicsEnergy2D(
                spec, c, fm.state_shape[0], divergence_weight=args.divergence_weight,
                vorticity_scale=physics_scales["vorticity_scale"],
                divergence_scale=physics_scales["divergence_scale"],
                spectral_whitener=spectral_whitener,
                spectral_residual_mean=spectral_residual_mean,
                substeps=args.physics_substeps, dt=dt_phys,
                device=fm.device)
            truth_energy = FMPhysicsEnergy2D(
                spec, truth[i, t], fm.state_shape[0],
                divergence_weight=args.divergence_weight,
                vorticity_scale=physics_scales["vorticity_scale"],
                divergence_scale=physics_scales["divergence_scale"],
                spectral_whitener=spectral_whitener,
                spectral_residual_mean=spectral_residual_mean,
                substeps=args.physics_substeps, dt=dt_phys, device=fm.device)

            if state_prior is None:
                predictive = LowRankGaussian.isotropic(
                    raw_mean, tau=sigma2, label="known-initial")
                prop_info = {"covariance_step": t + 1,
                             "covariance_source": "known_initial_isotropic",
                             "trace_retained": 0.0,
                             "trace_complement": 0.0,
                             "trace_injected": fm.N * sigma2,
                             "trace_total": fm.N * sigma2,
                             "frac_trace_in_rank": 0.0}
            else:
                # Common random numbers: every method/hyperparameter setting
                # receives the same Nyström probes at a given trajectory/step.
                # Without this, lambda/floor selection mixes actual performance
                # with the O(1/sqrt(m)) randomness of a small sketch.
                probe_gen = torch.Generator(device="cpu").manual_seed(
                    int(args.seed) + 100_003 * i + 1_009 * t)
                predictive, prop_info = propagate_step(
                    fm, c, state_prior, sigma2, k=args.k,
                    n_sketch=args.n_sketch, n_probe=args.n_probe,
                    chunk=args.chunk, complement="nystrom", generator=probe_gen)

            if method == "raw":
                corrected, opt_info, scored_prior = raw_mean, {}, None
            elif method == "raw_projected":
                corrected, opt_info, scored_prior = energy.project_incompressible(raw_mean), {}, None
            elif method == "physics_only":
                corrected, opt_info = _physics_solve(
                    raw_mean, energy, args.map_steps, args.map_lr)
                scored_prior = None
            else:
                if method == "hilp":
                    correction_prior = _scale_complement(
                        predictive, raw_mean, hilp_floor_scale)
                elif method == "hilp_blend":
                    correction_prior = _blend_hilp_isotropic(
                        _scale_complement(predictive, raw_mean, hilp_floor_scale),
                        raw_mean, covariance_beta)
                else:
                    correction_prior = _isotropize(predictive, raw_mean)
                lam_ref = force_scale(correction_prior, energy)
                post = LowRankPhysicsPosterior(
                    correction_prior, energy, lambda_rel * lam_ref)
                corrected, opt_info = post.map_estimate(
                    n_steps=args.map_steps, lr=args.map_lr)
                opt_info.update({"lambda": lambda_rel * lam_ref,
                                 "lambda_rel": lambda_rel,
                                 "lambda_ref": lam_ref,
                                 "hilp_floor_scale": hilp_floor_scale,
                                 "covariance_beta": covariance_beta})
                opt_info["posterior_grad_rms_final"] = float(
                    post.grad_log_prob(corrected).pow(2).mean().sqrt())
                scored_prior = correction_prior
                if capture_target and i == 0 and t == args.steps - 1:
                    captured = (post, corrected)

            if args.project_incompressible and method not in ("raw", "raw_projected"):
                corrected = energy.project_incompressible(corrected)
            if not torch.isfinite(corrected).all():
                raise RuntimeError(f"{method} produced non-finite state at trajectory {i}, step {t+1}")
            all_steps[t].append(_one_metrics(
                corrected, raw_mean, truth[i, t + 1], energy, truth_energy,
                scored_prior, same_conditioning_state=(t == 0)))
            diagnostics.append({"trajectory": i, "step": t + 1,
                                "predictive_rank": predictive.k,
                                "predictive_variance_per_dim": predictive.trace() / predictive.N,
                                **prop_info, **opt_info})
            c = corrected.to(fm.dtype)
            if method in ("hilp", "hilp_blend"):
                state_prior = _center(predictive, corrected)
            elif method == "isotropic":
                state_prior = _isotropize(predictive, corrected)
            else:
                state_prior = None
    return ([_aggregate(x) for x in all_steps], diagnostics,
            time.time() - started, captured)


def _summary_score(rows):
    return float(np.mean([r["rmse"] for r in rows]))


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=2)
    ap.add_argument("--n-test-traj", type=int, default=4)
    ap.add_argument("--sim-batch", type=int, default=2)
    ap.add_argument("--n-tail", type=int, default=8)
    ap.add_argument("--n-sketch", type=int, default=None)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    ap.add_argument("--hilp-floor-grid", nargs="+", type=float, default=[1.0],
                    help="HILP-only complement variance scales in (0,1]; "
                         "retained eigenvalues are preserved")
    ap.add_argument("--beta-grid", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="calibration grid for the HILP/isotropic covariance blend")
    ap.add_argument("--methods", nargs="+",
                    default=["raw", "physics_only", "isotropic", "hilp"])
    ap.add_argument("--map-steps", type=int, default=30)
    ap.add_argument("--map-lr", type=float, default=0.5)
    ap.add_argument("--physics-substeps", type=int, default=2)
    ap.add_argument("--divergence-weight", type=float, default=1.0)
    ap.add_argument("--spectral-whitening", action="store_true",
                    help="use a calibration-estimated diagonal Fourier residual metric")
    ap.add_argument("--project-incompressible", action="store_true",
                    help="Helmholtz-project every corrected velocity state")
    ap.add_argument("--sampler", choices=["none", "mala", "hmc"], default="none")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--burn-in", type=int, default=64)
    ap.add_argument("--n-leapfrog", type=int, default=8)
    ap.add_argument("--data-source", choices=["synthetic", "poseidon-native"],
                    default="synthetic")
    ap.add_argument("--fm-data-path", default=None)
    args = ap.parse_args()
    if any(not 0.0 < x <= 1.0 for x in args.hilp_floor_grid):
        raise SystemExit("--hilp-floor-grid values must all lie in (0, 1]")
    if any(not 0.0 <= x <= 1.0 for x in args.beta_grid):
        raise SystemExit("--beta-grid values must all lie in [0, 1]")
    if args.data_source == "poseidon-native":
        if args.fm != "poseidon" or not args.fm_data_path:
            raise SystemExit("poseidon-native requires --fm poseidon and --fm-data-path")
        dt_target = configure_native_poseidon_cadence(args)
    else:
        dt_target = configure_fm_cadence(args)
    # This legacy midpoint-residual script pre-dates the native discrete-flow
    # likelihood.  DPOT's released checkpoint has neither its dataset-specific
    # normalizer nor its true 10-frame history here, so assigning the synthetic
    # NS residual to its output would be a unit/cadence mismatch masquerading
    # as a physics experiment.  Keep the restriction explicit rather than
    # producing an attractive but invalid DPOT S4 table.
    if args.fm != "poseidon":
        raise SystemExit(
            "S4 physical correction is currently Poseidon-only. DPOT/MORPH "
            "have no verified matched physical trajectory, preprocessing, and "
            "history contract in this repository; use S1--S3 until those are "
            "provided. Do not reuse the Poseidon/AZEBAN residual for them.")

    set_seed(args.seed)
    fm = load_fm(args)
    print_header(f"FM S4 correction: {fm.info.name}, lead={args.lead_steps}, k={args.k}")
    if hasattr(fm, "dpot_history_policy"):
        print("WARNING: DPOT uses repeated-current history; treat as a pilot, not a final claim.")

    if args.data_source == "poseidon-native":
        spec = native_poseidon_spec(args, fm)
        print(f"  native cadence: {POSEIDON_NATIVE_RAW_STRIDE} raw frames / "
              f"Poseidon step; dt={spec.dt_out:g}; model lead={fm.lead_time:g}")
        cal = load_poseidon_trajectories(
            args.fm_data_path, fm, args.n_cal_traj, args.steps,
            lead_steps=args.lead_steps, offset=0)
        test = load_poseidon_trajectories(
            args.fm_data_path, fm, args.n_test_traj, args.steps,
            lead_steps=args.lead_steps, offset=args.n_cal_traj)
        args.trajectory_scaling = "native-physical-none"
    else:
        cal, spec = generate_fm_trajectories(
            args, fm, args.n_cal_traj, args.steps * args.lead_steps,
            seed=args.seed, preserve_physics=True)
        test, _ = generate_fm_trajectories(
            args, fm, args.n_test_traj, args.steps * args.lead_steps,
            seed=args.seed + 10000, preserve_physics=True)
        cal = cal[:, ::args.lead_steps]
        test = test[:, ::args.lead_steps]
    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma = fit_sigma2(fm, xs, ys)
    sigma2 = sigma["sigma2"]
    print(f"  physical target: {spec.name}, dt={dt_target:g}, "
          f"Poseidon lead={getattr(fm, 'lead_time', 'fixed')}")
    if args.data_source == "poseidon-native":
        print("  physics     AZEBAN smooth spectral-viscosity filter "
              "(published exponent=18); midpoint time-residual diagnostic")
    print(f"  fitted one-step sigma^2={sigma2:.5e}")

    calibration_floor = _truth_floor_audit(
        cal, spec, fm.state_shape[0], args, fm.device)
    test_floor = _truth_floor_audit(
        test, spec, fm.state_shape[0], args, fm.device)
    print("\n  physical-residual floor audit (true consecutive frames)")
    print(f"    calibration: truth={calibration_floor['truth_residual_mean']:.4g}, "
          f"persistence={calibration_floor['persistence_residual_mean']:.4g}, "
          f"ratio={calibration_floor['truth_over_persistence_mean']:.3f}")
    print(f"    held-out:   truth={test_floor['truth_residual_mean']:.4g}, "
          f"persistence={test_floor['persistence_residual_mean']:.4g}, "
          f"ratio={test_floor['truth_over_persistence_mean']:.3f}, "
          f"truth wins={100 * test_floor['truth_beats_persistence_fraction']:.1f}%")
    residual_is_discriminative = (
        test_floor["truth_over_persistence_mean"] < 0.9 and
        test_floor["truth_beats_persistence_fraction"] >= 0.75)
    if not residual_is_discriminative:
        print("    WARNING: approximate PDE residual does not reliably distinguish "
              "true evolution from persistence; correction results are diagnostic only")

    physics_scales = _calibration_physics_scales(
        args, fm, cal, spec, calibration_floor)
    spectral_whitener, spectral_residual_mean, whitening_meta = _calibration_spectral_whitener(
        args, fm, cal, spec)
    if spectral_whitener is not None:
        physics_scales["spectral_whitener"] = spectral_whitener
        physics_scales["spectral_residual_mean"] = spectral_residual_mean
        # The whitened residual needs its own calibration scale, while all
        # reported PDE residuals remain in the original physical units.
        physics_scales["vorticity_scale"] = whitening_meta["transformed_residual_scale"]
    print("\n  calibrated dimensionless physics scales (calibration only)")
    print(f"    vorticity residual scale={physics_scales['vorticity_scale']:.4g} "
          "(true-transition median)")
    print(f"    divergence scale={physics_scales['divergence_scale']:.4g} "
          "(raw-forecast median)")
    print(f"    divergence relative weight={args.divergence_weight:g}")
    if whitening_meta is not None:
        print(f"    spectral residual whitening=on; transformed scale="
              f"{whitening_meta['transformed_residual_scale']:.4g}")
    if args.project_incompressible:
        print("    incompressibility=exact Helmholtz projection of corrected states")

    selected, selected_floor, selected_beta, sweep = {}, {}, {}, {}
    for method in args.methods:
        if method in ("raw", "raw_projected", "physics_only"):
            selected[method] = 0.0
            selected_floor[method] = 1.0
            selected_beta[method] = 1.0
            continue
        print(f"\n  selecting lambda for {method} on calibration trajectories")
        cells = []
        floor_grid = args.hilp_floor_grid if method in ("hilp", "hilp_blend") else [1.0]
        beta_grid = args.beta_grid if method == "hilp_blend" else [1.0]
        for floor_scale in floor_grid:
            for beta in beta_grid:
                for lam in args.lambda_grid:
                    rows, _, secs, _ = run_rollouts(
                        args, fm, cal, spec, sigma2, method, lam,
                        hilp_floor_scale=floor_scale, covariance_beta=beta,
                        physics_scales=physics_scales)
                    cell = {"lambda_rel": lam, "hilp_floor_scale": floor_scale,
                            "covariance_beta": beta,
                            "mean_rollout_rmse": _summary_score(rows),
                            "rows": rows, "seconds": secs}
                    cells.append(cell)
                    extra = (f", floor={floor_scale:g}" if method in ("hilp", "hilp_blend") else "")
                    extra += (f", beta={beta:g}" if method == "hilp_blend" else "")
                    print(f"    lambda/ref={lam:g}{extra}: "
                          f"mean rollout RMSE={cell['mean_rollout_rmse']:.6g}")
        best = min(cells, key=lambda x: x["mean_rollout_rmse"])
        selected[method] = best["lambda_rel"]
        selected_floor[method] = best["hilp_floor_scale"]
        selected_beta[method] = best["covariance_beta"]
        sweep[method] = cells
        print(f"    selected {best['lambda_rel']:g} x reference, "
              f"floor={best['hilp_floor_scale']:g}, beta={best['covariance_beta']:g}")
        if best["lambda_rel"] in (min(args.lambda_grid), max(args.lambda_grid)):
            print("    WARNING: selected lambda is on the sweep boundary; "
                  "the optimum is not bracketed")

    print("\n  held-out physical rollout evaluation")
    final, opt, sample_target = {}, {}, None
    for method in args.methods:
        rows, diag, secs, captured = run_rollouts(
            args, fm, test, spec, sigma2, method, selected[method],
            hilp_floor_scale=selected_floor[method],
            covariance_beta=selected_beta[method],
            physics_scales=physics_scales,
            capture_target=(method == "hilp" and args.sampler != "none"))
        final[method] = {"steps": rows, "mean_rollout_rmse": _summary_score(rows),
                         "seconds": secs}
        opt[method] = diag
        if captured is not None:
            sample_target = captured

    raw_rmse = final["raw"]["mean_rollout_rmse"]
    table = Table("method", "RMSE", "gain%", "PDE resid", "step-1 PDE gain%",
                  "div", "enst err", "corr RMS")
    for method in args.methods:
        rows = final[method]["steps"]
        avg = {k: _finite_mean([r[k] for r in rows]) for k in rows[0]}
        table.add(method, avg["rmse"], 100 * (1 - avg["rmse"] / raw_rmse),
                  avg["pde_residual"],
                  100 * avg["first_step_pde_progress_to_truth_floor"], avg["divergence"],
                  avg["enstrophy_error"], avg["correction_rms"])
    print(table)
    if "hilp" in final:
        hrows = final["hilp"]["steps"]
        def havg(key):
            return _finite_mean([r[key] for r in hrows])
        print("\nHILP debugging diagnostics")
        print(f"  truth PDE residual sanity       {havg('truth_pde_residual'):.4g}")
        print(f"  residual / truth floor          {havg('pde_residual_over_truth_floor'):.3f}x")
        print(f"  step-1 raw -> truth-floor gain  "
              f"{100*havg('first_step_pde_progress_to_truth_floor'):.1f}%")
        print(f"  truth divergence sanity         {havg('truth_divergence'):.4g}")
        print(f"  correction vs true-error cosine {havg('correction_error_cosine'):+.3f}")
        print(f"  -physics-grad vs error cosine   {havg('negative_physics_grad_error_cosine'):+.3f}")
        print(f"  -Sigma*grad vs error cosine     {havg('preconditioned_grad_error_cosine'):+.3f}")
        print(f"  tangent oracle RMSE gain         {100*havg('tangent_oracle_rmse_gain'):.2f}%")
        print(f"  tangent oracle MSE fraction      {100*havg('tangent_oracle_mse_fraction'):.2f}%")
        print(f"  error rank enrichment           {havg('error_rank_enrichment'):.2f}x")
        print(f"  physics-grad rank enrichment    {havg('physics_grad_rank_enrichment'):.2f}x")
        print(f"  prior variance/dim              {havg('prior_variance_per_dim'):.4g}")
        print(f"  prior floor variance            {havg('prior_floor_variance'):.4g}")
        hdiag = opt["hilp"]
        propagated = [d for d in hdiag if d["predictive_rank"] > 0]
        if propagated:
            print(f"  propagated rank-trace fraction  "
                  f"{100*np.mean([d['frac_trace_in_rank'] for d in propagated]):.2f}%")

    sampling = None
    if args.sampler != "none":
        if sample_target is None:
            raise RuntimeError("sampling requested but the hilp method was not run")
        target, xmap = sample_target
        min_var = target.prior.alpha * min(
            target.prior.tau,
            float((target.prior.d + target.prior.tau).min())
            if target.prior.k else target.prior.tau)
        step_size = 0.05 * math.sqrt(max(min_var, 1e-30))
        gen = torch.Generator(device=fm.device).manual_seed(args.seed + 50000)
        if args.sampler == "mala":
            chain = langevin(target, xmap, n_samples=args.n_samples,
                             burn_in=args.burn_in, step_size=step_size,
                             metropolis=True, adapt=True, generator=gen)
        else:
            chain = hmc(target, xmap, n_samples=args.n_samples,
                        burn_in=args.burn_in, step_size=step_size,
                        n_leapfrog=args.n_leapfrog, adapt=True, generator=gen)
        dims = torch.linspace(0, chain.samples.shape[1] - 1,
                              min(64, chain.samples.shape[1])).long()
        ess = effective_sample_size(chain.samples[:, dims])
        sampling = {"sampler": args.sampler, "n_samples": int(chain.samples.shape[0]),
                    "accept_rate": chain.accept_rate, "step_size": chain.step_size,
                    "gradient_evals": chain.n_grad_evals,
                    "divergences": chain.diverged,
                    "ess_median_64_coordinates": float(np.median(ess)),
                    "posterior_mean_shift_from_map": float(
                        (chain.mean() - xmap).pow(2).mean().sqrt())}
        print(f"\n  {args.sampler}: accept={chain.accept_rate:.3f}, "
              f"divergences={chain.diverged}, median ESS={np.median(ess):.1f}")

    physics_scales_json = {k: v for k, v in physics_scales.items()
                           if k not in ("spectral_whitener", "spectral_residual_mean")}
    out = {"stage": "s4_fm_physics_correction",
           "metadata": fm_metadata(args, fm, spec), "sigma2": sigma,
           "steps": args.steps, "n_cal_traj": args.n_cal_traj,
           "n_test_traj": args.n_test_traj, "lambda_grid": args.lambda_grid,
           "hilp_floor_grid": args.hilp_floor_grid,
           "physics_scales": physics_scales_json,
           "spectral_whitening": whitening_meta,
           "incompressibility_projection": args.project_incompressible,
           "selected_lambda_rel": selected,
           "selected_hilp_floor_scale": selected_floor,
           "selected_covariance_beta": selected_beta,
           "calibration_sweep": sweep,
           "residual_floor_audit": {"calibration": calibration_floor,
                                    "test": test_floor,
                                    "discriminative": residual_is_discriminative},
           "test": final, "optimizer": opt,
           "sampling": sampling,
           "validity": {"physical_trajectory_scaling": "none",
                        "lambda_selected_on": "calibration trajectories",
                        "test_split_held_out": True,
                        "test_split_scope": (
                            "trajectory-disjoint from post-hoc calibration; "
                            "a single chunk does not by itself verify the "
                            "published Poseidon train/val/test membership"),
                        "residual_floor_measured_on_true_frame_pairs": True,
                        "residual_discriminative_vs_persistence": residual_is_discriminative,
                        "physics_residual": (
                            "AZEBAN smooth spectral-viscosity filter with a "
                            "midpoint discrete-time residual diagnostic"
                            if args.data_source == "poseidon-native" else
                            "synthetic solver equation")}}
    key = fm_result_key(args)
    p = save_json(out, results_path_scale("s4_fm_correction", key, "results.json", args.tag))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
