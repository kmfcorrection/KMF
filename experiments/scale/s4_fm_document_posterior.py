#!/usr/bin/env python3
"""Document-faithful S4: calibrated isotropic, pushforward, and GN posteriors."""
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
    configure_native_poseidon_cadence, fm_metadata, fm_result_key,
    generate_fm_trajectories, load_poseidon_trajectories, native_poseidon_spec)
from experiments.scale.s4_fm_physics_correction import (
    _aggregate, _calibration_physics_scales, _one_metrics, _physics_solve,
    _truth_floor_audit, make_physics_energy)
from hipp.scale.calibrate_scale import fit_alpha_scale
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.curvature_scale import estimate_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.scale.lowrank import LowRankGaussian
from hipp.scale.posterior_scale import (LowRankPhysicsPosterior,
                                         covariance_matvec, force_scale)
from hipp.scale.rollout import fit_sigma2, propagate_step
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("isotropic", "pushforward", "gauss_newton", "physics_only")


def _center(prior, mean):
    return LowRankGaussian(mean.double(), prior.U, prior.d, prior.tau,
                           alpha=prior.alpha, label=prior.label)


def _base_prior(method, fm, c, raw, sigma2, args, generator=None):
    """One-step prior shape centered at raw=f(c), with alpha deliberately one."""
    if method == "isotropic":
        return LowRankGaussian.isotropic(raw.double(), tau=1.0, label="isotropic")
    if method == "gauss_newton":
        return estimate_scale(
            fm, c, method="gn", k=args.k, tau_rel=args.tau_rel,
            oversample=args.oversample, n_iter=args.n_iter, chunk=args.chunk,
            n_tail=args.n_tail, generator=generator).prior
    raise ValueError(method)


def _calibration_summaries(args, fm, trajectories, sigma2, methods):
    """Raw autoregressive calibration of every covariance *shape*.

    No truth from the held-out set enters this routine.  Pushforward uses the
    Lyapunov recursion; GN is recomputed at the current raw conditioning state,
    as required by the document's local J^T J construction.
    """
    summaries = {m: [] for m in methods}
    for method in methods:
        if method == "physics_only":
            continue
        print(f"    building {method} calibration priors ({trajectories.shape[0]} trajectories x "
              f"{args.steps} rollout steps)", flush=True)
        for i in range(trajectories.shape[0]):
            c = trajectories[i, 0].to(fm.device, fm.dtype)
            pf_state = None
            for t in range(args.steps):
                print(f"      {method}: trajectory {i + 1}/{trajectories.shape[0]}, "
                      f"step {t + 1}/{args.steps}", flush=True)
                raw = fm.predict(c).double()
                gen = torch.Generator(device="cpu").manual_seed(
                    args.seed + 90_001 * i + 1_009 * t)
                if method == "pushforward":
                    if pf_state is None:
                        prior = LowRankGaussian.isotropic(
                            raw, tau=sigma2, label="pushforward-step1")
                    else:
                        prior, _ = propagate_step(
                            fm, c, pf_state, sigma2, k=args.k,
                            n_sketch=args.n_sketch, n_probe=args.n_probe,
                            chunk=args.chunk, generator=gen)
                else:
                    prior = _base_prior(method, fm, c, raw, sigma2, args, gen)
                summaries[method].append(prior.summarize(trajectories[i, t + 1]))
                c = raw.to(fm.dtype)
                if method == "pushforward":
                    pf_state = _center(prior, raw)
    return summaries


def _energy(spec, previous, channels, args, scales, device):
    return make_physics_energy(
        spec, previous, channels, args, device,
        vorticity_scale=scales["vorticity_scale"],
        divergence_scale=scales["divergence_scale"])


def _rollout_serial(args, fm, truth, spec, sigma2, method, alpha, lambda_rel, scales,
                    trace=False):
    all_steps, diagnostics = [[] for _ in range(args.steps)], []
    started = time.time()
    injected = alpha * sigma2
    for i in range(truth.shape[0]):
        c = truth[i, 0].to(fm.device, fm.dtype)
        pf_state = None
        for t in range(args.steps):
            raw = fm.predict(c).double()
            gen = torch.Generator(device="cpu").manual_seed(
                args.seed + 90_001 * i + 1_009 * t)
            if method == "physics_only":
                # Only a placeholder for common metric/diagnostic plumbing;
                # it is never used as a posterior prior.
                prior = LowRankGaussian.isotropic(raw, tau=1.0, label="physics-only")
            elif method == "pushforward":
                if pf_state is None:
                    prior = LowRankGaussian.isotropic(
                        raw, tau=injected, label="pushforward-step1")
                else:
                    prior, prop = propagate_step(
                        fm, c, pf_state, injected, k=args.k,
                        n_sketch=args.n_sketch, n_probe=args.n_probe,
                        chunk=args.chunk, generator=gen)
                    diagnostics.append({"trajectory": i, "step": t + 1, **prop})
            else:
                prior = _base_prior(method, fm, c, raw, sigma2, args, gen).rescaled(alpha)

            energy = _energy(spec, c, fm.state_shape[0], args, scales, fm.device)
            truth_energy = _energy(spec, truth[i, t], fm.state_shape[0], args, scales, fm.device)
            if method == "physics_only":
                corrected, info = _physics_solve(raw, energy, args.map_steps, args.map_lr)
                info.update({"lambda_rel": float("nan"), "lambda": float("nan"),
                             "lambda_ref": float("nan"), "alpha": float("nan"),
                             "prior_rank": 0, "prior_variance_per_dim": float("nan")})
            else:
                ref = force_scale(prior, energy)
                post = LowRankPhysicsPosterior(prior, energy, lambda_rel * ref)
                corrected, info = post.map_estimate(n_steps=args.map_steps, lr=args.map_lr)
                info.update({"lambda_rel": lambda_rel, "lambda": lambda_rel * ref,
                             "lambda_ref": ref, "alpha": alpha,
                             "prior_rank": prior.k,
                             "prior_variance_per_dim": prior.trace() / prior.N})
            metrics = _one_metrics(
                corrected, raw, truth[i, t + 1], energy, truth_energy,
                None if method == "physics_only" else prior,
                same_conditioning_state=(t == 0))
            all_steps[t].append(metrics)
            if trace and method != "physics_only":
                # This is the decisive local test.  If the physical gradient
                # has negligible projection into U, changing rank/covariance
                # cannot materially change the MAP correction.
                g = energy.grad(raw).reshape(-1)
                g2 = g.square().sum().clamp(min=1e-300)
                grad_in_rank = (prior.U.T @ g).square().sum() / g2 if prior.k else g2 * 0
                sg = covariance_matvec(prior, g).reshape(-1)
                sg_iso = (prior.trace() / prior.N) * g
                trace_record = {
                    "trace_physics_energy_raw": float(energy.energy(raw)[0]),
                    "trace_physics_energy_map": float(energy.energy(corrected)[0]),
                    "trace_weighted_physics_drop": float(
                        lambda_rel * ref * (energy.energy(raw)[0] - energy.energy(corrected)[0])),
                    "trace_prior_cost_map": float(0.5 * prior.mahalanobis_sq(corrected)),
                    "trace_grad_energy_in_rank": float(grad_in_rank),
                    "trace_anisotropic_action_ratio": float(
                        (sg - sg_iso).norm() / sg_iso.norm().clamp(min=1e-300)),
                    "trace_correction_rms": metrics["correction_rms"],
                    "trace_tangent_oracle_gain": metrics["tangent_oracle_rmse_gain"],
                    "trace_preconditioned_alignment": metrics["preconditioned_grad_error_cosine"],
                }
                info.update(trace_record)
            diagnostics.append({"trajectory": i, "step": t + 1, **info})
            c = corrected.to(fm.dtype)
            if method == "pushforward":
                pf_state = _center(prior, corrected)
    return [_aggregate(r) for r in all_steps], diagnostics, time.time() - started


def _rollout_batched_native_flow(args, fm, truth, spec, sigma2, method, alpha,
                                 lambda_rel, scales, trace=False):
    """Exact equivalent of the serial rollout with batched native flow maps.

    Autoregression still advances one physical time level at a time, but all
    independent trajectories at that level are sent through Poseidon and the
    adaptive AZEBAN map together.  Each member retains its own CFL step sizes;
    ``NativeFlowEnergy2D.member`` then provides the unchanged per-trajectory
    posterior objective.  Curvature estimates remain per state, because each
    has a different conditioning point and cannot be merged mathematically.
    """
    all_steps, diagnostics = [[] for _ in range(args.steps)], []
    started = time.time()
    injected = alpha * sigma2
    n_traj = truth.shape[0]
    states = [truth[i, 0].to(fm.device, fm.dtype) for i in range(n_traj)]
    pf_states = [None] * n_traj
    for t in range(args.steps):
        c_batch = torch.stack(states)
        raw_batch = fm.predict(c_batch).double()
        energy_batch = _energy(spec, c_batch, fm.state_shape[0], args, scales, fm.device)
        truth_batch = truth[:, t].to(fm.device, fm.dtype)
        truth_energy_batch = _energy(spec, truth_batch, fm.state_shape[0], args,
                                     scales, fm.device)
        if not hasattr(energy_batch, "member") or not hasattr(truth_energy_batch, "member"):
            raise RuntimeError("batched rollout requires a flow-endpoint physics likelihood")
        next_states, next_pf = [], []
        for i in range(n_traj):
            c, raw = states[i], raw_batch[i]
            energy, truth_energy = energy_batch.member(i), truth_energy_batch.member(i)
            gen = torch.Generator(device="cpu").manual_seed(
                args.seed + 90_001 * i + 1_009 * t)
            if method == "physics_only":
                prior = LowRankGaussian.isotropic(raw, tau=1.0, label="physics-only")
            elif method == "pushforward":
                if pf_states[i] is None:
                    prior = LowRankGaussian.isotropic(raw, tau=injected,
                                                       label="pushforward-step1")
                else:
                    prior, prop = propagate_step(
                        fm, c, pf_states[i], injected, k=args.k,
                        n_sketch=args.n_sketch, n_probe=args.n_probe,
                        chunk=args.chunk, generator=gen)
                    diagnostics.append({"trajectory": i, "step": t + 1, **prop})
            else:
                prior = _base_prior(method, fm, c, raw, sigma2, args, gen).rescaled(alpha)

            if method == "physics_only":
                # The discrete-flow target is exactly divergence-free, hence
                # it is the unique minimum of this physical-only objective.
                corrected = energy.flow_target.reshape(-1).clone()
                info = {"physics_only_closed_form": True,
                        "lambda_rel": float("nan"), "lambda": float("nan"),
                        "lambda_ref": float("nan"), "alpha": float("nan"),
                        "prior_rank": 0, "prior_variance_per_dim": float("nan")}
            else:
                ref = force_scale(prior, energy)
                post = LowRankPhysicsPosterior(prior, energy, lambda_rel * ref)
                corrected, info = post.map_estimate(n_steps=args.map_steps, lr=args.map_lr)
                info.update({"lambda_rel": lambda_rel, "lambda": lambda_rel * ref,
                             "lambda_ref": ref, "alpha": alpha,
                             "prior_rank": prior.k,
                             "prior_variance_per_dim": prior.trace() / prior.N})
            metrics = _one_metrics(
                corrected, raw, truth[i, t + 1], energy, truth_energy,
                None if method == "physics_only" else prior,
                same_conditioning_state=(t == 0))
            all_steps[t].append(metrics)
            if trace and method != "physics_only":
                g = energy.grad(raw).reshape(-1)
                g2 = g.square().sum().clamp(min=1e-300)
                grad_in_rank = (prior.U.T @ g).square().sum() / g2 if prior.k else g2 * 0
                sg = covariance_matvec(prior, g).reshape(-1)
                sg_iso = (prior.trace() / prior.N) * g
                info.update({
                    "trace_physics_energy_raw": float(energy.energy(raw)[0]),
                    "trace_physics_energy_map": float(energy.energy(corrected)[0]),
                    "trace_weighted_physics_drop": float(
                        lambda_rel * ref * (energy.energy(raw)[0] - energy.energy(corrected)[0])),
                    "trace_prior_cost_map": float(0.5 * prior.mahalanobis_sq(corrected)),
                    "trace_grad_energy_in_rank": float(grad_in_rank),
                    "trace_anisotropic_action_ratio": float(
                        (sg - sg_iso).norm() / sg_iso.norm().clamp(min=1e-300)),
                    "trace_correction_rms": metrics["correction_rms"],
                    "trace_tangent_oracle_gain": metrics["tangent_oracle_rmse_gain"],
                    "trace_preconditioned_alignment": metrics["preconditioned_grad_error_cosine"],
                })
            diagnostics.append({"trajectory": i, "step": t + 1, **info})
            next_states.append(corrected.to(fm.dtype))
            next_pf.append(_center(prior, corrected) if method == "pushforward" else None)
        states, pf_states = next_states, next_pf
    return [_aggregate(r) for r in all_steps], diagnostics, time.time() - started


def _rollout(args, fm, truth, spec, sigma2, method, alpha, lambda_rel, scales,
             trace=False):
    if (args.physics_likelihood in {"discrete_flow", "fixed_rk3_flow"} and
            not args.serial_native_flow):
        return _rollout_batched_native_flow(
            args, fm, truth, spec, sigma2, method, alpha, lambda_rel, scales, trace)
    return _rollout_serial(args, fm, truth, spec, sigma2, method, alpha, lambda_rel,
                           scales, trace)


def _score(rows):
    return float(np.mean([r["rmse"] for r in rows]))


def _print_posterior_trace(method, diagnostics, steps):
    """Compact held-out evidence for whether HILP geometry can act on physics."""
    print(f"\n  posterior-geometry trace [{method}]")
    print("  step  grad-in-rank  anisotropic-action  corr-RMS  tangent-gain  "
          "align(Sigma*g,error)  prior-cost  weighted-physics-drop")
    for step in range(1, steps + 1):
        rows = [d for d in diagnostics if d["step"] == step and
                "trace_grad_energy_in_rank" in d]
        if not rows:
            continue
        avg = lambda key: float(np.mean([r[key] for r in rows]))
        print(f"  {step:>4d}  {avg('trace_grad_energy_in_rank'):.3e}     "
              f"{avg('trace_anisotropic_action_ratio'):.3e}         "
              f"{avg('trace_correction_rms'):.3e}  "
              f"{100*avg('trace_tangent_oracle_gain'):.2f}%       "
              f"{avg('trace_preconditioned_alignment'):+.3f}              "
              f"{avg('trace_prior_cost_map'):.3g}      "
              f"{avg('trace_weighted_physics_drop'):.3g}")


def _print_physics_only_trace(diagnostics):
    """Show whether the residual optimizer genuinely moved and reduced E."""
    if not diagnostics:
        return
    if all(d.get("physics_only_closed_form", False) for d in diagnostics):
        print("\n  flow-endpoint physics-only audit")
        print("    correction is the exact minimum: the precomputed flow endpoint; "
              "no L-BFGS optimization is performed")
        return
    def avg(key):
        values = [r.get(key, float("nan")) for r in diagnostics]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")
    before, after = avg("energy_initial"), avg("energy_final")
    print("\n  residual-only optimizer audit")
    print(f"    standardized E: {before:.4g} -> {after:.4g} "
          f"({100 * (1 - after / max(before, 1e-300)):.2f}% reduction)")
    print(f"    correction RMS: {avg('physics_correction_rms'):.4g}; "
          f"mean L-BFGS evaluations: {avg('function_evals'):.1f}")


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=3)
    ap.add_argument("--n-cal-traj", type=int, default=8)
    ap.add_argument("--n-val-traj", type=int, default=0,
                    help="trajectory-disjoint validation trajectories used only to select lambda; "
                         "zero preserves the legacy calibration-only selection protocol")
    ap.add_argument("--n-test-traj", type=int, default=8)
    ap.add_argument("--n-sketch", type=int, default=None)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--n-tail", type=int, default=16)
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                    help="prior families to evaluate; use pushforward alone for a rank sweep")
    ap.add_argument("--map-steps", type=int, default=30)
    ap.add_argument("--map-lr", type=float, default=0.5)
    ap.add_argument("--physics-substeps", type=int, default=2)
    ap.add_argument("--physics-likelihood",
                    choices=["midpoint_residual", "midpoint", "discrete_flow", "fixed_rk3_flow"],
                    default="midpoint_residual",
                    help="physics energy: midpoint_residual evaluates the local "
                         "vorticity PDE defect E(x) directly; discrete_flow compares "
                         "against an adaptive AZEBAN endpoint; fixed_rk3_flow compares against "
                         "a full-grid prescribed-step SSP-RK3 endpoint. 'midpoint' is a backwards-compatible alias")
    ap.add_argument("--flow-cfl", type=float, default=0.5,
                    help="CFL number for the AZEBAN discrete-flow likelihood")
    ap.add_argument("--fixed-rk3-substeps", type=int, default=64,
                    help="exact SSP-RK3 steps for --physics-likelihood fixed_rk3_flow")
    ap.add_argument("--physics-resolution", type=int, default=128,
                    help="endpoint-solver grid; below target grid activates the fixed coarse-physics benchmark")
    ap.add_argument("--physics-only-calibration-gate", action="store_true",
                    help="score only raw versus physics-only on calibration trajectories, "
                         "then exit without loading held-out data; use this to select a "
                         "pre-specified coarse solver resolution")
    ap.add_argument("--serial-native-flow", action="store_true",
                    help="disable trajectory batching; correctness reference path")
    ap.add_argument("--verify-batched", action="store_true",
                    help="assert serial/batched Poseidon and AZEBAN agreement before a run")
    ap.add_argument("--divergence-weight", type=float, default=1.0)
    ap.add_argument("--trace-posterior", action="store_true",
                    help="print held-out diagnostics for the covariance/physics-gradient bottleneck")
    ap.add_argument("--data-source", choices=["synthetic", "poseidon-native"],
                    default="poseidon-native")
    ap.add_argument("--fm-data-path", required=True)
    ap.add_argument("--allow-native-transfer", action="store_true",
                    help="explicitly allow a fixed-step external FM (DPOT/MORPH) "
                    "to be evaluated on Poseidon NS-Gauss at one dt=0.1 transition")
    args = ap.parse_args()
    # Preserve command-line compatibility with previous experiments while
    # storing one unambiguous name in results JSON and console output.
    if args.physics_likelihood == "midpoint":
        args.physics_likelihood = "midpoint_residual"
    if args.data_source == "poseidon-native":
        if args.fm != "poseidon" and not args.allow_native_transfer:
            raise SystemExit(
                "A non-Poseidon FM on Poseidon NS-Gauss is an explicit OOD "
                "transfer experiment. Pass --allow-native-transfer to declare "
                "the convention: one fixed FM output equals one dt=0.1 target.")
        dt_target = configure_native_poseidon_cadence(args)
    else:
        dt_target = configure_fm_cadence(args)
    set_seed(args.seed)
    if args.physics_resolution > 128 or 128 % args.physics_resolution:
        raise SystemExit("--physics-resolution must divide the 128x128 target grid")
    if args.n_val_traj < 0:
        raise SystemExit("--n-val-traj must be non-negative")
    fm = load_fm(args)
    if args.data_source == "poseidon-native":
        spec = native_poseidon_spec(args, fm)
        cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj,
                                         args.steps, args.lead_steps, 0)
        val = None if args.n_val_traj == 0 else load_poseidon_trajectories(
            args.fm_data_path, fm, args.n_val_traj, args.steps, args.lead_steps,
            args.n_cal_traj)
        test = None if args.physics_only_calibration_gate else load_poseidon_trajectories(
            args.fm_data_path, fm, args.n_test_traj, args.steps, args.lead_steps,
            args.n_cal_traj + args.n_val_traj)
    else:
        cal, spec = generate_fm_trajectories(args, fm, args.n_cal_traj,
                                              args.steps * args.lead_steps,
                                              seed=args.seed, preserve_physics=True)
        val = None
        if args.n_val_traj:
            val, _ = generate_fm_trajectories(args, fm, args.n_val_traj,
                                               args.steps * args.lead_steps,
                                               seed=args.seed + 5_000, preserve_physics=True)
            val = val[:, ::args.lead_steps]
        if not args.physics_only_calibration_gate:
            test, _ = generate_fm_trajectories(args, fm, args.n_test_traj,
                                               args.steps * args.lead_steps,
                                               seed=args.seed + 10_000, preserve_physics=True)
            test = test[:, ::args.lead_steps]
        cal = cal[:, ::args.lead_steps]

    if args.verify_batched:
        if args.physics_likelihood not in {"discrete_flow", "fixed_rk3_flow"}:
            raise SystemExit("--verify-batched requires a flow-endpoint physics likelihood")
        ncheck = min(2, cal.shape[0])
        states = cal[:ncheck, 0].to(fm.device, fm.dtype)
        with torch.no_grad():
            serial_pred = torch.stack([fm.predict(states[i]) for i in range(ncheck)])
            batch_pred = fm.predict(states)
        rel = float((serial_pred - batch_pred).norm() /
                    serial_pred.norm().clamp(min=1e-30))
        if rel > 1e-6:
            raise RuntimeError(f"batched Poseidon prediction mismatch: relative error {rel:.3e}")
        serial_energy = _energy(spec, states[0], fm.state_shape[0], args,
                                {"vorticity_scale": 1.0, "divergence_scale": 1.0}, fm.device)
        batch_energy = _energy(spec, states, fm.state_shape[0], args,
                               {"vorticity_scale": 1.0, "divergence_scale": 1.0}, fm.device)
        flow_rel = float((serial_energy.flow_target - batch_energy.member(0).flow_target).norm() /
                         serial_energy.flow_target.norm().clamp(min=1e-30))
        if flow_rel > 1e-12:
            raise RuntimeError(f"batched AZEBAN flow mismatch: relative error {flow_rel:.3e}")
        print(f"  batched-equivalence check passed: Poseidon={rel:.2e}, "
              f"AZEBAN={flow_rel:.2e}", flush=True)

    label = "Document-faithful S4" if args.fm == "poseidon" else "Document-faithful S4 transfer"
    print_header(f"{label}: {fm.info.name}, lead={args.lead_steps}, k={args.k}")
    print(f"  native cadence: {POSEIDON_NATIVE_RAW_STRIDE} raw frames / shared-data step; "
          f"dt={spec.dt_out:g}; model lead={getattr(fm, 'lead_time', 'fixed/transfer')}")
    if args.data_source == "poseidon-native" and args.fm != "poseidon":
        print("  transfer convention: one fixed external-FM output is scored against "
              "one Poseidon NS-Gauss transition (dt=0.1); OOD result")
    if args.physics_resolution < spec.n:
        print(f"  physics benchmark: fixed coarse AZEBAN endpoint at "
              f"{args.physics_resolution}x{args.physics_resolution}, lifted spectrally to {spec.n}x{spec.n}")
    held_out_label = "not loaded (calibration gate)" if args.physics_only_calibration_gate else str(args.n_test_traj)
    val_label = str(args.n_val_traj) if args.n_val_traj else "none (legacy selection on calibration)"
    print(f"  target dt={dt_target:g}; calibration trajectories={args.n_cal_traj}; "
          f"validation trajectories={val_label}; held-out trajectories={held_out_label}")
    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma = fit_sigma2(fm, xs, ys)
    print(f"  fitted pushforward injection sigma^2={sigma['sigma2']:.5e}")
    floor = _truth_floor_audit(cal, spec, fm.state_shape[0], args, fm.device)
    scales = _calibration_physics_scales(args, fm, cal, spec, floor)
    scale_label = "endpoint" if args.physics_likelihood in {"discrete_flow", "fixed_rk3_flow"} else "residual"
    print(f"  physics likelihood={args.physics_likelihood}; {scale_label} scale={scales['vorticity_scale']:.4g}, "
          f"divergence={scales['divergence_scale']:.4g}")
    if args.physics_likelihood == "fixed_rk3_flow":
        print(f"  fixed-RK3 endpoint: full {spec.n}x{spec.n} grid; "
              f"exactly {args.fixed_rk3_substeps} SSP-RK3 steps per transition "
              "(approximate numerical PDE solve, not future-data residual fitting)")
    print("  residual-floor audit (calibration only): "
          f"true={floor['truth_residual_median']:.4g}, "
          f"persistence={floor['persistence_residual_mean']:.4g}, "
          f"true/persistence={floor['truth_over_persistence_mean']:.3f}, "
          f"truth wins={100 * floor['truth_beats_persistence_fraction']:.1f}%")

    if args.physics_only_calibration_gate:
        print("\n  calibration-only coarse-physics gate (no held-out trajectories loaded)")
        raw_rows, _, _ = _rollout(args, fm, cal, spec, sigma["sigma2"], "isotropic",
                                  1.0, 0.0, scales)
        phys_rows, phys_diag, secs = _rollout(
            args, fm, cal, spec, sigma["sigma2"], "physics_only", float("nan"),
            0.0, scales)
        raw_score, phys_score = _score(raw_rows), _score(phys_rows)
        gain = 100 * (1 - phys_score / raw_score)
        print(f"    raw calibration rollout RMSE={raw_score:.6g}")
        print(f"    physics-only calibration rollout RMSE={phys_score:.6g}")
        print(f"    physics-only calibration gain={gain:.2f}%")
        _print_physics_only_trace(phys_diag)
        out = {
            "stage": "s4_coarse_physics_calibration_gate",
            "metadata": fm_metadata(args, fm, spec), "sigma2": sigma,
            "physics_scales": scales, "residual_floor_calibration": floor,
            "calibration": {"raw_steps": raw_rows, "physics_only_steps": phys_rows,
                            "raw_rollout_rmse": raw_score,
                            "physics_only_rollout_rmse": phys_score,
                            "physics_only_gain_pct": gain,
                            "physics_only_seconds": secs,
                            "physics_only_diagnostics": phys_diag},
            "validity": {
                "selection": "calibration-only; no held-out trajectories were loaded",
                "physics_likelihood": args.physics_likelihood,
                "physics_resolution": args.physics_resolution,
                "native_transfer": bool(args.allow_native_transfer and args.fm != "poseidon"),
            },
        }
        p = save_json(out, results_path_scale("s4_physics_gate", fm_result_key(args),
                                               "results.json", args.tag))
        print(f"\nwrote {p}")
        return

    print("\n  calibrating prior scales alpha by the document MLE")
    summaries = _calibration_summaries(args, fm, cal, sigma["sigma2"], args.methods)
    alphas, alpha_diag = {}, {}
    for method in args.methods:
        if method == "physics_only":
            alphas[method] = float("nan")
            alpha_diag[method] = {"geometry_share": float("nan"),
                                  "error_energy_in_subspace": float("nan")}
            print(f"    {method:13s} no prior scale (physics-only control)")
            continue
        alphas[method], alpha_diag[method] = fit_alpha_scale(
            summaries[method], objective=args.alpha_objective,
            return_diagnostics=True, seed=args.seed)
        share = alpha_diag[method]["geometry_share"]
        share_text = (f"{share:.3g}" if np.isfinite(share)
                      else "N/A (signed GN)")
        print(f"    {method:13s} alpha={alphas[method]:.4g} "
              f"geometry-share={share_text} "
              f"error-in-rank={alpha_diag[method]['error_energy_in_subspace']:.3g}")

    selected, sweeps = {}, {}
    for method in args.methods:
        if method == "physics_only":
            selected[method], sweeps[method] = 0.0, []
            continue
        selection_traj = val if val is not None else cal
        selection_label = "validation" if val is not None else "calibration (legacy)"
        print(f"\n  selecting lambda for {method} on {selection_label} rollouts")
        cells = []
        for lam in args.lambda_grid:
            print(f"    evaluating lambda/ref={lam:g} ...", flush=True)
            rows, _, secs = _rollout(args, fm, selection_traj, spec, sigma["sigma2"], method,
                                     alphas[method], lam, scales)
            cell = {"lambda_rel": lam, "rmse": _score(rows), "seconds": secs}
            cells.append(cell)
            print(f"    lambda/ref={lam:g}: mean RMSE={cell['rmse']:.6g}")
        best = min(cells, key=lambda x: x["rmse"])
        selected[method], sweeps[method] = best["lambda_rel"], cells
        print(f"    selected {selected[method]:g} x reference")

    print("\n  held-out rollout evaluation")
    final, diagnostics = {}, {}
    for method in args.methods:
        rows, diag, secs = _rollout(args, fm, test, spec, sigma["sigma2"], method,
                                    alphas[method], selected[method], scales,
                                    trace=args.trace_posterior)
        final[method] = {"steps": rows, "mean_rollout_rmse": _score(rows), "seconds": secs}
        diagnostics[method] = diag
        if args.trace_posterior:
            _print_posterior_trace(method, diag, args.steps)
        if method == "physics_only":
            _print_physics_only_trace(diag)
    raw_rows, _, _ = _rollout(args, fm, test, spec, sigma["sigma2"], "isotropic",
                              1.0, 0.0, scales)
    raw_score = _score(raw_rows)
    # Do not print nonexistent lead columns as NaN on a deliberately short
    # smoke run.  The saved per-step JSON remains complete in either case.
    headers = ["method", "RMSE", "gain%"]
    headers += [f"step{t} gain%" for t in range(2, len(raw_rows) + 1)]
    table = Table(*headers)
    table.add("raw", raw_score, 0, *([0] * (len(raw_rows) - 1)))
    for method in args.methods:
        rows = final[method]["steps"]
        gains = [100 * (1 - r["rmse"] / raw_rows[t]["rmse"]) for t, r in enumerate(rows)]
        table.add(method, final[method]["mean_rollout_rmse"],
                  100 * (1 - final[method]["mean_rollout_rmse"] / raw_score),
                  *gains[1:])
    print(table)

    out = {"stage": "s4_document_faithful", "metadata": fm_metadata(args, fm, spec),
           "sigma2": sigma, "alpha": alphas, "alpha_diagnostics": alpha_diag,
           "selected_lambda_rel": selected, "calibration_sweep": sweeps,
           "physics_scales": scales, "residual_floor_calibration": floor,
           "test": final, "diagnostics": diagnostics,
           "validity": {"alpha": "fitted by document MLE on calibration raw rollouts",
                        "lambda_selection": ("trajectory-disjoint validation rollouts"
                                             if val is not None else "calibration rollouts (legacy protocol)"),
                        "n_calibration_trajectories": args.n_cal_traj,
                        "n_validation_trajectories": args.n_val_traj,
                        "n_test_trajectories": args.n_test_traj,
                        "methods": args.methods,
                        "physics_likelihood": args.physics_likelihood,
                        "physics_resolution": args.physics_resolution,
                        "native_transfer": bool(args.allow_native_transfer and args.fm != "poseidon"),
                        "native_flow_execution": (
                            "serial_reference" if args.serial_native_flow else "batched_per_trajectory_cfl"),
                        "step1": "known initial state; pushforward is necessarily isotropic",
                        "test_split": "trajectory-disjoint from calibration"}}
    p = save_json(out, results_path_scale("s4_document", fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
