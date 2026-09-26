#!/usr/bin/env python3
"""S4: exact coarse-resolved physics assimilation for a frozen Poseidon FM."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    configure_native_poseidon_cadence, fm_metadata, fm_result_key,
    load_poseidon_trajectories, native_poseidon_spec)
from experiments.scale.s4_fm_document_posterior import _calibration_summaries
from experiments.scale.s4_fm_physics_correction import _aggregate, _one_metrics
from hipp.scale.calibrate_scale import fit_alpha_scale
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import ResolvedCoarseFlowEnergy2D
from hipp.scale.lowrank import LowRankGaussian
from hipp.scale.posterior_scale import covariance_matvec
from hipp.scale.rollout import fit_sigma2, propagate_step
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("isotropic", "pushforward", "coarse_replace")


def _center(prior, mean):
    return LowRankGaussian(mean.double(), prior.U, prior.d, prior.tau,
                           alpha=prior.alpha, label=prior.label)


def _cosine(a, b):
    return float((a.reshape(-1) @ b.reshape(-1)) /
                 (a.norm() * b.norm()).clamp(min=1e-300))


def _energy(spec, previous, channels, args, discrepancy, device):
    return ResolvedCoarseFlowEnergy2D(
        spec, previous, channels, physics_n=args.physics_resolution,
        flow_cfl=args.flow_cfl, dt=max(1, args.lead_steps) * spec.dt_out,
        divergence_weight=0.0, discrepancy_bias=discrepancy["bias"],
        discrepancy_covariance=discrepancy["covariance"], device=device)


def _fit_discrepancy(args, calibration, spec, channels, device):
    """Fit b and R from fit trajectories only; no validation/test state enters."""
    residuals, solver_steps = [], []
    for i in range(calibration.shape[0]):
        for t in range(args.steps):
            energy = ResolvedCoarseFlowEnergy2D(
                spec, calibration[i, t].to(device), channels,
                physics_n=args.physics_resolution, flow_cfl=args.flow_cfl,
                dt=max(1, args.lead_steps) * spec.dt_out, divergence_weight=0.0,
                device=device)
            truth_low = energy.restrict(calibration[i, t + 1].to(device))[0]
            residuals.append((truth_low - energy.low_flow_target[0]).detach())
            solver_steps.append(float(energy.flow_internal_steps))
    D = torch.stack(residuals).double()
    bias = D.mean(0)
    centered = D - bias
    m = D.shape[1]
    # A fully empirical R needs far more samples than dimensions.  Shrinking
    # toward its diagonal is a predeclared, positive-definite compromise.
    if args.discrepancy_covariance == "diagonal":
        R = torch.diag(centered.square().mean(0))
    else:
        denom = max(D.shape[0] - 1, 1)
        sample = centered.T @ centered / denom
        diagonal = torch.diag(torch.diag(sample))
        R = ((1.0 - args.discrepancy_shrink) * sample +
             args.discrepancy_shrink * diagonal)
    scale = float(torch.diag(R).mean().clamp(min=1e-30))
    ridge = args.discrepancy_ridge_rel * scale
    R = 0.5 * (R + R.T) + ridge * torch.eye(m, dtype=R.dtype, device=R.device)
    chol = torch.linalg.cholesky(R)
    return {
        "bias": bias, "covariance": R, "n_samples": int(D.shape[0]),
        "bias_rms": float(bias.square().mean().sqrt()),
        "residual_rms": float(D.square().mean().sqrt()),
        "centered_rms": float(centered.square().mean().sqrt()),
        "covariance_trace_per_dim": float(torch.diag(R).mean()),
        "covariance_condition": float(torch.linalg.cond(R)),
        "ridge": float(ridge), "shrink": float(args.discrepancy_shrink),
        "kind": args.discrepancy_covariance,
        "flow_steps_mean": float(np.mean(solver_steps)),
        "_chol": chol,
    }


def _prior(method, fm, c, raw, sigma2, alpha, pf_state, args, generator):
    if method == "isotropic":
        return LowRankGaussian.isotropic(raw, tau=1.0, alpha=alpha, label="isotropic")
    if pf_state is None:
        return LowRankGaussian.isotropic(raw, tau=alpha * sigma2, label="pushforward-step1")
    return propagate_step(fm, c, pf_state, alpha * sigma2, k=args.k,
                          n_sketch=args.n_sketch, n_probe=args.n_probe,
                          chunk=args.chunk, generator=generator)[0]


def _exact_resolved_update(prior, energy, lambda_actual, *, condition_covariance=False):
    """Exact mean, and optionally exact covariance, for the Gaussian update.

    H is the actual FFT restriction implemented by ``energy``.  No assumed
    prolongation adjoint or optimizer approximation appears here.
    """
    if lambda_actual <= 0:
        return prior.mean.clone(), prior, {"lambda": float(lambda_actual), "exact": True}
    ht_rows, _ = energy.restriction_adjoint_and_gram()  # rows are H^T e_i
    # V rows are (Sigma H^T e_i)^T, so H V^T is H Sigma H^T.
    v_rows = covariance_matvec(prior, ht_rows)
    hsh = energy.restrict(v_rows).T
    hsh = 0.5 * (hsh + hsh.T)
    W = hsh + energy.discrepancy_covariance / float(lambda_actual)
    W = 0.5 * (W + W.T)
    chol = torch.linalg.cholesky(W)
    innovation = (energy.observation_target - energy.restrict(prior.mean))[0]
    weights = torch.cholesky_solve(innovation[:, None], chol)[:, 0]
    corrected = prior.mean + weights @ v_rows
    info = {
        "lambda": float(lambda_actual), "exact": True,
        "observation_dim": energy.observation_dim,
        "innovation_rms": float(innovation.square().mean().sqrt()),
        "resolved_system_condition": float(torch.linalg.cond(W)),
    }
    if not condition_covariance:
        return corrected, _center(prior, corrected), info

    # Sigma+ = Sigma - V W^-1 V^T.  Re-express the signed low-rank update
    # exactly as tau I + Q diag(d) Q^T so propagate_step can consume it.
    Winv = torch.cholesky_inverse(chol)
    V = v_rows.T
    B = torch.cat((prior.U.double(), V.double()), dim=1)
    Q, Rq = torch.linalg.qr(B, mode="reduced")
    k = prior.k
    C = torch.zeros(k + energy.observation_dim, k + energy.observation_dim,
                    dtype=torch.float64, device=Q.device)
    if k:
        C[:k, :k] = torch.diag(prior.alpha * prior.d.double())
    C[k:, k:] = -Winv.double()
    small = Rq @ C @ Rq.T
    d, Vsmall = torch.linalg.eigh(0.5 * (small + small.T))
    Uplus = Q @ Vsmall
    tau = prior.alpha * prior.tau
    min_eig = float((d + tau).min())
    if min_eig <= 0:
        raise RuntimeError(f"conditioned covariance lost PSD: min eigenvalue {min_eig:.3e}")
    posterior = LowRankGaussian(corrected.double(), Uplus, d, tau,
                                 alpha=1.0, label=f"{prior.label}-conditioned")
    info["posterior_rank"] = posterior.k
    info["posterior_trace"] = posterior.trace()
    return corrected, posterior, info


def _coarse_replace(raw, energy):
    """Hard low-mode replacement while retaining the raw forecast elsewhere."""
    ht_rows, hht = energy.restriction_adjoint_and_gram()
    delta = (energy.observation_target - energy.restrict(raw))[0]
    weights = torch.linalg.solve(hht, delta)
    return raw + weights @ ht_rows


def _rollout(args, fm, truth, spec, sigma2, discrepancy, method, alpha, lam):
    rows_by_step = [[] for _ in range(args.steps)]
    traces = []
    start = time.time()
    for i in range(truth.shape[0]):
        c, pf_state = truth[i, 0].to(fm.device, fm.dtype), None
        for t in range(args.steps):
            raw = fm.predict(c).double()
            energy = _energy(spec, c, fm.state_shape[0], args, discrepancy, fm.device)
            truth_energy = _energy(spec, truth[i, t].to(fm.device), fm.state_shape[0],
                                   args, discrepancy, fm.device)
            if method == "coarse_replace":
                corrected = _coarse_replace(raw, energy)
                posterior = None
                info = {"lambda": float("inf"), "exact": True}
            else:
                gen = torch.Generator(device="cpu").manual_seed(args.seed + 90001*i + 1009*t)
                prior = _prior(method, fm, c, raw, sigma2, alpha, pf_state, args, gen)
                corrected, posterior, info = _exact_resolved_update(
                    prior, energy, lam,
                    condition_covariance=(method == "pushforward" and
                                          args.resolved_covariance_update == "conditioned"))
            metrics = _one_metrics(corrected, raw, truth[i, t + 1], energy, truth_energy,
                                   prior if method != "coarse_replace" else None,
                                   same_conditioning_state=(t == 0))
            low_error = energy.restrict(truth[i, t + 1].to(fm.device)) - energy.restrict(raw)
            low_direction = energy.observation_target - energy.restrict(raw)
            metrics.update({
                "raw_resolved_rmse": float(low_error.square().mean().sqrt()),
                "resolved_direction_cosine": _cosine(low_direction, low_error),
                "resolved_energy_raw": float(energy.energy(raw)[0]),
                "resolved_energy_corrected": float(energy.energy(corrected)[0]),
            })
            rows_by_step[t].append(metrics)
            traces.append({"trajectory": i, "step": t + 1, **info, **metrics})
            c = corrected.to(fm.dtype)
            if method == "pushforward":
                pf_state = posterior
    return [_aggregate(r) for r in rows_by_step], traces, time.time() - start


def _alpha_summaries(args, fm, cal, sigma2, methods):
    # This reuses the established document alpha-MLE construction.  It does
    # not see the new physics likelihood or any validation/test trajectory.
    return _calibration_summaries(args, fm, cal, sigma2, methods)


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=16)
    ap.add_argument("--n-val-traj", type=int, default=8)
    ap.add_argument("--n-test-traj", type=int, default=16)
    ap.add_argument("--physics-resolution", type=int, default=8)
    ap.add_argument("--flow-cfl", type=float, default=0.5)
    ap.add_argument("--methods", nargs="+", choices=METHODS,
                    default=["isotropic", "pushforward", "coarse_replace"])
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    ap.add_argument("--discrepancy-covariance", choices=["diagonal", "full_shrink"],
                    default="diagonal")
    ap.add_argument("--discrepancy-shrink", type=float, default=0.75)
    ap.add_argument("--discrepancy-ridge-rel", type=float, default=1e-4)
    ap.add_argument("--resolved-covariance-update", choices=["frozen", "conditioned"],
                    default="frozen")
    ap.add_argument("--n-sketch", type=int, default=None)
    ap.add_argument("--n-probe", type=int, default=16)
    ap.add_argument("--n-tail", type=int, default=32)
    ap.add_argument("--data-source", choices=["poseidon-native"], default="poseidon-native")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.fm != "poseidon":
        raise SystemExit("resolved coarse-physics S4 is currently validated only for Poseidon velocity")
    if args.physics_resolution >= 128 or 128 % args.physics_resolution:
        raise SystemExit("--physics-resolution must divide 128 and be below 128")
    if not 0 <= args.discrepancy_shrink <= 1:
        raise SystemExit("--discrepancy-shrink must lie in [0,1]")
    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj,
                                     args.steps, args.lead_steps, offset=0)
    val = load_poseidon_trajectories(args.fm_data_path, fm, args.n_val_traj,
                                     args.steps, args.lead_steps, offset=args.n_cal_traj)
    test = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj,
                                      args.steps, args.lead_steps,
                                      offset=args.n_cal_traj + args.n_val_traj)

    print_header(f"S4 resolved coarse physics: {fm.info.name}, h={args.physics_resolution}, k={args.k}")
    print(f"  split: fit={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")
    print("  likelihood: resolved coarse AZEBAN endpoint + calibration-only bias/covariance")
    print(f"  covariance update: {args.resolved_covariance_update}; lambda is an actual observation precision")
    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma = fit_sigma2(fm, xs, ys)
    discrepancy = _fit_discrepancy(args, cal, spec, fm.state_shape[0], fm.device)
    print(f"  fitted sigma²={sigma['sigma2']:.4e}; coarse discrepancy samples={discrepancy['n_samples']}")
    print(f"  bias RMS={discrepancy['bias_rms']:.4e}; centered RMS={discrepancy['centered_rms']:.4e}; "
          f"R condition={discrepancy['covariance_condition']:.3g}")

    active_priors = [m for m in args.methods if m != "coarse_replace"]
    summaries = _alpha_summaries(args, fm, cal, sigma["sigma2"], active_priors)
    alphas = {}
    for method in active_priors:
        alphas[method], diag = fit_alpha_scale(summaries[method], objective=args.alpha_objective,
                                               return_diagnostics=True, seed=args.seed)
        print(f"  alpha {method}={alphas[method]:.5g}; error-in-rank={diag['error_energy_in_subspace']:.3g}")
    alphas["coarse_replace"] = float("nan")

    selected, sweeps = {}, {}
    for method in args.methods:
        if method == "coarse_replace":
            selected[method], sweeps[method] = float("inf"), []
            continue
        print(f"\n  selecting actual lambda for {method} on validation")
        cells = []
        for lam in args.lambda_grid:
            rows, _, secs = _rollout(args, fm, val, spec, sigma["sigma2"], discrepancy,
                                     method, alphas[method], lam)
            cell = {"lambda": float(lam), "rmse": float(np.mean([r["rmse"] for r in rows])),
                    "seconds": secs}
            cells.append(cell)
            print(f"    lambda={lam:g}: RMSE={cell['rmse']:.6g}")
        best = min(cells, key=lambda x: x["rmse"])
        selected[method], sweeps[method] = best["lambda"], cells
        print(f"    selected lambda={best['lambda']:g}")

    print("\n  held-out evaluation")
    final, diagnostics = {}, {}
    for method in args.methods:
        rows, diag, secs = _rollout(args, fm, test, spec, sigma["sigma2"], discrepancy,
                                    method, alphas[method], selected[method])
        final[method] = {"steps": rows, "mean_rollout_rmse": float(np.mean([r["rmse"] for r in rows])),
                         "seconds": secs}
        diagnostics[method] = diag
    raw_rows, _, _ = _rollout(args, fm, test, spec, sigma["sigma2"], discrepancy,
                               "isotropic", alphas["isotropic"], 0.0)
    raw = float(np.mean([r["rmse"] for r in raw_rows]))
    table = Table("method", "RMSE", "gain%", "resolved dir cosine")
    table.add("raw", raw, 0.0, float(np.mean([r["resolved_direction_cosine"] for r in raw_rows])))
    for method in args.methods:
        val = final[method]["mean_rollout_rmse"]
        cosine = float(np.mean([r["resolved_direction_cosine"] for r in final[method]["steps"]]))
        table.add(method, val, 100 * (1 - val / raw), cosine)
    print(table)
    out = {
        "stage": "s4_resolved_coarse_physics", "metadata": fm_metadata(args, fm, spec),
        "sigma2": sigma, "discrepancy": {k: v for k, v in discrepancy.items() if not k.startswith("_")},
        "alphas": alphas, "lambda_sweeps_validation": sweeps, "selected_lambda": selected,
        "raw": {"steps": raw_rows, "mean_rollout_rmse": raw},
        "held_out": final, "diagnostics": diagnostics,
        "validity": {"fit_validation_test_trajectory_disjoint": True,
                     "coarse_physics_resolution": args.physics_resolution,
                     "future_truth_used_for_inference": False,
                     "observation_space": "resolved coarse velocity modes only"},
    }
    p = save_json(out, results_path_scale("s4_resolved_physics", fm_result_key(args),
                                          "results.json", args.tag))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
