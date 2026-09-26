#!/usr/bin/env python3
"""FM S3 -- rollout uncertainty growth for Poseidon/DPOT."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    POSEIDON_NATIVE_RAW_STRIDE,
    configure_fm_cadence,
    configure_native_poseidon_cadence,
    draw_pairs,
    fm_metadata,
    fm_result_key,
    generate_fm_trajectories,
    load_poseidon_trajectories,
    native_poseidon_spec,
)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.jacobian import set_probe_precision
from hipp.scale.lowrank import LowRankGaussian
from hipp.scale.rollout import (
    aggregate_rollout,
    constant_covariance_rollout,
    fit_lowrank_innovation,
    fit_spectral_divfree_innovation,
    fit_sigma2,
    propagate_rollout,
    propagate_rollouts_batched,
    residual_temporal_diagnostic,
    score_rollout,
    shrink_innovation,
)
from hipp.utils import Table, print_header, save_json, set_seed


def _pushforward_prior0(args, fm, c, sigma2):
    from hipp.scale.curvature_scale import estimate_scale

    est = estimate_scale(
        fm, c, method="pushforward", k=args.k, tau_rel=args.tau_rel,
        oversample=args.oversample, n_iter=args.n_iter, chunk=args.chunk,
        n_tail=args.n_tail,
    )
    return est.prior.rescaled(sigma2 * fm.N / max(est.prior.trace(), 1e-300))


def _lyapunov_estimate(fm, c0, steps: int, seed: int) -> float:
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = c0.reshape(-1).to(fm.device, fm.dtype)
    d = torch.randn(a.numel(), generator=g, dtype=a.dtype).to(a.device)
    d = d / d.norm().clamp(min=1e-30) * 1e-4 * a.norm().clamp(min=1e-30)
    b = a + d
    r0 = float((b - a).norm())
    rates = []
    for _ in range(max(1, steps)):
        with torch.no_grad():
            a = fm.predict(a)
            b = fm.predict(b)
        r = float((b - a).norm())
        rates.append(math.log(max(r, 1e-30) / max(r0, 1e-30)))
        b = a + (b - a) / max(r, 1e-30) * r0
    return float(np.mean(np.diff([0.0] + rates))) if rates else float("nan")


def _select_innovation_shrinkage(args, fm, truth, sigma2, innovation):
    """Select Q shrinkage on calibration-validation trajectories only."""
    cells = []
    print("  selecting structured-Q shrinkage on calibration-validation rollouts")
    for rho in args.innovation_shrink_grid:
        q = shrink_innovation(innovation, rho)
        nlls, ratio_losses = [], []
        for i in range(truth.shape[0]):
            c0 = truth[i, 0].to(fm.device, fm.dtype)
            g = torch.Generator(device="cpu").manual_seed(args.seed + 70_000 + i)
            tr = propagate_rollout(
                fm, c0, args.steps, sigma2, k=args.k, n_sketch=args.n_sketch,
                n_probe=args.n_tail, chunk=args.chunk, generator=g,
                complement="nystrom", keep_priors=True, innovation=q)
            rows = score_rollout(tr, truth[i, :args.steps + 1])
            for p, y, row in zip(tr.priors, truth[i, 1:args.steps + 1], rows):
                nlls.append(float(-p.log_prob(y) / p.N))
                ratio_losses.append(abs(float(np.log(max(row["ratio"], 1e-30)))))
        cell = {"rho": float(rho), "joint_nll_per_dim": float(np.mean(nlls)),
                "mean_abs_log_ratio": float(np.mean(ratio_losses))}
        cells.append(cell)
        print(f"    rho={rho:g}: validation NLL/dim={cell['joint_nll_per_dim']:.6g}; "
              f"|log spread ratio|={cell['mean_abs_log_ratio']:.4g}")
    best = min(cells, key=lambda z: z["joint_nll_per_dim"])
    print(f"    selected rho={best['rho']:g} by validation joint NLL", flush=True)
    return shrink_innovation(innovation, best["rho"]), cells, best


def _propagate_known_initial_batch(args, fm, truth, steps, sigma2, mode, innovation):
    """Run S3 rollouts, optionally grouping independent trajectories.

    ``trajectory_batch=1`` is the long-standing reference implementation.  A
    larger value merely schedules independent local Jacobian products together;
    it never shares a covariance, a probe, or a calibration target across
    trajectories.  The pushforward-initial-prior ablation stays serial because
    its initial randomized SVD is still state-local.
    """
    traces = []
    width = int(args.trajectory_batch)
    use_batch = width > 1 and args.prior0 == "known"
    if use_batch:
        if not getattr(fm.module, "_hipp_batch_safe", False):
            raise SystemExit(
                "--trajectory-batch requires an adapter with a verified batch-safe "
                "forward contract; use --trajectory-batch 1 for this FM.")
        print(f"    trajectory-batched AD enabled: groups of up to {width} independent states")
    for lo in range(0, truth.shape[0], width if use_batch else 1):
        hi = min(truth.shape[0], lo + (width if use_batch else 1))
        if use_batch and hi - lo > 1:
            gs = [torch.Generator(device="cpu").manual_seed(args.seed + 1000 + i)
                  for i in range(lo, hi)]
            traces.extend(propagate_rollouts_batched(
                fm, truth[lo:hi, 0].to(fm.device, fm.dtype), steps, sigma2,
                k=args.k, n_sketch=args.n_sketch, n_probe=args.n_tail,
                chunk=args.chunk, generators=gs, complement=mode,
                keep_priors=True, innovation=innovation))
            continue
        # A one-member tail intentionally uses the established serial path.
        i = lo
        c0 = truth[i, 0].to(fm.device, fm.dtype)
        p0 = None
        if args.prior0 == "pushforward":
            p0 = _pushforward_prior0(args, fm, c0, sigma2)
        g = torch.Generator(device="cpu").manual_seed(args.seed + 1000 + i)
        traces.append(propagate_rollout(
            fm, c0, steps, sigma2, k=args.k, prior0=p0,
            n_sketch=args.n_sketch, chunk=args.chunk, generator=g,
            complement=mode, keep_priors=True, progress=False,
            innovation=innovation))
    return traces


def _verify_trajectory_batch(args, fm, truth, steps, sigma2, mode, innovation):
    """Fail closed unless batched and serial covariance recursions agree."""
    if args.trajectory_batch <= 1:
        raise SystemExit("--verify-trajectory-batch requires --trajectory-batch > 1")
    if args.prior0 != "known":
        raise SystemExit("trajectory batching verifies only the known-initial-state S3 protocol")
    if truth.shape[0] < 2:
        raise SystemExit("trajectory-batch verification needs at least two trajectories")
    n = min(int(args.trajectory_batch), 2, truth.shape[0])
    serial = []
    for i in range(n):
        g = torch.Generator(device="cpu").manual_seed(args.seed + 1000 + i)
        serial.append(propagate_rollout(
            fm, truth[i, 0].to(fm.device, fm.dtype), steps, sigma2,
            k=args.k, n_sketch=args.n_sketch, n_probe=args.n_tail,
            chunk=args.chunk, generator=g, complement=mode,
            keep_priors=True, innovation=innovation))
    gs = [torch.Generator(device="cpu").manual_seed(args.seed + 1000 + i)
          for i in range(n)]
    batched = propagate_rollouts_batched(
        fm, truth[:n, 0].to(fm.device, fm.dtype), steps, sigma2,
        k=args.k, n_sketch=args.n_sketch, n_probe=args.n_tail,
        chunk=args.chunk, generators=gs, complement=mode,
        keep_priors=True, innovation=innovation)
    max_mean, max_cov = 0.0, 0.0
    probe = torch.linspace(-1.0, 1.0, fm.N, dtype=torch.float64, device=fm.device)
    from hipp.scale.posterior_scale import covariance_matvec
    for a, b in zip(serial, batched):
        for ma, mb, pa, pb in zip(a.means, b.means, a.priors, b.priors):
            max_mean = max(max_mean, float((ma.double() - mb.double()).abs().max()))
            max_cov = max(max_cov, float((covariance_matvec(pa, probe) -
                                          covariance_matvec(pb, probe)).abs().max()))
    tol = 2e-6
    if max_mean > tol or max_cov > tol:
        raise RuntimeError(
            f"trajectory-batch equivalence failed: max mean={max_mean:.3e}, "
            f"max covariance-action={max_cov:.3e}, tolerance={tol:.1e}")
    print(f"  trajectory-batch equivalence passed: max mean={max_mean:.2e}, "
          f"max covariance-action={max_cov:.2e}")


def main() -> int:
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=4,
                    help="autoregressive rollout length")
    ap.add_argument("--n-traj", type=int, default=4,
                    help="test trajectories to score")
    ap.add_argument("--n-cal-traj", type=int, default=8,
                    help="trajectories used to fit sigma^2")
    ap.add_argument("--lead-steps", type=int, default=1,
                    help="truth spacing in generated trajectory snapshots")
    ap.add_argument("--sim-batch", type=int, default=2,
                    help="trajectory simulation batch size")
    ap.add_argument("--trajectory-batch", type=int, default=1,
                    help="optional number of independent known-initial-state rollouts "
                         "whose JVP/VJP calls are scheduled together; 1 is serial reference")
    ap.add_argument("--verify-trajectory-batch", action="store_true",
                    help="compare two serial rollouts to the optional batched AD path and fail on mismatch")
    ap.add_argument("--n-tail", type=int, default=8,
                    help="Hutchinson probes for discarded Jacobian trace")
    ap.add_argument("--n-sketch", type=int, default=None,
                    help="complement sketch width; defaults to k+10")
    ap.add_argument("--complements", nargs="+", default=["nystrom", "isotropic", "none"],
                    help="rollout propagation ablations")
    ap.add_argument("--sigma2", type=float, default=None,
                    help="override fitted per-step residual variance")
    ap.add_argument("--innovation", choices=["isotropic", "empirical_lowrank", "spectral_divfree"],
                    default="isotropic",
                    help="one-step Q: scalar sigma^2 I, raw empirical low-rank Q, or centered divergence-free spectral Q")
    ap.add_argument("--innovation-rank", type=int, default=16,
                    help="retained rank for a structured innovation Q")
    ap.add_argument("--innovation-validation-traj", type=int, default=0,
                    help="trajectory-disjoint subset of calibration trajectories used only to select Q shrinkage")
    ap.add_argument("--innovation-shrink-grid", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="trace-preserving Q shrinkage candidates; requires --innovation-validation-traj > 0")
    ap.add_argument("--prior0", default="known",
                    choices=["known", "pushforward"],
                    help="known uses Sigma_0=0; pushforward is an ablation with "
                         "a pre-existing uncertain initial state")
    ap.add_argument("--data-source", choices=["synthetic", "poseidon-native"],
                    default="synthetic")
    ap.add_argument("--fm-data-path", default=None,
                    help="NetCDF/HDF5 file for --data-source poseidon-native")
    args = ap.parse_args()

    if args.data_source == "poseidon-native":
        if not args.fm_data_path:
            raise SystemExit("poseidon-native requires --fm-data-path")
        dt_target = configure_native_poseidon_cadence(args)
    else:
        dt_target = configure_fm_cadence(args)
    if args.fm not in ("poseidon", "dpot", "morph"):
        raise SystemExit("FM S3 currently supports --fm poseidon, --fm dpot, or --fm morph.")
    if args.trajectory_batch < 1:
        raise SystemExit("--trajectory-batch must be positive")

    set_seed(args.seed)
    set_probe_precision(high=not args.allow_tf32)
    fm = load_fm(args)

    print_header(f"FM S3: {fm.info.name}  channels={args.fm_channels}  "
                 f"lead_steps={args.lead_steps}  k={args.k}  steps={args.steps}")
    print(f"  model      {fm}")
    if hasattr(fm, "dpot_history_policy"):
        print(f"  DPOT note  history policy = {fm.dpot_history_policy}")
    if hasattr(fm, "morph_normalization_policy"):
        print(f"  MORPH note normalization policy = {fm.morph_normalization_policy}; "
              "S3 is a geometry gate until a matched MORPH trajectory protocol is loaded")
    if args.data_source == "poseidon-native" and args.fm != "poseidon":
        print("  cadence    transfer convention: one fixed-step external-FM output "
              "is scored against one native dt=0.1 transition")

    lead = max(1, int(args.lead_steps))
    cal_steps = max(args.steps * lead, lead + 1)
    test_steps = args.steps * lead

    if args.data_source == "poseidon-native":
        spec = native_poseidon_spec(args, fm)
        print(f"  native cadence: {POSEIDON_NATIVE_RAW_STRIDE} raw frames / "
              f"shared-data step; dt={spec.dt_out:g}; model lead="
              f"{getattr(fm, 'lead_time', 'fixed/transfer')}")
        cal_traj = load_poseidon_trajectories(
            args.fm_data_path, fm, args.n_cal_traj, cal_steps, lead_steps=1,
            offset=0)
        args.trajectory_scaling = "native-physical-none"
    else:
        cal_traj, spec = generate_fm_trajectories(
            args, fm, args.n_cal_traj, cal_steps, seed=args.seed,
            preserve_physics=True)
    x_cal, y_cal = draw_pairs(
        cal_traj, min(args.n_cal, args.n_cal_traj * (cal_steps + 1 - lead)),
        lead, seed=args.seed + 101,
    )
    if args.innovation_validation_traj < 0 or args.innovation_validation_traj >= args.n_cal_traj:
        raise SystemExit("--innovation-validation-traj must be in [0, n-cal-traj)")
    n_fit_traj = args.n_cal_traj - args.innovation_validation_traj
    if args.innovation_validation_traj:
        fit_traj = cal_traj[:n_fit_traj]
        x_fit, y_fit = draw_pairs(
            fit_traj, min(args.n_cal, n_fit_traj * (cal_steps + 1 - lead)),
            lead, seed=args.seed + 101)
    else:
        x_fit, y_fit = x_cal, y_cal
    if args.sigma2 is None:
        sigma = fit_sigma2(fm, x_fit, y_fit)
        sigma2 = sigma["sigma2"]
    else:
        sigma2 = args.sigma2
        sigma = {"sigma2": sigma2, "note": "supplied on command line"}
    print(f"  target     {spec.name}, grid={spec.n}, stride={spec.stride}")
    innovation, innovation_selection = None, None
    temporal = residual_temporal_diagnostic(fm, fit_traj if args.innovation_validation_traj else cal_traj)
    if temporal["available"]:
        print("  calibration residual temporal gate: "
              f"lag-1 AR={temporal['lag1_ar_coefficient']:+.3f}, "
              f"mean cosine={temporal['lag1_cosine_mean']:+.3f} "
              f"({temporal['pairs']} adjacent pairs)")
    if args.innovation in ("empirical_lowrank", "spectral_divfree"):
        if args.innovation == "empirical_lowrank":
            innovation, q_diag = fit_lowrank_innovation(
                fm, x_fit, y_fit, rank=args.innovation_rank)
            label = "empirical low-rank"
        else:
            innovation, q_diag = fit_spectral_divfree_innovation(
                fm, x_fit, y_fit, rank=args.innovation_rank)
            label = "centered divergence-free spectral"
        sigma["structured_Q"] = q_diag
        print(f"  innovation {label} Q: rank={q_diag['rank']}, "
              f"trace share={q_diag['captured_trace_fraction']:.3f}")
        if q_diag.get("centered"):
            print(f"    calibration residual-mean RMS={q_diag['mean_residual_rms']:.4e} "
                  "(reported as bias; not inserted into Q)")
        if args.innovation_validation_traj:
            validation_truth = cal_traj[n_fit_traj:, ::lead, :].double()
            innovation, cells, chosen = _select_innovation_shrinkage(
                args, fm, validation_truth, sigma2, innovation)
            innovation_selection = {"fit_trajectories": n_fit_traj,
                                    "validation_trajectories": args.innovation_validation_traj,
                                    "candidates": cells, "selected": chosen}
    else:
        print(f"  sigma^2    {sigma2:.4e}")

    if args.data_source == "poseidon-native":
        test_traj = load_poseidon_trajectories(
            args.fm_data_path, fm, args.n_traj, test_steps, lead_steps=1,
            offset=args.n_cal_traj)
    else:
        test_traj, _ = generate_fm_trajectories(
            args, fm, args.n_traj, test_steps, seed=args.seed + 10000,
            preserve_physics=True)
    truth = test_traj[:, ::lead, :].double()
    steps = min(args.steps, truth.shape[1] - 1)
    print(f"  rollouts   {truth.shape[0]} trajectories x {steps} steps")

    lyap = [_lyapunov_estimate(fm, truth[i, 0], min(steps, 5), args.seed + i)
            for i in range(min(4, truth.shape[0]))]
    print(f"  model finite-time growth ≈ {np.nanmean(lyap):+.4f} per step")

    all_scores, controls, info = {}, {}, {}
    for mode in args.complements:
        print(f"\n  complement mode: {mode}", flush=True)
        if args.verify_trajectory_batch and mode == args.complements[0]:
            _verify_trajectory_batch(args, fm, truth, steps, sigma2, mode, innovation)
        traces = _propagate_known_initial_batch(
            args, fm, truth, steps, sigma2, mode, innovation)
        scores, infos = [], []
        for i, tr in enumerate(traces):
            scores.append(score_rollout(tr, truth[i, :steps + 1]))
            infos.append(tr.info)
            if mode == args.complements[0] and i == 0:
                pc = LowRankGaussian.isotropic(
                    tr.means[0].double(), tau=sigma2, label="constant")
                controls["constant"] = constant_covariance_rollout(
                    pc, tr.means, truth[i, :steps + 1], growth="none")
                controls["linear_in_t"] = constant_covariance_rollout(
                    pc, tr.means, truth[i, :steps + 1], growth="sqrt_t")
        all_scores[mode] = aggregate_rollout(scores)
        info[mode] = infos[0] if infos else []

    main_mode = args.complements[0]
    print(f"\nS3 summary [{main_mode}]")
    table = Table("step", "rmse", "pred_rms", "ratio", "D^2/N", "z",
                  "err-in-rank/kN")
    for row in all_scores[main_mode]:
        table.add(row["step"], row["rmse"], row["pred_rms"], row["ratio"],
                  row["maha_over_N"], row["z"],
                  row["frac_err_in_rank_over_isotropic"])
    print(table)
    print("  ratio≈1 means predicted spread matches actual rollout error growth.")

    if len(args.complements) > 1:
        print("\nAblation ratios")
        table = Table("mode", *[f"t={r['step']}" for r in all_scores[main_mode]])
        for mode, rows in all_scores.items():
            table.add(mode, *[f"{r['ratio']:.3f}" for r in rows])
        print(table)

    out = {
        "stage": "s3_fm",
        "metadata": fm_metadata(args, fm, spec),
        "k": args.k,
        "tau_rel": args.tau_rel,
        "lead_steps": args.lead_steps,
        "steps": steps,
        "n_traj": int(truth.shape[0]),
        "trajectory_batch": int(args.trajectory_batch),
        "trajectory_batch_verified": bool(args.verify_trajectory_batch),
        "sigma2": sigma,
        "innovation": args.innovation,
        "innovation_selection": innovation_selection,
        "temporal_residual_diagnostic": temporal,
        "prior0": args.prior0,
        "lyapunov_per_step": float(np.nanmean(lyap)),
        "propagated": all_scores,
        "controls": controls,
        "step_info": info,
        "complements": args.complements,
    }
    key = fm_result_key(args)
    p = save_json(out, results_path_scale("s3_fm", key, "results.json", args.tag))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
