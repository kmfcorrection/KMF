#!/usr/bin/env python3
"""s3 -- Does the Jacobian predict how uncertainty *grows* along a rollout?

The headline scaled experiment, and the one the 1D study set up but never ran.

Everything before this measures a one-step predictive covariance, where the
theory's own premise (Sigma_c != 0) has to be manufactured by injecting noise.
Here it is not manufactured: the model consumes its own output, its error
accumulates, and the claim under test is that the accumulation is governed by

    Sigma_{t+1} = J_t Sigma_t J_t^T + sigma^2 I

with sigma^2 the only fitted quantity -- estimated once, on held-out one-step
residuals under teacher forcing, before any rollout is run.

The comparison that matters is against growth models that need no Jacobian:

  constant      the one-step covariance held fixed. Any per-state UQ method
                applied naively at each step gives this; it has no accumulation
                mechanism at all.
  linear-in-t   the same covariance scaled by t, i.e. errors adding as
                independent increments. This is the strong heuristic control:
                reproducing linear variance growth is *not* evidence for the
                Jacobian, since diffusion gives it for free. The claim can only
                be about deviations from it -- which is exactly what a chaotic
                flow should produce and a dissipative one should not.

Ablations on the propagation itself (`--complements`) separate the parts of the
recursion: dropping the complement term shows how much comes from directions
entering the retained subspace, and the isotropic variant shows whether the
*geometry* of that term matters or only its trace.

Read `ratio = pred_rms / rmse` as the headline: flat and near 1 across steps
means the growth rate is right. `frac_err_in_rank_over_isotropic` says whether
the propagated basis is still tracking where the error actually is; if it decays
toward 1, the basis has drifted off the error and the rank is too small.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hipp.scale.common_scale import base_parser_scale, results_path_scale, setup_scale
from hipp.scale.curvature_scale import estimate_scale
from hipp.scale.lowrank import LowRankGaussian
from hipp.scale.physics2d import PhysicsEnergy2D
from hipp.scale.rollout import (aggregate_rollout, constant_covariance_rollout,
                                fit_sigma2, propagate_rollout, score_rollout)
from hipp.utils import Table, print_header, save_json


def lyapunov_estimate(fm, traj, steps: int, eps: float = 1e-4) -> float:
    """Crude finite-time Lyapunov exponent of the *model*, for context.

    Two nearby initial states, propagated by the model; the log growth rate of
    their separation. Positive means the model itself is chaotic, which is the
    condition under which propagated uncertainty should outgrow linear-in-t.
    Reported so the rollout numbers can be read against the dynamics rather
    than in isolation.
    """
    c0 = traj[0].reshape(-1)
    d0 = torch.randn_like(c0)
    d0 = d0 / d0.norm() * eps * float(c0.norm())
    a, b = c0.clone(), c0 + d0
    r0 = float((b - a).norm())
    rates = []
    for _ in range(steps):
        a, b = fm.predict(a), fm.predict(b)
        r = float((b - a).norm())
        rates.append(np.log(max(r, 1e-30) / max(r0, 1e-30)))
        # renormalize to stay in the linear regime
        b = a + (b - a) / max(r, 1e-30) * r0
    return float(np.mean(np.diff([0.0] + rates))) if len(rates) > 1 else float("nan")


def _pushforward_prior0(args, fm, c, sigma2):
    """One-step pushforward covariance, rescaled to per-dimension variance sigma^2.

    The rescaling matters: Sigma_0 has to carry the same total uncertainty as
    the isotropic default, or the ablation would be comparing growth rates that
    started from different amounts of variance.
    """
    est = estimate_scale(fm, c, method="pushforward", k=args.k,
                         tau_rel=args.tau_rel, oversample=args.oversample,
                         n_iter=args.n_iter, chunk=args.chunk)
    return est.prior.rescaled(sigma2 * fm.N / max(est.prior.trace(), 1e-300))


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=12, help="rollout length")
    ap.add_argument("--n-traj", type=int, default=16,
                    help="test trajectories to propagate")
    ap.add_argument("--n-sketch", type=int, default=None,
                    help="complement sketch width; defaults to k + 10")
    ap.add_argument("--complements", nargs="+",
                    default=["nystrom", "isotropic", "none"],
                    help="propagation ablations; 'nystrom' is the method")
    ap.add_argument("--sigma2", type=float, default=None,
                    help="override the fitted per-step injected variance")
    ap.add_argument("--prior0", default="isotropic",
                    choices=["isotropic", "pushforward"],
                    help="Sigma_0. 'isotropic' (sigma^2 I) is the honest initial "
                         "condition for a forecast started from known data, and "
                         "is the default. But from it the 'none' and 'isotropic' "
                         "complement ablations stay rank 0 forever -- nothing "
                         "ever creates a direction -- which makes the ablation "
                         "degenerate rather than informative. Use 'pushforward' "
                         "to start every mode from the same rank-k one-step "
                         "covariance, so the ablation isolates the ability to "
                         "*add* directions from the ability to propagate them.")
    ap.add_argument("--physics", action="store_true",
                    help="also report the PDE residual of the propagated mean")
    args = ap.parse_args()

    print_header(f"s3: rollout uncertainty propagation  [{args.pde}]  "
                 f"k={args.k}  steps={args.steps}")
    ctx = setup_scale(args)
    fm, ds, N = ctx["fm"], ctx["dataset"], ctx["N"]

    # ---- sigma^2 -------------------------------------------------------
    if args.sigma2 is None:
        s2 = fit_sigma2(fm, ctx["x_cal"], ctx["y_cal"])
        print(f"\n  sigma^2 (per-step injected variance), fitted on "
              f"{s2['n']} teacher-forced states:")
        print(f"    mean {s2['sigma2']:.4e}  median {s2['sigma2_median']:.4e}  "
              f"rel.SE {s2['rel_se']:.3f}  tail {s2['tail_ratio']:.2f}")
        sigma2 = s2["sigma2"]
    else:
        sigma2 = args.sigma2
        s2 = {"sigma2": sigma2, "note": "supplied on the command line"}
        print(f"\n  sigma^2 = {sigma2:.4e} (supplied)")

    # ---- trajectories ---------------------------------------------------
    arr = ds.split("test")
    n_traj = min(args.n_traj, arr.shape[0])
    steps = min(args.steps, arr.shape[1] - 1)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(arr.shape[0], size=n_traj, replace=False)
    print(f"  propagating {n_traj} trajectories x {steps} steps  "
          f"(N={N}, rank {args.k})")

    lyap = []
    for i in picks[:4]:
        traj = ds.trajectory("test", int(i), device=fm.device).reshape(-1, N)
        lyap.append(lyapunov_estimate(fm, traj.to(fm.dtype), min(steps, 8)))
    print(f"  finite-time Lyapunov exponent of the model: "
          f"{np.nanmean(lyap):+.4f} per step  "
          f"({'chaotic' if np.nanmean(lyap) > 0 else 'contracting'})")

    # ---- propagate -------------------------------------------------------
    all_scores: dict = {}
    controls: dict = {}
    per_step_info: dict = {}

    for mode in args.complements:
        scores = []
        infos = []
        for j, i in enumerate(picks):
            traj = ds.trajectory("test", int(i), device=fm.device).reshape(-1, N).double()
            g = torch.Generator(device="cpu").manual_seed(args.seed + j)
            p0 = None
            if args.prior0 == "pushforward":
                p0 = _pushforward_prior0(args, fm, traj[0].to(fm.dtype), sigma2)
            tr = propagate_rollout(fm, traj[0].to(fm.dtype), steps, sigma2,
                                   k=args.k, prior0=p0, n_sketch=args.n_sketch,
                                   chunk=args.chunk, generator=g, complement=mode)
            scores.append(score_rollout(tr, traj))
            infos.append(tr.info)
            if mode == args.complements[0] and j == 0:
                # Controls use the *one-step* curvature covariance, calibrated
                # to the same sigma^2, held fixed or grown heuristically.
                pc = _pushforward_prior0(args, fm, traj[0].to(fm.dtype), sigma2)
                for growth in ("none", "sqrt_t"):
                    controls[growth] = constant_covariance_rollout(
                        pc, tr.means, traj, growth=growth)
        all_scores[mode] = aggregate_rollout(scores)
        per_step_info[mode] = infos[0]
        print(f"    [{mode}] done", flush=True)

    # ---- report ----------------------------------------------------------
    main_mode = args.complements[0]
    print(f"\nPropagated covariance vs actual rollout error  [{main_mode}]")
    t = Table("step", "rmse", "pred rms", "ratio", "D^2/N", "z",
              "err in rank / isotropic", "trace in rank")
    for r in all_scores[main_mode]:
        t.add(r["step"], r["rmse"], r["pred_rms"], r["ratio"], r["maha_over_N"],
              r["z"], r["frac_err_in_rank_over_isotropic"], r["frac_trace_in_rank"])
    print(t)
    print("  'ratio' flat and near 1 = the growth rate is right. Drifting below 1")
    print("  = the recursion under-predicts accumulation; above = over-predicts.")

    print("\nControls that need no Jacobian")
    t = Table("model", *[f"t={r['step']}" for r in all_scores[main_mode][:6]])
    t.add("propagated (ratio)", *[f"{r['ratio']:.3f}" for r in all_scores[main_mode][:6]])
    for growth, rows in controls.items():
        lbl = "constant" if growth == "none" else "linear in t"
        t.add(f"{lbl} (ratio)", *[f"{r['ratio']:.3f}" for r in rows[:6]])
    print(t)
    print("  Linear-in-t growth is free from a diffusion argument. The Jacobian")
    print("  earns its place only if it tracks the *deviation* from that.")

    if len(args.complements) > 1:
        print("\nAblation: what the propagation's parts contribute")
        t = Table("complement mode", *[f"ratio t={r['step']}"
                                       for r in all_scores[main_mode][:5]])
        for mode, rows in all_scores.items():
            t.add(mode, *[f"{r['ratio']:.3f}" for r in rows[:5]])
        print(t)
        print("  'none' drops the term that lets new directions enter the")
        print("  retained subspace; 'isotropic' keeps its trace but not its")
        print("  geometry. The gap between them and 'nystrom' is the value of")
        print("  the directional information specifically.")

    # ---- physics --------------------------------------------------------
    phys = None
    if args.physics:
        print("\nPDE residual of the propagated mean (is the rollout still physical?)")
        traj = ds.trajectory("test", int(picks[0]), device=fm.device).reshape(-1, N)
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        tr = propagate_rollout(fm, traj[0].to(fm.dtype), steps, sigma2, k=args.k,
                               n_sketch=args.n_sketch, chunk=args.chunk,
                               generator=g, complement=main_mode)
        phys = []
        t = Table("step", "residual (model mean)", "residual (truth)")
        prev = traj[0].double()
        for s in range(len(tr.means)):
            E = PhysicsEnergy2D(ctx["spec"], prev)
            r_model = float(E.rms_residual(tr.means[s].double()))
            r_true = float(E.rms_residual(traj[s + 1].double()))
            phys.append({"step": s + 1, "residual_model": r_model,
                         "residual_truth": r_true})
            t.add(s + 1, r_model, r_true)
            prev = traj[s + 1].double()
        print(t)

    out = {"pde": args.pde, "k": args.k, "N": N, "steps": steps, "n_traj": n_traj,
           "prior0": args.prior0,
           "sigma2": s2, "lyapunov_per_step": float(np.nanmean(lyap)),
           "propagated": all_scores, "controls": controls,
           "step_info": per_step_info, "physics": phys,
           "complements": args.complements}
    p = save_json(out, results_path_scale("s3", args.pde, "results.json", args.tag))
    print(f"\nwrote {p}")

    if not args.no_plots:
        _plot(args, all_scores, controls, main_mode)


def _plot(args, all_scores, controls, main_mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = all_scores[main_mode]
    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].errorbar(steps, [r["rmse"] for r in rows],
                   yerr=[r["rmse_se"] for r in rows], marker="o", label="actual RMSE")
    ax[0].errorbar(steps, [r["pred_rms"] for r in rows],
                   yerr=[r["pred_rms_se"] for r in rows], marker="s",
                   label="propagated RMS")
    ax[0].set_xlabel("rollout step")
    ax[0].set_ylabel("RMS")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_title("growth of error vs predicted spread")

    for mode, rr in all_scores.items():
        ax[1].plot([r["step"] for r in rr], [r["ratio"] for r in rr], marker="o",
                   label=f"propagated ({mode})")
    for growth, rr in controls.items():
        ax[1].plot([r["step"] for r in rr], [r["ratio"] for r in rr], "--",
                   label="constant" if growth == "none" else "linear in t")
    ax[1].axhline(1.0, color="k", lw=1)
    ax[1].set_xlabel("rollout step")
    ax[1].set_ylabel("predicted / actual")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[1].set_title("calibration of the growth rate")

    ax[2].plot(steps, [r["frac_err_in_rank_over_isotropic"] for r in rows], marker="o")
    ax[2].axhline(1.0, color="k", lw=1, label="isotropic expectation")
    ax[2].set_xlabel("rollout step")
    ax[2].set_ylabel(f"error in rank-{args.k} / (k/N)")
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    ax[2].set_title("is the basis still tracking the error?")

    fig.suptitle(f"s3 rollout propagation -- {args.pde}, k={args.k}")
    fig.tight_layout()
    p = results_path_scale("s3", args.pde, "s3.png", args.tag)
    fig.savefig(p, dpi=140)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
