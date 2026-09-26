#!/usr/bin/env python3
"""s0 -- Does the low-rank machinery reproduce the exact object it replaces?

Run this first. Everything in the scaled study rests on the claim that a rank-k
randomized approximation of J carries the information the dense N x N Jacobian
carried, and that claim is checkable: at a resolution small enough for the exact
object to exist (32x32, N = 1024), both are computed and compared directly.

This is the bridge between the 1D results and any 2D claim. Without it, a
negative result at scale is unattributable -- it could be the method failing or
the approximation failing, and those have opposite implications.

Five questions:

  Q1  Spectrum. How well does the randomized SVD recover the singular values,
      and how much does that depend on the power-iteration count? For a residual
      operator J = I + dNet/dc the spectrum is clustered near 1, which is the
      hardest case for randomized range finding, so this is not a formality.

  Q2  Subspace. Overlap between the estimated and exact leading subspaces. The
      method's claims are about *directions*, so this matters more than Q1.

  Q3  Covariance. Relative error of the rank-k Sigma against the exact
      Sigma = J J^T + tau I, and -- the quantity that actually matters -- the
      error in the Mahalanobis distances and log-densities the metrics use.

  Q4  Rank. Where the above saturate as k grows, i.e. how much of the operator's
      geometry is genuinely low-dimensional.

  Q5  Cost. Measured network passes and wall time for each path, against the
      O(N) the dense route needs. This is the number that justifies the whole
      package.

Also sweeps the finite-difference epsilon, since the black-box estimator is only
usable if a stable window exists in float32 -- which is a different question at
N = 1024 than it was at N = 128.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hipp.scale.adapters import FrozenFM, check_adapter
from hipp.scale.common_scale import base_parser_scale, results_path_scale
from hipp.scale.curvature_scale import estimate_scale
from hipp.scale.data2d import build_dataset2d
from hipp.scale.jacobian import (fd_jvp_operator, jacobian_ops, randomized_svd,
                                 set_probe_precision, subspace_overlap)
from hipp.scale.lowrank import LowRankGaussian
from hipp.scale.models2d import build_model
from hipp.utils import Table, get_device, print_header, save_json, set_seed


def exact_jacobian_flat(f, x: torch.Tensor) -> torch.Tensor:
    """Dense J via reverse-mode AD. O(N) backward passes -- the thing we are
    replacing, computed here only so it can be compared against."""
    return torch.autograd.functional.jacobian(f, x, vectorize=True).detach()


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--n-states", type=int, default=6,
                    help="states to average over; each needs one dense Jacobian")
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--iters", type=int, nargs="+", default=[0, 1, 2, 4])
    ap.add_argument("--val-grid", type=int, default=32,
                    help="resolution at which the exact Jacobian is affordable")
    ap.add_argument("--untrained", action="store_true",
                    help="use a randomly initialized model instead of a "
                         "checkpoint, so the validation runs with no training")
    args = ap.parse_args()

    set_seed(args.seed)
    set_probe_precision(high=not args.allow_tf32)
    device = get_device(args.device)
    n = args.val_grid
    N = n * n

    print_header(f"s0: low-rank vs exact  [{args.pde}]  grid={n}x{n}  N={N}")

    # ---- model ---------------------------------------------------------
    if args.untrained:
        model = build_model(args.arch, in_channels=1, modes=min(16, n // 2),
                            width=32, n_layers=4).to(device)
        model.set_norm(0.0, 1.0)
        fm = FrozenFM(model, (1, n, n), name=f"{args.arch}-untrained", device=device)
        print("  using an untrained model: this validates the linear algebra, "
              "not the physics")
    else:
        from hipp.scale.common_scale import load_fm
        fm = load_fm(args, device=device)
        if fm.state_shape[-1] != n:
            print(f"  checkpoint is {fm.state_shape[-1]}x{fm.state_shape[-1]}, "
                  f"not {n}x{n}; falling back to --untrained sizing")
            model = build_model(args.arch, in_channels=1, modes=min(16, n // 2),
                                width=32, n_layers=4).to(device)
            model.set_norm(0.0, 1.0)
            fm = FrozenFM(model, (1, n, n), name=f"{args.arch}-resized", device=device)

    chk = check_adapter(fm, verbose=False)
    print(f"  adapter check: pass={chk['pass']}  "
          f"jvp-vs-fd {chk.get('jvp_vs_fd_rel', float('nan')):.2e}  "
          f"adjoint {chk.get('adjoint_rel', float('nan')):.2e}  "
          f"||J-I||/||I|| {chk.get('residual_dev', float('nan')):.3f}")
    if not chk["pass"]:
        print("  ADAPTER CHECK FAILED -- results below are not trustworthy")

    # ---- states --------------------------------------------------------
    ds = build_dataset2d(args.pde, n=n, n_train=8, n_val=4, n_test=4, n_steps=6,
                         seed=0, device=device)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(ds.n_pairs("test"), size=args.n_states, replace=False)
    states = [ds.pair("test", int(i), device=device)[0].reshape(-1) for i in idx]

    # ---- exact Jacobians ------------------------------------------------
    print(f"\n  computing {args.n_states} exact {N}x{N} Jacobians "
          f"({N} backward passes each) ...", flush=True)
    t0 = time.time()
    exact = []
    for c in states:
        J = exact_jacobian_flat(fm.flat_fn(), c).double()
        U, S, Vh = torch.linalg.svd(J)
        exact.append({"J": J, "U": U, "S": S, "V": Vh.T})
    dense_time = (time.time() - t0) / args.n_states
    print(f"  -> {dense_time:.1f}s per state")

    out: dict = {"pde": args.pde, "grid": n, "N": N, "n_states": args.n_states,
                 "dense_seconds_per_state": dense_time, "adapter_check": chk}

    # ---- Q1/Q2: spectrum and subspace vs power iterations ----------------
    print("\nQ1/Q2  Randomized SVD accuracy vs power iterations  (k = 32)")
    t = Table("n_iter", "sval rel err", "subspace overlap", "passes/state", "s/state")
    q1 = []
    for n_iter in args.iters:
        errs, ovl, passes, tt = [], [], 0, 0.0
        for c, ex in zip(states, exact):
            ops = jacobian_ops(fm.flat_fn(), c)
            t1 = time.time()
            U, S, V = randomized_svd(ops, k=32, oversample=args.oversample,
                                     n_iter=n_iter, chunk=args.chunk,
                                     generator=torch.Generator(device="cpu").manual_seed(0))
            tt += time.time() - t1
            errs.append(float((S - ex["S"][:32]).abs().max() / ex["S"][0]))
            ovl.append(subspace_overlap(U, ex["U"][:, :32]))
            passes = sum(ops.n_calls)
        row = {"n_iter": n_iter, "sval_rel_err": float(np.mean(errs)),
               "subspace_overlap": float(np.mean(ovl)), "passes": passes,
               "seconds": tt / args.n_states}
        q1.append(row)
        t.add(n_iter, row["sval_rel_err"], row["subspace_overlap"], passes,
              row["seconds"])
    print(t)
    print("  A residual operator has a spectrum clustered near 1, which is the")
    print("  worst case for randomized range finding. If overlap at n_iter=0 is")
    print("  far below 1, every downstream direction claim needs n_iter >= 2.")
    out["q1_power_iterations"] = q1

    # ---- Q3/Q4: covariance error vs rank ---------------------------------
    print("\nQ3/Q4  Rank-k covariance vs the exact J J^T + tau I")
    t = Table("k", "k/N", "Sigma rel err", "captured trace", "maha rel err",
              "logdet rel err", "passes/state")
    q3 = []
    for k in args.ranks:
        if k > N:
            continue
        sig_err, maha_err, ld_err, cap, passes = [], [], [], [], 0
        for c, ex in zip(states, exact):
            J = ex["J"]
            tau = float(args.tau_rel * (ex["S"] ** 2).mean())
            Sig_exact = J @ J.T + tau * torch.eye(N, dtype=torch.float64, device=J.device)
            mean = fm.predict(c).double()
            p_exact = LowRankGaussian.from_dense_covariance(mean, Sig_exact, k=N - 1)

            est = estimate_scale(fm, c, method="pushforward", k=k,
                                 tau_rel=args.tau_rel, oversample=args.oversample,
                                 n_iter=args.n_iter, chunk=args.chunk)
            passes = est.meta["jvp_calls"] + est.meta["vjp_calls"]
            Sig_k = est.prior.dense_covariance()
            sig_err.append(float((Sig_k - Sig_exact).norm() / Sig_exact.norm()))
            cap.append(float((ex["S"][:k] ** 2).sum() / (ex["S"] ** 2).sum()))

            # The quantities the metrics actually consume.
            y = mean + 0.1 * torch.randn(N, dtype=torch.float64, device=mean.device)
            m_e = float(p_exact.mahalanobis_sq(y))
            m_k = float(est.prior.mahalanobis_sq(y))
            maha_err.append(abs(m_k - m_e) / max(abs(m_e), 1e-30))
            ld_err.append(abs(est.prior.logdet_cov() - p_exact.logdet_cov())
                          / max(abs(p_exact.logdet_cov()), 1e-30))
        row = {"k": k, "k_over_N": k / N, "sigma_rel_err": float(np.mean(sig_err)),
               "captured_trace": float(np.mean(cap)),
               "maha_rel_err": float(np.mean(maha_err)),
               "logdet_rel_err": float(np.mean(ld_err)), "passes": passes}
        q3.append(row)
        t.add(k, f"{k/N:.4f}", row["sigma_rel_err"], row["captured_trace"],
              row["maha_rel_err"], row["logdet_rel_err"], passes)
    print(t)
    print("  'captured trace' is the share of sum(s_i^2) inside rank k -- the")
    print("  honest statement of how low-dimensional this operator's geometry is.")
    print("  If it is already ~1 at small k the method is cheap; if it needs")
    print("  k ~ N the low-rank route is not viable and that must be reported.")
    out["q3_rank"] = q3

    # ---- FD epsilon sweep ------------------------------------------------
    print("\nFinite-difference probe accuracy (black-box path), rel err of J v")
    t = Table("eps", "rel err mean", "rel err std")
    sweep = []
    v = torch.randn(N, device=device, dtype=fm.dtype)
    v = v / v.norm()
    for eps in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        errs = []
        for c, ex in zip(states, exact):
            jv_true = (ex["J"] @ v.double())
            jv_fd = fd_jvp_operator(fm.flat_fn(), c, eps=eps)(v).double()
            errs.append(float((jv_fd - jv_true).norm() / jv_true.norm()))
        sweep.append({"eps": eps, "rel_err_mean": float(np.mean(errs)),
                      "rel_err_std": float(np.std(errs))})
        t.add(f"{eps:.0e}", sweep[-1]["rel_err_mean"], sweep[-1]["rel_err_std"])
    print(t)
    best = min(sweep, key=lambda r: r["rel_err_mean"])
    print(f"  best eps = {best['eps']:.0e} at rel err {best['rel_err_mean']:.2e}")
    out["fd_eps_sweep"] = sweep
    out["fd_best_eps"] = best["eps"]

    # ---- Q5: cost --------------------------------------------------------
    print("\nQ5  Cost accounting")
    k = args.k
    passes_lr = (2 + 3 * args.n_iter) * (k + args.oversample)
    t = Table("route", "network passes / state", "relative to dense")
    t.add("dense exact J", N, 1.0)
    t.add(f"rank-{k} randomized", passes_lr, passes_lr / N)
    for grid in (64, 128, 256):
        t.add(f"  ... at {grid}x{grid} (N={grid*grid})", f"{passes_lr} vs {grid*grid}",
              passes_lr / (grid * grid))
    print(t)
    print(f"  The dense route is O(N) and the low-rank route is O(k), so the")
    print(f"  saving grows with resolution: {passes_lr/N:.3f}x here, "
          f"{passes_lr/(128*128):.4f}x at 128x128.")
    out["cost"] = {"N": N, "dense_passes": N, "lowrank_passes": passes_lr,
                   "k": k, "n_iter": args.n_iter, "oversample": args.oversample}

    # ---- verdict ---------------------------------------------------------
    # Two independent failure modes with different downstream consequences, so
    # they get separate verdicts rather than one label. A covariance can be
    # numerically accurate while the individual directions are not recovered at
    # all -- that happens whenever the spectrum is flat, since then the rank-k
    # subspace is nearly arbitrary but every choice of it gives nearly the same
    # Sigma. Calibration stages (s2, s3) are fine in that regime; the direction
    # test (s1) is not, because its whole claim is about which direction is which.
    ok_iter = [r for r in q1 if r["n_iter"] == args.n_iter]
    ok_rank = [r for r in q3 if r["k"] == args.k]
    verdict = {"directions": "unknown", "covariance": "unknown"}
    if ok_iter:
        verdict["directions"] = ("faithful" if ok_iter[0]["subspace_overlap"] > 0.9
                                 else "lossy")
        verdict["subspace_overlap"] = ok_iter[0]["subspace_overlap"]
    if ok_rank:
        verdict["covariance"] = ("faithful" if ok_rank[0]["maha_rel_err"] < 0.1
                                 else "lossy")
        verdict["maha_rel_err"] = ok_rank[0]["maha_rel_err"]
        verdict["captured_trace"] = ok_rank[0]["captured_trace"]

    print(f"\nVERDICT at k={args.k}, n_iter={args.n_iter}")
    print(f"  covariance (Mahalanobis error < 10%):     {verdict['covariance']}"
          f"   [{verdict.get('maha_rel_err', float('nan')):.3f}]")
    print(f"  directions (subspace overlap > 0.9):      {verdict['directions']}"
          f"   [{verdict.get('subspace_overlap', float('nan')):.3f}]")
    if verdict["covariance"] == "lossy":
        print("  -> s2/s3 results would not be attributable. Raise --k or --n-iter.")
    if verdict["directions"] == "lossy":
        cap = verdict.get("captured_trace", float("nan"))
        print(f"  -> s1's per-direction claims are not supported at this rank.")
        print(f"     With only {cap:.1%} of the spectral mass inside rank {args.k},")
        print(f"     the operator's geometry is not low-dimensional: the retained")
        print(f"     subspace is nearly arbitrary, which is a fact about this model")
        print(f"     (a near-identity residual map) rather than about the estimator.")
    out["verdict"] = verdict

    p = save_json(out, results_path_scale("s0", args.pde, "results.json", args.tag))
    print(f"\nwrote {p}")

    if not args.no_plots:
        _plot(args, q1, q3, sweep)


def _plot(args, q1, q3, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot([r["n_iter"] for r in q1], [r["subspace_overlap"] for r in q1], "o-")
    ax[0].set_xlabel("power iterations")
    ax[0].set_ylabel("subspace overlap with exact")
    ax[0].set_title("Q2: randomized SVD fidelity")
    ax[0].grid(alpha=0.3)

    ax[1].plot([r["k"] for r in q3], [r["captured_trace"] for r in q3], "o-",
               label="captured trace")
    ax[1].plot([r["k"] for r in q3], [r["maha_rel_err"] for r in q3], "s-",
               label="Mahalanobis rel err")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("rank k")
    ax[1].set_title("Q4: how low-dimensional is the geometry?")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    ax[2].loglog([r["eps"] for r in sweep], [r["rel_err_mean"] for r in sweep], "o-")
    ax[2].set_xlabel(r"$\varepsilon$")
    ax[2].set_ylabel("rel err of J v")
    ax[2].set_title("FD probe window")
    ax[2].grid(True, which="both", alpha=0.3)

    fig.suptitle(f"s0 low-rank validation -- {args.pde}")
    fig.tight_layout()
    p = results_path_scale("s0", args.pde, "s0.png", args.tag)
    fig.savefig(p, dpi=140)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
