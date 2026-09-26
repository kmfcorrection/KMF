#!/usr/bin/env python3
"""s1 -- Does curvature predict error direction, at scale and in the right regime?

The scaled form of the 1D Stage 1, with its two documented mis-specifications
(docs/method.md 2.4) built into the design rather than discovered afterwards:

  * The pushforward model is Sigma = J Sigma_c J^T + sigma^2 I. Teacher forcing
    sets Sigma_c = 0 and predicts *no curvature dependence at all*, so a null
    result there is not evidence against anything. This script therefore runs
    three conditioning regimes and reports them side by side:

        teacher     Sigma_c = 0. The predicted null. Included as a control, and
                    any signal here is a confound to be investigated.
        noise       synthetic input perturbation, Sigma_c = s^2 I. The 1D study's
                    positive regime.
        rollout     the model conditioned on its own output after `--warm` steps,
                    so Sigma_c is the model's real accumulated error. This is the
                    regime the theory actually describes and the 1D study could
                    not reach; it is the headline column.

  * Every correlation is judged against its own oracle ceiling, recomputed for
    the rank-k basis. With only k ranks the ceiling *differs* from the 1D value,
    so transferring the old threshold would repeat the original error. A value
    above the ceiling is reported as a confound, not a pass.

Three questions, as in 1D: Q1 per-direction (pooled), Q2 per-state scalars,
Q3 the bias diagnostic -- which at k/N ~ 0.004 has far more dynamic range than
it did at k/N = 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hipp.scale.common_scale import base_parser_scale, results_path_scale, setup_scale
from hipp.scale.curvature_scale import (dominant_wavenumber, estimate_scale,
                                        spectrum_stats_scale)
from hipp.scale.metrics_scale import (explained_error_fraction,
                                      oracle_direction_ceiling_lowrank, pearson,
                                      pooled_direction_test_lowrank, spearman)
from hipp.utils import Table, print_header, save_json


def conditioning_states(ctx, regime: str, noise: float, warm: int, seed: int):
    """Return (conditioning state, target) pairs for the requested regime."""
    fm, xs, ys = ctx["fm"], ctx["x_test"], ctx["y_test"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    if regime == "teacher":
        return list(zip(xs, ys))
    if regime == "noise":
        for x, y in zip(xs, ys):
            pert = torch.randn(x.numel(), generator=g).to(x.device, x.dtype)
            out.append((x + noise * float(x.std()) * pert, y))
        return out
    if regime == "rollout":
        # Walk the model forward `warm` steps from the true state; the target is
        # the true trajectory at the matching time, so the residual is the real
        # accumulated model error rather than an injected perturbation.
        ds, spec = ctx["dataset"], ctx["spec"]
        n_traj = ds.split("test").shape[0]
        rng = np.random.default_rng(seed)
        picks = rng.choice(n_traj, size=min(len(xs), n_traj), replace=False)
        for i in picks:
            traj = ds.trajectory("test", int(i), device=fm.device).reshape(
                -1, fm.N).double()
            if traj.shape[0] < warm + 2:
                continue
            c = traj[0].to(fm.dtype)
            for _ in range(warm):
                c = fm.predict(c)
            out.append((c, traj[warm + 1].to(fm.dtype)))
        return out
    raise ValueError(regime)


def run_regime(args, ctx, regime: str, pairs) -> dict:
    fm, N = ctx["fm"], ctx["N"]
    svals, projs, per_state, bias = [], [], [], []

    for c, y in pairs:
        est = estimate_scale(fm, c, method="pushforward", k=args.k,
                             tau_rel=args.tau_rel, oversample=args.oversample,
                             n_iter=args.n_iter, chunk=args.chunk)
        err = y.reshape(-1).double().to(est.mean.device) - est.mean
        s = est.S.double()
        svals.append(s.cpu().numpy())
        projs.append((est.U.T @ err).cpu().numpy())
        frac = explained_error_fraction(est.U, err, args.k)
        bias.append(frac / (args.k / N))
        st = spectrum_stats_scale(est)
        per_state.append({
            "err_norm": float(err.norm()), "explained_frac": frac,
            "trace_k": st["trace_k"], "cond_k": st["cond_k"],
            "eff_rank_k": st["eff_rank_k"], "smax": st["smax"],
            "smin": st["smin"], "s_mean": st["s_mean"],
            "captured_trace_frac": est.meta.get("captured_trace_frac", float("nan")),
        })

    rho, p = pooled_direction_test_lowrank(svals, projs)
    ceil = oracle_direction_ceiling_lowrank(np.mean(np.stack(svals), axis=0),
                                            pooled_over=len(svals), seed=args.seed)
    frac_ceiling = rho / ceil["median"] if abs(ceil["median"]) > 1e-9 else float("nan")

    errs = np.array([q["err_norm"] for q in per_state])
    q2 = {}
    for key in ("trace_k", "cond_k", "eff_rank_k", "smax", "smin", "s_mean"):
        v = np.array([q[key] for q in per_state])
        rs, ps = spearman(v, errs)
        rp, _ = pearson(v, errs)
        q2[key] = {"spearman": rs, "p": ps, "pearson": rp}

    passes = bool(rho > 0.5 * ceil["median"] and p < 0.01)
    # A rho above the oracle ceiling only means something if it is a rho at all:
    # with few states and few ranks the statistic is noisy, and calling an
    # insignificant fluctuation a "confound" overstates it in the same way that
    # calling it a "pass" would.
    confound = bool(rho > ceil["p95"] and p < 0.01)
    return {"regime": regime, "n_states": len(pairs),
            "rho_pooled": rho, "p_pooled": p, "ceiling": ceil,
            "frac_of_ceiling": frac_ceiling, "passes": passes,
            "confound": confound,
            "bias_ratio_mean": float(np.mean(bias)),
            "bias_ratio_median": float(np.median(bias)),
            "q2": q2, "per_state": per_state}


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--input-noise", type=float, default=0.05,
                    help="std of the injected perturbation for the 'noise' regime, "
                         "as a fraction of the state's own std")
    ap.add_argument("--warm", type=int, default=4,
                    help="autoregressive steps before measuring, for the "
                         "'rollout' regime -- the number that makes Sigma_c "
                         "non-zero for real rather than synthetically")
    ap.add_argument("--regimes", nargs="+",
                    default=["teacher", "noise", "rollout"])
    args = ap.parse_args()

    print_header(f"s1: curvature vs error direction  [{args.pde}]  k={args.k}")
    ctx = setup_scale(args)

    results = {}
    for regime in args.regimes:
        pairs = conditioning_states(ctx, regime, args.input_noise, args.warm,
                                    args.seed)
        if not pairs:
            print(f"\n  regime {regime}: no usable states, skipped")
            continue
        print(f"\n  regime {regime}: {len(pairs)} states ...", flush=True)
        results[regime] = run_regime(args, ctx, regime, pairs)

    # ---- Q1 ------------------------------------------------------------
    print("\nQ1  Pooled per-direction test, judged against the oracle ceiling")
    t = Table("regime", "n", "pooled rho", "ceiling", "obs/ceiling", "p", "verdict")
    for regime, r in results.items():
        v = ("CONFOUND" if r["confound"] else
             "pass" if r["passes"] else "no signal")
        t.add(regime, r["n_states"], r["rho_pooled"], r["ceiling"]["median"],
              r["frac_of_ceiling"], f"{r['p_pooled']:.2e}", v)
    print(t)
    print("  'teacher' is the predicted null: Sigma_c = 0 makes the pushforward")
    print("  model reduce to sigma^2 I, so a pass there is a confound, not a")
    print("  result. 'rollout' is the regime the theory describes.")

    # ---- Q2 ------------------------------------------------------------
    print("\nQ2  Per-state error vs scalar curvature summaries (Spearman rho)")
    keys = ["trace_k", "cond_k", "eff_rank_k", "smax", "smin", "s_mean"]
    t = Table("regime", *keys)
    for regime, r in results.items():
        t.add(regime, *[f"{r['q2'][kk]['spearman']:+.3f}" for kk in keys])
    print(t)
    print("  A heteroscedastic isotropic baseline Sigma = alpha(c) I driven by")
    print("  whichever of these correlates best is the control that a full")
    print("  covariance must beat; s2 runs it.")

    # ---- Q3 ------------------------------------------------------------
    print(f"\nQ3  Bias diagnostic: error energy in the top-{args.k} subspace, "
          f"vs the k/N = {args.k/ctx['N']:.4f} an isotropic error would give")
    t = Table("regime", "ratio (mean)", "ratio (median)", "reading")
    for regime, r in results.items():
        rr = r["bias_ratio_mean"]
        reading = ("bias-dominated" if rr < 2 else
                   "error concentrates where J amplifies")
        t.add(regime, rr, r["bias_ratio_median"], reading)
    print(t)
    print("  A ratio near 1 means the error looks isotropic to J -- model bias,")
    print("  which no Jacobian construction can capture (docs/method.md 9).")

    # ---- where the curvature lives --------------------------------------
    print("\nSpectral location of the leading curvature directions")
    c0 = ctx["x_test"][0]
    est = estimate_scale(ctx["fm"], c0, method="pushforward", k=args.k,
                         tau_rel=args.tau_rel, oversample=args.oversample,
                         n_iter=args.n_iter, chunk=args.chunk)
    wk = dominant_wavenumber(est.U, ctx["spec"].n).cpu().numpy()
    print(f"  peak wavenumber bin of the top 10 directions: {wk[:10].tolist()}")
    print(f"  (16 bins from |k|=0 to |k|_max; the claim is that these match the")
    print(f"   PDE's most-amplified band, not that they are arbitrary)")

    gate = results.get("rollout") or results.get("noise")
    verdict = "neither"
    if gate:
        verdict = ("confound" if gate["confound"]
                   else "pushforward" if gate["passes"] else "neither")
    print(f"\nVERDICT (on the {'rollout' if 'rollout' in results else 'noise'} "
          f"regime): {verdict}")

    out = {"pde": args.pde, "k": args.k, "N": ctx["N"], "tau_rel": args.tau_rel,
           "input_noise": args.input_noise, "warm": args.warm,
           "regimes": results, "verdict": verdict,
           "top_wavenumber_bins": wk[:16].tolist()}
    p = save_json(out, results_path_scale("s1", args.pde, "results.json", args.tag))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
