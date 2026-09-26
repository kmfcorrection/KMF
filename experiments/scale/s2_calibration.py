#!/usr/bin/env python3
"""s2 -- Is the predictive distribution valid and is it good? (1D Stages 2 + 3)

Merged because at scale they share every expensive computation: both need the
calibrated prior at every test state, and building that is the cost.

What is reported, and why each column is there:

  joint calibration    mean D^2/N and the standardized z = (D^2-N)/sqrt(2N).
                       The chi^2_N *test* is not the headline: at N = 16,384 it
                       has power to reject on deviations of no practical size,
                       so a table of p-values would read as "every method fails
                       including the correct one". z says how far off, in
                       interpretable units.

  subspace split       D^2 decomposed into the rank-k subspace (~ chi^2_k) and
                       its complement (~ chi^2_{N-k}). This is the column that
                       does not exist in the 1D study and is the most diagnostic
                       thing here: a method can look perfectly calibrated in
                       aggregate while its curvature contributes nothing, because
                       at k/N ~ 0.004 the aggregate is almost entirely the floor.

  alpha floor share    the same point from the calibration side: what fraction of
                       the fitted alpha is the isotropic floor rather than the
                       geometry. Near 1 means an isotropic Gaussian wearing a hat.

  marginal quality     NLL, CRPS, ECE, sharpness -- per pixel, so comparable
                       across resolutions and to the 1D numbers.

  conformal            distribution-free coverage from the same calibration
                       split, both the constant-width variant and the variant
                       normalized by each method's own sigma. The second is the
                       interesting one: it is where the curvature supplies shape
                       and conformal supplies the guarantee.

The success criterion is unchanged from docs/method.md 8: beating `identity` is
nearly free and proves nothing. The reference is `fixed_global_lowrank`, which is
state-independent -- if the curvature does not beat it, the Jacobian is
contributing no per-state information.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hipp.scale.baselines_scale import SplitConformal
from hipp.scale.common_scale import (DEFAULT_ESTIMATORS_SCALE, base_parser_scale,
                                     build_zoo_streaming, results_path_scale,
                                     setup_scale)
from hipp.scale.metrics_scale import (chi2_report, ellipsoid_coverage,
                                      maha_from_summaries, subspace_split_test)
from hipp.utils import Table, print_header, save_json

REFERENCE = "fixed_global_lowrank"


def conformal_block(args, ctx, zoo: dict) -> dict:
    """Split-conformal intervals, plain and normalized by each method's sigma."""
    from hipp.scale.curvature_scale import estimate_scale
    fm = ctx["fm"]
    out = {}

    # Calibration residuals, shared by every variant.
    res_cal, mu_cal = [], []
    for c, y in zip(ctx["x_cal"], ctx["y_cal"]):
        mu = fm.predict(c.reshape(-1)).double()
        res_cal.append((y.reshape(-1).double().to(mu.device) - mu).cpu().numpy())
        mu_cal.append(mu.cpu().numpy())
    res_cal = np.concatenate(res_cal)

    res_test, mu_test, y_test = [], [], []
    for c, y in zip(ctx["x_test"], ctx["y_test"]):
        mu = fm.predict(c.reshape(-1)).double()
        mu_test.append(mu.cpu().numpy())
        y_test.append(y.reshape(-1).double().cpu().numpy())
    mu_test = np.concatenate(mu_test)
    y_test = np.concatenate(y_test)

    out["absolute"] = SplitConformal(alpha=1 - args.level).fit(res_cal).report(
        mu_test, y_test)

    for name in args.conformal_on:
        if name not in zoo:
            continue
        # sigma on the calibration split, from the same calibrated prior
        sig_cal, sig_test = [], []
        build = (lambda c, m=name: estimate_scale(
            fm, c, method=m, k=args.k, tau_rel=args.tau_rel,
            oversample=args.oversample, n_iter=args.n_iter,
            chunk=args.chunk).prior.rescaled(zoo[m]["alpha"]))
        for c in ctx["x_cal"]:
            sig_cal.append(build(c).marginal_std().cpu().numpy())
        for c in ctx["x_test"]:
            sig_test.append(build(c).marginal_std().cpu().numpy())
        cf = SplitConformal(alpha=1 - args.level, normalized=True).fit(
            res_cal, np.concatenate(sig_cal))
        out[f"normalized_{name}"] = cf.report(mu_test, y_test,
                                              np.concatenate(sig_test))
    return out


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--estimators", nargs="+", default=DEFAULT_ESTIMATORS_SCALE)
    ap.add_argument("--baselines", nargs="+", default=None)
    ap.add_argument("--level", type=float, default=0.9,
                    help="nominal coverage for the conformal block")
    ap.add_argument("--conformal-on", nargs="+", default=["pushforward"],
                    help="methods whose sigma normalizes the conformal score")
    ap.add_argument("--no-conformal", action="store_true")
    args = ap.parse_args()

    print_header(f"s2: calibration and quality  [{args.pde}]  k={args.k}")
    ctx = setup_scale(args, need_ensemble=True, need_dropout=True)
    N = ctx["N"]

    print("\nBuilding the method zoo (streaming, one state at a time):")
    zoo = build_zoo_streaming(args, ctx, estimators=args.estimators,
                              baselines=args.baselines)

    # ---- joint calibration ---------------------------------------------
    print("\nJoint calibration of the whole state")
    t = Table("method", "kind", "alpha", "D^2/N", "z", "cov90", "cov95", "cov99")
    joint = {}
    for name, e in zoo.items():
        m = maha_from_summaries(e["summaries"], e["alpha"])
        rep = chi2_report(m, N)
        cov = ellipsoid_coverage(m, N)
        joint[name] = {**rep, **cov}
        t.add(name, e["kind"], e["alpha"], rep["mean_maha_over_N"], rep["z_mean"],
              cov["cov_90"], cov["cov_95"], cov["cov_99"])
    print(t)
    print("  z = (D^2 - N)/sqrt(2N): 0 is perfect, positive = under-dispersed.")
    print(f"  At N = {N} a chi^2 p-value is uninformative; z is the quantity.")

    # ---- subspace split -------------------------------------------------
    print("\nSubspace split -- where the calibration error lives")
    t = Table("method", "D_k^2/k", "z (subspace)", "D_perp^2/(N-k)", "z (perp)",
              "maha share in k", "alpha floor share")
    split = {}
    for name, e in zoo.items():
        s = subspace_split_test(e["summaries"], e["alpha"])
        split[name] = s
        t.add(name, s["sub"]["mean_over_dof"], s["sub"]["z_mean"],
              s["perp"]["mean_over_dof"], s["perp"]["z_mean"],
              s["share_of_maha_in_subspace"],
              e["alpha_diag"].get("floor_share", float("nan")))
    print(t)
    print("  If 'maha share in k' and 'alpha floor share' both say the geometry")
    print("  contributes almost nothing, then any apparent win in the table above")
    print("  is coming from the isotropic floor, which alpha then cancels.")

    # ---- marginal quality ------------------------------------------------
    print("\nMarginal (per-pixel) quality")
    t = Table("method", "NLL", "CRPS", "ECE", "sharpness", "cov90",
              "interval score", f"beats {REFERENCE}?")
    ref = zoo.get(REFERENCE, {}).get("marginal_agg", {})
    marg = {}
    for name, e in zoo.items():
        a = e["marginal_agg"]
        marg[name] = a
        better = "-"
        if ref and name != REFERENCE:
            better = "yes" if (a["nll"] < ref["nll"] and a["crps"] < ref["crps"]) else "no"
        t.add(name, a["nll"], a["crps"], a["ece"], a["sharpness"],
              a["cov_90_marginal"], a["interval_score_90"], better)
    print(t)
    print(f"  The 'beats' column is against {REFERENCE}, not against isotropic.")
    print("  A state-independent covariance is the sharpest null hypothesis: if")
    print("  the curvature loses to it, the Jacobian adds no per-state information.")

    # ---- conformal -------------------------------------------------------
    conf = {}
    if not args.no_conformal:
        print(f"\nSplit conformal, nominal {args.level:.0%} marginal coverage")
        conf = conformal_block(args, ctx, zoo)
        t = Table("variant", "coverage", "mean width", "interval score", "q")
        for name, r in conf.items():
            t.add(name, r["coverage"], r["mean_width"], r["interval_score"], r["q"])
        print(t)
        print("  'absolute' is a constant-width band -- it cannot express any")
        print("  state dependence, so it is the floor for 'does per-state")
        print("  information help'. The normalized variants take their shape from")
        print("  the named method and their calibration from conformal.")

    out = {"pde": args.pde, "k": args.k, "N": N, "tau_rel": args.tau_rel,
           "n_test": len(ctx["x_test"]), "n_cal": len(ctx["x_cal"]),
           "alpha": {n: e["alpha"] for n, e in zoo.items()},
           "alpha_diag": {n: e["alpha_diag"] for n, e in zoo.items()},
           "seconds": {n: e["seconds"] for n, e in zoo.items()},
           "joint": joint, "subspace_split": split, "marginal": marg,
           "conformal": conf, "reference": REFERENCE}
    p = save_json(out, results_path_scale("s2", args.pde, "results.json", args.tag))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
