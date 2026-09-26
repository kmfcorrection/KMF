#!/usr/bin/env python3
"""Phase 2: controlled input-noise test of pushforward covariance.

Teacher forcing has zero input uncertainty, so it cannot validate
J Sigma_in J^T.  Here x is the distribution mean, F(x+eps) is one draw from
the induced output distribution, and F(x) is its linearized mean.  This
isolates the known input-noise effect from the FM's physical forecast residual.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    configure_native_poseidon_cadence, fm_metadata, fm_result_key, split_for_s2)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.curvature_scale import estimate_scale
from hipp.scale.metrics_scale import aggregate, marginal_report
from hipp.scale.lowrank import LowRankGaussian, randn
from hipp.utils import Table, print_header, save_json, set_seed


def _noisy(x, rel_rms, seed):
    x = x.reshape(-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    std = float(rel_rms * x.double().square().mean().sqrt())
    eps = randn(x.shape, x.dtype, x.device, g)
    eps = eps / eps.square().mean().sqrt().clamp(min=1e-30) * std
    return x + eps, std


def _prior(method, fm, observed, variance, args, seed):
    est = estimate_scale(
        fm, observed, method="pushforward", k=args.k, tau_rel=args.tau_rel,
        oversample=args.oversample, n_iter=args.n_iter, chunk=args.chunk,
        n_tail=args.n_tail, generator=torch.Generator(device="cpu").manual_seed(seed))
    pf = est.prior.rescaled(variance)
    if method == "pushforward":
        return pf
    if method == "isotropic":
        return LowRankGaussian.isotropic(pf.mean, tau=pf.trace() / pf.N,
                                         label="trace_matched_isotropic")
    raise ValueError(method)


def _score(args, fm, method, xs, ys, split, start_seed):
    summaries, marginal, noise_stds, maha_over_n = [], [], [], []
    for i, (x, physical_y) in enumerate(zip(xs, ys)):
        clean_x = x.to(fm.device, fm.dtype)
        sampled_x, std = _noisy(clean_x, args.noise_rms_rel, start_seed + i)
        # Correct predictive orientation: F(clean_x) is the mean and
        # F(clean_x+eps) is a draw.  The prior/Jacobian must be evaluated at
        # clean_x, never at the sampled perturbation.
        with torch.no_grad():
            y = (fm.predict(sampled_x) if args.target == "clean_fm"
                 else physical_y.to(fm.device, fm.dtype)).double()
        prior = _prior(method, fm, clean_x, std ** 2, args, start_seed + 10_000 + i)
        summaries.append(prior.summarize(y, extra={"split": split, "noise_std": std}))
        # QuadSummary intentionally omits alpha because the ordinary S2 fit
        # supplies one shared alpha afterward.  Here alpha=sigma_i^2 differs
        # by example, so using summary.maha_sq(1) silently drops the known
        # input-noise scale.  Score the fully scaled prior directly.
        maha_over_n.append(float(prior.mahalanobis_sq(y) / prior.N))
        marginal.append(marginal_report(prior, y))
        noise_stds.append(std)
    maha = np.asarray(maha_over_n)
    return {
        "n": len(summaries), "mean_maha_over_N": float(maha.mean()),
        "z": float((maha.mean() - 1) / (maha.std(ddof=1) / max(len(maha), 1) ** .5)) if len(maha) > 1 else float("nan"),
        "marginal": aggregate(marginal), "mean_known_noise_std": float(np.mean(noise_stds)),
    }


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="+", choices=["isotropic", "pushforward"],
                    default=["isotropic", "pushforward"])
    ap.add_argument("--noise-rms-rel", type=float, default=0.02,
                    help="known RMS of white input noise relative to each clean state RMS")
    ap.add_argument("--target", choices=["clean_fm", "physical_future"],
                    default="clean_fm",
                    help="clean_fm isolates J Sigma_in J^T; physical_future is a labelled composite diagnostic including model error Q")
    ap.add_argument("--n-tail", type=int, default=16,
                    help="Hutchinson probes for the discarded pushforward trace")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-traj", type=int, default=16)
    ap.add_argument("--data-source", choices=["poseidon-native"], default="poseidon-native")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.noise_rms_rel <= 0:
        raise SystemExit("--noise-rms-rel must be positive")
    configure_native_poseidon_cadence(args)
    set_seed(args.seed)
    fm = load_fm(args)
    if args.fm != "poseidon":
        raise SystemExit("Phase 2 is native Poseidon only; external-FM transfer is intentionally excluded.")
    print_header(f"Phase 2: injected-input S2: {fm.info.name}, k={args.k}")
    print(f"  known input noise: white, RMS={100 * args.noise_rms_rel:g}% of each state RMS")
    print(f"  target: {args.target} (mean=F(clean x), sample=F(clean x+eps))")
    print("  predicted covariance: Sigma=J Sigma_in J^T; no alpha is calibrated")
    x_cal, y_cal, x_test, y_test, spec = split_for_s2(args, fm)
    started = time.time()
    out = {"stage": "phase2_injected_s2", "metadata": fm_metadata(args, fm, spec),
           "noise": {"kind": "white", "rms_rel": args.noise_rms_rel,
                     "covariance": "per-example sigma_i^2 I"}, "methods": {}}
    out["target"] = args.target
    table = Table("method", "cal D2/N", "test D2/N", "test z", "NLL", "CRPS", "ECE")
    for j, method in enumerate(args.methods):
        print(f"\n  scoring {method}", flush=True)
        # Common random perturbations make the isotropic/pushforward comparison
        # paired rather than accidentally sensitive to different eps draws.
        cal = _score(args, fm, method, x_cal, y_cal, "cal", args.seed + 1_000_000)
        test = _score(args, fm, method, x_test, y_test, "test", args.seed + 2_000_000)
        out["methods"][method] = {"calibration": cal, "held_out": test}
        m = test["marginal"]
        table.add(method, cal["mean_maha_over_N"], test["mean_maha_over_N"], test["z"],
                  m["nll"], m["crps"], m["ece"])
    out["seconds"] = time.time() - started
    print("\n" + str(table))
    print("  A valid pushforward result has lower held-out NLL/CRPS/ECE than the trace-matched isotropic control.")
    p = save_json(out, results_path_scale("phase2_injected_s2", fm_result_key(args),
                                           "results.json", args.tag))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
