#!/usr/bin/env python3
"""Matrix-free audit of the Phase-2 pushforward covariance algebra.

For known input noise eps, checks both the local linearization
F(x+eps)-F(x) ~= J eps and whether the rank-k-plus-floor covariance assigns
J eps the expected Mahalanobis size.  This is a diagnostic gate, not a score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import configure_native_poseidon_cadence, split_for_s2
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.curvature_scale import estimate_scale
from hipp.scale.jacobian import jacobian_ops
from hipp.scale.lowrank import randn
from hipp.utils import Table, print_header, save_json, set_seed


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--split", choices=["cal", "test"], default="cal",
                    help="which Phase-2 trajectory-disjoint split to audit")
    ap.add_argument("--noise-seed-base", type=int, default=None,
                    help="optional exact perturbation seed base; Phase-2 test uses seed+2000000")
    ap.add_argument("--probe-seed-base", type=int, default=None,
                    help="optional exact randomized-SVD seed base; Phase-2 test uses seed+2010000")
    ap.add_argument("--noise-rms-rel", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--data-source", choices=["poseidon-native"], default="poseidon-native")
    ap.add_argument("--fm-data-path", required=True)
    ap.add_argument("--n-tail", type=int, default=16)
    args = ap.parse_args()
    configure_native_poseidon_cadence(args)
    set_seed(args.seed)
    fm = load_fm(args)
    if args.fm != "poseidon":
        raise SystemExit("this algebra audit is native Poseidon only")
    xcal, ycal, xtest, ytest, _ = split_for_s2(args, fm)
    xs = xcal if args.split == "cal" else xtest
    print_header(f"Phase 2 covariance algebra audit: {fm.info.name}, k={args.k}, split={args.split}")
    rows = []
    noise_base = args.noise_seed_base if args.noise_seed_base is not None else args.seed
    probe_base = args.probe_seed_base if args.probe_seed_base is not None else args.seed + 10_000
    for i, x0 in enumerate(xs[:args.states]):
        x = x0.reshape(-1).to(fm.device, fm.dtype)
        gen = torch.Generator(device="cpu").manual_seed(probe_base + i)
        est = estimate_scale(fm, x, method="pushforward", k=args.k,
                             tau_rel=args.tau_rel, oversample=args.oversample,
                             n_iter=args.n_iter, chunk=args.chunk, n_tail=args.n_tail,
                             generator=gen)
        f = fm.flat_fn()
        clean = f(x).detach().double()
        mean_mismatch = float((clean - est.prior.mean).norm() / clean.norm().clamp(min=1e-30))
        ops = jacobian_ops(f, x)
        sigma = float(args.noise_rms_rel * x.double().square().mean().sqrt())
        prior = est.prior.rescaled(sigma ** 2)
        for j in range(args.samples):
            eps = randn(x.shape, x.dtype, x.device,
                        torch.Generator(device="cpu").manual_seed(noise_base + i + j))
            eps = eps / eps.square().mean().sqrt().clamp(min=1e-30) * sigma
            jeps = ops.matvec(eps).double()
            with torch.no_grad():
                finite = f(x + eps).detach().double() - clean
            lin_rel = float((finite - jeps).norm() / jeps.norm().clamp(min=1e-30))
            def maha(v):
                return float(prior.mahalanobis_sq(prior.mean + v) / prior.N)
            rows.append({
                "state": i + 1, "sample": j + 1, "mean_mismatch": mean_mismatch,
                "linearization_rel": lin_rel,
                "Jeps_rms": float(jeps.square().mean().sqrt()),
                "finite_rms": float(finite.square().mean().sqrt()),
                "pred_rms": float((prior.trace() / prior.N) ** .5),
                "Jeps_D2_over_N": maha(jeps), "finite_D2_over_N": maha(finite),
                "Jeps_in_rank": float((prior.U.T @ jeps).square().sum() /
                                         jeps.square().sum().clamp(min=1e-300)),
            })
    table = Table("quantity", "mean", "max")
    for key in ("mean_mismatch", "linearization_rel", "Jeps_rms", "finite_rms",
                "pred_rms", "Jeps_D2_over_N", "finite_D2_over_N", "Jeps_in_rank"):
        values = [r[key] for r in rows]
        table.add(key, float(np.mean(values)), float(np.max(values)))
    print(table)
    print("  Pass conditions: mean mismatch and linearization error are small; "
          "Jeps_D2/N is near one.  Failure of the last condition isolates the "
          "rank-k covariance approximation/algebra from model forecast error.")
    out = {"stage": "phase2_pushforward_algebra_audit", "rows": rows,
           "settings": {"noise_rms_rel": args.noise_rms_rel, "k": args.k,
                        "states": args.states, "samples": args.samples,
                        "split": args.split, "noise_seed_base": noise_base,
                        "probe_seed_base": probe_base}}
    p = save_json(out, results_path_scale("phase2_pushforward_audit", "poseidon", "results.json", args.tag))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
