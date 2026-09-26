#!/usr/bin/env python3
"""s4 -- What does the result depend on? Rank, resolution, model size, architecture.

The ablation table. Four axes, each answering a question a reviewer will ask, and
each cheap to misreport:

  rank k         Does the method need the geometry, or only a scalar? The
                 controlled comparison is at *fixed trace*: `LowRankGaussian.
                 truncated` folds the discarded mass into the floor, so a
                 smaller k is not automatically sharper. Without that, rank
                 sweeps measure sharpness, not information.

  resolution     N goes as n^2, so this is the axis on which the dense method is
                 impossible and the low-rank one is flat. Reported with cost, so
                 the O(k)-vs-O(N) claim is measured rather than asserted.
                 Note that changing resolution changes the *physics resolved*,
                 not just the discretization -- a finer grid resolves a longer
                 inertial range -- so quality differences across n are not
                 purely numerical and should not be presented as such.

  model size     Does a bigger, better-trained surrogate have more or less
                 usable curvature? Both directions are plausible: a better model
                 has smaller bias (helping, since bias is the failure mode) but
                 also smaller error overall (leaving less to explain).

  architecture   FNO vs U-Net. The method's claim is about learned operators, not
                 about spectral convolutions; if the geometry result only holds
                 for FNO it is an architectural artifact and must be reported as
                 one.

Each cell runs the same calibration and scoring path as s2, so the numbers are
directly comparable across the table.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hipp.scale.common_scale import (base_parser_scale, build_zoo_streaming,
                                     results_path_scale, setup_scale)
from hipp.scale.metrics_scale import (chi2_report, maha_from_summaries,
                                      subspace_split_test)
from hipp.scale.train2d import ckpt_path
from hipp.utils import Table, print_header, save_json


def score_cell(args, ctx, estimators, baselines) -> dict:
    zoo = build_zoo_streaming(args, ctx, estimators=estimators,
                              baselines=baselines, verbose=False)
    out = {}
    for name, e in zoo.items():
        m = maha_from_summaries(e["summaries"], e["alpha"])
        rep = chi2_report(m, ctx["N"])
        sp = subspace_split_test(e["summaries"], e["alpha"])
        out[name] = {
            "alpha": e["alpha"], "z": rep["z_mean"],
            "nll": e["marginal_agg"]["nll"], "crps": e["marginal_agg"]["crps"],
            "ece": e["marginal_agg"]["ece"],
            "sharpness": e["marginal_agg"]["sharpness"],
            "floor_share": e["alpha_diag"].get("floor_share", float("nan")),
            "maha_share_in_k": sp["share_of_maha_in_subspace"],
            "seconds": e["seconds"],
        }
    return out


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--axes", nargs="+",
                    default=["rank", "resolution", "architecture"],
                    choices=["rank", "resolution", "model_size", "architecture"])
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--archs", nargs="+", default=["fno2d", "unet2d"])
    ap.add_argument("--estimators", nargs="+", default=["pushforward", "identity"])
    ap.add_argument("--baselines", nargs="+", default=["fixed_global_lowrank"])
    args = ap.parse_args()

    print_header(f"s4: scaling ablation  [{args.pde}]")
    out: dict = {"pde": args.pde, "axes": args.axes}

    # ---- rank ------------------------------------------------------------
    if "rank" in args.axes:
        print("\nAxis: retained rank k  (fixed model, fixed resolution)")
        ctx = setup_scale(args, verbose=True)
        rows = []
        t = Table("k", "k/N", "z", "NLL", "CRPS", "sharpness",
                  "maha share in k", "alpha floor share", "passes", "s/state")
        for k in args.ranks:
            a = copy.copy(args)
            a.k = k
            t0 = time.time()
            cell = score_cell(a, ctx, ["pushforward"], [])
            r = cell["pushforward"]
            passes = (2 + 3 * args.n_iter) * (k + args.oversample) + 16
            rows.append({"k": k, **r, "passes": passes})
            t.add(k, f"{k/ctx['N']:.5f}", r["z"], r["nll"], r["crps"],
                  r["sharpness"], r["maha_share_in_k"], r["floor_share"], passes,
                  (time.time() - t0) / max(len(ctx["x_test"]) + len(ctx["x_cal"]), 1))
        print(t)
        print("  Trace is conserved across k, so a smaller k is not sharper by")
        print("  construction. If NLL is flat in k, the geometry beyond the first")
        print("  few directions is not being used and the method should say so.")
        out["rank"] = rows

    # ---- resolution ------------------------------------------------------
    if "resolution" in args.axes:
        print("\nAxis: resolution  (the axis where the dense method does not exist)")
        rows = []
        t = Table("grid", "N", "z", "NLL", "CRPS", "dense passes",
                  f"rank-{args.k} passes", "speedup")
        for n in args.grids:
            a = copy.copy(args)
            a.grid = n
            try:
                ctx_n = setup_scale(a, verbose=False)
            except FileNotFoundError:
                print(f"    {n}x{n}: no checkpoint "
                      f"({ckpt_path(args.pde, args.arch, 0).name}), skipped")
                continue
            if ctx_n["fm"].state_shape[-1] != n:
                print(f"    {n}x{n}: checkpoint is "
                      f"{ctx_n['fm'].state_shape[-1]}x{ctx_n['fm'].state_shape[-1]}, "
                      f"skipped -- train one at this resolution first")
                continue
            r = score_cell(a, ctx_n, ["pushforward"], [])["pushforward"]
            N = ctx_n["N"]
            passes = (2 + 3 * args.n_iter) * (args.k + args.oversample) + 16
            rows.append({"grid": n, "N": N, **r, "dense_passes": N,
                         "lowrank_passes": passes, "speedup": N / passes})
            t.add(n, N, r["z"], r["nll"], r["crps"], N, passes, f"{N/passes:.1f}x")
        print(t)
        print("  A finer grid resolves more of the inertial range, so quality")
        print("  changes here are physical as well as numerical; the cost columns")
        print("  are the unambiguous part.")
        out["resolution"] = rows

    # ---- model size ------------------------------------------------------
    if "model_size" in args.axes:
        print("\nAxis: model size")
        rows = []
        t = Table("width", "params (M)", "val MSE", "z", "NLL", "CRPS")
        for w in args.widths:
            a = copy.copy(args)
            a.ckpt = str(ckpt_path(args.pde, args.arch, 0, tag=f"w{w}"))
            if not Path(a.ckpt).exists():
                print(f"    width {w}: no checkpoint ({Path(a.ckpt).name}), skipped")
                continue
            ctx_w = setup_scale(a, verbose=False)
            payload = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            r = score_cell(a, ctx_w, ["pushforward"], [])["pushforward"]
            rows.append({"width": w, "val_mse": payload.get("val_mse"), **r})
            t.add(w, ctx_w["fm"].info.n_params / 1e6, payload.get("val_mse"),
                  r["z"], r["nll"], r["crps"])
        print(t)
        print("  A better model has less bias (which helps -- bias is the failure")
        print("  mode) but also less error to explain. Both effects are real and")
        print("  the net direction is an empirical question, not a prediction.")
        out["model_size"] = rows

    # ---- architecture ----------------------------------------------------
    if "architecture" in args.axes:
        print("\nAxis: architecture  (is the geometry result FNO-specific?)")
        from hipp.scale.adapters import check_adapter
        rows = []
        t = Table("arch", "params (M)", "||J-I||/||I||", "z", "NLL", "CRPS",
                  "maha share in k", "beats fixed-global?")
        for arch in args.archs:
            a = copy.copy(args)
            a.arch = arch
            if not ckpt_path(args.pde, arch, 0).exists():
                print(f"    {arch}: no checkpoint, skipped")
                continue
            ctx_a = setup_scale(a, verbose=False)
            chk = check_adapter(ctx_a["fm"], verbose=False)
            cell = score_cell(a, ctx_a, ["pushforward"], ["fixed_global_lowrank"])
            r = cell["pushforward"]
            ref = cell.get("fixed_global_lowrank", {})
            beats = ("yes" if ref and r["nll"] < ref["nll"] and r["crps"] < ref["crps"]
                     else "no" if ref else "-")
            rows.append({"arch": arch, "residual_dev": chk.get("residual_dev"),
                         **r, "beats_fixed_global": beats})
            t.add(arch, ctx_a["fm"].info.n_params / 1e6,
                  chk.get("residual_dev", float("nan")), r["z"], r["nll"],
                  r["crps"], r["maha_share_in_k"], beats)
        print(t)
        print("  ||J-I|| differs a lot between these two: the U-Net's Jacobian is")
        print("  further from the identity, which raises the oracle ceiling on the")
        print("  direction test and makes it the *easier* case for the method.")
        out["architecture"] = rows

    p = save_json(out, results_path_scale("s4", args.pde, "results.json", args.tag))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
