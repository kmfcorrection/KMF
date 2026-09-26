#!/usr/bin/env python3
"""FM S2 -- calibrated one-step predictive distributions for Poseidon/DPOT."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    configure_fm_cadence,
    configure_native_poseidon_cadence,
    fit_and_score_method,
    fm_metadata,
    fm_result_key,
    split_for_s2,
)
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.jacobian import set_probe_precision
from hipp.scale.metrics_scale import (
    chi2_report,
    ellipsoid_coverage,
    maha_from_summaries,
    subspace_split_test,
)
from hipp.utils import Table, print_header, save_json, set_seed


def main() -> int:
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="+", default=["pushforward", "identity"],
                    help="uncertainty estimators to evaluate")
    ap.add_argument("--steps", type=int, default=6,
                    help="synthetic trajectory length used to draw target pairs")
    ap.add_argument("--n-traj", type=int, default=16,
                    help="minimum synthetic trajectories to generate")
    ap.add_argument("--lead-steps", type=int, default=1,
                    help="target horizon in generated trajectory snapshots")
    ap.add_argument("--sim-batch", type=int, default=4,
                    help="trajectory simulation batch size")
    ap.add_argument("--n-tail", type=int, default=8,
                    help="Hutchinson probes for discarded Jacobian trace")
    ap.add_argument("--data-source", choices=["synthetic", "poseidon-native"],
                    default="synthetic")
    ap.add_argument("--fm-data-path", default=None,
                    help="NS-Gauss velocity HDF5 file for the shared native transfer corpus")
    args = ap.parse_args()

    if args.data_source == "poseidon-native":
        if not args.fm_data_path:
            raise SystemExit("poseidon-native S2 requires --fm-data-path")
        configure_native_poseidon_cadence(args)
    else:
        configure_fm_cadence(args)
    if args.fm == "poseidon":
        if args.fm_channels == "all":
            print("NOTE: Poseidon incompressible runs should normally use "
                  "--fm-channels velocity; all-channel is a control.", flush=True)
    if args.fm not in ("poseidon", "dpot", "morph"):
        raise SystemExit("FM S2 currently supports --fm poseidon, --fm dpot, or --fm morph.")

    set_seed(args.seed)
    set_probe_precision(high=not args.allow_tf32)
    fm = load_fm(args)

    print_header(f"FM S2: {fm.info.name}  channels={args.fm_channels}  "
                 f"lead_steps={args.lead_steps}  k={args.k}")
    print("  regime     teacher-forced negative control: Sigma_input=0, so the "
          "theoretical pushforward prediction is isotropic", flush=True)
    print(f"  model      {fm}")
    if hasattr(fm, "dpot_history_policy"):
        print(f"  DPOT note  history policy = {fm.dpot_history_policy}")
    if hasattr(fm, "morph_normalization_policy"):
        print(f"  MORPH note normalization policy = {fm.morph_normalization_policy}; "
              "S2 is a geometry gate until a matched MORPH trajectory protocol is loaded")

    x_cal, y_cal, x_test, y_test, spec = split_for_s2(args, fm)
    print(f"  target     {spec.name}, grid={spec.n}, stride={spec.stride}, "
          f"pairs: cal={len(x_cal)}, test={len(x_test)}")
    if args.data_source == "poseidon-native":
        print("  split      shared Poseidon NS-Gauss corpus; calibration and test "
              "use trajectory-disjoint subsets", flush=True)
        if args.fm != "poseidon":
            print("  cadence    transfer convention: one fixed-step external-FM "
                  "output is scored against one native dt=0.1 transition", flush=True)

    joint, split, marginal, alpha, alpha_diag, seconds = {}, {}, {}, {}, {}, {}
    table = Table("method", "alpha", "D^2/N", "z", "cov90", "NLL", "CRPS",
                  "ECE", "time(s)")

    for method in args.methods:
        print(f"\n  method: {method}", flush=True)
        entry = fit_and_score_method(args, fm, method, x_cal, y_cal, x_test, y_test)
        alpha[method] = entry["alpha"]
        alpha_diag[method] = entry["alpha_diag"]
        marginal[method] = entry["marginal_agg"]
        seconds[method] = entry["seconds"]

        m = maha_from_summaries(entry["summaries"], entry["alpha"])
        joint[method] = {**chi2_report(m, fm.N), **ellipsoid_coverage(m, fm.N)}
        split[method] = subspace_split_test(entry["summaries"], entry["alpha"])
        table.add(method, entry["alpha"], joint[method]["mean_maha_over_N"],
                  joint[method]["z_mean"], joint[method]["cov_90"],
                  marginal[method]["nll"], marginal[method]["crps"],
                  marginal[method]["ece"], entry["seconds"])

    print("\nS2 summary")
    print(table)
    print("  Read z≈0 as calibrated. Lower NLL/CRPS/ECE is better. The key "
          "comparison is pushforward vs identity, not just absolute numbers.")

    out = {
        "stage": "s2_fm",
        "metadata": fm_metadata(args, fm, spec),
        "k": args.k,
        "tau_rel": args.tau_rel,
        "lead_steps": args.lead_steps,
        "n_cal": len(x_cal),
        "n_test": len(x_test),
        "methods": args.methods,
        "alpha": alpha,
        "alpha_diag": alpha_diag,
        "joint": joint,
        "subspace_split": split,
        "marginal": marginal,
        "seconds": seconds,
    }
    key = fm_result_key(args)
    p = save_json(out, results_path_scale("s2_fm", key, "results.json", args.tag))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
