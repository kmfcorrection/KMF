#!/usr/bin/env python3
"""Measure cheap physics defects directly against full adaptive AZEBAN oracle defects."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (configure_native_poseidon_cadence,
                                               load_poseidon_trajectories, native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm
from hipp.scale.fm_physics import NativeFlowEnergy2D, spectral_resample_state
from hipp.scale.traditional_residuals import (fixed_step_azeban_endpoint,
    fit_pod_vorticity_basis, pod_galerkin_endpoint, smagorinsky_les_endpoint)
from hipp.utils import set_seed


def _cos(a, b):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-30)


def _endpoint_candidates(spec, current, *, fixed_steps, les_resolutions, les_cs,
                         les_substeps, pod_model, pod_substeps, device):
    """Classical endpoints; no future state is consumed here."""
    b, n = current.shape[0], spec.n
    out = {}
    for steps in fixed_steps:
        out[f"fixed_rk3_{steps}"] = fixed_step_azeban_endpoint(
            spec, current, dt=spec.dt_out, substeps=steps, device=device)
    for h in les_resolutions:
        low = smagorinsky_les_endpoint(spec, current, coarse_n=h, dt=spec.dt_out,
                                       cs=les_cs, substeps=les_substeps, device=device)
        out[f"les_smag_{h}"] = spectral_resample_state(low, n).reshape(b, -1)
    if pod_model is not None:
        mean, bases = pod_model
        for rank, basis in bases.items():
            out[f"pod_galerkin_{rank}"] = pod_galerkin_endpoint(
                spec, current, dt=spec.dt_out, mean=mean, basis=basis,
                substeps=pod_substeps, device=device).reshape(b, -1)
    return out


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=16)
    ap.add_argument("--n-test-traj", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fixed-rk3-substeps", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    ap.add_argument("--les-resolutions", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--les-cs", type=float, default=0.17)
    ap.add_argument("--les-substeps", type=int, default=16)
    ap.add_argument("--pod-ranks", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--pod-substeps", type=int, default=16)
    ap.add_argument("--out-dir", default="results/figures/oracle_residual_fidelity")
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.fm != "poseidon":
        raise SystemExit("this oracle audit is currently defined for native Poseidon velocity")
    set_seed(args.seed)
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    if args.lead_steps != 1:
        spec = type(spec)(**{**spec.__dict__, "dt": spec.dt * args.lead_steps})
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj, 1,
                                     args.lead_steps, offset=0)
    test = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj, 1,
                                      args.lead_steps, offset=args.n_cal_traj)
    print(f"Oracle residual-fidelity audit: {fm.info.name}; test transitions={len(test)}")
    snapshots = cal[:, 0]
    pod_model = None
    if args.pod_ranks:
        mean, basis = fit_pod_vorticity_basis(spec, snapshots, max(args.pod_ranks), fm.device)
        pod_model = (mean, {r: basis[:, :r] for r in args.pod_ranks})

    buckets = {}
    for start in range(0, len(test), args.batch):
        stop = min(start + args.batch, len(test))
        current = test[start:stop, 0]
        print(f"  test transitions {start + 1}-{stop}/{len(test)}")
        raw = torch.stack([fm.predict(x).double().reshape(-1) for x in current])
        # This is the evaluation-only high-fidelity defect.  It is never used
        # by a candidate method at inference.
        oracle = NativeFlowEnergy2D(spec, current, 2, divergence_weight=0.0,
                                    flow_cfl=0.5, dt=spec.dt_out, device=fm.device).flow_target
        oracle_defect = raw - oracle
        endpoints = _endpoint_candidates(
            spec, current, fixed_steps=args.fixed_rk3_substeps,
            les_resolutions=args.les_resolutions, les_cs=args.les_cs,
            les_substeps=args.les_substeps, pod_model=pod_model,
            pod_substeps=args.pod_substeps, device=fm.device)
        for name, endpoint in endpoints.items():
            candidate_defect = raw - endpoint
            endpoint_rel = (endpoint - oracle).norm(dim=1) / oracle.norm(dim=1).clamp(min=1e-30)
            defect_rel = (candidate_defect - oracle_defect).norm(dim=1) / oracle_defect.norm(dim=1).clamp(min=1e-30)
            values = buckets.setdefault(name, {"endpoint_rel": [], "defect_rel": [], "defect_cos": []})
            values["endpoint_rel"].append(endpoint_rel.detach().cpu())
            values["defect_rel"].append(defect_rel.detach().cpu())
            values["defect_cos"].append(_cos(candidate_defect, oracle_defect).detach().cpu())

    summary = {}
    for name, values in buckets.items():
        summary[name] = {}
        for metric, chunks in values.items():
            x = torch.cat(chunks).numpy()
            summary[name][f"{metric}_mean"] = float(np.mean(x))
            summary[name][f"{metric}_p90"] = float(np.quantile(x, .9))
    ordered = sorted(summary, key=lambda key: summary[key]["defect_rel_mean"])
    print("\nmethod                         defect rel. mean   p90     direction cosine")
    for name in ordered:
        q = summary[name]
        print(f"{name:28s} {q['defect_rel_mean']:>10.4g} {q['defect_rel_p90']:>8.4g} {q['defect_cos_mean']:>12.5f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    labels = ordered
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.25 * len(labels)), 4.6))
    axes[0].bar(range(len(labels)), [summary[x]["defect_rel_mean"] for x in labels])
    axes[0].set_title(r"Cheap defect error: $\|d_m-d_\star\|/\|d_\star\|$ (lower is better)")
    axes[1].bar(range(len(labels)), [summary[x]["defect_cos_mean"] for x in labels])
    axes[1].axhline(1, color="black", ls="--", lw=.8)
    axes[1].set_title(r"Direction agreement: $\cos(d_m,d_\star)$ (higher is better)")
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out / "oracle_defect_fidelity.png", dpi=180, bbox_inches="tight")
    payload = {"oracle": "full adaptive 128x128 AZEBAN", "test_trajectories": args.n_test_traj,
               "summary": summary,
               "interpretation": "Only an endpoint with defect_rel near zero and direction cosine near one reproduces the AZEBAN defect."
               }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out / 'oracle_defect_fidelity.png'}")
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
