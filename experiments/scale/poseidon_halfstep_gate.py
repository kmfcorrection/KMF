#!/usr/bin/env python3
"""Gate whether Poseidon's native time-conditioned half-step is usable.

The residual path evaluated here is generated exclusively by the frozen FM:
    x_t -> F_{0.05}(x_t) -> F_{0.05}(F_{0.05}(x_t)).
Ground-truth frames are used only after generation, to score forecast accuracy
and to test whether the resulting temporal PDE residual discriminates truth
from the FM path.  They never enter an FM call or the residual at inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (POSEIDON_RAW_DT, fm_metadata,
    fm_result_key, native_poseidon_spec)
from experiments.scale.s4_fm_dense_residual_window import DenseTemporalResidual
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.utils import Table, print_header, save_json, set_seed


def _load_three_frames(path, fm, n_traj, offset):
    import h5py
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        a = np.asarray(h[key][offset:offset + n_traj, :3, :2], dtype=np.float32)
    return torch.as_tensor(a, device=fm.device, dtype=fm.dtype).reshape(n_traj, 3, fm.N)


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--n-traj", type=int, default=32)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--divergence-weight", type=float, default=0.0)
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    args.fm, args.fm_channels, args.lead_time = "poseidon", "velocity", 1.0
    set_seed(args.seed)
    fm = load_fm(args)
    if not hasattr(fm, "set_lead_time"):
        raise SystemExit("this gate requires a time-conditioned Poseidon adapter")
    spec = native_poseidon_spec(args, fm)
    data = _load_three_frames(args.fm_data_path, fm, args.n_traj, args.offset)

    print_header(f"Poseidon FM-generated half-step gate: {fm.info.name}")
    print("  generated path: x_t -> F_lead=1(x_t) -> F_lead=1(F_lead=1(x_t))")
    print(f"  each FM half-step has physical dt={POSEIDON_RAW_DT:g}; no future truth enters generation")

    rows = []
    for i, trajectory in enumerate(data):
        x0, truth_half, truth_full = trajectory
        fm.set_lead_time(1.0)
        predicted_half = fm.predict(x0).detach().reshape(-1)
        predicted_full_two = fm.predict(predicted_half.to(fm.dtype)).detach().reshape(-1)
        fm.set_lead_time(2.0)
        predicted_full_direct = fm.predict(x0).detach().reshape(-1)
        fm.set_lead_time(1.0)

        energy = DenseTemporalResidual(spec, x0, coarse_steps=1, substeps=2,
                                       residual_scale=1.0, divergence_scale=1.0,
                                       divergence_weight=args.divergence_weight,
                                       device=fm.device)
        truth_rms = energy.rms(torch.stack((truth_half, truth_full)))
        fm_rms = energy.rms(torch.stack((predicted_half, predicted_full_two)))
        rows.append((
            float((predicted_half.double() - truth_half.double()).square().mean().sqrt()),
            float((predicted_full_two.double() - truth_full.double()).square().mean().sqrt()),
            float((predicted_full_direct.double() - truth_full.double()).square().mean().sqrt()),
            float((predicted_full_two.double() - predicted_full_direct.double()).square().mean().sqrt()),
            truth_rms, fm_rms,
        ))
        print(f"  trajectory {i + 1}/{len(data)}", flush=True)

    a = np.asarray(rows)
    truth_wins = float(np.mean(a[:, 4] < a[:, 5]))
    table = Table("half-step RMSE", "two-half full RMSE", "direct-full RMSE",
                  "two-half/direct gap", "truth residual", "FM-path residual")
    table.add(*a.mean(axis=0))
    print("\n  mean accuracy and residual audit")
    print(table)
    print(f"  residual gate: truth/FM={a[:, 4].mean() / max(a[:, 5].mean(), 1e-30):.3f}; "
          f"truth wins={100 * truth_wins:.1f}%")
    print("  PASS requires useful half-step accuracy and truth residual lower than the FM-generated path.")

    payload = {
        "stage": "poseidon_fm_generated_halfstep_gate",
        "metadata": fm_metadata(args, fm, spec),
        "n_trajectories": args.n_traj,
        "mean": dict(zip(["half_rmse", "two_half_full_rmse", "direct_full_rmse",
                             "two_half_direct_gap", "truth_residual_rms", "fm_path_residual_rms"],
                            map(float, a.mean(axis=0)))),
        "residual_truth_wins": truth_wins,
        "validity": {"generated_intermediate": "F_lead=1(x_t)",
                     "future_truth_used_in_generation": False},
    }
    path = save_json(payload, results_path_scale("poseidon_halfstep_gate", fm_result_key(args),
                                                 "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
