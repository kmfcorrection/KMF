#!/usr/bin/env python3
"""Trajectory-disjoint validation of KMF's autoregressive defect extension.

This script is deliberately separate from the Task 1 interpolation benchmark.
For each resampled calibration split, it selects alpha solely by autoregressive
calibration-rollout RMSE, freezes alpha, and evaluates one untouched test set.
It also reports a diagnostic cosine between the deployed unit correction and
the observed one-step error.  The latter uses truth only after prediction and
is never used to select or form a deployed correction.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.cross_fm_benchmark import load_dataset_trajectories, fm_predict
from experiments.scale.fm_eval_common import native_poseidon_spec
from hipp.scale.common_scale import load_fm, set_seed
from hipp.scale.data2d import SPECS2D
from hipp.scale.fm_physics import FMPhysicsEnergy2D


def project(op: FMPhysicsEnergy2D, x: torch.Tensor) -> torch.Tensor:
    return op.project_incompressible(x).reshape_as(x)


def unit_defect_update(op: FMPhysicsEnergy2D, previous: torch.Tensor,
                       raw_prediction: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return projected prediction and one unit, deployed Simpson correction.

    No truth field enters this function.  It reproduces the existing KMF
    rollout update: endpoint RHS evaluations, algebraic Hermite midpoint,
    Simpson defect, high-wavenumber taper, Biot--Savart lift, projection.
    """
    pred = project(op, raw_prediction)
    w_prev = op.to_vorticity(previous)
    w_pred = op.to_vorticity(pred)
    f_prev = op.rhs_vorticity(w_prev)
    f_pred = op.rhs_vorticity(w_pred)
    dt = float(op.dt)
    w_mid = 0.5 * (w_prev + w_pred) + (dt / 8.0) * (f_prev - f_pred)
    f_mid = op.rhs_vorticity(w_mid)
    w_simpson = w_prev + (dt / 6.0) * (f_prev + 4.0 * f_mid + f_pred)

    # Keep the established smooth spectral taper; this is part of the
    # deployed update, not a truth-dependent diagnostic.
    wh_delta = torch.fft.rfft2(w_simpson - w_pred)
    k_max = grid // 2
    taper = torch.exp(-36.0 * (op.grid.k2.sqrt() / k_max).pow(36))
    w_delta = torch.fft.irfft2(wh_delta * taper, s=(grid, grid))
    u_delta, v_delta = op.grid.velocity(torch.fft.rfft2(w_delta))
    delta = torch.stack((u_delta, v_delta), dim=1)
    return pred, delta


@torch.no_grad()
def rollout(fm: Any, op: FMPhysicsEnergy2D, trajectories: torch.Tensor, *,
            alpha: float, steps: int, stride: int, grid: int,
            collect_alignment: bool) -> dict[str, Any]:
    """Autoregressive rollout. Truth is used only to score completed states."""
    current = trajectories[:, 0].clone()
    per_step, cosines, update_norms, error_norms = [], [], [], []
    per_trajectory_sq = torch.zeros(len(trajectories), device=current.device, dtype=current.dtype)

    for step in range(steps):
        truth = trajectories[:, (step + 1) * stride]
        raw = fm_predict(fm, current, current_grid=grid)
        pred, delta = unit_defect_update(op, current, raw, grid)
        corrected = project(op, pred + alpha * delta)

        err = corrected - truth
        per_step.append(float(err.square().mean().sqrt().item()))
        per_trajectory_sq += err.square().mean(dim=(1, 2, 3))

        if collect_alignment:
            # The desired correction is truth - prediction.  This diagnostic
            # does not feed back into alpha choice or the deployed update.
            desired = truth - pred
            flat_delta = delta.reshape(len(delta), -1)
            flat_desired = desired.reshape(len(desired), -1)
            denom = flat_delta.norm(dim=1) * flat_desired.norm(dim=1)
            cosine = (flat_delta * flat_desired).sum(dim=1) / denom.clamp_min(1e-12)
            cosines.extend(cosine.detach().cpu().tolist())
            update_norms.extend(flat_delta.norm(dim=1).detach().cpu().tolist())
            error_norms.extend(flat_desired.norm(dim=1).detach().cpu().tolist())
        current = corrected

    trajectory_rmse = (per_trajectory_sq / steps).sqrt().detach().cpu().numpy()
    out: dict[str, Any] = {
        "window_rmse": float(trajectory_rmse.mean()),
        "final_step_rmse": per_step[-1],
        "per_step_rmse": per_step,
        "trajectory_window_rmse": trajectory_rmse.tolist(),
    }
    if collect_alignment:
        out["unit_update_error_cosine_mean"] = float(np.mean(cosines))
        out["unit_update_error_cosine_median"] = float(np.median(cosines))
        out["unit_update_rms"] = float(np.sqrt(np.mean(np.square(update_norms))))
        out["raw_error_rms"] = float(np.sqrt(np.mean(np.square(error_norms))))
    return out


def select_alpha(fm: Any, op: FMPhysicsEnergy2D, calibration: torch.Tensor, *,
                 alphas: list[float], steps: int, stride: int, grid: int) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for alpha in alphas:
        scores[str(alpha)] = rollout(
            fm, op, calibration, alpha=alpha, steps=steps, stride=stride,
            grid=grid, collect_alignment=False,
        )["window_rmse"]
    # Deterministic tie-breaker favors smaller interventions.
    chosen = min(alphas, key=lambda a: (scores[str(a)], a))
    return chosen, scores


def evaluate_system(args: argparse.Namespace, fm: Any, name: str, path: Path) -> dict[str, Any]:
    total = args.calibration_pool + args.n_test
    trajectories, _ = load_dataset_trajectories(
        path, fm, total, steps=args.steps * args.stride, stride=1,
        offset=args.offset, target_grid=args.grid,
    )
    pool = trajectories[:args.calibration_pool]
    test = trajectories[args.calibration_pool:]
    if name == "NS-Gauss":
        spec = dataclasses.replace(native_poseidon_spec(args, fm), n=args.grid, dt=args.dt)
    else:
        spec = dataclasses.replace(SPECS2D.get("kolmogorov", native_poseidon_spec(args, fm)), n=args.grid, dt=args.dt)
    op = FMPhysicsEnergy2D(spec, test[0, 0].to(fm.device, torch.float64), 2, dt=args.dt, device=fm.device)
    fm.set_lead_time(float(args.stride))

    rng = np.random.default_rng(args.seed + (0 if name == "NS-Gauss" else 1009))
    alphas = sorted(set([0.0] + args.alpha_grid))
    records = []
    print(f"\n{'=' * 78}\nTask 2 rollout-extension validation: {name}\n{'=' * 78}")
    print(f"calibration pool={args.calibration_pool}; calibration subset={args.n_cal}; "
          f"test={args.n_test}; splits={args.n_splits}; horizon={args.steps}; alpha grid={alphas}")

    for split in range(args.n_splits):
        idx = rng.choice(args.calibration_pool, size=args.n_cal, replace=False)
        cal = pool[torch.as_tensor(idx, device=pool.device)]
        alpha, cal_scores = select_alpha(
            fm, op, cal, alphas=alphas, steps=args.steps, stride=args.stride, grid=args.grid,
        )
        projection = rollout(fm, op, test, alpha=0.0, steps=args.steps, stride=args.stride,
                             grid=args.grid, collect_alignment=False)
        candidate = rollout(fm, op, test, alpha=alpha, steps=args.steps, stride=args.stride,
                            grid=args.grid, collect_alignment=True)
        base = np.asarray(projection["trajectory_window_rmse"])
        cand = np.asarray(candidate["trajectory_window_rmse"])
        gain = 100.0 * (1.0 - cand / np.maximum(base, 1e-12))
        record = {
            "split": split,
            "calibration_indices": idx.tolist(),
            "selected_alpha": alpha,
            "calibration_window_rmse_by_alpha": cal_scores,
            "projection_window_rmse": projection["window_rmse"],
            "candidate_window_rmse": candidate["window_rmse"],
            "mean_heldout_gain_vs_projection_pct": float(gain.mean()),
            "heldout_gain_vs_projection_pct_per_trajectory": gain.tolist(),
            "alignment": {k: candidate[k] for k in candidate if k.startswith("unit_") or k == "raw_error_rms"},
        }
        records.append(record)
        print(f"split {split + 1:2d}/{args.n_splits}: alpha={alpha:g}; "
              f"held-out gain={record['mean_heldout_gain_vs_projection_pct']:+.3f}%; "
              f"cos={record['alignment']['unit_update_error_cosine_mean']:+.3f}")

    selected = np.asarray([r["selected_alpha"] for r in records])
    gains = np.asarray([r["mean_heldout_gain_vs_projection_pct"] for r in records])
    cosines = np.asarray([r["alignment"]["unit_update_error_cosine_mean"] for r in records])
    return {
        "protocol": {
            "selection": "alpha selected only from autoregressive calibration rollouts",
            "test": "fixed held-out trajectories, never used in alpha selection",
            "alignment": "truth-only post-hoc diagnostic; never used by deployed correction",
            "baseline": "projection-only rollout (alpha=0)",
        },
        "config": {"dt": args.dt, "stride": args.stride, "horizon": args.steps,
                   "calibration_pool": args.calibration_pool, "n_cal": args.n_cal,
                   "n_test": args.n_test, "n_splits": args.n_splits, "alpha_grid": alphas},
        "records": records,
        "summary": {
            "alpha_positive_fraction": float(np.mean(selected > 0)),
            "alpha_median": float(np.median(selected)),
            "heldout_gain_mean_pct": float(gains.mean()),
            "heldout_gain_median_pct": float(np.median(gains)),
            "heldout_gain_positive_split_fraction": float(np.mean(gains > 0)),
            "alignment_cosine_mean": float(cosines.mean()),
            "alignment_cosine_median": float(np.median(cosines)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fm", default="poseidon", choices=["poseidon"])
    parser.add_argument("--fm-size", default="B", choices=["T", "B", "L"])
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--calibration-pool", type=int, default=20)
    parser.add_argument("--n-cal", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--offset", type=int, default=19760)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.n_cal > args.calibration_pool:
        raise ValueError("--n-cal must not exceed --calibration-pool")
    if abs(args.dt - 0.05 * args.stride) > 1e-9:
        raise ValueError("this dataset has raw cadence 0.05, so require dt = 0.05 * stride")

    set_seed(args.seed)
    args.fm_channels = "velocity"
    fm = load_fm(args, device=torch.device(args.device))
    root = Path(args.data_root)
    output: dict[str, Any] = {"metadata": vars(args)}
    for name in ("NS-Gauss", "FNS-KF"):
        path = root / f"{name}.nc"
        if not path.exists():
            print(f"Skipping {name}: missing {path}")
            continue
        output[name] = evaluate_system(args, fm, name, path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "task2_rollout_extension_validation.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
