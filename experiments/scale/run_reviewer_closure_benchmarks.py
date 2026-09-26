#!/usr/bin/env python3
"""Reviewer-closure experiments for KMF on verified Poseidon fluid settings.

This script intentionally separates two information contracts:
  * KMF uses the observed pair (u0, u1) for endpoint-conditioned reconstruction.
  * Explicit RK controls use u0 only and are causal solver controls.

It produces four pre-registered reviewer experiments without claiming that a
causal solver and a two-sided reconstructor have identical information:
  E1 stable-solver accuracy/latency Pareto;
  E2 operator-mismatch robustness;
  E3 calibration-size resampling on both fluid datasets;
  E4 empirical cadence scaling on turbulent trajectories.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.cross_fm_benchmark import (
    GenericPDEPhysics,
    fm_predict,
    load_dataset_trajectories,
)
from experiments.scale.fm_eval_common import native_poseidon_spec
from hipp.scale.common_scale import load_fm, set_seed
from hipp.scale.data2d import SPECS2D
from hipp.scale.fm_physics import FMPhysicsEnergy2D


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rmse_per_trajectory(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    return pred.sub(truth).square().mean(dim=(1, 2, 3)).sqrt().detach().cpu().numpy()


def _project(op: FMPhysicsEnergy2D, x: torch.Tensor) -> torch.Tensor:
    return op.project_incompressible(x).reshape_as(x)


def _state_from_vorticity(op: FMPhysicsEnergy2D, w: torch.Tensor,
                          reference: torch.Tensor) -> torch.Tensor:
    """Lift vorticity to velocity, preserving the reference's mean velocity."""
    u, v = op.grid.velocity(torch.fft.rfft2(w))
    u = u + reference[:, 0].mean(dim=(-2, -1), keepdim=True)
    v = v + reference[:, 1].mean(dim=(-2, -1), keepdim=True)
    return torch.stack((u, v), dim=1)


def _velocity_tangent(op: FMPhysicsEnergy2D, state: torch.Tensor,
                      rhs: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    wdot = rhs(op.to_vorticity(state))
    du, dv = op.grid.velocity(torch.fft.rfft2(wdot))
    return torch.stack((du, dv), dim=1)


def _rhs_with_mismatch(op: FMPhysicsEnergy2D, viscosity_mult: float,
                       forcing_mult: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Known RHS with intentionally controlled coefficient mismatch.

    NS-Gauss uses the released AZEBAN smooth spectral-viscosity form; its
    effective viscosity coefficient is perturbed. FNS-KF perturbs physical
    viscosity and forcing amplitude independently.
    """
    if viscosity_mult <= 0 or forcing_mult < 0:
        raise ValueError("viscosity multiplier must be positive and forcing multiplier nonnegative")

    if op.poseidon_spectral_viscosity:
        def native_rhs(w: torch.Tensor) -> torch.Tensor:
            wh = torch.fft.rfft2(w)
            adv = op.grid.advection(wh)
            mode = op.grid.k2.sqrt()
            cutoff = 2 * math.pi * math.sqrt(op.n) / op.spec.L
            filt = 1.0 - torch.exp(-(mode / cutoff).pow(18))
            eps = viscosity_mult * 0.05 / op.n
            return torch.fft.irfft2(adv - eps * op.grid.k2 * filt * wh, s=(op.n, op.n))
        return native_rhs

    def generic_rhs(w: torch.Tensor) -> torch.Tensor:
        wh = torch.fft.rfft2(w)
        out = op.grid.advection(wh)
        out = out - viscosity_mult * op.spec.nu * op.grid.k2 * wh
        if op.spec.drag:
            out = out - op.spec.drag * wh
        if op.grid.forcing is not None:
            out = out + forcing_mult * torch.fft.rfft2(op.grid.forcing)
        return torch.fft.irfft2(out, s=(op.n, op.n))
    return generic_rhs


def _bridge(op: FMPhysicsEnergy2D, u0: torch.Tensor, u1: torch.Tensor,
            dt: float, rhs: Callable[[torch.Tensor], torch.Tensor],
            s: float = 0.5) -> torch.Tensor:
    """Two-sided physical Hermite bridge; algebraic, never a flow solve."""
    f0 = _velocity_tangent(op, u0, rhs)
    f1 = _velocity_tangent(op, u1, rhs)
    h00 = 1 - 3 * s * s + 2 * s * s * s
    h10 = 3 * s * s - 2 * s * s * s
    h01 = dt * (s - 2 * s * s + s * s * s)
    h11 = dt * (-s * s + s * s * s)
    return _project(op, h00 * u0 + h10 * u1 + h01 * f0 + h11 * f1)


def _fit_weight(bridge: torch.Tensor, fm: torch.Tensor, truth: torch.Tensor) -> float:
    d_fm = (fm - bridge).reshape(-1)
    d_truth = (truth - bridge).reshape(-1)
    denom = float(d_fm.square().sum())
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(float((d_fm * d_truth).sum()) / denom, 0.0, 1.0))


def _explicit_integrate(op: FMPhysicsEnergy2D, u0: torch.Tensor, horizon: float,
                        substeps: int, order: int,
                        rhs: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Causal explicit RK2/RK4 solver control from u0 only."""
    if order not in (2, 4) or substeps < 1:
        raise ValueError("order must be 2 or 4 and substeps must be positive")
    w = op.to_vorticity(u0)
    h = float(horizon) / substeps
    for _ in range(substeps):
        if order == 2:
            k1 = rhs(w)
            k2 = rhs(w + 0.5 * h * k1)
            w = w + h * k2
        else:
            k1 = rhs(w)
            k2 = rhs(w + 0.5 * h * k1)
            k3 = rhs(w + 0.5 * h * k2)
            k4 = rhs(w + h * k3)
            w = w + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return _project(op, _state_from_vorticity(op, w, u0))


def _timed_ms(fn: Callable[[], torch.Tensor], device: torch.device,
              warmup: int, trials: int) -> tuple[float, float]:
    for _ in range(warmup):
        out = fn()
        if not torch.isfinite(out).all():
            return float("inf"), float("nan")
    _sync(device)
    values = []
    for _ in range(trials):
        _sync(device)
        start = time.perf_counter()
        out = fn()
        _sync(device)
        if not torch.isfinite(out).all():
            return float("inf"), float("nan")
        values.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(values)), float(np.std(values))


def _load_fluid_contexts(args, fm) -> Iterable[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, FMPhysicsEnergy2D]]:
    data_root = Path(args.data_root)
    # Preserve the real-audit primary split: the first N_CAL trajectories are
    # calibration and the following N_TEST are the fixed test set.  Draw E3
    # calibration subsets only from a later, disjoint candidate pool.
    total = args.n_cal + args.n_test + args.calibration_pool
    for name in ("NS-Gauss", "FNS-KF"):
        path = data_root / f"{name}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        trajs, _ = load_dataset_trajectories(path, fm, total, steps=args.frames - 1,
                                              stride=1, offset=args.offset,
                                              target_grid=args.grid)
        if name == "NS-Gauss":
            spec = dataclasses.replace(native_poseidon_spec(args, fm), n=args.grid)
        else:
            spec = dataclasses.replace(SPECS2D.get("kolmogorov", native_poseidon_spec(args, fm)),
                                       n=args.grid, dt=0.10)
        op = FMPhysicsEnergy2D(spec, trajs[0, 0].to(fm.device, torch.float64), 2,
                               dt=0.10, device=fm.device)
        baseline_cal = trajs[:args.n_cal]
        fixed_test = trajs[args.n_cal:args.n_cal + args.n_test]
        pool = trajs[args.n_cal + args.n_test:]
        yield name, pool, baseline_cal, fixed_test, op


def _predict_fm_query(fm, u0: torch.Tensor, raw_lead: int, grid: int) -> torch.Tensor:
    """Query the FM at an integer number of stored 0.05 s frames."""
    fm.set_lead_time(float(raw_lead))
    with torch.no_grad():
        return fm_predict(fm, u0, current_grid=grid)


def _predict_fm_midpoint(fm, u0: torch.Tensor, grid: int) -> torch.Tensor:
    return _predict_fm_query(fm, u0, 1, grid)


def run_pareto(args, fm, name, cal, test, op) -> Dict[str, object]:
    dt, stride, mid = 0.10, 2, 1
    rhs = _rhs_with_mismatch(op, 1.0, 1.0)
    u0, u1, truth = test[:, 0], test[:, stride], test[:, mid]
    c0, c1, ctruth = cal[:, 0], cal[:, stride], cal[:, mid]
    fm_test = _project(op, _predict_fm_midpoint(fm, u0, args.grid))
    calibration_start = time.perf_counter()
    fm_cal = _project(op, _predict_fm_midpoint(fm, c0, args.grid))
    bridge_cal = _bridge(op, c0, c1, dt, rhs)
    weight = _fit_weight(bridge_cal, fm_cal, ctruth)
    _sync(fm.device)
    calibration_ms = (time.perf_counter() - calibration_start) * 1000.0
    bridge_test = _bridge(op, u0, u1, dt, rhs)
    fused = _project(op, (1 - weight) * bridge_test + weight * fm_test)

    rows: Dict[str, Dict[str, float | int | str | None]] = {}
    for label, pred, contract, rhs_evals in [
        ("raw_fm", fm_test, "causal FM query", 0),
        ("linear_endpoint", _project(op, 0.5 * (u0 + u1)), "two observed endpoints", 0),
        ("kmf_bridge", bridge_test, "two observed endpoints", 2),
        ("kmf_calibrated", fused, "two endpoints + FM query", 2),
    ]:
        rows[label] = {
            "rmse": float(np.mean(_rmse_per_trajectory(pred, truth))),
            "information_contract": contract,
            "rhs_evaluations": rhs_evals,
        }

    sample0, sample1 = u0[:1], u1[:1]
    rows["raw_fm"].update(dict(zip(("latency_ms", "latency_std_ms"), _timed_ms(
        lambda: _project(op, _predict_fm_midpoint(fm, sample0, args.grid)), fm.device, args.warmup, args.timing_trials))))
    rows["linear_endpoint"].update(dict(zip(("latency_ms", "latency_std_ms"), _timed_ms(
        lambda: _project(op, 0.5 * (sample0 + sample1)), fm.device, args.warmup, args.timing_trials))))
    rows["kmf_bridge"].update(dict(zip(("latency_ms", "latency_std_ms"), _timed_ms(
        lambda: _bridge(op, sample0, sample1, dt, rhs), fm.device, args.warmup, args.timing_trials))))
    rows["kmf_calibrated"].update(dict(zip(("latency_ms", "latency_std_ms"), _timed_ms(
        lambda: _project(op, (1 - weight) * _bridge(op, sample0, sample1, dt, rhs) + weight * _project(op, _predict_fm_midpoint(fm, sample0, args.grid))),
        fm.device, args.warmup, args.timing_trials))))

    for order in (2, 4):
        for substeps in args.solver_substeps:
            label = f"rk{order}_{substeps}substeps"
            try:
                pred = _explicit_integrate(op, u0, dt / 2, substeps, order, rhs)
                finite = bool(torch.isfinite(pred).all())
                rms = float(np.mean(_rmse_per_trajectory(pred, truth))) if finite else None
                latency, latency_std = _timed_ms(
                    lambda: _explicit_integrate(op, sample0, dt / 2, substeps, order, rhs),
                    fm.device, args.warmup, args.timing_trials)
                rows[label] = {"rmse": rms, "latency_ms": latency, "latency_std_ms": latency_std,
                               "information_contract": "causal solver from u0", "rhs_evaluations": order * substeps,
                               "finite": finite}
            except RuntimeError as exc:
                rows[label] = {"rmse": None, "latency_ms": None, "latency_std_ms": None,
                               "information_contract": "causal solver from u0", "rhs_evaluations": order * substeps,
                               "finite": False, "failure": str(exc)}

    # A solver curve is interpretable only if its highest-resolution endpoint
    # replay is compatible with the dataset's governing discrete dynamics.
    replay_pred = _explicit_integrate(op, u0, dt, max(args.solver_substeps), 4, rhs)
    replay_rmse = float(np.mean(_rmse_per_trajectory(replay_pred, u1))) if bool(torch.isfinite(replay_pred).all()) else None

    endpoint_labels = ["raw_fm", "linear_endpoint", "kmf_bridge", "kmf_calibrated"]
    solver_labels = [k for k in rows if k.startswith("rk")]
    print(f"\n[E1 fair endpoint-conditioned audit: {name}] weight={weight:.4f}; "
          f"offline calibration={calibration_ms:.2f} ms")
    print("  endpoint-conditioned methods: online latency excludes calibration")
    for label in endpoint_labels:
        row = rows[label]
        print(f"    {label:20s} RMSE={row.get('rmse')} latency_ms={row.get('latency_ms')}")
    print("  causal solver controls: no future endpoint or FM query")
    for label in solver_labels:
        row = rows[label]
        print(f"    {label:20s} RMSE={row.get('rmse')} latency_ms={row.get('latency_ms')}")
    return {"span_dt": dt, "query_dt": dt / 2, "weight": weight,
            "offline_calibration_ms": calibration_ms,
            "online_latency_definition": "per-query method latency; calibration excluded",
            "comparison_groups": {
                "endpoint_conditioned": endpoint_labels,
                "causal_solver_controls": solver_labels,
            },
            "methods": rows,
            "solver_replay": {"method": f"rk4_{max(args.solver_substeps)}substeps", "endpoint_span_dt": dt,
                              "endpoint_replay_rmse": replay_rmse,
                              "interpretation": "Only solver rows with a small endpoint replay error are valid numerical-baseline evidence."}}


def run_mismatch(args, fm, name, cal, test, op) -> Dict[str, object]:
    # At s=1/2 a constant additive forcing cancels exactly from a two-sided
    # Hermite bridge. Use 0.20 s spans and off-center stored queries to make a
    # forcing mismatch identifiable.
    dt, stride = 0.20, 4
    u0, u1 = test[:, 0], test[:, stride]
    c0, c1 = cal[:, 0], cal[:, stride]
    rows = []
    force_values = [1.0] if op.poseidon_spectral_viscosity else args.forcing_multipliers
    for s in args.mismatch_query_s:
        raw_lead = int(round(stride * s))
        if raw_lead <= 0 or raw_lead >= stride:
            raise ValueError("mismatch query locations must be strictly inside the endpoint span")
        truth, ctruth = test[:, raw_lead], cal[:, raw_lead]
        fm_test = _project(op, _predict_fm_query(fm, u0, raw_lead, args.grid))
        fm_cal = _project(op, _predict_fm_query(fm, c0, raw_lead, args.grid))
        baseline = _project(op, (1 - s) * u0 + s * u1)
        nominal_rhs = _rhs_with_mismatch(op, 1.0, 1.0)
        nominal_bridge_cal = _bridge(op, c0, c1, dt, nominal_rhs, s=s)
        nominal_w = _fit_weight(nominal_bridge_cal, fm_cal, ctruth)
        for visc in args.viscosity_multipliers:
            for force in force_values:
                rhs = _rhs_with_mismatch(op, visc, force)
                bridge_test = _bridge(op, u0, u1, dt, rhs, s=s)
                bridge_cal = _bridge(op, c0, c1, dt, rhs, s=s)
                recal_w = _fit_weight(bridge_cal, fm_cal, ctruth)
                nominal = _project(op, (1 - nominal_w) * bridge_test + nominal_w * fm_test)
                recal = _project(op, (1 - recal_w) * bridge_test + recal_w * fm_test)
                fm_err = _rmse_per_trajectory(fm_test, truth)
                rows.append({
                    "span_dt": dt, "query_s": s, "query_dt": s * dt,
                    "viscosity_multiplier": visc, "forcing_multiplier": force,
                    "weight_nominal": nominal_w, "weight_recalibrated": recal_w,
                    "linear_projection_rmse": float(np.mean(_rmse_per_trajectory(baseline, truth))),
                    "fm_projection_rmse": float(np.mean(fm_err)),
                    "bridge_rmse": float(np.mean(_rmse_per_trajectory(bridge_test, truth))),
                    "nominal_weight_rmse": float(np.mean(_rmse_per_trajectory(nominal, truth))),
                    "recalibrated_rmse": float(np.mean(_rmse_per_trajectory(recal, truth))),
                    "gain_recal_vs_fm_projection_pct": float(100 * (1 - np.mean(_rmse_per_trajectory(recal, truth)) / np.mean(fm_err))),
                })
    print(f"\n[E2 {name}] completed {len(rows)} operator perturbations")
    return {"span_dt": dt, "rows": rows}


def run_calibration_robustness(args, fm, name, pool, test, op) -> Dict[str, object]:
    """E3: resample calibration subsets while retaining one untouched test set."""
    dt, stride, mid = 0.10, 2, 1
    rhs = _rhs_with_mismatch(op, 1.0, 1.0)
    p0, p1, ptruth = pool[:, 0], pool[:, stride], pool[:, mid]
    t0, t1, ttruth = test[:, 0], test[:, stride], test[:, mid]
    pbridge = _bridge(op, p0, p1, dt, rhs)
    tbridge = _bridge(op, t0, t1, dt, rhs)
    pfm = _project(op, _predict_fm_midpoint(fm, p0, args.grid))
    tfm = _project(op, _predict_fm_midpoint(fm, t0, args.grid))
    bridge_error = float(np.mean(_rmse_per_trajectory(tbridge, ttruth)))
    fm_error = float(np.mean(_rmse_per_trajectory(tfm, ttruth)))
    rng = np.random.default_rng(args.seed + (0 if name == "NS-Gauss" else 1))
    rows = {}
    for budget in [1, 2, 5, 10, 20]:
        if budget > len(pool):
            continue
        weights, rmses, degradations = [], [], []
        for _ in range(args.calibration_splits):
            idx = rng.choice(len(pool), size=budget, replace=False)
            w = _fit_weight(pbridge[idx], pfm[idx], ptruth[idx])
            fused = _project(op, (1 - w) * tbridge + w * tfm)
            rmse = float(np.mean(_rmse_per_trajectory(fused, ttruth)))
            weights.append(w)
            rmses.append(rmse)
            degradations.append(rmse > min(bridge_error, fm_error))
        rows[str(budget)] = {
            "M": budget,
            "n_calibration_subsets": args.calibration_splits,
            "fixed_heldout_test_trajectories": len(test),
            "weight_median": float(np.median(weights)),
            "weight_iqr": [float(np.percentile(weights, 25)), float(np.percentile(weights, 75))],
            "heldout_rmse_mean": float(np.mean(rmses)),
            "heldout_rmse_std": float(np.std(rmses)),
            "heldout_degradation_fraction": float(np.mean(degradations)),
            "bridge_rmse_fixed_test": bridge_error,
            "fm_projection_rmse_fixed_test": fm_error,
        }
        r = rows[str(budget)]
        print(f"  [E3 {name}] M={budget:2d} w*={r['weight_median']:.3f} "
              f"IQR={r['weight_iqr']} test RMSE={r['heldout_rmse_mean']:.6f} "
              f"degradation={r['heldout_degradation_fraction']:.1%}")
    return {"candidate_pool_trajectories": len(pool), "rows": rows,
            "protocol": "calibration subsets are sampled only from the candidate pool; the evaluation set is fixed and never selected from"}


def run_cadence_scaling(args, fm, name, cal, test, op) -> Dict[str, object]:
    rows = []
    rhs = _rhs_with_mismatch(op, 1.0, 1.0)
    for dt in args.cadence_spans:
        stride = int(round(dt / 0.05))
        if stride % 2 or stride >= test.shape[1]:
            raise ValueError(f"span {dt} needs an even valid raw-frame stride; got {stride}")
        mid = stride // 2
        u0, u1, truth = test[:, 0], test[:, stride], test[:, mid]
        c0, c1, ctruth = cal[:, 0], cal[:, stride], cal[:, mid]
        bridge = _bridge(op, u0, u1, dt, rhs)
        bridge_cal = _bridge(op, c0, c1, dt, rhs)
        fm_test = _project(op, _predict_fm_midpoint(fm, u0, args.grid)) if mid == 1 else None
        fm_cal = _project(op, _predict_fm_midpoint(fm, c0, args.grid)) if mid == 1 else None
        linear = _project(op, 0.5 * (u0 + u1))
        row = {"span_dt": dt, "query_s": 0.5,
               "linear_rmse": float(np.mean(_rmse_per_trajectory(linear, truth))),
               "bridge_rmse": float(np.mean(_rmse_per_trajectory(bridge, truth)))}
        if fm_test is not None and fm_cal is not None:
            w = _fit_weight(bridge_cal, fm_cal, ctruth)
            fused = _project(op, (1 - w) * bridge + w * fm_test)
            row.update({"fm_projection_rmse": float(np.mean(_rmse_per_trajectory(fm_test, truth)),),
                        "calibrated_weight": w,
                        "calibrated_rmse": float(np.mean(_rmse_per_trajectory(fused, truth)))})
        rows.append(row)
    x = np.log(np.asarray([r["span_dt"] for r in rows]))
    y = np.log(np.asarray([r["bridge_rmse"] for r in rows]))
    slope = float(np.polyfit(x, y, 1)[0]) if len(rows) >= 2 else float("nan")
    print(f"\n[E4 {name}] empirical bridge log-log slope={slope:.3f} (diagnostic, not a C4 proof)")
    return {"rows": rows, "bridge_loglog_slope": slope,
            "interpretation": "empirical cadence scaling only; no claim of uniform turbulent C4 regularity"}


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj).__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="KMF reviewer-closure GPU benchmark suite")
    ap.add_argument("--data-root", default="data/assembled")
    ap.add_argument("--out-dir", default="results/audit_experiments/reviewer_closure")
    ap.add_argument("--fm", default="poseidon", choices=["poseidon"])
    ap.add_argument("--fm-size", default="B", choices=["T", "B", "L"])
    # The fluid datasets contain only (u, v).  Poseidon's density and pressure
    # channels are fixed constants for this corpus and must remain pinned in
    # the adapter, exactly as in the real-audit pipeline.  Leaving the common
    # loader default ("all") would instead pad the absent channels with zeros
    # and query Poseidon off its incompressible-data manifold.
    ap.add_argument("--fm-channels", default="velocity", choices=["velocity"],
                    help="Fluid state contract; keeps Poseidon rho/p channels pinned.")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--n-cal", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--calibration-pool", type=int, default=70)
    ap.add_argument("--calibration-splits", type=int, default=50)
    ap.add_argument("--frames", type=int, default=10, help="raw stored frames per trajectory to load")
    ap.add_argument("--solver-substeps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--viscosity-multipliers", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--forcing-multipliers", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--mismatch-query-s", type=float, nargs="+", default=[0.25, 0.75],
                    help="off-center locations for identifiable forcing-mismatch experiments")
    ap.add_argument("--cadence-spans", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    ap.add_argument("--timing-trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--offset", type=int, default=19760)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--experiments", nargs="+", choices=["pareto", "mismatch", "calibration", "cadence"],
                    default=["pareto", "mismatch", "calibration", "cadence"])
    args = ap.parse_args()
    if args.frames < 9:
        ap.error("--frames must be at least 9 for the 0.40 s cadence diagnostic")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fm = load_fm(args, device=torch.device(args.device))
    payload: Dict[str, object] = {"metadata": {
        "suite": "KMF reviewer-closure experiments", "model": f"poseidon_{args.fm_size}",
        "device": args.device, "grid": args.grid, "n_cal": args.n_cal, "n_test": args.n_test,
        "calibration_splits": args.calibration_splits, "seed": args.seed,
        "contracts": {"kmf": "two observed endpoints", "rk_controls": "causal u0-only solver"},
    }}
    for name, pool, cal, test, op in _load_fluid_contexts(args, fm):
        print(f"\n{'=' * 78}\nReviewer closure: {name}; cal={len(cal)}, test={len(test)}, grid={args.grid}\n{'=' * 78}")
        if "pareto" in args.experiments:
            payload[f"E1_pareto_{name}"] = run_pareto(args, fm, name, cal, test, op)
        if "mismatch" in args.experiments:
            payload[f"E2_mismatch_{name}"] = run_mismatch(args, fm, name, cal, test, op)
        if "calibration" in args.experiments:
            payload[f"E3_calibration_{name}"] = run_calibration_robustness(
                args, fm, name, pool, test, op)
        if "cadence" in args.experiments:
            payload[f"E4_cadence_{name}"] = run_cadence_scaling(args, fm, name, cal, test, op)

    result_path = out_dir / "reviewer_closure_results.json"
    result_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"\nWrote {result_path}")


if __name__ == "__main__":
    main()
