#!/usr/bin/env python3
"""Stress-test matrix-free GMRES transport for sparse PDE residuals.

This is a local diagnostic, not a correction method.  It deliberately builds
a deterministic, translation-equivariant surrogate forecast with large error:
a wrong Euler tendency plus a fixed spectral error operator.  The amplitude is
swept so the relevant local transport gain

    || A e || / ||e||,  A = dt/2 J_F((x+y)/2),

can exceed one.  No branch integrates a PDE endpoint.  GMRES applies only
matrix-free JVPs of the known RHS.

For each candidate it reports three distinct quantities:
  * inverse recovery: solve (I-A)z=(I-A)e;
  * truth-centered:  solve (I-A)z=R(y)-R(y*);
  * deployable:      solve (I-A)z=R(y)-b_cal.
The first two use truth solely as an offline diagnosis; the last is the only
inference-available observation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipp.scale.data2d import PDESpec2D, SpectralGrid2D
from experiments.scale.audit_local_residual_only import _history_pairs, _load_spec


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-cal-traj", type=int, default=32)
    p.add_argument("--n-test-traj", type=int, default=16)
    p.add_argument("--transitions-per-traj", type=int, default=4)
    p.add_argument("--dt", type=float, default=.1)
    p.add_argument("--error-strengths", type=float, nargs="+", default=[.1, .25, .5, 1., 2.])
    p.add_argument("--gmres-iters", type=int, default=48)
    p.add_argument("--gmres-tol", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=20260910)
    return p.parse_args()


def _rhs(grid: SpectralGrid2D, x: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfft2(grid.rhs_hat(torch.fft.rfft2(x)), s=(grid.spec.n, grid.spec.n))


def _midpoint_defect(grid: SpectralGrid2D, x: torch.Tensor, y: torch.Tensor, dt: float) -> torch.Tensor:
    return y - x - dt * _rhs(grid, .5 * (x + y))


def _surrogate_error_operator(x: torch.Tensor) -> torch.Tensor:
    """Fixed real Fourier multiplier: a simple learned-operator-like error."""
    n = x.shape[-1]
    kx = torch.fft.fftfreq(n, device=x.device, dtype=x.dtype).view(-1, 1)
    ky = torch.fft.rfftfreq(n, device=x.device, dtype=x.dtype).view(1, -1)
    radius = (kx.square() + ky.square()).sqrt()
    # Band-weighted, sign-changing multiplier. It is fixed across all samples,
    # translation equivariant, and deliberately has no dependence on truth.
    multiplier = (0.2 + 1.8 * (radius / radius.max().clamp_min(1e-30)).square())
    multiplier = multiplier * torch.cos(5. * kx - 7. * ky)
    z = torch.fft.irfft2(torch.fft.rfft2(x) * multiplier, s=(n, n))
    return z / z.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-30)[:, None, None]


def _large_error_forecast(grid: SpectralGrid2D, x: torch.Tensor, dt: float,
                          strength: float) -> torch.Tensor:
    """Causal, deliberately imperfect one-step surrogate forecast."""
    bad = PDESpec2D(**{**asdict(grid.spec), "nu": 4. * grid.spec.nu,
                         "forcing_amp": .5 * grid.spec.forcing_amp})
    bad_grid = SpectralGrid2D(bad, dtype=x.dtype, device=x.device)
    base = x + dt * _rhs(bad_grid, x)
    rms = x.flatten(1).pow(2).mean(1).sqrt()[:, None, None]
    return base + float(strength) * rms * _surrogate_error_operator(x)


def _apply_M(grid: SpectralGrid2D, midpoint: torch.Tensor, v: torch.Tensor,
             dt: float) -> torch.Tensor:
    """M v=(I-dt/2 J_F(midpoint))v, one JVP, no matrix formation."""
    _, jv = torch.func.jvp(lambda z: _rhs(grid, z), (midpoint,), (v,))
    return v - .5 * dt * jv


def _gmres(apply, rhs: torch.Tensor, maxiter: int, tol: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Unrestarted GMRES for a single field; all vectors remain matrix-free."""
    shape = rhs.shape
    b = rhs.reshape(-1)
    beta = float(b.norm())
    if beta < 1e-30:
        return torch.zeros_like(rhs), {"iterations": 0., "relative_residual": 0.}
    vectors = [b / beta]
    h = torch.zeros((maxiter + 1, maxiter), dtype=rhs.dtype, device=rhs.device)
    target = torch.zeros(maxiter + 1, dtype=rhs.dtype, device=rhs.device)
    target[0] = beta
    best, rel = torch.zeros_like(b), float("inf")
    for j in range(maxiter):
        w = apply(vectors[j].reshape(shape)).reshape(-1)
        for i in range(j + 1):
            h[i, j] = torch.dot(vectors[i], w)
            w = w - h[i, j] * vectors[i]
        # A second modified Gram--Schmidt pass is important once the transport
        # operator is non-normal.  Without it, loss of Krylov orthogonality can
        # masquerade as a failure of the physical linearization.
        for i in range(j + 1):
            correction = torch.dot(vectors[i], w)
            h[i, j] = h[i, j] + correction
            w = w - correction * vectors[i]
        h[j + 1, j] = w.norm()
        if float(h[j + 1, j]) > 1e-30 and j + 1 < maxiter:
            vectors.append(w / h[j + 1, j])
        solution = torch.linalg.lstsq(h[:j + 2, :j + 1], target[:j + 2]).solution
        basis = torch.stack(vectors[:j + 1], dim=1)
        best = basis @ solution
        rel = float((apply(best.reshape(shape)).reshape(-1) - b).norm() / beta)
        if rel <= tol or float(h[j + 1, j]) <= 1e-30:
            return best.reshape(shape), {"iterations": float(j + 1), "relative_residual": rel}
    return best.reshape(shape), {"iterations": float(maxiter), "relative_residual": rel}


def _rel_cos(x: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    a, b = x.reshape(-1), target.reshape(-1)
    return (float((a - b).norm() / b.norm().clamp_min(1e-30)),
            float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30)))


def _means(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}


def main() -> None:
    args = _args()
    if args.dt <= 0 or args.gmres_iters < 1 or args.transitions_per_traj < 1:
        raise SystemExit("dt, gmres-iters, and transitions-per-traj must be positive")
    torch.manual_seed(args.seed)
    base, test, calibration = _load_spec(args.data_root, args.n_test_traj, args.n_cal_traj)
    stride = round(args.dt / base.dt_out)
    if not math.isclose(stride * base.dt_out, args.dt, abs_tol=1e-12):
        raise SystemExit("dt must be an integer multiple of stored dataset cadence")
    spec = PDESpec2D(**{**asdict(base), "stride": stride})
    grid = SpectralGrid2D(spec, dtype=torch.float64)
    _, _, cal_x, cal_truth = _history_pairs(calibration, stride, args.transitions_per_traj)
    bias = _midpoint_defect(grid, cal_x, cal_truth, args.dt).mean(0)
    _, _, x, truth = _history_pairs(test, stride, args.transitions_per_traj)
    print("=" * 78)
    print(f"Local GMRES transport stress audit: {args.data_root.name}")
    print("=" * 78)
    print(f"  dt={args.dt:g}; pairs={len(x)}; GMRES max iterations={args.gmres_iters}; "
          f"calibration midpoint-bias RMS={float(bias.square().mean().sqrt()):.4g}")
    print("  forecast: wrong Euler tendency + fixed translation-equivariant spectral surrogate error")
    print("  diagnostics: inverse recovery / truth-centered / deployable calibration-biased residual")

    all_results = {}
    for strength in args.error_strengths:
        rows = {"inverse_recovery": [], "truth_centered": [], "deployable": [], "state": []}
        started = time.time()
        for i in range(len(x)):
            candidate = _large_error_forecast(grid, x[i:i + 1], args.dt, strength)
            error = candidate - truth[i:i + 1]
            midpoint = .5 * (x[i:i + 1] + candidate)
            apply = lambda v: _apply_M(grid, midpoint, v, args.dt)
            A_error_gain = float((error - apply(error)).norm() / error.norm().clamp_min(1e-30))
            r_raw = _midpoint_defect(grid, x[i:i + 1], candidate, args.dt)
            r_truth = _midpoint_defect(grid, x[i:i + 1], truth[i:i + 1], args.dt)
            diagnostics = {
                "inverse_recovery": apply(error),
                "truth_centered": r_raw - r_truth,
                "deployable": r_raw - bias,
            }
            for name, rhs in diagnostics.items():
                recovered, info = _gmres(apply, rhs, args.gmres_iters, args.gmres_tol)
                rel, cosine = _rel_cos(recovered, error)
                rows[name].append({"recovered_relative_error": rel,
                                   "recovered_cosine": cosine,
                                   **info})
            rows["state"].append({
                "raw_error_over_state_rms": float(error.square().mean().sqrt() /
                                                    x[i:i + 1].square().mean().sqrt().clamp_min(1e-30)),
                "A_on_error_gain": A_error_gain,
            })
        metrics = {name: _means(values) for name, values in rows.items()}
        metrics["seconds"] = time.time() - started
        all_results[str(strength)] = metrics
        s = metrics["state"]
        print(f"\n  strength={strength:g}: raw-error/state-RMS={s['raw_error_over_state_rms']:.3g}; "
              f"||A e||/||e||={s['A_on_error_gain']:.3g}")
        for name in ("inverse_recovery", "truth_centered", "deployable"):
            q = metrics[name]
            print(f"    {name:17s} residual={q['relative_residual']:.2e}; "
                  f"error={q['recovered_relative_error']:.3g}; cosine={q['recovered_cosine']:+.4f}; "
                  f"iters={q['iterations']:.1f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "GMRES diagnostic for large-J sparse residual transport; no method integrates a PDE endpoint",
        "forecast": "wrong Euler tendency plus fixed causal spectral surrogate error",
        "operator": "M=I-dt/2 J_F(midpoint), applied only via PDE-RHS JVPs",
        "calibration_bias_rms": float(bias.square().mean().sqrt()),
        "n_calibration_trajectories": args.n_cal_traj,
        "n_test_pairs": int(len(x)),
        "results": all_results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
