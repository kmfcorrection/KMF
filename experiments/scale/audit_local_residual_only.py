#!/usr/bin/env python3
"""Residual-only fidelity audit for the local dense Navier--Stokes corpus.

No proposed method integrates the PDE or constructs a flow endpoint.  Each
method takes only an observed current field ``x`` and a candidate FM endpoint
``y`` and returns an approximate one-step defect in state units.  The densely
stored solver target is used strictly *afterwards* as an oracle reference:

    d_star(x,y) = y - x_true_next.

The audit asks two separate questions:
  1. Does the proxy vector approximate ``d_star`` for a misspecified forecast?
  2. Does its norm score a true transition lower than a bad candidate?

Linear and Hermite paths are algebraic endpoint bridges.  Their quadrature
evaluates the PDE RHS at a fixed set of bridge nodes; it never advances a
state in time and therefore is not a numerical PDE solve.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipp.scale.data2d import PDESpec2D, SpectralGrid2D
from hipp.scale.fm_physics import spectral_resample_state


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cadences", nargs="+", type=float, default=[.01, .05, .1])
    p.add_argument("--n-cal-traj", type=int, default=32,
                   help="Calibration trajectories used only for residual-bias estimation.")
    p.add_argument("--n-test-traj", type=int, default=8)
    p.add_argument("--transitions-per-traj", type=int, default=8)
    p.add_argument("--weak-modes", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def _load_prefix(root: Path, split: str, n: int) -> torch.Tensor:
    """Load only a requested trajectory prefix from a flat or sharded corpus."""
    flat = root / f"{split}.npy"
    if flat.exists():
        values = np.load(flat, mmap_mode="r")
        if n > len(values):
            raise ValueError(f"requested {n} {split} trajectories; dataset has {len(values)}")
        return torch.from_numpy(np.array(values[:n], dtype=np.float64))
    directory = root / split
    paths = sorted(directory.glob("*.npy")) if directory.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"cannot find {split}.npy or {split}/ shards under {root}")
    blocks, remaining = [], int(n)
    for path in paths:
        if remaining <= 0:
            break
        values = np.load(path, mmap_mode="r")
        take = min(remaining, len(values))
        blocks.append(np.array(values[:take], dtype=np.float64))
        remaining -= take
    if remaining:
        raise ValueError(f"requested {n} {split} trajectories; sharded corpus has too few")
    return torch.from_numpy(np.concatenate(blocks, axis=0))


def _load_spec(root: Path, n_test: int, n_cal: int) -> tuple[PDESpec2D, torch.Tensor, torch.Tensor]:
    manifest = json.loads((root / "manifest.json").read_text())
    return (PDESpec2D(**manifest["spec"]), _load_prefix(root, "test", n_test),
            _load_prefix(root, "train", n_cal))


def _rhs(grid: SpectralGrid2D, x: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfft2(grid.rhs_hat(torch.fft.rfft2(x)), s=(grid.spec.n, grid.spec.n))


def _midpoint(grid: SpectralGrid2D, x: torch.Tensor, y: torch.Tensor, dt: float) -> torch.Tensor:
    return y - x - dt * _rhs(grid, .5 * (x + y))


def _quadrature(order: int) -> tuple[torch.Tensor, torch.Tensor]:
    if order == 2:
        return torch.tensor([.5 - 1 / math.sqrt(12), .5 + 1 / math.sqrt(12)]), torch.tensor([.5, .5])
    if order == 4:
        # Gauss--Legendre nodes/weights mapped from [-1, 1] to [0, 1].
        a, b = math.sqrt(3 / 7 + 2 / 7 * math.sqrt(6 / 5)), math.sqrt(3 / 7 - 2 / 7 * math.sqrt(6 / 5))
        nodes = [.5 * (1 - a), .5 * (1 - b), .5 * (1 + b), .5 * (1 + a)]
        wa, wb = (18 - math.sqrt(30)) / 72, (18 + math.sqrt(30)) / 72
        return torch.tensor(nodes), torch.tensor([wa, wb, wb, wa])
    raise ValueError(order)


def _linear_path(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (1 - s) * x + s * y


def _hermite_path(grid: SpectralGrid2D, x: torch.Tensor, y: torch.Tensor, dt: float,
                  s: torch.Tensor) -> torch.Tensor:
    """Cubic endpoint bridge with PDE tangents; algebraic, not an integration."""
    f0, f1 = _rhs(grid, x), _rhs(grid, y)
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * x + h10 * dt * f0 + h01 * y + h11 * dt * f1


def _path_defect(grid: SpectralGrid2D, x: torch.Tensor, y: torch.Tensor, dt: float,
                 kind: str, order: int) -> torch.Tensor:
    nodes, weights = _quadrature(order)
    integral = torch.zeros_like(x)
    for s, weight in zip(nodes, weights):
        state = _linear_path(x, y, s) if kind == "linear" else _hermite_path(grid, x, y, dt, s)
        integral += float(weight) * _rhs(grid, state)
    return y - x - dt * integral


def _phi12(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable spectral phi_1(z), phi_2(z) for an exponential defect."""
    tiny = z.abs() < 1e-7
    phi1 = torch.where(tiny, 1 + z / 2 + z.square() / 6, torch.expm1(z) / z)
    phi2 = torch.where(tiny, .5 + z / 6 + z.square() / 24,
                       (torch.expm1(z) - z) / z.square())
    return phi1, phi2


def _etd_trapezoid_defect(grid: SpectralGrid2D, x: torch.Tensor, y: torch.Tensor,
                          dt: float) -> torch.Tensor:
    """Endpoint-only exponential-trapezoid defect.

    For w'=Lw+N(w), variation of constants gives this exact identity with an
    integral of N.  We approximate that integral from N(x), N(y), but evaluate
    exp(dt L) analytically in Fourier space.  It is a defect calculation, not
    an ETD rollout.
    """
    z = dt * grid.lin_symbol
    E = torch.exp(z)
    phi1, phi2 = _phi12(z)
    xh, yh = torch.fft.rfft2(x), torch.fft.rfft2(y)
    n0, n1 = grid.nonlinear_hat(xh), grid.nonlinear_hat(yh)
    defect = yh - E * xh - dt * ((phi1 - phi2) * n0 + phi2 * n1)
    return torch.fft.irfft2(defect, s=(grid.spec.n, grid.spec.n))


def _if_path_defect(grid: SpectralGrid2D, x: torch.Tensor, y: torch.Tensor, dt: float,
                    kind: str, order: int) -> torch.Tensor:
    """Integrating-factor bridge defect with no intermediate prediction.

    The bridge is constructed algebraically in q(s)=exp(-s dt L)w(s).  Fixed
    Gauss nodes merely evaluate the RHS on that bridge; they do not step it.
    The returned defect is mapped back to the physical final-state coordinates.
    """
    z = dt * grid.lin_symbol
    E, Einv = torch.exp(z), torch.exp(-z)
    q0, q1 = torch.fft.rfft2(x), Einv * torch.fft.rfft2(y)
    n0 = grid.nonlinear_hat(q0)
    n1 = Einv * grid.nonlinear_hat(torch.fft.rfft2(y))
    nodes, weights = _quadrature(order)
    integral = torch.zeros_like(q0)
    for s, weight in zip(nodes, weights):
        if kind == "linear":
            q = (1 - s) * q0 + s * q1
        elif kind == "hermite":
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            q = h00 * q0 + h10 * dt * n0 + h01 * q1 + h11 * dt * n1
        else:
            raise ValueError(kind)
        wh = torch.exp(s * z) * q
        integrand = torch.exp(-s * z) * grid.nonlinear_hat(wh)
        integral += float(weight) * integrand
    return torch.fft.irfft2(E * (q1 - q0 - dt * integral), s=(grid.spec.n, grid.spec.n))


def _jvp_rhs(grid: SpectralGrid2D, x: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    # A single directional derivative of the known PDE RHS, not a FM Jacobian
    # and not a time integration.
    _, out = torch.func.jvp(lambda value: _rhs(grid, value), (x,), (tangent,))
    return out


def _taylor2_defect(grid: SpectralGrid2D, x: torch.Tensor, y: torch.Tensor,
                    dt: float, *, symmetric: bool) -> torch.Tensor:
    fx, fy = _rhs(grid, x), _rhs(grid, y)
    forward = y - x - dt * fx - .5 * dt * dt * _jvp_rhs(grid, x, fx)
    if not symmetric:
        return forward
    # Backward expansion of x about y, rearranged into final-state defect form.
    backward = y - x - dt * fy + .5 * dt * dt * _jvp_rhs(grid, y, fy)
    return .5 * (forward + backward)


def _midpoint_transport_defect(grid: SpectralGrid2D, x: torch.Tensor, y: torch.Tensor,
                               dt: float, bias: torch.Tensor, terms: int) -> torch.Tensor:
    """Bias-corrected, matrix-free transport of a midpoint defect.

    If r(y)-b ~= (I-dt/2 J_F(mid)) (y-y_star), a finite Neumann expansion of
    that inverse maps the residual into endpoint-error coordinates.  It uses
    ``terms`` JVPs at fixed states; it does not advance a PDE state or form a
    Jacobian matrix.  ``bias`` is fitted once from calibration truth paths.
    """
    mid = .5 * (x + y)
    value = _midpoint(grid, x, y, dt) - bias
    out = value
    for _ in range(int(terms)):
        value = .5 * dt * _jvp_rhs(grid, mid, value)
        out = out + value
    return out


def _linear_precondition(grid: SpectralGrid2D, residual: torch.Tensor, alpha: float) -> torch.Tensor:
    """Apply (alpha I-L)^{-1} exactly in Fourier space.

    This is a closed-form spectral filter, not a PDE time step.  It converts a
    multistep equation residual into an approximate final-state error while
    treating only the known linear viscous/drag operator implicitly.
    """
    rh = torch.fft.rfft2(residual)
    out = rh / (alpha - grid.lin_symbol)
    return torch.fft.irfft2(out, s=(grid.spec.n, grid.spec.n))


def _bdf2_defect(grid: SpectralGrid2D, xm1: torch.Tensor, x: torch.Tensor,
                 y: torch.Tensor, dt: float) -> torch.Tensor:
    residual = (3 * y - 4 * x + xm1) / (2 * dt) - _rhs(grid, y)
    return _linear_precondition(grid, residual, 3 / (2 * dt))


def _bdf3_defect(grid: SpectralGrid2D, xm2: torch.Tensor, xm1: torch.Tensor,
                 x: torch.Tensor, y: torch.Tensor, dt: float) -> torch.Tensor:
    residual = (11 * y - 18 * x + 9 * xm1 - 2 * xm2) / (6 * dt) - _rhs(grid, y)
    return _linear_precondition(grid, residual, 11 / (6 * dt))


def _adams_moulton3_defect(grid: SpectralGrid2D, xm1: torch.Tensor, x: torch.Tensor,
                           y: torch.Tensor, dt: float) -> torch.Tensor:
    return y - x - dt / 12 * (5 * _rhs(grid, y) + 8 * _rhs(grid, x) - _rhs(grid, xm1))


def _adams_moulton4_defect(grid: SpectralGrid2D, xm2: torch.Tensor, xm1: torch.Tensor,
                           x: torch.Tensor, y: torch.Tensor, dt: float) -> torch.Tensor:
    return y - x - dt / 24 * (9 * _rhs(grid, y) + 19 * _rhs(grid, x)
                              - 5 * _rhs(grid, xm1) + _rhs(grid, xm2))


def _lowpass(x: torch.Tensor, modes: int) -> torch.Tensor:
    n = x.shape[-1]
    return spectral_resample_state(spectral_resample_state(x[:, None], modes), n)[:, 0]


def _history_pairs(trajectories: torch.Tensor, stride: int, per_trajectory: int) -> tuple[torch.Tensor, ...]:
    """Trajectory-disjoint pairs with two causally available sparse history states."""
    first, last = 2 * stride, trajectories.shape[1] - stride - 1
    starts = np.linspace(first, last, min(per_trajectory, last - first + 1), dtype=int)
    return (torch.cat([trajectories[:, t - 2 * stride] for t in starts]),
            torch.cat([trajectories[:, t - stride] for t in starts]),
            torch.cat([trajectories[:, t] for t in starts]),
            torch.cat([trajectories[:, t + stride] for t in starts]))


def _misspecified_euler(spec: PDESpec2D, x: torch.Tensor, dt: float) -> torch.Tensor:
    """A deterministic stand-in for an imperfect FM forecast, with no solve."""
    bad_spec = PDESpec2D(**{**asdict(spec), "nu": 4 * spec.nu,
                            "forcing_amp": .5 * spec.forcing_amp})
    bad_grid = SpectralGrid2D(bad_spec, dtype=torch.float64)
    return x + dt * _rhs(bad_grid, x)


def _metrics(proxy: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p, t = proxy.flatten(1), target.flatten(1)
    rel = (p - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-14)
    cos = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-14)
    return {"relative_error": float(rel.mean()), "cosine": float(cos.mean()),
            "rms": float(proxy.square().mean().sqrt())}


def main() -> None:
    args = _parser().parse_args()
    root = args.data_root.expanduser().resolve()
    base_spec, trajectories, calibration = _load_spec(root, args.n_test_traj, args.n_cal_traj)
    stored_dt = base_spec.dt_out
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"purpose": "residual-only fidelity; no proposed method integrates a PDE",
                                  "data_root": str(root), "stored_dt": stored_dt, "cadences": {}}
    print("=" * 78)
    print(f"Residual-only local fidelity audit: {root.name}")
    print("=" * 78)
    print("Methods only evaluate a fixed RHS quadrature over endpoint bridges; no method solves Phi_dt.")

    for dt in args.cadences:
        stride = round(dt / stored_dt)
        if not math.isclose(dt, stride * stored_dt, abs_tol=1e-12):
            raise ValueError(f"dt={dt} is not an integer multiple of stored dt={stored_dt}")
        spec = PDESpec2D(**{**asdict(base_spec), "stride": stride})
        grid = SpectralGrid2D(spec, dtype=torch.float64)
        xm2, xm1, x, truth = _history_pairs(trajectories, stride, args.transitions_per_traj)
        # Calibration only: this estimates numerical quadrature bias b(dt), not
        # an endpoint predictor.  The held-out trajectory is never consulted.
        _, _, cal_x, cal_truth = _history_pairs(calibration, stride, args.transitions_per_traj)
        midpoint_bias = _midpoint(grid, cal_x, cal_truth, dt).mean(0, keepdim=True)
        candidates = {"truth": truth, "persistence": x,
                      "misspecified_euler": _misspecified_euler(spec, x, dt)}
        all_metrics: dict[str, dict[str, dict[str, float]]] = {}
        for label, y in candidates.items():
            methods = {
                "spectral_midpoint": _midpoint(grid, x, y, dt),
                "linear_GL2": _path_defect(grid, x, y, dt, "linear", 2),
                "linear_GL4": _path_defect(grid, x, y, dt, "linear", 4),
                "hermite_PDE_GL2": _path_defect(grid, x, y, dt, "hermite", 2),
                "hermite_PDE_GL4": _path_defect(grid, x, y, dt, "hermite", 4),
                "ETD_endpoint_trapezoid": _etd_trapezoid_defect(grid, x, y, dt),
                "IF_linear_GL4": _if_path_defect(grid, x, y, dt, "linear", 4),
                "IF_hermite_GL4": _if_path_defect(grid, x, y, dt, "hermite", 4),
                "Taylor2_forward": _taylor2_defect(grid, x, y, dt, symmetric=False),
                "Taylor2_symmetric": _taylor2_defect(grid, x, y, dt, symmetric=True),
                "midpoint_bias_corrected": _midpoint(grid, x, y, dt) - midpoint_bias,
                "midpoint_transport_1JVP": _midpoint_transport_defect(grid, x, y, dt, midpoint_bias, 1),
                "midpoint_transport_2JVP": _midpoint_transport_defect(grid, x, y, dt, midpoint_bias, 2),
                "midpoint_transport_4JVP": _midpoint_transport_defect(grid, x, y, dt, midpoint_bias, 4),
                "BDF2_linear_preconditioned": _bdf2_defect(grid, xm1, x, y, dt),
                "BDF3_linear_preconditioned": _bdf3_defect(grid, xm2, xm1, x, y, dt),
                "Adams_Moulton3": _adams_moulton3_defect(grid, xm1, x, y, dt),
                "Adams_Moulton4": _adams_moulton4_defect(grid, xm2, xm1, x, y, dt),
            }
            for modes in args.weak_modes:
                methods[f"weak_spectral_{modes}"] = _lowpass(methods["spectral_midpoint"], modes)
            all_metrics[label] = {name: _metrics(value, y - truth) for name, value in methods.items()}
        rows = []
        for name, metric in all_metrics["misspecified_euler"].items():
            true_rms = all_metrics["truth"][name]["rms"]
            rows.append((name, metric["relative_error"], metric["cosine"],
                         true_rms / all_metrics["persistence"][name]["rms"],
                         true_rms / metric["rms"]))
        rows.sort(key=lambda row: row[1])
        print(f"\n-- dt={dt:g}; held-out transitions={len(x)} --")
        print("method                    rel.error  cosine  truth/persist truth/misspecified")
        for name, rel, cosine, rp, rm in rows:
            print(f"{name:24s} {rel:>9.4g} {cosine:>7.4f} {rp:>14.4g} {rm:>18.4g}")
        payload["cadences"][str(dt)] = {"stride": stride, "n_transitions": len(x),
                                         "history": "two exact prior sparse states; upper bound for autoregressive use",
                                         "midpoint_bias_rms": float(midpoint_bias.square().mean().sqrt()),
                                         "metrics": all_metrics}

    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    labels = list(next(iter(payload["cadences"].values()))["metrics"]["truth"])
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.15), 4.2))
    for dt, result in payload["cadences"].items():
        q = result["metrics"]["misspecified_euler"]
        axes[0].plot(range(len(labels)), [q[name]["relative_error"] for name in labels], "o-", label=f"dt={dt}")
        axes[1].plot(range(len(labels)), [q[name]["cosine"] for name in labels], "o-", label=f"dt={dt}")
    axes[0].set_title("Residual-vector error vs oracle defect")
    axes[1].set_title("Residual-vector direction cosine")
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.grid(alpha=.2); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "residual_only_fidelity.png", dpi=180)
    print(f"\nwrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'residual_only_fidelity.png'}")


if __name__ == "__main__":
    main()
