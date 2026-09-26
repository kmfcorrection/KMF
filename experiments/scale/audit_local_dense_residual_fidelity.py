#!/usr/bin/env python3
"""Audit residual proxies on the fully local dense-time NS development corpus.

The stored next state is *only* an offline reference.  Every proxy receives a
current vorticity field and a candidate next state, exactly as it would at
inference.  We report whether a proxy approximates the reference flow defect

    d_star(x, y) = y - Phi_dt(x),

where ``Phi_dt`` is represented by the densely stored, high-accuracy solver
trajectory.  Thus this is a residual-fidelity test, not a correction result.

Methods are all matched to the local forced 2-D vorticity equation:
dealiased pseudospectral midpoint, centered finite differences (2/4/6), a
finite-volume flux form, weak Fourier/Galerkin residuals, fixed-step SSP-RK3,
coarse Smagorinsky LES, and calibration-only POD--Galerkin ROMs.
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

from hipp.scale.data2d import PDESpec2D, SpectralGrid2D, ifrk4_step
from hipp.scale.fm_physics import spectral_resample_state


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cadences", type=float, nargs="+", default=[.01, .05, .1])
    p.add_argument("--n-cal-traj", type=int, default=16)
    p.add_argument("--n-test-traj", type=int, default=8)
    p.add_argument("--transitions-per-traj", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--fd-orders", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--weak-modes", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--rk3-substeps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--les-resolutions", type=int, nargs="+", default=[16, 32])
    p.add_argument("--les-cs", type=float, default=.17)
    p.add_argument("--les-substeps", type=int, default=8)
    p.add_argument("--pod-ranks", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--pod-substeps", type=int, default=8)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260909)
    return p


def _load(root: Path, split: str, n: int) -> torch.Tensor:
    a = np.load(root / f"{split}.npy", mmap_mode="r")
    if n > len(a):
        raise ValueError(f"requested {n} {split} trajectories; dataset has {len(a)}")
    return torch.from_numpy(np.array(a[:n], dtype=np.float64))


def _spec(root: Path) -> PDESpec2D:
    data = json.loads((root / "manifest.json").read_text())
    return PDESpec2D(**data["spec"])


def _velocity(grid: SpectralGrid2D, w: torch.Tensor) -> torch.Tensor:
    u, v = grid.velocity(torch.fft.rfft2(w))
    return torch.stack((u, v), dim=1)


def _rhs(grid: SpectralGrid2D, w: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfft2(grid.rhs_hat(torch.fft.rfft2(w)), s=(grid.spec.n, grid.spec.n))


def _d1(x: torch.Tensor, axis: int, dx: float, order: int) -> torch.Tensor:
    coeffs = {
        2: ((-1, -.5), (1, .5)),
        4: ((-2, 1 / 12), (-1, -2 / 3), (1, 2 / 3), (2, -1 / 12)),
        6: ((-3, -1 / 60), (-2, 9 / 60), (-1, -45 / 60),
            (1, 45 / 60), (2, -9 / 60), (3, 1 / 60)),
    }[order]
    dim = -2 if axis == 0 else -1
    return sum(c * torch.roll(x, -o, dim) for o, c in coeffs) / dx


def _d2(x: torch.Tensor, axis: int, dx: float, order: int) -> torch.Tensor:
    coeffs = {
        2: ((-1, 1.), (0, -2.), (1, 1.)),
        4: ((-2, -1 / 12), (-1, 4 / 3), (0, -5 / 2), (1, 4 / 3), (2, -1 / 12)),
        6: ((-3, 1 / 90), (-2, -3 / 20), (-1, 3 / 2), (0, -49 / 18),
            (1, 3 / 2), (2, -3 / 20), (3, 1 / 90)),
    }[order]
    dim = -2 if axis == 0 else -1
    return sum(c * torch.roll(x, -o, dim) for o, c in coeffs) / dx**2


def _fd_residual(grid: SpectralGrid2D, previous: torch.Tensor, candidate: torch.Tensor,
                 dt: float, order: int) -> torch.Tensor:
    """Strong-form FD defect for *the same* forced NS equation as the corpus."""
    dx = grid.spec.L / grid.spec.n
    mid = .5 * (previous + candidate)
    velocity = _velocity(grid, mid)
    adv = velocity[:, 0] * _d1(mid, 0, dx, order) + velocity[:, 1] * _d1(mid, 1, dx, order)
    lap = _d2(mid, 0, dx, order) + _d2(mid, 1, dx, order)
    forcing = grid.forcing.unsqueeze(0) if grid.forcing is not None else 0.0
    return (candidate - previous) / dt + adv - grid.spec.nu * lap - forcing


def _fv_residual(grid: SpectralGrid2D, previous: torch.Tensor, candidate: torch.Tensor,
                 dt: float) -> torch.Tensor:
    dx = grid.spec.L / grid.spec.n
    mid = .5 * (previous + candidate)
    velocity = _velocity(grid, mid)
    u, v = velocity[:, 0], velocity[:, 1]
    fx = .25 * (u + torch.roll(u, -1, -2)) * (mid + torch.roll(mid, -1, -2))
    fy = .25 * (v + torch.roll(v, -1, -1)) * (mid + torch.roll(mid, -1, -1))
    div = ((fx - torch.roll(fx, 1, -2)) + (fy - torch.roll(fy, 1, -1))) / dx
    lap = _d2(mid, 0, dx, 2) + _d2(mid, 1, dx, 2)
    forcing = grid.forcing.unsqueeze(0) if grid.forcing is not None else 0.0
    return (candidate - previous) / dt + div - grid.spec.nu * lap - forcing


def _spectral_residual(grid: SpectralGrid2D, previous: torch.Tensor, candidate: torch.Tensor,
                       dt: float) -> torch.Tensor:
    mid = .5 * (previous + candidate)
    return (candidate - previous) / dt - _rhs(grid, mid)


def _lowpass(field: torch.Tensor, modes: int) -> torch.Tensor:
    n = field.shape[-1]
    low = spectral_resample_state(field[:, None], modes)
    return spectral_resample_state(low, n)[:, 0]


def _ssprk3(grid: SpectralGrid2D, initial: torch.Tensor, dt: float, substeps: int) -> torch.Tensor:
    h, w = dt / substeps, initial
    for _ in range(substeps):
        w1 = w + h * _rhs(grid, w)
        w2 = .75 * w + .25 * (w1 + h * _rhs(grid, w1))
        w = w / 3 + (2 / 3) * (w2 + h * _rhs(grid, w2))
    return w


def _les_endpoint(spec: PDESpec2D, initial: torch.Tensor, dt: float, coarse_n: int,
                  cs: float, substeps: int) -> torch.Tensor:
    # A conventional coarse Smagorinsky closure.  It remains a solver control,
    # not a learned residual estimator.
    coarse_spec = PDESpec2D(**{**asdict(spec), "n": coarse_n})
    grid = SpectralGrid2D(coarse_spec, dtype=torch.float64)
    w = spectral_resample_state(initial[:, None], coarse_n)[:, 0]
    h = dt / substeps
    dx = coarse_spec.L / coarse_n
    for _ in range(substeps):
        def rhs_les(a: torch.Tensor) -> torch.Tensor:
            vel = _velocity(grid, a)
            ux = torch.fft.irfft2(1j * grid.kx * torch.fft.rfft2(vel[:, 0]), s=(coarse_n, coarse_n))
            uy = torch.fft.irfft2(1j * grid.ky * torch.fft.rfft2(vel[:, 0]), s=(coarse_n, coarse_n))
            vx = torch.fft.irfft2(1j * grid.kx * torch.fft.rfft2(vel[:, 1]), s=(coarse_n, coarse_n))
            vy = torch.fft.irfft2(1j * grid.ky * torch.fft.rfft2(vel[:, 1]), s=(coarse_n, coarse_n))
            strain = torch.sqrt((2 * ux.square() + 2 * vy.square() + (uy + vx).square()).clamp_min(0))
            nu_t = (cs * dx) ** 2 * strain
            ax = _d1(a, 0, dx, 4); ay = _d1(a, 1, dx, 4)
            closure = _d1(nu_t * ax, 0, dx, 4) + _d1(nu_t * ay, 1, dx, 4)
            return _rhs(grid, a) + closure
        a1 = w + h * rhs_les(w)
        a2 = .75 * w + .25 * (a1 + h * rhs_les(a1))
        w = w / 3 + (2 / 3) * (a2 + h * rhs_les(a2))
    return spectral_resample_state(w[:, None], spec.n)[:, 0]


def _fit_pod(cal: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = cal.reshape(-1, cal.shape[-2] * cal.shape[-1])
    mean = x.mean(0)
    _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
    return mean, vh[:rank].T.contiguous()


def _pod_endpoint(grid: SpectralGrid2D, initial: torch.Tensor, dt: float,
                  mean: torch.Tensor, basis: torch.Tensor, substeps: int) -> torch.Tensor:
    a = (initial.reshape(len(initial), -1) - mean) @ basis
    h = dt / substeps
    def rhs(c: torch.Tensor) -> torch.Tensor:
        w = (mean + c @ basis.T).reshape(-1, grid.spec.n, grid.spec.n)
        return _rhs(grid, w).reshape(len(c), -1) @ basis
    for _ in range(substeps):
        a1 = a + h * rhs(a)
        a2 = .75 * a + .25 * (a1 + h * rhs(a1))
        a = a / 3 + (2 / 3) * (a2 + h * rhs(a2))
    return (mean + a @ basis.T).reshape_as(initial)


def _metrics(proxy: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    p, r = proxy.flatten(1), reference.flatten(1)
    rel = (p - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-14)
    cos = (p * r).sum(1) / (p.norm(dim=1) * r.norm(dim=1)).clamp_min(1e-14)
    return {"relative_defect_error": float(rel.mean()), "direction_cosine": float(cos.mean()),
            "proxy_rms": float(proxy.square().mean().sqrt()),
            "reference_rms": float(reference.square().mean().sqrt())}


def _candidate(grid: SpectralGrid2D, previous: torch.Tensor, dt: float, kind: str) -> torch.Tensor:
    if kind == "persistence":
        return previous
    if kind == "wrong_physics":
        bad = PDESpec2D(**{**asdict(grid.spec), "nu": grid.spec.nu * 4,
                           "forcing_amp": grid.spec.forcing_amp * .5})
        bad_grid = SpectralGrid2D(bad, dtype=torch.float64)
        wh = torch.fft.rfft2(previous)
        # Integrating-factor step with intentionally wrong coefficients.
        return torch.fft.irfft2(ifrk4_step(bad_grid, wh, dt), s=(bad.n, bad.n))
    raise ValueError(kind)


def _pairs(traj: torch.Tensor, stride: int, per_traj: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.linspace(0, traj.shape[1] - stride - 1, min(per_traj, traj.shape[1] - stride), dtype=int)
    previous = torch.cat([traj[:, t] for t in starts])
    truth = torch.cat([traj[:, t + stride] for t in starts])
    return previous, truth


def main() -> None:
    args = _parser().parse_args()
    torch.manual_seed(args.seed)
    root = args.data_root.expanduser().resolve()
    spec = _spec(root)
    cal = _load(root, "train", args.n_cal_traj)
    test = _load(root, "test", args.n_test_traj)
    stored_dt = spec.dt_out
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"Local dense residual-fidelity audit: {root.name}")
    print("=" * 78)
    print(f"PDE=forced 2D NS; state=vorticity; reference stored dt={stored_dt:g}; "
          f"test trajectories={len(test)}")
    print("Future truth forms only offline d_star=candidate-truth; no proxy receives it.")
    payload: dict[str, object] = {"dataset": str(root), "stored_dt": stored_dt, "cadences": {}}

    for dt in args.cadences:
        stride = round(dt / stored_dt)
        if not math.isclose(dt, stride * stored_dt, abs_tol=1e-12):
            raise ValueError(f"cadence {dt} is not a multiple of stored dt {stored_dt}")
        if stride >= test.shape[1]:
            raise ValueError(f"cadence {dt} exceeds trajectory duration")
        local_spec = PDESpec2D(**{**asdict(spec), "stride": stride})
        grid = SpectralGrid2D(local_spec, dtype=torch.float64)
        previous, truth = _pairs(test, stride, args.transitions_per_traj)
        pod_snapshots, _ = _pairs(cal, stride, args.transitions_per_traj)
        max_rank = min(max(args.pod_ranks), len(pod_snapshots), local_spec.N)
        pod_mean, pod_basis = _fit_pod(pod_snapshots, max_rank)
        print(f"\n-- cadence dt={dt:g}: {len(previous)} held-out transitions --")
        by_candidate: dict[str, dict[str, dict[str, float]]] = {}
        for candidate_name in ("truth", "persistence", "wrong_physics"):
            candidate = truth if candidate_name == "truth" else _candidate(grid, previous, dt, candidate_name)
            reference = candidate - truth
            methods: dict[str, torch.Tensor] = {
                "spectral_midpoint": dt * _spectral_residual(grid, previous, candidate, dt),
                "finite_volume_2": dt * _fv_residual(grid, previous, candidate, dt),
            }
            for order in args.fd_orders:
                methods[f"finite_difference_{order}"] = dt * _fd_residual(grid, previous, candidate, dt, order)
            full = methods["spectral_midpoint"]
            for modes in args.weak_modes:
                methods[f"weak_galerkin_{modes}"] = _lowpass(full, modes)
            for substeps in args.rk3_substeps:
                methods[f"ssprk3_{substeps}"] = candidate - _ssprk3(grid, previous, dt, substeps)
            for resolution in args.les_resolutions:
                methods[f"les_smag_{resolution}"] = candidate - _les_endpoint(
                    local_spec, previous, dt, resolution, args.les_cs, args.les_substeps)
            for rank in args.pod_ranks:
                methods[f"pod_galerkin_{rank}"] = candidate - _pod_endpoint(
                    grid, previous, dt, pod_mean, pod_basis[:, :rank], args.pod_substeps)
            by_candidate[candidate_name] = {name: _metrics(proxy, reference)
                                            for name, proxy in methods.items()}
        # Truth has zero reference defect, so report its raw residual magnitude
        # separately; relative errors are undefined there.
        truth_scores = {}
        for name in by_candidate["truth"]:
            proxy_name = name
            candidate = truth
            if proxy_name == "spectral_midpoint": proxy = dt * _spectral_residual(grid, previous, candidate, dt)
            elif proxy_name == "finite_volume_2": proxy = dt * _fv_residual(grid, previous, candidate, dt)
            elif proxy_name.startswith("finite_difference_"):
                proxy = dt * _fd_residual(grid, previous, candidate, dt, int(proxy_name.rsplit("_", 1)[1]))
            elif proxy_name.startswith("weak_galerkin_"):
                proxy = _lowpass(dt * _spectral_residual(grid, previous, candidate, dt), int(proxy_name.rsplit("_", 1)[1]))
            elif proxy_name.startswith("ssprk3_"):
                proxy = candidate - _ssprk3(grid, previous, dt, int(proxy_name.rsplit("_", 1)[1]))
            elif proxy_name.startswith("les_smag_"):
                proxy = candidate - _les_endpoint(local_spec, previous, dt, int(proxy_name.rsplit("_", 1)[1]), args.les_cs, args.les_substeps)
            else:
                proxy = candidate - _pod_endpoint(grid, previous, dt, pod_mean, pod_basis[:, :int(proxy_name.rsplit("_", 1)[1])], args.pod_substeps)
            truth_scores[name] = float(proxy.square().mean().sqrt())
        # Candidate relative error / cosine report, plus an inference-relevant
        # discrimination ratio.  A useful residual has truth/candidate < 1.
        chosen = "wrong_physics"
        rows = []
        for name, metrics in by_candidate[chosen].items():
            truth_rms = truth_scores[name]
            persistence_rms = by_candidate["persistence"][name]["proxy_rms"]
            wrong_rms = metrics["proxy_rms"]
            rows.append((name, metrics["relative_defect_error"], metrics["direction_cosine"], truth_rms,
                         truth_rms / max(persistence_rms, 1e-30),
                         truth_rms / max(wrong_rms, 1e-30)))
        rows.sort(key=lambda x: x[1])
        print("method                      defect rel.err  cosine   truth/persist truth/wrong")
        for name, rel, cos, truth_rms, persist_ratio, wrong_ratio in rows:
            print(f"{name:26s} {rel:>10.4g} {cos:>8.4f} {persist_ratio:>13.4g} {wrong_ratio:>11.4g}")
        payload["cadences"][str(dt)] = {"stride": stride, "n_test_transitions": len(previous),
                                         "truth_proxy_rms": truth_scores,
                                         "candidate_metrics": by_candidate}

    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    # Plot only the inference-relevant wrong-physics candidate across cadence.
    methods = list(next(iter(payload["cadences"].values()))["candidate_metrics"]["wrong_physics"])
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(methods) * .75), 4.5))
    for cadence, value in payload["cadences"].items():
        met = value["candidate_metrics"]["wrong_physics"]
        axes[0].plot(range(len(methods)), [met[m]["relative_defect_error"] for m in methods], marker="o", label=f"dt={cadence}")
        axes[1].plot(range(len(methods)), [met[m]["direction_cosine"] for m in methods], marker="o", label=f"dt={cadence}")
    axes[0].set_title("Proxy defect relative error (lower is better)")
    axes[1].set_title("Proxy/reference defect cosine (higher is better)")
    for ax in axes:
        ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "residual_fidelity_by_cadence.png", dpi=180)
    print(f"\nwrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'residual_fidelity_by_cadence.png'}")


if __name__ == "__main__":
    main()
