#!/usr/bin/env python3
"""s5 -- Is the testbed capable of testing the hypothesis at all?

A prerequisite check that has nothing to do with the model, and the one that
should have existed before any of the others.

HILP asks whether the local geometry of a learned operator carries predictive
information. That question is only meaningful if the operator *has* geometry.
For an autoregressive surrogate of a PDE, the operator being approximated is the
true solution map over one output interval dt_out, and if dt_out is short
compared with the flow's own timescale then that map is close to the identity:

    J = I + dt_out * (linearized dynamics) + O(dt_out^2)

A near-identity J has a flat singular spectrum. Every rank-k subspace is then
equivalent, both curvature surrogates degenerate to isotropic, and the fraction
of spectral mass captured by rank k is just k/N -- the isotropic value. A null
result in that regime is a statement about the experimental design, not about
the method, and reporting it as evidence would be wrong.

This script measures the *true solver's* Jacobian directly, with no network
involved, across a sweep of dt_out. It answers:

  - at what output cadence does the solution operator stop being near-identity?
  - how much spectral mass is genuinely low-rank there?
  - which cadence should the datasets use?

It is deliberately run at a small grid: the exact Jacobian of the solver is
O(N) backward passes through `stride` RK4 steps, which is the most expensive
thing in the repository per unit of N. The conclusion is about the ratio
dt_out / (eddy turnover), which is resolution-independent to leading order.

Measured for ns2d_forced at N=144, nu=1e-4 (the numbers that set SPECS2D):

    dt_out   ||J-I||/||I||   captured trace @ k/N=0.22   cond(J)
    0.05         0.033                 0.23               1.16
    0.20         0.134                 0.28               1.79
    0.80         0.559                 0.48               8.71
    3.20         2.707                 0.91             269.4
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hipp.scale.data2d import (LEGACY_STRIDES, SPECS2D, PDESpec2D, SpectralGrid2D,
                               ifrk4_step, random_initial_vorticity)
from hipp.scale.common_scale import results_path_scale
from hipp.utils import Table, print_header, save_json, set_seed


def true_step_fn(grid: SpectralGrid2D, n: int, stride: int, dt: float):
    def step(v: torch.Tensor) -> torch.Tensor:
        h = torch.fft.rfft2(v.reshape(1, n, n))
        for _ in range(stride):
            h = ifrk4_step(grid, h, dt)
        return torch.fft.irfft2(h, s=(n, n)).reshape(-1)
    return step


def analyse(spec: PDESpec2D, n_states: int, burn: int, seed: int) -> dict:
    """Exact Jacobian spectrum of the true solution operator at this cadence."""
    n, N = spec.n, spec.n ** 2
    grid = SpectralGrid2D(spec, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    eye = torch.eye(N, dtype=torch.float64)

    rows = []
    for _ in range(n_states):
        w = random_initial_vorticity(spec, 1, generator=g, dtype=torch.float64,
                                     grid=grid)
        wh = torch.fft.rfft2(w)
        for _ in range(burn):                       # settle onto the attractor
            wh = ifrk4_step(grid, wh, spec.dt)
        w0 = torch.fft.irfft2(wh, s=(n, n)).reshape(-1)

        J = torch.autograd.functional.jacobian(
            true_step_fn(grid, n, spec.stride, spec.dt), w0, vectorize=True).double()
        S = torch.linalg.svdvals(J)
        s2 = S ** 2
        tot = float(s2.sum())
        rows.append({
            "dev_from_identity": float((J - eye).norm() / eye.norm()),
            "cond": float(S.max() / S.min().clamp(min=1e-30)),
            "captured": {f: float(s2[:max(1, int(f * N))].sum() / tot)
                         for f in (0.01, 0.05, 0.10, 0.25, 0.50)},
            "smax": float(S.max()), "smin": float(S.min()),
        })

    def avg(key):
        return float(np.mean([r[key] for r in rows]))
    return {
        "stride": spec.stride, "dt_out": spec.dt_out, "N": N,
        "dev_from_identity": avg("dev_from_identity"), "cond": avg("cond"),
        "smax": avg("smax"), "smin": avg("smin"),
        "captured": {f: float(np.mean([r["captured"][f] for r in rows]))
                     for f in rows[0]["captured"]},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pde", default="ns2d_forced", choices=list(SPECS2D))
    ap.add_argument("--grid", type=int, default=12,
                    help="grid for the exact Jacobian. Small on purpose: this "
                         "is O(N) backward passes through `stride` RK4 steps.")
    ap.add_argument("--strides", type=int, nargs="+", default=None,
                    help="output cadences to sweep; defaults to a geometric "
                         "sweep around the spec value")
    ap.add_argument("--n-states", type=int, default=2)
    ap.add_argument("--burn", type=int, default=2000,
                    help="solver steps to settle onto the attractor first")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    base = SPECS2D[args.pde]
    strides = args.strides or sorted({LEGACY_STRIDES.get(args.pde, 50),
                                      base.stride // 4, base.stride,
                                      base.stride * 2, base.stride * 6})
    strides = [s for s in strides if s >= 1]

    print_header(f"s5: does the testbed have geometry to find?  [{args.pde}]  "
                 f"grid={args.grid}x{args.grid}")
    print(f"  true solution operator, no network involved")
    print(f"  nu={base.nu:g}  dt={base.dt:g}  N={args.grid**2}  "
          f"averaged over {args.n_states} states on the attractor")

    N = args.grid ** 2
    out = []
    t = Table("stride", "dt_out", "||J-I||/||I||", "cond(J)",
              "cap@k/N=.01", "cap@.05", "cap@.10", "cap@.25", "verdict")
    for stride in strides:
        spec = PDESpec2D(**{**asdict(base), "n": args.grid, "stride": stride,
                            "warmup": 0})
        r = analyse(spec, args.n_states, args.burn, args.seed)
        # "Degenerate" means the captured trace is no better than the isotropic
        # k/N an operator with a flat spectrum gives. That is the condition
        # under which the method has nothing to find, whatever the model.
        ratio = r["captured"][0.10] / 0.10
        r["captured_over_isotropic_at_10pct"] = ratio
        verdict = ("degenerate" if ratio < 1.5 else
                   "weak" if ratio < 3 else "usable")
        r["verdict"] = verdict
        r["is_spec_default"] = (stride == base.stride)
        out.append(r)
        t.add(stride, f"{r['dt_out']:.3f}", r["dev_from_identity"], r["cond"],
              r["captured"][0.01], r["captured"][0.05], r["captured"][0.10],
              r["captured"][0.25], verdict + (" *" if r["is_spec_default"] else ""))
    print(t)
    print("  'cap@k/N=f' is the share of sum(s_i^2) inside the leading fraction f")
    print("  of directions. An isotropic operator gives exactly f, so a value")
    print("  near f means there is no geometry to exploit and every rank-k")
    print("  subspace is equivalent. '*' marks the current spec default.")

    usable = [r for r in out if r["verdict"] == "usable"]
    print(f"\nVERDICT for {args.pde}")
    cur = [r for r in out if r["is_spec_default"]]
    if cur:
        c = cur[0]
        print(f"  current spec: stride={c['stride']} dt_out={c['dt_out']:.3f} "
              f"-> {c['verdict']}  "
              f"({c['captured_over_isotropic_at_10pct']:.2f}x isotropic)")
    if usable:
        print(f"  smallest cadence with usable geometry: "
              f"stride={usable[0]['stride']} (dt_out={usable[0]['dt_out']:.3f})")
    else:
        print("  no swept cadence reaches 3x isotropic -- sweep larger strides, "
              "or this PDE/viscosity combination is not a useful testbed")

    p = save_json({"pde": args.pde, "grid": args.grid, "nu": base.nu,
                   "dt": base.dt, "spec_stride": base.stride,
                   "legacy_stride": LEGACY_STRIDES.get(args.pde),
                   "n_states": args.n_states, "rows": out},
                  results_path_scale("s5", args.pde, "results.json"))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
