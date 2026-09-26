#!/usr/bin/env python3
"""Create a small, dense-time Navier--Stokes corpus for local residual work.

This is deliberately a *development fixture*, not a paper benchmark.  It gives
us solver-generated trajectories at a cadence where temporal PDE defects can
be audited honestly.  In particular, a method evaluated against this corpus
must never use the stored future state at inference; future states are only
used offline to measure residual fidelity and forecast/correction error.

The format is the repository's standard sharded ``.npy`` trajectory format:

    <out-root>/<name>/train.npy, val.npy, test.npy, manifest.json

Each array has shape ``(trajectory, time, y, x)`` and is memory mappable, so
the corpus remains comfortable on an 8 GB laptop.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Direct ``python experiments/scale/<script>.py`` invocation sets sys.path to
# this directory rather than the repository root.  Keep this script standalone
# and make the project package import deterministic.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipp.scale.data2d import PDESpec2D, simulate


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--name", default="dense_ns2d_forced_n64_dt001")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--solver-dt", type=float, default=5e-4)
    p.add_argument("--snapshot-dt", type=float, default=1e-2)
    p.add_argument("--frames", type=int, default=101,
                   help="Stored frames per trajectory, including t=0.")
    p.add_argument("--train", type=int, default=32)
    p.add_argument("--val", type=int, default=8)
    p.add_argument("--test", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--nu", type=float, default=1e-4)
    p.add_argument("--forcing-amp", type=float, default=0.1)
    p.add_argument("--device", default="cpu", choices=("cpu",),
                   help="Float64 spectral integration is intentionally CPU-only on macOS.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate completed shards.  Existing data are otherwise reused.")
    return p


def _complete(path: Path, shape: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    try:
        return np.load(path, mmap_mode="r").shape == shape
    except Exception:
        return False


def _normalizer(train: Path) -> tuple[float, float]:
    a = np.load(train, mmap_mode="r")
    total = squared = 0.0
    count = 0
    for i in range(a.shape[0]):
        x = np.asarray(a[i])
        total += float(x.sum(dtype=np.float64))
        squared += float(np.square(x, dtype=np.float64).sum())
        count += x.size
    mean = total / count
    return mean, math.sqrt(max(squared / count - mean * mean, 1e-12))


def main() -> None:
    args = _parser().parse_args()
    stride_float = args.snapshot_dt / args.solver_dt
    stride = round(stride_float)
    if not math.isclose(stride_float, stride, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("snapshot-dt must be an integer multiple of solver-dt")
    if args.frames < 3:
        raise ValueError("frames must be at least 3")
    if min(args.train, args.val, args.test) < 1:
        raise ValueError("every split needs at least one trajectory")

    spec = PDESpec2D(
        name="local_ns2d_forced_dense",
        n=args.grid,
        nu=args.nu,
        dt=args.solver_dt,
        stride=stride,
        forcing="li",
        forcing_amp=args.forcing_amp,
        warmup=0,
        ic_peak_k=4.0,
    )
    root = args.out_root.expanduser().resolve() / args.name
    root.mkdir(parents=True, exist_ok=True)
    counts = {"train": args.train, "val": args.val, "test": args.test}
    print("=" * 78)
    print(f"Local dense NS residual-development corpus: {root}")
    print("=" * 78)
    print(f"system=forced 2D Navier--Stokes; grid={spec.n}^2; "
          f"solver dt={spec.dt:g}; stored dt={spec.dt_out:g}; frames={args.frames}")
    print("Stored future frames are offline evaluation targets only; they are not inference inputs.")

    for split_id, (split, n_traj) in enumerate(counts.items()):
        final = root / f"{split}.npy"
        shape = (n_traj, args.frames, spec.n, spec.n)
        if _complete(final, shape) and not args.force:
            print(f"  {split}: reusing complete shard {final.name}")
            continue
        final.unlink(missing_ok=True)
        part = root / f"{split}.npy.part"
        part.unlink(missing_ok=True)
        mb = np.prod(shape) * 4 / 2**20
        print(f"  {split}: generating {n_traj} trajectories ({mb:.1f} MiB)")
        data = simulate(spec, n_traj=n_traj, n_steps=args.frames - 1,
                        seed=args.seed + 1000 * split_id, device=args.device,
                        batch=args.batch, dtype=torch.float64, out_path=part)
        del data
        part.replace(final)

    mu, sigma = _normalizer(root / "train.npy")
    manifest = {
        "purpose": "local dense-time residual-development fixture; not a paper benchmark",
        "inference_rule": "future stored frames are unavailable to all residual/correction methods at inference",
        "state": "vorticity",
        "layout": "(trajectory, time, y, x), float32 .npy memory maps",
        "spec": asdict(spec),
        "solver_dt": spec.dt,
        "snapshot_dt": spec.dt_out,
        "frames": args.frames,
        "counts": counts,
        "seed": args.seed,
        "mu": mu,
        "sigma": sigma,
    }
    with open(root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {root / 'manifest.json'}")
    print(f"train normalization: mean={mu:.4e}, std={sigma:.4e}")


if __name__ == "__main__":
    main()
