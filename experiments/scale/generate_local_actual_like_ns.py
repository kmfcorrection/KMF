#!/usr/bin/env python3
"""Build a resumable, actual-protocol-scale local NS residual corpus.

The default has the same trajectory count and sparse 21-frame layout as the
released NS-Gauss corpus (20,000 trajectories), but uses a 64x64 vorticity
state so it remains feasible on a laptop CPU.  Every shard is independently
complete and atomically renamed, allowing safe interruption and resumption.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipp.scale.data2d import PDESpec2D, simulate


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument(
        "--scratch-root", type=Path,
        default=Path("/private/tmp/hilp_actual_like_ns_staging"),
        help="Local staging directory. Shards are simulated here before a buffered copy to out-root.",
    )
    p.add_argument(
        "--copy-chunk-mib", type=int, default=1,
        help="Maximum buffered write size used while copying a completed shard to out-root.",
    )
    p.add_argument("--name", default="actual_like_ns2d_forced_n64_dt01_20k")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--solver-dt", type=float, default=5e-4)
    p.add_argument("--snapshot-dt", type=float, default=.1)
    p.add_argument("--frames", type=int, default=21)
    p.add_argument("--train", type=int, default=16000)
    p.add_argument("--val", type=int, default=2000)
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--nu", type=float, default=1e-4)
    p.add_argument("--forcing-amp", type=float, default=.1)
    return p


def _is_complete(path: Path, shape: tuple[int, ...]) -> bool:
    try:
        return path.exists() and np.load(path, mmap_mode="r").shape == shape
    except Exception:
        return False


def _stats(shards: list[Path]) -> tuple[float, float, int]:
    total = squared = 0.0
    count = 0
    for path in shards:
        values = np.load(path, mmap_mode="r")
        for i in range(len(values)):
            a = np.asarray(values[i])
            total += float(a.sum(dtype=np.float64))
            squared += float(np.square(a, dtype=np.float64).sum())
            count += a.size
    mean = total / count
    return mean, math.sqrt(max(squared / count - mean * mean, 1e-12)), count


def _buffered_copy(source: Path, destination: Path, chunk_bytes: int) -> None:
    """Copy without memory-mapping the destination volume.

    This deliberately uses small sequential writes. It is slower than a direct
    mmap write to an external SSD, but avoids large USB-hub transactions and
    leaves a destination file only after the full local shard is available.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=chunk_bytes)
        dst.flush()
        os.fsync(dst.fileno())


def main() -> None:
    args = _parser().parse_args()
    stride = round(args.snapshot_dt / args.solver_dt)
    if not math.isclose(stride * args.solver_dt, args.snapshot_dt, abs_tol=1e-12):
        raise ValueError("snapshot-dt must be divisible by solver-dt")
    if min(args.train, args.val, args.test, args.shard_size, args.batch) <= 0:
        raise ValueError("counts, shard size, and batch must be positive")
    if args.copy_chunk_mib <= 0:
        raise ValueError("copy-chunk-mib must be positive")
    spec = PDESpec2D(name="local_ns2d_forced_actual_like", n=args.grid, nu=args.nu,
                     dt=args.solver_dt, stride=stride, forcing="li",
                     forcing_amp=args.forcing_amp, warmup=0, ic_peak_k=4.)
    root = args.out_root.expanduser().resolve() / args.name
    root.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch_root.expanduser().resolve() / args.name
    scratch.mkdir(parents=True, exist_ok=True)
    copy_chunk_bytes = args.copy_chunk_mib * 1024 * 1024
    counts = {"train": args.train, "val": args.val, "test": args.test}
    print("=" * 78)
    print(f"Actual-like local NS corpus: {root}")
    print("=" * 78)
    print(f"trajectories={sum(counts.values())}; splits={counts}; grid={args.grid}^2; "
          f"frames={args.frames}; sparse dt={spec.dt_out:g}; solver dt={spec.dt:g}")
    print("Each shard is resumable. Future states are offline targets only.")
    print(f"staging={scratch}; external copy chunks={args.copy_chunk_mib} MiB")

    split_shards: dict[str, list[Path]] = {}
    for split_id, (split, count) in enumerate(counts.items()):
        directory = root / split
        directory.mkdir(exist_ok=True)
        paths = []
        for begin in range(0, count, args.shard_size):
            n = min(args.shard_size, count - begin)
            final = directory / f"{begin:06d}.npy"
            paths.append(final)
            shape = (n, args.frames, args.grid, args.grid)
            if _is_complete(final, shape):
                print(f"  {split} {begin:05d}-{begin + n - 1:05d}: reuse", flush=True)
                continue
            final.unlink(missing_ok=True)
            # A pre-buffered-transfer run used this external memmap suffix.
            # It is necessarily incomplete here because `final` was not valid.
            final.with_suffix(".npy.part").unlink(missing_ok=True)
            external_part = final.with_suffix(".npy.copying")
            external_part.unlink(missing_ok=True)
            part = scratch / split / f"{begin:06d}.npy.part"
            part.parent.mkdir(parents=True, exist_ok=True)
            part.unlink(missing_ok=True)
            print(f"  {split} {begin:05d}-{begin + n - 1:05d}: generate", flush=True)
            values = simulate(spec, n_traj=n, n_steps=args.frames - 1,
                              seed=args.seed + split_id * 1_000_000 + begin,
                              device="cpu", batch=args.batch, dtype=torch.float64,
                              out_path=part)
            del values
            print(f"  {split} {begin:05d}-{begin + n - 1:05d}: buffered copy", flush=True)
            _buffered_copy(part, external_part, copy_chunk_bytes)
            if not _is_complete(external_part, shape):
                raise RuntimeError(f"buffered copy produced an invalid shard: {external_part}")
            external_part.replace(final)
            part.unlink(missing_ok=True)
        split_shards[split] = paths

    mean, sigma, count = _stats(split_shards["train"])
    manifest = {
        "purpose": "local actual-protocol-scale residual development fixture; not a paper benchmark",
        "inference_rule": "future stored frames are unavailable at inference",
        "state": "vorticity",
        "layout": "split/000000.npy shards, each (trajectory,time,y,x), float32",
        "spec": asdict(spec), "solver_dt": spec.dt, "snapshot_dt": spec.dt_out,
        "frames": args.frames, "counts": counts, "shard_size": args.shard_size,
        "seed": args.seed, "mu": mean, "sigma": sigma, "n_train_elems": count,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {root / 'manifest.json'}")
    print(f"train normalization: mean={mean:.4e}, std={sigma:.4e}")


if __name__ == "__main__":
    main()
