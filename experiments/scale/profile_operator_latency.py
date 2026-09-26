#!/usr/bin/env python3
"""Measure spatial-operator and admissibility costs across PDE families."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.scale.cross_fm_benchmark import GenericPDEPhysics


def timed(fn, device, warmup, trials):
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    for _ in range(warmup):
        fn()
    sync()
    vals = []
    for _ in range(trials):
        sync(); start = time.perf_counter(); fn(); sync()
        vals.append((time.perf_counter() - start) * 1000.0)
    vals.sort()
    return {"mean_ms": sum(vals) / len(vals), "p50_ms": vals[len(vals)//2],
            "p95_ms": vals[max(0, int(0.95 * len(vals)) - 1)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()
    device = torch.device(args.device)
    rows = []
    for pde, channels in (("NS-Gauss", 2), ("FNS-KF", 2), ("ACE", 1), ("Wave-Gauss", 1)):
        op = GenericPDEPhysics(pde, n=args.grid, device=device, dtype=torch.float32)
        x = torch.randn(1, channels, args.grid, args.grid, device=device)
        rhs = timed(lambda: op.rhs(x), device, args.warmup, args.trials)
        proj = timed(lambda: op.project_manifold(x), device, args.warmup, args.trials)
        rows.append({"pde": pde, "channels": channels, "grid": args.grid,
                     "rhs": rhs, "projection": proj,
                     "two_rhs_plus_projection_ms": 2 * rhs["mean_ms"] + proj["mean_ms"]})
        print(f"{pde:12s} RHS={rhs['mean_ms']:.3f} ms  projection={proj['mean_ms']:.3f} ms")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": str(device), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
