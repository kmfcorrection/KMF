"""Shared helpers: device handling, seeding, IO."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # float64 and several linalg ops are unsupported on MPS; the curvature
        # code needs both, so we stay on CPU rather than fail deep in a stage.
        return torch.device("cpu")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)
    return path


def load_json(path: str | Path):
    with open(path) as fh:
        return json.load(fh)


def results_path(stage: str, pde: str, name: str) -> Path:
    p = RESULTS_DIR / stage / pde
    p.mkdir(parents=True, exist_ok=True)
    return p / name


class Table:
    """Minimal aligned-text table for stage summaries."""

    def __init__(self, *cols: str):
        self.cols = list(cols)
        self.rows: list[list[str]] = []

    def add(self, *vals) -> None:
        out = []
        for v in vals:
            if isinstance(v, float):
                out.append(f"{v:.4g}")
            else:
                out.append(str(v))
        self.rows.append(out)

    def __str__(self) -> str:
        widths = [len(c) for c in self.cols]
        for r in self.rows:
            for i, v in enumerate(r):
                widths[i] = max(widths[i], len(v))
        line = "  ".join(c.ljust(w) for c, w in zip(self.cols, widths))
        sep = "  ".join("-" * w for w in widths)
        body = "\n".join("  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in self.rows)
        return f"{line}\n{sep}\n{body}"


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
