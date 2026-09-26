#!/usr/bin/env python3
"""Create a RAM-safe, raw-cadence NS-Gauss development corpus.

The output preserves all 21 original snapshots per trajectory (raw dt=0.05),
but retains only velocity channels (u,v).  It is intended for local residual
fidelity development, not final paper evaluation.  The source file is read one
trajectory at a time and the result is HDF5-chunked, so neither machine needs
to hold the corpus in memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


RAW_DT = 1.0 / 20.0
OFFICIAL_TEST_START = 19_760


def _indices(n_train: int, n_val: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use early official-training trajectories and the official test tail."""
    train = np.arange(n_train, dtype=np.int32)
    val = np.arange(n_train, n_train + n_val, dtype=np.int32)
    test = np.arange(OFFICIAL_TEST_START, OFFICIAL_TEST_START + n_test, dtype=np.int32)
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="assembled official NS-Gauss HDF5/NetCDF file")
    ap.add_argument("--output", required=True, help="new local development HDF5 file")
    ap.add_argument("--n-train", type=int, default=160)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    ap.add_argument("--compression-level", type=int, default=1)
    args = ap.parse_args()

    if min(args.n_train, args.n_val, args.n_test) < 1:
        raise SystemExit("every split must contain at least one trajectory")
    source, output = Path(args.source), Path(args.output)
    if not source.is_file():
        raise SystemExit(f"source does not exist: {source}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    train, val, test = _indices(args.n_train, args.n_val, args.n_test)
    indices = np.concatenate((train, val, test))

    compression = None if args.compression == "none" else args.compression
    kwargs: dict[str, object] = {"chunks": (1, 21, 2, 128, 128), "compression": compression}
    if compression == "gzip":
        kwargs["compression_opts"] = args.compression_level

    temporary = output.with_suffix(output.suffix + ".partial")
    try:
        with h5py.File(source, "r") as inp, h5py.File(temporary, "w") as out:
            key = "velocity" if "velocity" in inp else "solution" if "solution" in inp else None
            if key is None:
                raise SystemExit("source has neither 'velocity' nor 'solution'")
            source_data = inp[key]
            if tuple(source_data.shape[1:]) != (21, 3, 128, 128):
                raise SystemExit(
                    f"expected full NS-Gauss shape (*,21,3,128,128), got {tuple(source_data.shape)}")
            if int(indices.max()) >= source_data.shape[0]:
                raise SystemExit(f"source has only {source_data.shape[0]} trajectories; requested index {indices.max()}")
            out_data = out.create_dataset("velocity", (len(indices), 21, 2, 128, 128), dtype="f4", **kwargs)
            for dst, src in enumerate(indices):
                out_data[dst] = np.asarray(source_data[int(src), :, :2], dtype=np.float32)
                if (dst + 1) % 16 == 0 or dst + 1 == len(indices):
                    print(f"copied {dst + 1}/{len(indices)} trajectories", flush=True)
            splits = out.create_group("splits")
            splits.create_dataset("train", data=np.arange(len(train), dtype=np.int32))
            splits.create_dataset("validation", data=np.arange(len(train), len(train) + len(val), dtype=np.int32))
            splits.create_dataset("test", data=np.arange(len(train) + len(val), len(indices), dtype=np.int32))
            out.create_dataset("source_indices", data=indices)
            out.attrs.update({
                "dataset": "camlab-ethz/NS-Gauss raw-cadence development subset",
                "source": str(source.resolve()),
                "state_variable": key,
                "channels": "velocity_uv",
                "raw_dt": RAW_DT,
                "raw_frames": 21,
                "native_poseidon_raw_stride": 2,
                "purpose": "local residual-fidelity development only; not a paper split",
            })
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "path": str(output.resolve()), "shape": [len(indices), 21, 2, 128, 128],
        "raw_dt": RAW_DT, "source_indices": {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()},
        "purpose": "local residual-fidelity development only; use official full split for paper results",
    }, indent=2) + "\n")
    print(f"wrote {output}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
