#!/usr/bin/env python3
"""Create split-safe canonical and FM-specific views of Poseidon PDEGym data.

This is an *offline data preparation* step.  It extracts a fixed pool only
from the publisher-declared test partition, then partitions that pool into
calibration/validation/test trajectories.  It never computes a trajectory-wide
normalizer, so future frames cannot enter a causal forecast input.

Supported frozen-model views are intentionally conservative:

* Poseidon and MORPH receive a runnable two-velocity-channel FNS-KF view.
* DPOT receives true chronological 10-frame histories; for FNS-KF, the known
  incompressible embedding [rho=1,u,v,p=0] is materialized as ``history4``.
* ACE and Wave-Gauss are stored canonically, with static wave speed retained,
  but are marked ``requires_adapter``.  The current frozen-model adapters do
  not implement their released scalar downstream embedding/recovery contracts.

The script records this status explicitly rather than padding scalar states
into arbitrary channels and calling them valid FM evaluations.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


SPECS = {
    "FNS-KF": dict(test_start=19760, test_count=240, variable="solution",
                   channels=2, kind="forced_incompressible_ns"),
    "ACE": dict(test_start=14760, test_count=240, variable="solution",
                channels=1, kind="allen_cahn"),
    "Wave-Gauss": dict(test_start=10272, test_count=240, variable="solution",
                       channels=1, kind="variable_speed_wave", static="c"),
}


def _copy(dst: h5py.Dataset, src: h5py.Dataset, indices: np.ndarray, channels: int):
    """Copy selected trajectories without materializing the whole source file."""
    for out_i, src_i in enumerate(indices):
        a = np.asarray(src[int(src_i)], dtype=np.float32)
        if a.ndim == 3:  # (time, H, W)
            a = a[:, None]
        if a.ndim != 4 or a.shape[1] != channels or a.shape[-2:] != (128, 128):
            raise ValueError(f"unexpected trajectory shape {a.shape}")
        dst[out_i] = a


def _write_str_attr(group, key: str, value):
    group.attrs[key] = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value


def _create_dpot_views(out: h5py.File, state: h5py.Dataset, dataset: str):
    """Create genuine past-to-future windows, never repeated-current history."""
    g = out.create_group("dpot")
    n, t, c, h, w = state.shape
    history = 10
    starts = [(i, stop) for i in range(n) for stop in range(history, t)]
    m = len(starts)
    g.create_dataset("history", (m, history, c, h, w), dtype="f4",
                     chunks=(1, history, c, h, w), compression="gzip", compression_opts=1)
    g.create_dataset("target", (m, c, h, w), dtype="f4",
                     chunks=(1, c, h, w), compression="gzip", compression_opts=1)
    index = g.create_dataset("index", (m, 2), dtype="i4")
    for j, (i, stop) in enumerate(starts):
        g["history"][j] = state[i, stop - history:stop]
        g["target"][j] = state[i, stop]
        index[j] = (i, stop)
    _write_str_attr(g, "layout", "sample,time,channel,height,width")
    _write_str_attr(g, "history_policy", "true chronological ten-frame history")
    if dataset == "FNS-KF":
        h4 = g.create_dataset("history4", (m, history, 4, h, w), dtype="f4",
                              chunks=(1, history, 4, h, w), compression="gzip", compression_opts=1)
        y4 = g.create_dataset("target4", (m, 4, h, w), dtype="f4",
                              chunks=(1, 4, h, w), compression="gzip", compression_opts=1)
        for j in range(m):
            h4[j, :, 0] = 1.0
            h4[j, :, 1:3] = g["history"][j]
            y4[j, 0] = 1.0
            y4[j, 1:3] = g["target"][j]
        _write_str_attr(g, "status", "candidate-ready: known FNS channel embedding; dataset-specific DPOT normalization still requires audit")
        _write_str_attr(g, "channel_map", ["rho=1", "u", "v", "p=0"])
    else:
        _write_str_attr(g, "status", "requires_adapter: scalar field must not be padded into DPOT's four channels without a published mapping and normalization")


def _create_poseidon_morph_views(out: h5py.File, state: h5py.Dataset, dataset: str):
    """Create only views justified by an existing frozen-model state contract."""
    n, t, c, h, w = state.shape
    pg = out.create_group("poseidon")
    mg = out.create_group("morph")
    if dataset == "FNS-KF":
        p4 = pg.create_dataset("state4", (n, t, 4, h, w), dtype="f4",
                               chunks=(1, 1, 4, h, w), compression="gzip", compression_opts=1)
        p4[:, :, 0] = 1.0
        for i in range(n):
            p4[i, :, 1:3] = state[i]
        _write_str_attr(pg, "status", "candidate-ready: known incompressible Poseidon channel embedding")
        _write_str_attr(pg, "channel_map", ["rho=1", "u", "v", "p=0"])
        morph = mg.create_dataset("state", state.shape, dtype="f4",
                                  chunks=(1, 1, 2, h, w), compression="gzip", compression_opts=1)
        for i in range(n):
            morph[i] = state[i]
        _write_str_attr(mg, "status", "candidate-ready geometry view: causal one-frame RevIN only; S4 requires a separate MORPH protocol audit")
    else:
        _write_str_attr(pg, "status", "requires_adapter: released downstream embedding/recovery contract for this scalar PDE is not implemented")
        _write_str_attr(mg, "status", "requires_adapter: current MORPH adapter is two-component velocity only")


def prepare_one(args, name: str):
    spec = SPECS[name]
    source = Path(args.data_root) / "assembled" / f"{name}.nc"
    if not source.exists():
        raise FileNotFoundError(f"missing assembled dataset: {source}")
    n_total = args.n_cal + args.n_val + args.n_test
    if args.test_offset < 0 or args.test_offset + n_total > spec["test_count"]:
        raise ValueError(f"{name}: requested {n_total} trajectories at offset {args.test_offset}, "
                         f"but official test partition has {spec['test_count']}")
    destination = Path(args.out_root) / f"{name}_testpool_{n_total}.h5"
    if destination.exists() and not args.overwrite:
        print(f"exists, skipping: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.h5")
    temporary.unlink(missing_ok=True)
    source_indices = np.arange(spec["test_start"] + args.test_offset,
                               spec["test_start"] + args.test_offset + n_total,
                               dtype=np.int64)
    with h5py.File(source, "r") as inp, h5py.File(temporary, "w") as out:
        var = spec["variable"]
        if var not in inp:
            raise KeyError(f"{source}: expected variable {var!r}")
        ds = inp[var]
        if ds.shape[0] < spec["test_start"] + spec["test_count"]:
            raise ValueError(f"{source}: source has {ds.shape[0]} trajectories, "
                             "shorter than the official split metadata")
        state = out.create_dataset("canonical/state", (n_total, ds.shape[1], spec["channels"], 128, 128),
                                   dtype="f4", chunks=(1, 1, spec["channels"], 128, 128),
                                   compression="gzip", compression_opts=1)
        _copy(state, ds, source_indices, spec["channels"])
        if "static" in spec:
            key = spec["static"]
            if key not in inp:
                raise KeyError(f"{source}: expected static variable {key!r}")
            static = out.create_dataset(f"canonical/static/{key}", (n_total, 1, 128, 128),
                                        dtype="f4", chunks=(1, 1, 128, 128), compression="gzip", compression_opts=1)
            for j, src_i in enumerate(source_indices):
                static[j, 0] = np.asarray(inp[key][int(src_i)], dtype=np.float32)
        splits = out.create_group("splits")
        splits.create_dataset("calibration", data=np.arange(0, args.n_cal, dtype=np.int32))
        splits.create_dataset("validation", data=np.arange(args.n_cal, args.n_cal + args.n_val, dtype=np.int32))
        splits.create_dataset("test", data=np.arange(args.n_cal + args.n_val, n_total, dtype=np.int32))
        _create_dpot_views(out, state, name)
        _create_poseidon_morph_views(out, state, name)
        _write_str_attr(out, "dataset", name)
        _write_str_attr(out, "pde_kind", spec["kind"])
        _write_str_attr(out, "source", str(source.resolve()))
        _write_str_attr(out, "source_trajectory_indices", source_indices.tolist())
        _write_str_attr(out, "selection", "official test partition only; chronological state history; no future normalization")
        _write_str_attr(out, "split_sizes", {"calibration": args.n_cal, "validation": args.n_val, "test": args.n_test})
    shutil.move(temporary, destination)
    print(f"wrote {destination}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="root containing assembled/<dataset>.nc")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--datasets", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--test-offset", type=int, default=0,
                    help="offset inside the publisher-declared test partition")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if min(args.n_cal, args.n_val, args.n_test) <= 0:
        raise SystemExit("all split sizes must be positive")
    for name in args.datasets:
        prepare_one(args, name)


if __name__ == "__main__":
    main()
