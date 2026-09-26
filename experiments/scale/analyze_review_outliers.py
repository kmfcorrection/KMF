#!/usr/bin/env python3
"""Audit negative-gain and unavailable-contract rows in the master matrix.

This is a descriptive audit only. It never selects a method or changes a
reported result. It records the raw endpoint availability, calibration weight,
and the two endpoint-refinement gains so failure mechanisms are visible.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def model_name(row):
    if row.get("fm") in ("the_well", "well"):
        return f"Polymathic-{row.get('fm_family', '')}"
    return f"{str(row.get('fm', '')).capitalize()}-{row.get('fm_size', '')}"


def collect(root):
    paths = {}
    for pattern in (
        str(root / "master_matrix_*/json/*_results.json"),
        str(root / "cross_benchmark/*_results.json"),
    ):
        for path in glob.glob(pattern):
            paths[os.path.basename(path)] = path
    rows = []
    for path in sorted(paths.values()):
        with open(path) as handle:
            row = json.load(handle)
        sub = row.get("sub_cadence", {})
        rows.append({
            "file": os.path.basename(path),
            "pde": row.get("pde", ""),
            "model": model_name(row),
            "requested_dt": row.get("requested_dt"),
            "endpoint_span_dt": row.get("endpoint_span_dt"),
            "linear_rmse": sub.get("linear_rmse"),
            "fm_rmse": sub.get("fm_rmse"),
            "hermite_rmse": sub.get("hermite_rmse"),
            "calibrated_rmse": sub.get("calibrated_rmse"),
            "gain_vs_linear": sub.get("gain_cal_vs_linear"),
            "gain_vs_fm": sub.get("gain_cal_vs_fm"),
            "calibrated_weight": sub.get("calibrated_weight"),
            "raw_fm_available": finite(sub.get("fm_rmse")),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/scale"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.results_root)
    rows.sort(key=lambda r: (
        r["gain_vs_linear"] if finite(r["gain_vs_linear"]) else float("inf"),
        r["pde"], r["model"], r["requested_dt"] or 0,
    ))
    with open(args.out_dir / "master_outlier_audit.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    negative = [r for r in rows if finite(r["gain_vs_linear"]) and r["gain_vs_linear"] < 0]
    unavailable = [r for r in rows if not r["raw_fm_available"]]
    summary = {
        "n_rows": len(rows),
        "n_negative_vs_linear": len(negative),
        "n_unavailable_raw_fm": len(unavailable),
        "negative_rows": negative,
        "unavailable_rows": unavailable,
        "interpretation": {
            "negative_gain": "A negative held-out gain indicates that the calibrated physical candidate did not transfer to that configuration; it is not hidden or removed.",
            "unavailable_raw_fm": "When the adapter exposes no off-cadence query, calibrated fusion is not comparable to a direct FM query and must be interpreted as the physical bridge.",
        },
    }
    with open(args.out_dir / "master_outlier_audit.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: summary[k] for k in ("n_rows", "n_negative_vs_linear", "n_unavailable_raw_fm")}, indent=2))


if __name__ == "__main__":
    main()
