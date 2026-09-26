#!/usr/bin/env python3
"""
Summarizes all results from results/scale/cross_benchmark/ across:
- All PDEs (NS-Gauss, FNS-KF, ACE, Wave-Gauss, NS-SL, Wave-Layer)
- All FMs (Poseidon T/B/L, DPOT Ti/S, MORPH Ti/S, Polymathic TFNO/UNetConvNext/FNO)
- All Spatial Resolutions (64x64, 128x128, 256x256)
- Rollout (Step 1 RMSE, Window RMSE, Gain %) & Sub-cadence (Midpoint RMSE, Gain %)
"""
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path


def load_all_results(results_dir: str = "results/scale/cross_benchmark"):
    p = Path(results_dir)
    all_files = sorted(list(p.glob("*_results.json")))
    # Prefer systematic parameter sweep files (containing '_dt')
    sweep_files = [f for f in all_files if "_dt" in f.name]
    files = sweep_files if sweep_files else all_files
    records = []
    for f in files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            fm = data.get("fm", "")
            fm_size = data.get("fm_size", "")
            fm_family = data.get("fm_family", "")
            if fm in ("the_well", "well"):
                model_name = f"Polymathic-{fm_family}"
            elif fm_size:
                model_name = f"{fm.capitalize()}-{fm_size}"
            else:
                model_name = fm.capitalize()

            grid = data.get("grid", 128)
            req_dt = data.get("requested_dt", data.get("coarse_dt", 0.10))
            span_dt = data.get("endpoint_span_dt", data.get("coarse_dt", 0.10))
            pde = data.get("pde", "")
            sub = data.get("sub_cadence", {})
            roll = data.get("rollout", {})

            # Causal pre-selected metrics (strictly zero test leakage):
            sub_ours = sub.get("calibrated_rmse", sub.get("ours_midpoint_rmse", float("nan")))
            sub_gain_lin = sub.get("gain_cal_vs_linear", sub.get("gain_vs_linear", float("nan")))
            sub_gain_fm = sub.get("gain_cal_vs_fm", sub.get("gain_vs_fm", float("nan")))
            roll_ours = roll.get("window_ours_rmse", roll.get("window_simpson_rmse", roll.get("window_proj_rmse", float("nan"))))
            roll_gain = roll.get("gain_rollout", roll.get("gain_rollout_simpson", float("nan")))

            adapter_class = data.get("adapter_class", "UNKNOWN")

            rec = {
                "PDE": pde,
                "Model": model_name,
                "Adapter_Class": adapter_class,
                "Rollout_Cadence_dt": f"{req_dt:.2f}s",
                "Sub_Cadence_dt": f"{span_dt:.2f}s",
                "Cadence_dt": f"{span_dt:.2f}s",
                "dt": span_dt,
                "req_dt": req_dt,
                "Resolution": f"{grid}x{grid}",
                "Grid": grid,
                # Sub-cadence (t = 0.5 * dt)
                "Linear_RMSE": sub.get("linear_rmse", float("nan")),
                "Raw_Sub_RMSE": sub.get("fm_rmse", float("nan")),
                "Hermite_RMSE": sub.get("hermite_rmse", float("nan")),
                "Equal_RMSE": sub.get("equal_consensus_rmse", float("nan")),
                "Physics_RMSE": sub.get("physics_gated_rmse", float("nan")),
                "Calibrated_RMSE": sub.get("calibrated_rmse", float("nan")),
                "Ours_Sub_RMSE": sub_ours,
                "Sub_Gain_vs_Lin": sub_gain_lin,
                "Sub_Gain_vs_FM": sub_gain_fm,
                # Autoregressive Rollout
                "Roll_Steps": roll.get("steps", 4),
                "Step1_Raw_RMSE": roll.get("step1_raw_rmse", float("nan")),
                "Step1_Ours_RMSE": roll.get("step1_proj_rmse", float("nan")),
                "Rollout_Raw_RMSE": roll.get("window_raw_rmse", float("nan")),
                "Rollout_Proj_RMSE": roll.get("window_proj_rmse", float("nan")),
                "Rollout_Simpson_RMSE": roll.get("window_simpson_rmse", float("nan")),
                "Rollout_Ours_RMSE": roll_ours,
                "Rollout_Gain": roll_gain,
                "Rollout_Gain_Proj": roll.get("gain_rollout_proj", float("nan")),
                "Rollout_Gain_Simpson": roll.get("gain_rollout_simpson", float("nan")),
            }
            records.append(rec)
        except Exception as e:
            print(f"[WARN] Error reading {f}: {e}")

    if not records:
        print(f"No results found in {results_dir}")
        return None

    records.sort(key=lambda x: (x["PDE"], x["Model"], x["req_dt"], x["Grid"]))
    return records


def format_val(val, is_pct=False, decimals=6):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if is_pct:
        return f"{val:+.2f}%" if isinstance(val, (int, float)) else str(val)
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def print_table(headers, rows, float_cols=None, pct_cols=None):
    if float_cols is None:
        float_cols = set()
    if pct_cols is None:
        pct_cols = set()

    formatted_rows = []
    for row in rows:
        fmt_row = []
        for i, val in enumerate(row):
            h = headers[i]
            if h in pct_cols:
                fmt_row.append(format_val(val, is_pct=True))
            elif h in float_cols:
                fmt_row.append(format_val(val, decimals=6))
            else:
                fmt_row.append(str(val) if val is not None else "N/A")
        formatted_rows.append(fmt_row)

    col_widths = [len(h) for h in headers]
    for row in formatted_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_str = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_str = "  ".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_str)
    print(sep_str)
    for row in formatted_rows:
        print("  ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))


def print_summary_tables(records):
    print("\n" + "=" * 140)
    print("TABLE 1: MULTI-TEMPORAL CADENCE & MULTI-RESOLUTION AUTOREGRESSIVE ROLLOUT BENCHMARK (0 FORWARD ODE STEPS)")
    print("=" * 140)
    h1 = [
        "PDE", "Model", "Rollout_Cadence_dt", "Resolution", "Roll_Steps",
        "Step1_Raw_RMSE", "Step1_Ours_RMSE", "Rollout_Raw_RMSE",
        "Rollout_Proj_RMSE", "Rollout_Simpson_RMSE", "Rollout_Gain"
    ]
    rows1 = [[r[k] for k in h1] for r in records]
    float_cols1 = {
        "Step1_Raw_RMSE", "Step1_Ours_RMSE", "Rollout_Raw_RMSE",
        "Rollout_Proj_RMSE", "Rollout_Simpson_RMSE"
    }
    pct_cols1 = {"Rollout_Gain"}
    print_table(h1, rows1, float_cols=float_cols1, pct_cols=pct_cols1)

    print("\n" + "=" * 150)
    print("TABLE 2: FM-ENDPOINT TEMPORAL REFINEMENT: ALL VARIANTS vs BASELINES (0 FORWARD ODE STEPS)")
    print("=" * 150)
    h2 = [
        "PDE", "Model", "Sub_Cadence_dt", "Resolution", "Linear_RMSE",
        "Raw_Sub_RMSE", "Hermite_RMSE", "Equal_RMSE", "Physics_RMSE",
        "Calibrated_RMSE", "Sub_Gain_vs_Lin", "Sub_Gain_vs_FM"
    ]
    # A midpoint requires an endpoint span of at least two stored 0.05 s frames.
    sub_records = [r for r in records if r.get("req_dt", r.get("dt", 0.10)) >= 0.099]
    rows2 = [[r[k] for k in h2] for r in sub_records]
    float_cols2 = {
        "Linear_RMSE", "Raw_Sub_RMSE", "Hermite_RMSE", "Equal_RMSE",
        "Physics_RMSE", "Calibrated_RMSE"
    }
    pct_cols2 = {"Sub_Gain_vs_Lin", "Sub_Gain_vs_FM"}
    print_table(h2, rows2, float_cols=float_cols2, pct_cols=pct_cols2)


def save_to_csv(records, out_path: str = "results/scale/master_benchmark_summary.csv"):
    if not records:
        return
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(out_p, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSummary exported to {out_p} ({len(records)} benchmark configurations)")


def main():
    res_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/scale/cross_benchmark"
    records = load_all_results(res_dir)
    if records:
        print_summary_tables(records)
        csv_out = f"{res_dir}/master_benchmark_summary.csv" if res_dir != "results/scale/cross_benchmark" else "results/scale/master_benchmark_summary.csv"
        save_to_csv(records, out_path=csv_out)


if __name__ == "__main__":
    main()
