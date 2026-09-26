#!/usr/bin/env python3
"""Generate appendix tables from the evaluated master-matrix JSON files."""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "iclr2027" / "generated_master_tables.tex"


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def fmt(x, digits=5):
    return "N/A" if not finite(x) else f"{x:.{digits}f}"


def pct(x):
    return "N/A" if not finite(x) else f"{x:+.2f}\\%"


def model_name(d):
    if d.get("fm") in ("the_well", "well"):
        return f"Polymathic-{d.get('fm_family', '')}"
    if d.get("fm") == "cno":
        return "CNO-FM"
    return f"{str(d.get('fm', '')).capitalize()}-{d.get('fm_size', '')}"


def load():
    # Later directories overwrite an identical basename, if one is present.
    by_name = {}
    for directory in sorted(glob.glob(str(ROOT / "results/scale/master_matrix_20260924*"))):
        for path in glob.glob(os.path.join(directory, "json", "*_results.json")):
            by_name[os.path.basename(path)] = path
    rows = []
    for path in sorted(by_name.values()):
        with open(path) as f:
            d = json.load(f)
        d["_path"] = path
        d["_model"] = model_name(d)
        rows.append(d)
    return rows


def generate(rows):
    lines = [
        "% Automatically generated from the evaluated master-matrix JSON files.",
        "\\clearpage",
        "\\begingroup\\tiny",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\renewcommand{\\arraystretch}{1.02}",
        "\\begin{longtable}{llccrrrrrr}",
        "\\caption{FM-endpoint temporal-refinement matrix. Every row is an evaluated configuration; unavailable dataset or contract pairs are not imputed.\\label{tab:subcadence_master}}\\\\",
        "\\toprule",
        "PDE & Model & req. $\\Delta t$ & span $\\Delta t$ & Linear & Raw FM & Hermite & Calibrated & Gain/Lin. & Gain/FM \\\\",
        "\\midrule\\endfirsthead",
        "\\multicolumn{10}{c}{\\small\\textit{Table \\ref{tab:subcadence_master} continued}} \\\\",
        "\\toprule",
        "PDE & Model & req. $\\Delta t$ & span $\\Delta t$ & Linear & Raw FM & Hermite & Calibrated & Gain/Lin. & Gain/FM \\\\",
        "\\midrule\\endhead",
        "\\midrule\\multicolumn{10}{r}{\\small\\textit{Continued on next page}} \\\\",
        "\\endfoot\\bottomrule\\endlastfoot",
    ]
    for d in sorted(rows, key=lambda x: (x.get("pde", ""), x["_model"], x.get("requested_dt", 0))):
        s = d.get("sub_cadence", {})
        lines.append(
            f"{d.get('pde','')} & {d['_model']} & {d.get('requested_dt',0):.2f} & "
            f"{d.get('endpoint_span_dt', d.get('requested_dt',0)):.2f} & "
            f"{fmt(s.get('linear_rmse'))} & {fmt(s.get('fm_rmse'))} & "
            f"{fmt(s.get('hermite_rmse'))} & {fmt(s.get('calibrated_rmse'))} & "
            f"{pct(s.get('gain_cal_vs_linear'))} & {pct(s.get('gain_cal_vs_fm'))} \\\\")
    lines += ["\\end{longtable}", "\\clearpage"]

    lines += [
        "\\begin{longtable}{llccrrrrrrr}",
        "\\caption{Causal rollout and dense-bridge matrix. The rollout columns use the calibration-selected correction and the dense columns are reported only where an intermediate truth state was available for scoring.\\label{tab:rollout_master}}\\\\",
        "\\toprule",
        "PDE & Model & $\\Delta t$ & Steps & Raw & Projected & KMF & Gain & $\\alpha^*$ & Dense raw-lin. & Dense KMF \\\\",
        "\\midrule\\endfirsthead",
        "\\multicolumn{11}{c}{\\small\\textit{Table \\ref{tab:rollout_master} continued}} \\\\",
        "\\toprule",
        "PDE & Model & $\\Delta t$ & Steps & Raw & Projected & KMF & Gain & $\\alpha^*$ & Dense raw-lin. & Dense KMF \\\\",
        "\\midrule\\endhead",
        "\\midrule\\multicolumn{11}{r}{\\small\\textit{Continued on next page}} \\\\",
        "\\endfoot\\bottomrule\\endlastfoot",
    ]
    for d in sorted(rows, key=lambda x: (x.get("pde", ""), x["_model"], x.get("requested_dt", 0))):
        r = d.get("rollout", {})
        lines.append(
            f"{d.get('pde','')} & {d['_model']} & {d.get('requested_dt',0):.2f} & {r.get('steps','N/A')} & "
            f"{fmt(r.get('window_raw_rmse'))} & {fmt(r.get('window_proj_rmse'))} & "
            f"{fmt(r.get('window_ours_rmse'))} & {pct(r.get('gain_rollout'))} & "
            f"{fmt(r.get('alpha_hs_calibrated'), 3)} & "
            f"{fmt(r.get('dense_window_rmse_raw_linear'))} & {fmt(r.get('dense_window_rmse_kmf'))} \\\\")
    lines += ["\\end{longtable}", "\\endgroup", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} from {len(rows)} JSON files")


if __name__ == "__main__":
    generate(load())
