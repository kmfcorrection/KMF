#!/usr/bin/env python3
"""Generate publication-ready Figure 1 for ICLR paper: Fine-Cadence Super-Resolution Curve.

Plots:
(A) RMSE vs Target Query Time t in [0.01s, 0.10s] comparing:
    - Linear Interpolation (dashed gray)
    - Poseidon FM Raw (red, showing severe phase jitter up to 0.195)
    - Hermite Kinematic Spline (blue)
    - Physical-Neural Consensus (solid green)
(B) Percentage Error Reduction vs Raw Poseidon (all-positive +11.8% to +92.0%).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None, help="Path to benchmark JSON results")
    parser.add_argument("--out", type=str, default="results/scale/fine_cadence_curve.pdf")
    args = parser.parse_args()

    # If json provided, load from it; otherwise use the exact Delta GPU benchmark data
    if args.json and Path(args.json).exists():
        with open(args.json) as f:
            data = json.load(f)
        t_vals = np.array(data["timestamps"])
        lin_rmse = np.array(data["linear_rmse"])
        fm_rmse = np.array(data["poseidon_rmse"])
        herm_rmse = np.array(data["hermite_rmse"])
        cons_rmse = np.array(data["consensus_rmse"])
        gain_fm = np.array(data["gain_vs_poseidon"])
    else:
        t_vals = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        lin_rmse = np.array([0.023016, 0.041413, 0.054559, 0.062262, 0.064535, 0.061446, 0.053127, 0.039788, 0.021837, 0.000000])
        fm_rmse = np.array([0.060662, 0.132928, 0.195308, 0.150141, 0.055596, 0.030391, 0.021718, 0.014027, 0.008301, 0.006515])
        herm_rmse = np.array([0.004833, 0.015151, 0.025523, 0.032542, 0.034610, 0.031419, 0.023766, 0.013586, 0.004182, 0.000001])
        cons_rmse = np.array([0.004833, 0.015151, 0.025523, 0.032542, 0.032196, 0.025859, 0.019154, 0.011588, 0.003917, 0.000001])
        gain_fm = np.array([92.03, 88.60, 86.93, 78.33, 42.09, 14.91, 11.80, 17.39, 52.81, 0.0])

    # Publication style setup
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

    # Plot A: Absolute RMSE Comparison
    ax1.plot(t_vals[:-1], lin_rmse[:-1], "--", color="#7f7f7f", label="Linear Midpoint", marker="x", alpha=0.7)
    ax1.plot(t_vals[:-1], fm_rmse[:-1], "-s", color="#d62728", label="Poseidon FM Raw (Direct Step)", alpha=0.9)
    ax1.plot(t_vals[:-1], herm_rmse[:-1], "-^", color="#1f77b4", label="Hermite Kinematic Spline", alpha=0.9)
    ax1.plot(t_vals[:-1], cons_rmse[:-1], "-o", color="#2ca02c", label="Consensus Fusion (Ours)", linewidth=2.5)

    # Mark Poseidon's native trained step
    ax1.scatter([0.10], [fm_rmse[-1]], color="#d62728", s=80, zorder=5, edgecolors="black", label="Poseidon Trained Step ($t=0.10$s)")

    ax1.set_xlabel("Target Query Timestamp $t$ (seconds)")
    ax1.set_ylabel("Velocity RMSE vs. Ground Truth")
    ax1.set_title("(a) Sub-Cadence Error & Phase Jitter")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.95)
    ax1.set_xlim(0.005, 0.105)

    # Plot B: Percentage Error Reduction vs Raw Poseidon
    interior_t = t_vals[:-1]
    interior_gain = gain_fm[:-1]

    bars = ax2.bar(interior_t, interior_gain, width=0.007, color="#2ca02c", alpha=0.85, edgecolor="#1b611b", label="Gain vs. Raw Poseidon (%)")
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("Target Query Timestamp $t$ (seconds)")
    ax2.set_ylabel("RMSE Reduction vs. Raw Poseidon (%)")
    ax2.set_title("(b) Zero-ODE Error Reduction Spectrum")
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0.005, 0.095)

    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"+{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Add callout box for speedup
    speedup_box = dict(boxstyle="round,pad=0.5", facecolor="#eef8ea", edgecolor="#2ca02c", alpha=0.95)
    ax2.text(0.96, 0.93, "Strictly 0 ODE Solves\n$30.1\\times$ Faster than Poseidon\n($1.73$ms vs $52.11$ms)",
             transform=ax2.transAxes, fontsize=9.5, verticalalignment="top", horizontalalignment="right",
             bbox=speedup_box)

    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    plt.savefig(png_path, bbox_inches="tight")
    print(f"Generated Figure 1 plots:\n  {out_path}\n  {png_path}")


if __name__ == "__main__":
    main()
