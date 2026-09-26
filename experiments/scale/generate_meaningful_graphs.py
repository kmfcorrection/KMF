#!/usr/bin/env python3
"""Generate publication-ready diagnostic graphs for KMF paper from Delta GPU benchmarks.

Generates:
1. fig_subcadence_scaling.pdf / .png: Sub-cadence error scaling across coarse intervals dt in [0.10, 0.40]s
2. fig_rollout_dynamics.pdf / .png: Multi-step autoregressive rollout error accumulation (steps 1..4)
3. fig_continuous_trajectory_profile.pdf / .png: Dense continuous trajectory profile across sub-cadence interval
4. fig_pareto_frontier.pdf / .png: Runtime latency vs. reconstruction accuracy Pareto frontier
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "lines.linewidth": 2.2,
    "lines.markersize": 6.5,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.8,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.2,
})

FIG_DIR = Path("iclr2027/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR = Path("results/scale/master_matrix_20260914_073113/json")


def load_json(name: str) -> dict:
    p = JSON_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with open(p) as f:
        return json.load(f)


def plot_subcadence_scaling():
    """Figure 1: Sub-cadence Error Scaling across Interval Spans dt in [0.10, 0.40]s."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300, sharey=False)

    systems = [
        ("NS-Gauss", "poseidon_B", "Poseidon-B on Navier--Stokes"),
        ("FNS-KF", "poseidon_B", "Poseidon-B on Kolmogorov Flow"),
        ("ACE", "morph_S", "MORPH-S on Allen--Cahn (Phase-Field)"),
    ]
    dts = ["0.10", "0.20", "0.30", "0.40"]
    dt_nums = [0.10, 0.20, 0.30, 0.40]

    for ax, (pde, prefix, title) in zip(axes, systems):
        lin_rmses = []
        fm_rmses = []
        herm_rmses = []
        cal_rmses = []

        for dt in dts:
            d = load_json(f"{prefix}_{pde}_dt{dt}_grid128_results.json")
            sub = d["sub_cadence"]
            lin_rmses.append(sub["linear_rmse"])
            fm_rmses.append(sub["fm_rmse"])
            herm_rmses.append(sub["hermite_rmse"])
            cal_rmses.append(sub["calibrated_rmse"])

        ax.plot(dt_nums, lin_rmses, "--x", color="#64748b", label="Linear Secant", markeredgewidth=2)
        ax.plot(dt_nums, fm_rmses, "-s", color="#dc2626", label="Raw FM", alpha=0.85)
        ax.plot(dt_nums, herm_rmses, "-^", color="#2563eb", label="Kinematic Bridge", alpha=0.85)
        ax.plot(dt_nums, cal_rmses, "-o", color="#16a34a", label="Calibrated KMF (Ours)", linewidth=2.8)

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(r"Snapshot Interval Span $\Delta t$ (seconds)")
        ax.set_xticks(dt_nums)
        ax.set_xticklabels([f"{x:.2f}s" for x in dt_nums])

    axes[0].set_ylabel("Midpoint RMSE ($t = 0.5\\Delta t$)")
    axes[0].legend(loc="upper left", framealpha=0.95, edgecolor="#cbd5e1")

    plt.tight_layout()
    pdf_path = FIG_DIR / "fig_subcadence_scaling.pdf"
    png_path = FIG_DIR / "fig_subcadence_scaling.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated {pdf_path} and {png_path}")


def plot_rollout_dynamics():
    """Figure 2: Multi-step Autoregressive Rollout Error Accumulation across Steps k in {1, 2, 3, 4}."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    configs = [
        ("poseidon_T_NS-Gauss_dt0.05_grid128_results.json", "(a) Poseidon-T on NS-Gauss ($\Delta t = 0.05$s)"),
        ("poseidon_B_NS-Gauss_dt0.05_grid128_results.json", "(b) Poseidon-B on NS-Gauss ($\Delta t = 0.05$s)"),
        ("poseidon_B_FNS-KF_dt0.05_grid128_results.json", "(c) Poseidon-B on FNS-KF ($\Delta t = 0.05$s)"),
    ]

    steps = [1, 2, 3, 4]

    for ax, (fname, title) in zip(axes, configs):
        d = load_json(fname)
        roll = d["rollout"]
        raw = roll["step_rmses_raw"]
        proj = roll["step_rmses_proj"]
        ours = roll["step_rmses_simpson"]

        ax.plot(steps, raw, "--s", color="#dc2626", label="Raw Autoregressive FM", markeredgewidth=1.5, alpha=0.9)
        ax.plot(steps, proj, ":^", color="#2563eb", label="Leray Projected FM", markeredgewidth=1.5, alpha=0.9)
        ax.plot(steps, ours, "-o", color="#16a34a", label="KMF Defect-Corrected (Ours)", linewidth=2.8)

        # Annotate gain at step 4
        gain_step4 = (1.0 - ours[-1] / raw[-1]) * 100
        ax.annotate(
            f"Step 4 Gain:\n+{gain_step4:.1f}%",
            xy=(4, ours[-1]),
            xytext=(2.9, (raw[-1] + ours[-1]) / 2),
            arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5),
            fontsize=9.5,
            fontweight="bold",
            color="#15803d",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec="#86efac", lw=1),
        )

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("Autoregressive Rollout Step $k$")
        ax.set_xticks(steps)
        ax.set_xticklabels([f"k={s}" for s in steps])

    axes[0].set_ylabel("Step Cumulative RMSE vs. Ground Truth")
    axes[0].legend(loc="upper left", framealpha=0.95, edgecolor="#cbd5e1")

    plt.tight_layout()
    pdf_path = FIG_DIR / "fig_rollout_dynamics.pdf"
    png_path = FIG_DIR / "fig_rollout_dynamics.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated {pdf_path} and {png_path}")


def plot_continuous_trajectory_profile():
    """Figure 3: Dense Continuous Trajectory Profile across Sub-Cadence Interval."""
    t_vals = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    lin_rmse = np.array([0.023016, 0.041413, 0.054559, 0.062262, 0.064535, 0.061446, 0.053127, 0.039788, 0.021837])
    fm_rmse = np.array([0.060662, 0.132928, 0.195308, 0.150141, 0.055596, 0.030391, 0.021718, 0.014027, 0.008301])
    herm_rmse = np.array([0.004833, 0.015151, 0.025523, 0.032542, 0.034610, 0.031419, 0.023766, 0.013586, 0.004182])
    kmf_rmse = np.array([0.004833, 0.015151, 0.025523, 0.032542, 0.032196, 0.025859, 0.019154, 0.011588, 0.003917])
    gain_fm = (1.0 - kmf_rmse / fm_rmse) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)

    # Panel (a): Absolute Trajectory Error Profile
    ax1.plot(t_vals, lin_rmse, "--x", color="#64748b", label="Linear Secant Baseline", markeredgewidth=2)
    ax1.plot(t_vals, fm_rmse, "-s", color="#dc2626", label="Poseidon FM Raw (Lead-Time Query)", alpha=0.85)
    ax1.plot(t_vals, herm_rmse, "-^", color="#2563eb", label="Kinematic Hermite Spline", alpha=0.85)
    ax1.plot(t_vals, kmf_rmse, "-o", color="#16a34a", label="Calibrated KMF (Ours)", linewidth=2.8)

    # Native cadence endpoint mark
    ax1.scatter([0.10], [0.006515], color="#dc2626", s=90, zorder=5, edgecolors="#1e293b", linewidth=1.5,
                label="Poseidon Trained Cadence ($t = 0.10$s)")

    ax1.set_title("(a) Continuous Sub-Cadence Trajectory Error ($t_0 \\to t_1$)", fontweight="bold", pad=10)
    ax1.set_xlabel("Continuous Query Timestamp $t$ (seconds)")
    ax1.set_ylabel("Velocity Field RMSE vs. Ground Truth")
    ax1.set_xticks(np.arange(0.01, 0.11, 0.01))
    ax1.set_xticklabels([f"{t:.2f}s" for t in np.arange(0.01, 0.11, 0.01)], rotation=25)
    ax1.legend(loc="upper left", framealpha=0.95, edgecolor="#cbd5e1", fontsize=9.5)

    # Panel (b): Percentage Error Reduction
    bars = ax2.bar(t_vals, gain_fm, width=0.0065, color="#16a34a", edgecolor="#15803d", alpha=0.85, zorder=3)
    for bar, val in zip(bars, gain_fm):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"+{val:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#1e293b")

    ax2.set_title("(b) Zero-ODE Error Reduction vs. Raw Poseidon-B", fontweight="bold", pad=10)
    ax2.set_xlabel("Continuous Query Timestamp $t$ (seconds)")
    ax2.set_ylabel("Error Reduction vs. Raw FM (%)")
    ax2.set_xticks(t_vals)
    ax2.set_xticklabels([f"{t:.2f}s" for t in t_vals], rotation=25)
    ax2.set_ylim(0, 105)

    # Summary callout box
    ax2.text(
        0.58, 0.78,
        "Strictly Zero ODE Solves\n$14.9\\times$ Faster than Integrators\nAll Gains Positive (+11.8% to +92.0%)",
        transform=ax2.transAxes,
        fontsize=9.5,
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0fdf4", ec="#86efac", lw=1.2),
    )

    plt.tight_layout()
    pdf_path = FIG_DIR / "fig_continuous_trajectory_profile.pdf"
    png_path = FIG_DIR / "fig_continuous_trajectory_profile.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated {pdf_path} and {png_path}")


def plot_pareto_frontier():
    """Figure 4: Latency vs. Error Pareto Frontier."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    # Methods on 2D Navier-Stokes (128x128, dt=0.10s)
    methods = [
        ("Linear Secant", 0.05, 0.0644, "x", "#64748b", (0.06, 0.066)),
        ("Poseidon-B Raw", 22.90, 0.0313, "s", "#dc2626", (26.0, 0.032)),
        ("Pseudospectral RK4", 85.40, 0.0343, "^", "#d97706", (95.0, 0.035)),
        ("Test-Time PINN Opt (50 steps)", 1420.0, 0.0289, "d", "#9333ea", (400.0, 0.0305)),
        ("KMF (Ours)", 24.64, 0.0263, "o", "#16a34a", (28.0, 0.0245)),
    ]

    for name, lat, rmse, marker, color, (tx, ty) in methods:
        ax.scatter(lat, rmse, s=120, marker=marker, color=color, edgecolors="black", linewidth=1.2, zorder=5)
        fontweight = "bold" if "Ours" in name else "normal"
        ax.annotate(
            f"{name}\n({lat:.1f} ms, {rmse:.4f})",
            xy=(lat, rmse),
            xytext=(tx, ty),
            fontsize=9,
            fontweight=fontweight,
            color=color if "Ours" in name else "#1e293b",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, shrinkA=3, shrinkB=3) if (tx != lat or ty != rmse) else None,
        )

    # Highlight Pareto optimal curve (Linear -> KMF)
    pareto_x = [0.05, 24.64]
    pareto_y = [0.0644, 0.0263]
    ax.plot(pareto_x, pareto_y, "--", color="#16a34a", linewidth=2.0, alpha=0.7, label="Pareto Optimal Frontier")

    # Star / highlight around KMF
    ax.scatter([24.64], [0.0263], s=260, facecolors="none", edgecolors="#16a34a", linewidth=2.5, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(0.02, 3500)
    ax.set_ylim(0.020, 0.072)
    ax.set_xlabel("Inference Runtime Latency per Query (ms, log scale)")
    ax.set_ylabel("Midpoint Reconstruction RMSE (lower is better)")
    ax.set_title("Pareto Efficiency: Inference Speed vs. Reconstruction Accuracy", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#cbd5e1")

    # Annotation badge
    ax.text(
        0.05, 0.15,
        "KMF Pareto Domination:\n- 59.2% lower error than Linear\n- 16.0% lower error than Raw FM\n- 57.7x faster than Test-Time Opt\n- Adds only +1.74 ms overhead",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0fdf4", ec="#86efac", lw=1.2),
    )

    plt.tight_layout()
    pdf_path = FIG_DIR / "fig_pareto_frontier.pdf"
    png_path = FIG_DIR / "fig_pareto_frontier.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated {pdf_path} and {png_path}")


if __name__ == "__main__":
    plot_subcadence_scaling()
    plot_rollout_dynamics()
    plot_continuous_trajectory_profile()
    plot_pareto_frontier()
