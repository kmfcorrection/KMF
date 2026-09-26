#!/usr/bin/env python3
"""Generate the two compact figures used by the KMF paper."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "iclr2027" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2F6FDB"
GREEN = "#169B55"
ORANGE = "#D97706"
RED = "#C83C3C"
INK = "#1F2937"
MUTED = "#52606D"
GRID = "#D9E2EC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def box(ax, xy, wh, text, face="#F7FAFC", edge="#B8C7D9", color=INK,
        fontsize=9, weight="normal", radius=0.02):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.2, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=color, fontsize=fontsize, weight=weight,
            transform=ax.transAxes, linespacing=1.15)


def arrow(ax, start, end, color=MUTED, lw=1.4, style="-|>", curve=0.0):
    connection = "arc3" if curve == 0 else f"arc3,rad={curve}"
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle=style,
        mutation_scale=12, linewidth=lw, color=color,
        connectionstyle=connection, clip_on=False,
    ))


def architecture():
    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.965, "Kinematic Manifold Fusion", fontsize=11.5,
            weight="normal", color=INK, va="top")
    ax.text(0.02, 0.905,
            "Inference layer around a frozen PDE foundation model",
            fontsize=8.0, color=MUTED, va="top")

    ax.plot([0.505, 0.505], [0.10, 0.84], color=GRID, lw=1.0)
    ax.text(0.25, 0.835, "Task 1: temporal refinement", ha="center",
            fontsize=9.5, weight="normal", color=BLUE)
    ax.text(0.75, 0.835, "Task 2: causal rollout correction", ha="center",
            fontsize=9.5, weight="normal", color=GREEN)

    # Task 1: two model endpoints -> local rates -> bridge -> optional fusion.
    box(ax, (0.04, 0.67), (0.18, 0.08), "$\\tilde u_k$\nFM endpoint", fontsize=8.2)
    box(ax, (0.28, 0.67), (0.18, 0.08), "$\\tilde u_{k+1}$\nFM endpoint", fontsize=8.2)
    box(ax, (0.15, 0.52), (0.20, 0.075), "evaluate PDE RHS\nat both endpoints",
        face="#EEF5FF", edge=BLUE, color=BLUE, fontsize=7.8)
    box(ax, (0.15, 0.37), (0.20, 0.075), "Hermite bridge\n$u_H(s)$",
        face="#EFFAF4", edge=GREEN, color=GREEN, fontsize=8.0)
    box(ax, (0.39, 0.37), (0.10, 0.075), "direct FM\nquery",
        face="#FFF7ED", edge=ORANGE, color=ORANGE, fontsize=7.0)
    box(ax, (0.15, 0.19), (0.20, 0.075), "fuse candidates\n+ project",
        face="#F3F0FF", edge="#7456B8", color="#5B3D96", fontsize=7.9)
    arrow(ax, (0.13, 0.67), (0.205, 0.595), color=BLUE)
    arrow(ax, (0.37, 0.67), (0.295, 0.595), color=BLUE)
    arrow(ax, (0.25, 0.52), (0.25, 0.445), color=GREEN)
    arrow(ax, (0.25, 0.37), (0.25, 0.265), color=GREEN)
    arrow(ax, (0.44, 0.37), (0.35, 0.255), color=ORANGE)
    ax.text(0.25, 0.105, r"return state at any $t\in[t_k,t_{k+1}]$", ha="center",
            fontsize=7.0, color=MUTED)

    # Task 2: causal candidate -> defect -> corrected state -> reinjection.
    box(ax, (0.55, 0.67), (0.18, 0.08), "$\\hat u_k$\ncorrected input", fontsize=7.9)
    box(ax, (0.77, 0.67), (0.18, 0.08), "$\\tilde u_{k+1}$\nFM candidate", fontsize=7.9)
    box(ax, (0.665, 0.52), (0.18, 0.075), "evaluate local\nphysical defect",
        face="#EEF5FF", edge=BLUE, color=BLUE, fontsize=7.7)
    box(ax, (0.665, 0.37), (0.18, 0.075), "relax by $\\alpha$\n+ project",
        face="#EFFAF4", edge=GREEN, color=GREEN, fontsize=7.9)
    box(ax, (0.665, 0.19), (0.18, 0.075), "$\\hat u_{k+1}$\ncorrected output",
        face="#F3F0FF", edge="#7456B8", color="#5B3D96", fontsize=7.9)
    arrow(ax, (0.73, 0.71), (0.77, 0.71), color=MUTED)
    arrow(ax, (0.64, 0.67), (0.70, 0.595), color=BLUE)
    arrow(ax, (0.86, 0.67), (0.81, 0.595), color=BLUE)
    arrow(ax, (0.755, 0.52), (0.755, 0.445), color=GREEN)
    arrow(ax, (0.755, 0.37), (0.755, 0.265), color=GREEN)
    arrow(ax, (0.845, 0.23), (0.95, 0.71), color="#7456B8", curve=0.28)
    ax.text(0.92, 0.47, "reinject", ha="center", fontsize=7.0,
            color="#5B3D96", rotation=72)
    ax.text(0.755, 0.105, "repeat causally; no temporal ODE substeps", ha="center",
            fontsize=7.0, color=MUTED)

    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "fig_kmf_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_kmf_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def results_summary():
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.85))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.82,
                        wspace=0.32)

    # Task 1: selected endpoint-refinement gains versus raw FM.
    ax = axes[0]
    labels = ["NS-G\nB .10", "NS-G\nT .10", "FNS\nB .10", "FNS\nT .10",
              "ACE\nB .10", "Wave\nB .40"]
    gains = [16.49, 46.15, 30.01, 86.11, 27.37, 13.87]
    colors = [BLUE, BLUE, GREEN, GREEN, ORANGE, ORANGE]
    ax.bar(np.arange(len(labels)), gains, color=colors, width=0.68)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("KMF gain vs raw FM (%)")
    ax.set_title("Task 1: refinement", loc="left",
                 fontsize=8, weight="normal")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for i, v in enumerate(gains):
        ax.text(i, v + 2.0, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5,
                color=INK)
    ax.set_ylim(0, 100)

    # Task 2: both regimes, representative four-step rows.
    ax = axes[1]
    labels = ["NS-G\nB", "NS-G\nT", "FNS\nB", "FNS\nT", "ACE\nB", "Wave\nB"]
    off = [16.52, 40.85, 7.01, 17.48, 10.66, 1.55]
    native = [6.21, 4.47, 0.23, 4.03, 8.28, 12.84]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, off, width=0.34, color=GREEN, label="off-native $\\Delta t=0.05$")
    ax.bar(x + 0.18, native, width=0.34, color=BLUE, label="native $\\Delta t=0.10$")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("KMF gain vs raw (%)")
    ax.set_title("Task 2: rollout regimes", loc="left",
                 fontsize=8, weight="normal")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=7)
    ax.set_ylim(0, 48)

    # Long horizon.
    ax = axes[2]
    horizons = np.array([4, 10, 20])
    ns = [16.52, 16.08, 14.00]
    fns = [7.01, 11.79, 14.64]
    ax.plot(horizons, ns, marker="o", lw=2.2, color=BLUE, label="NS-Gauss")
    ax.plot(horizons, fns, marker="o", lw=2.2, color=GREEN, label="FNS-KF")
    ax.set_xticks(horizons)
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel("Gain vs projection (%)")
    ax.set_title("Task 2: long horizon", loc="left",
                 fontsize=8, weight="normal")
    ax.grid(color=GRID, lw=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_ylim(0, 20)

    fig.suptitle("KMF results at a glance", fontsize=10, weight="normal", y=0.96)
    fig.savefig(OUT / "fig_kmf_results_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_kmf_results_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def solver_controls():
    """Plot the ACE and Wave-Gauss fair solver-control audit."""
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.95))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.84,
                        wspace=0.30)
    audits = {
        "FNS-KF": {
            "endpoint": [("FM endpoint", .012534127, 93.349),
                         ("linear", .010176356, .510),
                         ("KMF bridge", .008928397, 2.251),
                         ("KMF fusion", .008390486, 95.858)],
            "rk2": [(.036589108, 1.946), (.036591009, 2.979),
                    (.036591917, 5.033), (.036592195, 9.114),
                    (.036592270, 17.735)],
            "rk4": [(.036592280, 3.076), (.036592298, 5.201),
                    (.036592297, 9.458), (.036592296, 17.929),
                    (.036592296, 34.952)],
        },
        "ACE": {
            "endpoint": [("FM endpoint", .16839938, .184),
                         ("linear", .099232878, .386),
                         ("KMF bridge", .098375761, .779),
                         ("KMF fusion", .098375761, .784)],
            "rk2": [(0.022755466, .756), (0.022755281, 1.153),
                    (0.022755225, 1.841), (0.022755210, 3.264),
                    (0.022755206, 6.201)],
            "rk4": [(0.022755204, 1.158), (0.022755205, 1.883),
                    (0.022755205, 3.472), (0.022755205, 6.444),
                    (0.022755205, 12.601)],
        },
        "Wave-Gauss": {
            "endpoint": [("FM endpoint", .092019539, .190),
                         ("linear", .074334079, .588),
                         ("KMF bridge", .069417108, .917),
                         ("KMF fusion", .081420401, 1.218)],
            "rk2": [(85.026716, .909), (529611.47, 1.445),
                    (3.4124003e14, 2.553), (4.6612618e30, 4.761),
                    (6.9846859e60, 9.024)],
            "rk4": [(1641249.4, 1.251), (2.8936169e15, 2.200),
                    (6.1763282e32, 4.086), (1.1679987e66, 7.756),
                    (7.8259739e123, 15.095)],
        },
    }
    for ax, (title, data) in zip(axes, audits.items()):
        endpoint = data["endpoint"]
        ax.scatter([row[2] for row in endpoint], [row[1] for row in endpoint],
                   marker="o", s=34, color=ORANGE, label="endpoint-conditioned")
        for label, y, x in endpoint:
            if label.startswith("KMF"):
                dy = 8 if label == "KMF bridge" else -12
                ax.annotate(label, (x, y), xytext=(4, dy),
                            textcoords="offset points", fontsize=6.5, color=INK)
        for key, color, marker in (("rk2", BLUE, "s"), ("rk4", RED, "^")):
            vals = data[key]
            ax.plot([row[1] for row in vals], [row[0] for row in vals],
                    marker=marker, ms=4.5, lw=1.4, color=color,
                    label=key.upper())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9, weight="normal")
        ax.set_xlabel("Online latency (ms)")
        ax.set_ylabel("Midpoint RMSE")
        ax.grid(color=GRID, lw=0.8, alpha=0.8, which="both")
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), fontsize=7.5)
    fig.suptitle("Endpoint-conditioned KMF and causal solver controls",
                 fontsize=11, weight="normal", y=1.08)
    fig.savefig(OUT / "fig_kmf_solver_controls.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_kmf_solver_controls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    results_summary()
    print(f"wrote figures to {OUT}")
