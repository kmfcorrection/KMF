#!/usr/bin/env python3
"""Replace stale experimental prose with the refreshed 20260924 protocol."""
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "iclr2027" / "iclr2027_conference.tex"
text = path.read_text()

methodology_start = text.index("\\subsection{Benchmark Specifications and Evaluation Protocols}")
results_start = text.index("\\section{Results and Architectural Analysis}")
methodology = r'''\\subsection{Benchmark Specifications and Evaluation Protocols}

The refreshed benchmark contains 126 completed configuration-level JSON evaluations from the 20260924 reruns. It covers five frozen foundation-model families that were available under the released interfaces: Poseidon, CNO, Polymathic, DPOT, and MORPH. KMF operates outside each model and does not update foundation-model parameters. The primary Task~1 analysis is restricted to models for which the endpoint-conditioned refinement contract is meaningful, namely Poseidon and CNO. DPOT, MORPH, and Polymathic transfer rows are retained in the appendix as contract-specific diagnostics rather than used to define the primary temporal-refinement claim.

The available data systems are NS-Gauss, FNS-KF, ACE, and Wave-Gauss on $128\\times128$ grids. Requested spans range from $0.05$ to $0.40$ seconds. Because the assembled trajectories contain a finite number of raw frames, the longest-span runs use the largest valid rollout horizon for each dataset; unavailable cells are not imputed from the superseded matrix. The refreshed matrix distinguishes the requested cadence from the endpoint span used by the two-sided refinement calculation. Calibration and test trajectories remain disjoint, and all reported fusion or correction strengths are selected without using held-out test errors.

For Task~2 we report two operational regimes. The off-native regime uses $\\Delta t=0.05$ seconds and tests active physical-defect relaxation during causal rollout. The native-cadence regime uses $\\Delta t=0.10$ seconds and tests projection and defect correction at the model's standard endpoint spacing. In both regimes, the corrected endpoint is reinjected before the next foundation-model call, and the dense bridge is evaluated between successive corrected endpoints whenever the dataset provides a scoreable intermediate reference. The appendix contains every completed refreshed JSON row, including rows that are not part of the primary claims.

'''
text = text[:methodology_start] + methodology + text[results_start:]

sub_start = text.index("\\subsection{Sub-Cadence Super-Resolution: Contribution of Kinematic Boundary Physics}")
lat_start = text.index("\\subsection{Inference Latency}")
results = r'''\\subsection{Sub-Cadence Super-Resolution: Contribution of Kinematic Boundary Physics}
\\label{sec:subcadence_results}

Task~1 evaluates the state at an intermediate time between two foundation-model endpoints. The primary table reports the representative Poseidon-B and CNO rows on the two verified fluid systems at the shortest and longest completed spans. All candidates receive the same endpoint information; the raw FM column is an additional practical reference, while linear interpolation is the matched endpoint baseline. The complete refreshed matrix is given in Appendix Table~\\ref{tab:subcadence_master}.

\\begin{table}[t]
\\centering
\\caption{\\textbf{Representative endpoint-conditioned temporal refinement.} Values are held-out RMSE. KMF uses the calibration-selected fusion candidate. The primary table reports Poseidon-B and CNO on the two verified fluid systems; all completed transfer rows are in the appendix.}
\\label{tab:subcadence_results}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{llcccccc}
\\toprule
PDE & Model & $\\Delta t$ & Linear & Raw FM & Hermite & KMF & Gain vs. linear / raw FM \\\\
\\midrule
NS-Gauss & Poseidon-B & 0.10 & 0.06165 & 0.03004 & 0.03200 & \\textbf{0.02509} & +59.30\\% / +16.47\\% \\\\
FNS-KF & Poseidon-B & 0.10 & 0.01018 & 0.01248 & 0.00908 & \\textbf{0.00873} & +14.33\\% / +30.08\\% \\\\
NS-Gauss & CNO & 0.10 & 0.06106 & 0.03681 & 0.03296 & \\textbf{0.02684} & +56.05\\% / +27.09\\% \\\\
FNS-KF & CNO & 0.10 & 0.03110 & 0.09023 & 0.03059 & \\textbf{0.03059} & +1.67\\% / +66.10\\% \\\\
NS-Gauss & Poseidon-B & 0.40 & 0.22715 & 0.00425 & 0.21346 & \\textbf{0.00425} & +98.13\\% / 0.00\\% \\\\
FNS-KF & Poseidon-B & 0.40 & 0.05972 & 0.01076 & 0.05475 & \\textbf{0.01076} & +81.99\\% / 0.00\\% \\\\
NS-Gauss & CNO & 0.40 & 0.22013 & 0.04075 & 0.20168 & \\textbf{0.04075} & +81.49\\% / 0.00\\% \\\\
FNS-KF & CNO & 0.40 & 0.06441 & 0.04502 & 0.06006 & \\textbf{0.04437} & +31.11\\% / +1.44\\% \\\\
\\bottomrule
\\end{tabular}
}
\\end{table}

At the shortest completed fluid span, KMF improves over the matched linear reference in all four representative rows and improves over the raw FM in the rows where the raw model is not already the strongest candidate. At longer spans, calibration appropriately selects the raw FM candidate in several cases. This is a strength of the deployable selection rule, not a missing correction: KMF does not force a physics bridge when the model endpoint is already more accurate. The result is therefore a calibrated inference-time selection between model and physical candidates, rather than a claim that the physical bridge dominates every regime.

\\paragraph{Scope of Task 1.}
The primary Task~1 claim concerns endpoint-conditioned refinement for models with a meaningful endpoint-query or endpoint-transition contract. Polymathic, DPOT, and MORPH rows are included in the appendix to document transfer behavior, but are not used to inflate the main claim. Wave-Gauss is retained as a dispersive stress test; its completed rows show that endpoint-polynomial candidates can be sensitive to phase dispersion, which is precisely why the calibration rule and the matched baselines are reported.

\\subsection{Defect-Corrected Short-Horizon Autoregressive Rollout}
\\label{sec:rollout_results}

Task~2 has two distinct regimes, and they should not be pooled. In the off-native $\\Delta t=0.05$ regime, physical defect relaxation is active and the corrected endpoint is reinjected at every step. In the native $\\Delta t=0.10$ regime, the foundation model is already operating at its standard endpoint spacing, so calibration often selects projection-only correction. The table reports representative Poseidon-B and CNO rows; the full completed rollout matrix is in Appendix Table~\\ref{tab:rollout_master}.

\\begin{table}[t]
\\centering
\\caption{\\textbf{Two Task~2 rollout regimes.} Four-step causal rollout from the same initial state. KMF is selected on calibration trajectories and reinjected before the next model call.}
\\label{tab:rollout_results}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{llcccccc}
\\toprule
Regime & PDE / Model & Raw & Projected & KMF & Gain vs. raw & $\\alpha^*$ & Dense KMF vs. raw-linear \\\\
\\midrule
Off-native $0.05$ & NS-Gauss / Poseidon-T & 0.14967 & 0.10534 & \\textbf{0.08853} & +40.85\\% & 0.20 & not scoreable \\\\
Off-native $0.05$ & FNS-KF / Poseidon-T & 0.15862 & 0.14629 & \\textbf{0.13089} & +17.48\\% & 0.20 & not scoreable \\\\
Off-native $0.05$ & NS-Gauss / CNO & 0.07801 & 0.07592 & \\textbf{0.06277} & +19.53\\% & 0.20 & not scoreable \\\\
Off-native $0.05$ & FNS-KF / CNO & 0.15573 & 0.15410 & \\textbf{0.13710} & +11.96\\% & 0.20 & not scoreable \\\\
Native $0.10$ & NS-Gauss / Poseidon-B & 0.00420 & 0.00394 & 0.00394 & +6.21\\% & 0.00 & 0.02571 vs. 0.11326 \\\\
Native $0.10$ & FNS-KF / Poseidon-B & 0.01434 & 0.01431 & 0.01431 & +0.23\\% & 0.00 & 0.01412 vs. 0.02895 \\\\
Native $0.10$ & NS-Gauss / CNO & 0.07639 & 0.07535 & \\textbf{0.06775} & +11.31\\% & 0.20 & 0.06567 vs. 0.10045 \\\\
Native $0.10$ & FNS-KF / CNO & 0.12601 & 0.12091 & \\textbf{0.11154} & +11.48\\% & 0.20 & 0.09285 vs. 0.12518 \\\\
\\bottomrule
\\end{tabular}
}
\\end{table}

The off-native regime gives the clearest evidence for active physical correction: every representative row selects a positive relaxation factor and improves over the raw causal rollout by 11.96--40.85\\%. In the native regime, Poseidon-B selects $\\alpha^*=0$ on these fluid rows, so its gain is attributable to projection rather than defect relaxation; CNO selects an active correction on both systems. This separation avoids treating projection-only native-cadence gains as evidence for universal defect relaxation. Dense evaluation is scoreable at $\\Delta t\\ge0.10$ because an intermediate reference frame exists. For example, at native $0.10$ seconds the Poseidon-B NS-Gauss dense KMF window RMSE is $0.02571$ versus $0.11326$ for the raw-linear bridge, while CNO gives $0.06567$ versus $0.10045$. At $0.05$ seconds the raw data provide no additional intermediate frame, so the dense metric is correctly marked unavailable while the causal endpoint rollout remains fully scored.

The independent calibration-resampling audit supports the off-native conclusion: the same fixed 50-trajectory test set was improved over projection-only rollout by 16.42\\% on NS-Gauss and 5.43\\% on FNS-KF when positive relaxation was selected from repeated five-trajectory calibration subsets. The correction-suite audit further compares KMF with residual-gradient, live Gauss--Newton, a cached-Jacobian PhysicsCorrect-style adapter, and a solver control under the same causal contract. These are algorithmic comparisons, not claims of official reproduction when the released state representations differ.

\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/fig_rollout_dynamics.pdf}
\\caption{\\textbf{Causal rollout and dense-bridge diagnostics.} The figure illustrates the two Task~2 regimes: off-native active defect correction and native-cadence projection or calibrated correction. Dense intermediate states are evaluated by chaining bridges between corrected endpoints and require no forward PDE substeps.}
\\label{fig:rollout_dynamics}
\\end{figure*}

'''
text = text[:sub_start] + results + text[lat_start:]

# Replace stale exact-number captions and conclusion claims while preserving the figures.
text = text.replace(
    r"\\caption{\\textbf{Pointwise residual error fields $\\|u - u_{\\text{true}}\\|_2$ at sub-cadence midpoint $t_{1/2}$.} Comparison of Ground Truth, Linear Secant Baseline, Raw Foundation Model (Poseidon-B), and Calibrated Kinematic Manifold Fusion (KMF, Ours) on 2D incompressible Navier--Stokes ($128\\times 128$, $\\Delta t = 0.10\\,\\text{s}$). KMF eliminates convective phase distortion and suppresses turbulent filament error, achieving a 14.9\\% error reduction over the raw foundation model and 57.9\\% over linear interpolation.}",
    r"\\caption{\\textbf{Illustrative endpoint-conditioned refinement.} Pointwise error fields compare the matched endpoint reference, a raw foundation-model estimate, and KMF at a sub-cadence query. The figure is qualitative; numerical claims are taken from the refreshed held-out tables.}")
text = text.replace(
    r"We introduce KMF (Kinematic Manifold Fusion), an endpoint-PDE interface for two-sided sub-cadence interpolation and short-horizon autoregressive correction. It combines classical endpoint Hermite reconstruction with calibration-selected fusion and, for periodic incompressible fields, exact Leray projection, without forward PDE substeps. A real Poseidon-B audit with projection-matched controls confirms gains of 14.28\\% on NS-Gauss and 33.07\\% on FNS-KF over Raw FM+Projection, with a 2.15 ms operator overhead. The full matrix also identifies dispersive and transfer settings where endpoint interpolation or one-step defect calibration is insufficient. KMF is therefore a practical, explicitly scoped complement to continuous-time learned operators and numerical solvers.",
    r"We introduce KMF (Kinematic Manifold Fusion), an endpoint-PDE interface for temporal refinement and short-horizon autoregressive correction. It combines endpoint Hermite reconstruction, calibration-selected fusion, and exact Leray projection where the constraint is known, without forward PDE substeps. The refreshed evaluation separates off-native active correction from native-cadence projection, reports dense bridge scores only when intermediate references exist, and retains transfer and dispersive cases as scope diagnostics. KMF is therefore a practical, explicitly scoped complement to continuous-time learned operators and numerical solvers.")

app_start = text.index("\\subsection{Comprehensive 200-Benchmark Evaluation Breakdown}")
end_doc = text.index("\\end{document}")
appendix = r'''\\subsection{Refreshed Benchmark Matrix}
\\label{sec:app_full_benchmark_matrix}

The following appendix tables are generated exclusively from the 126 completed JSON files in the refreshed 20260924 runs. They report the requested cadence, the endpoint span used by Task~1, all completed baselines, the calibration-selected KMF value, and the corresponding causal rollout fields. Missing combinations are absent rather than copied from the earlier matrix. In particular, longer Wave-Gauss rows are limited by the number of available raw frames. The primary discussion intentionally focuses Task~1 on Poseidon and CNO, while these tables retain the completed transfer rows for auditability.

\\input{generated_master_tables.tex}

'''
text = text[:app_start] + appendix + text[end_doc:]
path.write_text(text)
print(f"updated {path}")
