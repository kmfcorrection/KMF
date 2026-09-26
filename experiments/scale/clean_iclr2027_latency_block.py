from pathlib import Path

p = Path(__file__).resolve().parents[2] / "iclr2027" / "iclr2027_conference.tex"
s = p.read_text()
start = s.index("\\subsection{Inference Latency}")
end = s.index("\\section{Discussion and Limitations}", start)
block = r'''\\subsection{Inference Latency}
\\label{sec:latency_results}

Table~\\ref{tab:latency} reports the refreshed synchronized Poseidon-B real-pipeline timing audit on an NVIDIA A100-SXM4-40GB (CUDA 12.4, PyTorch 2.4, batch size one, $128\\times128$). The measured operator overhead is $3.33$ ms on top of a $96.33$ ms FM forward pass, for $99.66$ ms end-to-end. This is an endpoint-reconstruction measurement; it does not claim a universal runtime ordering against numerical solvers.

\\begin{table}[t]
\\centering
\\caption{\\textbf{Refreshed real-pipeline latency audit.} Synchronized wall-clock time on an NVIDIA A100-SXM4-40GB, CUDA 12.4, PyTorch 2.4, batch size one, and a $128\\times128$ field.}
\\label{tab:latency}
\\resizebox{0.72\\columnwidth}{!}{
\\begin{tabular}{lrr}
\\toprule
\\textbf{Pipeline configuration} & \\textbf{Latency} & \\textbf{vs. FM} \\\\
\\midrule
Raw foundation model (Poseidon-B) & $96.33\\,\\text{ms}$ & $1.00\\times$ \\\\
Two RHS evaluations & $2.40\\,\\text{ms}$ & $0.025\\times$ \\\\
Helmholtz--Leray projection & $0.52\\,\\text{ms}$ & $0.005\\times$ \\\\
Bridge arithmetic & $0.41\\,\\text{ms}$ & $0.004\\times$ \\\\
\\textbf{FM + KMF operator pipeline} & $\\mathbf{99.66\\,\\text{ms}}$ & $\\mathbf{1.03\\times$} \\\\
\\bottomrule
\\end{tabular}
}
\\end{table}

'''
s = s[:start] + block + s[end:]
p.write_text(s)
print(f"updated {p}")
