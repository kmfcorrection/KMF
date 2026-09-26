from pathlib import Path

p = Path(__file__).resolve().parents[2] / "iclr2027" / "iclr2027_conference.tex"
s = p.read_text()
start = s.index("\\\\subsection{Benchmark Specifications and Evaluation Protocols}")
end = s.index("\\section{Discussion and Limitations}", start)
block = s[start:end]
while "\\\\" in block:
    block = block.replace("\\\\", "\\")
s = s[:start] + block + s[end:]
p.write_text(s)
print(f"normalized rewritten block in {p}")
