from pathlib import Path

p = Path(__file__).resolve().parents[2] / "iclr2027" / "iclr2027_conference.tex"
s = p.read_text()
start = s.index("\\subsection{Benchmark Specifications and Evaluation Protocols}")
end = s.index("\\section{Discussion and Limitations}", start)
lines = s[start:end].splitlines()
out = []
for line in lines:
    stripped = line.rstrip()
    if "&" in stripped and stripped.endswith("\\") and not stripped.endswith("\\\\"):
        line = stripped + "\\"
    out.append(line)
s = s[:start] + "\n".join(out) + "\n" + s[end:]
p.write_text(s)
print(f"fixed table row breaks in {p}")
