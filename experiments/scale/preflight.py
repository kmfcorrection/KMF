#!/usr/bin/env python3
"""preflight -- what this node actually gives you, and where the run will die.

Run this on a new machine before launching anything long. It reports the limits
that matter and then executes the first few steps of the real pipeline one at a
time, printing host and GPU memory after each, so a kill lands on a labelled
line instead of an unattributable "Killed".

"Killed" with no traceback is SIGKILL: the host OOM killer or a cgroup memory
limit. It is *not* GPU OOM, which raises `torch.cuda.OutOfMemoryError` with a
stack trace. The distinction matters because the fixes are opposite -- GPU OOM
wants a smaller batch, host OOM usually wants a smaller dataset generation or
more --mem.

    python3 experiments/scale/preflight.py --pde ns2d_forced --grid 128
"""
from __future__ import annotations

import argparse
import os
import resource
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _rss_gb() -> float:
    """Peak resident set size in GB. ru_maxrss is bytes on macOS, KB on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def _host_mem() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            info = dict(l.split(":", 1) for l in fh)
        out["total_gb"] = float(info["MemTotal"].split()[0]) / 1e6
        out["available_gb"] = float(info["MemAvailable"].split()[0]) / 1e6
    except Exception:
        pass
    # cgroup limits are what SLURM actually enforces; the machine's MemTotal is
    # irrelevant if the job is capped well below it. The limit lives on the
    # *job's own* nested cgroup (.../job_12345/memory.max), not at the root --
    # reading the root returns "max" and makes a capped job look uncapped, which
    # is exactly how a 2 GB allocation on a 2 TB node hides.
    candidates = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            rel = line.split(":")[-1].lstrip("/")
            node = Path("/sys/fs/cgroup") / rel
            # walk up: the limit may be set on an ancestor of the leaf
            while True:
                candidates.append(node / "memory.max")
                candidates.append(node / "memory.limit_in_bytes")
                if node == Path("/sys/fs/cgroup") or node.parent == node:
                    break
                node = node.parent
    except Exception:
        pass
    candidates += [Path("/sys/fs/cgroup/memory.max"),
                   Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")]
    limits = []
    for path in candidates:
        try:
            v = path.read_text().strip()
            if v not in ("max", ""):
                limits.append(int(v) / 1e9)
        except Exception:
            continue
    # A limit larger than the machine is the kernel's "unlimited" sentinel.
    limits = [v for v in limits if 0 < v < out.get("total_gb", 1e12) * 1.01]
    if limits:
        out["cgroup_limit_gb"] = min(limits)
    return out


def report(label: str) -> None:
    import torch
    line = f"  [{label:<28s}] host peak RSS {_rss_gb():6.2f} GB"
    if torch.cuda.is_available():
        line += (f" | gpu alloc {torch.cuda.memory_allocated()/1e9:5.2f} GB"
                 f" reserved {torch.cuda.max_memory_reserved()/1e9:5.2f} GB")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pde", default="ns2d_forced")
    ap.add_argument("--arch", default="fno2d")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=64)
    args = ap.parse_args()

    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    import torch
    mem = _host_mem()
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count()
    print(f"  python           {sys.version.split()[0]}")
    print(f"  torch            {torch.__version__}")
    print(f"  cpus available   {ncpu}   (os.cpu_count reports {os.cpu_count()})")
    for k, v in mem.items():
        print(f"  {k:16s} {v:.1f}")
    if "cgroup_limit_gb" in mem:
        print(f"  -> the cgroup limit is the one SLURM enforces; a kill happens "
              f"there, not at MemTotal")
    slurm = {v: os.environ[v] for v in
             ("SLURM_JOB_ID", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
              "SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "SLURM_JOB_GPUS")
             if os.environ.get(v)}
    for k, v in slurm.items():
        print(f"  {k:16s} {v}")
    if slurm and "SLURM_MEM_PER_NODE" not in slurm:
        per_cpu = slurm.get("SLURM_MEM_PER_CPU")
        budget = (f"{int(per_cpu) * ncpu / 1024:.1f} GB "
                  f"({per_cpu} MB/cpu x {ncpu} cpu)") if per_cpu else "unknown"
        print(f"  -> no --mem set, so the job budget is mem-per-cpu x cpus = "
              f"{budget}.")
        print(f"     With {ncpu} cpu(s) that is the real ceiling, not the "
              f"{mem.get('total_gb', 0):.0f} GB the node has.")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  gpu              {p.name}, {p.total_memory/1e9:.1f} GB, "
              f"sm_{p.major}{p.minor}, n={torch.cuda.device_count()}")
        print(f"  bf16 supported   {torch.cuda.is_bf16_supported()}")
    else:
        print("  gpu              NONE -- everything below will run on CPU")
    du = shutil.disk_usage(".")
    print(f"  disk free        {du.free/1e9:.1f} GB")

    print("\n" + "=" * 78)
    print("PIPELINE STEPS  (a kill lands on the line after the last one printed)")
    print("=" * 78)
    from hipp.scale.adapters import FrozenFM, check_adapter
    from hipp.scale.curvature_scale import estimate_scale
    from hipp.scale.data2d import SPECS2D, dataset_dir
    from hipp.scale.models2d import build_model, count_params
    from hipp.utils import get_device

    device = get_device()
    report("start")

    # --- dataset -------------------------------------------------------
    root = dataset_dir(args.pde, args.grid, 0, args.stride)
    if (root / "manifest.json").exists():
        sizes = {p.name: p.stat().st_size / 1e9 for p in root.glob("*.npy")}
        print(f"  dataset present: {root}")
        for n, gb in sorted(sizes.items()):
            print(f"      {n:12s} {gb:6.2f} GB")
        print("  -> training memory-maps these; it does not load them whole")
    else:
        gb = 512 * 41 * args.grid ** 2 * 4 / 1e9
        print(f"  dataset MISSING: {root}")
        print(f"  -> train2d generates it implicitly with defaults (512/128/128 "
              f"trajectories, 40 steps, ~{gb:.2f} GB on disk)")
        print(f"  -> it streams to disk a batch at a time, so peak host RAM is "
              f"one batch, not the whole split")
        print("  -> generate it explicitly first anyway, so a failure there is "
              "not mistaken for a training failure:")
        print(f"       python3 -m hipp.scale.data2d --pde {args.pde} "
              f"--n {args.grid}")
        return

    # --- model ----------------------------------------------------------
    model = build_model(args.arch, in_channels=1, width=args.width,
                        modes=args.modes, n_layers=args.layers)
    n_par = count_params(model)
    print(f"  model {args.arch}: {n_par/1e6:.1f}M params "
          f"({n_par*4/1e9:.2f} GB fp32; ~{n_par*4*4/1e9:.2f} GB with AdamW states)")
    report("model built on host")
    model = model.to(device)
    model.set_norm(0.0, 1.0)
    report("model moved to device")

    # --- one training step ------------------------------------------------
    x = torch.randn(args.batch, 1, args.grid, args.grid, device=device)
    use_amp = device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        loss = torch.nn.functional.mse_loss(model(x), torch.randn_like(x))
    loss.backward()
    report(f"fwd+bwd batch={args.batch}")
    model.zero_grad(set_to_none=True)

    # --- one curvature estimate -------------------------------------------
    for p in model.parameters():
        p.requires_grad_(False)
    fm = FrozenFM(model, (1, args.grid, args.grid), name=args.arch, device=device)
    chk = check_adapter(fm, verbose=False)
    print(f"  adapter check: pass={chk['pass']}  "
          f"jvp-vs-fd {chk.get('jvp_vs_fd_rel', float('nan')):.2e}  "
          f"adjoint {chk.get('adjoint_rel', float('nan')):.2e}")
    report("adapter check")

    c = torch.randn(fm.N, device=device)
    est = estimate_scale(fm, c, method="pushforward", k=args.k, chunk=8)
    print(f"  rank-{args.k} estimate: {est.meta['jvp_calls']} JVP + "
          f"{est.meta['vjp_calls']} VJP passes  (dense would need {fm.N})")
    report(f"curvature k={args.k}")

    print("\nAll preflight steps completed. If the real run still dies, it dies "
          "later than\nanything checked here -- most likely in dataset generation "
          "or the dataloader.")


if __name__ == "__main__":
    import torch  # noqa: E402  (imported after the env report starts)
    main()
