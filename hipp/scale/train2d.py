"""Train the frozen surrogate at scale, on GPU.

Same contract as `hipp/train.py`: whatever comes out is treated downstream as a
frozen black box. What is different is that the data no longer fits in memory
(a 512-trajectory 128x128 set is ~4 GB per split) and the model no longer trains
in five minutes, so this adds streaming, AMP, checkpoint/resume and optional DDP.

Design notes that matter for the method rather than for throughput:

* **One-step MSE, nothing else.** The whole premise is a model trained on a
  deterministic one-step reconstruction loss with no probabilistic head. Adding
  a rollout loss or a noise-injection schedule would change what f is and make
  the small-scale comparison meaningless. Rollout RMSE is *reported* every
  validation pass, never optimized.

* **Ensembles get different data order and init, not different data.** The deep
  ensemble is a baseline for the same predictive question, so it has to see the
  same training set; varying the data would make it a different estimator.

* **SWAG is collected at constant LR over the final epochs**, matching the 1D
  code, so the SWAG baseline means the same thing at both scales.

* **float32 master weights, bf16 autocast, fp32 FFT.** fp16 is avoided: the
  spectral weights of an FNO have a wide dynamic range and fp16 loss scaling
  interacts badly with the FFT. bf16 has the range and needs no scaler -- but
  cuFFT has no reduced-precision kernel at all, so `SpectralConv2d` disables
  autocast internally and runs in fp32. Everything else stays in bf16.

Usage
-----
    python -m hipp.scale.train2d --pde ns2d_forced --arch fno2d \
        --width 64 --modes 32 --layers 4 --epochs 60 --batch 16 --ensemble 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils import CKPT_DIR, get_device, print_header, set_seed
from .data2d import SPECS2D, ShardedTrajectories, build_dataset2d
from .models2d import build_model, count_params


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """One-step (input, target) pairs indexed over a memory-mapped split.

    The memmap is opened lazily per worker: an already-open np.memmap does not
    survive fork cleanly on all platforms and silently returns zeros in the
    child, which shows up as a model that trains to a plateau for no reason.
    """

    def __init__(self, root: Path, split: str):
        self.root, self.split = Path(root), split
        self._arr = None
        with open(self.root / "manifest.json") as fh:
            man = json.load(fh)
        shape = np.load(self.root / f"{split}.npy", mmap_mode="r").shape
        self.n_traj, self.t_len = shape[0], shape[1] - 1
        self.grid = shape[-1]
        self.mu, self.sigma = man["mu"], man["sigma"]

    def __len__(self) -> int:
        return self.n_traj * self.t_len

    def _array(self):
        if self._arr is None:
            self._arr = np.load(self.root / f"{self.split}.npy", mmap_mode="r")
        return self._arr

    def __getitem__(self, idx: int):
        a = self._array()
        i, t = divmod(int(idx), self.t_len)
        x = torch.from_numpy(np.array(a[i, t], dtype=np.float32))
        y = torch.from_numpy(np.array(a[i, t + 1], dtype=np.float32))
        return x.unsqueeze(0), y.unsqueeze(0)          # (1, H, W)


def _default_workers() -> int:
    """CPUs this process may actually use, capped at 4.

    `os.cpu_count()` reports the machine, not the allocation: on a SLURM node
    with --cpus-per-task=1 it can return 128 while the process is pinned to one
    core. `sched_getaffinity` reports the pinned set, which is the number that
    matters -- oversubscribed workers contend for the same core and the loader
    ends up slower than num_workers=0.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:                       # not on Linux
        n = os.cpu_count() or 1
    return max(0, min(4, n - 1))


def make_loader(root: Path, split: str, batch: int, workers: int = 0,
                shuffle: bool = True, seed: int = 0) -> DataLoader:
    ds = PairDataset(root, split)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=shuffle, generator=g,
                      persistent_workers=workers > 0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 40) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        tot += float(torch.nn.functional.mse_loss(model(x), y)) * x.shape[0]
        n += x.shape[0]
    return tot / max(n, 1)


@torch.no_grad()
def rollout_rmse(model, ds: ShardedTrajectories, split: str, device,
                 n_traj: int = 8, steps: int = 10) -> list[float]:
    """Per-step RMSE under autoregressive rollout, reported not optimized.

    This is the quantity the uncertainty is ultimately about: it shows where the
    model's own error stops being one-step noise and starts being accumulated,
    which is exactly the regime (Sigma_c != 0) in which the pushforward
    mechanism is predicted to exist at all.
    """
    model.eval()
    arr = ds.split(split)
    n_traj = min(n_traj, arr.shape[0])
    steps = min(steps, arr.shape[1] - 1)
    truth = torch.as_tensor(np.array(arr[:n_traj, :steps + 1]), dtype=torch.float32,
                            device=device)
    c = truth[:, 0].unsqueeze(1)
    out = []
    for t in range(steps):
        c = model(c)
        err = (c.squeeze(1) - truth[:, t + 1])
        out.append(float(err.pow(2).mean().sqrt()))
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def ckpt_path(pde: str, arch: str, member: int = 0, dropout: float = 0.0,
              tag: str = "") -> Path:
    d = CKPT_DIR / "scale"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{pde}_{arch}_m{member}"
    if dropout > 0:
        name += f"_do{dropout:g}"
    if tag:
        name += f"_{tag}"
    return d / f"{name}.pt"


def train_one(pde: str, arch: str = "fno2d", member: int = 0, epochs: int = 60,
              batch: int = 16, lr: float = 1e-3, weight_decay: float = 1e-5,
              width: int = 64, modes: int = 32, n_layers: int = 4,
              dropout: float = 0.0, in_channels: int = 1, workers: int | None = None,
              swag_epochs: int = 0, device=None, amp: bool = True,
              grid: int | None = None, seed: int | None = None,
              data_seed: int = 0, stride: int | None = None,
              resume: bool = True, verbose: bool = True,
              max_swag_dev: int = 20):
    device = device or get_device()
    workers = _default_workers() if workers is None else workers
    set_seed(1234 + member if seed is None else seed)

    ds = build_dataset2d(pde, n=grid, seed=data_seed, device=device,
                         **({'stride': stride} if stride else {}))
    root = ds.root
    mu, sd = ds.normalizer()

    model = build_model(arch, in_channels=in_channels, width=width, modes=modes,
                        n_layers=n_layers, dropout=dropout).to(device)
    model.set_norm(mu, sd)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - swag_epochs, 1))

    path = ckpt_path(pde, arch, member, dropout)
    start_ep, best = 0, float("inf")
    if resume and path.exists():
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("epochs_done", 0) < epochs:
            model.load_state_dict(payload["state_dict"])
            if "opt" in payload:
                opt.load_state_dict(payload["opt"])
            start_ep = payload.get("epochs_done", 0)
            best = payload.get("val_mse", float("inf"))
            if verbose:
                print(f"  resuming {path.name} from epoch {start_ep}")

    tr = make_loader(root, "train", batch, workers, shuffle=True, seed=member)
    va = make_loader(root, "val", batch, max(workers // 2, 0), shuffle=False)

    use_amp = amp and device.type == "cuda"
    swag_mean = swag_sq = None
    swag_dev: list[torch.Tensor] = []
    n_swag = 0
    best_state = None
    t0 = time.time()

    for ep in range(start_ep, epochs):
        model.train()
        tot, seen = 0.0, 0
        for x, y in tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]

        in_swag = swag_epochs > 0 and ep >= epochs - swag_epochs
        if in_swag:
            for g in opt.param_groups:
                g["lr"] = lr * 0.1
            flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
            if swag_mean is None:
                swag_mean = torch.zeros_like(flat)
                swag_sq = torch.zeros_like(flat)
            n_swag += 1
            swag_mean += (flat - swag_mean) / n_swag
            swag_sq += (flat ** 2 - swag_sq) / n_swag
            swag_dev.append((flat - swag_mean).clone())
            swag_dev[:] = swag_dev[-max_swag_dev:]
        else:
            sched.step()

        vl = evaluate(model, va, device)
        if vl < best and not in_swag:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  [{pde}/{arch} m{member}] ep {ep:3d}  train {tot/max(seen,1):.4e}  "
                  f"val {vl:.4e}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        _save(path, model, opt, arch, pde, ds, dict(
            in_channels=in_channels, width=width, modes=modes,
            n_layers=n_layers, dropout=dropout), best, ep + 1, swag_mean,
            swag_sq, swag_dev, time.time() - t0)

    if best_state is not None and swag_epochs == 0:
        model.load_state_dict(best_state)
        _save(path, model, opt, arch, pde, ds, dict(
            in_channels=in_channels, width=width, modes=modes,
            n_layers=n_layers, dropout=dropout), best, epochs, swag_mean,
            swag_sq, swag_dev, time.time() - t0)

    if verbose:
        rr = rollout_rmse(model, ds, "val", device)
        print(f"  saved {path.name}  val MSE {best:.4e}  "
              f"({count_params(model)/1e6:.1f}M params, {time.time()-t0:.0f}s)")
        print(f"  rollout RMSE steps 1/5/10: "
              f"{rr[0]:.4f} / {rr[min(4,len(rr)-1)]:.4f} / {rr[-1]:.4f}")
    return model, path


def _save(path, model, opt, arch, pde, ds, config, best, epochs_done,
          swag_mean, swag_sq, swag_dev, seconds):
    payload = {
        "state_dict": model.state_dict(), "opt": opt.state_dict(),
        "arch": arch, "config": config, "pde": pde,
        "spec": asdict(ds.spec), "norm": ds.normalizer(),
        "val_mse": best, "epochs_done": epochs_done, "seconds": seconds,
        "name": f"{pde}_{arch}",
    }
    if swag_mean is not None:
        payload["swag"] = {"mean": swag_mean.cpu(),
                           "var": torch.clamp(swag_sq - swag_mean ** 2, min=1e-12).cpu(),
                           "dev": torch.stack(swag_dev).cpu()}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pde", default="ns2d_forced", choices=list(SPECS2D) + ["all"])
    ap.add_argument("--arch", default="fno2d", choices=["fno2d", "unet2d"])
    ap.add_argument("--grid", type=int, default=None, help="resolution override")
    ap.add_argument("--stride", type=int, default=None,
                    help="dataset output cadence; defaults to the spec")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ensemble", type=int, default=4,
                    help="members for the deep-ensemble baseline")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="rate for the extra MC-dropout model (0 to skip)")
    ap.add_argument("--swag-epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=_default_workers(),
                    help="dataloader workers; defaults to the CPUs actually "
                         "available to this process, which on a SLURM node is "
                         "the --cpus-per-task allocation and not the machine's "
                         "core count. Oversubscribing here makes the loader "
                         "slower, not faster, and can deadlock.")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    for pde in (list(SPECS2D) if args.pde == "all" else [args.pde]):
        print_header(f"Training {args.arch} on {pde}  (device={device})")
        for m in range(args.ensemble):
            train_one(pde, arch=args.arch, member=m, epochs=args.epochs,
                      batch=args.batch, lr=args.lr, width=args.width,
                      modes=args.modes, n_layers=args.layers, grid=args.grid,
                      stride=args.stride, workers=args.workers, device=device,
                      amp=not args.no_amp, resume=not args.no_resume,
                      swag_epochs=args.swag_epochs if m == 0 else 0)
        if args.dropout > 0:
            train_one(pde, arch=args.arch, member=0, epochs=args.epochs,
                      batch=args.batch, lr=args.lr, width=args.width,
                      modes=args.modes, n_layers=args.layers, grid=args.grid,
                      stride=args.stride, dropout=args.dropout,
                      workers=args.workers, device=device,
                      amp=not args.no_amp, resume=not args.no_resume)


if __name__ == "__main__":
    main()
