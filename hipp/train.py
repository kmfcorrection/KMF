"""Train the deterministic surrogate(s). Everything downstream treats the
resulting checkpoint as a frozen black box."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .data import SPECS, build_dataset, normalizer, pairs
from .models import FNO1d, count_params
from .utils import CKPT_DIR, get_device, print_header, set_seed


def ckpt_path(pde: str, member: int = 0, dropout: float = 0.0):
    tag = f"{pde}_m{member}" + (f"_do{dropout:g}" if dropout > 0 else "")
    return CKPT_DIR / f"{tag}.pt"


def train_one(pde: str, member: int = 0, epochs: int = 120, dropout: float = 0.0,
              lr: float = 2e-3, batch: int = 64, width: int = 32, modes: int = 16,
              n_layers: int = 4, seed: int | None = None, device=None,
              swag_epochs: int = 0, verbose: bool = True):
    """Train one model. If swag_epochs > 0, also collect SWAG statistics over
    the final epochs at constant LR (used by the SWAG baseline in Stage 2/3)."""
    device = device or get_device()
    set_seed(1234 + member if seed is None else seed)
    ds = build_dataset(pde)
    mu, sd = normalizer(ds["train"])

    xtr, ytr = pairs(ds["train"], device=device)
    xva, yva = pairs(ds["val"], device=device)

    model = FNO1d(modes=modes, width=width, n_layers=n_layers, dropout=dropout).to(device)
    model.set_norm(mu, sd)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - swag_epochs, 1))

    swag_mean = swag_sq = None
    swag_dev: list[torch.Tensor] = []
    n_swag = 0

    best = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(xtr.shape[0], device=device)
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            pred = model(xtr[idx])
            loss = torch.nn.functional.mse_loss(pred, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        in_swag = swag_epochs > 0 and ep >= epochs - swag_epochs
        if in_swag:
            for g in opt.param_groups:      # constant LR during SWAG collection
                g["lr"] = lr * 0.1
            flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
            if swag_mean is None:
                swag_mean, swag_sq = torch.zeros_like(flat), torch.zeros_like(flat)
            n_swag += 1
            swag_mean += (flat - swag_mean) / n_swag
            swag_sq += (flat ** 2 - swag_sq) / n_swag
            swag_dev.append((flat - swag_mean).clone())
            swag_dev[:] = swag_dev[-20:]
        else:
            sched.step()

        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.mse_loss(model(xva), yva).item()
        if vl < best and not in_swag:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"  [{pde} m{member}] ep {ep:3d}  train {tot/len(perm):.3e}  val {vl:.3e}")

    if best_state is not None and swag_epochs == 0:
        model.load_state_dict(best_state)

    payload = {
        "state_dict": model.state_dict(),
        "config": dict(modes=modes, width=width, n_layers=n_layers, dropout=dropout),
        "norm": (mu, sd), "pde": pde, "val_mse": best, "seconds": time.time() - t0,
    }
    if swag_mean is not None:
        payload["swag"] = {
            "mean": swag_mean.cpu(),
            "var": torch.clamp(swag_sq - swag_mean ** 2, min=1e-12).cpu(),
            "dev": torch.stack(swag_dev).cpu(),
        }
    path = ckpt_path(pde, member, dropout)
    torch.save(payload, path)
    if verbose:
        print(f"  saved {path.name}  best val MSE {best:.3e}  ({count_params(model)} params)")
    return model, payload


def load_model(pde: str, member: int = 0, dropout: float = 0.0, device=None,
               with_payload: bool = False):
    device = device or get_device()
    path = ckpt_path(pde, member, dropout)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:  python -m hipp.train --pde {pde}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FNO1d(**payload["config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return (model, payload) if with_payload else model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="burgers", choices=list(SPECS) + ["all"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ensemble", type=int, default=5, help="number of ensemble members")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="dropout rate for the extra MC-dropout model")
    ap.add_argument("--swag-epochs", type=int, default=20)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pdes = list(SPECS) if args.pde == "all" else [args.pde]
    dev = get_device(args.device)
    for pde in pdes:
        print_header(f"Training surrogates for {pde}  (device={dev})")
        for m in range(args.ensemble):
            # member 0 additionally collects SWAG statistics
            train_one(pde, member=m, epochs=args.epochs, device=dev,
                      swag_epochs=args.swag_epochs if m == 0 else 0)
        if args.dropout > 0:
            train_one(pde, member=0, epochs=args.epochs, dropout=args.dropout, device=dev)


if __name__ == "__main__":
    main()
