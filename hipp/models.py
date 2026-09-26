"""Surrogate "foundation model": a small 1D Fourier Neural Operator.

The model is the frozen deterministic predictor f_theta the whole method is
built around. It maps a physical state c_t -> c_{t+1} directly (normalization
lives inside the module) so that the Jacobian dJ/dc is expressed in physical
units and the physics residual can be applied to its output without rescaling.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes, 2, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, N)
        B, _, N = x.shape
        xh = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xh.shape[-1])
        w = torch.view_as_complex(self.weight[..., :m, :].contiguous())
        out = torch.zeros(B, self.out_ch, xh.shape[-1], dtype=xh.dtype, device=x.device)
        out[..., :m] = torch.einsum("bim,iom->bom", xh[..., :m], w)
        return torch.fft.irfft(out, n=N, dim=-1)


class FNO1d(nn.Module):
    """Residual FNO. Predicts the *increment*, which is standard practice for
    autoregressive PDE models and keeps the Jacobian close to identity."""

    def __init__(self, modes: int = 16, width: int = 32, n_layers: int = 4,
                 dropout: float = 0.0, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.dropout_p = dropout
        self.lift = nn.Linear(2, width)  # (u, x) -> width
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))

    def set_norm(self, mu: float, sigma: float) -> None:
        self.mu.fill_(mu)
        self.sigma.fill_(sigma)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, N) or (N,) physical state -> same shape."""
        squeeze = c.dim() == 1
        if squeeze:
            c = c.unsqueeze(0)
        B, N = c.shape
        u = (c - self.mu) / self.sigma
        grid = torch.linspace(0, 1, N, device=c.device, dtype=c.dtype).expand(B, N)
        h = self.lift(torch.stack([u, grid], dim=-1))          # (B, N, W)
        h = h.permute(0, 2, 1)                                  # (B, W, N)
        for sp, pw in zip(self.spectral, self.pointwise):
            h = torch.nn.functional.gelu(sp(h) + pw(h))
            h = self.drop(h)
        out = self.proj(h.permute(0, 2, 1)).squeeze(-1)         # (B, N)
        out = out * self.sigma
        if self.residual:
            out = out + c
        return out.squeeze(0) if squeeze else out

    def rollout(self, c0: torch.Tensor, steps: int) -> torch.Tensor:
        """(B, N) -> (B, steps + 1, N)."""
        seq = [c0]
        c = c0
        for _ in range(steps):
            c = self(c)
            seq.append(c)
        return torch.stack(seq, dim=1)


def enable_dropout(model: nn.Module) -> None:
    """Put *only* dropout layers into train mode (for MC-dropout sampling)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
