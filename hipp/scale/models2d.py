"""Surrogates that scale: 2D FNO and U-Net.

These stand in for "the frozen deterministic foundation model" when a pretrained
checkpoint is not being used. Two architectures rather than one, deliberately:
the method's claim is about the local geometry of a learned operator, and a
result that only holds for FNO would be a result about spectral convolutions.
The U-Net has a genuinely different Jacobian structure (local receptive field
built up through scales, rather than a global spectral multiplier), so it is the
control that says whether the geometry finding is architectural.

Both keep two properties the 1D `FNO1d` had, because the whole method depends on
them:

  * normalization lives *inside* the module, so J = df/dc is in physical units
    and the physics residual applies to the output without rescaling;
  * the model predicts the increment (residual connection), so J = I + dNet/dc.
    That keeps cond(J) small, which is not a detail -- it sets the oracle
    ceiling on Stage 1's rank correlation (docs/method.md 2.4).

Sizing. `--width 64 --modes 32 --layers 4` gives ~34M parameters, which is the
range where "foundation model" is a fair description and where the O(N) exact
Jacobian of the 1D code is definitively out of reach.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FNO
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """Truncated spectral convolution. Keeps modes1 x modes2 low frequencies.

    Both corners of the rfft2 output are kept (positive and negative kx), which
    is required for the operator to be real-symmetric; dropping the second block
    is a common bug that shows up as a spurious phase drift under rollout.

    Runs in float32 with autocast explicitly disabled. cuFFT has no bfloat16 (or
    float16) kernel at all -- `torch.fft.rfft2` on an autocast-produced bf16
    tensor raises "Unsupported dtype BFloat16" -- so the choice is not between
    precisions but between fp32 here and no AMP anywhere. Confining the override
    to this layer keeps the pointwise convolutions and the projection MLP in
    bf16, which is where most of the parameters and nearly all of the matmul
    time live. The cast is also free in accuracy terms: this is the layer whose
    weights span the widest dynamic range, and it is the one worth keeping exact.
    """

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        shape = (in_ch, out_ch, modes1, modes2, 2)
        self.w1 = nn.Parameter(scale * torch.rand(*shape))
        self.w2 = nn.Parameter(scale * torch.rand(*shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, C, H, W)
        with torch.autocast(device_type=x.device.type, enabled=False):
            # float64 is preserved (the s0 validation path runs the model in
            # double); only a reduced-precision autocast dtype is promoted.
            if x.dtype not in (torch.float32, torch.float64):
                x = x.float()
            B, _, H, W = x.shape
            xh = torch.fft.rfft2(x, norm="ortho")
            m1 = min(self.modes1, H // 2)
            m2 = min(self.modes2, W // 2 + 1)
            out = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=xh.dtype,
                              device=x.device)
            w1 = torch.view_as_complex(self.w1[:, :, :m1, :m2].to(x.dtype).contiguous())
            w2 = torch.view_as_complex(self.w2[:, :, :m1, :m2].to(x.dtype).contiguous())
            out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xh[:, :, :m1, :m2], w1)
            out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xh[:, :, -m1:, :m2], w2)
            return torch.fft.irfft2(out, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    """Residual 2D FNO mapping a physical state to the next physical state."""

    def __init__(self, in_channels: int = 1, out_channels: int | None = None,
                 modes: int = 32, width: int = 64, n_layers: int = 4,
                 dropout: float = 0.0, residual: bool = True,
                 use_grid: bool = True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels, self.out_channels = in_channels, out_channels
        self.residual, self.use_grid = residual, use_grid
        self.dropout_p = dropout
        lift_in = in_channels + (2 if use_grid else 0)
        self.lift = nn.Conv2d(lift_in, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(min(8, width), width) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Sequential(nn.Conv2d(width, 2 * width, 1), nn.GELU(),
                                  nn.Conv2d(2 * width, out_channels, 1))
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))

    def set_norm(self, mu: float, sigma: float) -> None:
        self.mu.fill_(float(mu))
        self.sigma.fill_(float(sigma))

    def _grid(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        gy = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        gx = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
        return torch.cat([gy.expand(B, 1, H, W), gx.expand(B, 1, H, W)], dim=1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) or (C, H, W) or (H, W) physical state -> same shape."""
        shape_in = c.shape
        c = _as_bchw(c, self.in_channels)
        u = (c - self.mu) / self.sigma
        h = self.lift(torch.cat([u, self._grid(u)], dim=1) if self.use_grid else u)
        for sp, pw, nrm in zip(self.spectral, self.pointwise, self.norms):
            h = h + self.drop(F.gelu(nrm(sp(h) + pw(h))))
        out = self.proj(h) * self.sigma
        if self.residual:
            out = out + c
        return out.reshape(shape_in)

    def rollout(self, c0: torch.Tensor, steps: int) -> torch.Tensor:
        seq, c = [c0], c0
        for _ in range(steps):
            c = self(c)
            seq.append(c)
        return torch.stack(seq, dim=1)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular")
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.drop(h)
        h = self.n2(self.c2(h))
        return F.gelu(h + self.skip(x))


class UNet2d(nn.Module):
    """Residual U-Net with circular padding (the domain is periodic).

    Present as an architecture control, not because it is expected to win.
    """

    def __init__(self, in_channels: int = 1, out_channels: int | None = None,
                 width: int = 64, depth: int = 4, dropout: float = 0.0,
                 residual: bool = True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels, self.out_channels = in_channels, out_channels
        self.residual, self.depth = residual, depth
        self.dropout_p = dropout
        chans = [width * min(2 ** i, 8) for i in range(depth + 1)]
        self.stem = _Block(in_channels, chans[0], dropout)
        self.down = nn.ModuleList([_Block(chans[i], chans[i + 1], dropout)
                                   for i in range(depth)])
        self.up = nn.ModuleList([_Block(chans[i + 1] + chans[i], chans[i], dropout)
                                 for i in reversed(range(depth))])
        self.head = nn.Conv2d(chans[0], out_channels, 1)
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))

    def set_norm(self, mu: float, sigma: float) -> None:
        self.mu.fill_(float(mu))
        self.sigma.fill_(float(sigma))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        shape_in = c.shape
        c = _as_bchw(c, self.in_channels)
        h = self.stem((c - self.mu) / self.sigma)
        skips = []
        for blk in self.down:
            skips.append(h)
            h = blk(F.avg_pool2d(h, 2))
        for blk, s in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, s], dim=1))
        out = self.head(h) * self.sigma
        if self.residual:
            out = out + c
        return out.reshape(shape_in)

    def rollout(self, c0: torch.Tensor, steps: int) -> torch.Tensor:
        seq, c = [c0], c0
        for _ in range(steps):
            c = self(c)
            seq.append(c)
        return torch.stack(seq, dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_bchw(c: torch.Tensor, channels: int) -> torch.Tensor:
    """Accept (H,W), (C,H,W)/(B,H,W) or (B,C,H,W); return (B,C,H,W).

    Flat (N,) input is deliberately *not* accepted here. `torch.func.jvp` hands
    the model a flat vector, and the reshape has to happen inside the
    differentiated function -- but the state shape is the adapter's knowledge,
    not the module's, so `adapters.FrozenFM.flat_fn` owns that reshape.
    """
    if c.dim() == 4:
        return c
    if c.dim() == 3:
        return c.unsqueeze(0) if c.shape[0] == channels else c.unsqueeze(1)
    if c.dim() == 2:
        return c.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"cannot interpret a {c.dim()}-d tensor as a 2D state; "
                     "flatten/reshape in the adapter, not here")


def build_model(arch: str, **kw) -> nn.Module:
    if arch == "fno2d":
        return FNO2d(**kw)
    if arch == "unet2d":
        kw.pop("modes", None)
        kw.pop("use_grid", None)
        kw["depth"] = kw.pop("n_layers", 4)
        return UNet2d(**kw)
    raise ValueError(f"unknown arch {arch!r}; choices: fno2d, unet2d")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def enable_dropout(model: nn.Module) -> None:
    """Put only dropout layers into train mode, for MC-dropout sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
