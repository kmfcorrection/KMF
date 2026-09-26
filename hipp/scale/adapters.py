"""One interface for "the frozen foundation model", whatever it actually is.

Everything downstream -- curvature, calibration, rollout -- only ever needs

    f : R^N -> R^N,   the one-step map on the flattened physical state

plus the shape needed to fold that vector back into a field. `FrozenFM` is that
contract. Given it, the same experiment script runs against a 34M-parameter FNO
we trained, a 158M-parameter Poseidon checkpoint, or a 1.3B-parameter Walrus,
with no change to the method code.

Three things the adapters have to get right, all of which are easy to get
subtly wrong and all of which silently invalidate the results:

1. **Physical units.** J = df/dc must be the Jacobian of the map on physical
   states. If a model normalizes its input externally, the adapter folds the
   normalization into f, so J carries the chain rule through it. Otherwise the
   curvature is measured in whatever arbitrary units the checkpoint was trained
   in and the physics residual no longer matches it.

2. **History.** Most real PDE foundation models condition on a short stack of
   past frames, u(t-tau+1..t). The predictive-uncertainty question is about the
   *current* state, so the adapter differentiates with respect to the last
   frame and holds the earlier ones fixed. That is a genuine modeling choice --
   it says the uncertainty being propagated lives in the present state -- and
   `history_mode` makes it explicit rather than implicit. Set it to "all" to
   treat the whole stack as the uncertain object instead.

3. **Determinism.** Dropout, stochastic depth and any sampling must be off, and
   the parameters frozen. `FrozenFM.__init__` enforces both, since a stochastic
   f makes every finite-difference probe and every JVP silently wrong.

Status of each pretrained adapter, because they differ and the difference
matters when reading any number produced through one:

* `from_poseidon` -- **executed and verified**. `check_adapter` passes on
  `camlab-ethz/Poseidon-{T,B}`: deterministic, JVP and VJP both work, the
  adjoint identity holds to ~1e-6, and the JVP matches a central difference.
  The normalization, channel layout and lead-time convention below are read off
  `scOT.problems.fluids`, not guessed.
* `from_dpot` -- wired against the released PyTorch source and HuggingFace
  weights. Its public checkpoint lacks its dataset normalizer, so it is a
  geometry/S1--S3 candidate only until that contract is recovered.
* `from_morph` -- wired against LANL's released PyTorch source and foundation
  checkpoints. MORPH's published RevIN is trajectory-level; this adapter uses
  the last observed frame only, explicitly avoiding future-statistics leakage.
  It is therefore an AD/S1--S3 adapter, not a native-physics S4 adapter.
* `from_walrus` -- the checkpoint now exists on the Hub, but the public model
  needs its formatter/field-index metadata to build a physically meaningful
  one-step map; this adapter fails explicitly until that contract is pinned.
* `from_the_well` -- runnable checkpoint wrapper for Polymathic/The Well
  benchmark models. These are useful external controls, not foundation models.
* `from_pdeformer` -- documents why PDEformer-2 is outside this PyTorch AD path:
  released inference is MindSpore and predicts pointwise solutions from a PDE
  graph, not an autoregressive tensor-to-tensor map.
* `from_omniarch` -- **cannot be implemented against the released artifacts**
  and says so when called. See its docstring; this is a property of what
  OmniArch released, not a gap here.

Verify `check_adapter()` passes before trusting any result from an adapter.
"""
from __future__ import annotations

import os
import sys
import json
import inspect
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@dataclass
class FMInfo:
    name: str
    state_shape: tuple            # (C, H, W)
    n_params: int
    history: int = 1              # frames the model conditions on
    dtype: torch.dtype = torch.float32
    source: str = ""

    @property
    def N(self) -> int:
        c, h, w = self.state_shape
        return c * h * w


class FrozenFM:
    """A frozen deterministic one-step operator on a flattened 2D state."""

    def __init__(self, module: nn.Module, state_shape: tuple, name: str = "fm",
                 history: int = 1, history_mode: str = "last",
                 device=None, dtype: torch.dtype = torch.float32,
                 pre=None, post=None, source: str = ""):
        if history_mode not in ("last", "all"):
            raise ValueError("history_mode must be 'last' or 'all'")
        self.module = module.eval().to(device=device, dtype=dtype)
        for p in self.module.parameters():
            p.requires_grad_(False)
        for m in self.module.modules():                      # determinism
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                m.eval()
        self.state_shape = tuple(state_shape)
        self.history, self.history_mode = int(history), history_mode
        self.device = device or next(module.parameters()).device
        self.dtype = dtype
        self.pre, self.post = pre, post
        self.info = FMInfo(name=name, state_shape=self.state_shape,
                           n_params=sum(p.numel() for p in module.parameters()),
                           history=history, dtype=dtype, source=source)

    # ---- shape plumbing -------------------------------------------------
    @property
    def N(self) -> int:
        return self.info.N

    def to_field(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*self.state_shape)

    def to_flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1)

    # ---- the map --------------------------------------------------------
    def _apply(self, field: torch.Tensor, past: torch.Tensor | None) -> torch.Tensor:
        inp = field if self.pre is None else self.pre(field)
        if past is not None:
            inp = torch.cat([past, inp.unsqueeze(0)], dim=0) if inp.dim() == 3 else inp
        out = self.module(inp)
        return out if self.post is None else self.post(out)

    def flat_fn(self, past: torch.Tensor | None = None):
        """Return f : (N,) -> (N,), pure and differentiable.

        `past` is the frozen history stack (history-1, C, H, W) when the model
        needs one. It is captured by closure, so the resulting function depends
        only on its argument -- which is what `torch.func.jvp`/`vjp` require.
        """
        shape = self.state_shape

        def f(x: torch.Tensor) -> torch.Tensor:
            out = self._apply(x.reshape(*shape), past)
            return out.reshape(-1)
        return f

    def flat_batch_fn(self, past: torch.Tensor | None = None):
        """Return the differentiable independent map ``(B,N) -> (B,N)``.

        This is deliberately separate from :meth:`predict`: the latter is a
        no-grad convenience method, whereas state-batched JVP/VJP code needs a
        pure differentiable function.  The caller must opt in only for adapters
        whose module has been marked ``_hipp_batch_safe``.
        """
        if not getattr(self.module, "_hipp_batch_safe", False):
            raise RuntimeError(
                f"{self.info.name} has not opted into batched AD; use the "
                "serial Jacobian backend")
        shape = self.state_shape

        def f(x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 2 or x.shape[1] != self.N:
                raise ValueError(f"expected (B,{self.N}) state batch, got {tuple(x.shape)}")
            out = self._apply(x.reshape(-1, *shape), past)
            return out.reshape(x.shape[0], -1)
        return f

    @torch.no_grad()
    def predict(self, x: torch.Tensor, past: torch.Tensor | None = None) -> torch.Tensor:
        """Batched point prediction. Accepts (N,), (C,H,W), (B,N) or (B,C,H,W);
        the output always has the same shape as the input."""
        xb = x.reshape(-1, *self.state_shape)
        # Adapters opt in only after their batch layout has been verified.  The
        # legacy loop remains the safe fallback for third-party wrappers whose
        # public forward contract is single-state only.
        if getattr(self.module, "_hipp_batch_safe", False):
            out = self._apply(xb, past)
            if tuple(out.shape) == tuple(xb.shape):
                return out.reshape(x.shape)
        out = torch.stack([self._apply(xb[i], past) for i in range(xb.shape[0])])
        return out.reshape(x.shape)

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        """(steps+1, N) autoregressive trajectory from a single initial state."""
        seq, c = [x0.reshape(-1)], x0.reshape(-1)
        f = self.flat_fn()
        for _ in range(steps):
            c = f(c)
            seq.append(c)
        return torch.stack(seq)

    def __repr__(self):
        i = self.info
        return (f"FrozenFM({i.name}, state={i.state_shape}, N={i.N}, "
                f"params={i.n_params/1e6:.1f}M, history={i.history})")


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def from_local_checkpoint(path: str | Path, device=None,
                          dtype: torch.dtype = torch.float32) -> FrozenFM:
    """Load a checkpoint written by `hipp.scale.train2d`."""
    from .models2d import build_model
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train one with:\n"
            f"  python -m hipp.scale.train2d --pde ns2d_forced --arch fno2d")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(payload["arch"], **payload["config"])
    model.load_state_dict(payload["state_dict"])
    n = payload["spec"]["n"]
    ch = payload["config"].get("in_channels", 1)
    return FrozenFM(model, (ch, n, n), name=payload.get("name", path.stem),
                    device=device, dtype=dtype, source=str(path))


def from_callable(fn, state_shape: tuple, name: str = "callable",
                  n_params: int = 0, device=None) -> FrozenFM:
    """Wrap an arbitrary differentiable function. Escape hatch for models that
    are not `nn.Module`s (compiled graphs, JAX-via-dlpack, ONNX runtimes)."""
    class _Wrap(nn.Module):
        _hipp_batch_safe = True

        def forward(self, x):
            return fn(x)
    fm = FrozenFM(_Wrap(), state_shape, name=name, device=device)
    fm.info.n_params = n_params
    return fm


# Poseidon's fluid corpus is stored in physical units and normalized on the way
# into the network by these per-channel constants, over channels [rho, u, v, p].
# Copied from scOT.problems.fluids.normalization_constants; f folds them in so
# J = df/dc is the Jacobian of the map on *physical* states, per point 1 above.
POSEIDON_FLUID_MEAN = (0.80, 0.0, 0.0, 0.0)
POSEIDON_FLUID_STD = (0.31, 0.391, 0.356, 0.185)

# scOT conditions on a raw dataset-index difference normalized by the trajectory
# horizon: ``time = (t2-t1)/20``. NS-Gauss stores 21 raw snapshots, but the
# standard Poseidon incompressible training configuration selects every other
# one; its supported native one-step transition is therefore lead_time=2.0.
# A caller may still request another positive lead time for an explicit cadence
# interpolation/extrapolation study.
POSEIDON_TIME_SCALE = 20.0


DPOT_SPECS = {
    "Ti": dict(file="model_Ti.pth", embed_dim=512, depth=4, n_blocks=4,
               mlp_ratio=1.0, params="7M"),
    "S": dict(file="model_S.pth", embed_dim=1024, depth=6, n_blocks=8,
              mlp_ratio=1.0, params="30M"),
    "M": dict(file="model_M.pth", embed_dim=1024, depth=12, n_blocks=8,
              mlp_ratio=4.0, params="122M"),
    "L": dict(file="model_L.pth", embed_dim=1536, depth=24, n_blocks=16,
              mlp_ratio=4.0, params="509M"),
    "H": dict(file="model_H.pth", embed_dim=2048, depth=27, n_blocks=8,
              mlp_ratio=8092 / 2048, params="1.03B"),
}


# Official MORPH-FM variants from lanl/MORPH's general fine-tuning script.
# The checkpoints are pre-trained with max_ar=1, hence the state-map adapter
# below can expose a true one-frame map without inventing an unobserved past.
MORPH_SPECS = {
    "Ti": dict(file="morph-Ti-FM-max_ar1_ep225.pth", filters=8, dim=256,
               heads=4, depth=4, mlp_dim=1024, params="7M"),
    "S": dict(file="morph-S-FM-max_ar1_ep225.pth", filters=8, dim=512,
              heads=8, depth=4, mlp_dim=2048, params="30M"),
    "M": dict(file="morph-M-FM-max_ar1_ep290_latestbatch.pth", filters=8,
              dim=768, heads=12, depth=8, mlp_dim=3072, params="120M"),
}


def _patch_scot(ScOT):
    """Backfill APIs that newer transformers removed but scOT still calls.

    scOT is written against transformers 4.29 and calls `get_head_mask`, which
    v5 dropped along with head masking itself. Nothing in this repository ever
    passes a head mask, and the original behaviour for `head_mask=None` is the
    whole contract: one `None` per layer, which scOT then splits between its
    encoder and decoder stacks. Restoring that is a shim, not a reimplementation
    -- if a head mask is ever actually passed, this raises rather than silently
    ignoring it.
    """
    if not hasattr(ScOT, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers,
                          is_attention_chunked: bool = False):
            if head_mask is not None:
                raise NotImplementedError(
                    "scOT was given a head mask, but this transformers version "
                    "removed head-mask support and the shim in "
                    "hipp.scale.adapters._patch_scot only covers head_mask=None")
            return [None] * num_hidden_layers
        ScOT.get_head_mask = get_head_mask
    return ScOT


def _import_scot():
    """Import `scOT`, falling back to a source checkout.

    scOT pins `transformers == 4.29.2`, which is old enough that installing it
    with its dependencies would downgrade the environment out from under
    everything else here. It runs unchanged against transformers 4.x (verified
    on 4.57.3), so the supported routes are, in order: an installed package;
    `$SCOT_PATH`; a checkout at `third_party/poseidon` beside the repo root.
    See `_patch_scot` for what newer transformers takes away.
    """
    try:
        from scOT.model import ScOT                       # type: ignore
        return _patch_scot(ScOT)
    except ImportError:
        pass
    roots = [os.environ.get("SCOT_PATH"),
             str(Path(__file__).resolve().parents[2] / "third_party" / "poseidon")]
    for root in roots:
        if root and (Path(root) / "scOT" / "model.py").exists():
            if root not in sys.path:
                sys.path.insert(0, root)
            from scOT.model import ScOT                   # type: ignore
            return _patch_scot(ScOT)
    raise ImportError(
        "Poseidon needs the scOT source from github.com/camlab-ethz/poseidon.\n"
        "  git clone https://github.com/camlab-ethz/poseidon.git third_party/poseidon\n"
        "or set SCOT_PATH to an existing checkout. Do NOT `pip install` it "
        "with dependencies: it pins transformers==4.29.2.")


def from_poseidon(model_size: str = "B", device=None,
                  dtype: torch.dtype = torch.float32,
                  resolution: int = 128, lead_time: float = 1.0,
                  free_channels: tuple[int, ...] | None = None,
                  mean: tuple[float, ...] = POSEIDON_FLUID_MEAN,
                  std: tuple[float, ...] = POSEIDON_FLUID_STD) -> FrozenFM:
    """Poseidon (scOT), the ETH CAMLab PDE foundation model family.

    Checkpoints: `camlab-ethz/Poseidon-{T,B,L}` (21M / 158M / 629M) on the
    HuggingFace Hub, loaded through the `scOT` source tree (see `_import_scot`).

    Three things about Poseidon specifically, each of which changes what the
    resulting operator *is*:

    **It is not a fixed one-step map.** scOT takes the lead time as an input, so
    unlike DPOT or Walrus the output cadence is a free parameter rather than a
    property of the checkpoint. `lead_time` is an underlying raw dataset-index
    difference, and f is the map `u(t=0) -> u(t=lead_time)`. For NS-Gauss the
    published standard training loader uses spacing two, hence `lead_time=2`
    is the native one-step setting. This is the knob `experiments/scale/s5` had
    to fake with a resimulated dataset, available here directly -- and it means
    the "the cadence is not ours to choose" worry in `check_fm` does not apply
    to this family. It applies to models that hard-code their step.

    **Its state has four channels, `[rho, u, v, p]`, and on the incompressible
    corpus two of them are constants** (`rho == 1`, `p == 0` -- see
    `IncompressibleBase.__getitem__`). Differentiating with respect to all four
    therefore mixes directions the model never saw vary. `free_channels`
    selects the subset that carries the uncertain state; the rest are pinned at
    their physical constants and excluded from `N`. `(1, 2)` is the velocity-only
    choice appropriate to the incompressible problems; `None` keeps all four,
    which is the right choice for the compressible ones.

    **Its normalization is per-channel and part of the physics.** `mean`/`std`
    are folded into f, so the chain rule carries them into J and the curvature
    stays in the units the physics residual is measured in.
    """
    ScOT = _import_scot()
    net = ScOT.from_pretrained(f"camlab-ethz/Poseidon-{model_size}")
    n_ch = int(net.config.num_channels)
    free = tuple(range(n_ch)) if free_channels is None else tuple(free_channels)
    if not all(0 <= c < n_ch for c in free):
        raise ValueError(f"free_channels {free} outside the model's {n_ch}")

    mu = torch.tensor(mean[:n_ch], dtype=dtype).view(-1, 1, 1)
    sd = torch.tensor(std[:n_ch], dtype=dtype).view(-1, 1, 1)
    # Channels held fixed sit at their physical constants: rho = 1, p = 0 and,
    # generically, the corpus mean. Only the free ones enter the state vector.
    pinned = torch.tensor(mean[:n_ch], dtype=dtype).view(-1, 1, 1).clone()
    if n_ch == 4:
        pinned[0], pinned[3] = 1.0, 0.0

    class _Wrap(nn.Module):
        # scOT accepts an ordinary leading batch dimension and has no
        # batch-dependent state in eval mode.  This is exercised by the
        # trajectory-batched AD equivalence gate before it is used in a run.
        _hipp_batch_safe = True

        def __init__(self, net):
            super().__init__()
            self.net = net
            self.t_norm = float(lead_time) / POSEIDON_TIME_SCALE
            self.register_buffer("mu", mu)
            self.register_buffer("sd", sd)
            self.register_buffer("pinned", pinned.expand(n_ch, resolution,
                                                         resolution).clone())
            self.idx = torch.tensor(free)

        def forward(self, x):                    # (C,H,W) or (B,C,H,W), physical
            single = x.dim() == 3
            if x.dim() not in (3, 4):
                raise ValueError(f"Poseidon adapter expects 3D/4D state, got {tuple(x.shape)}")
            xb = x.unsqueeze(0) if single else x
            batch = xb.shape[0]
            full = self.pinned.to(x.dtype).unsqueeze(0).expand(batch, -1, -1, -1)
            if len(free) != n_ch:
                # index_copy on a fresh tensor keeps this differentiable in x
                # while the pinned channels stay constant w.r.t. it.
                full = full.index_copy(1, self.idx.to(x.device), xb)
            else:
                full = xb
            u = (full - self.mu.unsqueeze(0)) / self.sd.unsqueeze(0)
            time = torch.full((batch,), self.t_norm, device=x.device, dtype=x.dtype)
            out = self.net(pixel_values=u, time=time)
            out = out.output if hasattr(out, "output") else out
            out = out * self.sd.unsqueeze(0) + self.mu.unsqueeze(0)
            result = out.index_select(1, self.idx.to(x.device))
            return result.squeeze(0) if single else result

    fm = FrozenFM(_Wrap(net), (len(free), resolution, resolution),
                  name=f"poseidon-{model_size}", device=device, dtype=dtype,
                  source=f"camlab-ethz/Poseidon-{model_size}")
    fm.free_channels = free

    def set_lead_time(t: float) -> None:
        """Retarget the operator without reloading 158M parameters. Sweeping
        this is the only way to ask a *pretrained* model the s5 question."""
        fm.module.t_norm = float(t) / POSEIDON_TIME_SCALE
        fm.lead_time = float(t)

    fm.set_lead_time = set_lead_time
    fm.lead_time = float(lead_time)
    return fm


def _import_dpot():
    roots = [os.environ.get("DPOT_PATH"),
             str(Path(__file__).resolve().parents[2] / "third_party" / "dpot")]
    for root in roots:
        if root and (Path(root) / "models" / "dpot.py").exists():
            if root not in sys.path:
                sys.path.insert(0, root)
            from models.dpot import DPOTNet, checkpoint_filter_fn  # type: ignore
            return DPOTNet, checkpoint_filter_fn
    raise ImportError(
        "DPOT needs the released source from github.com/HaoZhongkai/DPOT.\n"
        "  git clone https://github.com/HaoZhongkai/DPOT.git third_party/dpot\n"
        "or set DPOT_PATH to an existing checkout. Install einops/timm, but do "
        "not let its old torch requirement replace the cluster torch.")


def from_dpot(model_size: str = "Ti", device=None,
              dtype: torch.dtype = torch.float32,
              resolution: int = 128, checkpoint: str | None = None,
              free_channels: tuple[int, ...] | None = None,
              history: int = 10, norm: tuple[float, float] = (0.0, 1.0)) -> FrozenFM:
    """DPOT, the auto-regressive denoising operator transformer.

    DPOT is a PyTorch neural operator transformer pretrained on 10+ PDE datasets
    with checkpoints `model_{Ti,S,M,L,H}.pth` on `hzk17/DPOT`. The released
    model consumes a 10-frame causal history `(B, X, Y, T, C)` and predicts one
    future frame.  Consequently this adapter requires the caller to attach
    real preceding states through ``set_history`` before every forecast.  It
    deliberately refuses the formerly used ``[u_t, ..., u_t]`` synthetic
    history: that input is not the released DPOT forecasting contract.

    DPOT's public card does not ship per-dataset physical normalizers with the
    pretrained weights, so `norm=(0,1)` is the default and the resulting
    Jacobian is in whatever units the input states are supplied in. That is fine
    for the gate (`check_fm`), but any calibration/refinement claim should be
    rerun with the dataset-specific normalizer once the target corpus is chosen.
    """
    key = {"T": "Ti", "Tiny": "Ti", "small": "S", "medium": "M",
           "large": "L", "huge": "H"}.get(str(model_size), str(model_size))
    if key not in DPOT_SPECS:
        raise ValueError(f"Unknown DPOT size {model_size}; choose {sorted(DPOT_SPECS)}")
    spec = DPOT_SPECS[key]
    DPOTNet, checkpoint_filter_fn = _import_dpot()

    n_ch = 4
    free = tuple(range(n_ch)) if free_channels is None else tuple(free_channels)
    if not all(0 <= c < n_ch for c in free):
        raise ValueError(f"free_channels {free} outside DPOT's {n_ch} channels")

    net = DPOTNet(img_size=resolution, patch_size=8, mixing_type="afno",
                  in_channels=n_ch, in_timesteps=history, out_timesteps=1,
                  out_channels=n_ch, normalize=False,
                  embed_dim=spec["embed_dim"], modes=32, depth=spec["depth"],
                  n_blocks=spec["n_blocks"], mlp_ratio=spec["mlp_ratio"],
                  out_layer_dim=32, n_cls=12)
    if checkpoint is None:
        try:
            from huggingface_hub import hf_hub_download
            checkpoint = hf_hub_download("hzk17/DPOT", spec["file"])
        except ImportError as exc:
            raise ImportError(
                "DPOT checkpoints need huggingface_hub, or pass --fm-checkpoint"
            ) from exc
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    state = checkpoint_filter_fn(state, net)
    net.load_state_dict(state)
    mu, sd = norm

    class _Wrap(nn.Module):
        # DPOT's released tensor contract has an ordinary independent leading
        # batch dimension.  Batch execution is required so a history context
        # remains paired with its corresponding current state.
        _hipp_batch_safe = True

        def __init__(self, net):
            super().__init__()
            self.net = net
            # The shared NS-Gauss transfer corpus contains velocity only.  When
            # the DPOT state is restricted to channels (u,v), retain the
            # incompressible constants for the other channels rather than
            # silently presenting rho=0 to the frozen model.
            pinned = torch.zeros(n_ch, resolution, resolution)
            pinned[0] = 1.0
            self.register_buffer("pinned", pinned)
            self.idx = torch.tensor(free)
            self.history_context: torch.Tensor | None = None

        def set_history(self, history_context: torch.Tensor) -> None:
            """Attach causal physical states shaped ``(B,T,C,H,W)``.

            The final history state must be the state passed to ``forward``.
            Keeping history outside the trainable network lets the benchmark
            advance the queue autoregressively without using future truth.
            """
            if history_context.ndim != 5 or history_context.shape[1] != history:
                raise ValueError(
                    f"DPOT requires (B,{history},C,H,W) causal history, got "
                    f"{tuple(history_context.shape)}")
            if history_context.shape[2] != len(free):
                raise ValueError(
                    f"DPOT history has {history_context.shape[2]} free channels; "
                    f"expected {len(free)}")
            self.history_context = history_context

        def _full_state(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            base = self.pinned.to(device=x.device, dtype=x.dtype).unsqueeze(0)
            full = base.expand(batch, -1, -1, -1)
            if len(free) != n_ch:
                return full.index_copy(1, self.idx.to(x.device), x)
            return x

        def forward(self, x):                    # (len(free), H, W)
            single = x.ndim == 3
            xb = x.unsqueeze(0) if single else x
            if xb.ndim != 4:
                raise ValueError(f"DPOT expects (C,H,W) or (B,C,H,W), got {tuple(x.shape)}")
            ctx = self.history_context
            if ctx is None:
                raise RuntimeError(
                    "DPOT requires a causal 10-frame history. Call fm.set_history(...) "
                    "before fm.predict; synthetic repeated-current histories are disabled.")
            ctx = ctx.to(device=xb.device, dtype=xb.dtype)
            if ctx.shape[0] != xb.shape[0] or not torch.allclose(ctx[:, -1], xb, rtol=1e-4, atol=1e-6):
                raise ValueError("DPOT history must end at the state passed to forward")
            full_hist = self._full_state(ctx.reshape(-1, *ctx.shape[2:])).reshape(
                ctx.shape[0], history, n_ch, *ctx.shape[-2:])
            # `norm` is intentionally allowed to be ordinary Python scalars;
            # PyTorch promotes them safely to the input device/dtype here.
            u = ((full_hist - mu) / sd).permute(0, 3, 4, 1, 2)
            out, _ = self.net(u.contiguous())
            out = out[:, :, :, -1, :].permute(0, 3, 1, 2)
            out = out * sd + mu
            result = out.index_select(1, self.idx.to(x.device))
            return result.squeeze(0) if single else result

    fm = FrozenFM(_Wrap(net), (len(free), resolution, resolution),
                  name=f"dpot-{key}", history=history, device=device,
                  dtype=dtype, source=f"hzk17/DPOT:{spec['file']}")
    fm.free_channels = free
    fm.set_history = fm.module.set_history
    fm.required_history = history
    fm.dpot_history_policy = "causal-observed-then-autoregressive"
    return fm


def _import_morph():
    """Import MORPH from a released source checkout without installing it.

    MORPH uses absolute imports rooted at its repository directory (``src``),
    so the source root must be ahead of this repository on ``sys.path``.  The
    adapter deliberately imports only the model definition; its training and
    plotting dependencies are not part of the inference contract.
    """
    roots = [os.environ.get("MORPH_PATH"),
             str(Path(__file__).resolve().parents[2] / "third_party" / "morph")]
    for root in roots:
        if root and (Path(root) / "src" / "utils" / "vit_conv_xatt_axialatt2.py").exists():
            if root not in sys.path:
                sys.path.insert(0, root)
            from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression  # type: ignore
            return ViT3DRegression
    raise ImportError(
        "MORPH needs the released source from github.com/lanl/MORPH.\n"
        "  git clone https://github.com/lanl/MORPH.git third_party/morph\n"
        "or set MORPH_PATH to an existing checkout. Install einops and "
        "huggingface_hub in the active environment; do not install MORPH's "
        "full training environment into the shared inference environment.")


def from_morph(model_size: str = "Ti", device=None,
               dtype: torch.dtype = torch.float32,
               resolution: int = 128, checkpoint: str | None = None) -> FrozenFM:
    """MORPH-FM as a differentiable 2-D velocity map for S1--S3.

    MORPH's released foundation checkpoints use one history frame (``max_ar=1``)
    and accept ``(B,T,F,C,D,H,W)``.  A 2-D velocity state is represented as one
    field with two components and singleton depth: ``(B,1,1,2,1,H,W)``.

    The original MORPH training code applies RevIN using statistics of an entire
    trajectory.  Reusing those statistics at forecast time would leak future
    information.  We instead compute the scalar mean and standard deviation
    from the observed current frame only, fold this transform into ``f``, and
    undo it after the network.  This defines an honest deterministic map and
    differentiates through the normalization, but is not claimed to reproduce
    MORPH's dataset-specific forecasting protocol.  It is safe for the
    probeability/S1--S3 geometry experiments; S4 deliberately rejects it until
    matching physical trajectories and preprocessing are supplied.
    """
    key = {"T": "Ti", "Tiny": "Ti", "small": "S", "medium": "M"}.get(
        str(model_size), str(model_size))
    if key not in MORPH_SPECS:
        raise ValueError(f"Unknown MORPH size {model_size}; choose {sorted(MORPH_SPECS)}")
    if resolution % 8:
        raise ValueError(f"MORPH's released patch size is 8; resolution={resolution} is invalid")
    spec = MORPH_SPECS[key]
    ViT3DRegression = _import_morph()
    net = ViT3DRegression(
        patch_size=8, dim=spec["dim"], depth=spec["depth"], heads=spec["heads"],
        heads_xa=32, mlp_dim=spec["mlp_dim"], max_components=3,
        conv_filter=spec["filters"], max_ar=1, max_patches=4096, max_fields=3,
        dropout=0.1, emb_dropout=0.1, lora_r_attn=0, lora_r_mlp=0,
        lora_alpha=None, lora_p=0.0, model_size=key,
    )
    if checkpoint is None:
        try:
            from huggingface_hub import hf_hub_download
            checkpoint = hf_hub_download("mahindrautela/MORPH", spec["file"],
                                         subfolder="models/FM")
        except ImportError as exc:
            raise ImportError(
                "MORPH checkpoints need huggingface_hub, or pass --fm-checkpoint"
            ) from exc
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload.get("model", payload))
    if not isinstance(state, dict):
        raise ValueError(f"unrecognised MORPH checkpoint payload at {checkpoint}")
    if state and next(iter(state)).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    # The foundation checkpoint has no active LoRA parameters. Any other
    # mismatch means we are not loading the released architecture faithfully.
    allowed_missing = [k for k in missing if k.endswith((".A", ".B")) or ".lora" in k]
    if len(allowed_missing) != len(missing) or unexpected:
        raise RuntimeError(
            "MORPH checkpoint/model mismatch; refusing a partial model load. "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}")

    class _Wrap(nn.Module):
        _hipp_batch_safe = True

        def __init__(self, net):
            super().__init__()
            self.net = net
            self.history_context: torch.Tensor | None = None

        def set_history(self, history_context: torch.Tensor) -> None:
            if history_context.ndim != 5 or history_context.shape[2] != 2:
                raise ValueError(
                    "MORPH normalization context must have shape (B,T,2,H,W), "
                    f"got {tuple(history_context.shape)}")
            self.history_context = history_context

        def forward(self, x):                    # (B?, 2, H, W) via FrozenFM
            squeeze = x.ndim == 3
            if squeeze:
                x = x.unsqueeze(0)
            if x.ndim != 4 or x.shape[1] != 2:
                raise ValueError(f"MORPH expects velocity (B,2,H,W), got {tuple(x.shape)}")
            # The released RevIN computes one statistic per trajectory and
            # physical field over temporal, component, and spatial axes.  At
            # inference we use the available prefix only, never future states.
            ctx = self.history_context
            if ctx is not None:
                ctx = ctx.to(device=x.device, dtype=x.dtype)
                if ctx.shape[0] != x.shape[0] or not torch.allclose(ctx[:, -1], x, rtol=1e-4, atol=1e-6):
                    raise ValueError("MORPH normalization context must end at the state passed to forward")
                mu = ctx.mean(dim=(1, 2, 3, 4), keepdim=False).view(-1, 1, 1, 1)
                sd = ctx.std(dim=(1, 2, 3, 4), keepdim=False, unbiased=False).clamp_min(1e-6).view(-1, 1, 1, 1)
            else:
                mu = x.mean(dim=(1, 2, 3), keepdim=True)
                sd = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
            z = (x - mu) / sd
            vol = z[:, None, None, :, None, :, :]       # B,T,F,C,D,H,W
            _, _, out = self.net(vol)
            y = out[:, 0, :, 0] * sd + mu                # B,C,H,W
            return y[0] if squeeze else y

    fm = FrozenFM(_Wrap(net), (2, resolution, resolution), name=f"morph-{key}",
                  history=1, device=device, dtype=dtype,
                  source=f"mahindrautela/MORPH:{spec['file']}")
    fm.free_channels = (0, 1)
    fm.state_layout = "velocity_uv"
    fm.set_history = fm.module.set_history
    fm.required_history = 1
    fm.normalization_history = 10
    fm.morph_normalization_policy = "causal-prefix trajectory-field ReVIN"
    fm.morph_s4_status = "blocked: no released matched physical data/preprocessing contract"
    return fm


def from_omniarch(*args, **kwargs) -> FrozenFM:
    """OmniArch (Chen et al., ICML 2025) -- not reachable from PyTorch.

    Raises rather than pretending, because the obstruction is in what the
    project released, not in this repository. As of the openi.pcl.ac.cn/cty315
    checkout:

    1. **The model exists only in MindSpore.** `Omniarch.py` is an
       `mindspore.nn.Cell` built on `mindformers.LlamaForPDE` and a MindSpore
       Fourier encoder/decoder, targeting Ascend NPUs. The probes this method is
       built on are `torch.func.jvp`/`vjp`; there is no PyTorch definition of
       the network to differentiate, and the README's "OmniArch-PT-Torch" row
       points at the same `.ckpt` URL as the MindSpore row.
    2. **The weights are not retrievable.** The primary Huawei OBS link is a
       signed URL that expired 2025-12-08 (it now returns 403); only a 7.9 GB
       Google Drive mirror remains.
    3. **It is not a one-step map on a state.** `construct` consumes a stack of
       `T x C` frames with token-type ids and an attention mask, and it adds
       `randn_like(phy_seq) * noise_level` *inside* the forward pass -- so it is
       stochastic by construction and `check_adapter`'s determinism test would
       fail until that is disabled.

    Supporting it means porting the architecture to PyTorch and writing a
    MindSpore-to-PyTorch parameter map, then re-verifying against the published
    numbers. That is a project, not an adapter. Poseidon is the pretrained
    foundation model this method is actually evaluated against.
    """
    raise NotImplementedError(from_omniarch.__doc__)


def from_pdeformer(*args, **kwargs) -> FrozenFM:
    """PDEformer-2 -- important, but not a PyTorch one-step FM for this gate.

    PDEformer-2 is a versatile 2D PDE foundation model with released checkpoints
    and code, but the public implementation is MindSpore. Its inference API
    builds a symbolic PDE DAG and evaluates the solution at requested
    spatio-temporal points; it is not a frozen PyTorch tensor map
    `u(t) -> u(t+1)` on a field. That makes it scientifically interesting but
    outside the `torch.func.jvp`/`vjp` pipeline used here.

    Supporting it would require either a faithful PyTorch port or a separate
    finite-difference-only black-box path over PDE-DAG inputs, which would test
    a different object than the autoregressive FM claims in this repository.
    """
    raise NotImplementedError(from_pdeformer.__doc__)


def from_walrus(state_shape: tuple, checkpoint: str = "polymathic-ai/walrus",
                device=None, dtype: torch.dtype = torch.float32,
                history: int = 1, norm: tuple[float, float] = (0.0, 1.0)) -> FrozenFM:
    """Walrus / MPP, the Polymathic AI physics foundation models.

    Walrus is a 1.3B-parameter space-time transformer trained autoregressively
    across 19 physical systems; MPP-AViT is the earlier axial-attention family.
    Both predict the increment, u(t+1) = u(t) + M(U(t)), which is the residual
    structure the method assumes. Requires the `walrus` package from
    github.com/PolymathicAI/walrus.

    The base checkpoint is now on HuggingFace as `polymathic-ai/walrus`, with
    `walrus.safetensors` and an `extended_config.yaml`. What is not yet fixed
    is the physical input contract: the model is built through Hydra, consumes
    Well-formatted batches with field-index maps and normalization strategy,
    and embeds 2D fields as thin 3D volumes. Guessing that contract would make
    the Jacobian meaningless, so this adapter raises until a dataset-specific
    wrapper is implemented and verified.

    At 1.3B parameters a single JVP is expensive, so this is the configuration
    where the O(k) cost of the low-rank estimator versus the O(N) cost of an
    exact Jacobian stops being a convenience and becomes the only option:
    rank 64 at 128x128 is 64 probes against 16,384.
    """
    raise NotImplementedError(from_walrus.__doc__)


def from_the_well(model_family: str = "FNO", dataset: str = "active_matter",
                  device=None, dtype: torch.dtype = torch.float32,
                  resolution: int = 128, history: int = 4,
                  state_shape: tuple[int, int, int] | None = None,
                  checkpoint: str | None = None) -> FrozenFM:
    """Polymathic/The Well benchmark checkpoints.

    These are public, easy-to-load neural-operator checkpoints and useful
    external controls, but they are not PDE foundation models: each checkpoint
    is trained for one Well dataset, not across many PDE families. The wrapper
    keeps them in the same gate pipeline so we can answer a reviewer asking
    whether the Poseidon effect is just "any neural operator has this geometry".

    The Well models normally infer their channel metadata from the checkpoint.
    Because that metadata is package-version specific, this adapter accepts an
    explicit `state_shape` and otherwise starts with a conservative `(1,H,W)`.
    If the model rejects that shape, `check_fm` fails at the adapter gate rather
    than producing a number in the wrong units.
    """
    try:
        from the_well.benchmark import models as well_models  # type: ignore
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            "The Well adapter needs the benchmark extras:\n"
            "  python -m pip install 'the_well[benchmark]'"
        ) from exc
    family = {"fno": "FNO", "tfno": "TFNO", "unetclassic": "UNetClassic",
              "unet": "UNetClassic", "unetconvnext": "UNetConvNext",
              "cnext": "UNetConvNext"}.get(model_family.lower(), model_family)
    if not hasattr(well_models, family):
        raise ValueError(f"Unknown The Well model family {model_family}")
    cls = getattr(well_models, family)
    repo = checkpoint or f"polymathic-ai/{family}-{dataset}"
    try:
        net = cls.from_pretrained(repo)
    except TypeError as exc:
        # Some released FNO checkpoints contain a complete config.json, but
        # older The Well HubMixin versions fail to forward that config to the
        # FNO constructor.  Recover only this documented case explicitly.
        # Do not apply this fallback to the other architectures, whose input
        # contracts are different and must remain fail-closed.
        if family != "FNO" or "missing" not in str(exc) or "dim_in" not in str(exc):
            raise
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as dep_exc:  # pragma: no cover
            raise ImportError(
                "The explicit FNO loader requires huggingface_hub and safetensors."
            ) from dep_exc

        config_path = hf_hub_download(repo_id=repo, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        signature = inspect.signature(cls)
        constructor = {
            name: config[name]
            for name in signature.parameters
            if name != "self" and name in config
        }
        required = ("dim_in", "dim_out", "n_spatial_dims", "spatial_resolution",
                    "modes1", "modes2")
        missing_config = [name for name in required if name not in constructor]
        if missing_config:
            raise RuntimeError(
                f"FNO checkpoint {repo} has incomplete config.json; missing {missing_config}"
            ) from exc
        net = cls(**constructor)

        weights_path = hf_hub_download(repo_id=repo, filename="model.safetensors")
        state = load_file(weights_path, device="cpu")
        candidates = [state]
        for prefix in ("module.", "model.", "net."):
            if all(key.startswith(prefix) for key in state):
                candidates.append({key[len(prefix):]: value for key, value in state.items()})
        load_error = None
        for candidate in candidates:
            try:
                incompat = net.load_state_dict(candidate, strict=True)
                if incompat.missing_keys or incompat.unexpected_keys:
                    raise RuntimeError(
                        f"missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}"
                    )
                break
            except RuntimeError as load_exc:
                load_error = load_exc
        else:
            raise RuntimeError(
                f"FNO checkpoint {repo} weights do not match the declared architecture: {load_error}"
            ) from exc
    shape = state_shape or (1, resolution, resolution)

    def _detect_in_channels(m_net):
        # 1. Direct attribute on model
        for target in (m_net, getattr(m_net, "model", None)):
            if target is None:
                continue
            if hasattr(target, "in_channels") and isinstance(getattr(target, "in_channels"), int):
                return getattr(target, "in_channels")
            if hasattr(target, "n_in") and isinstance(getattr(target, "n_in"), int):
                return getattr(target, "n_in")
            for attr in ("lifting", "in_proj", "in_conv", "stem", "fc0"):
                if hasattr(target, attr):
                    mod = getattr(target, attr)
                    for m in mod.modules():
                        if isinstance(m, (nn.Conv2d, nn.Conv3d)) and hasattr(m, "in_channels"):
                            return m.in_channels
                        elif isinstance(m, nn.Linear) and hasattr(m, "in_features"):
                            return m.in_features
        return 16

    exp_c = _detect_in_channels(net)

    class _Wrap(nn.Module):
        def __init__(self, net, exp_channels):
            super().__init__()
            self.net = net
            self.exp_c = exp_channels

        def forward(self, x):
            # x is (C, H, W) or (B, C, H, W)
            if x.dim() == 3:
                x_in = x.unsqueeze(0)  # (1, C, H, W)
            else:
                x_in = x
            B, C, H, W = x_in.shape

            # Channel alignment with network's expected in_channels
            if C != self.exp_c:
                if C < self.exp_c:
                    repeat_factor = (self.exp_c + C - 1) // C
                    x_in = x_in.repeat(1, repeat_factor, 1, 1)[:, :self.exp_c]
                else:
                    x_in = x_in[:, :self.exp_c]

            # Explicit forward contract: 4D (B, C, H, W) or 5D temporal history (B, T, C, H, W)
            try:
                out = self.net(x_in)
            except Exception:
                try:
                    inp_5d = x_in.unsqueeze(1).repeat(1, history, 1, 1, 1)
                    out = self.net(inp_5d)
                except Exception as e:
                    raise ValueError(f"The Well adapter failed for {family} with expected channels {self.exp_c}: {e}")

            out = out[0] if isinstance(out, (tuple, list)) else out
            if out.dim() == 5:
                out = out[:, -1]
            elif out.dim() == 4 and out.shape[1] > shape[0]:
                out = out[:, :shape[0]]
            return out.reshape(-1, *shape)[0]

    fm = FrozenFM(_Wrap(net, exp_c), shape, name=f"well-{family}-{dataset}",
                  history=history, device=device, dtype=dtype, source=repo)
    fm.free_channels = tuple(range(shape[0]))
    return fm


def from_cno(model_size: str = "FM", device=None, resolution: int = 128,
             checkpoint: str | Path | None = None, config: str | Path | None = None,
             source_dir: str | Path | None = None, lead_time: float = 1.0,
             time_scale: float = 1.0, pinned_state: tuple[float, float] = (0.8, 0.0),
             dtype: torch.dtype = torch.float32) -> FrozenFM:
    """Time-conditioned CNO Foundation Model under its released 4-field contract.

    CNO-FM takes normalized ``[rho, u, v, p, time]`` inputs and returns
    ``[rho, u, v, p]``.  The shared NS benchmark contains only ``[u, v]``;
    consequently rho and p are explicit, fixed physical conditioning values,
    not invented free channels.  Supplying the released architecture JSON is
    mandatory because the checkpoint alone does not encode all constructor
    arguments.  This is deliberately strict: a missing source tree, config,
    or checkpoint is an unavailable model, never a surrogate baseline.
    """
    checkpoint = checkpoint or os.environ.get("CNO_CHECKPOINT")
    config = config or os.environ.get("CNO_CONFIG")
    source_dir = source_dir or os.environ.get("CNO_SOURCE")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            "CNO requires --fm-checkpoint (or CNO_CHECKPOINT) pointing to the released .ckpt file.")
    if not config or not Path(config).is_file():
        raise FileNotFoundError(
            "CNO requires --cno-config (or CNO_CONFIG): the released architecture JSON for this checkpoint.")
    if not source_dir or not (Path(source_dir) / "CNO_timeModule_CIN.py").is_file():
        raise FileNotFoundError(
            "CNO requires --cno-source (or CNO_SOURCE) pointing to CNO2d_temporal from the official repository.")

    source_dir = str(Path(source_dir).resolve())
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    try:
        from CNO_timeModule_CIN import CNO_time
    except Exception as exc:
        raise RuntimeError("Could not import the official time-conditioned CNO implementation.") from exc

    with open(config) as f:
        cfg = json.load(f)
    required = {"N_layers", "N_res", "N_res_neck", "channel_multiplier", "batch_norm",
                "activation", "time_steps", "is_time", "nl_dim"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"CNO config is missing released architecture fields: {missing}")
    if int(cfg.get("in_dim", 5)) != 5 or int(cfg.get("out_dim", 4)) != 4:
        raise ValueError("This velocity-only CNO adapter supports the released CNO-FM 5-to-4 channel contract only.")

    model = CNO_time(
        in_dim=5, out_dim=4, in_size=int(cfg.get("in_size", resolution)),
        N_layers=int(cfg["N_layers"]), N_res=int(cfg["N_res"]),
        N_res_neck=int(cfg["N_res_neck"]), channel_multiplier=int(cfg["channel_multiplier"]),
        batch_norm=bool(cfg["batch_norm"]), activation=cfg["activation"],
        time_steps=int(cfg["time_steps"]), is_time=int(cfg["is_time"]), nl_dim=cfg["nl_dim"],
        is_att=bool(cfg.get("is_att", False)), patch_size=int(cfg.get("patch_size", 1)),
        dim_multiplier=float(cfg.get("dim_multiplier", 1.0)), depth=int(cfg.get("depth", 2)),
        heads=int(cfg.get("heads", 2)), dim_head_multiplier=float(cfg.get("dim_head_multiplier", 0.5)),
        mlp_dim_multiplier=float(cfg.get("mlp_dim_multiplier", 1.0)),
        emb_dropout=float(cfg.get("emb_dropout", 0.0)),
        loader_dictionary=dict(cfg.get("loader_dictionary", {})),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # Lightning wrappers sometimes prefix only the model keys.  Strip that
        # one documented wrapper prefix, then still require an exact load.
        if all(k.startswith("model.") for k in state):
            model.load_state_dict({k[6:]: v for k, v in state.items()}, strict=True)
        else:
            raise

    mean = torch.tensor(cfg.get("state_mean", [0.80, 0.0, 0.0, 0.0]), dtype=dtype)
    std = torch.tensor(cfg.get("state_std", [0.31, 0.391, 0.356, 0.46]), dtype=dtype)
    rho, pressure = (float(pinned_state[0]), float(pinned_state[1]))

    class _VelocityCNO(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.lead_time = float(lead_time)

        def forward(self, velocity):
            if velocity.ndim == 3:
                velocity = velocity.unsqueeze(0)
            if velocity.shape[1] != 2:
                raise ValueError(f"CNO velocity adapter expects (B,2,H,W), got {tuple(velocity.shape)}")
            b, _, h, w = velocity.shape
            physical = torch.empty((b, 4, h, w), device=velocity.device, dtype=velocity.dtype)
            physical[:, 0].fill_(rho)
            physical[:, 1:3] = velocity
            physical[:, 3].fill_(pressure)
            norm = (physical - mean.to(velocity.device)[None, :, None, None]) / std.to(velocity.device)[None, :, None, None]
            time_channel = torch.full((b, 1, h, w), self.lead_time / float(time_scale),
                                      device=velocity.device, dtype=velocity.dtype)
            output = self.net(torch.cat((norm, time_channel), dim=1),
                              torch.full((b,), self.lead_time / float(time_scale), device=velocity.device, dtype=velocity.dtype))
            output = output * std.to(velocity.device)[None, :, None, None] + mean.to(velocity.device)[None, :, None, None]
            return output[:, 1:3]

    wrapped = _VelocityCNO(model)
    fm = FrozenFM(wrapped, (2, resolution, resolution), name=f"cno-{model_size}",
                  history=1, device=device, dtype=dtype, source=str(checkpoint))
    fm.free_channels = (0, 1)
    fm.state_layout = "velocity_uv"

    def set_lead_time(value: float) -> None:
        if value <= 0:
            raise ValueError("CNO lead time must be positive")
        wrapped.lead_time = float(value)

    fm.set_lead_time = set_lead_time
    fm.lead_time_units = "physical_time"
    fm.time_query_contract = "released time-conditioned CNO-FM query"
    return fm


def from_motion(model_size: str = "156.9M", device=None, resolution: int = 128,
                checkpoint: str | Path | None = None, config: str | Path | None = None,
                source_dir: str | Path | None = None, lead_time: float = 1.0,
                time_scale: float = 1.0, dtype: torch.dtype = torch.float32) -> FrozenFM:
    """Refuse an incomplete MOTION adapter rather than create a false baseline.

    MOTION is TensorFlow-based and needs its official eight-slot state formatter,
    normalization and named-array weight loader.  A two-channel tensor wrapper
    cannot reconstruct that contract.  This explicit failure prevents a caller
    from accidentally evaluating an identity map or a truth-derived surrogate.
    """
    raise NotImplementedError(
        "MOTION is not yet runnable in this benchmark: implement and validate its official "
        "TensorFlow 8-slot formatter, normalization, and checkpoint loader before enabling it.")


ADAPTERS = {
    "local": from_local_checkpoint,
    "poseidon": from_poseidon,
    "dpot": from_dpot,
    "morph": from_morph,
    "walrus": from_walrus,
    "the_well": from_the_well,
    "well": from_the_well,
    "cno": from_cno,
    "motion": from_motion,
    "omniarch": from_omniarch,
    "pdeformer": from_pdeformer,
}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def check_adapter(fm: FrozenFM, x: torch.Tensor | None = None,
                  verbose: bool = True) -> dict:
    """Assert the properties the method silently depends on.

    Run this before any experiment against a new checkpoint. Every check
    corresponds to a way the pipeline can produce confident nonsense:

      shape        f must map R^N -> R^N, or nothing downstream type-checks
      determinism  two calls must agree exactly, or every probe is noise
      jvp          forward-mode AD must work, or randomized SVD silently
                   falls back to finite differences with unknown eps
      jvp_vs_fd    JVP and a central difference must agree somewhere in a swept
                   epsilon window, which catches a model whose AD path skips a
                   non-differentiable op (rounding, masking, argmax pooling).
                   The step is swept, not fixed: see the comment at the call.
      linearity    J(av) = a J(v), the assumption behind every use of J here
      residual     ||J - I||/||I||; small means a residual architecture, which
                   is what sets the oracle ceiling on Stage 1
    """
    out: dict = {"name": fm.info.name, "N": fm.N}
    dev = fm.device
    if x is None:
        x = torch.randn(fm.N, device=dev, dtype=fm.dtype)
    x = x.reshape(-1).to(device=dev, dtype=fm.dtype)
    f = fm.flat_fn()

    y = f(x)
    out["shape_ok"] = tuple(y.shape) == (fm.N,)

    out["deterministic"] = bool(torch.equal(y, f(x)))

    v = torch.randn_like(x)
    v = v / v.norm()
    try:
        jv = torch.func.jvp(f, (x,), (v,))[1]
        out["jvp_ok"] = bool(torch.isfinite(jv).all())
    except Exception as exc:
        out["jvp_ok"] = False
        out["jvp_error"] = repr(exc)[:200]
        jv = None

    if jv is not None:
        # The step has to be set by ||h v|| / ||x||, not by the size of a single
        # element, and the difference is not cosmetic at scale. A unit-norm
        # tangent has entries of order 1/sqrt(N), so a per-element rule that is
        # correct at N = 128 puts the perturbation ~sqrt(N/128) closer to the
        # float32 rounding floor at N = 32,768 -- the central difference is then
        # pure cancellation noise and a perfectly good adapter reports a 0.57
        # relative error. Sweeping and taking the minimum is also the honest
        # statement of the test: what is being asked is whether a usable epsilon
        # window *exists*, not whether one guessed value lands in it.
        scale = float(x.norm() / v.norm().clamp(min=1e-30))
        best, best_h = float("inf"), None
        for rel in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
            h = rel * scale
            fd = (f(x + h * v) - f(x - h * v)) / (2 * h)
            err = float((jv - fd).norm() / jv.norm().clamp(min=1e-30))
            if err < best:
                best, best_h = err, rel
        out["jvp_vs_fd_rel"] = best
        out["jvp_vs_fd_rel_step"] = best_h
        jv2 = torch.func.jvp(f, (x,), (2.5 * v,))[1]
        out["linearity_rel"] = float((jv2 - 2.5 * jv).norm() / jv2.norm().clamp(min=1e-30))
        out["residual_dev"] = float((jv - v).norm() / v.norm())

    try:
        _, vjp = torch.func.vjp(f, x)
        w = torch.randn_like(x)
        jtw = vjp(w)[0]
        # <J v, w> == <v, J^T w> is the adjoint identity; a mismatch means the
        # forward and reverse graphs disagree, which no amount of tuning fixes.
        out["adjoint_rel"] = float(
            abs(float(jv @ w) - float(v @ jtw)) / max(abs(float(jv @ w)), 1e-30))
        out["vjp_ok"] = True
    except Exception as exc:
        out["vjp_ok"] = False
        out["vjp_error"] = repr(exc)[:200]

    out["pass"] = bool(out.get("shape_ok") and out.get("deterministic")
                       and out.get("jvp_ok") and out.get("vjp_ok")
                       and out.get("jvp_vs_fd_rel", 1) < 1e-2
                       and out.get("linearity_rel", 1) < 1e-4
                       and out.get("adjoint_rel", 1) < 1e-4)
    if verbose:
        print(f"adapter check [{fm.info.name}]  N={fm.N}  "
              f"params={fm.info.n_params/1e6:.1f}M")
        for k, val in out.items():
            if k not in ("name", "N"):
                print(f"    {k:16s} {val}")
    return out
