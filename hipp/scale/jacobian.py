"""Matrix-free Jacobian access, batched for GPUs.

`hipp/curvature.py` builds J with `jacrev` (N backward passes) and loops one
probe at a time. At N = 128 that costs 128 passes and is the right call, since
it gives exact ground truth. At N = 49,152 it is 49,152 passes per state and the
result does not fit in memory anyway.

Here J is never formed. Everything goes through two primitives,

    J v   (forward-mode JVP)     and    J^T w   (reverse-mode VJP)

each costing about one network evaluation, batched over a block of vectors so a
GPU is actually saturated. Rank-k geometry then costs O(k) network evaluations
per state rather than O(N).

Batching notes
--------------
`torch.vmap` over `jvp` is the fast path but not every operator survives it --
custom kernels, some FFT paths, attention with flash backends and anything using
`.item()` will raise. `probe_map` tries vmap once per call site, caches the
outcome on the function object, and falls back to a plain loop. The fallback is
slower but numerically identical, so a model that cannot be vmapped still runs.

Precision
---------
Probes run in the model's own dtype (float32 or bf16 under autocast); only the
small (ell x ell) QR/SVD factors are promoted to float64. Doing the reduction in
float64 matters: the singular values of a residual model are clustered near 1
(cond(J) ~ 1.03-1.7 in the 1D experiments) and a float32 QR loses enough digits
to reorder them, which corrupts exactly the rank ordering Stage 1 tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .lowrank import rademacher, randn


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

def set_probe_precision(high: bool = True) -> dict:
    """Turn TF32 off (high=True) or on for the Jacobian probe paths.

    On Ampere and later, cuDNN enables TF32 by default. TF32 keeps float32's
    exponent but only 10 mantissa bits, i.e. ~1e-3 relative precision, and that
    is catastrophic *here specifically* for a reason worth stating:

    the surrogate is residual, J = I + dNet/dc, and the measured deviation is
    ||J - I||/||I|| ~ 0.025. The entire signal is a 2.5% perturbation of the
    identity, so 1e-3 noise on J v is ~4% noise on the only part that carries
    information. Measured on an H200: the adjoint identity <Jv,w> = <v,J^T w>
    degrades from 3e-7 (CPU, true fp32) to 4e-4, and the finite-difference
    estimator loses its usable epsilon window entirely -- its best relative
    error moves from 1.4e-5 at eps=1e-2 to 1.3e-3 at eps=1e-1, because
    differencing two nearly equal numbers is exactly what TF32 cannot do.

    Training is left alone: there TF32 is a large speedup and the loss does not
    care about the tenth mantissa bit. Only the analysis stages call this.

    Returns the previous settings so a caller can restore them.
    """
    prev = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    torch.backends.cuda.matmul.allow_tf32 = not high
    torch.backends.cudnn.allow_tf32 = not high
    torch.set_float32_matmul_precision("highest" if high else "high")
    return prev


def restore_probe_precision(prev: dict) -> None:
    torch.backends.cuda.matmul.allow_tf32 = prev["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = prev["cudnn_tf32"]
    torch.set_float32_matmul_precision(prev["matmul_precision"])


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def probe_map(fn, M: torch.Tensor, chunk: int = 8, vectorize: bool = True):
    """Apply `fn` to every column of M, returning columns stacked.

    M: (N, m) -> (N_out, m). `chunk` bounds peak memory: vmap materializes
    `chunk` copies of every intermediate activation at once, so this is the knob
    that trades throughput against a 128x128x3 model's activation footprint.
    """
    if M.shape[1] == 0:
        return torch.zeros(M.shape[0], 0, dtype=M.dtype, device=M.device)
    if vectorize and getattr(fn, "_vmap_ok", True):
        try:
            out = torch.vmap(fn, in_dims=0, out_dims=0,
                             chunk_size=max(1, chunk))(M.T.contiguous())
            fn._vmap_ok = True
            return out.T.contiguous()
        except Exception:
            # Cache the failure so we pay the vmap trace cost once, not per state.
            try:
                fn._vmap_ok = False
            except AttributeError:
                pass
    return torch.stack([fn(M[:, i]) for i in range(M.shape[1])], dim=1)


def jvp_operator(f, x: torch.Tensor):
    """v -> J v at x. Forward mode: one primal+tangent pass, no stored graph."""
    x = x.detach()

    def apply(v: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(f, (x,), (v.reshape(x.shape),))[1].reshape(-1)
    return apply


def vjp_operator(f, x: torch.Tensor):
    """w -> J^T w at x. The graph is built once and reused across all w."""
    x = x.detach()
    _, fn = torch.func.vjp(f, x)

    def apply(w: torch.Tensor) -> torch.Tensor:
        return fn(w.reshape(x.shape))[0].reshape(-1)
    return apply


@dataclass
class JacobianOps:
    """Both directions plus the bookkeeping the estimators need."""
    matvec: object                   # v -> J v
    rmatvec: object                  # w -> J^T w
    N: int
    dtype: torch.dtype
    device: torch.device
    n_calls: list                    # [jvp_count, vjp_count], mutated in place

    def mm(self, V: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        self.n_calls[0] += V.shape[1]
        return probe_map(self.matvec, V, chunk=chunk)

    def rmm(self, W: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        self.n_calls[1] += W.shape[1]
        return probe_map(self.rmatvec, W, chunk=chunk)


def jacobian_ops(f, x: torch.Tensor) -> JacobianOps:
    """Wrap a flat function f: (N,) -> (N,) into matrix-free J operators."""
    x = x.detach().reshape(-1)
    return JacobianOps(matvec=jvp_operator(f, x), rmatvec=vjp_operator(f, x),
                       N=int(x.numel()), dtype=x.dtype, device=x.device,
                       n_calls=[0, 0])


def batched_jacobian_apply(fm, states: torch.Tensor, vectors: torch.Tensor, *,
                           adjoint: bool = False, chunk: int = 8) -> torch.Tensor:
    """Apply independent local Jacobians to a batch of state/probe pairs.

    Parameters are ``states=(B,N)`` and ``vectors=(B,N,M)``.  The result is
    ``(B,N,M)`` with member ``b`` equal to ``J(states[b]) vectors[b]`` (or its
    adjoint).  Flattening the ``B x chunk`` independent AD calls lets vmap send
    one ordinary large batch through a batch-safe frozen model.  It is exactly
    the serial collection of JVPs/VJPs: every state and probe is retained, only
    their scheduling changes.

    This function deliberately rejects adapters that have not declared a
    batch-safe forward contract.  Falling back silently would make a speed flag
    appear active while still executing a serial loop.
    """
    if states.ndim != 2 or vectors.ndim != 3:
        raise ValueError("states must be (B,N) and vectors must be (B,N,M)")
    b, n = states.shape
    if vectors.shape[:2] != (b, n):
        raise ValueError(f"incompatible state/vector shapes {states.shape}, {vectors.shape}")
    if b < 1 or vectors.shape[2] < 1:
        raise ValueError("batched Jacobian application needs nonempty batches")
    f = fm.flat_batch_fn()
    out = []
    # `torch.func.jvp`/`vjp` act on an entire state batch.  For a separable FM
    # in eval mode this is block diagonal across B, so the tangent/pullback is
    # exactly the requested per-state Jacobian action.
    for lo in range(0, vectors.shape[2], max(1, int(chunk))):
        hi = min(vectors.shape[2], lo + max(1, int(chunk)))
        v = vectors[:, :, lo:hi].permute(2, 0, 1).contiguous()  # (m,B,N)
        if adjoint:
            def one(w):
                _, pullback = torch.func.vjp(f, states)
                return pullback(w)[0]
            z = torch.vmap(one)(v)
        else:
            def one(tangent):
                return torch.func.jvp(f, (states,), (tangent,))[1]
            z = torch.vmap(one)(v)
        out.append(z.permute(1, 2, 0).contiguous())
    return torch.cat(out, dim=2)


# ---------------------------------------------------------------------------
# Randomized SVD
# ---------------------------------------------------------------------------

def randomized_svd(ops: JacobianOps, k: int = 32, oversample: int = 10,
                   n_iter: int = 2, chunk: int = 8, generator=None,
                   dtype_reduce: torch.dtype = torch.float64):
    """Rank-k SVD of J from O(k) network passes (Halko, Martinsson & Tropp).

    Cost: (1 + 2*n_iter) * ell JVPs and (1 + n_iter) * ell VJPs, ell = k+oversample.

    `n_iter` power iterations sharpen the separation between retained and
    discarded singular values. For a residual PDE operator J = I + dNet/dc the
    spectrum is flat near 1, which is the worst case for randomized range
    finding, so n_iter >= 2 is not optional here -- n_iter = 0 mixes ranks
    badly enough to change Stage 1's conclusion. `s0_validate_lowrank.py`
    measures this against the exact SVD.

    Returns (U, S, V) with U: (N, k), S: (k,) descending, V: (N, k).
    """
    N = ops.N
    ell = min(k + oversample, N)
    Omega = randn((N, ell), ops.dtype, ops.device, generator)

    Y = ops.mm(Omega, chunk=chunk)
    Q = _qr_q(Y, dtype_reduce)
    for _ in range(n_iter):
        Q = _qr_q(ops.rmm(Q, chunk=chunk), dtype_reduce)
        Q = _qr_q(ops.mm(Q, chunk=chunk), dtype_reduce)

    B = ops.rmm(Q, chunk=chunk).to(dtype_reduce)            # (N, ell) = J^T Q
    # B^T = Q^T J is (ell, N); its SVD lifts back through Q.
    Ub, S, Vh = torch.linalg.svd(B.T, full_matrices=False)
    U = (Q.to(dtype_reduce) @ Ub)[:, :k].contiguous()
    return U, S[:k].contiguous(), Vh[:k].T.contiguous()


def _qr_q(A: torch.Tensor, dtype_reduce: torch.dtype) -> torch.Tensor:
    """Orthonormal basis for range(A), reduced in high precision."""
    Q, _ = torch.linalg.qr(A.to(dtype_reduce), mode="reduced")
    return Q.to(A.dtype) if A.dtype != dtype_reduce else Q


def jacobian_spectrum_probe(ops: JacobianOps, k: int = 32, **kw) -> dict:
    """Singular values plus the summary scalars Stage 1's Q2 correlates on."""
    U, S, V = randomized_svd(ops, k=k, **kw)
    s2 = (S.double() ** 2)
    p = s2 / s2.sum().clamp(min=1e-300)
    return {
        "U": U, "S": S, "V": V,
        "smax": float(S.max()), "smin": float(S.min()),
        "s_mean": float(S.mean()),
        "cond_k": float(S.max() / S.min().clamp(min=1e-30)),
        "trace_k": float(s2.sum()),
        "eff_rank_k": float(torch.exp(-(p * p.clamp(min=1e-300).log()).sum())),
        "jvp_calls": ops.n_calls[0], "vjp_calls": ops.n_calls[1],
    }


# ---------------------------------------------------------------------------
# Diagnostics that need no basis
# ---------------------------------------------------------------------------

def hutchinson_diag_gramian(ops: JacobianOps, n_probe: int = 32, chunk: int = 8,
                            generator=None) -> torch.Tensor:
    """diag(J^T J) by Rademacher probes: E[(J z) * (J z)] summed the right way.

    diag(J^T J)_i = sum_j J_ji^2, and for Rademacher z, E[(J^T J z)_i z_i] gives
    the diagonal of J^T J directly. Two matvecs per probe.
    """
    Z = rademacher((ops.N, n_probe), ops.dtype, ops.device, generator)
    JZ = ops.mm(Z, chunk=chunk)
    JtJZ = ops.rmm(JZ, chunk=chunk)
    return (JtJZ * Z).mean(dim=1).clamp(min=0)


def frobenius_norm_estimate(ops: JacobianOps, n_probe: int = 16, chunk: int = 8,
                            generator=None) -> float:
    """E[||J z||^2] / n = ||J||_F^2 / N for z ~ N(0, I). One matvec per probe."""
    Z = randn((ops.N, n_probe), ops.dtype, ops.device, generator)
    JZ = ops.mm(Z, chunk=chunk)
    return float((JZ.double() ** 2).sum() / n_probe)


def subspace_overlap(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Mean squared principal cosine between two orthonormal bases, in [0, 1].

    1.0 means the leading geometry is identical between the two states, i.e. a
    single fixed covariance would do as well as a state-dependent one. This is
    the number that turned `advdiff` into a negative control in the 1D study,
    and it is the first thing to check at scale.
    """
    k = min(U1.shape[1], U2.shape[1])
    if k == 0:
        return float("nan")
    M = U1[:, :k].T.to(torch.float64) @ U2[:, :k].to(torch.float64)
    return float((M ** 2).sum() / k)


# ---------------------------------------------------------------------------
# Finite-difference probes (black-box models)
# ---------------------------------------------------------------------------

def fd_jvp_operator(f, x: torch.Tensor, eps: float = 1e-3, central: bool = True):
    """Forward-only surrogate for J v, for models with no usable AD path.

    Needed for genuinely black-box foundation models (compiled inference graphs,
    served checkpoints, non-differentiable preprocessing). `eps` must be swept:
    too small and float32 cancellation dominates, too large and the
    linearization breaks. The 1D study found a usable window; whether one exists
    for a large model in float32/bf16 is an empirical question that
    `s0_validate_lowrank.py` answers.
    """
    x = x.detach()
    base = None if central else f(x).reshape(-1).detach()

    def apply(v: torch.Tensor) -> torch.Tensor:
        v = v.reshape(x.shape)
        nrm = v.norm().clamp(min=1e-30)
        h = eps / nrm * math.sqrt(x.numel())          # keep the step size scale-free
        plus = f(x + h * v).reshape(-1).detach()
        if central:
            minus = f(x - h * v).reshape(-1).detach()
            return (plus - minus) / (2 * h)
        return (plus - base) / h
    return apply
