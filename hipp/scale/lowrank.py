"""Low-rank-plus-isotropic Gaussians: the representation that makes HILP scale.

    Sigma = alpha * S,      S = U diag(d) U^T + tau I,     U^T U = I_k

`U` is (N, k) with k << N, `d` is (k,) and may be **negative**; the only
requirement is that every eigenvalue of S is positive, i.e. `d_i + tau > 0` and
`tau > 0`. Allowing signed `d` is what lets a single class carry both curvature
surrogates, which is otherwise the awkward part of scaling this method:

    pushforward   Sigma = J J^T + tau_j I
                  -> U = left singular vectors, d = s^2,       tau = tau_j
    Gauss-Newton  Sigma = (J^T J + tau_j I)^{-1}
                  -> U = right singular vectors,
                     d = 1/(s^2 + tau_j) - 1/tau_j  (negative),  tau = 1/tau_j

Both are exact re-writings, not approximations -- the approximation is only in
truncating the SVD of J to rank k, which `jacobian.randomized_svd` does and
`experiments/scale/s0_validate_lowrank.py` quantifies against the exact object.

Every quantity below is O(N k) time and O(N k) memory. The dense-N algebra of
`hipp/priors.py` (Cholesky, eigh, explicit inverse) never appears.

Streaming
---------
At N ~ 5e4 a rank-64 basis is ~13 MB in float64, so holding one per calibration
state is not possible for thousands of states. `LowRankGaussian.summarize()`
reduces a state to the handful of numbers that alpha, NLL and coverage actually
depend on -- ||r||^2, the k projections U^T r, d, tau -- which is a few hundred
bytes. Calibration is then a single streaming pass; see `calibrate_scale.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

EPS_EIG = 1e-30


# ---------------------------------------------------------------------------
# Seeded randomness that survives a device change
# ---------------------------------------------------------------------------

def randn(shape, dtype, device, generator=None) -> torch.Tensor:
    """`torch.randn` that tolerates a generator living on another device.

    torch refuses `torch.randn(..., device='cuda', generator=<cpu generator>)`
    with "Expected a 'cuda' device type for generator". Every probe in this
    package is seeded for reproducibility, and the natural thing for a caller to
    write is `torch.Generator().manual_seed(s)` -- which is a *CPU* generator.
    That works on CPU and raises on GPU, i.e. the failure appears only once the
    study moves to the hardware it was written for.

    Drawing on the generator's own device and copying keeps a given seed
    producing the same probes on CPU and GPU, which also makes a CPU debugging
    run reproduce a GPU run exactly. The copy is negligible: these are (N, k)
    probe blocks, not activations.
    """
    device = torch.device(device) if device is not None else None
    if generator is not None and device is not None \
            and generator.device.type != device.type:
        return torch.randn(*shape, dtype=dtype, generator=generator).to(device)
    return torch.randn(*shape, dtype=dtype, device=device, generator=generator)


def rademacher(shape, dtype, device, generator=None) -> torch.Tensor:
    """+/-1 probes, with the same cross-device generator tolerance as `randn`."""
    device = torch.device(device) if device is not None else None
    if generator is not None and device is not None \
            and generator.device.type != device.type:
        bits = torch.randint(0, 2, tuple(shape), generator=generator,
                             dtype=torch.int8).to(device)
    else:
        bits = torch.randint(0, 2, tuple(shape), device=device,
                             generator=generator, dtype=torch.int8)
    return bits.to(dtype) * 2 - 1


# ---------------------------------------------------------------------------
# Streaming summary
# ---------------------------------------------------------------------------

@dataclass
class QuadSummary:
    """Everything a scale fit / NLL / coverage needs from one state.

    Storing this instead of the (N, k) basis is what makes calibration over
    thousands of states affordable. `r_sq` is ||y - mean||^2 and `proj` is
    U^T (y - mean); together with (d, tau, N) they determine the Mahalanobis
    distance and the log-determinant exactly.
    """
    r_sq: float
    proj: np.ndarray          # (k,) U^T r
    d: np.ndarray             # (k,)
    tau: float
    N: int
    extra: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.d.shape[0])

    def quad(self) -> float:
        """r^T S^{-1} r, with S the *unit-scale* shape (alpha factored out)."""
        w = self.d / (self.d + self.tau)
        return float((self.r_sq - float((w * self.proj ** 2).sum())) / self.tau)

    def logdet(self) -> float:
        """log det S."""
        return float(np.log(self.d + self.tau).sum()
                     + (self.N - self.k) * math.log(self.tau))

    def maha_sq(self, alpha: float) -> float:
        return self.quad() / alpha

    def nll(self, alpha: float, per_dim: bool = True) -> float:
        """Joint Gaussian negative log-likelihood under Sigma = alpha * S."""
        val = 0.5 * (self.quad() / alpha + self.logdet()
                     + self.N * math.log(alpha)
                     + self.N * math.log(2 * math.pi))
        return val / self.N if per_dim else val


# ---------------------------------------------------------------------------
# The Gaussian
# ---------------------------------------------------------------------------

class LowRankGaussian:
    """N(mean, alpha * (U diag(d) U^T + tau I)).

    Parameters
    ----------
    mean : (N,) tensor
    U    : (N, k) tensor with orthonormal columns. Orthonormality is *assumed*,
           not enforced, because every producer in this package (randomized SVD,
           QR-truncated propagation, residual PCA) already guarantees it and a
           re-orthogonalization here would double the dominant cost. Pass
           `check=True` to verify during development.
    d    : (k,) eigenvalue offsets, may be negative; d + tau must be > 0.
    tau  : isotropic floor, > 0. This is the variance the model assigns to every
           direction outside span(U) -- at scale it is *not* a numerical nuisance
           parameter but the thing that keeps the (N-k)-dimensional complement
           from having zero variance, so it is fitted, not hard-coded.
    """

    def __init__(self, mean: torch.Tensor, U: torch.Tensor, d: torch.Tensor,
                 tau: float, alpha: float = 1.0, label: str = "",
                 check: bool = False):
        self.mean = mean.detach().reshape(-1)
        self.U = U.detach()
        self.d = d.detach().reshape(-1).to(self.U.dtype)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.label = label
        self.N = int(self.mean.numel())
        self.k = int(self.d.numel())
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")
        eig = self.d + self.tau
        if bool((eig <= 0).any()):
            raise ValueError(
                f"{label or 'LowRankGaussian'}: d + tau has non-positive entries "
                f"(min {float(eig.min()):.3e}); the covariance is not PSD")
        if self.U.shape != (self.N, self.k):
            raise ValueError(f"U has shape {tuple(self.U.shape)}, expected {(self.N, self.k)}")
        if check:
            gram = self.U.T @ self.U
            off = float((gram - torch.eye(self.k, dtype=gram.dtype,
                                          device=gram.device)).abs().max())
            if off > 1e-4:
                raise ValueError(f"U is not orthonormal (max |U^T U - I| = {off:.2e})")

    # ---- constructors --------------------------------------------------
    @classmethod
    def isotropic(cls, mean: torch.Tensor, tau: float = 1.0, alpha: float = 1.0,
                  label: str = "isotropic") -> "LowRankGaussian":
        U = torch.zeros(mean.numel(), 0, dtype=mean.dtype, device=mean.device)
        d = torch.zeros(0, dtype=mean.dtype, device=mean.device)
        return cls(mean, U, d, tau, alpha=alpha, label=label)

    @classmethod
    def from_pushforward(cls, mean: torch.Tensor, U: torch.Tensor, svals: torch.Tensor,
                         tau_rel: float = 1e-2, alpha: float = 1.0,
                         label: str = "pushforward",
                         tail_trace: float | None = None) -> "LowRankGaussian":
        """Sigma = J J^T + tau I, truncated to the leading k singular triples.

        **`tau` means something different here than it did in 1D, and getting it
        wrong is the single easiest way to invalidate a scaled result.** In the
        dense code all N directions were retained and tau was a regularizer
        keeping J^T J invertible; `tau_rel = 1e-2` was a safe nominal value. At
        rank k << N, tau is the variance assigned to N-k *real* directions of the
        operator, and setting it to 1% of the mean retained eigenvalue
        under-predicts the complement by orders of magnitude -- which shows up as
        Mahalanobis distances inflated by ~N/k and cannot be repaired by the
        scale calibration, since alpha multiplies subspace and floor alike.

        `tail_trace` is sum_{i>k} s_i^2, the discarded spectral mass, which
        `curvature_scale.estimate_scale` estimates by Hutchinson probes for a few
        extra network passes (||J||_F^2 is cheap; subtract the captured part).
        Given it, the floor is the mean discarded eigenvalue -- the same choice
        `from_dense_covariance` makes, and the one that conserves the trace.
        `tau_rel` then returns to being what it was: a small explicit damping.

        Passing `tail_trace=None` reproduces the 1D convention and is kept only
        so the two can be compared; it is not the right default at scale.
        """
        s2 = (svals.to(U.dtype) ** 2)
        damp = float(tau_rel * float(s2.mean().clamp(min=EPS_EIG)))
        if tail_trace is None:
            tau = damp
        else:
            n_perp = max(mean.numel() - U.shape[1], 1)
            tau = max(float(tail_trace) / n_perp, 0.0) + damp
        return cls(mean, U, s2, max(tau, EPS_EIG), alpha=alpha, label=label)

    @classmethod
    def from_gauss_newton(cls, mean: torch.Tensor, V: torch.Tensor, svals: torch.Tensor,
                          tau_rel: float = 1e-2, alpha: float = 1.0,
                          label: str = "gn",
                          tail_trace: float | None = None) -> "LowRankGaussian":
        """Sigma = (J^T J + tau_j I)^{-1}, exactly, in low-rank-plus-isotropic form.

        (V diag(s^2) V^T + tau_j I)^{-1}
            = V diag(1/(s^2+tau_j)) V^T + (1/tau_j)(I - V V^T)
            = V diag(1/(s^2+tau_j) - 1/tau_j) V^T + (1/tau_j) I

        so `d` is negative and `tau = 1/tau_j`.

        The truncation error runs the *opposite* way to the pushforward case and
        is just as damaging. Directions outside span(V) get 1/tau_j, the maximum
        admissible variance, so a rank-k Gauss-Newton covariance over-predicts
        the complement rather than under-predicting it. With `tail_trace` the
        floor becomes 1/(mean discarded s^2 + tau_j), which is what the dense
        object actually assigns there.
        """
        s2 = (svals.to(V.dtype) ** 2)
        tau_j = float(tau_rel * float(s2.mean().clamp(min=EPS_EIG)))
        if tail_trace is None:
            tail_mean = 0.0
        else:
            n_perp = max(mean.numel() - V.shape[1], 1)
            tail_mean = max(float(tail_trace) / n_perp, 0.0)
        tau = 1.0 / (tail_mean + tau_j)
        d = 1.0 / (s2 + tau_j) - tau
        return cls(mean, V, d, tau, alpha=alpha, label=label)

    @classmethod
    def from_dense_covariance(cls, mean: torch.Tensor, Sigma: torch.Tensor, k: int,
                              alpha: float = 1.0, label: str = "dense") -> "LowRankGaussian":
        """Best rank-k-plus-isotropic fit to a dense Sigma. Test/reference only.

        The isotropic floor takes the mean of the discarded eigenvalues, which
        is the maximum-likelihood choice for the complement under a probabilistic
        PCA reading and preserves the trace.
        """
        evals, evecs = torch.linalg.eigh(Sigma)
        evals, evecs = evals.flip(0), evecs.flip(1)         # descending
        k = min(k, evals.numel() - 1)
        tail = evals[k:].clamp(min=EPS_EIG)
        tau = float(tail.mean())
        return cls(mean, evecs[:, :k].contiguous(), evals[:k] - tau, tau,
                   alpha=alpha, label=label)

    # ---- core quantities, all O(N k) ------------------------------------
    def _resid(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.mean.dtype).reshape(-1, self.N) - self.mean

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """U^T (x - mean), shape (B, k)."""
        return self._resid(x) @ self.U

    def quad(self, x: torch.Tensor) -> torch.Tensor:
        """(x-mean)^T S^{-1} (x-mean), alpha excluded. Batched over rows.

        Woodbury with orthonormal U:
            S^{-1} = (1/tau) [ I - U diag(d/(d+tau)) U^T ]
        """
        r = self._resid(x)
        p = r @ self.U
        w = self.d / (self.d + self.tau)
        return ((r * r).sum(-1) - (w * p * p).sum(-1)) / self.tau

    def mahalanobis_sq(self, x: torch.Tensor) -> torch.Tensor:
        return self.quad(x) / self.alpha

    def logdet_cov(self) -> float:
        """log det(alpha * S), by the matrix determinant lemma."""
        return float(torch.log(self.d + self.tau).sum()
                     + (self.N - self.k) * math.log(self.tau)
                     + self.N * math.log(self.alpha))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (self.mahalanobis_sq(x) + self.logdet_cov()
                       + self.N * math.log(2 * math.pi))

    def grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """-Sigma^{-1} (x - mean). The gradient a sampler needs, matrix-free."""
        r = self._resid(x)
        p = r @ self.U
        w = self.d / (self.d + self.tau)
        return -(r - (p * w) @ self.U.T) / (self.tau * self.alpha)

    def marginal_var(self) -> torch.Tensor:
        """diag(Sigma), (N,). O(N k)."""
        v = self.alpha * (self.tau + (self.U * self.U) @ self.d)
        return v.clamp(min=EPS_EIG)

    def marginal_std(self) -> torch.Tensor:
        return self.marginal_var().sqrt()

    def trace(self) -> float:
        return float(self.alpha * (self.d.sum() + self.N * self.tau))

    def sample(self, n: int = 1, generator=None) -> torch.Tensor:
        """Exact draws via the symmetric square root, O(N k) per draw.

        S^{1/2} = U diag(sqrt(d+tau)) U^T + sqrt(tau) (I - U U^T), so
        S^{1/2} z = sqrt(tau) z + U [ (sqrt(d+tau) - sqrt(tau)) * (U^T z) ].
        """
        z = randn((n, self.N), self.mean.dtype, self.mean.device, generator)
        c = torch.sqrt(self.d + self.tau) - math.sqrt(self.tau)
        w = math.sqrt(self.tau) * z + ((z @ self.U) * c) @ self.U.T
        return self.mean + math.sqrt(self.alpha) * w

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """S^{-1/2} (x - mean) / sqrt(alpha). Componentwise N(0,1) if the model holds."""
        r = self._resid(x)
        p = r @ self.U
        c = 1.0 / torch.sqrt(self.d + self.tau) - 1.0 / math.sqrt(self.tau)
        w = r / math.sqrt(self.tau) + (p * c) @ self.U.T
        return w / math.sqrt(self.alpha)

    # ---- streaming ------------------------------------------------------
    def summarize(self, y: torch.Tensor, extra: dict | None = None) -> QuadSummary:
        """Reduce one state to the few numbers downstream fits need."""
        r = self._resid(y).reshape(-1)
        p = (r @ self.U)
        return QuadSummary(
            r_sq=float((r * r).sum()),
            proj=p.double().cpu().numpy(),
            d=self.d.double().cpu().numpy(),
            tau=self.tau,
            N=self.N,
            extra=extra or {},
        )

    # ---- manipulation ---------------------------------------------------
    def rescaled(self, alpha: float) -> "LowRankGaussian":
        return LowRankGaussian(self.mean, self.U, self.d, self.tau, alpha=alpha,
                               label=self.label)

    def truncated(self, k: int) -> "LowRankGaussian":
        """Keep the k largest-|d| directions; fold the rest into the floor.

        Used by the rank ablation. The discarded mass is added to tau so that
        the trace is preserved, which keeps the comparison across k honest --
        otherwise smaller k would look better simply by being sharper.
        """
        if k >= self.k:
            return self
        order = torch.argsort(self.d.abs(), descending=True)[:k]
        dropped = float(self.d.sum() - self.d[order].sum())
        # Spread the discarded mass over the complement only: tau applies to all
        # N directions, so raising it by delta must be compensated on the k kept
        # ones. Without that compensation the trace grows by k*delta.
        delta = dropped / max(self.N - k, 1)
        return LowRankGaussian(self.mean, self.U[:, order].contiguous(),
                               self.d[order] - delta, max(self.tau + delta, EPS_EIG),
                               alpha=self.alpha, label=self.label)

    def to(self, device=None, dtype=None) -> "LowRankGaussian":
        return LowRankGaussian(self.mean.to(device=device, dtype=dtype),
                               self.U.to(device=device, dtype=dtype),
                               self.d.to(device=device, dtype=dtype),
                               self.tau, alpha=self.alpha, label=self.label)

    def dense_covariance(self) -> torch.Tensor:
        """Materialize Sigma. Test/reference only -- O(N^2), never call at scale."""
        if self.N > 4096:
            raise RuntimeError(
                f"refusing to materialize a {self.N}x{self.N} covariance; "
                "this method exists only to check the low-rank algebra at small N")
        eye = torch.eye(self.N, dtype=self.U.dtype, device=self.U.device)
        return self.alpha * ((self.U * self.d) @ self.U.T + self.tau * eye)

    def __repr__(self):
        return (f"LowRankGaussian(N={self.N}, k={self.k}, tau={self.tau:.3e}, "
                f"alpha={self.alpha:.4g}, label={self.label!r})")


# ---------------------------------------------------------------------------
# Combining low-rank factors
# ---------------------------------------------------------------------------

def merge_lowrank(factors: list[torch.Tensor], k: int,
                  tau: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compress sum_i A_i A_i^T + tau I into rank k plus an isotropic floor.

    `factors` are (N, m_i) matrices whose outer products sum to the target.
    Concatenating them gives A = [A_1 ... A_p] with sum = A A^T, whose leading
    left singular vectors are obtained from a thin QR of A followed by a small
    SVD -- no N x N object appears.

    Returns (U, d, tau_new). The truncated tail is folded into `tau_new` so the
    trace is preserved; this is what stops repeated propagation steps from
    silently losing variance.
    """
    keep = [f for f in factors if f.numel() > 0]
    if not keep:
        # Every factor was empty: the target is tau I, which is already in the
        # representation. Reached whenever a rollout starts from an isotropic
        # Sigma_0 with the complement term disabled.
        ref = factors[0] if factors else torch.zeros(0)
        return (torch.zeros(ref.shape[0] if factors else 0, 0, dtype=ref.dtype,
                            device=ref.device),
                torch.zeros(0, dtype=ref.dtype, device=ref.device), tau)
    A = torch.cat(keep, dim=1)
    N = A.shape[0]
    Q, R = torch.linalg.qr(A, mode="reduced")               # A = Q R, Q: (N, m)
    Ur, S, _ = torch.linalg.svd(R, full_matrices=False)     # R R^T = Ur S^2 Ur^T
    kk = min(k, S.numel())
    U = (Q @ Ur[:, :kk]).contiguous()
    dropped = float((S[kk:] ** 2).sum())
    # As in LowRankGaussian.truncated: tau covers all N directions, so the mass
    # folded into it has to come back off the k retained ones for the trace to
    # be conserved. Trace conservation is what keeps repeated propagation steps
    # from bleeding variance (or manufacturing it) purely through truncation.
    delta = dropped / max(N - kk, 1)
    d = S[:kk] ** 2 - delta
    return U, d, tau + delta
