"""Curvature estimators at scale -- the low-rank counterpart of `hipp/curvature.py`.

Same taxonomy, same names, same `tau_rel` convention, so a result here is
comparable to the 1D result. What changes is that nothing dense is ever formed:
each estimator returns a `LowRankGaussian` built from O(k) network passes.

Which estimators survive the move
---------------------------------
    gn                   yes -- exact rewrite of (J^T J + tau I)^{-1}, rank-k truncated
    pushforward          yes -- JJ^T + tau I, rank-k truncated
    lowrank_*            these *are* the scaled versions; the names are kept
                         for continuity with the 1D results tables
    diag_gn              yes -- Hutchinson, no basis needed
    fd_pushforward       yes -- finite-difference probes, for black-box models
    identity             yes -- trivially

    fd_gn                dropped. It needs J-hat = D V^+ with m >= N probes to be
                         anything other than noise, i.e. the full N passes the
                         low-rank path exists to avoid. Requesting it raises
                         rather than silently returning a rank-deficient object.

    dircov               dropped, for a reason worth stating: it differs from
                         fd_pushforward only by a global constant (the probe
                         count), and every method here goes through the same
                         one-parameter scale calibration, which absorbs exactly
                         that constant. At scale the two are the *same
                         estimator*. In the 1D code they were listed separately
                         and produced near-identical numbers; that was not a
                         coincidence and should not be reported as two results.

The one genuinely new parameter is `k`. It is not a nuisance: the rank is the
statement of how much of the operator's geometry the method claims to need, and
`s4_scaling_ablation.py` sweeps it. `s0_validate_lowrank.py` measures, at a
resolution where the exact object is still computable, how much is lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .jacobian import (JacobianOps, fd_jvp_operator, frobenius_norm_estimate,
                       hutchinson_diag_gramian, jacobian_ops, randomized_svd)
from .lowrank import LowRankGaussian, randn

ESTIMATORS_SCALE: dict[str, str] = {
    "pushforward": "Sigma = J J^T + tau I, rank-k randomized SVD",
    "gn": "Sigma = (J^T J + tau I)^{-1}, rank-k randomized SVD",
    "gn_soft": "Gauss--Newton covariance retaining soft J^T J directions by shifted inverse iteration",
    "diag_gn": "diag(J^T J) by Hutchinson probes; no basis",
    "fd_pushforward": "pushforward from forward-only finite-difference probes",
    "identity": "isotropic control (no curvature information)",
}


def _gram_mm(ops: JacobianOps, V: torch.Tensor, chunk: int) -> torch.Tensor:
    """Matrix-free (J^T J)V using batched JVPs and VJPs."""
    return ops.rmm(ops.mm(V, chunk=chunk), chunk=chunk)


def _shifted_cg(ops: JacobianOps, B: torch.Tensor, shift: float, *, n_iter: int,
                chunk: int) -> torch.Tensor:
    """Independent batched CG solves of (J^T J + shift I)X=B."""
    x = torch.zeros_like(B)
    r = p = B.clone()
    rr = (r * r).sum(0)
    for _ in range(n_iter):
        ap = _gram_mm(ops, p, chunk) + shift * p
        a = rr / (p * ap).sum(0).clamp(min=1e-30)
        x, r = x + p * a, r - ap * a
        rr_new = (r * r).sum(0)
        p, rr = r + p * (rr_new / rr.clamp(min=1e-30)), rr_new
    return x


def _soft_gram_eigenpairs(ops: JacobianOps, *, k: int, oversample: int,
                          inverse_iters: int, cg_iters: int, shift: float,
                          chunk: int, generator=None):
    """Smallest J^T J eigendirections via shifted inverse subspace iteration."""
    ell = min(k + oversample, ops.N)
    q, _ = torch.linalg.qr(randn((ops.N, ell), ops.dtype, ops.device, generator).double())
    for _ in range(inverse_iters):
        x = _shifted_cg(ops, q.to(ops.dtype), shift, n_iter=cg_iters, chunk=chunk)
        q, _ = torch.linalg.qr(x.double())
    aq = _gram_mm(ops, q.to(ops.dtype), chunk).double()
    values, vectors = torch.linalg.eigh(q.T @ aq)
    values, vectors = values.clamp(min=0)[:k], vectors[:, :k]
    V = (q @ vectors).contiguous()
    # A small Ritz residual is the matrix-free convergence certificate; it is
    # essential because an inexpensive smoke setting may not resolve a nearly
    # flat spectrum well enough to interpret downstream alignment statistics.
    residual = (aq @ vectors - V * values).norm(dim=0) / \
               (aq @ vectors).norm(dim=0).clamp(min=1e-30)
    return V, values.contiguous(), residual.contiguous()


@dataclass
class ScaleEstimate:
    """A low-rank curvature estimate plus the spectral bookkeeping."""
    prior: LowRankGaussian
    mean: torch.Tensor
    U: torch.Tensor | None = None
    S: torch.Tensor | None = None
    V: torch.Tensor | None = None
    method: str = ""
    meta: dict = field(default_factory=dict)

    def rescaled(self, alpha: float) -> LowRankGaussian:
        return self.prior.rescaled(alpha)


def estimate_scale(fm, c: torch.Tensor, method: str = "pushforward",
                   k: int = 64, tau_rel: float = 1e-2, oversample: int = 10,
                   n_iter: int = 2, chunk: int = 8, eps: float = 1e-3,
                   n_probe: int = 64, n_tail: int = 16,
                   past: torch.Tensor | None = None,
                   generator=None, ops: JacobianOps | None = None,
                   soft_inverse_iters: int = 2, soft_cg_iters: int = 24,
                   soft_shift_rel: float = 0.1) -> ScaleEstimate:
    """Curvature at conditioning state `c` for a `FrozenFM`.

    Cost, in network passes:
        pushforward / gn   (2 + 3*n_iter) * (k + oversample) + n_tail
        diag_gn            2 * n_probe
        fd_pushforward     2 * (k + oversample) * (1 + n_iter)   (forward only)
        identity           1

    `ops` lets a caller share one set of J operators across several estimators
    at the same state, which is what makes sweeping the zoo affordable -- the
    VJP graph is built once.
    """
    c = c.reshape(-1).detach()
    N = c.numel()
    f = fm.flat_fn(past)
    with torch.no_grad():
        mean = f(c).detach()
    meta = {"method": method, "tau_rel": tau_rel, "k": k, "N": N}

    if method == "identity":
        # tau is arbitrary here: alpha absorbs it exactly. Fixed at 1 so the
        # fitted alpha is directly the per-dimension error variance.
        prior = LowRankGaussian.isotropic(mean, tau=1.0, label="identity")
        return ScaleEstimate(prior, mean, method=method, meta=meta)

    if method == "diag_gn":
        ops = ops or jacobian_ops(f, c)
        d = hutchinson_diag_gramian(ops, n_probe=n_probe, chunk=chunk,
                                    generator=generator).double()
        # Sigma = diag(1/(d + tau_j)) has no low-rank structure, so it is carried
        # as a rank-k object on the k *most extreme* coordinates plus a floor at
        # the median. That is an approximation the dense code did not need; it is
        # reported as such and is why diag_gn is a weak baseline at scale.
        tau_j = float(tau_rel * d.mean().clamp(min=1e-30))
        var = 1.0 / (d + tau_j)
        floor = float(var.median())
        idx = torch.topk((var - floor).abs(), min(k, N)).indices
        U = torch.zeros(N, idx.numel(), dtype=mean.dtype, device=mean.device)
        U[idx, torch.arange(idx.numel(), device=mean.device)] = 1.0
        prior = LowRankGaussian(mean, U, (var[idx] - floor).to(mean.dtype), floor,
                                label="diag_gn")
        meta["n_probe"] = n_probe
        return ScaleEstimate(prior, mean, method=method, meta=meta)

    if method == "gn_soft":
        ops = ops or jacobian_ops(f, c)
        fro = frobenius_norm_estimate(ops, n_probe=max(n_tail, 8), chunk=chunk,
                                      generator=generator)
        bulk = max(fro / N, 1e-30)
        shift = max(float(soft_shift_rel) * bulk, 1e-12)
        Vsoft, lam, ritz_residual = _soft_gram_eigenpairs(
            ops, k=k, oversample=oversample, inverse_iters=soft_inverse_iters,
            cg_iters=soft_cg_iters, shift=shift, chunk=chunk, generator=generator)
        tau_j = max(float(tau_rel) * bulk, 1e-30)
        floor = 1.0 / (bulk + tau_j)
        soft_var = 1.0 / (lam + tau_j)
        mean64 = mean.to(torch.float64)
        prior = LowRankGaussian(mean64, Vsoft, (soft_var - floor).clamp(min=0),
                                floor, label="gn_soft")
        meta.update({"frobenius_sq": fro, "bulk_gram_eigenvalue": bulk,
                     "soft_eigenvalues": lam.cpu(),
                     "soft_ritz_residual_rel": ritz_residual.cpu(), "soft_shift": shift,
                     "soft_cg_iters": soft_cg_iters,
                     "soft_inverse_iters": soft_inverse_iters,
                     "jvp_calls": ops.n_calls[0], "vjp_calls": ops.n_calls[1]})
        return ScaleEstimate(prior, mean64, V=Vsoft, method=method, meta=meta)

    if method == "fd_pushforward":
        fd = fd_jvp_operator(f, c, eps=eps)
        ops = JacobianOps(matvec=fd, rmatvec=_no_vjp, N=N, dtype=c.dtype,
                          device=c.device, n_calls=[0, 0])
        U, S = _range_finder_svd(ops, k=k, oversample=oversample, n_iter=n_iter,
                                 chunk=chunk, generator=generator)
        meta["eps"] = eps
        V = None
    else:
        ops = ops or jacobian_ops(f, c)
        U, S, V = randomized_svd(ops, k=k, oversample=oversample, n_iter=n_iter,
                                 chunk=chunk, generator=generator)

    U = U.to(torch.float64)
    S = S.to(torch.float64)
    mean64 = mean.to(torch.float64)

    # The discarded spectral mass sets the isotropic floor, and at rank k << N
    # that floor covers almost every direction -- see LowRankGaussian.
    # from_pushforward for why inheriting the 1D tau_rel instead is wrong.
    # ||J||_F^2 costs n_tail extra JVPs; the captured part is already known.
    tail_trace = None
    if n_tail > 0:
        fro = frobenius_norm_estimate(ops, n_probe=n_tail, chunk=chunk,
                                      generator=generator)
        tail_trace = max(fro - float((S ** 2).sum()), 0.0)
        meta["frobenius_sq"] = fro
        meta["tail_trace"] = tail_trace
        meta["captured_trace_frac"] = float((S ** 2).sum()) / max(fro, 1e-300)

    if method in ("pushforward", "fd_pushforward"):
        prior = LowRankGaussian.from_pushforward(mean64, U, S, tau_rel=tau_rel,
                                                 label=method, tail_trace=tail_trace)
    elif method == "gn":
        if V is None:
            raise ValueError("gn needs the right singular vectors; it is not "
                             "available from forward-only probes")
        prior = LowRankGaussian.from_gauss_newton(mean64, V.to(torch.float64), S,
                                                  tau_rel=tau_rel, label=method,
                                                  tail_trace=tail_trace)
    elif method == "fd_gn":
        raise ValueError(
            "fd_gn is not available at scale: recovering J^T J from probes needs "
            "m >= N of them, which is the O(N) cost the low-rank path exists to "
            "avoid. Use 'gn' (needs AD) or 'fd_pushforward' (forward-only).")
    else:
        raise ValueError(f"unknown estimator {method!r}; "
                         f"choices: {list(ESTIMATORS_SCALE)}")

    meta["jvp_calls"], meta["vjp_calls"] = ops.n_calls
    meta["svals"] = S.cpu()
    return ScaleEstimate(prior, mean64, U=U, S=S, V=V, method=method, meta=meta)


def _no_vjp(w):                                          # pragma: no cover
    raise RuntimeError("no adjoint available for finite-difference probes")


def _range_finder_svd(ops: JacobianOps, k: int, oversample: int, n_iter: int,
                      chunk: int, generator=None):
    """Leading left singular subspace of J using forward products only.

    With no adjoint, power iteration on J^T J is unavailable; instead the range
    is refined by re-projecting the sample matrix, which converges more slowly
    but needs nothing but f. Returns (U, S) with S from the Gram matrix of the
    sketch, so the singular values are those of the *projected* operator -- an
    underestimate whose size `s0_validate_lowrank.py` reports.
    """
    N = ops.N
    ell = min(k + oversample, N)
    Omega = randn((N, ell), ops.dtype, ops.device, generator)
    Y = ops.mm(Omega, chunk=chunk)
    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Y.double())
        Y = ops.mm(Q.to(ops.dtype), chunk=chunk)
    Uy, S, _ = torch.linalg.svd(Y.double(), full_matrices=False)
    # Y = J Omega with Omega ~ N(0, I) has E[Y Y^T] = J J^T * ell, so the
    # singular values of Y overestimate those of J by sqrt(ell).
    return Uy[:, :k].contiguous(), (S[:k] / (ell ** 0.5)).contiguous()


# ---------------------------------------------------------------------------
# Spectral diagnostics
# ---------------------------------------------------------------------------

def spectrum_stats_scale(est: ScaleEstimate) -> dict:
    """Scalar summaries of the retained spectrum.

    Every quantity here is over the rank-k subspace only, and is labelled `_k`
    accordingly: `trace` and `logdet` of the full covariance are dominated by
    the (N-k)-dimensional isotropic floor, so reporting them unqualified would
    be reporting tau. This is a real difference from the 1D tables and the
    reason Stage 1's Q2 correlations must be re-derived rather than transferred.
    """
    p = est.prior
    d = p.d.double()
    eig = (d + p.tau).clamp(min=1e-300)
    frac = eig / eig.sum()
    out = {
        "k": p.k, "tau": p.tau, "N": p.N,
        "trace_full": p.trace() / p.alpha,
        "trace_k": float(d.sum()),
        "trace_frac_in_k": float(d.sum() / (d.sum() + p.N * p.tau)),
        "logdet_full": float(torch.log(eig).sum()) + (p.N - p.k) * float(
            torch.log(torch.tensor(p.tau, dtype=torch.float64))),
        "eig_max_k": float(eig.max()), "eig_min_k": float(eig.min()),
        "cond_k": float(eig.max() / eig.min()),
        "eff_rank_k": float(torch.exp(-(frac * frac.clamp(min=1e-300).log()).sum())),
        "n_modes_90pct": int((torch.cumsum(frac.flip(0), 0) < 0.9).sum()) + 1,
    }
    if est.S is not None:
        s = est.S.double()
        out.update(smax=float(s.max()), smin=float(s.min()), s_mean=float(s.mean()),
                   cond_J_k=float(s.max() / s.min().clamp(min=1e-30)))
    return out


def spectral_content(U: torch.Tensor, n: int, n_bins: int = 16) -> torch.Tensor:
    """Where each curvature direction lives in wavenumber space, (k, n_bins).

    The 2D form of "do the top eigenvectors match the physically amplified
    modes". In 1D this was a list of |k| values; on a 2D grid it has to be an
    isotropically binned spectrum per direction.
    """
    k = U.shape[1]
    fields = U.T.reshape(k, n, n)
    fh = torch.fft.rfft2(fields.to(torch.float64))
    kx = torch.fft.fftfreq(n, device=U.device).view(-1, 1) * n
    ky = torch.fft.rfftfreq(n, device=U.device).view(1, -1) * n
    kk = (kx ** 2 + ky ** 2).sqrt()
    idx = (kk / float(kk.max()) * (n_bins - 1)).round().long().clamp(0, n_bins - 1)
    power = (fh.abs() ** 2).reshape(k, -1)
    out = torch.zeros(k, n_bins, dtype=torch.float64, device=U.device)
    out.scatter_add_(1, idx.reshape(1, -1).expand(k, -1), power)
    return out / out.sum(dim=1, keepdim=True).clamp(min=1e-300)


def dominant_wavenumber(U: torch.Tensor, n: int, n_bins: int = 16) -> torch.Tensor:
    """Peak wavenumber bin of each curvature direction, (k,)."""
    return spectral_content(U, n, n_bins).argmax(dim=1)
