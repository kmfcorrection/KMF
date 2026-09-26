"""Uncertainty propagation along an autoregressive rollout.

This is the experiment the 1D study could set up but not run, and it is where
the method's central claim is actually testable.

The theory (docs/method.md 2.1b, background.md 2) says the predictive covariance
of a deterministic model comes from uncertainty in the state it conditions on,
pushed through the linearization:

    Sigma_{t+1} = J_t Sigma_t J_t^T + sigma^2 I                        (*)

Under teacher forcing Sigma_c = 0 and (*) collapses to sigma^2 I, which predicts
*no curvature dependence at all* -- the regime in which the original Stage 1 was
mistakenly run. During autoregressive rollout the model consumes its own output,
so Sigma_t is genuinely non-zero and grows, and (*) has content. A single-step
study can only inject synthetic input noise to stand in for this; a rollout
study measures the real thing.

Keeping (*) tractable
---------------------
Sigma_t is N x N and (*) is a dense recursion, so at N = 16,384 it cannot be
run literally. Written in the low-rank-plus-isotropic representation with
Sigma_t = U_t diag(e_t) U_t^T + tau_t (I - U_t U_t^T), e_t > 0:

    J Sigma_t J^T = (J U_t) diag(e_t) (J U_t)^T  +  tau_t * J P_perp J^T

The first term is exact and costs k JVPs. The second is the mass in the
(N-k)-dimensional complement being pushed forward; it is sketched by drawing
random vectors, projecting out span(U_t) and applying J, which is what keeps
new directions from being permanently invisible once they leave the retained
subspace. The sum is compressed back to rank k by a thin QR.

Trace conservation is enforced exactly rather than approximately. Every step
computes the true trace of the target -- tr = sum_i e_i ||J u_i||^2 +
tau_t * tr(J P_perp J^T) + N sigma^2, all available from products already
computed -- and sets the new floor so the compressed object matches it. Without
this the recursion bleeds variance through truncation at every step and the
propagated uncertainty drifts down for purely numerical reasons, which would
look exactly like the method under-predicting rollout error.

What sigma^2 is
---------------
The per-step injected variance: the model's irreducible one-step error, the part
not explained by input uncertainty. It is fitted once on held-out one-step
residuals (`fit_sigma2`) and is the *only* free parameter of the recursion
beyond the initial condition. It plays the role alpha plays in the single-step
study, and like alpha it sets a scale rather than a shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .jacobian import batched_jacobian_apply, jacobian_ops
from .lowrank import LowRankGaussian, merge_lowrank, randn


# ---------------------------------------------------------------------------
# One propagation step
# ---------------------------------------------------------------------------

def propagate_step(fm, c: torch.Tensor, prior: LowRankGaussian, sigma2: float,
                   k: int | None = None, n_sketch: int | None = None, n_probe: int = 8, chunk: int = 8,
                   generator=None, complement: str = "nystrom",
                   past: torch.Tensor | None = None,
                   innovation: LowRankGaussian | None = None) -> tuple[LowRankGaussian, dict]:
    """Advance one step of (*): returns N(f(c), Sigma_{t+1}) in low-rank form.

    Parameters
    ----------
    prior : the current Sigma_t. Its `alpha` is folded into the eigenvalues here
            so the returned object carries alpha = 1 and an absolute covariance;
            the recursion has no free scale left once sigma2 is fixed.
    complement : how to handle tau_t * J P_perp J^T, the mass outside the
        retained subspace being pushed forward. This term is what lets a
        direction that was not in span(U_t) enter the covariance, so dropping it
        makes the retained basis permanently blind to newly growing modes.

        "nystrom"   randomized eigendecomposition of M M^T with M = J P_perp:
                    sketch, orthogonalize, then form the exact projected
                    Gram matrix. Costs n_sketch JVPs + n_sketch VJPs and is
                    *exact* once n_sketch reaches the rank of M. Default.
        "sketch"    the raw Monte-Carlo estimate (1/m) Y Y^T with Y = M Omega.
                    Half the cost, no adjoint needed -- but its eigenvalues
                    carry O(1/sqrt(m)) noise, which at m ~ k is large enough to
                    dominate the covariance *shape*. Kept for the ablation and
                    for black-box models with no adjoint; not recommended.
        "isotropic" replace the term by its trace spread evenly. Correct when J
                    is near-scalar on the complement, wrong when the complement
                    holds the growing modes -- which for a chaotic flow is
                    exactly where they are. The cheap control.
        "none"      drop it entirely. Only for measuring what it contributes.
    """
    c = c.reshape(-1).detach()
    k = k or prior.k
    # The complement sketch has to be at least as wide as the rank being
    # retained, or the compressed object cannot be filled to rank k and the
    # recursion loses rank monotonically.
    n_sketch = n_sketch or (k + 10)
    f = fm.flat_fn(past)
    ops = jacobian_ops(f, c)
    with torch.no_grad():
        mean_next = f(c).detach().double()

    U = prior.U.to(ops.dtype)
    e = (prior.d + prior.tau).clamp(min=0) * prior.alpha        # (k,) > 0
    tau_t = prior.tau * prior.alpha

    # --- term 1: the retained subspace, exact -----------------------------
    JU = ops.mm(U, chunk=chunk) if prior.k else U
    A1 = JU.double() * e.double().sqrt()
    trace_1 = float((A1 ** 2).sum())

    # --- term 2: the complement -------------------------------------------
    factors = [A1]
    trace_2 = 0.0
    if complement != "none" and tau_t > 0:
        Om = randn((ops.N, n_sketch), ops.dtype, ops.device, generator)
        if prior.k:
            Om = Om - U @ (U.T @ Om)                # project onto span(U)^perp
        Y = ops.mm(Om, chunk=chunk).double()
        if complement == "nystrom":
            F, captured = _complement_eigenfactor(ops, U if prior.k else None, Y, chunk)
            # The captured mass is exact; only what falls outside range(Q) needs
            # estimating, and it must be estimated with *fresh* probes -- Q is
            # built from Y, so (I - QQ^T)Y is identically zero and reusing Y
            # would report a residual of exactly 0 no matter the true rank.
            residual = _residual_trace(ops, U if prior.k else None, F, n_probe,
                                       chunk, generator)
            trace_2 = tau_t * (captured + residual)
            factors.append(F * math.sqrt(tau_t))
        else:
            # tr(M M^T) = E[||M om||^2] for om ~ N(0, I).
            trace_2 = tau_t * float((Y ** 2).sum()) / n_sketch
            if complement == "sketch":
                factors.append(Y * math.sqrt(tau_t / n_sketch))

    # --- add innovation, then compress and conserve trace -----------------
    # A structured one-step innovation Q may replace the conventional scalar
    # sigma^2 I.  Its low-rank directions and isotropic tail are added before
    # the common rank-k compression, so trace conservation still holds.
    if innovation is None:
        innovation_trace = ops.N * sigma2
    else:
        if innovation.N != ops.N:
            raise ValueError("innovation covariance has incompatible dimension")
        if innovation.k:
            factors.append(innovation.U.double() *
                           (innovation.alpha * innovation.d).clamp(min=0).sqrt().double())
        innovation_trace = innovation.trace()
    Unew, dnew, _ = merge_lowrank(factors, k=k, tau=0.0)
    target_trace = trace_1 + trace_2 + innovation_trace
    tau_new = (target_trace - float(dnew.sum())) / ops.N
    if tau_new <= 0:
        # Can only happen if the retained rank already accounts for more than
        # the whole trace, i.e. numerical round-off at very small sigma2.
        tau_new = max(sigma2 if innovation is None else innovation.alpha * innovation.tau, 1e-30)
        dnew = dnew * (target_trace - ops.N * tau_new) / float(dnew.sum().clamp(min=1e-300))
    dnew = dnew.clamp(min=-tau_new * (1 - 1e-9))

    out = LowRankGaussian(mean_next, Unew.to(torch.float64), dnew.to(torch.float64),
                          float(tau_new), alpha=1.0, label=prior.label)
    info = {"trace_retained": trace_1, "trace_complement": trace_2,
            "trace_injected": innovation_trace, "trace_total": target_trace,
            "tau": float(tau_new), "k": out.k,
            "jvp_calls": ops.n_calls[0], "vjp_calls": ops.n_calls[1],
            "frac_trace_in_rank": float(dnew.sum()) / max(target_trace, 1e-300)}
    return out, info


def _complement_eigenfactor(ops, U, Y: torch.Tensor, chunk: int):
    """Factor F with F F^T ~= M M^T, M = J P_perp, from a sketch Y = M Omega.

    Q = orth(Y) is a basis for the sampled range of M. Projecting,

        M M^T ~= Q (Q^T M)(Q^T M)^T Q^T = Q (B^T B) Q^T,   B = M^T Q,

    and eigendecomposing the small (l x l) matrix B^T B = W L W^T gives
    F = Q W sqrt(L). Unlike the raw sketch this reproduces the *eigenvalues* of
    M M^T rather than a Monte-Carlo estimate of them, and it is exact once
    range(Q) contains range(M) -- no power iterations are needed for that, since
    the projection is onto the sampled range itself.

    Costs one VJP block on top of the JVP block already spent on Y. Returns
    (F, captured_trace) with captured_trace = ||B||_F^2 = tr(Q^T M M^T Q) exact.
    """
    Q, _ = torch.linalg.qr(Y)                            # (N, l)
    B = ops.rmm(Q.to(ops.dtype), chunk=chunk).double()   # (N, l) = J^T Q
    if U is not None:
        B = B - U.double() @ (U.double().T @ B)          # M^T = P_perp J^T
    G = B.T @ B                                          # (l, l)
    evals, W = torch.linalg.eigh(0.5 * (G + G.T))
    evals = evals.clamp(min=0)
    return (Q @ W) * evals.sqrt(), float(evals.sum())


def _residual_trace(ops, U, F: torch.Tensor, n_probe: int, chunk: int,
                    generator=None) -> float:
    """Unbiased estimate of tr(M M^T) - tr(F F^T): the mass F does not capture.

    Uses probes independent of the ones that built F, which is the whole point:
    E ||(I - PP^T) M om||^2 = ||(I - PP^T) M||_F^2 for om ~ N(0, I), and with
    P the orthonormal basis of range(F) this is exactly the uncaptured trace.
    Returns 0 (up to noise) when F already spans the range, so the total trace
    stays consistent with the captured eigenvalues rather than fighting them.
    """
    if n_probe <= 0:
        return 0.0
    N = ops.N
    Om = randn((N, n_probe), ops.dtype, ops.device, generator)
    if U is not None:
        Om = Om - U @ (U.T @ Om)
    Y = ops.mm(Om, chunk=chunk).double()
    if F.numel():
        P, _ = torch.linalg.qr(F)
        Y = Y - P @ (P.T @ Y)
    return max(float((Y ** 2).sum()) / n_probe, 0.0)


# ---------------------------------------------------------------------------
# Full rollout
# ---------------------------------------------------------------------------

@dataclass
class RolloutTrace:
    """Everything one propagated rollout produced, ready to aggregate."""
    steps: int
    means: list = field(default_factory=list)        # predicted state per step
    priors: list = field(default_factory=list)       # LowRankGaussian per step
    info: list = field(default_factory=list)


def propagate_rollout(fm, c0: torch.Tensor, steps: int, sigma2: float,
                      k: int = 64, prior0: LowRankGaussian | None = None,
                      n_sketch: int | None = None, n_probe: int = 8, chunk: int = 8, generator=None,
                      complement: str = "nystrom", keep_priors: bool = True,
                      progress: bool = False,
                      innovation: LowRankGaussian | None = None) -> RolloutTrace:
    """Run (*) for `steps` autoregressive steps from a known initial state.

    `prior0` is Sigma_0. When it is None, the initial state is known exactly:
    Sigma_0 = 0 and the first forecast covariance is therefore sigma^2 I.
    Everything after that propagates the accumulated forecast uncertainty.

    Memory: with `keep_priors` every step holds an (N, k) basis. At N = 16,384,
    k = 64 and 20 steps that is 168 MB in float64 -- fine for one trajectory,
    not for a batch, which is why the experiment loops over states.
    """
    c = c0.reshape(-1).detach()
    tr = RolloutTrace(steps=steps)
    start = 0
    if prior0 is None and steps > 0:
        with torch.no_grad():
            mean1 = fm.predict(c).detach().double()
        if innovation is None:
            prior = LowRankGaussian.isotropic(mean1, tau=sigma2, label="rollout")
            injected_trace = fm.N * sigma2
        else:
            prior = LowRankGaussian(mean1.double(), innovation.U, innovation.d,
                                    innovation.tau, alpha=innovation.alpha,
                                    label="rollout+structured-Q")
            injected_trace = innovation.trace()
        tr.means.append(mean1.to(c.dtype).clone())
        tr.priors.append(prior if keep_priors else None)
        tr.info.append({
            "step": 1, "trace_retained": 0.0, "trace_complement": 0.0,
            "trace_injected": injected_trace, "trace_total": injected_trace,
            "tau": prior.alpha * prior.tau, "k": prior.k, "jvp_calls": 0, "vjp_calls": 0,
            "frac_trace_in_rank": float((prior.alpha * prior.d).sum()) / max(injected_trace, 1e-300),
        })
        c = mean1.to(c.dtype)
        start = 1
        if progress:
            print(f"      step 1/{steps}  trace {injected_trace:.4e}  "
                  f"in-rank {tr.info[-1]['frac_trace_in_rank']:.3f}", flush=True)
    else:
        prior = prior0

    for t in range(start, steps):
        prior, info = propagate_step(fm, c, prior, sigma2, k=k, n_sketch=n_sketch,
                                     n_probe=n_probe, chunk=chunk, generator=generator,
                                     complement=complement, innovation=innovation)
        c = prior.mean.to(c.dtype)
        tr.means.append(c.clone())
        tr.priors.append(prior if keep_priors else None)
        info["step"] = t + 1
        tr.info.append(info)
        if progress:
            print(f"      step {t+1}/{steps}  trace {info['trace_total']:.4e}  "
                  f"in-rank {info['frac_trace_in_rank']:.3f}", flush=True)
    return tr


def _member_randn(generators, n: int, m: int, dtype, device) -> torch.Tensor:
    """Draw independent, reproducible probe blocks for each trajectory."""
    return torch.stack([randn((n, m), dtype, device, g) for g in generators])


def propagate_rollouts_batched(fm, c0: torch.Tensor, steps: int, sigma2: float,
                               *, k: int = 64, n_sketch: int | None = None,
                               n_probe: int = 8, chunk: int = 8,
                               generators=None, complement: str = "nystrom",
                               keep_priors: bool = True,
                               innovation: LowRankGaussian | None = None) -> list[RolloutTrace]:
    """Trajectory-batched equivalent of :func:`propagate_rollout`.

    All members retain separate states, random probes, covariance factors and
    low-rank compression.  The only shared operation is the *evaluation* of
    their independent JVP/VJP blocks by :func:`batched_jacobian_apply`.  It is
    therefore an execution optimization, not a coupled covariance model.

    The initial-prior-known path used by S3 is supported intentionally.  The
    ``prior0=pushforward`` ablation remains serial because each member's
    independently estimated initial prior would otherwise require a separate
    randomized SVD before the batched recursion begins.
    """
    if c0.ndim != 2:
        raise ValueError("c0 must have shape (trajectory_batch,N)")
    if not getattr(fm.module, "_hipp_batch_safe", False):
        raise RuntimeError("trajectory-batched rollout requires a batch-safe FM adapter")
    B, N = c0.shape
    if N != fm.N:
        raise ValueError(f"state width {N} != FM width {fm.N}")
    if B < 1:
        return []
    if generators is None:
        generators = [None] * B
    if len(generators) != B:
        raise ValueError("one generator per trajectory is required")
    if complement not in ("nystrom", "isotropic", "none", "sketch"):
        raise ValueError(f"unsupported complement {complement!r}")
    if complement == "sketch":
        # The raw sketch is retained as an ablation in the serial path.  The
        # batched backend is intentionally limited to the paper controls so a
        # new execution path cannot silently change that noisy estimator.
        raise ValueError("trajectory-batched rollout does not support complement='sketch'")
    n_sketch = int(n_sketch or (k + 10))
    if n_sketch < k:
        raise ValueError("n_sketch must be at least k")

    traces = [RolloutTrace(steps=steps) for _ in range(B)]
    states = c0.reshape(B, N).detach().to(fm.device, fm.dtype)
    if steps == 0:
        return traces
    means = fm.predict(states).detach().double()
    priors = []
    for b in range(B):
        if innovation is None:
            prior = LowRankGaussian.isotropic(means[b], tau=sigma2, label="rollout")
            injected = fm.N * sigma2
        else:
            prior = LowRankGaussian(means[b], innovation.U, innovation.d,
                                    innovation.tau, alpha=innovation.alpha,
                                    label="rollout+structured-Q")
            injected = innovation.trace()
        priors.append(prior)
        traces[b].means.append(means[b].to(fm.dtype).clone())
        traces[b].priors.append(prior if keep_priors else None)
        traces[b].info.append({
            "step": 1, "trace_retained": 0.0, "trace_complement": 0.0,
            "trace_injected": injected, "trace_total": injected,
            "tau": prior.alpha * prior.tau, "k": prior.k,
            "jvp_calls": 0, "vjp_calls": 0,
            "frac_trace_in_rank": float((prior.alpha * prior.d).sum()) / max(injected, 1e-300),
            "trajectory_batched": True,
        })
    states = means.to(fm.dtype)

    for t in range(1, steps):
        # Every member has rank k after the first injected covariance.  Keeping
        # that invariant explicit catches a future covariance change rather
        # than mixing incompatible batch shapes.
        ranks = {p.k for p in priors}
        if len(ranks) != 1:
            raise RuntimeError(f"trajectory-batched rollout requires equal ranks, got {sorted(ranks)}")
        rk = next(iter(ranks))
        mean_next = fm.predict(states).detach().double()
        U = torch.stack([p.U.to(fm.dtype) for p in priors])
        e = torch.stack([(p.d + p.tau).clamp(min=0).double() * p.alpha for p in priors])
        tau = torch.tensor([p.tau * p.alpha for p in priors], dtype=torch.float64,
                           device=fm.device)
        JU = batched_jacobian_apply(fm, states, U, chunk=chunk) if rk else U
        A1 = JU.double() * e.sqrt()[:, None, :]
        trace1 = A1.square().sum(dim=(1, 2))
        factors = [[A1[b]] for b in range(B)]
        trace2 = torch.zeros(B, dtype=torch.float64, device=fm.device)
        jvp_calls = torch.full((B,), rk, dtype=torch.int64, device=fm.device)
        vjp_calls = torch.zeros(B, dtype=torch.int64, device=fm.device)

        if complement != "none" and bool((tau > 0).any()):
            Om = _member_randn(generators, N, n_sketch, fm.dtype, fm.device)
            if rk:
                Om = Om - torch.einsum("bnk,bkm->bnm", U, torch.einsum("bnk,bnm->bkm", U, Om))
            Y = batched_jacobian_apply(fm, states, Om, chunk=chunk).double()
            jvp_calls += n_sketch
            if complement == "isotropic":
                trace2 = tau * Y.square().sum(dim=(1, 2)) / n_sketch
            else:
                Q = torch.stack([torch.linalg.qr(Y[b], mode="reduced")[0] for b in range(B)])
                Bmat = batched_jacobian_apply(fm, states, Q.to(fm.dtype),
                                               adjoint=True, chunk=chunk).double()
                vjp_calls += n_sketch
                if rk:
                    Bmat = Bmat - torch.einsum("bnk,bkm->bnm", U.double(),
                                                torch.einsum("bnk,bnm->bkm", U.double(), Bmat))
                F, captured = [], []
                for b in range(B):
                    G = Bmat[b].T @ Bmat[b]
                    vals, W = torch.linalg.eigh(0.5 * (G + G.T))
                    vals = vals.clamp(min=0)
                    F.append((Q[b] @ W) * vals.sqrt())
                    captured.append(vals.sum())
                F = torch.stack(F)
                ResidualOm = _member_randn(generators, N, n_probe, fm.dtype, fm.device)
                if rk:
                    ResidualOm = ResidualOm - torch.einsum(
                        "bnk,bkm->bnm", U,
                        torch.einsum("bnk,bnm->bkm", U, ResidualOm))
                RY = batched_jacobian_apply(fm, states, ResidualOm, chunk=chunk).double()
                jvp_calls += n_probe
                P = torch.stack([torch.linalg.qr(F[b], mode="reduced")[0] for b in range(B)])
                RY = RY - torch.einsum("bnk,bkm->bnm", P, torch.einsum("bnk,bnm->bkm", P, RY))
                residual = RY.square().sum(dim=(1, 2)) / max(n_probe, 1)
                trace2 = tau * (torch.stack(captured) + residual)
                for b in range(B):
                    factors[b].append(F[b] * tau[b].sqrt())

        new_priors = []
        for b in range(B):
            injected = fm.N * sigma2 if innovation is None else innovation.trace()
            if innovation is not None and innovation.k:
                factors[b].append(innovation.U.double() *
                                  (innovation.alpha * innovation.d).clamp(min=0).sqrt().double())
            Unew, dnew, _ = merge_lowrank(factors[b], k=k, tau=0.0)
            target = float(trace1[b] + trace2[b] + injected)
            tau_new = (target - float(dnew.sum())) / fm.N
            if tau_new <= 0:
                tau_new = max(sigma2 if innovation is None else innovation.alpha * innovation.tau, 1e-30)
                dnew = dnew * (target - fm.N * tau_new) / float(dnew.sum().clamp(min=1e-300))
            dnew = dnew.clamp(min=-tau_new * (1 - 1e-9))
            prior = LowRankGaussian(mean_next[b], Unew.to(torch.float64), dnew.to(torch.float64),
                                    float(tau_new), alpha=1.0, label=priors[b].label)
            new_priors.append(prior)
            traces[b].means.append(mean_next[b].to(fm.dtype).clone())
            traces[b].priors.append(prior if keep_priors else None)
            traces[b].info.append({
                "step": t + 1, "trace_retained": float(trace1[b]),
                "trace_complement": float(trace2[b]), "trace_injected": injected,
                "trace_total": target, "tau": float(tau_new), "k": prior.k,
                "jvp_calls": int(jvp_calls[b]), "vjp_calls": int(vjp_calls[b]),
                "frac_trace_in_rank": float(dnew.sum()) / max(target, 1e-300),
                "trajectory_batched": True,
            })
        priors = new_priors
        states = mean_next.to(fm.dtype)
    return traces


# ---------------------------------------------------------------------------
# The injected-variance parameter
# ---------------------------------------------------------------------------

def fit_sigma2(fm, xs, ys, max_states: int = 128) -> dict:
    """Per-dimension one-step residual variance, the sigma^2 of (*).

    Fitted on held-out one-step pairs under teacher forcing, which is the right
    place: teacher forcing sets Sigma_c = 0, so the entire residual there *is*
    the irreducible term. That is the one useful thing the teacher-forced regime
    is good for, and it is why the regime is kept in the pipeline rather than
    discarded after the Stage 1 correction.
    """
    n = min(max_states, len(xs))
    vals = []
    for i in range(n):
        x = xs[i].reshape(-1).to(fm.device, fm.dtype)
        y = ys[i].reshape(-1).to(fm.device, fm.dtype)
        r = (y - fm.predict(x)).double()
        vals.append(float((r ** 2).mean()))
    v = np.array(vals)
    return {"sigma2": float(v.mean()), "sigma2_median": float(np.median(v)),
            "n": n, "rel_se": float(v.std() / math.sqrt(max(n, 1)) / max(v.mean(), 1e-300)),
            "tail_ratio": float(v.mean() / max(np.median(v), 1e-300))}


def fit_lowrank_innovation(fm, xs, ys, rank: int = 16, max_states: int = 128):
    """Fit a zero-mean empirical residual covariance Q on calibration pairs.

    Q is a forecast-error covariance, not a correction model: no held-out
    target enters it, and its mean remains exactly zero.  With m calibration
    pairs, its nonzero part has rank at most m, so the thin SVD is inexpensive
    even when the state dimension is large.
    """
    n = min(max_states, len(xs))
    if n < 2:
        raise ValueError("structured innovation needs at least two calibration pairs")
    residuals = []
    for i in range(n):
        x = xs[i].reshape(-1).to(fm.device, fm.dtype)
        y = ys[i].reshape(-1).to(fm.device, fm.dtype)
        residuals.append((y - fm.predict(x)).double())
    R = torch.stack(residuals)                         # (m, N)
    _, s, vh = torch.linalg.svd(R, full_matrices=False)
    q = min(int(rank), int(s.numel()), R.shape[1] - 1)
    if q < 1:
        raise ValueError("innovation rank must be positive")
    eig = s[:q].square() / n
    total_trace = float(R.square().sum() / n)
    tail_trace = max(total_trace - float(eig.sum()), 0.0)
    tau = max(tail_trace / max(R.shape[1] - q, 1), 1e-30)
    U = vh[:q].T.contiguous()
    # Retained eigenvalues are exact empirical eigenvalues; the unobserved
    # complement is represented by its mean variance.
    prior = LowRankGaussian(torch.zeros(R.shape[1], dtype=torch.float64, device=R.device),
                             U, (eig - tau).to(torch.float64), tau,
                             label="empirical_residual_Q")
    return prior, {"kind": "empirical_lowrank", "n": n, "rank": q,
                   "trace": total_trace, "tail_trace": tail_trace,
                   "variance_per_dim": total_trace / R.shape[1],
                   "captured_trace_fraction": float(eig.sum()) / max(total_trace, 1e-300)}


def _velocity_fourier_modes(n: int, max_modes: int) -> list[tuple[int, int]]:
    """Non-duplicated, non-Nyquist Fourier wave numbers for real fields."""
    limit = n // 2
    modes = []
    for ky in range(-limit + 1, limit):
        for kx in range(-limit + 1, limit):
            if kx == 0 and ky == 0:
                continue
            # Keep one representative of the conjugate pair (k, -k).
            if ky < 0 or (ky == 0 and kx < 0):
                continue
            modes.append((ky, kx))
    return modes[:max_modes]


def fit_spectral_divfree_innovation(fm, xs, ys, rank: int = 16,
                                    max_states: int = 128):
    """Fit a centered, divergence-free spectral innovation covariance.

    This deliberately differs from ``fit_lowrank_innovation`` in two ways.
    It removes the calibration residual mean (a systematic forecast bias is
    not uncertainty), and its retained directions are real transverse Fourier
    modes.  Thus a small calibration corpus cannot memorize arbitrary pixel
    patterns.  Longitudinal and unrepresented error energy is retained in the
    isotropic tail, preserving the total centered residual trace.

    The current shared benchmark consists of two velocity channels on a square
    periodic grid; using this routine for another state layout is an error
    rather than an implicit, invalid reinterpretation of its channels.
    """
    n_samples = min(max_states, len(xs))
    if n_samples < 2:
        raise ValueError("spectral innovation needs at least two calibration pairs")
    residuals = []
    for i in range(n_samples):
        x = xs[i].reshape(-1).to(fm.device, fm.dtype)
        y = ys[i].reshape(-1).to(fm.device, fm.dtype)
        residuals.append((y - fm.predict(x)).double())
    R = torch.stack(residuals)
    N = R.shape[1]
    n = math.isqrt(N // 2)
    if 2 * n * n != N:
        raise ValueError("spectral divergence-free Q requires a two-channel square velocity state")

    mean = R.mean(dim=0)
    Rc = R - mean
    total_trace = float(Rc.square().sum() / n_samples)
    requested_pairs = max(1, min(int(rank) // 2, (N - 1) // 2))

    # Score the transverse component of every real Fourier mode.  FFT uses an
    # orthonormal convention, so energies can be compared directly.
    fields = Rc.reshape(n_samples, 2, n, n)
    hats = torch.fft.fft2(fields, norm="ortho")
    candidates = _velocity_fourier_modes(n, max_modes=n * n)
    scored = []
    for ky, kx in candidates:
        knorm = math.hypot(kx, ky)
        transverse = (-ky / knorm, kx / knorm)
        coeff = transverse[0] * hats[:, 0, ky % n, kx % n] + \
                transverse[1] * hats[:, 1, ky % n, kx % n]
        scored.append((float(coeff.abs().square().mean()), ky, kx))
    selected = sorted(scored, reverse=True)[:requested_pairs]
    if not selected:
        raise ValueError("could not construct spectral innovation modes")

    yy, xx = torch.meshgrid(torch.arange(n, device=R.device, dtype=torch.float64),
                            torch.arange(n, device=R.device, dtype=torch.float64),
                            indexing="ij")
    columns, selected_out = [], []
    amp = math.sqrt(2.0 / (n * n))
    for energy, ky, kx in selected:
        knorm = math.hypot(kx, ky)
        theta = 2.0 * math.pi * (kx * xx + ky * yy) / n
        transverse = torch.tensor([-ky / knorm, kx / knorm], device=R.device,
                                  dtype=torch.float64)
        for wave in (torch.cos(theta), torch.sin(theta)):
            field = amp * transverse[:, None, None] * wave[None]
            columns.append(field.reshape(-1))
        selected_out.append({"ky": ky, "kx": kx, "transverse_energy": energy})
    U = torch.stack(columns, dim=1)
    # Numerical orthonormalization protects the covariance algebra from any
    # finite-grid normalization corner cases.
    U, _ = torch.linalg.qr(U, mode="reduced")
    q = U.shape[1]
    coeff = Rc @ U
    eig = coeff.square().mean(dim=0)
    retained_trace = float(eig.sum())
    tail_trace = max(total_trace - retained_trace, 0.0)
    tau = max(tail_trace / max(N - q, 1), 1e-30)
    prior = LowRankGaussian(torch.zeros(N, dtype=torch.float64, device=R.device),
                             U, (eig - tau).clamp(min=0), tau,
                             label="spectral_divfree_residual_Q")
    return prior, {
        "kind": "spectral_divfree", "n": n_samples, "rank": q,
        "centered": True,
        "mean_residual_rms": float(mean.square().mean().sqrt()),
        "trace": total_trace, "tail_trace": tail_trace,
        "variance_per_dim": total_trace / N,
        "captured_trace_fraction": retained_trace / max(total_trace, 1e-300),
        "selected_modes": selected_out,
    }


def residual_temporal_diagnostic(fm, trajectories: torch.Tensor) -> dict:
    """Calibration-only lag-one diagnostic for the independent-Q assumption.

    ``trajectories`` is (B,T,N).  It returns no fitted correction and is used
    only to decide whether a future AR discrepancy model is warranted.
    """
    if trajectories.ndim != 3 or trajectories.shape[1] < 3:
        return {"available": False, "reason": "need trajectories with at least three snapshots"}
    residuals = []
    with torch.no_grad():
        for b in range(trajectories.shape[0]):
            for t in range(trajectories.shape[1] - 1):
                x = trajectories[b, t].reshape(-1).to(fm.device, fm.dtype)
                y = trajectories[b, t + 1].reshape(-1).to(fm.device, fm.dtype)
                residuals.append((y - fm.predict(x)).double())
    r = torch.stack(residuals).reshape(trajectories.shape[0], -1, trajectories.shape[-1])
    centered = r - r.mean(dim=(0, 1), keepdim=True)
    a, b = centered[:, :-1], centered[:, 1:]
    denom = a.square().sum().clamp(min=1e-300)
    rho = float((a * b).sum() / denom)
    cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-300)
    return {"available": True, "pairs": int(a.shape[0] * a.shape[1]),
            "lag1_ar_coefficient": rho, "lag1_cosine_mean": float(cos.mean()),
            "mean_residual_rms": float(r.mean(dim=(0, 1)).square().mean().sqrt())}


def shrink_innovation(innovation: LowRankGaussian, rho: float) -> LowRankGaussian:
    """Trace-preserving shrinkage of Q toward its isotropic counterpart.

    Q_rho = rho Q + (1-rho) tr(Q)/N I.  ``rho`` is selected using only a
    trajectory-disjoint validation subset of the calibration corpus; it is
    never fitted on the held-out S3 trajectories.
    """
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("innovation shrinkage rho must lie in [0, 1]")
    trace_per_dim = innovation.trace() / innovation.N
    return LowRankGaussian(innovation.mean, innovation.U,
                            rho * innovation.alpha * innovation.d,
                            rho * innovation.alpha * innovation.tau +
                            (1.0 - rho) * trace_per_dim,
                            alpha=1.0,
                            label=f"{innovation.label}-shrink{rho:g}")


# ---------------------------------------------------------------------------
# Scoring a propagated rollout against the truth
# ---------------------------------------------------------------------------

def score_rollout(trace: RolloutTrace, truth: torch.Tensor) -> list[dict]:
    """Per-step calibration of the propagated covariance against actual error.

    `truth` is (steps+1, N) with truth[0] the initial state, so truth[t+1] is
    the target for step t+1.

    The columns that decide whether propagation works:

      rmse            actual rollout error, grows
      pred_rms        sqrt(trace/N) of the propagated covariance, should grow
                      *with the same shape*
      ratio           pred_rms / rmse. Flat and near 1 means the recursion has
                      the growth rate right. This is the headline number: an
                      isotropic or fixed covariance cannot produce a flat ratio
                      because it has no growth mechanism at all.
      z               (D^2 - N)/sqrt(2N), the joint calibration
      frac_err_in_rank share of squared error inside the retained subspace,
                      against the k/N an isotropic error would give. If this
                      collapses toward k/N as t grows, the propagated basis has
                      stopped tracking where the error actually is.
    """
    out = []
    for t, (p, info) in enumerate(zip(trace.priors, trace.info)):
        if p is None:
            continue
        y = truth[t + 1].reshape(-1).to(p.mean.device, p.mean.dtype)
        err = y - p.mean
        err_sq = float((err ** 2).sum())
        maha = float(p.mahalanobis_sq(y))
        # k can be 0: starting from an isotropic Sigma_0 with the complement
        # term disabled, nothing ever creates a direction, so the object stays
        # rank 0 for the whole rollout. That is the correct behaviour of that
        # ablation, not a failure, but the subspace diagnostics are undefined.
        proj = p.U.T @ err if p.k else err.new_zeros(0)
        in_rank = float((proj ** 2).sum() / max(err_sq, 1e-300))
        iso_ref = (in_rank / (p.k / p.N)) if p.k else float("nan")
        out.append({
            "step": t + 1,
            "rmse": math.sqrt(err_sq / p.N),
            "pred_rms": math.sqrt(p.trace() / p.N),
            "ratio": math.sqrt(p.trace() / max(err_sq, 1e-300)),
            "maha_over_N": maha / p.N,
            "z": (maha - p.N) / math.sqrt(2.0 * p.N),
            "frac_err_in_rank": in_rank,
            "frac_err_in_rank_over_isotropic": iso_ref,
            "trace_total": info["trace_total"],
            "frac_trace_in_rank": info["frac_trace_in_rank"],
        })
    return out


def aggregate_rollout(scores: list[list[dict]]) -> list[dict]:
    """Mean and standard error per step across trajectories."""
    if not scores:
        return []
    n_steps = min(len(s) for s in scores)
    keys = [k for k in scores[0][0] if k != "step"]
    out = []
    for t in range(n_steps):
        row = {"step": t + 1, "n": len(scores)}
        for kk in keys:
            v = np.array([s[t][kk] for s in scores], dtype=np.float64)
            row[kk] = float(v.mean())
            row[kk + "_se"] = float(v.std() / math.sqrt(max(v.size, 1)))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Reference: what the baselines can do here
# ---------------------------------------------------------------------------

def constant_covariance_rollout(prior: LowRankGaussian, means: list,
                                truth: torch.Tensor, growth: str = "none",
                                sigma2: float | None = None) -> list[dict]:
    """Score a *non-propagated* covariance along the same rollout.

    Two controls, and the comparison against them is the whole point of the
    rollout experiment:

      "none"    the single-step covariance held fixed for all t. This is what
                any per-state uncertainty method gives if you simply apply it at
                each step -- it has no mechanism for accumulation.
      "sqrt_t"  scaled by t, i.e. assuming errors add as independent increments
                (a random walk). This is the strongest *heuristic* growth model
                and the fair thing to beat: reproducing linear-in-t variance
                growth is not evidence for the Jacobian, since diffusion gives
                it for free. The claim can only be that the *shape* and the
                *deviation from* sqrt(t) growth are right.
    """
    out = []
    for t, mu in enumerate(means):
        y = truth[t + 1].reshape(-1).to(prior.mean.device, prior.mean.dtype)
        scale = 1.0 if growth == "none" else float(t + 1)
        p = LowRankGaussian(mu.double(), prior.U, prior.d, prior.tau,
                            alpha=prior.alpha * scale, label=prior.label)
        err = y - p.mean
        err_sq = float((err ** 2).sum())
        maha = float(p.mahalanobis_sq(y))
        out.append({"step": t + 1, "rmse": math.sqrt(err_sq / p.N),
                    "pred_rms": math.sqrt(p.trace() / p.N),
                    "ratio": math.sqrt(p.trace() / max(err_sq, 1e-300)),
                    "maha_over_N": maha / p.N,
                    "z": (maha - p.N) / math.sqrt(2.0 * p.N),
                    "growth": growth})
    return out
