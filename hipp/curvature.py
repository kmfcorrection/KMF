"""Jacobian probing and curvature estimators.

Every estimator returns a `CurvatureEstimate` holding a precision *shape* H
(so that Sigma = alpha H^{-1}), plus whatever spectral information it produced
along the way. See docs/method.md section 3 for the taxonomy.

The central methodological point: `gn` (H = J^T J) and `pushforward`
(Sigma = J J^T) order directions oppositely once turned into a covariance.
Both are implemented; Stage 1 decides between them empirically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.func import jacrev, jvp, vjp

from .priors import GaussianPrior, _sym, safe_cholesky


@dataclass
class CurvatureEstimate:
    H: torch.Tensor                     # precision shape (N, N), float64
    mean: torch.Tensor                  # prediction c_hat (N,), float64
    name: str = ""
    J: torch.Tensor | None = None       # Jacobian, when available
    svals: torch.Tensor | None = None   # singular values of J, descending
    U: torch.Tensor | None = None       # left singular vectors (output space)
    V: torch.Tensor | None = None       # right singular vectors (input space)
    meta: dict = field(default_factory=dict)

    def prior(self, alpha: float = 1.0) -> GaussianPrior:
        return GaussianPrior(self.mean, self.H, alpha=alpha, label=self.name)


# ---------------------------------------------------------------------------
# Jacobian access
# ---------------------------------------------------------------------------

def predict(model, c: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(c.reshape(-1))


def exact_jacobian(model, c: torch.Tensor) -> torch.Tensor:
    """Exact J = df/dc via reverse-mode AD. O(N) backward passes."""
    c = c.reshape(-1).detach()
    return jacrev(lambda z: model(z))(c).detach()


def jvp_fn(model, c: torch.Tensor):
    c = c.reshape(-1).detach()

    def apply(v: torch.Tensor) -> torch.Tensor:
        return jvp(lambda z: model(z), (c,), (v.reshape(-1),))[1].detach()
    return apply


def vjp_fn(model, c: torch.Tensor):
    c = c.reshape(-1).detach()
    _, fn = vjp(lambda z: model(z), c)

    def apply(w: torch.Tensor) -> torch.Tensor:
        return fn(w.reshape(-1))[0].detach()
    return apply


def fd_directional(model, c: torch.Tensor, V: torch.Tensor, eps: float,
                   central: bool = True) -> torch.Tensor:
    """Delta_i = (f(c + eps v_i) - f(c)) / eps for each column of V.

    V: (N, m). Returns (N, m). Forward passes only -- works on a black-box model.
    """
    c = c.reshape(-1).detach()
    m = V.shape[1]
    base = None if central else model(c).detach()
    out = []
    for i in range(m):
        v = V[:, i]
        plus = model(c + eps * v).detach()
        if central:
            minus = model(c - eps * v).detach()
            out.append((plus - minus) / (2 * eps))
        else:
            out.append((plus - base) / eps)
    return torch.stack(out, dim=1)


def fd_jacobian(model, c: torch.Tensor, m: int | None = None, eps: float = 1e-3,
                central: bool = True, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares Jacobian from random probes. Returns (J_hat, Delta)."""
    N = c.numel()
    m = m or N
    V = torch.randn(N, m, dtype=c.dtype, device=c.device, generator=generator)
    V = V / V.norm(dim=0, keepdim=True) * (N ** 0.5)
    D = fd_directional(model, c, V, eps, central=central)
    # D ~ J V  =>  J ~ D V^+ (right pseudo-inverse; exact when m >= N)
    Jhat = D @ torch.linalg.pinv(V)
    return Jhat, D


def randomized_svd_jacobian(model, c: torch.Tensor, k: int = 16, oversample: int = 8,
                            n_iter: int = 2, generator=None):
    """Rank-k randomized SVD of J using matrix-free JVP/VJP products."""
    N = c.numel()
    ell = min(k + oversample, N)
    Jv, Jtw = jvp_fn(model, c), vjp_fn(model, c)
    Omega = torch.randn(N, ell, dtype=c.dtype, device=c.device, generator=generator)
    Y = torch.stack([Jv(Omega[:, i]) for i in range(ell)], dim=1)
    Q, _ = torch.linalg.qr(Y)
    for _ in range(n_iter):                       # power iterations for accuracy
        Z = torch.stack([Jtw(Q[:, i]) for i in range(Q.shape[1])], dim=1)
        Q, _ = torch.linalg.qr(Z)
        Y = torch.stack([Jv(Q[:, i]) for i in range(Q.shape[1])], dim=1)
        Q, _ = torch.linalg.qr(Y)
    B = torch.stack([Jtw(Q[:, i]) for i in range(Q.shape[1])], dim=1).T   # (ell, N)
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], S[:k], Vh[:k].T


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------

def damp(M: torch.Tensor, tau_rel: float) -> torch.Tensor:
    """Add tau_rel * mean(diag(M)) * I. Relative damping keeps tau_rel
    dimensionless so the same value is meaningful across PDEs."""
    scale = torch.diagonal(M).mean().clamp(min=1e-30)
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    return _sym(M) + tau_rel * scale * eye


def invert(M: torch.Tensor) -> torch.Tensor:
    L = safe_cholesky(M)
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    return _sym(torch.cholesky_solve(eye, L))


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

ESTIMATORS: dict[str, str] = {
    "gn": "Gauss-Newton input-space curvature, H = J^T J + tau I",
    "pushforward": "Linearized error propagation, Sigma = J J^T + tau I",
    "fd_gn": "Gauss-Newton from finite-difference probes (black-box)",
    "fd_pushforward": "Pushforward from finite-difference probes (black-box)",
    "dircov": "Empirical covariance of directional derivatives",
    "diag_gn": "Diagonal of J^T J (Hutchinson-style, cheapest anisotropy)",
    "lowrank_gn": "Rank-k randomized approximation of J^T J",
    "lowrank_pushforward": "Rank-k randomized approximation of J J^T",
    "identity": "Isotropic control (no curvature information)",
}


def estimate(model, c: torch.Tensor, method: str = "pushforward", tau_rel: float = 1e-2,
             k: int = 16, m: int | None = None, eps: float = 1e-3,
             generator=None, J_exact: torch.Tensor | None = None) -> CurvatureEstimate:
    """Compute a curvature estimate at conditioning state `c`.

    `J_exact` lets a caller supply an already-computed exact Jacobian; the
    estimators that need it (gn / pushforward / diag_gn) then skip the O(N)
    backward passes, which is what makes sweeping several estimators over a
    few hundred calibration states affordable.
    """
    c = c.reshape(-1).detach()
    N = c.numel()
    mean = predict(model, c).to(torch.float64)
    f64 = dict(dtype=torch.float64, device=c.device)
    meta = {"method": method, "tau_rel": tau_rel}

    if method == "identity":
        H = torch.eye(N, **f64)
        return CurvatureEstimate(H, mean, method, meta=meta)

    J = U = S = V = None

    if method in ("gn", "pushforward", "diag_gn"):
        J = (exact_jacobian(model, c) if J_exact is None else J_exact).to(torch.float64)
        U, S, Vh = torch.linalg.svd(J)
        V = Vh.T
    elif method in ("fd_gn", "fd_pushforward", "dircov"):
        Jhat, D = fd_jacobian(model, c, m=m, eps=eps, generator=generator)
        J = Jhat.to(torch.float64)
        D = D.to(torch.float64)
        U, S, Vh = torch.linalg.svd(J)
        V = Vh.T
        meta["eps"] = eps
        meta["m_probes"] = D.shape[1]
    elif method in ("lowrank_gn", "lowrank_pushforward"):
        Uk, Sk, Vk = randomized_svd_jacobian(model, c, k=k, generator=generator)
        U, S, V = Uk.to(torch.float64), Sk.to(torch.float64), Vk.to(torch.float64)
        meta["k"] = k
    else:
        raise ValueError(f"unknown estimator {method!r}; choices: {list(ESTIMATORS)}")

    if method in ("gn", "fd_gn"):
        H = damp(J.T @ J, tau_rel)
    elif method in ("pushforward", "fd_pushforward"):
        H = invert(damp(J @ J.T, tau_rel))
    elif method == "dircov":
        Sigma = (D @ D.T) / D.shape[1]
        H = invert(damp(Sigma, tau_rel))
    elif method == "diag_gn":
        d = (J * J).sum(dim=0)
        H = damp(torch.diag(d), tau_rel)
    elif method == "lowrank_gn":
        H = damp(V @ torch.diag(S ** 2) @ V.T, tau_rel)
    elif method == "lowrank_pushforward":
        H = invert(damp(U @ torch.diag(S ** 2) @ U.T, tau_rel))

    return CurvatureEstimate(H, mean, method, J=J, svals=S, U=U, V=V, meta=meta)


def spectrum_stats(est: CurvatureEstimate) -> dict:
    """Summary statistics of the curvature shape H."""
    ev = torch.linalg.eigvalsh(est.H).clamp(min=1e-300)
    ev_desc = ev.flip(0)
    csum = torch.cumsum(ev_desc, 0) / ev_desc.sum()
    return {
        "trace": float(ev.sum()),
        "logdet": float(torch.log(ev).sum()),
        "eig_max": float(ev[-1]),
        "eig_min": float(ev[0]),
        "condition_number": float(ev[-1] / ev[0]),
        "effective_rank": float(torch.exp(-(ev / ev.sum() * torch.log(ev / ev.sum())).sum())),
        "n_modes_90pct": int((csum < 0.9).sum().item()) + 1,
        "eigenvalues": ev_desc.cpu(),
    }
