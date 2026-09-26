"""Pseudo-spectral solvers for the three 1D PDEs used as testbeds.

All problems are periodic on [0, L) with N grid points, integrated with an
integrating-factor RK4 scheme (exact on the linear part, RK4 on the nonlinear
part). We deliberately use three PDEs with different character:

  advdiff  linear, smoothing   -> posterior is exactly Gaussian (correctness test)
  burgers  nonlinear, smoothing -> mild non-Gaussianity, shock formation
  ks       nonlinear, chaotic   -> positive Lyapunov exponent, the hard case
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .utils import DATA_DIR


@dataclass(frozen=True)
class PDESpec:
    name: str
    L: float
    N: int
    dt: float            # solver step
    stride: int          # solver steps between stored snapshots
    nu: float = 0.0      # viscosity (advdiff, burgers)
    c_adv: float = 0.0   # advection speed (advdiff)
    warmup: int = 0      # stored snapshots discarded from the front

    @property
    def dt_out(self) -> float:
        """Time between consecutive stored snapshots -- the model's step."""
        return self.dt * self.stride


SPECS = {
    "advdiff": PDESpec("advdiff", L=2 * np.pi, N=128, dt=2e-3, stride=25,
                       nu=0.01, c_adv=1.0, warmup=0),
    "burgers": PDESpec("burgers", L=2 * np.pi, N=128, dt=1e-3, stride=50,
                       nu=0.02, warmup=0),
    "ks": PDESpec("ks", L=32 * np.pi, N=128, dt=0.05, stride=5, warmup=200),
}


def wavenumbers(spec: PDESpec, device=None, dtype=torch.float64) -> torch.Tensor:
    return torch.fft.rfftfreq(spec.N, d=spec.L / spec.N, device=device,
                              dtype=dtype) * 2 * np.pi


def _linear_symbol(spec: PDESpec, k: np.ndarray) -> np.ndarray:
    """Symbol of the linear operator L in u_t = L u + N(u)."""
    if spec.name == "advdiff":
        return -1j * spec.c_adv * k - spec.nu * k ** 2
    if spec.name == "burgers":
        return -spec.nu * k ** 2
    if spec.name == "ks":
        return k ** 2 - k ** 4
    raise ValueError(spec.name)


def _nonlinear(spec: PDESpec, uh: np.ndarray, k: np.ndarray, N: int) -> np.ndarray:
    """Spectral nonlinear term, 2/3-dealiased."""
    if spec.name == "advdiff":
        return np.zeros_like(uh)
    u = np.fft.irfft(uh, n=N)
    nh = -0.5j * k * np.fft.rfft(u * u)
    cutoff = int(N / 3)
    nh[cutoff:] = 0.0
    return nh


def _ifrk4_step(spec: PDESpec, uh, k, N, dt):
    """Integrating-factor RK4: exact linear propagation, RK4 nonlinear."""
    Lsym = _linear_symbol(spec, k)
    E, E2 = np.exp(dt * Lsym), np.exp(dt * Lsym / 2)
    a = _nonlinear(spec, uh, k, N)
    b = _nonlinear(spec, E2 * (uh + dt / 2 * a), k, N)
    c = _nonlinear(spec, E2 * uh + dt / 2 * b, k, N)
    d = _nonlinear(spec, E * uh + dt * E2 * c, k, N)
    return E * uh + dt / 6 * (E * a + 2 * E2 * (b + c) + d)


def random_initial_condition(spec: PDESpec, rng: np.random.Generator) -> np.ndarray:
    """Smooth random field: random Fourier series with decaying amplitudes."""
    x = np.linspace(0, spec.L, spec.N, endpoint=False)
    if spec.name == "ks":
        # KS forgets its initial condition; a small perturbation off the
        # attractor plus a long warmup is the standard construction.
        u = 0.1 * rng.standard_normal(spec.N)
        u = np.real(np.fft.irfft(np.fft.rfft(u) * np.exp(-0.01 * np.arange(spec.N // 2 + 1) ** 2),
                                 n=spec.N))
        return u + 0.5 * np.cos(2 * np.pi * x / spec.L) * rng.standard_normal()
    n_modes = 8
    u = np.zeros(spec.N)
    for m in range(1, n_modes + 1):
        amp = rng.standard_normal() / m ** 1.2
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(2 * np.pi * m * x / spec.L + phase)
    return u / (np.abs(u).max() + 1e-12)


def simulate(spec: PDESpec, n_traj: int, n_steps: int, seed: int = 0) -> np.ndarray:
    """Return trajectories of shape (n_traj, n_steps + 1, N)."""
    rng = np.random.default_rng(seed)
    k = np.fft.rfftfreq(spec.N, d=spec.L / spec.N) * 2 * np.pi
    total = n_steps + spec.warmup
    out = np.empty((n_traj, total + 1, spec.N), dtype=np.float64)
    for i in range(n_traj):
        u = random_initial_condition(spec, rng)
        uh = np.fft.rfft(u)
        out[i, 0] = u
        for s in range(total):
            for _ in range(spec.stride):
                uh = _ifrk4_step(spec, uh, k, spec.N, spec.dt)
            out[i, s + 1] = np.fft.irfft(uh, n=spec.N)
        if not np.isfinite(out[i]).all():
            raise RuntimeError(f"{spec.name}: trajectory {i} diverged; reduce dt")
    return out[:, spec.warmup:]


def build_dataset(pde: str, n_train: int = 96, n_val: int = 32, n_test: int = 32,
                  n_steps: int = 60, seed: int = 0, force: bool = False) -> dict:
    """Simulate (or load) trajectories and split into train/cal/test."""
    spec = SPECS[pde]
    path = DATA_DIR / f"{pde}_{n_train}_{n_val}_{n_test}_{n_steps}_{seed}.npz"
    if path.exists() and not force:
        z = np.load(path)
        return {"train": z["train"], "val": z["val"], "test": z["test"], "spec": spec}
    total = n_train + n_val + n_test
    traj = simulate(spec, total, n_steps, seed=seed)
    train, val, test = traj[:n_train], traj[n_train:n_train + n_val], traj[n_train + n_val:]
    np.savez_compressed(path, train=train, val=val, test=test)
    return {"train": train, "val": val, "test": test, "spec": spec}


def pairs(traj: np.ndarray, device=None, dtype=torch.float32):
    """Flatten trajectories into (input, target) one-step pairs."""
    x = torch.as_tensor(traj[:, :-1], dtype=dtype, device=device).reshape(-1, traj.shape[-1])
    y = torch.as_tensor(traj[:, 1:], dtype=dtype, device=device).reshape(-1, traj.shape[-1])
    return x, y


def normalizer(traj: np.ndarray) -> tuple[float, float]:
    return float(traj.mean()), float(traj.std() + 1e-8)
