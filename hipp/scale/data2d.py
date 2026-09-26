"""2D PDE data at foundation-model scale.

Three systems, chosen to span the same character axis as the 1D suite so the
small-scale conclusions have something to be compared against:

  ns2d_decay   decaying 2D turbulence, no forcing -- dissipative, the state
               relaxes; the analogue of `burgers`. Error should become
               bias-dominated as the flow smooths, which is the regime where
               the 1D study found the Jacobian construction has nothing to say.
  ns2d_forced  the standard FNO / PDEBench forced Navier-Stokes benchmark
               (Li et al. forcing, nu in {1e-3, 1e-4, 1e-5}). Statistically
               stationary; the workhorse.
  kolmogorov   Kolmogorov flow at Re ~ 1000 with linear drag -- genuinely
               chaotic, positive Lyapunov exponent. This is the 2D `ks`: the
               case where predictive uncertainty is real rather than an
               artifact of an under-trained model, and the only one where the
               pushforward mechanism is predicted to dominate.

Solver: pseudo-spectral vorticity formulation on a periodic square, RK4 with an
integrating factor on the viscous term (exact linear propagation) and 2/3
dealiasing on the advection. Written batched in torch so it runs on the GPU --
generating the ~10^3 trajectories a scaled study needs is otherwise a
multi-day CPU job.

    omega_t + u . grad(omega) = nu * lap(omega) + f - drag * omega
    lap(psi) = -omega,   u = (psi_y, -psi_x)

Storage is sharded float32 memmaps plus a JSON manifest rather than one npz:
at 128x128 with 60 snapshots a single trajectory is 3.9 MB, so a 1024-trajectory
dataset is 4 GB and must be streamed rather than loaded.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..utils import DATA_DIR


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PDESpec2D:
    name: str
    L: float = 2 * math.pi
    n: int = 128              # grid is n x n
    nu: float = 1e-4
    dt: float = 1e-3          # solver step
    stride: int = 50          # solver steps between stored snapshots
    drag: float = 0.0
    forcing: str = "none"     # none | li | kolmogorov
    forcing_amp: float = 0.1
    forcing_k: int = 4
    warmup: int = 0           # stored snapshots discarded from the front
    ic_peak_k: float = 4.0    # spectral peak of the random initial condition

    @property
    def dt_out(self) -> float:
        return self.dt * self.stride

    @property
    def N(self) -> int:
        """Dimension of the flattened state -- the number that decides whether
        any of the dense 1D machinery is usable. 128x128 gives 16,384."""
        return self.n * self.n


# The output cadence `stride` is a *scientific* parameter here, not a numerical
# one, and it is the parameter this study is most sensitive to.
#
# The method asks whether the local geometry of the learned operator carries
# predictive information. If the operator is near-identity, it has no geometry to
# carry any: J = I + eps has a flat spectrum, every rank-k subspace is
# equivalent, and both curvature surrogates degenerate to isotropic. Measured on
# the *true* solver (experiments/scale/s5_timescale.py, N=144, nu=1e-4):
#
#   dt_out    ||J-I||/||I||   captured trace at k/N=0.22   cond(J)
#   0.05          0.033                 0.23                1.16     <- degenerate
#   0.20          0.134                 0.28                1.79
#   0.80          0.559                 0.48                8.71
#   3.20          2.707                 0.91              269.4
#
# At dt_out = 0.05 one step advances the flow ~5% of an eddy turnover time, so
# captured trace ~ k/N, which is exactly what an isotropic operator gives. A
# surrogate trained on that target is near-identity *because the target is*, and
# a null result says nothing about the method. The original strides were chosen
# for easy learning and produced precisely that regime.
#
# There is a real trade-off in the other direction, and it is not resolved here:
# a larger dt_out gives more geometry but a harder learning problem, and if the
# surrogate cannot fit the map then the error becomes model bias, which is
# invisible to J for a different reason (docs/method.md 9). Both ends of the
# sweep produce a null; the usable window is in between.
#
# The defaults below target dt_out ~ 1.0, roughly one eddy turnover. s5 rates
# that "weak-to-usable" (~2.5x isotropic at k/N = 0.1) -- deliberately the
# conservative end of the window, because a surrogate that cannot learn its
# target invalidates the study more thoroughly than a weakly anisotropic
# operator does. Pin it down empirically before the large runs: sweep --stride
# at low resolution, train briefly at each, and take the largest cadence whose
# one-step error is still acceptable. This is also the regime real PDE
# foundation models operate in -- they take large steps relative to the
# dynamics, which is the whole reason they are faster than a solver.
SPECS2D: dict[str, PDESpec2D] = {
    "ns2d_decay": PDESpec2D("ns2d_decay", nu=1e-3, dt=1e-3, stride=1000,
                            forcing="none", ic_peak_k=6.0),
    "ns2d_forced": PDESpec2D("ns2d_forced", nu=1e-4, dt=1e-3, stride=1000,
                             forcing="li", forcing_amp=0.1, warmup=10),
    "kolmogorov": PDESpec2D("kolmogorov", nu=1e-3, dt=5e-4, stride=2000,
                            drag=0.1, forcing="kolmogorov", forcing_amp=1.0,
                            forcing_k=4, warmup=40, ic_peak_k=4.0),
}

# The pre-correction cadence, kept reachable so the degenerate regime can be
# reproduced as a negative control rather than only described.
LEGACY_STRIDES = {"ns2d_decay": 50, "ns2d_forced": 50, "kolmogorov": 40}


# ---------------------------------------------------------------------------
# Spectral operators
# ---------------------------------------------------------------------------

class SpectralGrid2D:
    """Wavenumbers, dealiasing mask and the Biot-Savart inverse for one grid."""

    def __init__(self, spec: PDESpec2D, device=None, dtype=torch.float64):
        n, L = spec.n, spec.L
        self.spec, self.device, self.dtype = spec, device, dtype
        kx = torch.fft.fftfreq(n, d=L / n, device=device, dtype=dtype) * 2 * math.pi
        ky = torch.fft.rfftfreq(n, d=L / n, device=device, dtype=dtype) * 2 * math.pi
        self.kx = kx.view(-1, 1)                 # (n, 1)
        self.ky = ky.view(1, -1)                 # (1, n//2+1)
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.inv_k2 = torch.where(self.k2 > 0, 1.0 / self.k2.clamp(min=1e-30),
                                  torch.zeros_like(self.k2))
        kmax = float(ky.max())
        self.dealias = ((self.kx.abs() <= 2 / 3 * kmax) &
                        (self.ky.abs() <= 2 / 3 * kmax)).to(dtype)
        x = torch.arange(n, device=device, dtype=dtype) * (L / n)
        self.X, self.Y = torch.meshgrid(x, x, indexing="ij")
        self.forcing = self._build_forcing()

    def _build_forcing(self) -> torch.Tensor | None:
        s = self.spec
        if s.forcing == "none":
            return None
        if s.forcing == "li":
            # Li et al. (FNO) benchmark forcing on [0, 2pi)^2.
            return s.forcing_amp * (torch.sin(self.X + self.Y)
                                    + torch.cos(self.X + self.Y))
        if s.forcing == "kolmogorov":
            # Monochromatic shear forcing; the curl of n*cos(n*y) x-hat.
            return -s.forcing_amp * s.forcing_k * torch.cos(s.forcing_k * self.Y)
        raise ValueError(s.forcing)

    def velocity(self, wh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(u, v) in physical space from vorticity in spectral space."""
        psih = wh * self.inv_k2
        u = torch.fft.irfft2(1j * self.ky * psih, s=(self.spec.n, self.spec.n))
        v = torch.fft.irfft2(-1j * self.kx * psih, s=(self.spec.n, self.spec.n))
        return u, v

    def advection(self, wh: torch.Tensor) -> torch.Tensor:
        """-(u . grad) omega in spectral space, 2/3-dealiased."""
        n = self.spec.n
        u, v = self.velocity(wh)
        wx = torch.fft.irfft2(1j * self.kx * wh, s=(n, n))
        wy = torch.fft.irfft2(1j * self.ky * wh, s=(n, n))
        return -torch.fft.rfft2(u * wx + v * wy) * self.dealias

    def rhs_hat(self, wh: torch.Tensor) -> torch.Tensor:
        """Full right-hand side of omega_t = rhs, in spectral space."""
        out = self.advection(wh) - self.spec.nu * self.k2 * wh
        if self.spec.drag:
            out = out - self.spec.drag * wh
        if self.forcing is not None:
            out = out + torch.fft.rfft2(self.forcing)
        return out

    def nonlinear_hat(self, wh: torch.Tensor) -> torch.Tensor:
        """Everything except the stiff linear part (handled by the integrating
        factor): advection, forcing. Drag is linear and goes in the symbol."""
        out = self.advection(wh)
        if self.forcing is not None:
            out = out + torch.fft.rfft2(self.forcing)
        return out

    @property
    def lin_symbol(self) -> torch.Tensor:
        return -self.spec.nu * self.k2 - self.spec.drag


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------

def ifrk4_step(grid: SpectralGrid2D, wh: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrating-factor RK4: exact on the viscous/drag part, RK4 on the rest.

    Same scheme as the 1D solver, so a resolution comparison is not confounded
    by a change of integrator.
    """
    Lsym = grid.lin_symbol
    E, E2 = torch.exp(dt * Lsym), torch.exp(dt * Lsym / 2)
    a = grid.nonlinear_hat(wh)
    b = grid.nonlinear_hat(E2 * (wh + dt / 2 * a))
    c = grid.nonlinear_hat(E2 * wh + dt / 2 * b)
    d = grid.nonlinear_hat(E * wh + dt * E2 * c)
    return E * wh + dt / 6 * (E * a + 2 * E2 * (b + c) + d)


def random_initial_vorticity(spec: PDESpec2D, n_traj: int, generator=None,
                             device=None, dtype=torch.float64,
                             grid: "SpectralGrid2D | None" = None) -> torch.Tensor:
    """Gaussian random field with a peaked energy spectrum, unit RMS vorticity.

    E(k) ~ k^4 / (k + k_peak)^8 is the standard McWilliams-style initialization
    for 2D turbulence: smooth, isotropic, and with the energy concentrated at a
    controllable scale so the resolution is genuinely used.
    """
    n = spec.n
    grid = grid or SpectralGrid2D(spec, device=device, dtype=dtype)
    k = grid.k2.sqrt()
    amp = (k ** 4) / (k + spec.ic_peak_k) ** 8
    amp = torch.where(torch.isfinite(amp), amp, torch.zeros_like(amp)).sqrt()
    noise = torch.randn(n_traj, n, n, device=device, dtype=dtype, generator=generator)
    wh = torch.fft.rfft2(noise) * amp
    w = torch.fft.irfft2(wh, s=(n, n))
    rms = w.flatten(1).std(dim=1).clamp(min=1e-12).view(-1, 1, 1)
    return w / rms


@torch.no_grad()
def simulate(spec: PDESpec2D, n_traj: int, n_steps: int, seed: int = 0,
             device=None, batch: int = 32, dtype=torch.float64,
             progress: bool = True, out_path: "Path | None" = None) -> np.ndarray:
    """Trajectories of shape (n_traj, n_steps + 1, n, n), float32.

    Integration is float64 (the enstrophy cascade at nu = 1e-4 is not stable in
    float32 over thousands of steps); only the stored output is cast down.

    With `out_path`, results are written straight into an on-disk .npy memmap a
    batch at a time and peak host memory is one batch rather than the whole
    split. That is not a micro-optimization: the in-memory array for the default
    512 trajectories at 128x128 is 1.4 GB, and a SLURM job that did not pass
    --mem gets mem-per-cpu times cpus -- often 2-4 GB total -- so allocating it
    is enough to get the process SIGKILLed on a node with 2 TB free. Streaming
    also removes the ceiling on how large a dataset can be generated at all.
    """
    grid = SpectralGrid2D(spec, device=device, dtype=dtype)
    g = torch.Generator(device=device if device is not None else "cpu")
    g.manual_seed(seed)
    shape = (n_traj, n_steps + 1, spec.n, spec.n)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                        shape=shape)
    else:
        out = np.empty(shape, dtype=np.float32)

    def advance(wh, n_snap, tag):
        for _ in range(n_snap):
            for _ in range(spec.stride):
                wh = ifrk4_step(grid, wh, spec.dt)
            if not torch.isfinite(wh).all():
                raise RuntimeError(
                    f"{spec.name}: diverged during {tag} -- reduce dt "
                    f"(currently {spec.dt:g}) or raise nu")
        return wh

    def to_host(wh):
        return torch.fft.irfft2(wh, s=(spec.n, spec.n)).float().cpu().numpy()

    for start in range(0, n_traj, batch):
        b = min(batch, n_traj - start)
        w0 = random_initial_vorticity(spec, b, generator=g, device=device,
                                      dtype=dtype, grid=grid)
        wh = advance(torch.fft.rfft2(w0), spec.warmup, "warmup")
        out[start:start + b, 0] = to_host(wh)
        for s in range(n_steps):
            wh = advance(wh, 1, f"snapshot {s + 1}")
            out[start:start + b, s + 1] = to_host(wh)
        if out_path is not None:
            out.flush()
        if progress:
            print(f"    simulated {start + b}/{n_traj} trajectories", flush=True)
    return out


# ---------------------------------------------------------------------------
# Sharded on-disk dataset
# ---------------------------------------------------------------------------

def dataset_dir(pde: str, n: int, seed: int, stride: int | None = None) -> Path:
    """Directory for one generated dataset.

    `stride` is part of the identity. It was not, and that was a silent
    correctness bug: changing the output cadence changes the target operator
    entirely (see SPECS2D), but the old key was (pde, n, seed) alone, so a run
    with a new stride would find the old directory, decide the data already
    existed, and train on trajectories from a different physical setup than the
    one its spec described. Nothing would error; the results would simply be
    about the wrong problem.
    """
    if stride is None:
        stride = SPECS2D[pde].stride
    return DATA_DIR / "scale" / f"{pde}_n{n}_s{stride}_seed{seed}"


class ShardedTrajectories:
    """Memory-mapped access to a generated dataset.

    Each split is one .npy of shape (n_traj, n_steps+1, n, n) opened with
    mmap_mode='r'. Random access to one-step pairs is then a page fault rather
    than a 4 GB read, which is what makes a 128x128 study fit on a normal node.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        with open(self.root / "manifest.json") as fh:
            self.manifest = json.load(fh)
        self.spec = PDESpec2D(**self.manifest["spec"])
        self._arrays: dict[str, np.ndarray] = {}

    def split(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            self._arrays[name] = np.load(self.root / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def n_pairs(self, name: str) -> int:
        a = self.split(name)
        return a.shape[0] * (a.shape[1] - 1)

    def pair(self, name: str, idx: int, device=None, dtype=torch.float32):
        """One (input, target) snapshot pair, indexed over the flattened
        (trajectory, time) grid."""
        a = self.split(name)
        t_len = a.shape[1] - 1
        i, t = divmod(int(idx), t_len)
        x = torch.as_tensor(np.array(a[i, t]), dtype=dtype, device=device)
        y = torch.as_tensor(np.array(a[i, t + 1]), dtype=dtype, device=device)
        return x, y

    def batch(self, name: str, idxs, device=None, dtype=torch.float32):
        xs, ys = zip(*(self.pair(name, i, device=device, dtype=dtype) for i in idxs))
        return torch.stack(xs), torch.stack(ys)

    def trajectory(self, name: str, i: int, device=None, dtype=torch.float32):
        return torch.as_tensor(np.array(self.split(name)[int(i)]), dtype=dtype,
                               device=device)

    def normalizer(self) -> tuple[float, float]:
        return self.manifest["mu"], self.manifest["sigma"]


def build_dataset2d(pde: str, n_train: int = 512, n_val: int = 128, n_test: int = 128,
                    n_steps: int = 40, seed: int = 0, n: int | None = None,
                    device=None, batch: int = 32, force: bool = False,
                    **spec_overrides) -> ShardedTrajectories:
    """Generate (or reuse) a sharded 2D dataset.

    Defaults give ~10^4 one-step pairs per split at 128x128, which is the scale
    at which a 10-50M parameter surrogate actually trains rather than memorizes.
    """
    spec = SPECS2D[pde]
    if n is not None:
        spec = PDESpec2D(**{**asdict(spec), "n": n})
    if spec_overrides:
        spec = PDESpec2D(**{**asdict(spec), **spec_overrides})
    root = dataset_dir(pde, spec.n, seed, spec.stride)
    if (root / "manifest.json").exists() and not force:
        return ShardedTrajectories(root)

    root.mkdir(parents=True, exist_ok=True)
    counts = {"train": n_train, "val": n_val, "test": n_test}
    mu_acc, sq_acc, cnt = 0.0, 0.0, 0
    for i, (name, cnt_i) in enumerate(counts.items()):
        gb = cnt_i * (n_steps + 1) * spec.n ** 2 * 4 / 1e9
        final = root / f"{name}.npy"
        want = (cnt_i, n_steps + 1, spec.n, spec.n)
        # Splits are written to a .part file and renamed only once complete, so
        # the presence of the final name means the data is whole. Generation at
        # 128x128 takes tens of minutes per split; without this, dying anywhere
        # after the first split (as happens under a tight memory limit) throws
        # away everything already computed.
        reuse = False
        if final.exists() and not force:
            try:
                reuse = np.load(final, mmap_mode="r").shape == want
            except Exception:
                reuse = False
            if not reuse:
                final.unlink(missing_ok=True)

        if reuse:
            print(f"  [{pde}] {name}: already complete, reusing", flush=True)
        else:
            print(f"  [{pde}] generating {name}: {cnt_i} trajectories x {n_steps} "
                  f"steps at {spec.n}x{spec.n}  ({gb:.2f} GB, streamed to disk)",
                  flush=True)
            part = root / f"{name}.npy.part"
            del_traj = simulate(spec, cnt_i, n_steps, seed=seed + 1000 * i,
                                device=device, batch=batch, out_path=part)
            del del_traj                       # close the memmap before renaming
            part.replace(final)
        traj = np.load(final, mmap_mode="r")
        if name == "train":
            # One trajectory at a time, accumulating in float64 without ever
            # materializing a float64 copy of a block. `.mean()` over the whole
            # memmap would fault the entire split back in, and casting a
            # batch-sized block to float64 doubles it again -- enough to be
            # SIGKILLed under a 1 GB cgroup, which is what a SLURM job with no
            # --mem gets. Per-trajectory temporaries are a few MB.
            tot = sq = 0.0
            for s in range(traj.shape[0]):
                blk = np.asarray(traj[s])                      # float32 view
                tot += float(blk.sum(dtype=np.float64))
                sq += float(np.square(blk, dtype=np.float64).sum())
            cnt = int(np.prod(traj.shape))
            mu_acc = tot / cnt
            sq_acc = sq / cnt
        del traj
    sigma = math.sqrt(max(sq_acc - mu_acc ** 2, 1e-12))
    with open(root / "manifest.json", "w") as fh:
        json.dump({"spec": asdict(spec), "counts": counts, "n_steps": n_steps,
                   "seed": seed, "mu": mu_acc, "sigma": sigma,
                   "n_train_elems": cnt}, fh, indent=2)
    return ShardedTrajectories(root)


# ---------------------------------------------------------------------------
# PDEBench / external benchmark loader
# ---------------------------------------------------------------------------

def load_pdebench(path: str | Path, field: str = "velocity",
                  max_traj: int | None = None) -> dict:
    """Load a PDEBench HDF5 file into the same interface as the generated data.

    PDEBench layouts vary by system; the two common ones are a top-level dataset
    of shape (n_traj, n_t, n_x, n_y[, c]) and per-trajectory groups '0000',
    '0001', ... each holding the fields. Both are handled. Which file you want:

        2D_CFD_*.hdf5              compressible NS, fields density/pressure/Vx/Vy
        2D_diff-react_*.hdf5       two-species reaction-diffusion
        ns_incom_inhom_2d_*.h5     incompressible NS

    Downloads live at the DaRUS repository linked from github.com/pdebench/PDEBench.
    This is the path for claiming results on an established benchmark rather
    than only on self-generated data, which reviewers will ask for.
    """
    try:
        import h5py
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("PDEBench files need h5py: pip install h5py") from exc

    path = Path(path)
    with h5py.File(path, "r") as fh:
        keys = list(fh.keys())
        if field in fh:
            arr = np.asarray(fh[field][:max_traj])
        elif keys and isinstance(fh[keys[0]], h5py.Group):
            groups = keys[:max_traj]
            sub = field if field in fh[groups[0]] else list(fh[groups[0]].keys())[0]
            arr = np.stack([np.asarray(fh[g][sub]) for g in groups])
        else:
            arr = np.asarray(fh[keys[0]][:max_traj])
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[-1] <= 4:         # (traj, t, x, y, c) -> channels first
        arr = np.moveaxis(arr, -1, 2)
    return {"data": arr, "shape": arr.shape, "source": str(path)}


def main():                                          # pragma: no cover
    import argparse
    from ..utils import get_device
    ap = argparse.ArgumentParser(description="Generate 2D PDE datasets")
    ap.add_argument("--pde", default="ns2d_forced", choices=list(SPECS2D) + ["all"])
    ap.add_argument("--n", type=int, default=128, help="grid resolution (n x n)")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--stride", type=int, default=None,
                    help="solver steps between stored snapshots, i.e. the "
                         "model's time step. Defaults to the spec; see SPECS2D "
                         "for why this is the parameter the whole study is most "
                         "sensitive to. Pass a legacy value to reproduce the "
                         "near-identity regime as a negative control.")
    ap.add_argument("--batch", type=int, default=32, help="trajectories per GPU batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"generating on {device}")
    for pde in (list(SPECS2D) if args.pde == "all" else [args.pde]):
        over = {"stride": args.stride} if args.stride else {}
        ds = build_dataset2d(pde, n_train=args.n_train, n_val=args.n_val,
                             n_test=args.n_test, n_steps=args.n_steps, n=args.n,
                             seed=args.seed, device=device, batch=args.batch,
                             force=args.force, **over)
        print(f"  -> {ds.root}  N={ds.spec.N}  dt_out={ds.spec.dt_out:g}  "
              f"train pairs={ds.n_pairs('train')}")


if __name__ == "__main__":                           # pragma: no cover
    main()
