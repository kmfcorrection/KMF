"""Shared utilities for evaluating pretrained PDE foundation models.

The original scale stages use a cached local one-channel vorticity dataset.
Poseidon and DPOT expose different state interfaces, so the FM stages build
paired states directly in the adapter's state space:

  * Poseidon velocity mode:          [u, v]
  * Poseidon/DPOT all-channel mode:  [rho, u, v, p] with rho=1, p=0
  * Local one-channel fallback:      [omega]

This is deliberately honest about the current target distribution: until the
public pretraining corpora are loaded, these are FM evaluations on our
Navier--Stokes probe distribution, not on each model's exact training split.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from hipp.scale.adapters import POSEIDON_FLUID_STD
from hipp.scale.calibrate_scale import fit_alpha_scale
from hipp.scale.curvature_scale import estimate_scale
from hipp.scale.data2d import PDESpec2D, SPECS2D, SpectralGrid2D, simulate
from hipp.scale.metrics_scale import aggregate, marginal_report


# NS-Gauss stores 21 raw frames on [0,1], hence raw_dt=0.05.  Poseidon's
# standard incompressible dataset constructor uses ``time_step_size=2`` (see
# scOT.problems.base), so its in-distribution single transition spans two raw
# frames: raw index difference 2, physical time 0.1, and model time 2/20.
# Calling the checkpoint on adjacent raw frames at time=1/20 is an off-training
# cadence interpolation, not its native one-step prediction problem.
POSEIDON_RAW_DT = 1.0 / 20.0
POSEIDON_NATIVE_RAW_STRIDE = 2
POSEIDON_REFERENCE_DT = POSEIDON_RAW_DT * POSEIDON_NATIVE_RAW_STRIDE


def configure_fm_cadence(args) -> float:
    """Set the FM time input to the physical interval represented by a target.

    Poseidon's adapter accepts *dataset index units*: one unit is one of the
    20 equal intervals on [0,1], hence 0.05 physical time.  Equating an index
    unit to an arbitrary synthetic solver snapshot was a silent cadence bug.
    """
    base = SPECS2D[args.pde]
    stride = int(args.stride if getattr(args, "stride", None) is not None else base.stride)
    dt_target = base.dt * stride * max(1, int(getattr(args, "lead_steps", 1)))
    if args.fm == "poseidon":
        args.lead_time = dt_target / POSEIDON_REFERENCE_DT
    return dt_target


def configure_native_poseidon_cadence(args) -> float:
    """Configure Poseidon's *trained* native cadence on NS-Gauss.

    ``lead_steps`` counts Poseidon transitions, each of which spans two raw
    NetCDF frames.  The adapter's ``lead_time`` is measured in raw dataset
    indices, as required by scOT's ``time = (t2-t1)/20`` convention.
    """
    dt_target = POSEIDON_REFERENCE_DT * max(1, int(args.lead_steps))
    if args.fm == "poseidon":
        args.lead_time = float(
            POSEIDON_NATIVE_RAW_STRIDE * max(1, int(args.lead_steps)))
    return dt_target


def native_poseidon_spec(args, fm) -> PDESpec2D:
    """Published incompressible Poseidon grid/time convention.

    The released solver uses high-mode spectral viscosity.  `nu` records its
    nominal magnitude; S3 needs only the trajectory, while S4 labels its current
    constant-viscosity residual as an approximation until the exact AZEBAN
    multiplier is reproduced.
    """
    return PDESpec2D("poseidon_ns_native", L=1.0, n=int(fm.state_shape[-1]),
                     nu=0.0, dt=POSEIDON_REFERENCE_DT, stride=1,
                     forcing="none", warmup=0, ic_peak_k=4.0)


def load_poseidon_trajectories(path: str | Path, fm, n_traj: int, steps: int,
                               lead_steps: int = 1, offset: int = 0):
    """Load physical [u,v] trajectories from an assembled or chunk NetCDF/HDF5.

    Expected variables are `velocity` (NS-Gauss/NS-* datasets) or `solution`
    (FNS-KF), with shape (sample,time,channel,128,128). ``lead_steps`` counts
    Poseidon-native transitions, each spanning two raw NS-Gauss snapshots.
    No normalization or per-snapshot rescaling is applied; the FM adapter
    performs its own channel normalization internally.
    """
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"native Poseidon data not found: {path}")
    raw_stride = POSEIDON_NATIVE_RAW_STRIDE * int(lead_steps)
    if raw_stride < 1:
        raise ValueError(f"lead_steps must be positive, got {lead_steps}")
    need_t = steps * raw_stride + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution" if "solution" in h else None
        if key is None:
            raise KeyError(f"{path} has neither a 'velocity' nor 'solution' variable")
        ds = h[key]
        if ds.ndim != 5 or ds.shape[2] < 2:
            raise ValueError(f"{key} has shape {ds.shape}; expected (sample,time,channel,x,y)")
        if need_t > ds.shape[1]:
            raise ValueError(f"need {need_t} snapshots but {path} contains {ds.shape[1]}")
        if offset < 0 or offset + n_traj > ds.shape[0]:
            raise ValueError(f"trajectory slice [{offset}:{offset+n_traj}] outside {ds.shape[0]}")
        arr = np.asarray(ds[offset:offset+n_traj, :need_t:raw_stride, :2],
                         dtype=np.float32)
    x = torch.as_tensor(arr, device=fm.device, dtype=fm.dtype)
    if tuple(x.shape[-2:]) != tuple(fm.state_shape[-2:]):
        raise ValueError(f"native data grid {tuple(x.shape[-2:])} != FM grid {fm.state_shape[-2:]}")
    return x.reshape(n_traj, steps + 1, -1)


def fm_result_key(args) -> str:
    bits = [str(args.fm), str(args.fm_size), str(args.fm_channels)]
    if getattr(args, "lead_steps", None) is not None:
        bits.append(f"lead{args.lead_steps:g}")
    return "_".join(b.replace(",", "-") for b in bits)


def fm_metadata(args, fm, spec: PDESpec2D) -> dict:
    meta = {
        "fm": args.fm,
        "fm_size": args.fm_size,
        "fm_channels": args.fm_channels,
        "state_shape": tuple(int(x) for x in fm.state_shape),
        "N": int(fm.N),
        "source": getattr(fm, "source", ""),
        "target_distribution": (
            "synthetic Navier-Stokes trajectories generated by hipp.scale.data2d; "
            "not the model's original pretraining corpus"
        ),
        "pde_spec": asdict(spec),
        "physical_dt_target": spec.dt_out * max(1, int(getattr(args, "lead_steps", 1))),
        "trajectory_scaling": getattr(args, "trajectory_scaling", "unspecified"),
    }
    if hasattr(fm, "lead_time"):
        meta["fm_lead_time"] = float(fm.lead_time)
    if (getattr(args, "data_source", "synthetic") == "poseidon-native"
            and args.fm != "poseidon"):
        meta["transfer_timestep_convention"] = (
            "One fixed-step external-FM output is evaluated against one native "
            "Poseidon NS-Gauss transition (physical dt=0.1). This is an explicit "
            "out-of-distribution comparison convention, not a claim about the "
            "external model's original training cadence.")
    if hasattr(fm, "dpot_history_policy"):
        meta["dpot_history_policy"] = fm.dpot_history_policy
        meta["dpot_caveat"] = (
            "The public DPOT model consumes a history window. This adapter "
            "currently repeats the current state through that history window; "
            "useful for the geometry/calibration gate, but final predictive "
            "claims should be rerun with true 10-frame histories."
        )
    if hasattr(fm, "morph_normalization_policy"):
        meta["morph_normalization_policy"] = fm.morph_normalization_policy
        meta["morph_caveat"] = (
            "MORPH's released RevIN is trajectory-level. The adapter uses only "
            "the observed current frame, so no future statistics leak into the "
            "map. It is appropriate for S1--S3 geometry checks, not native S4 "
            "physical-correction claims without matched MORPH data and metadata.")
    return meta


def make_spec(args, fm) -> PDESpec2D:
    base = SPECS2D[args.pde]
    overrides = {"n": int(fm.state_shape[-1])}
    if getattr(args, "stride", None) is not None:
        overrides["stride"] = int(args.stride)
    return PDESpec2D(**{**asdict(base), **overrides})


def generate_fm_trajectories(args, fm, n_traj: int, steps: int,
                             seed: int | None = None,
                             preserve_physics: bool = False) -> tuple[torch.Tensor, PDESpec2D]:
    """Return trajectories in FM state space, shape (n_traj, steps+1, N)."""
    spec = make_spec(args, fm)
    args.trajectory_scaling = ("physical-none" if preserve_physics
                               else "per-snapshot-corpus-rms")
    vort_np = simulate(
        spec, n_traj=n_traj, n_steps=steps,
        seed=args.seed if seed is None else int(seed),
        device=fm.device, batch=max(1, min(getattr(args, "sim_batch", 8), n_traj)),
        dtype=torch.float64, progress=True,
    )
    vort = torch.as_tensor(vort_np, dtype=torch.float64, device=fm.device)
    n_traj, n_steps_plus, n, _ = vort.shape
    c, _, _ = fm.state_shape

    free = getattr(fm, "free_channels", None)
    if c == 1 and (free is None or free == (0,)):
        fields = vort.unsqueeze(2)
    else:
        grid = SpectralGrid2D(spec, device=fm.device, dtype=torch.float64)
        flat_vort = vort.reshape(n_traj * n_steps_plus, n, n)
        u, v = grid.velocity(torch.fft.rfft2(flat_vort))
        u = u.reshape(n_traj, n_steps_plus, n, n)
        v = v.reshape(n_traj, n_steps_plus, n, n)

        if not preserve_physics:
            # Geometry probes use the scale of Poseidon's fluid corpus.  This
            # is deliberately disabled for correction experiments: changing
            # the scale independently at every snapshot breaks the PDE in time.
            target = 0.5 * (POSEIDON_FLUID_STD[1] + POSEIDON_FLUID_STD[2])
            rms = torch.stack([u, v], dim=2).flatten(2).std(dim=2).clamp(min=1e-12)
            scale = (target / rms).view(n_traj, n_steps_plus, 1, 1)
            u, v = u * scale, v * scale

        if c == 2:
            fields = torch.stack([u, v], dim=2)
        elif c == 4:
            rho = torch.ones_like(u)
            p = torch.zeros_like(u)
            fields = torch.stack([rho, u, v, p], dim=2)
        else:
            raise ValueError(f"do not know how to build FM states with {c} channels")

    return fields.reshape(n_traj, n_steps_plus, -1).to(dtype=fm.dtype), spec


def draw_pairs(traj: torch.Tensor, n_pairs: int, lead_steps: int, seed: int):
    """Draw teacher-forced pairs from a trajectory tensor."""
    n_traj, n_steps_plus, _ = traj.shape
    t_len = n_steps_plus - lead_steps
    if t_len <= 0:
        raise ValueError(f"lead_steps={lead_steps} exceeds available trajectory length")
    total = n_traj * t_len
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=min(n_pairs, total), replace=False)
    xs, ys = [], []
    for z in idx:
        i, t = divmod(int(z), t_len)
        xs.append(traj[i, t].reshape(-1))
        ys.append(traj[i, t + lead_steps].reshape(-1))
    return xs, ys


def fit_and_score_method(args, fm, method: str, x_cal, y_cal, x_test, y_test,
                         verbose: bool = True) -> dict:
    kw = dict(k=args.k, tau_rel=args.tau_rel, oversample=args.oversample,
              n_iter=args.n_iter, chunk=args.chunk, n_tail=args.n_tail)
    t0 = time.time()
    cal_summaries = []
    for i, (x, y) in enumerate(zip(x_cal, y_cal), 1):
        est = estimate_scale(fm, x, method=method, **kw)
        cal_summaries.append(est.prior.summarize(y, extra={"split": "cal", "i": i}))
        if verbose:
            print(f"      cal {i}/{len(x_cal)}", flush=True)

    alpha, diag = fit_alpha_scale(
        cal_summaries, objective=args.alpha_objective,
        return_diagnostics=True, seed=args.seed,
    )

    summaries, marginals = [], []
    for i, (x, y) in enumerate(zip(x_test, y_test), 1):
        est = estimate_scale(fm, x, method=method, **kw)
        prior = est.prior.rescaled(alpha)
        summaries.append(prior.summarize(y, extra={"split": "test", "i": i}))
        marginals.append(marginal_report(prior, y))
        if verbose:
            print(f"      test {i}/{len(x_test)}", flush=True)

    return {
        "alpha": alpha,
        "alpha_diag": diag,
        "summaries": summaries,
        "marginal_agg": aggregate(marginals),
        "seconds": time.time() - t0,
    }


def split_for_s2(args, fm):
    lead = max(1, int(args.lead_steps))
    if getattr(args, "data_source", "synthetic") == "poseidon-native":
        if not getattr(args, "fm_data_path", None):
            raise ValueError("poseidon-native S2 requires --fm-data-path")
        # Keep calibration and test trajectory-disjoint. Pair selection occurs
        # only within each split, so alpha never sees a held-out trajectory.
        pairs_per_traj = max(1, int(args.steps) + 1 - lead)
        n_cal_traj = max(1, math.ceil(int(args.n_cal) / pairs_per_traj))
        n_test_traj = max(1, math.ceil(int(args.n_test) / pairs_per_traj))
        cal = load_poseidon_trajectories(args.fm_data_path, fm, n_cal_traj,
                                         args.steps, lead_steps=1, offset=0)
        test = load_poseidon_trajectories(args.fm_data_path, fm, n_test_traj,
                                          args.steps, lead_steps=1, offset=n_cal_traj)
        x_cal, y_cal = draw_pairs(cal, args.n_cal, lead, seed=args.seed + 17)
        x_test, y_test = draw_pairs(test, args.n_test, lead, seed=args.seed + 18)
        args.trajectory_scaling = "native-physical-none"
        return x_cal, y_cal, x_test, y_test, native_poseidon_spec(args, fm)
    needed = int(args.n_cal) + int(args.n_test)
    pairs_per_traj = max(1, int(args.steps) + 1 - lead)
    n_traj = max(int(args.n_traj), math.ceil(needed / pairs_per_traj))
    traj, spec = generate_fm_trajectories(
        args, fm, n_traj=n_traj, steps=args.steps, preserve_physics=True)
    x_all, y_all = draw_pairs(traj, needed, lead, seed=args.seed + 17)
    return x_all[:args.n_cal], y_all[:args.n_cal], x_all[args.n_cal:], y_all[args.n_cal:], spec
