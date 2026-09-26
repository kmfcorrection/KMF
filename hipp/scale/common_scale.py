"""Shared scaffolding for the scaled experiments, mirroring `hipp/common.py`.

Every `experiments/scale/s*.py` starts with `setup_scale(args)` so the stages
are directly comparable to each other and, where the quantity is the same, to
the 1D stages.

The one structural difference from `hipp/common.py` is that nothing is held in
memory. `build_zoo_streaming` evaluates one state at a time and keeps only the
`QuadSummary` and the marginal report, because a rank-64 basis at N = 16,384 is
8 MB and the 1D version's habit of building a list of priors for every method
would need hundreds of gigabytes.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..utils import get_device, set_seed
from .adapters import ADAPTERS, POSEIDON_FLUID_STD, from_local_checkpoint
from .baselines_scale import BASELINES_SCALE, build_baseline_fn, fit_residual_stats
from .calibrate_scale import fit_alpha_scale
from .curvature_scale import ESTIMATORS_SCALE, estimate_scale
from .jacobian import set_probe_precision
from .data2d import SPECS2D, build_dataset2d
from .metrics_scale import aggregate, marginal_report
from .train2d import ckpt_path

DEFAULT_ESTIMATORS_SCALE = ["pushforward", "gn", "diag_gn", "identity"]


def base_parser_scale(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--pde", default="ns2d_forced", choices=list(SPECS2D))
    ap.add_argument("--arch", default="fno2d", choices=["fno2d", "unet2d"])
    ap.add_argument("--grid", type=int, default=None,
                    help="resolution; defaults to whatever the checkpoint used")
    ap.add_argument("--stride", type=int, default=None,
                    help="output cadence of the dataset to use; defaults to the "
                         "spec. Must match what the checkpoint was trained on.")
    ap.add_argument("--ckpt", default=None,
                    help="explicit checkpoint path, overriding --pde/--arch")
    ap.add_argument("--fm", default="local", choices=sorted(ADAPTERS),
                    help="which frozen model to load. 'local' is a checkpoint "
                         "from hipp.scale.train2d; 'poseidon' is verified "
                         "against camlab-ethz/Poseidon-{T,B}. Run "
                         "experiments/scale/check_fm.py against any pretrained "
                         "checkpoint before trusting a number from it.")
    ap.add_argument("--fm-size", default="B",
                    help="model size for adapters that have one "
                         "(Poseidon: T/B/L; DPOT: Ti/S/M/L/H; MORPH: Ti/S/M)")
    ap.add_argument("--fm-checkpoint", default=None,
                    help="explicit pretrained checkpoint/repo id for adapters "
                         "whose public weights are not encoded by --fm-size")
    ap.add_argument("--fm-dataset", default="active_matter",
                    help="dataset name for dataset-specific public checkpoints "
                         "such as Polymathic/The Well models")
    ap.add_argument("--fm-family", default="FNO",
                    help="model family inside a multi-checkpoint source "
                         "(The Well: FNO/TFNO/UNetClassic/UNetConvNext)")
    ap.add_argument("--fm-history", type=int, default=None,
                    help="history length for autoregressive FMs. If omitted, "
                         "the adapter's native default is used.")
    ap.add_argument("--lead-time", type=float, default=1.0,
                    help="output cadence for time-conditioned FMs, in native "
                         "dataset snapshots (Poseidon: trajectories span 20). "
                         "This is the --stride of a pretrained model, and the "
                         "axis s5 shows the whole method depends on.")
    ap.add_argument("--fm-channels", default="all",
                    help="'all', 'velocity', or a comma-separated channel list. "
                         "Which channels of the FM's state are the uncertain "
                         "object; the rest are pinned at their physical "
                         "constants. Poseidon's incompressible corpus holds "
                         "rho and p constant, so 'velocity' is the honest "
                         "choice there -- see adapters.from_poseidon.")
    ap.add_argument("--k", type=int, default=64,
                    help="retained rank. The method's claim is that the "
                         "operator's geometry is k-dimensional to a useful "
                         "approximation; s4 sweeps this.")
    ap.add_argument("--tau-rel", type=float, default=1e-2, help="relative damping")
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--n-cal", type=int, default=256,
                    help="states used to fit alpha (a mean of a heavy-tailed "
                         "quantity, so not small)")
    ap.add_argument("--n-fit", type=int, default=512,
                    help="states used to fit baseline residual statistics")
    ap.add_argument("--alpha-objective", default="mle", choices=["mle", "median"])
    ap.add_argument("--oversample", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=2,
                    help="power iterations in the randomized SVD. 0 is not safe "
                         "for a residual operator; see s0.")
    ap.add_argument("--chunk", type=int, default=8,
                    help="probes per vmap batch; the memory/throughput knob")
    ap.add_argument("--allow-tf32", action="store_true",
                    help="keep TF32 enabled for the probe paths. Off by "
                         "default: TF32 has ~1e-3 relative precision, and since "
                         "the surrogate is residual (||J-I||/||I|| ~ 0.025) that "
                         "is ~4%% noise on the only part of J that carries "
                         "information. See jacobian.set_probe_precision.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the results directory")
    return ap


def parse_fm_channels(spec: str | None) -> tuple[int, ...] | None:
    """'all' -> None (every channel), 'velocity' -> (1, 2), '0,3' -> (0, 3)."""
    if spec in (None, "", "all"):
        return None
    if spec == "velocity":
        return (1, 2)
    return tuple(int(c) for c in str(spec).split(","))


def load_fm(args, member: int = 0, dropout: float = 0.0, device=None):
    """Load the frozen model under test: local checkpoint or pretrained FM."""
    device = device or get_device(args.device)
    kind = getattr(args, "fm", "local")
    if kind == "local":
        path = Path(args.ckpt) if args.ckpt else ckpt_path(
            args.pde, args.arch, member=member, dropout=dropout)
        return from_local_checkpoint(path, device=device)
    if kind == "poseidon":
        n = args.grid or 128
        return ADAPTERS["poseidon"](
            model_size=getattr(args, "fm_size", "B"), device=device,
            resolution=n, lead_time=getattr(args, "lead_time", 1.0),
            free_channels=parse_fm_channels(getattr(args, "fm_channels", "all")))
    if kind == "dpot":
        n = args.grid or 128
        kwargs = {}
        if getattr(args, "fm_history", None):
            kwargs["history"] = args.fm_history
        return ADAPTERS["dpot"](
            model_size=getattr(args, "fm_size", "Ti"), device=device,
            resolution=n, checkpoint=getattr(args, "fm_checkpoint", None),
            free_channels=parse_fm_channels(getattr(args, "fm_channels", "all")),
            **kwargs)
    if kind == "morph":
        n = args.grid or 128
        channels = getattr(args, "fm_channels", "velocity")
        if channels not in ("velocity", "0,1", "1,2"):
            raise ValueError(
                "MORPH's current adapter represents a 2-D vector state; use "
                "--fm-channels velocity (or 0,1), not %r" % channels)
        return ADAPTERS["morph"](
            model_size=getattr(args, "fm_size", "Ti"), device=device,
            resolution=n, checkpoint=getattr(args, "fm_checkpoint", None))
    if kind == "cno":
        n = args.grid or 128
        return ADAPTERS["cno"](
            model_size=getattr(args, "fm_size", "FM"), device=device, resolution=n,
            checkpoint=getattr(args, "fm_checkpoint", None),
            config=getattr(args, "cno_config", None),
            source_dir=getattr(args, "cno_source", None),
            lead_time=getattr(args, "lead_time", 1.0),
            time_scale=getattr(args, "cno_time_scale", 1.0),
            pinned_state=(getattr(args, "cno_rho", 0.8), getattr(args, "cno_pressure", 0.0)))
    if kind == "motion":
        n = args.grid or 128
        return ADAPTERS["motion"](
            model_size=getattr(args, "fm_size", "156.9M"), device=device, resolution=n,
            checkpoint=getattr(args, "fm_checkpoint", None),
            config=getattr(args, "motion_config", None),
            source_dir=getattr(args, "motion_source", None),
            lead_time=getattr(args, "lead_time", 1.0),
            time_scale=getattr(args, "motion_time_scale", 1.0))
    if kind == "walrus":
        n = args.grid or 128
        hist = getattr(args, "fm_history", None) or 6
        return ADAPTERS["walrus"](
            (1, n, n), checkpoint=getattr(args, "fm_checkpoint", "polymathic-ai/walrus"),
            device=device, history=hist)
    if kind in ("the_well", "well"):
        n = args.grid or 128
        hist = getattr(args, "fm_history", None) or 4
        return ADAPTERS[kind](
            model_family=getattr(args, "fm_family", "FNO"),
            dataset=getattr(args, "fm_dataset", "active_matter"),
            checkpoint=getattr(args, "fm_checkpoint", None),
            resolution=n, history=hist, device=device)
    return ADAPTERS[kind](device=device)


def fm_probe_states(fm, n_states: int, kind: str = "fluid", seed: int = 0,
                    device=None, rollout_steps: int = 2) -> torch.Tensor:
    """States to probe the operator at, as `(n_states, N)`.

    Where J is evaluated is not a detail. A pretrained foundation model is a
    function of the whole input space but was only ever fit on a thin manifold
    of physical fields, and its Jacobian off that manifold is not a fact about
    the physics -- it is a fact about extrapolation. Probing at `randn` is the
    mistake that makes a real model look either trivially near-identity or
    wildly ill-conditioned, depending on the architecture, and neither reading
    transfers to the regime the method is claimed for.

      gaussian  iid N(0,1). Kept as an explicit off-manifold *control*, not as
                a default, so the gap to the on-manifold numbers is measurable.
      fluid     divergence-free velocity from a McWilliams-style random field,
                `E(k) ~ k^4/(k+k_peak)^8`, scaled to the corpus per-channel RMS.
                Smooth, isotropic and spectrally like the training data.
      rollout   `fluid` pushed `rollout_steps` steps through the model itself.
                The strongest notion of on-manifold available without the
                pretraining corpus, and the regime the method targets, since
                rollout states are exactly what an autoregressive predictive
                distribution conditions on.
    """
    from .data2d import PDESpec2D, SpectralGrid2D, random_initial_vorticity

    device = device or fm.device
    c, h, w = fm.state_shape
    g = torch.Generator(device="cpu").manual_seed(seed)

    if kind == "gaussian":
        x = torch.randn(n_states, fm.N, generator=g)
        return x.to(device=device, dtype=fm.dtype)

    spec = PDESpec2D("probe", n=h, ic_peak_k=4.0)
    grid = SpectralGrid2D(spec, device="cpu", dtype=torch.float64)
    vort = random_initial_vorticity(spec, n_states, generator=g, device="cpu",
                                    dtype=torch.float64, grid=grid)
    u, v = grid.velocity(torch.fft.rfft2(vort))

    # MORPH represents its physical state directly as [u,v], whereas
    # Poseidon/DPOT use their published four-channel fluid convention where
    # velocity lives at channels 1 and 2.  Do not reuse the latter's channel
    # indices for MORPH: it would silently replace u by a constant density.
    if getattr(fm, "state_layout", None) == "velocity_uv":
        fields = torch.stack([u, v], dim=1)
    else:
        free = getattr(fm, "free_channels", None)
        if free is None and c == 1:
            fields = vort.unsqueeze(1)                     # local surrogates: omega
        else:
            free = free if free is not None else tuple(range(c))
            # One scale factor for both velocity components, not one each: u and v
            # come from the Biot-Savart inverse and are therefore divergence-free,
            # and rescaling them independently would break that. The incompressible
            # corpus contains divergence-free fields, so an input that is not one is
            # off-manifold in exactly the way this function exists to avoid.
            target = 0.5 * (POSEIDON_FLUID_STD[1] + POSEIDON_FLUID_STD[2])
            rms = torch.stack([u, v], 1).flatten(1).std(dim=1).clamp(min=1e-12)
            u, v = [f * (target / rms).view(-1, 1, 1) for f in (u, v)]
            cols = []
            for ch in free:
                if ch in (1, 2):
                    cols.append(u if ch == 1 else v)
                else:                       # rho / p: constant in the fluid corpus
                    cols.append(torch.full((n_states, h, w), 1.0 if ch == 0 else 0.0,
                                           dtype=torch.float64))
            fields = torch.stack(cols, dim=1)

    x = fields.reshape(n_states, -1).to(device=device, dtype=fm.dtype)
    if kind == "fluid":
        return x
    if kind != "rollout":
        raise ValueError(f"unknown state kind {kind!r}")
    f = fm.flat_fn()
    with torch.no_grad():
        for _ in range(rollout_steps):
            x = torch.stack([f(row) for row in x])
    return x


def setup_scale(args, need_ensemble: bool = False, need_dropout: bool = False,
                verbose: bool = True) -> dict:
    """Load model + data and carve out calibration / fit / test splits."""
    set_seed(args.seed)
    set_probe_precision(high=not getattr(args, "allow_tf32", False))
    device = get_device(args.device)
    fm = load_fm(args, device=device)
    over = {'stride': args.stride} if getattr(args, 'stride', None) else {}
    ds = build_dataset2d(args.pde, n=args.grid, seed=0, device=device, **over)

    rng = np.random.default_rng(args.seed)
    # The calibration split (which fits alpha) and the statistics split (which
    # fits the baselines' residual covariance) are carved from *disjoint
    # trajectories*, not from a shared pool of one-step pairs.
    #
    # This is not fastidiousness. Pairs from the same trajectory are strongly
    # correlated -- consecutive states differ by one model step -- so a rank-k
    # basis fitted on a few pairs of a trajectory explains almost all the
    # residual energy of every other pair of that same trajectory. Drawing both
    # splits from a shared pool therefore leaks: the fixed-global baseline looks
    # near-perfect on the calibration states, alpha is fitted to that inflated
    # fit, and the test split (genuinely different trajectories) then reports a
    # catastrophic miscalibration that is an artifact of the split, not of the
    # method. Measured here at 32x32: held-out-pair energy in the fitted basis
    # was 0.94 within-trajectory versus 0.045 across-trajectory.
    n_val_traj = ds.split("val").shape[0]
    perm = rng.permutation(n_val_traj)
    cal_traj, fit_traj = perm[:n_val_traj // 2], perm[n_val_traj // 2:]
    x_cal, y_cal = _draw(ds, "val", args.n_cal, rng, device, trajs=cal_traj)
    x_fit, y_fit = _draw(ds, "val", args.n_fit, rng, device, trajs=fit_traj)
    x_test, y_test = _draw(ds, "test", args.n_test, rng, device)

    ctx = {"fm": fm, "spec": ds.spec, "dataset": ds, "device": device,
           "pde": args.pde, "N": fm.N,
           "x_cal": x_cal, "y_cal": y_cal, "x_fit": x_fit, "y_fit": y_fit,
           "x_test": x_test, "y_test": y_test}

    if need_ensemble:
        members = []
        for m in range(8):
            try:
                members.append(load_fm(args, member=m, device=device))
            except FileNotFoundError:
                break
        ctx["ensemble"] = members
    if need_dropout:
        try:
            ctx["fm_do"] = load_fm(args, dropout=0.1, device=device)
        except FileNotFoundError:
            ctx["fm_do"] = None

    if verbose:
        print(f"  model      {fm}")
        print(f"  data       {ds.root.name}  N={fm.N}  "
              f"grid={ds.spec.n}x{ds.spec.n}  dt_out={ds.spec.dt_out:g}")
        print(f"  splits     cal={len(x_cal)}  fit={len(x_fit)}  test={len(x_test)}")
        print(f"  tf32       {'ENABLED (--allow-tf32)' if getattr(args, 'allow_tf32', False) else 'disabled'}"
              f"   [probe precision]")
        dense_gb = fm.N ** 2 * 8 / 1e9
        print(f"  a dense NxN covariance here would be {dense_gb:.1f} GB "
              f"and need {fm.N} backward passes per state; "
              f"rank {args.k} needs {(2 + 3*args.n_iter) * (args.k + args.oversample)}")
    return ctx


def _draw(ds, split: str, n: int, rng, device, trajs=None):
    """Sample n one-step pairs, optionally restricted to given trajectories."""
    arr = ds.split(split)
    t_len = arr.shape[1] - 1
    if trajs is None:
        pool = np.arange(arr.shape[0] * t_len)
    else:
        pool = np.concatenate([np.arange(int(i) * t_len, (int(i) + 1) * t_len)
                               for i in trajs])
    idx = rng.choice(pool, size=min(n, pool.size), replace=False)
    xs, ys = [], []
    for i in idx:
        x, y = ds.pair(split, int(i), device=device)
        xs.append(x.reshape(-1))
        ys.append(y.reshape(-1))
    return xs, ys


# ---------------------------------------------------------------------------
# The method zoo, streamed
# ---------------------------------------------------------------------------

def build_zoo_streaming(args, ctx, estimators=None, baselines=None,
                        verbose: bool = True) -> dict:
    """Calibrate and evaluate every method, one state at a time.

    Returns {name: {alpha, alpha_diag, summaries, marginals, kind, seconds}}
    where `summaries` are the streamed `QuadSummary` objects on the *test*
    split -- enough for every joint calibration metric -- and `marginals` are
    the per-state per-pixel reports.

    Two passes per method: one over the calibration split to fit alpha, one over
    the test split to score. Both are streaming, so peak memory is one prior.
    """
    estimators = DEFAULT_ESTIMATORS_SCALE if estimators is None else estimators
    baselines = BASELINES_SCALE if baselines is None else baselines
    fm = ctx["fm"]
    zoo: dict = {}

    kw = dict(k=args.k, tau_rel=args.tau_rel, oversample=args.oversample,
              n_iter=args.n_iter, chunk=args.chunk)

    for meth in estimators:
        t0 = time.time()
        if verbose:
            print(f"  curvature: {meth:16s} ...", end="", flush=True)
        build = lambda c, m=meth: estimate_scale(fm, c, method=m, **kw).prior
        entry = _calibrate_and_score(build, ctx, args)
        entry.update(kind="curvature", seconds=time.time() - t0)
        zoo[meth] = entry
        if verbose:
            print(f" alpha={entry['alpha']:.4g}  "
                  f"floor share={entry['alpha_diag'].get('floor_share', float('nan')):.3f}  "
                  f"({entry['seconds']:.0f}s)", flush=True)

    if not baselines:
        return zoo

    bctx = {"fm": fm, "k": args.k}
    if any(b in baselines for b in ("diagonal_fitted", "fixed_global_lowrank")):
        if verbose:
            print(f"  fitting residual statistics on {len(ctx['x_fit'])} states ...",
                  flush=True)
        bctx["res_stats"] = fit_residual_stats(fm, ctx["x_fit"], ctx["y_fit"],
                                               k=args.k, device=ctx["device"])
    if "deep_ensemble" in baselines:
        bctx["ensemble"] = ctx.get("ensemble") or []
        if len(bctx["ensemble"]) < 2:
            baselines = [b for b in baselines if b != "deep_ensemble"]
    if "mc_dropout" in baselines:
        bctx["fm_do"] = ctx.get("fm_do")
        if bctx["fm_do"] is None:
            baselines = [b for b in baselines if b != "mc_dropout"]
    if "swag" in baselines:
        swag = _load_swag(args, ctx)
        if swag is None:
            baselines = [b for b in baselines if b != "swag"]
        else:
            bctx["swag"] = swag

    for name in baselines:
        t0 = time.time()
        if verbose:
            print(f"  baseline:  {name:16s} ...", end="", flush=True)
        entry = _calibrate_and_score(build_baseline_fn(name, bctx), ctx, args)
        entry.update(kind="baseline", seconds=time.time() - t0)
        zoo[name] = entry
        if verbose:
            print(f" alpha={entry['alpha']:.4g}  ({entry['seconds']:.0f}s)", flush=True)
    return zoo


def _calibrate_and_score(build_fn, ctx, args) -> dict:
    cal = [build_fn(c).summarize(y)
           for c, y in zip(ctx["x_cal"], ctx["y_cal"])]
    alpha, diag = fit_alpha_scale(cal, objective=args.alpha_objective,
                                  return_diagnostics=True)
    summaries, marginals = [], []
    for c, y in zip(ctx["x_test"], ctx["y_test"]):
        p = build_fn(c).rescaled(alpha)
        summaries.append(p.summarize(y))
        marginals.append(marginal_report(p, y))
    return {"alpha": alpha, "alpha_diag": diag, "summaries": summaries,
            "marginals": marginals, "marginal_agg": aggregate(marginals)}


def _load_swag(args, ctx):
    path = Path(args.ckpt) if args.ckpt else ckpt_path(args.pde, args.arch, 0)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("swag")


def estimator_choices_scale() -> list[str]:
    return list(ESTIMATORS_SCALE)


def results_path_scale(stage: str, pde: str, name: str, tag: str = "") -> Path:
    from ..utils import RESULTS_DIR
    p = RESULTS_DIR / "scale" / stage / (pde + (f"_{tag}" if tag else ""))
    p.mkdir(parents=True, exist_ok=True)
    return p / name
