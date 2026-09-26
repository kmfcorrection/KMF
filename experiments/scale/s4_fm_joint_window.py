#!/usr/bin/env python3
"""Joint-window residual HILP for frozen Poseidon rollouts.

Unlike one-step S4, this program forms a whole raw forecast window first and
then corrects the future states jointly.  Its physical term is the centered
vorticity equation on interior predicted frames:

    (omega_{n+1}-omega_{n-1})/(2 dt) - F_AZEBAN(omega_n).

The initial frame is observed and fixed.  No future ground-truth frame enters
the residual, prior construction, lambda selection, or MAP solve.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (configure_native_poseidon_cadence,
    fm_metadata, fm_result_key, load_poseidon_trajectories, native_poseidon_spec)
from hipp.scale.calibrate_scale import fit_alpha_scale
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.scale.jacobian import jacobian_ops
from hipp.scale.lowrank import LowRankGaussian, merge_lowrank, randn
from hipp.scale.posterior_scale import LowRankPhysicsPosterior, force_scale
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("isotropic", "joint_pushforward")


class JointWindowResidualEnergy:
    """Centered, differentiable NS residual over a future state window.

    Given a known x_0 and candidate z=(z_1,...,z_T), only interior temporal
    indices n=1,...,T-1 are scored.  This supplies a centered derivative using
    (z_{n-1}, z_n, z_{n+1}); z_T is constrained through the last interior
    residual and the trajectory prior.  The construction intentionally does
    not use an endpoint solver target.
    """
    def __init__(self, spec, x0, *, steps, dt, residual_scale, divergence_scale,
                 divergence_weight, device):
        if steps < 2:
            raise ValueError("joint residual needs --steps >= 2")
        self.spec, self.steps, self.dt = spec, int(steps), float(dt)
        self.N, self.device = 2 * spec.N, device
        self.residual_scale = float(residual_scale)
        self.divergence_scale = float(divergence_scale)
        self.divergence_weight = float(divergence_weight)
        if self.residual_scale <= 0 or self.divergence_scale <= 0:
            raise ValueError("joint residual scales must be positive")
        # This object supplies the published AZEBAN RHS and spectral operators.
        # Its stored previous state is irrelevant after construction.
        self.operator = FMPhysicsEnergy2D(spec, x0, 2, divergence_weight=0.0,
                                          dt=dt, device=device)
        self.x0 = x0.detach().to(device, torch.float64).reshape(1, self.N)

    def _window(self, z):
        z = z.to(self.device, torch.float64).reshape(-1, self.steps, self.N)
        if z.shape[0] != 1:
            raise ValueError("joint-window posterior currently solves one rollout at a time")
        return z

    def residual_fields(self, z):
        future = self._window(z)[0]
        states = torch.cat((self.x0, future), dim=0)
        vort = self.operator.to_vorticity(states)
        # Interior states 1,...,T-1.  The same published pseudospectral RHS is
        # used as in the native-flow audit, but no flow integration occurs.
        return ((vort[2:] - vort[:-2]) / (2.0 * self.dt) -
                self.operator._native_rhs(vort[1:-1]))

    def energy(self, z):
        future = self._window(z)[0]
        r = self.residual_fields(future)
        residual_sq = r.square().mean()
        div_sq = self.operator.divergence(future).square().mean()
        e = 0.5 * residual_sq / self.residual_scale ** 2
        e = e + 0.5 * self.divergence_weight * div_sq / self.divergence_scale ** 2
        return e.reshape(1)

    def grad(self, z):
        x = z.detach().to(self.device, torch.float64).reshape(1, -1).requires_grad_(True)
        (g,) = torch.autograd.grad(self.energy(x).sum(), x)
        return g

    def rms_residual(self, z):
        return self.residual_fields(z).square().mean().sqrt().reshape(1)

    def rms_divergence(self, z):
        future = self._window(z)[0]
        return self.operator.divergence(future).square().mean().sqrt().reshape(1)


def _raw_window(fm, x0, steps):
    """Frozen autoregressive mean path, with x0 retained separately."""
    states, c = [], x0.to(fm.device, fm.dtype)
    for _ in range(steps):
        c = fm.predict(c).detach()
        states.append(c.double().reshape(-1))
    return torch.stack(states)


def _joint_pushforward_prior(fm, x0, raw, sigma2, args, generator):
    """Tangent-ensemble covariance of the complete future trajectory.

    Each column follows delta z_1=e_1 and
    delta z_{t+1}=J_t delta z_t+e_{t+1}, with iid e_t~N(0,sigma²I).
    Stacking its states gives a Monte-Carlo low-rank factor of the *joint*
    rollout covariance, including cross-time blocks.  The scalar floor is
    deliberately matched in trace by the isotropic control.
    """
    m = int(args.tangent_samples or (args.k + args.oversample))
    if m < 2:
        raise ValueError("--tangent-samples must be at least 2")
    n = raw.shape[1]
    delta = math.sqrt(max(float(sigma2), 1e-30)) * randn(
        (n, m), fm.dtype, fm.device, generator)
    factors = [delta.double()]
    calls = 0
    for t in range(1, args.steps):
        # raw[t-1] is the conditioning state that produced raw[t].
        ops = jacobian_ops(fm.flat_fn(), raw[t - 1].to(fm.device, fm.dtype))
        innovation = math.sqrt(max(float(sigma2), 1e-30)) * randn(
            (n, m), fm.dtype, fm.device, generator)
        delta = ops.mm(delta, chunk=args.chunk) + innovation
        calls += ops.n_calls[0]
        factors.append(delta.double())
    # A A^T is the sample joint covariance.  The ridge is explicit rather than
    # hidden: it regularizes unseen tangent directions and its scale is fitted.
    A = torch.cat(factors, dim=0) / math.sqrt(m)
    trace_sample = float(A.square().sum())
    floor = max(float(args.joint_floor_rel) * trace_sample / A.shape[0], 1e-30)
    joint_rank = min(int(args.joint_rank or args.k), A.shape[1])
    U, d, tau = merge_lowrank([A], k=joint_rank, tau=floor)
    prior = LowRankGaussian(raw.reshape(-1).double(), U, d, tau,
                             label="joint_pushforward")
    return prior, {"tangent_samples": m, "joint_rank": prior.k,
                   "joint_floor": float(tau), "joint_sample_trace": trace_sample,
                   "joint_jvp_calls": calls}


def _prior(method, fm, x0, raw, sigma2, args, generator):
    mean = raw.reshape(-1).double()
    if method == "isotropic":
        # Match the pushforward sample covariance trace exactly.  This prevents
        # a sharper or broader control from masquerading as geometry.
        pf, info = _joint_pushforward_prior(fm, x0, raw, sigma2, args, generator)
        return LowRankGaussian.isotropic(mean, tau=pf.trace() / pf.N,
                                         label="joint_isotropic"), info
    if method == "joint_pushforward":
        return _joint_pushforward_prior(fm, x0, raw, sigma2, args, generator)
    raise ValueError(method)


def _calibration_scales(fm, cal, spec, sigma2, args):
    residuals, divergences = [], []
    for i in range(cal.shape[0]):
        raw = _raw_window(fm, cal[i, 0], args.steps)
        # Unit scales here merely expose raw physical units for robust fitting.
        e = JointWindowResidualEnergy(spec, cal[i, 0], steps=args.steps,
                                      dt=args.lead_steps * spec.dt_out,
                                      residual_scale=1.0, divergence_scale=1.0,
                                      divergence_weight=args.divergence_weight,
                                      device=fm.device)
        residuals.append(float(e.rms_residual(raw)))
        divergences.append(float(e.rms_divergence(raw)))
    return {"residual_scale": max(float(np.median(residuals)), 1e-30),
            "divergence_scale": max(float(np.median(divergences)), 1e-30),
            "calibration_residual_median": float(np.median(residuals)),
            "calibration_divergence_median": float(np.median(divergences))}


def _temporal_residual_floor_audit(fm, cal, spec, scales, args):
    """Measure the discretization floor of the window residual on calibration.

    The centered residual is not expected to vanish even on true consecutive
    samples: the released trajectories are separated by a finite time step and
    are generated by a discrete solver.  This audit therefore compares its
    RMS on the true window with the raw FM window using *calibration only*.
    A truth/raw ratio well below one says the residual can discriminate a
    physically better trajectory at the available temporal cadence.  It is an
    audit of the likelihood, not a quantity used to tune a held-out result.
    """
    truth_rms, raw_rms = [], []
    for i in range(cal.shape[0]):
        energy = JointWindowResidualEnergy(
            spec, cal[i, 0], steps=args.steps,
            dt=args.lead_steps * spec.dt_out,
            residual_scale=scales["residual_scale"],
            divergence_scale=scales["divergence_scale"],
            divergence_weight=args.divergence_weight, device=fm.device)
        truth = cal[i, 1:args.steps + 1].to(fm.device, torch.float64)
        raw = _raw_window(fm, cal[i, 0], args.steps)
        truth_rms.append(float(energy.rms_residual(truth)))
        raw_rms.append(float(energy.rms_residual(raw)))
    truth_med, raw_med = float(np.median(truth_rms)), float(np.median(raw_rms))
    return {"truth_residual_median": truth_med,
            "raw_residual_median": raw_med,
            "truth_over_raw": truth_med / max(raw_med, 1e-30),
            "truth_wins_fraction": float(np.mean(np.asarray(truth_rms) < np.asarray(raw_rms)))}


def _prepare(method, fm, trajectories, sigma2, scales, spec, args, reference=None):
    """Materialize one immutable joint prior/energy package per trajectory.

    This is intentionally outside the lambda loop.  Lambda changes only the
    posterior multiplier; rebuilding random tangent trajectories for each
    candidate would waste AD work and make a sweep needlessly stochastic.
    """
    packages = []
    for i in range(trajectories.shape[0]):
        print(f"    {method}: trajectory {i + 1}/{trajectories.shape[0]}", flush=True)
        if method == "isotropic" and reference is not None:
            # Exact trace-matched control from the already-built tangent
            # covariance. This avoids a second identical set of JVPs.
            raw, energy = reference[i]["raw"], reference[i]["energy"]
            pf = reference[i]["prior"]
            prior = LowRankGaussian.isotropic(raw.reshape(-1).double(),
                                               tau=pf.trace() / pf.N,
                                               label="joint_isotropic")
            info = {"matched_pushforward_trace": pf.trace(),
                    "tangent_samples": reference[i]["prior_info"]["tangent_samples"]}
        else:
            raw = _raw_window(fm, trajectories[i, 0], args.steps)
            gen = torch.Generator(device="cpu").manual_seed(args.seed + 10_009 * i)
            prior, info = _prior(method, fm, trajectories[i, 0], raw, sigma2, args, gen)
            energy = JointWindowResidualEnergy(
                spec, trajectories[i, 0], steps=args.steps,
                dt=args.lead_steps * spec.dt_out, residual_scale=scales["residual_scale"],
                divergence_scale=scales["divergence_scale"],
                divergence_weight=args.divergence_weight, device=fm.device)
        packages.append({"raw": raw, "prior": prior, "energy": energy,
                         "truth": trajectories[i, 1:args.steps + 1].to(fm.device, torch.float64),
                         "prior_info": info})
    return packages


def _summaries(packages):
    return [p["prior"].summarize(p["truth"].reshape(-1)) for p in packages]


def _evaluate(method, alpha, lam_rel, packages, args, collect_trace=False):
    rmses, per_step, diagnostics = [], [[] for _ in range(args.steps)], []
    started = time.time()
    for i, package in enumerate(packages):
        raw, pinfo = package["raw"], package["prior_info"]
        prior, energy, truth = package["prior"].rescaled(alpha), package["energy"], package["truth"]
        ref = force_scale(prior, energy)
        post = LowRankPhysicsPosterior(prior, energy, lam_rel * ref)
        corrected, info = post.map_estimate(n_steps=args.map_steps, lr=args.map_lr)
        corrected_window = corrected.reshape(args.steps, -1)
        raw_err = (raw - truth).square().mean().sqrt()
        corr_err = (corrected_window - truth).square().mean().sqrt()
        rmses.append(float(corr_err))
        for t in range(args.steps):
            per_step[t].append(float((corrected_window[t] - truth[t]).square().mean().sqrt()))
        if collect_trace:
            g = energy.grad(prior.mean).reshape(-1)
            frac = ((prior.U.T @ g).square().sum() / g.square().sum().clamp(min=1e-300)
                    if prior.k else g.new_zeros(()))
            diagnostics.append({"trajectory": i, "raw_rmse": float(raw_err),
                                "corrected_rmse": float(corr_err),
                                "grad_in_joint_rank": float(frac),
                                "correction_rms": float((corrected - prior.mean).square().mean().sqrt()),
                                "physics_energy_raw": float(energy.energy(prior.mean)[0]),
                                "physics_energy_map": float(energy.energy(corrected)[0]),
                                "lambda": lam_rel * ref, "lambda_ref": ref,
                                **pinfo, **info})
    return {"rmse": float(np.mean(rmses)), "step_rmse": [float(np.mean(x)) for x in per_step],
            "seconds": time.time() - started, "diagnostics": diagnostics}


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=8)
    ap.add_argument("--n-val-traj", type=int, default=4)
    ap.add_argument("--n-test-traj", type=int, default=8)
    ap.add_argument("--joint-rank", type=int, default=None,
                    help="rank of complete stacked trajectory covariance; default --k")
    ap.add_argument("--tangent-samples", type=int, default=None,
                    help="independent tangent trajectories; default k+oversample")
    ap.add_argument("--joint-floor-rel", type=float, default=1e-3,
                    help="explicit isotropic ridge as fraction of sampled mean variance")
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--map-steps", type=int, default=30)
    ap.add_argument("--map-lr", type=float, default=0.5)
    ap.add_argument("--divergence-weight", type=float, default=1.0)
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.steps < 2 or args.n_val_traj < 1:
        raise SystemExit("joint-window S4 needs --steps >= 2 and --n-val-traj >= 1")
    set_seed(args.seed)
    args.fm, args.fm_size, args.fm_channels = "poseidon", args.fm_size, "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj,
                                     args.steps, args.lead_steps, 0)
    val = load_poseidon_trajectories(args.fm_data_path, fm, args.n_val_traj,
                                     args.steps, args.lead_steps, args.n_cal_traj)
    test = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj,
                                      args.steps, args.lead_steps,
                                      args.n_cal_traj + args.n_val_traj)
    print_header(f"S4 joint-window residual HILP: {fm.info.name}, horizon={args.steps}, k={args.k}")
    print("  likelihood: centered temporal NS vorticity residual on generated future window")
    print("  prior: isotropic control versus tangent-ensemble joint pushforward covariance")
    print(f"  split: calibration={args.n_cal_traj}, validation={args.n_val_traj}, held-out={args.n_test_traj}")
    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]
    scales = _calibration_scales(fm, cal, spec, sigma2, args)
    print(f"  fitted one-step sigma²={sigma2:.5e}; residual scale={scales['residual_scale']:.4g}; "
          f"divergence scale={scales['divergence_scale']:.4g}")
    floor = _temporal_residual_floor_audit(fm, cal, spec, scales, args)
    print("  temporal-residual floor audit (calibration only): "
          f"truth={floor['truth_residual_median']:.4g}, raw={floor['raw_residual_median']:.4g}, "
          f"truth/raw={floor['truth_over_raw']:.3f}, truth wins={100 * floor['truth_wins_fraction']:.1f}%")

    alphas, summaries_meta, prepared = {}, {}, {}
    print("\n  calibrating joint prior scales")
    method_order = (["joint_pushforward"] if "joint_pushforward" in args.methods else []) + \
                   [m for m in args.methods if m != "joint_pushforward"]
    for method in method_order:
        packages = _prepare(method, fm, cal, sigma2, scales, spec, args,
                            reference=prepared.get(("calibration", "joint_pushforward")))
        prepared[("calibration", method)] = packages
        summaries = _summaries(packages)
        alpha, diag = fit_alpha_scale(summaries, return_diagnostics=True, seed=args.seed)
        alphas[method], summaries_meta[method] = alpha, {
            "fit": diag, "prior_meta": [p["prior_info"] for p in packages]}
        print(f"    {method:18s} alpha={alpha:.4g}; error-in-joint-rank="
              f"{diag.get('error_energy_in_subspace', float('nan')):.4g}")

    selected, selection = {}, {}
    for method in method_order:
        print(f"\n  selecting lambda for {method} on validation windows")
        packages = _prepare(method, fm, val, sigma2, scales, spec, args,
                            reference=prepared.get(("validation", "joint_pushforward")))
        prepared[("validation", method)] = packages
        curve = []
        for lam in args.lambda_grid:
            out = _evaluate(method, alphas[method], lam, packages, args)
            curve.append({"lambda_rel": lam, "rmse": out["rmse"]})
            print(f"    lambda/ref={lam:g}: mean window RMSE={out['rmse']:.6g}", flush=True)
        best = min(curve, key=lambda x: x["rmse"])
        selected[method], selection[method] = best["lambda_rel"], curve
        print(f"    selected {best['lambda_rel']:g} x reference")

    print("\n  held-out joint-window evaluation")
    table, records = Table("method", "window RMSE", "gain%",
                           *[f"step{i+1} gain%" for i in range(args.steps)]), {}
    # Raw metrics are shared: lambda=0 leaves every posterior at the FM mean.
    test_packages = {}
    for method in method_order:
        test_packages[method] = _prepare(
            method, fm, test, sigma2, scales, spec, args,
            reference=test_packages.get("joint_pushforward"))
        prepared[("held_out", method)] = test_packages[method]
    raw = _evaluate("isotropic", 1.0, 0.0, test_packages["isotropic"], args)
    records["raw"] = raw
    table.add("raw", raw["rmse"], 0.0, *([0.0] * args.steps))
    for method in args.methods:
        out = _evaluate(method, alphas[method], selected[method], test_packages[method],
                        args, collect_trace=True)
        records[method] = out
        gains = [100 * (1 - out["step_rmse"][t] / raw["step_rmse"][t]) for t in range(args.steps)]
        table.add(method, out["rmse"], 100 * (1 - out["rmse"] / raw["rmse"]), *gains)
        if out["diagnostics"]:
            print(f"  {method}: mean physics-gradient energy in joint rank="
                  f"{np.mean([d['grad_in_joint_rank'] for d in out['diagnostics']]):.3e}")
    print(table)
    out = {"stage": "s4_joint_window_residual", "metadata": fm_metadata(args, fm, spec),
           "sigma2": sigma2, "physics_scales": scales,
           "temporal_residual_floor_calibration": floor, "alphas": alphas,
           "calibration": summaries_meta, "lambda_selection": selection,
           "selected_lambda": selected, "held_out": records,
           "validity": {"future_truth_used_by_method": False,
                        "temporal_residual": "centered interior finite difference",
                        "spatial_operator": "published AZEBAN pseudospectral RHS",
                        "prior": "Monte-Carlo tangent joint rollout covariance"}}
    path = save_json(out, results_path_scale("s4_joint_window", fm_result_key(args),
                                             "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
