#!/usr/bin/env python3
"""S4 latent-state HILP: apply input-space GN before a frozen FM transition.

This is deliberately separate from ``s4_fm_document_posterior.py``.  The
existing document posterior corrects the output state directly.  Here the
random variable is a latent correction ``z_t`` to an *uncertain rollout
conditioning state* ``c_t``:

    z_t ~ N(c_t, alpha (J_t^T J_t + tau I)^-1),
    c_{t+1} = F(z_t),

and the local physics likelihood is E_phys(F(z_t); c_t).  Its flow target is
constructed from the fixed current rollout state c_t, rather than z_t.  This
is the first-order, fixed-conditioning likelihood: differentiating an
adaptive-step PDE solver through z_t would introduce a different, nonsmooth
model and is not needed to test the FM's J^T grad(E) correction direction.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (
    POSEIDON_NATIVE_RAW_STRIDE, configure_native_poseidon_cadence, fm_metadata,
    fm_result_key, load_poseidon_trajectories, native_poseidon_spec)
from experiments.scale.s4_fm_physics_correction import (
    _aggregate, _calibration_physics_scales, _one_metrics, _truth_floor_audit,
    make_physics_energy)
from hipp.scale.calibrate_scale import fit_alpha_scale
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.curvature_scale import estimate_scale, _gram_mm, _shifted_cg
from hipp.scale.jacobian import hutchinson_diag_gramian, jacobian_ops
from hipp.scale.lowrank import LowRankGaussian
from hipp.scale.posterior_scale import LowRankPhysicsPosterior, force_scale
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("latent_isotropic", "latent_gauss_newton", "latent_soft_gauss_newton",
           "latent_full_gauss_newton")


def _pcg(ops, b, shift, diag_inv, *, n_iter, chunk, refresh_every=8):
    """Preconditioned CG for (J^T J + shift I)x=b with residual replacement.

    The diagonal preconditioner changes only the coordinate system in which CG
    searches; it leaves the target linear system and its exact solution intact.
    Periodic residual replacement controls float32 recurrence drift in the
    large, mildly ill-conditioned FM operator.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    z = diag_inv[:, None] * r
    p = z.clone()
    rz = (r * z).sum(0)
    for it in range(n_iter):
        ap = _gram_mm(ops, p, chunk) + shift * p
        step = rz / (p * ap).sum(0).clamp(min=1e-30)
        x = x + p * step
        if (it + 1) % refresh_every == 0 or it + 1 == n_iter:
            r = b - (_gram_mm(ops, x, chunk) + shift * x)
        else:
            r = r - ap * step
        z = diag_inv[:, None] * r
        rz_new = (r * z).sum(0)
        p = z + p * (rz_new / rz.clamp(min=1e-30))
        rz = rz_new
    return x


class LatentTransitionEnergy:
    """E_phys(F(z); c) with c and the physical endpoint held fixed.

    The class exposes the same energy/grad API as an ordinary likelihood, so
    ``LowRankPhysicsPosterior`` supplies the MAP solve without special cases.
    """
    def __init__(self, fm, endpoint_energy):
        self.fm = fm
        self.endpoint_energy = endpoint_energy
        self.device = fm.device
        self.dtype = torch.float64

    def predict(self, z):
        # FrozenFM.predict is intentionally no-grad for ordinary rollout use.
        # flat_fn is its differentiable, single-state contract.
        return self.fm.flat_fn()(z.reshape(-1).to(self.fm.dtype)).double().reshape(1, -1)

    def energy(self, z):
        z = z.reshape(-1, self.endpoint_energy.N).to(self.device, self.dtype)
        rows = [self.endpoint_energy.energy(self.predict(z[i])) for i in range(z.shape[0])]
        return torch.cat(rows)

    def grad(self, z):
        x = z.detach().to(self.device, self.dtype).reshape(-1, self.endpoint_energy.N)
        x.requires_grad_(True)
        (g,) = torch.autograd.grad(self.energy(x).sum(), x)
        return g


class FullGNPrecision:
    """Matrix-free fixed-linearization GN precision A=J_c^T J_c+tau I."""
    def __init__(self, fm, c, tau_rel, n_tail, chunk, diag_probes, generator):
        from hipp.scale.jacobian import frobenius_norm_estimate
        self.mean = c.detach().double().reshape(-1)
        self.ops = jacobian_ops(fm.flat_fn(), c)
        fro = frobenius_norm_estimate(self.ops, n_probe=max(8, n_tail),
                                      chunk=chunk, generator=generator)
        self.tau = max(float(tau_rel) * fro / self.mean.numel(), 1e-12)
        self.chunk, self.N = chunk, self.mean.numel()
        # Hutchinson estimates diag(J^T J).  This affects *only* PCG's search
        # metric; A itself below is always applied exactly through JVP/VJP.
        diag = hutchinson_diag_gramian(self.ops, n_probe=diag_probes,
                                       chunk=chunk, generator=generator)
        self.diag_inv = 1.0 / (diag + self.tau).clamp(min=self.tau)
        self.diag_probes = int(diag_probes)
        self.last_cg_rel_residual = float("nan")

    def apply(self, v):
        v = v.reshape(-1, 1).to(self.ops.dtype)
        return (_gram_mm(self.ops, v, self.chunk) + self.tau * v).reshape(-1).double()

    def quad(self, z):
        d = z.detach().double().reshape(-1) - self.mean
        return float(d @ self.apply(d))

    def covariance_action(self, g, alpha, cg_iters):
        b = g.reshape(-1, 1).to(self.ops.dtype)
        x = _pcg(self.ops, b, self.tau, self.diag_inv, n_iter=cg_iters,
                 chunk=self.chunk)
        # This is a convergence certificate for the actual solve A x = g,
        # unlike the failed soft-GN Ritz residual, which certified a selected
        # eigenspace.  It is recorded with every held-out full-GN correction.
        residual = _gram_mm(self.ops, x, self.chunk) + self.tau * x - b
        self.last_cg_rel_residual = float(residual.norm() / b.norm().clamp(min=1e-30))
        return float(alpha) * x.reshape(-1).double()


def _full_gn_map(fm, c, endpoint_energy, args, alpha, lambda_rel, generator):
    """MAP with full matrix-free GN precision and exact composed physics gradient."""
    prior = FullGNPrecision(fm, c, args.tau_rel, args.n_tail, args.chunk,
                            args.full_gn_diag_probes, generator)
    latent = LatentTransitionEnergy(fm, endpoint_energy)
    g0 = latent.grad(prior.mean).reshape(-1)
    sg0 = prior.covariance_action(g0, alpha, args.full_gn_cg_iters)
    if prior.last_cg_rel_residual > args.full_gn_cg_tol:
        raise RuntimeError(
            "full-GN CG did not converge for the lambda normalization: "
            f"relative residual={prior.last_cg_rel_residual:.3e}, "
            f"required <= {args.full_gn_cg_tol:.3e}. Increase --full-gn-cg-iters "
            "or use a larger pre-specified --tau-rel; do not interpret this run.")
    ref = 1.0 / float((g0 @ sg0).clamp(min=1e-30).sqrt())
    lam = lambda_rel * ref
    x = prior.mean.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=args.map_lr, max_iter=args.map_steps,
                            history_size=10, line_search_fn="strong_wolfe")
    calls = [0]
    def closure():
        opt.zero_grad(set_to_none=True)
        d = x.detach() - prior.mean
        aq = prior.apply(d)
        p_loss = 0.5 * (d @ aq) / alpha
        e = latent.energy(x)[0]
        g_phys = latent.grad(x).reshape(-1)
        x.grad = (aq / alpha + lam * g_phys).reshape_as(x)
        calls[0] += 1
        return (p_loss + lam * e).detach()
    opt.step(closure)
    return latent.predict(x.detach()).reshape(-1).detach(), {
        "lambda_rel": lambda_rel, "lambda_ref": ref, "lambda": lam,
        "full_gn_tau": prior.tau, "full_gn_calls": calls[0],
        "full_gn_jvp_calls": prior.ops.n_calls[0], "full_gn_vjp_calls": prior.ops.n_calls[1],
    }, prior, latent


def _input_prior(method, fm, c, args, generator):
    """Prior on the conditioning state, centered at c rather than F(c)."""
    c64 = c.detach().double().reshape(-1)
    if method == "latent_isotropic":
        return LowRankGaussian.isotropic(c64, tau=1.0, label="latent-isotropic")
    estimator = "gn_soft" if method == "latent_soft_gauss_newton" else "gn"
    est = estimate_scale(
        fm, c, method=estimator, k=args.k, tau_rel=args.tau_rel,
        oversample=args.oversample, n_iter=args.n_iter, chunk=args.chunk,
        n_tail=args.n_tail, generator=generator,
        soft_inverse_iters=args.soft_inverse_iters,
        soft_cg_iters=args.soft_cg_iters,
        soft_shift_rel=args.soft_shift_rel)
    p = est.prior
    prior = LowRankGaussian(c64, p.U, p.d, p.tau, alpha=1.0, label=method)
    prior.geometry_meta = est.meta
    return prior


def _energy(spec, current, channels, args, scales, device):
    return make_physics_energy(
        spec, current, channels, args, device,
        vorticity_scale=scales["vorticity_scale"],
        divergence_scale=scales["divergence_scale"])


def _calibration_summaries(args, fm, cal, methods):
    """Fit input-state covariance scale from predicted-current-state errors.

    Step one is excluded because x_0 is observed exactly.  At t>=1 the raw
    autoregressive state c_t is compared with the known calibration x_t.
    """
    summaries = {m: [] for m in methods}
    for method in methods:
        print(f"    building {method} latent priors ({cal.shape[0]} trajectories x "
              f"{max(args.steps - 1, 0)} uncertain steps)", flush=True)
        for i in range(cal.shape[0]):
            c = cal[i, 0].to(fm.device, fm.dtype)
            for t in range(args.steps):
                if t > 0:
                    print(f"      {method}: trajectory {i + 1}/{cal.shape[0]}, "
                          f"latent step {t + 1}/{args.steps}", flush=True)
                    gen = torch.Generator(device="cpu").manual_seed(
                        args.seed + 90_001 * i + 1_009 * t)
                    if method == "latent_full_gauss_newton":
                        prior = FullGNPrecision(fm, c, args.tau_rel, args.n_tail,
                                                args.chunk, args.full_gn_diag_probes, gen)
                        error = cal[i, t].to(fm.device).double().reshape(-1)
                        summaries[method].append({
                            "quad": prior.quad(error), "N": prior.N,
                            "jvp_calls": prior.ops.n_calls[0],
                            "vjp_calls": prior.ops.n_calls[1],
                        })
                    else:
                        prior = _input_prior(method, fm, c, args, gen)
                        summaries[method].append(prior.summarize(cal[i, t]))
                c = fm.predict(c).to(fm.dtype)
    return summaries


def _rollout(args, fm, truth, spec, method, alpha, lambda_rel, scales,
             trace=False):
    rows, diagnostics = [[] for _ in range(args.steps)], []
    started = time.time()
    for i in range(truth.shape[0]):
        c = truth[i, 0].to(fm.device, fm.dtype)
        for t in range(args.steps):
            raw = fm.predict(c).double()
            endpoint_energy = _energy(spec, c, fm.state_shape[0], args, scales, fm.device)
            truth_energy = _energy(spec, truth[i, t], fm.state_shape[0], args,
                                   scales, fm.device)
            if t == 0:
                # x_0 is observed, so latent input correction would be data
                # leakage.  All latent methods have the same raw first step.
                corrected, info = raw, {"known_initial_state": True,
                                         "lambda_rel": 0.0, "lambda": 0.0,
                                         "prior_rank": 0}
            else:
                gen = torch.Generator(device="cpu").manual_seed(
                    args.seed + 90_001 * i + 1_009 * t)
                input_error = truth[i, t].to(fm.device).double().reshape(-1) - c.double().reshape(-1)
                if method == "latent_full_gauss_newton":
                    corrected, info, prior, latent_energy = _full_gn_map(
                        fm, c, endpoint_energy, args, alpha, lambda_rel, gen)
                    g = latent_energy.grad(prior.mean).reshape(-1)
                    hdir = -prior.covariance_action(g, alpha, args.full_gn_cg_iters)
                    if prior.last_cg_rel_residual > args.full_gn_cg_tol:
                        raise RuntimeError(
                            "full-GN CG did not converge for the held-out direction: "
                            f"relative residual={prior.last_cg_rel_residual:.3e}, "
                            f"required <= {args.full_gn_cg_tol:.3e}. Increase "
                            "--full-gn-cg-iters or use a larger pre-specified --tau-rel; "
                            "do not interpret this run.")
                    cosine = float((hdir @ input_error) /
                                   (hdir.norm() * input_error.norm()).clamp(min=1e-300))
                    info.update({"known_initial_state": False, "prior_rank": "full",
                                 "latent_direction_error_cosine": cosine,
                                 "latent_grad_energy_in_rank": float("nan"),
                                 "full_gn_preconditioned_grad_norm": float(hdir.norm()),
                                 "full_gn_cg_rel_residual": prior.last_cg_rel_residual,
                                 "full_gn_jvp_calls": prior.ops.n_calls[0],
                                 "full_gn_vjp_calls": prior.ops.n_calls[1]})
                else:
                    base_prior = _input_prior(method, fm, c, args, gen)
                    prior = base_prior.rescaled(alpha)
                    # LowRankGaussian.rescaled intentionally constructs a fresh
                    # algebra object; retain estimator diagnostics separately.
                    if hasattr(base_prior, "geometry_meta"):
                        prior.geometry_meta = base_prior.geometry_meta
                    latent_energy = LatentTransitionEnergy(fm, endpoint_energy)
                    ref = force_scale(prior, latent_energy)
                    posterior = LowRankPhysicsPosterior(prior, latent_energy, lambda_rel * ref)
                    z_star, info = posterior.map_estimate(n_steps=args.map_steps, lr=args.map_lr)
                    corrected = latent_energy.predict(z_star).reshape(-1).detach()
                    g = latent_energy.grad(prior.mean).reshape(-1)
                    # The actual local HILP direction is -Sigma grad E.  Computing
                    # it through grad_log_prob would invert Sigma, so use the prior
                    # covariance formula directly in the diagnostic below.
                    from hipp.scale.posterior_scale import covariance_matvec
                    hdir = -covariance_matvec(prior, g).reshape(-1)
                    cosine = float((hdir @ input_error) /
                                   (hdir.norm() * input_error.norm()).clamp(min=1e-300))
                    info.update({"known_initial_state": False, "lambda_rel": lambda_rel,
                                 "lambda": lambda_rel * ref, "lambda_ref": ref,
                                 "prior_rank": prior.k,
                                 "latent_input_shift_rms": float(
                                     (z_star.reshape(-1) - c.double().reshape(-1)).square().mean().sqrt()),
                                 "latent_direction_error_cosine": cosine,
                                 "latent_grad_energy_in_rank": float(
                                     (prior.U.T @ g).square().sum() / g.square().sum().clamp(min=1e-300))
                                 if prior.k else 0.0})
                    if hasattr(prior, "geometry_meta") and "soft_ritz_residual_rel" in prior.geometry_meta:
                        info["soft_ritz_residual_rel"] = float(
                            prior.geometry_meta["soft_ritz_residual_rel"].mean())
            metrics = _one_metrics(corrected, raw, truth[i, t + 1], endpoint_energy,
                                   truth_energy, prior=None,
                                   same_conditioning_state=(t == 0))
            rows[t].append(metrics)
            diagnostics.append({"trajectory": i, "step": t + 1, **info})
            c = corrected.to(fm.dtype)
    return [_aggregate(r) for r in rows], diagnostics, time.time() - started


def _score(rows):
    return float(np.mean([r["rmse"] for r in rows]))


def _select_full_gn_damping(args, fm, cal, spec, scales):
    """Choose the smallest numerically resolved GN damping on calibration only.

    This gate deliberately examines solver residuals, never forecast error.  It
    is therefore a numerical-validity choice, not a hidden performance sweep.
    """
    grid = args.full_gn_tau_grid or [args.tau_rel]
    rows = []
    original_tau = args.tau_rel
    print("\n  full-GN calibration damping gate (solver residual only)")
    for tau_rel in grid:
        args.tau_rel = float(tau_rel)
        residuals = []
        for i in range(cal.shape[0]):
            c = cal[i, 0].to(fm.device, fm.dtype)
            for t in range(args.steps):
                if t > 0:
                    gen = torch.Generator(device="cpu").manual_seed(
                        args.seed + 90_001 * i + 1_009 * t)
                    endpoint_energy = _energy(spec, c, fm.state_shape[0], args,
                                              scales, fm.device)
                    prior = FullGNPrecision(
                        fm, c, args.tau_rel, args.n_tail, args.chunk,
                        args.full_gn_diag_probes, gen)
                    g = LatentTransitionEnergy(fm, endpoint_energy).grad(prior.mean)
                    prior.covariance_action(g, alpha=1.0,
                                            cg_iters=args.full_gn_cg_iters)
                    residuals.append(prior.last_cg_rel_residual)
                c = fm.predict(c).to(fm.dtype)
        record = {"tau_rel": float(tau_rel), "n_states": len(residuals),
                  "mean_cg_rel_residual": float(np.mean(residuals)),
                  "max_cg_rel_residual": float(np.max(residuals)),
                  "passes": bool(max(residuals) <= args.full_gn_cg_tol)}
        rows.append(record)
        print(f"    tau_rel={record['tau_rel']:g}: mean/max residual="
              f"{record['mean_cg_rel_residual']:.2e}/{record['max_cg_rel_residual']:.2e} "
              f"({'PASS' if record['passes'] else 'fail'})")
        if record["passes"]:
            args.tau_rel = record["tau_rel"]
            print(f"    selected tau_rel={args.tau_rel:g}: smallest passing damping")
            return rows
    args.tau_rel = original_tau
    raise RuntimeError(
        "no full-GN damping candidate met the required CG residual on all "
        "calibration states; expand --full-gn-tau-grid or increase CG iterations.")


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1)
    ap.add_argument("--n-cal-traj", type=int, default=8)
    ap.add_argument("--n-test-traj", type=int, default=8)
    ap.add_argument("--n-tail", type=int, default=16)
    ap.add_argument("--soft-inverse-iters", type=int, default=2)
    ap.add_argument("--soft-cg-iters", type=int, default=24)
    ap.add_argument("--soft-shift-rel", type=float, default=0.1)
    ap.add_argument("--full-gn-cg-iters", type=int, default=24,
                    help="CG iterations for each full matrix-free GN precision solve")
    ap.add_argument("--full-gn-cg-tol", type=float, default=1e-3,
                    help="required relative residual for full-GN A^{-1}g solves")
    ap.add_argument("--full-gn-diag-probes", type=int, default=16,
                    help="Hutchinson probes for a PCG-only diagonal preconditioner")
    ap.add_argument("--full-gn-tau-grid", nargs="+", type=float, default=None,
                    help="calibration-only solver-validity grid; selects smallest passing tau")
    ap.add_argument("--full-gn-convergence-only", action="store_true",
                    help="run the calibration damping gate then exit without held-out evaluation")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                    help="latent priors to evaluate")
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=[0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    ap.add_argument("--map-steps", type=int, default=30)
    ap.add_argument("--map-lr", type=float, default=0.5)
    ap.add_argument("--physics-likelihood", choices=["discrete_flow"], default="discrete_flow")
    ap.add_argument("--flow-cfl", type=float, default=0.5)
    ap.add_argument("--physics-resolution", type=int, default=8)
    ap.add_argument("--divergence-weight", type=float, default=1.0)
    ap.add_argument("--fm-data-path", required=True)
    ap.add_argument("--allow-native-transfer", action="store_true")
    args = ap.parse_args()
    args.data_source = "poseidon-native"
    if args.fm != "poseidon" and not args.allow_native_transfer:
        raise SystemExit("non-Poseidon native-data experiments require --allow-native-transfer")
    if args.physics_resolution <= 0 or 128 % args.physics_resolution:
        raise SystemExit("--physics-resolution must be a positive divisor of 128")
    set_seed(args.seed)
    dt_target = configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    spec = native_poseidon_spec(args, fm)
    cal = load_poseidon_trajectories(args.fm_data_path, fm, args.n_cal_traj,
                                     args.steps, args.lead_steps, 0)
    print_header(f"S4 latent GN-HILP: {fm.info.name}, lead={args.lead_steps}, k={args.k}")
    print("  first step is raw: x_0 is known exactly; latent correction begins at step 2")
    print(f"  physics likelihood: fixed {args.physics_resolution}x{args.physics_resolution} "
          f"AZEBAN endpoint; target dt={dt_target:g}")
    if "latent_full_gauss_newton" in args.methods:
        print(f"  full GN mode: matrix-free J^T J precision with {args.full_gn_cg_iters} "
              f"PCG iterations and {args.full_gn_diag_probes} diagonal probes "
              "per inverse action (no spectral truncation)")
    xs = [cal[i, t] for i in range(cal.shape[0]) for t in range(args.steps)]
    ys = [cal[i, t + 1] for i in range(cal.shape[0]) for t in range(args.steps)]
    sigma = fit_sigma2(fm, xs, ys)
    floor = _truth_floor_audit(cal, spec, fm.state_shape[0], args, fm.device)
    scales = _calibration_physics_scales(args, fm, cal, spec, floor)
    print(f"  residual floor: true/persistence={floor['truth_over_persistence_mean']:.3f}; "
          f"truth wins={100 * floor['truth_beats_persistence_fraction']:.1f}%")
    damping_gate = []
    if "latent_full_gauss_newton" in args.methods and args.full_gn_tau_grid:
        damping_gate = _select_full_gn_damping(args, fm, cal, spec, scales)
    if args.full_gn_convergence_only:
        if "latent_full_gauss_newton" not in args.methods:
            raise SystemExit("--full-gn-convergence-only requires latent_full_gauss_newton")
        out = {"stage": "s4_latent_full_gn_convergence_gate",
               "metadata": fm_metadata(args, fm, spec), "physics_scales": scales,
               "residual_floor_calibration": floor, "damping_gate": damping_gate,
               "validity": {"split": "calibration trajectories only; no held-out evaluation",
                            "criterion": "smallest tau with every PCG solve below tolerance"}}
        p = save_json(out, results_path_scale("s4_latent_gn", fm_result_key(args),
                                              "convergence_gate.json", args.tag))
        print(f"\nwrote {p}")
        return
    print("\n  calibrating latent input-prior scale alpha")
    summaries = _calibration_summaries(args, fm, cal, args.methods)
    alphas, alpha_diag = {}, {}
    for method in args.methods:
        if method == "latent_full_gauss_newton":
            quads = np.asarray([s["quad"] / s["N"] for s in summaries[method]], dtype=float)
            alphas[method] = max(float(np.mean(quads)), 1e-300)
            alpha_diag[method] = {
                "alpha": alphas[method], "objective": "mle",
                "n": int(len(quads)), "N": int(summaries[method][0]["N"]),
                "error_energy_in_subspace": float("nan"),
                "decomposition_note": "full matrix-free GN precision; no retained subspace",
            }
        else:
            alphas[method], alpha_diag[method] = fit_alpha_scale(
                summaries[method], objective=args.alpha_objective,
                return_diagnostics=True, seed=args.seed)
        diag_label = ("full precision" if method == "latent_full_gauss_newton"
                      else f"error-in-rank={alpha_diag[method]['error_energy_in_subspace']:.3g}")
        print(f"    {method:22s} alpha={alphas[method]:.4g} {diag_label}")
    selected, sweeps = {}, {}
    for method in args.methods:
        print(f"\n  selecting lambda for {method} on calibration rollouts")
        cells = []
        for lam in args.lambda_grid:
            run, _, secs = _rollout(args, fm, cal, spec, method, alphas[method], lam, scales)
            cell = {"lambda_rel": lam, "rmse": _score(run), "seconds": secs}
            cells.append(cell)
            print(f"    lambda/ref={lam:g}: mean RMSE={cell['rmse']:.6g}")
        best = min(cells, key=lambda x: x["rmse"])
        selected[method], sweeps[method] = best["lambda_rel"], cells
        print(f"    selected {best['lambda_rel']:g} x reference")
    print("\n  held-out latent-state rollout evaluation")
    test = load_poseidon_trajectories(args.fm_data_path, fm, args.n_test_traj,
                                      args.steps, args.lead_steps, args.n_cal_traj)
    final, diag = {}, {}
    raw_rows, _, _ = _rollout(args, fm, test, spec, "latent_isotropic", 1.0, 0.0, scales)
    raw_score = _score(raw_rows)
    for method in args.methods:
        out, d, secs = _rollout(args, fm, test, spec, method, alphas[method],
                                selected[method], scales)
        final[method] = {"steps": out, "mean_rollout_rmse": _score(out), "seconds": secs}
        diag[method] = d
    headers = ["method", "RMSE", "gain%"] + [f"step{t} gain%" for t in range(2, args.steps + 1)]
    table = Table(*headers)
    table.add("raw", raw_score, 0, *([0] * (args.steps - 1)))
    for method in args.methods:
        out = final[method]["steps"]
        gains = [100 * (1 - out[t]["rmse"] / raw_rows[t]["rmse"]) for t in range(args.steps)]
        table.add(method, final[method]["mean_rollout_rmse"],
                  100 * (1 - final[method]["mean_rollout_rmse"] / raw_score), *gains[1:])
        useful = [d for d in diag[method] if not d["known_initial_state"]]
        if method == "latent_full_gauss_newton":
            print(f"  {method}: latent direction cosine="
                  f"{np.mean([d['latent_direction_error_cosine'] for d in useful]):+.3f}; "
                  f"full GN precision (mean JVP/VJP calls "
                  f"{np.mean([d['full_gn_jvp_calls'] for d in useful]):.0f}/"
                  f"{np.mean([d['full_gn_vjp_calls'] for d in useful]):.0f}; "
                  f"CG residual {np.mean([d['full_gn_cg_rel_residual'] for d in useful]):.2e})")
        else:
            print(f"  {method}: latent direction cosine={np.mean([d['latent_direction_error_cosine'] for d in useful]):+.3f}; "
                  f"physics-grad in GN rank={np.mean([d['latent_grad_energy_in_rank'] for d in useful]):.3e}")
        if useful and "soft_ritz_residual_rel" in useful[0]:
            print(f"    soft-GN Ritz residual={np.mean([d['soft_ritz_residual_rel'] for d in useful]):.3e}")
    print(table)
    out = {"stage": "s4_latent_gn", "metadata": fm_metadata(args, fm, spec),
           "sigma2": sigma, "physics_scales": scales, "residual_floor_calibration": floor,
           "alpha": alphas, "alpha_diagnostics": alpha_diag,
           "full_gn_damping_gate": damping_gate,
           "selected_lambda_rel": selected, "calibration_sweep": sweeps,
           "test": final, "diagnostics": diag,
           "validity": {"latent_objective": "E_phys(F(z); c) with c fixed per local transition",
                        "first_step": "raw because initial state is known", "test_split": "trajectory-disjoint"}}
    if "latent_full_gauss_newton" in args.methods:
        out["validity"]["full_gn"] = (
            "A=J^T J+tau I is applied matrix-free using JVP/VJP; "
            "CG approximates A^{-1}g without selecting a low-rank eigenspace")
    p = save_json(out, results_path_scale("s4_latent_gn", fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
