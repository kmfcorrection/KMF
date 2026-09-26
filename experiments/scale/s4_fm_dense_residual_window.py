#!/usr/bin/env python3
"""Dense-time residual HILP for frozen Poseidon forecasts.

This is the residual-only alternative to endpoint-flow S4.  Every Poseidon
coarse transition (dt=0.1 for lead=1) is split into observed-native substeps
(dt=0.05).  The intervening states are latent optimization variables.  Physics
enters only through the centered temporal Navier--Stokes residual; no RK/AZEBAN
flow endpoint is constructed.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale.fm_eval_common import (POSEIDON_NATIVE_RAW_STRIDE,
    POSEIDON_RAW_DT, configure_native_poseidon_cadence, fm_metadata,
    fm_result_key, native_poseidon_spec)
from experiments.scale.s4_fm_revised_hilp import InnovationPrior, IsotropicPrior
from hipp.scale.common_scale import base_parser_scale, load_fm, results_path_scale
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.scale.jacobian import jacobian_ops
from hipp.scale.rollout import fit_sigma2
from hipp.utils import Table, print_header, save_json, set_seed


METHODS = ("isotropic", "block_innovation")


def _load_fine(path, fm, n_traj, coarse_steps, substeps, offset):
    """Load raw native frames, keeping intermediates for calibration/audit only."""
    import h5py
    raw_stride = POSEIDON_NATIVE_RAW_STRIDE
    if raw_stride % substeps:
        raise ValueError(f"dense_substeps={substeps} must divide native raw stride={raw_stride}")
    need = coarse_steps * raw_stride + 1
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        ds = h[key]
        if offset + n_traj > ds.shape[0] or need > ds.shape[1]:
            raise ValueError("requested dense trajectory slice is outside the dataset")
        # substeps=2 means every raw 0.05 frame; subsampling is deliberate.
        stride = raw_stride // substeps
        arr = np.asarray(ds[offset:offset + n_traj, :need:stride, :2], dtype=np.float32)
    return torch.as_tensor(arr, device=fm.device, dtype=fm.dtype).reshape(n_traj, -1, fm.N)


def _raw_endpoints_and_ops(fm, x0, coarse_steps):
    raw, ops, current = [], [], x0.to(fm.device, fm.dtype)
    for t in range(coarse_steps):
        nxt = fm.predict(current).detach().reshape(-1)
        raw.append(nxt.double())
        if t + 1 < coarse_steps:
            ops.append(jacobian_ops(fm.flat_fn(), nxt))
        current = nxt
    return torch.stack(raw), ops


def _linear_nodes(x0, endpoints, substeps):
    """Full latent path excluding the known x0, initially piecewise linear."""
    nodes, previous = [], x0.double().reshape(-1)
    for end in endpoints:
        for j in range(1, substeps + 1):
            nodes.append((1.0 - j / substeps) * previous + (j / substeps) * end)
        previous = end
    return torch.stack(nodes)


class DenseTemporalResidual:
    """Centered NS residual on all fine temporal pairs; no forward solve."""
    def __init__(self, spec, x0, coarse_steps, substeps, residual_scale,
                 divergence_scale, divergence_weight, device):
        self.steps, self.substeps = int(coarse_steps), int(substeps)
        self.nodes = self.steps * self.substeps
        self.N, self.dt = 2 * spec.N, POSEIDON_RAW_DT / self.substeps * POSEIDON_NATIVE_RAW_STRIDE
        # For lead=1 and substeps=2 this is 0.05.  More generally each coarse
        # 0.1 transition is split evenly by the requested number of nodes.
        self.dt = (POSEIDON_RAW_DT * POSEIDON_NATIVE_RAW_STRIDE) / self.substeps
        self.operator = FMPhysicsEnergy2D(spec, x0, 2, divergence_weight=0.0,
                                          dt=self.dt, device=device)
        self.x0 = x0.detach().to(device, torch.float64).reshape(1, self.N)
        self.residual_scale, self.divergence_scale = float(residual_scale), float(divergence_scale)
        self.divergence_weight = float(divergence_weight)

    def _states(self, z):
        future = z.reshape(self.nodes, self.N).to(self.operator.device, torch.float64)
        return torch.cat((self.x0, future), dim=0)

    def residual_fields(self, z):
        states = self._states(z)
        vort = self.operator.to_vorticity(states)
        mid = 0.5 * (vort[:-1] + vort[1:])
        return (vort[1:] - vort[:-1]) / self.dt - self.operator._native_rhs(mid)

    def velocity_rhs(self, state):
        """PDE tangent d(u,v)/dt, evaluated but never integrated."""
        flat = state.reshape(-1, self.N).to(self.operator.device, torch.float64)
        vort = self.operator.to_vorticity(flat)
        rhs_vort = self.operator._native_rhs(vort)
        uh = torch.fft.rfft2(rhs_vort)
        du, dv = self.operator.grid.velocity(uh)
        # The spatially constant velocity mode is conserved by this periodic,
        # unforced PDE, so its time derivative is zero.
        return torch.stack((du, dv), dim=1).reshape_as(flat)

    def bridge_nodes(self, endpoints, bridge):
        """Latent fine nodes conditioned on coarse endpoints.

        ``hermite_pde`` is cubic Hermite interpolation with endpoint slopes
        from the PDE RHS.  It evaluates F twice per coarse interval but does
        not advance a numerical flow map.
        """
        nodes, previous = [], self.x0.reshape(-1)
        coarse_dt = self.dt * self.substeps
        for end in endpoints.reshape(self.steps, self.N):
            if bridge == "linear":
                for j in range(1, self.substeps + 1):
                    a = j / self.substeps
                    nodes.append((1.0 - a) * previous + a * end)
            elif bridge == "hermite_pde":
                f0, f1 = self.velocity_rhs(previous)[0], self.velocity_rhs(end)[0]
                for j in range(1, self.substeps + 1):
                    a = j / self.substeps
                    h00, h10 = 2*a**3 - 3*a**2 + 1, a**3 - 2*a**2 + a
                    h01, h11 = -2*a**3 + 3*a**2, a**3 - a**2
                    nodes.append(h00 * previous + h10 * coarse_dt * f0 +
                                 h01 * end + h11 * coarse_dt * f1)
            else:
                raise ValueError(f"unknown latent bridge {bridge!r}")
            previous = end
        return torch.stack(nodes)

    def components(self, z):
        """Return separately normalized temporal and divergence energies."""
        states = self._states(z)
        # Use a sum of per-time, per-gridpoint standardized norms.  This
        # matches revised S4's direct-lambda convention; a global mean here
        # silently divided the physics gradient by nodes*N and made a
        # previously meaningful lambda appear to do nothing.
        residual_sq = self.residual_fields(z).flatten(1).square().mean(1)
        divergence_sq = self.operator.divergence(states[1:]).flatten(1).square().mean(1)
        temporal = self.N * 0.5 * (residual_sq / self.residual_scale ** 2).sum()
        divergence = self.N * 0.5 * self.divergence_weight * (
            divergence_sq / self.divergence_scale ** 2).sum()
        return temporal, divergence

    def value(self, z):
        temporal, divergence = self.components(z)
        return temporal + divergence

    def rms(self, z):
        return float(self.residual_fields(z).square().mean().sqrt())


class DensePrior:
    """Endpoint HILP prior plus a calibrated temporal bridge."""
    def __init__(self, endpoint_prior, x0, substeps, bridge_variance, bridge, node_builder):
        self.endpoint_prior = endpoint_prior
        self.x0 = x0.detach().double().reshape(-1)
        self.substeps = int(substeps)
        self.steps, self.N = endpoint_prior.T, endpoint_prior.N
        self.nodes = self.steps * self.substeps
        self.bridge, self.node_builder = bridge, node_builder
        self.mean = node_builder(endpoint_prior.mean, bridge).detach()
        self.bridge_variance = float(bridge_variance)
        if self.bridge_variance <= 0:
            raise ValueError("bridge variance must be positive")

    def rescaled(self, alpha):
        return DensePrior(self.endpoint_prior.rescaled(alpha), self.x0, self.substeps,
                          self.bridge_variance, self.bridge, self.node_builder)

    def endpoints(self, z):
        return z.reshape(self.nodes, self.N)[self.substeps - 1::self.substeps]

    def value_grad(self, candidate):
        z = candidate.detach().double().reshape(self.nodes, self.N)
        ends = self.endpoints(z)
        value, endpoint_grad = self.endpoint_prior.value_grad(ends)
        grad = torch.zeros_like(z)
        grad[self.substeps - 1::self.substeps] += endpoint_grad
        if self.bridge == "hermite_pde":
            # Differentiate the bridge target through endpoint RHS evaluations;
            # this is a PDE tangent calculation, not a numerical flow solve.
            probe = z.detach().clone().requires_grad_(True)
            target = self.node_builder(self.endpoints(probe), self.bridge)
            bridge_value = 0.5 * (probe - target).square().sum() / self.bridge_variance
            bridge_grad = torch.autograd.grad(bridge_value, probe)[0]
            return value + float(bridge_value.detach()), grad + bridge_grad.detach()
        previous = self.x0
        for segment in range(self.steps):
            end_index = (segment + 1) * self.substeps - 1
            end = z[end_index]
            for j in range(1, self.substeps):
                index, a = segment * self.substeps + (j - 1), j / self.substeps
                d = z[index] - ((1.0 - a) * previous + a * end)
                value += 0.5 * float(d.square().sum() / self.bridge_variance)
                g = d / self.bridge_variance
                grad[index] += g
                grad[end_index] -= a * g
                if segment:
                    grad[segment * self.substeps - 1] -= (1.0 - a) * g
            previous = end
        return value, grad


class CollapsedEndpointPrior:
    """Endpoint prior for the constrained, collapsed-Hermite likelihood.

    There are deliberately no independently optimizable fine nodes in this
    construction.  The residual wrapper deterministically rebuilds them from
    the endpoint window on every objective evaluation.
    """
    def __init__(self, endpoint_prior):
        self.endpoint_prior = endpoint_prior
        self.mean = endpoint_prior.mean
        self.T, self.N = endpoint_prior.T, endpoint_prior.N

    def rescaled(self, alpha):
        return CollapsedEndpointPrior(self.endpoint_prior.rescaled(alpha))

    def endpoints(self, z):
        return z.reshape(self.T, self.N)

    def value_grad(self, candidate):
        return self.endpoint_prior.value_grad(candidate)


class CollapsedHermiteResidual:
    """Fine-window PDE residual with all midpoint nodes fixed to Hermite paths."""
    def __init__(self, dense):
        self.dense = dense

    def _nodes(self, endpoints):
        return self.dense.bridge_nodes(endpoints, "hermite_pde")

    def value(self, endpoints):
        return self.dense.value(self._nodes(endpoints))

    def rms(self, endpoints):
        return self.dense.rms(self._nodes(endpoints))

    def components(self, endpoints):
        return self.dense.components(self._nodes(endpoints))


def _map(prior, residual, lam, args):
    z = prior.mean.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([z], lr=args.map_lr, max_iter=args.map_steps,
                            history_size=10, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(set_to_none=True)
        prior_value, prior_grad = prior.value_grad(z)
        loss = z.new_tensor(prior_value) + lam * residual.value(z)
        loss.backward()
        with torch.no_grad():
            z.grad.add_(prior_grad)
        return loss
    opt.step(closure)
    return z.detach()


def _bridge_variance(fine, spec, substeps, bridge, divergence_weight, device):
    """Fit the latent-bridge variance using each trajectory's own x_0.

    A bridge's first fine segment is conditioned on its initial state.  Reusing
    one residual object here silently conditioned every later calibration
    trajectory on trajectory zero's x_0, which inflated this variance in a
    split-dependent way and weakened the bridge precision in the MAP problem.
    """
    values = []
    for trajectory in fine:
        residual = DenseTemporalResidual(spec, trajectory[0],
                                         (fine.shape[1] - 1) // substeps,
                                         substeps, 1.0, 1.0,
                                         divergence_weight, device)
        for s in range((fine.shape[1] - 1) // substeps):
            endpoints = trajectory[substeps::substeps]
            target = residual.bridge_nodes(endpoints, bridge)
            for j in range(1, substeps):
                d = trajectory[s * substeps + j] - target[s * substeps + j - 1]
                values.append(float(d.double().square().mean()))
    return max(float(np.mean(values)), 1e-30)


def _scales(fine, spec, steps, substeps, args, device):
    # The first temporal difference requires that trajectory's own known x_0.
    # Do not reuse trajectory zero's residual operator for every trajectory.
    probes = [DenseTemporalResidual(spec, t[0], steps, substeps, 1.0, 1.0,
                                    args.divergence_weight, device)
              for t in fine]
    truth = [probe.rms(t[1:]) for probe, t in zip(probes, fine)]
    # Raw comparison is built later per trajectory; scale must be calibration-only.
    div = [float(probe.operator.divergence(t[1:].double()).square().mean().sqrt())
           for probe, t in zip(probes, fine)]
    return max(float(np.median(truth)), 1e-30), max(float(np.median(div)), 1e-30)


def _package(fm, fine, spec, sigma2, bridge, scales, args):
    packages = []
    for i, truth_fine in enumerate(fine):
        print(f"    preparing trajectory {i + 1}/{len(fine)}", flush=True)
        raw, ops = _raw_endpoints_and_ops(fm, truth_fine[0], args.steps)
        endpoint_truth = truth_fine[args.dense_substeps::args.dense_substeps]
        endpoint_block = InnovationPrior(raw, ops, sigma2)
        residual = DenseTemporalResidual(spec, truth_fine[0], args.steps, args.dense_substeps,
                                         scales[0], scales[1], args.divergence_weight, fm.device)
        endpoint_isotropic = IsotropicPrior(raw)
        if args.latent_bridge == "collapsed_hermite":
            # The only candidate variables are coarse FM endpoints.  The
            # Hermite midpoint is rebuilt inside the residual, so it cannot
            # absorb a PDE defect without changing an endpoint.
            collapsed = CollapsedHermiteResidual(residual)
            priors = {"block_innovation": CollapsedEndpointPrior(endpoint_block),
                      "isotropic": CollapsedEndpointPrior(endpoint_isotropic)}
            package_residual = collapsed
        else:
            priors = {"block_innovation": DensePrior(endpoint_block, truth_fine[0], args.dense_substeps, bridge,
                                                       args.latent_bridge, residual.bridge_nodes),
                      "isotropic": DensePrior(endpoint_isotropic, truth_fine[0], args.dense_substeps, bridge,
                                                args.latent_bridge, residual.bridge_nodes)}
            package_residual = residual
        packages.append({"raw": raw, "truth_end": endpoint_truth.double(),
                         "truth_fine": truth_fine[1:].double(), "residual": package_residual,
                         **priors})
    return packages


def _fit_alpha(packages, method):
    values = []
    for p in packages:
        prior = p[method]
        if method == "isotropic":
            values.append(float((p["truth_end"] - p["raw"]).square().mean()))
        else:
            base = prior.endpoint_prior
            values.append(base.normalized_innovation_mse(p["truth_end"]))
    return max(float(np.mean(values)), 1e-30)


def _evaluate(packages, method, alpha, lam, args):
    rows = []
    for p in packages:
        corrected = _map(p[method].rescaled(alpha), p["residual"], lam, args)
        ends = p[method].endpoints(corrected)
        truth, raw = p["truth_end"], p["raw"]
        temporal, divergence = p["residual"].components(corrected)
        raw_temporal, raw_divergence = p["residual"].components(p[method].mean)
        rows.append((float((ends - truth).square().mean().sqrt()),
                     float((raw - truth).square().mean().sqrt()),
                     p["residual"].rms(corrected), p["residual"].rms(p[method].mean),
                     float(p["residual"].value(corrected).detach()),
                     float(p["residual"].value(p[method].mean).detach()),
                     float((ends - raw).square().mean().sqrt()),
                     float(temporal.detach()), float(raw_temporal.detach()),
                     float(divergence.detach()), float(raw_divergence.detach())))
    arr = np.asarray(rows)
    return {"rmse": float(arr[:, 0].mean()), "raw_rmse": float(arr[:, 1].mean()),
            "physics_rms": float(arr[:, 2].mean()), "raw_physics_rms": float(arr[:, 3].mean()),
            "physics_energy": float(arr[:, 4].mean()), "raw_physics_energy": float(arr[:, 5].mean()),
            "endpoint_correction_rms": float(arr[:, 6].mean()),
            "temporal_energy": float(arr[:, 7].mean()), "raw_temporal_energy": float(arr[:, 8].mean()),
            "divergence_energy": float(arr[:, 9].mean()), "raw_divergence_energy": float(arr[:, 10].mean())}


def main():
    ap = base_parser_scale(__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lead-steps", type=int, default=1,
                    help="native Poseidon transition count; dense mode currently requires 1")
    ap.add_argument("--dense-substeps", type=int, default=2)
    ap.add_argument("--latent-bridge", choices=["linear", "hermite_pde", "collapsed_hermite"],
                    default="hermite_pde")
    ap.add_argument("--n-cal-traj", type=int, default=16)
    ap.add_argument("--n-val-traj", type=int, default=8)
    ap.add_argument("--n-test-traj", type=int, default=16)
    ap.add_argument("--lambda-grid", nargs="+", type=float, default=[0, 1e-6, 1e-5, 1e-4])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--map-steps", type=int, default=25)
    ap.add_argument("--map-lr", type=float, default=0.5)
    ap.add_argument("--divergence-weight", type=float, default=1.0)
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.lead_steps != 1 or args.dense_substeps != 2:
        raise SystemExit("first dense residual implementation supports native Poseidon lead=1, dense_substeps=2 only")
    set_seed(args.seed)
    args.fm, args.fm_channels = "poseidon", "velocity"
    configure_native_poseidon_cadence(args)
    fm, spec = load_fm(args), None
    spec = native_poseidon_spec(args, fm)
    cal = _load_fine(args.fm_data_path, fm, args.n_cal_traj, args.steps, args.dense_substeps, 0)
    val = _load_fine(args.fm_data_path, fm, args.n_val_traj, args.steps, args.dense_substeps, args.n_cal_traj)
    test = _load_fine(args.fm_data_path, fm, args.n_test_traj, args.steps, args.dense_substeps, args.n_cal_traj + args.n_val_traj)
    print_header(f"S4 dense temporal-residual HILP: {fm.info.name}, horizon={args.steps}")
    print(f"  latent temporal nodes: {args.dense_substeps} per native Poseidon transition; fine dt={POSEIDON_RAW_DT:.3g}")
    # q keeps the previous calibration convention, while all residual/bridge
    # quantities are fitted on calibration trajectories only.
    xs = [cal[i, 2 * t] for i in range(len(cal)) for t in range(args.steps)]
    ys = [cal[i, 2 * (t + 1)] for i in range(len(cal)) for t in range(args.steps)]
    sigma2 = fit_sigma2(fm, xs, ys)["sigma2"]
    scales = _scales(cal, spec, args.steps, args.dense_substeps, args, fm.device)
    if args.latent_bridge == "collapsed_hermite":
        bridge = None
        print(f"  calibration q={sigma2:.4g}; bridge=collapsed Hermite (no latent variance); residual scale={scales[0]:.4g}")
    else:
        bridge = _bridge_variance(cal, spec, args.dense_substeps, args.latent_bridge,
                                  args.divergence_weight, fm.device)
        print(f"  calibration q={sigma2:.4g}; bridge={args.latent_bridge}; bridge variance={bridge:.4g}; residual scale={scales[0]:.4g}")
    cal_p = _package(fm, cal, spec, sigma2, bridge, scales, args)
    val_p = _package(fm, val, spec, sigma2, bridge, scales, args)
    # Gate: truth fine paths must be more physically consistent than raw path
    # with latent linear interpolation, otherwise this residual is rejected.
    # In collapsed mode the test is exactly the endpoint-only energy evaluated
    # on true versus raw endpoint windows.  Free-node modes retain the dense
    # true path comparison.
    gate_input = lambda p: p["truth_end"] if args.latent_bridge == "collapsed_hermite" else p["truth_fine"]
    gate_truth = np.median([p["residual"].rms(gate_input(p)) for p in cal_p])
    gate_raw = np.median([p["residual"].rms(p["isotropic"].mean) for p in cal_p])
    wins = np.mean([p["residual"].rms(gate_input(p)) < p["residual"].rms(p["isotropic"].mean) for p in cal_p])
    print(f"  dense residual gate: truth={gate_truth:.4g}, raw-{args.latent_bridge}={gate_raw:.4g}, truth/raw={gate_truth/max(gate_raw,1e-30):.3f}, truth wins={100*wins:.1f}%")
    if wins < 0.5:
        raise SystemExit("dense residual gate failed: do not interpret a correction result")
    alphas = {m: _fit_alpha(cal_p, m) for m in args.methods}
    selected, curves = {}, {}
    for m in args.methods:
        cells = []
        print(f"\n  selecting lambda for {m} on validation")
        for lam in args.lambda_grid:
            out = _evaluate(val_p, m, alphas[m], lam, args)
            cells.append({"lambda": lam, **out})
            print(f"    lambda={lam:g}: endpoint RMSE={out['rmse']:.6g}")
        selected[m] = min(cells, key=lambda x: x["rmse"])["lambda"]
        curves[m] = cells
    test_p = _package(fm, test, spec, sigma2, bridge, scales, args)
    raw = _evaluate(test_p, "isotropic", 1.0, 0.0, args)
    table = Table("method", "endpoint RMSE", "gain%", "corr RMS", "temporal E ratio", "div E ratio", "physics E ratio")
    table.add("raw", raw["rmse"], 0.0, 0.0, 1.0, 1.0, 1.0)
    held = {"raw": raw}
    for m in args.methods:
        out = _evaluate(test_p, m, alphas[m], selected[m], args)
        held[m] = out
        table.add(m, out["rmse"], 100 * (1 - out["rmse"] / raw["rmse"]),
                  out["endpoint_correction_rms"],
                  out["temporal_energy"] / max(out["raw_temporal_energy"], 1e-30),
                  out["divergence_energy"] / max(out["raw_divergence_energy"], 1e-30),
                  out["physics_energy"] / max(out["raw_physics_energy"], 1e-30),
                  )
    print("\n  held-out dense-window evaluation")
    print(table)
    payload = {"stage": "s4_dense_temporal_residual", "metadata": fm_metadata(args, fm, spec),
               "sigma2": sigma2, "bridge_variance": bridge, "residual_scale": scales[0],
               "divergence_scale": scales[1], "gate": {"truth": gate_truth, "raw": gate_raw, "wins": float(wins)},
               "alphas": alphas, "lambda_validation": curves, "selected_lambda": selected, "held_out": held,
               "validity": {"physics": "centered fine-window PDE residual; no endpoint flow is evaluated",
                            "future_truth_used_at_inference": False, "latent_nodes": args.dense_substeps,
                            "latent_bridge": args.latent_bridge}}
    path = save_json(payload, results_path_scale("s4_dense_residual", fm_result_key(args), "results.json", args.tag))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
