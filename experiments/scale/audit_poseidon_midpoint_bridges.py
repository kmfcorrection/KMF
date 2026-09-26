#!/usr/bin/env python3
"""Evaluate inference-time midpoint bridges for the strong PDE residual.

The stored raw Poseidon frame at t+.05 is used only after bridge construction
to measure bridge-state accuracy and residual alignment.  At inference each
bridge receives only x_t, an endpoint y (either a truth endpoint for an upper
bound or a frozen Poseidon endpoint), and known PDE RHS evaluations.  No
candidate advances a PDE endpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scale.fm_eval_common import (POSEIDON_RAW_DT,
                                               configure_native_poseidon_cadence,
                                               native_poseidon_spec)
from hipp.scale.common_scale import base_parser_scale, load_fm
from hipp.scale.fm_physics import FMPhysicsEnergy2D
from hipp.utils import Table, print_header, save_json, set_seed


def _d7(w: torch.Tensor, j: int, h: float) -> torch.Tensor:
    return (-w[:, j - 3] + 9 * w[:, j - 2] - 45 * w[:, j - 1]
            + 45 * w[:, j + 1] - 9 * w[:, j + 2] + w[:, j + 3]) / (60 * h)


def _metric(estimate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    e, r = estimate.flatten(1), reference.flatten(1)
    relative = (e - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)
    cosine = (e * r).sum(1) / (e.norm(dim=1) * r.norm(dim=1)).clamp_min(1e-30)
    return float(relative.mean()), float(cosine.mean())


def _velocity_rhs(operator: FMPhysicsEnergy2D, state: torch.Tensor) -> torch.Tensor:
    """Known PDE tendency in [u,v] coordinates; evaluates RHS, never a flow."""
    flat = state.reshape(-1, operator.N).to(operator.device, torch.float64)
    w = operator.to_vorticity(flat)
    rhs_w = operator.rhs_vorticity(w)
    du, dv = operator.grid.velocity(torch.fft.rfft2(rhs_w))
    return torch.stack((du, dv), dim=1).reshape_as(flat)


def _bridges(operator: FMPhysicsEnergy2D, x0: torch.Tensor, endpoint: torch.Tensor,
             dt: float) -> dict[str, torch.Tensor]:
    """Algebraic midpoint candidates.  All return a t+.5dt velocity state."""
    x0, endpoint = x0.double(), endpoint.double()
    f0, f1 = _velocity_rhs(operator, x0), _velocity_rhs(operator, endpoint)
    linear = .5 * (x0 + endpoint)
    # Cubic Hermite at a=1/2: uses endpoint tangent evaluations only.
    hermite = .5 * (x0 + endpoint) + .125 * dt * (f0 - f1)
    # One Picard iterate of a half-time collocation equation, initialized by
    # the endpoint average.  This is one RHS evaluation, not a time step.
    picard_half = x0 + .5 * dt * _velocity_rhs(operator, linear)
    # SSPRK3's c=1/2 internal stage.  It is an RHS-stage diagnostic control,
    # deliberately reported separately because it ignores the supplied endpoint.
    stage1 = x0 + dt * f0
    rk3_c_half = .75 * x0 + .25 * (stage1 + dt * _velocity_rhs(operator, stage1))
    return {"linear": linear, "hermite_pde": hermite,
            "picard_half": picard_half, "rk3_c_half_control": rk3_c_half}


def _load_raw(path: Path, n_traj: int, offset: int, device: torch.device) -> torch.Tensor:
    with h5py.File(path, "r") as h:
        key = "velocity" if "velocity" in h else "solution"
        ds = h[key]
        arr = np.asarray(ds[offset:offset + n_traj, :, :2], dtype=np.float64)
    return torch.as_tensor(arr, device=device, dtype=torch.float64)


def main() -> None:
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--lead-steps", type=int, default=1,
                    help="must be one native Poseidon transition (dt=0.1)")
    ap.add_argument("--n-traj", type=int, default=32)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--centers", nargs="*", type=int, default=None)
    ap.add_argument("--max-centers", type=int, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/scale/poseidon_midpoint_bridge_audit"))
    ap.add_argument("--fm-data-path", required=True)
    args = ap.parse_args()
    if args.n_traj < 1:
        raise SystemExit("n-traj must be positive")
    if args.lead_steps != 1:
        raise SystemExit("this midpoint bridge audit is defined only for one native Poseidon transition (lead-steps=1)")
    set_seed(args.seed)
    args.fm, args.fm_size, args.fm_channels = "poseidon", args.fm_size, "velocity"
    configure_native_poseidon_cadence(args)
    fm = load_fm(args)
    raw = _load_raw(Path(args.fm_data_path), args.n_traj, args.offset, fm.device)
    n_raw, n = raw.shape[1], raw.shape[-1]
    centers = (args.centers if args.centers is not None else list(range(3, n_raw - 3, 2)))
    if args.max_centers is not None:
        centers = centers[:args.max_centers]
    if not centers or any(j < 3 or j + 3 >= n_raw for j in centers):
        raise SystemExit(f"centers must lie in [3, {n_raw - 4}]")
    spec = native_poseidon_spec(args, fm)
    all_states = raw.reshape(args.n_traj * n_raw, -1)
    operator = FMPhysicsEnergy2D(spec, all_states, 2, divergence_weight=0.0,
                                 dt=2 * POSEIDON_RAW_DT, device=fm.device)
    with torch.no_grad():
        all_w = operator.to_vorticity(all_states).reshape(args.n_traj, n_raw, n, n)

    print_header(f"Poseidon midpoint-bridge fidelity audit: {fm.info.name}")
    print(f"  split: trajectories={args.n_traj}; raw centers={centers}; grid={n}x{n}")
    print("  truth midpoints are diagnostic only; FM endpoints are teacher-forced from x_t")
    print("  bridges use RHS evaluations only; no endpoint flow integration")

    buckets: dict[str, dict[str, list[torch.Tensor]]] = {
        source: {name: [] for name in ("linear", "hermite_pde", "picard_half", "rk3_c_half_control")}
        for source in ("truth_endpoint_upper_bound", "fm_endpoint_inference")}
    residuals: dict[str, dict[str, list[torch.Tensor]]] = {
        source: {name: [] for name in ("linear", "hermite_pde", "picard_half", "rk3_c_half_control")}
        for source in buckets}
    references = []

    for j in centers:
        x0 = raw[:, j - 1].reshape(args.n_traj, -1)
        midpoint_truth = raw[:, j].reshape(args.n_traj, -1)
        endpoint_truth = raw[:, j + 1].reshape(args.n_traj, -1)
        with torch.no_grad():
            endpoint_fm = fm.predict(x0.to(fm.device, fm.dtype)).double().reshape(args.n_traj, -1)
            ref = _d7(all_w, j, POSEIDON_RAW_DT) - operator.rhs_vorticity(all_w[:, j])
        references.append(ref.cpu())
        for source, endpoint in (("truth_endpoint_upper_bound", endpoint_truth),
                                 ("fm_endpoint_inference", endpoint_fm)):
            for name, bridge in _bridges(operator, x0, endpoint, 2 * POSEIDON_RAW_DT).items():
                with torch.no_grad():
                    buckets[source][name].append(bridge.cpu())
                    w0 = operator.to_vorticity(x0)
                    wy = operator.to_vorticity(endpoint)
                    wb = operator.to_vorticity(bridge)
                    # This is exactly the candidate-path PDE residual used by
                    # the proposed midpoint HILP energy, with a bridge state.
                    r = (wy - w0) / (2 * POSEIDON_RAW_DT) - operator.rhs_vorticity(wb)
                    residuals[source][name].append(r.cpu())

    midpoint_ref = torch.cat([raw[:, j].reshape(args.n_traj, -1).cpu() for j in centers])
    residual_ref = torch.cat(references)
    rows, payload = [], {"reference": "sixth-order truth derivative minus native RHS", "centers": centers}
    for source in buckets:
        payload[source] = {}
        for name in buckets[source]:
            bridge = torch.cat(buckets[source][name])
            r = torch.cat(residuals[source][name])
            bridge_rel, bridge_cos = _metric(bridge, midpoint_ref)
            residual_rel, residual_cos = _metric(r, residual_ref)
            payload[source][name] = {"midpoint_state_relative_error": bridge_rel,
                                     "midpoint_state_cosine": bridge_cos,
                                     "residual_relative_error": residual_rel,
                                     "residual_direction_cosine": residual_cos}
            rows.append((source, name, bridge_rel, bridge_cos, residual_rel, residual_cos))
    table = Table("endpoint source", "bridge", "mid state rel.err", "mid cosine", "resid rel.err", "resid cosine")
    for row in rows:
        table.add(*row)
    print("\n" + str(table))
    print("\n  Deployment gate: inspect fm_endpoint_inference only. A useful bridge needs")
    print("  a low midpoint-state error and strongly positive residual cosine against the reference.")
    path = save_json(payload, args.out_dir / f"{args.tag}_results.json")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
