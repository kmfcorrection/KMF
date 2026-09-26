#!/usr/bin/env python3
"""Causal Task-2 correction benchmark for a frozen PDE foundation model.

Rows share the same initial condition, Poseidon rollout, operator, calibration
trajectories, test trajectories, horizon, and state normalization:
  raw, projection, KMF Simpson, residual gradient, live Gauss--Newton,
  cached PhysicsCorrect-style linearized residual correction, and (NS only)
  a fixed-budget native-flow intervention.

The cached row ports PhysicsCorrect's *algorithm* (one calibration-only
Jacobian/pseudoinverse cache and per-step linear residual correction) to a
reduced vorticity state. It is explicitly labelled as an adapter because the
official release uses a 64x64 stream-function model, not Poseidon's 128x128
two-component velocity contract. No row uses test truth to choose a parameter.
"""
from __future__ import annotations

import argparse, dataclasses, json, sys, time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from experiments.scale.cross_fm_benchmark import load_dataset_trajectories, fm_predict
from experiments.scale.fm_eval_common import native_poseidon_spec
from experiments.scale.validate_task2_rollout_extension import project, unit_defect_update
from hipp.scale.common_scale import load_fm, set_seed
from hipp.scale.data2d import SPECS2D
from hipp.scale.fm_physics import FMPhysicsEnergy2D, spectral_resample_state


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)).mean())


def low_spec(args: argparse.Namespace, fm: Any, name: str):
    if name == "NS-Gauss":
        return dataclasses.replace(native_poseidon_spec(args, fm), n=args.pc_grid, dt=args.dt)
    return dataclasses.replace(SPECS2D.get("kolmogorov", native_poseidon_spec(args, fm)), n=args.pc_grid, dt=args.dt)


def vorticity_to_velocity(op: FMPhysicsEnergy2D, w: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    u, v = op.grid.velocity(torch.fft.rfft2(w))
    return torch.stack((u, v), dim=1) + mean


class ReducedResidual:
    """Common reduced residual representation, lifted back to the FM state."""
    def __init__(self, full_op: FMPhysicsEnergy2D, spec, args: argparse.Namespace):
        zero = torch.zeros(1, 2, args.pc_grid, args.pc_grid, device=full_op.device, dtype=torch.float64)
        self.full_op, self.low_op, self.n = full_op, FMPhysicsEnergy2D(
            spec, zero, 2, dt=args.dt, device=full_op.device), args.pc_grid
        self.dt = args.dt

    def down(self, velocity: torch.Tensor) -> torch.Tensor:
        return spectral_resample_state(self.full_op.to_vorticity(velocity), self.n)

    def up_delta(self, delta: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        high = spectral_resample_state(delta, self.full_op.n)
        return vorticity_to_velocity(self.full_op, high, torch.zeros_like(previous[:, :1]))

    def residual(self, previous_low: torch.Tensor, candidate_low: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (previous_low + candidate_low)
        return (candidate_low - previous_low) / self.dt - self.low_op.rhs_vorticity(midpoint)


def cache_physicscorrect(repr_: ReducedResidual, previous: torch.Tensor,
                         calibration_prediction: torch.Tensor) -> tuple[torch.Tensor, float]:
    """PhysicsCorrect cache: one calibration-only Jacobian and pseudoinverse."""
    w_prev = repr_.down(previous[:1])[0]
    w_target = repr_.down(calibration_prediction[:1])[0]
    start = time.perf_counter()
    jac = torch.autograd.functional.jacobian(
        lambda z: repr_.residual(w_prev.unsqueeze(0), z.unsqueeze(0))[0], w_target,
        vectorize=True,
    ).reshape(repr_.n * repr_.n, repr_.n * repr_.n)
    pinv = torch.linalg.pinv(jac, rtol=1e-5)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return pinv.detach(), time.perf_counter() - start


def correction(method: str, raw: torch.Tensor, previous: torch.Tensor, op: FMPhysicsEnergy2D,
               repr_: ReducedResidual | None, pc_pinv: torch.Tensor | None, step: float,
               gn_damping: float, flow_substeps: int) -> torch.Tensor:
    if method == "raw": return raw
    pred = project(op, raw)
    if method == "projection": return pred
    if method == "kmf":
        _, delta = unit_defect_update(op, previous, raw, op.n)
        return project(op, pred + step * delta)
    if method == "solver":
        if not op.poseidon_spectral_viscosity: return pred
        op.previous, op.previous_vorticity = previous.detach(), op.to_vorticity(previous).detach()
        w = op.native_flow_vorticity(op.previous_vorticity, substeps=flow_substeps)
        state = vorticity_to_velocity(op, w, previous.mean(dim=(-2, -1), keepdim=True))
        return project(op, (1.0 - step) * pred + step * state)
    if repr_ is None: raise RuntimeError(f"{method} requires reduced residual representation")
    w_prev, w_pred = repr_.down(previous), repr_.down(pred)
    w = w_pred.detach().requires_grad_(method in {"gradient", "gn"})
    r = repr_.residual(w_prev, w)
    if method == "physicscorrect":
        assert pc_pinv is not None
        delta = -torch.einsum("ij,bj->bi", pc_pinv, r.reshape(len(r), -1)).reshape_as(w)
    elif method == "gradient":
        energy = 0.5 * r.square().mean()
        delta = -torch.autograd.grad(energy, w)[0]
    elif method == "gn":
        # Live per-state Gauss--Newton, intentionally expensive but not cached.
        out = []
        for b in range(len(w)):
            wb, wp = w[b].detach(), w_prev[b].detach()
            jac = torch.autograd.functional.jacobian(
                lambda z: repr_.residual(wp.unsqueeze(0), z.unsqueeze(0))[0], wb,
                vectorize=True,
            ).reshape(repr_.n * repr_.n, repr_.n * repr_.n)
            rb = r[b].reshape(-1).detach()
            eye = torch.eye(len(rb), device=rb.device, dtype=rb.dtype)
            out.append(torch.linalg.solve(jac.T @ jac + gn_damping * eye, -jac.T @ rb).reshape_as(wb))
        delta = torch.stack(out)
    else: raise ValueError(method)
    delta_high = repr_.up_delta(delta.detach(), previous)
    return project(op, pred + step * delta_high)


@torch.no_grad()
def rollout(fm: Any, trajectories: torch.Tensor, op: FMPhysicsEnergy2D, *, method: str,
            repr_: ReducedResidual | None, pc_pinv: torch.Tensor | None, step: float,
            args: argparse.Namespace, diagnostics: bool = False) -> dict[str, Any]:
    current = trajectories[:, 0].clone(); errors=[]; cos=[]; times=[]
    for t in range(args.steps):
        truth = trajectories[:, (t + 1) * args.stride]
        raw = fm_predict(fm, current, current_grid=args.grid)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.enable_grad() if method in {"gradient", "gn"} else torch.no_grad():
            nxt = correction(method, raw, current, op, repr_, pc_pinv, step, args.gn_damping, args.flow_substeps)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1e3)
        if diagnostics and method not in {"projection", "solver"}:
            baseline=project(op, raw); cos.append(cosine(nxt-baseline, truth-baseline))
        errors.append((nxt-truth).square().mean(dim=(1,2,3)).sqrt().detach().cpu().numpy())
        current=nxt.detach()
    arr=np.stack(errors,1)
    return {"window_rmse":float(arr.mean()),"final_rmse":float(arr[:,-1].mean()),
            "trajectory_window_rmse":arr.mean(1).tolist(),"per_step_rmse":arr.mean(0).tolist(),
            "online_correction_ms_mean":float(np.mean(times)),"online_correction_ms_p90":float(np.percentile(times,90)),
            "correction_error_cosine":float(np.mean(cos)) if cos else None,
            "finite":bool(np.isfinite(arr).all())}


def run_system(args: argparse.Namespace, fm: Any, name: str, path: Path) -> dict[str, Any]:
    total=args.n_cal+args.n_test
    trajectories,_=load_dataset_trajectories(path,fm,total,steps=args.steps*args.stride,stride=1,offset=args.offset,target_grid=args.grid)
    calibration,test=trajectories[:args.n_cal],trajectories[args.n_cal:]
    spec = dataclasses.replace(native_poseidon_spec(args,fm),n=args.grid,dt=args.dt) if name=="NS-Gauss" else dataclasses.replace(SPECS2D.get("kolmogorov",native_poseidon_spec(args,fm)),n=args.grid,dt=args.dt)
    op=FMPhysicsEnergy2D(spec,test[:1,0],2,dt=args.dt,device=fm.device)
    fm.set_lead_time(float(args.stride))
    reduced=ReducedResidual(op,low_spec(args,fm,name),args)
    with torch.no_grad():
        cache_prediction = project(op, fm_predict(fm, calibration[:1, 0], current_grid=args.grid))
    pc_pinv, cache_s=cache_physicscorrect(reduced, calibration[:1, 0], cache_prediction)
    available=["raw", "projection","kmf","gradient","gn","physicscorrect"] + (["solver"] if name=="NS-Gauss" else [])
    methods=[m for m in args.methods if m in available]
    for required in reversed(("raw", "projection")):
        if required not in methods:
            methods.insert(0, required)
    out={"physicscorrect_adapter":{"state":"reduced vorticity", "grid":args.pc_grid,
                                    "cache":"one calibration-only dense residual Jacobian pseudoinverse",
                                    "cache_seconds":cache_s,"official_contract_note":"algorithmic port; official release uses 64x64 stream-function"},"methods":{}}
    print(f"\n{name}: building calibration-only PhysicsCorrect cache: {cache_s:.2f}s")
    for method in methods:
        candidates=[0.0]+args.step_grid if method not in {"raw", "projection"} else [0.0]
        scores={}
        for step in candidates:
            scores[str(step)]=rollout(fm,calibration,op,method=method,repr_=reduced,pc_pinv=pc_pinv,step=step,args=args)["window_rmse"]
        chosen=min(candidates,key=lambda s:(scores[str(s)],s))
        test_out=rollout(fm,test,op,method=method,repr_=reduced,pc_pinv=pc_pinv,step=chosen,args=args,diagnostics=True)
        out["methods"][method]={"selected_step":chosen,"calibration_window_rmse_by_step":scores,**test_out}
        print(f"  {method:16s} step={chosen:g} test-window={test_out['window_rmse']:.6g} online={test_out['online_correction_ms_mean']:.2f}ms")
    base=np.asarray(out["methods"]["projection"]["trajectory_window_rmse"])
    for m,v in out["methods"].items():
        v["gain_vs_projection_pct"]=float(100*(1-np.mean(v["trajectory_window_rmse"])/np.mean(base)))
    return out


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--data-root",required=True);p.add_argument("--out-dir",required=True)
    p.add_argument("--fm",default="poseidon",choices=["poseidon"]);p.add_argument("--fm-size",default="B",choices=["T","B","L"]);p.add_argument("--fm-channels",default="velocity")
    p.add_argument("--grid",type=int,default=128);p.add_argument("--pc-grid",type=int,default=16);p.add_argument("--dt",type=float,default=.05);p.add_argument("--stride",type=int,default=1);p.add_argument("--steps",type=int,default=4)
    p.add_argument("--n-cal",type=int,default=5);p.add_argument("--n-test",type=int,default=20);p.add_argument("--step-grid",nargs="+",type=float,default=[.01,.02,.05,.1,.2]);p.add_argument("--gn-damping",type=float,default=1e-3);p.add_argument("--flow-substeps",type=int,default=16)
    p.add_argument("--methods",nargs="+",choices=["raw","projection","kmf","gradient","gn","physicscorrect","solver"],default=["raw","projection","kmf","gradient","gn","physicscorrect","solver"])
    p.add_argument("--offset",type=int,default=19760);p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");p.add_argument("--seed",type=int,default=20260924)
    args=p.parse_args()
    if args.dt != .05*args.stride: raise ValueError("require dt = 0.05 * stride for stored dataset cadence")
    set_seed(args.seed);fm=load_fm(args,device=torch.device(args.device)); results={"metadata":vars(args)}
    for name in ("NS-Gauss","FNS-KF"):
        path=Path(args.data_root)/f"{name}.nc"
        if path.exists(): results[name]=run_system(args,fm,name,path)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);(out/"task2_correction_suite.json").write_text(json.dumps(results,indent=2));print(f"wrote {out/'task2_correction_suite.json'}")
if __name__=="__main__": main()
