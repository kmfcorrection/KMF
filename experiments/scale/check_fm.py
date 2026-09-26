#!/usr/bin/env python3
"""check_fm -- can this pretrained foundation model be used at all?

Run against any checkpoint *before* building an experiment on it. Two questions,
and a "no" to either means the model cannot support this method's claims -- for
entirely different reasons.

1. **Is it probeable?** `check_adapter` verifies the properties every downstream
   stage silently assumes: f maps R^N -> R^N, it is deterministic, forward-mode
   AD works, JVP agrees with a central difference, J is linear in the tangent,
   and the adjoint identity <Jv, w> = <v, J^T w> holds. Each catches a distinct
   way an unfamiliar checkpoint breaks the probes -- a non-differentiable
   preprocessing step, a stochastic head left in train mode, a custom kernel
   with a wrong backward. These fail silently otherwise: the estimators still
   return a covariance, it is just meaningless.

2. **Does it have geometry?** This is the lesson of s5. If the model's output
   step is short relative to the flow's own timescale, its Jacobian is
   near-identity, its spectrum is flat, and every rank-k subspace is equivalent
   -- so the method finds nothing regardless of how good the model is.
   `||J-I||/||I||` and the captured spectral fraction are reported here so that
   is known before any GPU time is committed.

   For most foundation models (DPOT, Walrus) the step is fixed by the
   pretraining corpus and this is a pass/fail gate. Poseidon takes the lead time
   as an *input*, so `--lead-times` sweeps it and the gate becomes a
   measurement: the same table s5 produced on the true solver, produced on a
   pretrained model.

Both questions are asked at states drawn from `--states`, which matters more
than it looks: a foundation model's Jacobian at `randn` is a fact about
extrapolation, not about physics. The default is on-manifold; `gaussian` is kept
as a control.

    python3 experiments/scale/check_fm.py --fm poseidon --fm-size B \
        --fm-channels velocity --states rollout --lead-times 0.25,1,4,16 --k 64
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hipp.scale.adapters import check_adapter
from hipp.scale.common_scale import (base_parser_scale, fm_probe_states,
                                     load_fm, results_path_scale)
from hipp.scale.jacobian import (frobenius_norm_estimate, jacobian_ops,
                                 randomized_svd, set_probe_precision,
                                 subspace_overlap)
from hipp.utils import Table, get_device, print_header, save_json, set_seed


def lowpass_basis(state_shape, k: int, device, dtype) -> torch.Tensor:
    """Orthonormal basis of the `k` lowest-wavenumber real Fourier modes.

    The reference subspace for "J is just a smoothing filter". It is fixed --
    it does not depend on the state at all -- so a leading subspace that
    coincides with it is by construction not state-dependent geometry.
    """
    c, h, w = state_shape
    ky = torch.fft.fftfreq(h) * h
    kx = torch.fft.fftfreq(w) * w
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    order = torch.argsort((KX ** 2 + KY ** 2).reshape(-1))
    y = torch.arange(h, dtype=torch.float64).view(-1, 1)
    x = torch.arange(w, dtype=torch.float64).view(1, -1)
    cols = []
    for idx in order.tolist():
        a, b = float(KY.reshape(-1)[idx]), float(KX.reshape(-1)[idx])
        ph = 2 * math.pi * (a * y / h + b * x / w)
        for field in (torch.cos(ph), torch.sin(ph)):
            if float(field.abs().max()) < 1e-12:
                continue
            for ch in range(c):
                v = torch.zeros(c, h, w, dtype=torch.float64)
                v[ch] = field
                cols.append(v.reshape(-1))
                if len(cols) >= k:
                    Q, _ = torch.linalg.qr(torch.stack(cols, 1))
                    return Q.to(device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(torch.stack(cols, 1))
    return Q.to(device=device, dtype=dtype)


def geometry_at(fm, states, args, device, lowpass=None) -> dict:
    """Deviation from identity, conditioning, captured trace and -- the part
    that decides whether any of it means anything -- how much the leading
    subspace moves between states. Matrix-free: rank k costs O(k) passes.

    A steep spectrum on its own is *not* evidence for the method. A strongly
    dissipative operator has one trivially, and if its leading subspace is the
    same at every state then a single global covariance reproduces it and the
    per-state curvature buys nothing. That is the `advdiff` outcome from the 1D
    study (overlap 0.97), reached there through a linear PDE rather than through
    a long lead time, and it is the alternative explanation for a large
    captured-trace ratio that has to be excluded rather than assumed away.
    """
    dev_i, cap, cond, lp, bases = [], [], [], [], []
    t0, vmap_ok = time.time(), None
    for i, c in enumerate(states):
        ops = jacobian_ops(fm.flat_fn(), c)
        U, S, _ = randomized_svd(
            ops, k=args.k, oversample=args.oversample, n_iter=args.n_iter,
            chunk=args.chunk,
            generator=torch.Generator(device="cpu").manual_seed(args.seed + i))
        fro = frobenius_norm_estimate(jacobian_ops(fm.flat_fn(), c),
                                      n_probe=args.n_probe, chunk=args.chunk)
        cap.append(float((S.double() ** 2).sum()) / max(fro, 1e-30))
        cond.append(float(S.max() / S.min().clamp(min=1e-30)))
        bases.append(U)
        if lowpass is not None:
            lp.append(subspace_overlap(U, lowpass))
        v = torch.randn(fm.N, device=device, dtype=fm.dtype)
        v = v / v.norm()
        dev_i.append(float((ops.matvec(v) - v).norm()))
        # Whether torch.vmap survives this architecture is the single biggest
        # throughput factor -- probe_map falls back to a one-at-a-time loop for
        # any operator it cannot trace, which is numerically identical and
        # several times slower. Report it so a slow run is diagnosable.
        if vmap_ok is None:
            vmap_ok = getattr(ops.matvec, "_vmap_ok", None)

    pairs = [subspace_overlap(bases[i], bases[j])
             for i in range(len(bases)) for j in range(i + 1, len(bases))]
    frac = args.k / fm.N
    return {"dev_from_identity": float(np.mean(dev_i)),
            "cond_k": float(np.mean(cond)),
            "captured_trace": float(np.mean(cap)),
            "isotropic_trace": frac,
            "captured_over_isotropic": float(np.mean(cap)) / frac,
            "subspace_overlap_across_states": float(np.mean(pairs)) if pairs else float("nan"),
            "subspace_overlap_lowpass": float(np.mean(lp)) if lp else float("nan"),
            # Two *independent* k-dim subspaces of R^N already share k/N by
            # chance, so the overlaps above are read against this, not against
            # zero. At k=64, N=32,768 chance is 0.002 -- an overlap of 0.01
            # is five times chance, not "no shared structure".
            "subspace_overlap_chance": frac,
            "seconds_per_state": (time.time() - t0) / max(len(states), 1),
            "vmap": vmap_ok}


def verdict_of(ratio: float) -> str:
    return "degenerate" if ratio < 1.5 else "weak" if ratio < 3 else "usable"


def main():
    ap = base_parser_scale(__doc__.splitlines()[0])
    ap.add_argument("--n-states", type=int, default=4)
    ap.add_argument("--n-probe", type=int, default=16,
                    help="Hutchinson probes for ||J||_F, the denominator of the "
                         "captured fraction")
    ap.add_argument("--states", default="fluid",
                    choices=["fluid", "rollout", "gaussian"],
                    help="where J is evaluated; see common_scale.fm_probe_states")
    ap.add_argument("--rollout-steps", type=int, default=2)
    ap.add_argument("--lead-times", default=None,
                    help="comma-separated cadence sweep for time-conditioned "
                         "models (Poseidon). Defaults to --lead-time alone.")
    args = ap.parse_args()

    set_seed(args.seed)
    set_probe_precision(high=not args.allow_tf32)
    device = get_device(args.device)

    print_header(f"check_fm: {args.fm}" + (f"-{args.fm_size}"
                                           if args.fm != "local" else ""))
    try:
        fm = load_fm(args, device=device)
    except ImportError as exc:
        print(f"  cannot load: {exc}")
        print("  -> install the adapter's package (see requirements-scale.txt)")
        return 2
    except (FileNotFoundError, NotImplementedError) as exc:
        print(f"  cannot load: {str(exc).strip().splitlines()[0]}")
        print(f"\n{exc}")
        return 2
    print(f"  {fm}")
    print(f"  states={args.states}  free channels={getattr(fm, 'free_channels', 'all')}")

    states = fm_probe_states(fm, args.n_states, kind=args.states, seed=args.seed,
                             device=device, rollout_steps=args.rollout_steps)

    # ---- 1. probeable? --------------------------------------------------
    # Checked at a real state, not at randn: a non-differentiable preprocessing
    # step or a wrong custom backward can be invisible off the data manifold.
    print("\n1. Is it probeable?")
    chk = check_adapter(fm, x=states[0], verbose=False)
    t = Table("check", "value", "requirement", "ok")
    rows = [("shape f: R^N -> R^N", chk.get("shape_ok"), "True"),
            ("deterministic", chk.get("deterministic"), "True"),
            ("forward-mode AD (jvp)", chk.get("jvp_ok"), "True"),
            ("reverse-mode AD (vjp)", chk.get("vjp_ok"), "True"),
            ("jvp vs central difference", chk.get("jvp_vs_fd_rel"), "< 1e-2"),
            ("  at relative step", chk.get("jvp_vs_fd_rel_step"), None),
            ("J linear in tangent", chk.get("linearity_rel"), "< 1e-4"),
            ("adjoint <Jv,w>=<v,J^T w>", chk.get("adjoint_rel"), "< 1e-4")]
    for name, val, req in rows:
        if req is None:                       # context line, nothing to assert
            t.add(name, val, "", "")
            continue
        ok = (val is True if isinstance(val, bool) else
              (val is not None and val < float(req.split("<")[1])))
        t.add(name, val, req, "yes" if ok else "NO")
    print(t)
    for key in ("jvp_error", "vjp_error"):
        if chk.get(key):
            print(f"  {key}: {chk[key]}")
    if not chk["pass"]:
        print("  -> FAILED. Nothing downstream is trustworthy against this "
              "checkpoint; fix the adapter before running any stage.")

    # ---- 2. does it have geometry? --------------------------------------
    leads = ([float(s) for s in args.lead_times.split(",")]
             if args.lead_times else [getattr(fm, "lead_time", None)])
    sweeps = hasattr(fm, "set_lead_time") and leads[0] is not None
    print(f"\n2. Does it have geometry?  (rank {args.k} of N={fm.N}, "
          f"isotropic = {args.k / fm.N:.4f})")

    lowpass = lowpass_basis(fm.state_shape, args.k, device, fm.dtype)
    rows_out, geo = [], None
    t = Table("lead time", "||Jv-v||/||v||", "cond(J) rank k", "captured trace",
              "/ isotropic", "U overlap: states", "U overlap: lowpass",
              "s/state", "verdict")
    for lt in leads:
        if sweeps:
            fm.set_lead_time(lt)
            # Rollout states are cadence-dependent by construction, so they are
            # regenerated per lead time rather than shared across the sweep.
            if args.states == "rollout":
                states = fm_probe_states(fm, args.n_states, kind=args.states,
                                         seed=args.seed, device=device,
                                         rollout_steps=args.rollout_steps)
        geo = geometry_at(fm, states, args, device, lowpass=lowpass)
        geo["lead_time"] = lt
        geo["verdict"] = verdict_of(geo["captured_over_isotropic"])
        rows_out.append(geo)
        t.add("native" if lt is None else f"{lt:g}", geo["dev_from_identity"],
              geo["cond_k"], geo["captured_trace"],
              geo["captured_over_isotropic"],
              geo["subspace_overlap_across_states"],
              geo["subspace_overlap_lowpass"],
              round(geo["seconds_per_state"], 2), geo["verdict"])
    print(t)
    print(f"  probe batching: "
          f"{'vmap' if rows_out[0].get('vmap') else 'sequential fallback'}"
          f" (chunk {args.chunk}); ~{6 * (args.k + args.oversample) + args.n_probe}"
          f" network passes per state")
    print(f"  U overlap: 1.0 = the leading subspace does not move, and a single "
          f"global covariance would do\n  as well as a per-state one. "
          f"{args.k / fm.N:.2e} = chance for independent subspaces.")

    best = max(rows_out, key=lambda r: r["captured_over_isotropic"])
    # Ranked by how much the subspace moves, not by how anisotropic J is: an
    # operator can be extremely anisotropic and still carry a fixed subspace,
    # and only the latter distinguishes this method from a global prior.
    moving = min(rows_out, key=lambda r: r["subspace_overlap_across_states"])
    print("\nVERDICT")
    print(f"  probeable:  {'yes' if chk['pass'] else 'NO'}")
    print(f"  geometry:   {best['verdict']}  "
          f"({best['captured_over_isotropic']:.2f}x isotropic"
          + (f" at lead time {best['lead_time']:g})" if sweeps else ")"))
    ov = moving["subspace_overlap_across_states"]
    print(f"  state-dependent: {'yes' if ov < 0.5 else 'WEAK' if ov < 0.9 else 'NO'}"
          f"  (lowest cross-state subspace overlap {ov:.4f}, "
          f"chance {args.k / fm.N:.2e}"
          + (f", at lead time {moving['lead_time']:g})" if sweeps else ")"))
    if ov >= 0.9:
        print("  -> The leading subspace barely moves between states. The "
              "captured-trace number above is then a fact about the operator "
              "being dissipative, not about per-state geometry, and s1's "
              "direction claim is the advdiff null. Check the low-pass overlap "
              "column: if that is also near 1, J is a smoothing filter.")
    if best["verdict"] == "degenerate":
        print("  -> This model's operator is effectively isotropic at rank "
              f"{args.k}. That is usually the output cadence being short "
              "relative to the flow timescale (docs/scaling_plan.md 7), and it "
              "is not fixable by training or by raising k -- the geometry is "
              "not there. Check what dt the checkpoint was pretrained at.")
    elif sweeps and len(rows_out) > 1:
        print("  -> The cadence is an input to this model, so the row above is "
              "a choice, not a constraint. Run the stages at the lead time that "
              "the one-step error still supports.")

    out = {"fm": args.fm, "size": getattr(args, "fm_size", None), "N": fm.N,
           "params": fm.info.n_params, "k": args.k,
           "states": args.states, "n_states": args.n_states,
           "free_channels": list(getattr(fm, "free_channels", []) or []),
           "adapter_check": chk, "sweep": rows_out,
           "verdict": best["verdict"],
           "captured_over_isotropic": best["captured_over_isotropic"]}
    tag = "_".join(x for x in [getattr(args, "fm_size", "") or "", args.states,
                               args.tag] if x)
    p = save_json(out, results_path_scale("check_fm", args.fm, "results.json", tag))
    print(f"\nwrote {p}")
    return 0 if (chk["pass"] and best["verdict"] != "degenerate") else 1


if __name__ == "__main__":
    sys.exit(main())
