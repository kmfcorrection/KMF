#!/usr/bin/env python3
"""Convert the official CNO training text files into the adapter JSON contract.

The CNO checkpoint stores weights but not every constructor argument.  This
tool keeps the benchmark tied to the architecture files shipped beside that
checkpoint rather than reconstructing its architecture by guesswork.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _read_pairs(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                out[row[0].strip()] = row[1].strip()
    return out


def _number(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _boolean(value: str) -> bool:
    """Parse the boolean spellings used by released CNO metadata."""
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected a boolean metadata value, got {value!r}")


def _nl_dims(value: str):
    """Parse compact or comma/bracketed CNO normalization dimensions."""
    raw = str(value).strip()
    compact = raw.replace(" ", "").replace("[", "").replace("]", "")
    compact = compact.replace("(", "").replace(")", "")
    if compact in {"23", "2,3"}:
        return [2, 3]
    if compact in {"023", "0,2,3"}:
        return [0, 2, 3]
    if compact in {"123", "1,2,3"}:
        return [1, 2, 3]
    parts = [p for p in re.split(r"[,;]", compact) if p]
    if parts and all(p.isdigit() for p in parts):
        dims = [int(p) for p in parts]
        if dims in ([2, 3], [0, 2, 3], [1, 2, 3]):
            return dims
    raise ValueError(f"unrecognised nl_dim metadata value: {value!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True,
                    help="Directory holding net_architecture.txt and training_properties.txt from the released CNO run.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--in-dim", type=int, default=5)
    ap.add_argument("--out-dim", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.artifact_dir)
    arch = _read_pairs(root / "net_architecture.txt")
    train = _read_pairs(root / "training_properties.txt")
    keys = ("N_layers", "N_res", "N_res_neck", "channel_multiplier", "batch_norm",
            "activation", "is_time", "nl_dim", "is_att", "patch_size",
            "dim_multiplier", "depth", "heads", "dim_head_multiplier",
            "mlp_dim_multiplier", "emb_dropout")
    missing = [key for key in keys if key not in arch]
    if missing:
        raise ValueError(f"{root / 'net_architecture.txt'} lacks {missing}")
    if "time_steps" not in train:
        raise ValueError(f"{root / 'training_properties.txt'} lacks time_steps")

    cfg = {key: _number(arch[key]) for key in keys}
    cfg["time_steps"] = int(_number(train["time_steps"]))
    cfg["in_size"] = int(_number(arch.get("in_size", "128")))
    cfg["in_dim"] = args.in_dim
    cfg["out_dim"] = args.out_dim
    cfg["batch_norm"] = _boolean(cfg["batch_norm"])
    # The released CNO metadata writes this as ``True``/``False``.  Passing
    # the literal string through makes CNO_time skip its FiLM modules because
    # the official constructor checks ``is_time == True``.
    raw_is_time = str(arch["is_time"]).strip().lower()
    if raw_is_time in {"0", "1", "true", "false", "yes", "no", "y", "n", "on", "off"}:
        cfg["is_time"] = _boolean(arch["is_time"])
    else:
        # Some released CNO metadata records the temporal mode as an integer
        # code (for example ``4``), while CNO_time expects a boolean switch.
        # The checkpoint's FiLM keys establish that this is the time-conditioned
        # CNO-FM variant, so any positive numeric code means enabled.
        try:
            cfg["is_time"] = int(float(arch["is_time"])) != 0
        except ValueError as exc:
            raise ValueError(f"unrecognised is_time metadata value: {arch['is_time']!r}") from exc
    cfg["is_att"] = _boolean(cfg["is_att"])
    cfg["nl_dim"] = _nl_dims(arch["nl_dim"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
