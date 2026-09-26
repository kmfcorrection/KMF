#!/usr/bin/env python3
"""Fail-closed placeholder for a future validated MOTION benchmark adapter.

MOTION is a TensorFlow model with an eight-slot state formatter, family masks,
normalization, and a named-array checkpoint loader. This repository does not
yet execute that official path. The script intentionally writes no results:
an unavailable comparator is preferable to a fabricated one.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="MOTION benchmark status")
    ap.add_argument("--preflight-only", action="store_true")
    ap.parse_known_args()
    raise SystemExit(
        "MOTION is disabled: its official TensorFlow state formatter, normalization, "
        "and named-array checkpoint loader have not yet been integrated and validated. "
        "No benchmark JSON was written.")


if __name__ == "__main__":
    main()
