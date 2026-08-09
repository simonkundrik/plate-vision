#!/usr/bin/env python
"""Emit the golden fixture the TypeScript depth mirror is checked against.

`packages/client/src/depth.ts` is a hand-written mirror of `platevision/depth.py`, because
the app has to normalise a depth map the same way training did and the two live in different
languages. A mirror checked only against its own tests can be wrong in exactly the same way
twice, and the failure it produces is a model quietly fed inputs it was never trained on.

So the expected values come from the real implementation. This writes them next to the
TypeScript test, which reads them back and asserts agreement.

Usage:
    python scripts/emit_depth_fixture.py
    python scripts/emit_depth_fixture.py --check    # fail if the committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from platevision import depth as d

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "client"
    / "src"
    / "__tests__"
    / "depth-fixture.json"
)

# Fixed, so regenerating on another machine produces the same file rather than a diff.
SEED = 20260809


def cases() -> list[dict]:
    rng = np.random.default_rng(SEED)
    maps: list[tuple[str, np.ndarray]] = [
        # A rig-like frame: tight around 3.5 m with a sixth of the pixels dropped out.
        ("rig", np.where(rng.random((8, 8)) < 0.16, 0, rng.integers(3400, 4100, (8, 8)))),
        # A phone-like frame: 300 mm, with food standing proud of the table.
        ("phone", np.where(rng.random((8, 8)) < 0.05, 0, rng.integers(255, 320, (8, 8)))),
        # Saturation and dropout together, the same failure at either end.
        ("broken", np.array([[0, 65535, 3000, 3200], [0, 0, 3100, 65535]])),
        # No valid pixel anywhere, which is the division by zero this code exists to avoid.
        ("empty", np.zeros((2, 3))),
    ]

    out = []
    for name, raw in maps:
        raw = raw.astype(np.uint16)
        out.append(
            {
                "name": name,
                "width": int(raw.shape[1]),
                "height": int(raw.shape[0]),
                "data": [int(v) for v in raw.ravel()],
                "normalised": [round(float(v), 9) for v in d.normalise_depth(raw).ravel()],
                "heightAbove": [round(float(v), 9) for v in d.height_above(raw).ravel()],
                "dropoutFraction": round(float(d.dropout_fraction(raw)), 9),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail rather than rewrite")
    args = parser.parse_args(argv)

    text = json.dumps(cases(), indent=2) + "\n"

    if args.check:
        if not FIXTURE.exists():
            print(f"{FIXTURE} is missing; run this without --check")
            return 1
        if FIXTURE.read_text(encoding="utf-8") != text:
            print(f"{FIXTURE} is stale; run this without --check and commit the result")
            return 1
        print(f"{FIXTURE.name} is current")
        return 0

    FIXTURE.write_text(text, encoding="utf-8")
    print(f"wrote {len(cases())} cases to {FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
