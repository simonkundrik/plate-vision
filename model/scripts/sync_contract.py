#!/usr/bin/env python
"""Copy the shared contract into the package so an installed copy carries it.

``shared/`` sits above this package in the repository. That is fine from a checkout and
wrong once installed: nothing above ``site-packages/platevision`` is a repository, so the
contract simply is not there and every function that touches it raises FileNotFoundError.

A published package cannot reference files outside itself, so the contract has to be copied
in. The copies are committed and a test asserts they still match ``shared/``, which is what
stops the two from drifting. Drift here does not crash anything: a moved label ordering
renames every prediction.

Run after editing anything in shared/. `pytest` fails if the copies are stale.

Usage:
    python scripts/sync_contract.py
    python scripts/sync_contract.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
SHARED_DIR = MODEL_DIR.parent / "shared"
PACKAGED_DIR = MODEL_DIR / "platevision" / "_data"

CONTRACT_FILES = ("model_meta.json", "food101_labels.json")


def rendered(name: str) -> str:
    """Reserialised rather than copied verbatim, so line endings cannot fail the check."""
    source = json.loads((SHARED_DIR / name).read_text(encoding="utf-8"))
    return json.dumps(source, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if a copy is stale")
    args = parser.parse_args(argv)

    PACKAGED_DIR.mkdir(parents=True, exist_ok=True)
    stale = []

    for name in CONTRACT_FILES:
        target = PACKAGED_DIR / name
        expected = rendered(name)

        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != expected:
                stale.append(name)
            continue

        target.write_text(expected, encoding="utf-8")
        print(f"synced {name}")

    if stale:
        print(f"stale: {', '.join(stale)}. Run python scripts/sync_contract.py", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
