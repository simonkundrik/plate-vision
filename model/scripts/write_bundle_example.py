#!/usr/bin/env python
"""Regenerate `shared/bundle.example.json`.

The example is a real manifest built by `platevision.bundle`, not a hand-written sample.
It is the fixture the TypeScript client's tests parse, which is what makes the Python and
TypeScript sides of the boundary testable against one definition rather than each against
its own idea of the other.

Run this after changing the manifest shape. `pytest` fails if the committed file and the
builder disagree.

Usage:
    python scripts/write_bundle_example.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from platevision import bundle, meta

# Fixed rather than `datetime.now()`. A timestamp that moves on every run would make the
# committed example differ from a freshly built one for a reason that means nothing, and
# the test that compares them would have to ignore the only field guaranteed to change.
GENERATED_UTC = "2026-01-01T00:00:00+00:00"

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "shared" / "bundle.example.json"


def example() -> dict:
    """A manifest for a trained classifier and a trained nutrition head.

    Values are representative rather than measured. The example exists to pin the *shape*
    of the manifest across two languages; the numbers in it describe nothing and the file
    says so.
    """
    targets = meta.target_keys()

    provenance = {
        "classifier_checkpoint": "runs/kaggle-baseline/runs/baseline/best.pt",
        "classifier_backbone": "efficientnet_b0",
        "classifier_metric": 86.12,
        "nutrition_source": "runs/nutrition/best.pt",
        "target_transform": {
            "mean": [5.09, 2.32, 2.05, 2.64, 5.10],
            "std": [1.10, 1.23, 1.19, 0.93, 0.82],
            "keys": targets,
        },
    }

    return bundle.build_bundle(
        artifact={
            "name": "plate-vision-fp32.onnx",
            "bytes": 17280512,
            "sha256": "0" * 64,
        },
        heads_trained={"logits": True, "nutrition_quantiles": True},
        provenance=provenance,
        quantization=None,
        generated_utc=GENERATED_UTC,
    )


def main() -> int:
    EXAMPLE_PATH.write_text(json.dumps(example(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {EXAMPLE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
