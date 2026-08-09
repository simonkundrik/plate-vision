#!/usr/bin/env python
"""Fit conformal offsets for a checkpoint, without retraining it.

``train_nutrition.py`` fits these at the end of a run, so a checkpoint trained before that
code was corrected carries offsets fitted the wrong way. Refitting is a forward pass over one
split, and retraining a model to repair its calibration file would be absurd.

**Calibration comes from a slice of the split the intervals will be used on.** A conformal
guarantee needs exchangeability, not merely data the model was not fitted to. Offsets fitted
on held-out *training* data came out near zero on this model, energy +/-1.0 kcal, and moved
test coverage from 82.2% to 83.4% while claiming 90%. That is the failure this script exists
to avoid repeating, and it is why the calibration slice is carved out of the evaluation split
and then excluded from the reported numbers.

Coverage is reported on the remainder, which is the only number that means anything: scoring
a model on the data used to calibrate its intervals is the same circularity in a new place.

Usage:
    python scripts/fit_conformal.py --checkpoint runs/nutrition/best.pt \\
        --data-root data/nutrition5k --out runs/nutrition/conformal.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from platevision import checkpoint, conformal, datasets, meta, regression, transforms


def predict(model, samples, target_transform, device, batch_size, workers):
    dataset = datasets.Nutrition5kDataset(
        samples, transform=transforms.eval_transform(), target_transform=target_transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    predictions, targets = [], []
    with torch.no_grad():
        for images, batch_targets in loader:
            predictions.append(model(images.to(device)).float().cpu())
            targets.append(batch_targets.float().cpu())

    # Monotonic first: an inverted interval makes a conformity score meaningless, because the
    # miss would be measured against the wrong bound.
    stacked = regression.enforce_monotonic(torch.cat(predictions))
    return target_transform.inverse(stacked), target_transform.inverse(torch.cat(targets))


def coverage(predictions: torch.Tensor, targets: torch.Tensor, index: int) -> float:
    inside = (targets[:, index] >= predictions[:, index, 0]) & (
        targets[:, index] <= predictions[:, index, -1]
    )
    return float(inside.float().mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
        help="share of the split used to fit; the rest reports coverage",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if not 0 < args.calibration_fraction < 1:
        raise SystemExit("--calibration-fraction must leave data on both sides of the split")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, target_transform, payload = checkpoint.restore_nutrition_model(args.checkpoint)
    model.to(device).eval()
    print(f"checkpoint: {args.checkpoint} (epoch {payload['epoch']}, {payload['backbone']})")

    samples, _ = datasets.build_nutrition5k_index(args.data_root, args.split)
    shuffled = list(samples)
    random.Random(args.seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * args.calibration_fraction))
    calibration_samples, holdout_samples = shuffled[:cut], shuffled[cut:]
    print(
        f"{args.split}: {len(calibration_samples):,} dishes to calibrate on, "
        f"{len(holdout_samples):,} held back to measure the result"
    )

    keys = meta.target_keys()
    energy = keys.index("energy")

    predicted, actual = predict(
        model, calibration_samples, target_transform, device, args.batch_size, args.workers
    )
    calibration = conformal.ConformalCalibration.fit(predicted, actual, keys, alpha=args.alpha)

    print(f"\noffsets at alpha {args.alpha:.2f}")
    for key, offset in zip(calibration.keys, calibration.offsets, strict=True):
        print(f"  {key:<14} widen by +/- {offset:.1f}")

    held_predicted, held_actual = predict(
        model, holdout_samples, target_transform, device, args.batch_size, args.workers
    )
    before = coverage(held_predicted, held_actual, energy)
    after = coverage(calibration.apply(held_predicted), held_actual, energy)

    print(f"\nenergy coverage on the {len(holdout_samples):,} dishes not used to calibrate")
    print(f"  raw            {before * 100:.1f}%")
    print(f"  conformalised  {after * 100:.1f}%   (target {(1 - args.alpha) * 100:.0f}%)")

    # Loud, because shipping an offset that does not deliver its stated level is exactly the
    # failure that produced this script. Two points of slack for a finite holdout.
    target = 1 - args.alpha
    if after < target - 0.02:
        print(
            f"\n  WARNING: still {(target - after) * 100:.1f} points short of {target * 100:.0f}%. "
            "Check that the calibration slice is exchangeable with the split it is used on."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload_out = calibration.to_dict()
    payload_out["holdout_coverage"] = {"energy_raw": before, "energy_conformalised": after}
    args.out.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
