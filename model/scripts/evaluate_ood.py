#!/usr/bin/env python
"""Measure classification accuracy on the out-of-distribution set.

Food-101's own test split is drawn from the same curated pool as its training data. This
answers a different and more useful question: what happens on photos the model has no
reason to find easy, taken by strangers with ordinary cameras.

The gap between the two numbers is the result. A model that scores 86 percent on its own
test split and far less here has learned the dataset as much as the task, and that is worth
knowing before shipping it to a phone.

Labels on this set are weak, derived from search terms rather than inspection, so accuracy
here has a noise floor. Pass --noise-rate with the figure from reviewing the sample
manifest to get a corrected estimate alongside the raw one.

Usage:
    python scripts/evaluate_ood.py --checkpoint runs/kaggle-baseline/runs/baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from platevision import checkpoint, datasets, food101, models, transforms

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "ood" / "manifest.json"
DEFAULT_IMAGES = Path(__file__).resolve().parents[1] / "data" / "ood" / "images"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--noise-rate",
        type=float,
        help="fraction of labels believed wrong, from reviewing the sample manifest",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    device = models.resolve_device(args.device)
    model, payload = checkpoint.restore_classifier(args.checkpoint)
    model.to(device).eval()
    print(f"checkpoint: {args.checkpoint}")
    print(
        f"  backbone {checkpoint.backbone_of(payload)}, epoch {payload['epoch']}, "
        f"reported {payload['best_metric']:.2f}% on the Food-101 test split"
    )

    samples, missing = datasets.build_ood_index(args.manifest, args.images)
    if args.limit:
        samples = samples[: args.limit]
    print(f"\nood set: {len(samples):,} images, {missing:,} manifest entries not on disk")

    loader = DataLoader(
        datasets.Food101Dataset(samples, transform=transforms.eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    correct1 = correct5 = total = 0
    per_class = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            outputs = model(images.to(device)).cpu()
            top5 = outputs.topk(5, dim=1).indices
            hits1 = top5[:, 0] == labels
            hits5 = (top5 == labels.view(-1, 1)).any(dim=1)

            correct1 += int(hits1.sum())
            correct5 += int(hits5.sum())
            total += labels.numel()
            for label, hit in zip(labels.tolist(), hits1.tolist(), strict=True):
                per_class[label][0] += int(hit)
                per_class[label][1] += 1

            if index % 20 == 0:
                print(f"  batch {index}  running top1 {100 * correct1 / total:.2f}%", flush=True)

    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    reported = payload["best_metric"]

    print(f"\nout-of-distribution top-1: {top1:.2f}%")
    print(f"out-of-distribution top-5: {top5:.2f}%")
    print(f"Food-101 test split top-1: {reported:.2f}%")
    print(f"drop: {reported - top1:.2f} points")

    if args.noise_rate:
        # Wrong labels cap achievable accuracy. Dividing by the fraction believed correct
        # gives a rough upper bound on what the model could have scored on clean labels.
        corrected = top1 / (1.0 - args.noise_rate)
        print(
            f"\nwith {args.noise_rate:.0%} label noise, corrected top-1 is about "
            f"{min(corrected, 100.0):.2f}%"
        )

    keys = food101.class_keys()
    ranked = sorted(per_class.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
    print("\nweakest classes")
    for label, (hits, count) in ranked[:8]:
        print(f"  {keys[label]:<24} {hits:>3}/{count:<3} {100 * hits / max(count, 1):5.1f}%")
    print("\nstrongest classes")
    for label, (hits, count) in ranked[-5:]:
        print(f"  {keys[label]:<24} {hits:>3}/{count:<3} {100 * hits / max(count, 1):5.1f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "images": total,
                    "ood_top1": top1,
                    "ood_top5": top5,
                    "food101_test_top1": reported,
                    "drop_points": reported - top1,
                    "label_quality": "weak, search-derived",
                    "per_class": {keys[k]: v for k, v in per_class.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
