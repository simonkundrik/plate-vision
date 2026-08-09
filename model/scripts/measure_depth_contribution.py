#!/usr/bin/env python
"""Did the depth channel do anything?

Phase H3 fed depth to the network as a fourth input channel, following the Nutrition5k
paper, which reports 70.6 to 47.6 kcal MAE from that change. Here it went the other way:
54.7 to 57.6 against phase D, whose configuration is identical but for `--depth`.

"Depth made it worse" has two very different explanations, and they call for opposite next
steps:

  1. the model used depth and still lost, meaning the channel costs more than it pays, or
  2. the model learned to ignore the fourth channel, meaning nothing was measured about
     depth at all and the 2.9 kcal is the price of a wider stem.

Replacing the depth plane and seeing whether anything moves separates them. Two
replacements, because they fail differently:

  ``mean``      a constant, so the channel carries no information whatsoever
  ``shuffled``  another dish's depth, which keeps the channel's statistics and destroys
                only the pairing, so a model reading typical values rather than this
                dish's values scores the same as it did on real depth

Usage:
    python scripts/measure_depth_contribution.py \\
        --checkpoint runs/kaggle-nutrition-h3/best.pt --data-root data/nutrition5k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from platevision import checkpoint, datasets, evaluation, meta, models, transforms
from platevision.quantile import enforce_monotonic

MODES = ("real", "mean", "shuffled")


def measure(ckpt: Path, data_root: Path, mode: str, split: str, batch_size: int, workers: int):
    device = models.resolve_device(None)
    model, target_transform, payload = checkpoint.restore_nutrition_model(ckpt)
    model.to(device).eval()

    channels = checkpoint.stem_channels(payload["model"])
    if channels != 4:
        raise SystemExit(
            f"{ckpt} takes {channels} channels; this only means anything for a depth model"
        )

    samples, _ = datasets.build_nutrition5k_index(data_root, split, require_depth=True)
    dataset = datasets.Nutrition5kDataset(
        samples,
        transform=transforms.eval_transform(channels=4),
        target_transform=target_transform,
        with_depth=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    predictions, targets = [], []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            if mode == "mean":
                # Zero in normalised space is the value training standardised to, so the
                # channel reads as "nothing unusual". A raw zero would instead say the
                # surface is touching the lens, which is a strong and wrong signal.
                images[:, 3, :, :] = 0.0
            elif mode == "shuffled":
                images[:, 3, :, :] = images[:, 3, :, :].flip(0)
            predictions.append(model(images).float().cpu())
            targets.append(target)

    # The model predicts in standardised log space. Skipping the inverse gives an MAE of
    # about 0.26 "kcal", which is the tell that the numbers are unitless.
    return evaluation.build_report(
        target_transform.inverse(enforce_monotonic(torch.cat(predictions))),
        target_transform.inverse(torch.cat(targets).float()),
        target_keys=meta.target_keys(),
        quantiles=meta.quantiles(),
        dish_ids=[s.dish_id for s in samples],
        crossing_rate=0.0,
        top_n=1,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)

    results = {}
    for mode in MODES:
        report = measure(
            args.checkpoint, args.data_root, mode, args.split, args.batch_size, args.workers
        )
        results[mode] = (report.mae["energy"], report.median_ape["energy"] * 100)
        print(f"{mode:<10} MAE {results[mode][0]:7.2f} kcal   median APE {results[mode][1]:5.2f}%")

    cost = results["mean"][0] - results["real"][0]
    print(f"\nreal depth is worth {cost:.2f} kcal against a constant channel")
    if cost < 2.0:
        print("  which is close to nothing; the model is not reading this channel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
