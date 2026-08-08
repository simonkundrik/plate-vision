#!/usr/bin/env python
"""How much of this model's accuracy is the capture rig handing it for free?

Nutrition5k is shot from a fixed overhead camera at constant height, so apparent size in
pixels maps directly to real size and the model can read scale straight off the image. A
phone held at an unknown distance destroys that mapping.

Zooming the test images simulates the loss and measures what survives. It is the closest
thing to a deployment number obtainable without weighed photographs, and it is the reason
this project should not quote its Nutrition5k figures as though they describe a user's
dinner.

Usage:
    python scripts/measure_scale_dependence.py --checkpoint runs/nutrition/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from platevision import checkpoint, datasets, meta, regression, transforms

# Roughly the range a handheld phone spans: arm's length down to leaning over the plate.
DEFAULT_ZOOMS = (1.0, 1.15, 1.3, 1.6, 2.0)


class ZoomedDishes(Dataset):
    """Centre-zooms each image by a fixed random factor before the usual eval transform."""

    def __init__(self, samples, transform, target_transform, zoom: float, seed: int = 0):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        # Drawn once per dish rather than per epoch, so every zoom level sees the same
        # assignment and the comparison is not confounded by which dish got which factor.
        rng = np.random.default_rng(seed)
        self.factors = rng.uniform(1 / zoom, zoom, size=len(samples)) if zoom > 1 else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")

        if self.factors is not None:
            factor = float(self.factors[index])
            width, height = image.size
            side = min(width, height) / factor
            left, top = (width - side) / 2, (height - side) / 2
            image = image.crop((left, top, left + side, top + side))

        targets = torch.tensor(sample.targets, dtype=torch.float32)
        if self.target_transform is not None:
            targets = self.target_transform.forward(targets)
        return self.transform(image), targets


def median_ape(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    return float(((predicted - actual).abs() / actual.clamp(min=1)).median() * 100)


def evaluate(model, samples, transform, target_transform, zoom, device, workers):
    loader = DataLoader(
        ZoomedDishes(samples, transform, target_transform, zoom),
        batch_size=32,
        num_workers=workers,
    )

    predictions, targets = [], []
    with torch.no_grad():
        for images, batch_targets in loader:
            predictions.append(model(images.to(device)).float().cpu())
            targets.append(batch_targets.float())

    predicted = target_transform.inverse(regression.enforce_monotonic(torch.cat(predictions)))
    actual = target_transform.inverse(torch.cat(targets))

    keys = meta.target_keys()
    energy, mass = keys.index("energy"), keys.index("mass")
    median = meta.quantiles().index(0.5)

    return {
        "zoom": zoom,
        "mass_mae": float((predicted[:, mass, median] - actual[:, mass]).abs().mean()),
        "mass_median_ape": median_ape(predicted[:, mass, median], actual[:, mass]),
        "energy_mae": float((predicted[:, energy, median] - actual[:, energy]).abs().mean()),
        "energy_median_ape": median_ape(predicted[:, energy, median], actual[:, energy]),
        "coverage": float(
            (
                (actual[:, energy] >= predicted[:, energy, 0])
                & (actual[:, energy] <= predicted[:, energy, -1])
            )
            .float()
            .mean()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    model, target_transform, _ = checkpoint.restore_nutrition_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    samples, _ = datasets.build_nutrition5k_index(args.data_root, args.split)
    transform = transforms.eval_transform()

    print(f"{len(samples):,} dishes, zoom simulates unknown camera distance\n")
    print(
        f"{'zoom':>8}{'mass MAE':>11}{'mass APE':>10}{'kcal MAE':>11}{'kcal APE':>10}{'cover':>9}"
    )

    rows = []
    for zoom in DEFAULT_ZOOMS:
        row = evaluate(model, samples, transform, target_transform, zoom, device, args.workers)
        rows.append(row)
        label = "none" if zoom == 1.0 else f"{zoom:.2f}x"
        print(
            f"{label:>8}{row['mass_mae']:>11.1f}{row['mass_median_ape']:>9.1f}%"
            f"{row['energy_mae']:>11.1f}{row['energy_median_ape']:>9.1f}%"
            f"{row['coverage'] * 100:>8.1f}%"
        )

    baseline, worst = rows[0], rows[-1]
    cost = worst["energy_median_ape"] - baseline["energy_median_ape"]
    print(
        f"\nUnknown camera distance costs {cost:.1f} points of calorie error "
        f"({baseline['energy_median_ape']:.1f}% to {worst['energy_median_ape']:.1f}%)."
    )
    print("That gap is the rig, not the model. A phone does not come with a fixed tripod.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
