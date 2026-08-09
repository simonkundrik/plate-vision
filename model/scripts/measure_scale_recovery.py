#!/usr/bin/env python
"""How accurate would a plate detector have to be to be worth building?

``measure_scale_dependence.py`` shows what unknown camera distance costs. It does not show
whether that cost is *recoverable*, and the scale-reference route assumes it is: that the
damage is a systematic error in apparent size, correctable once something of known physical
size is found in the frame.

That assumption has two ways to fail, and both are cheaper to check than to discover after
writing a detector:

1. **The error may not be systematic.** If zooming degrades the representation rather than
   rescaling the prediction, no measurement of scale corrects it, however exact.
2. **The correction may need precision no detector can deliver.** A plate is 26 to 28 cm
   depending on the plate, which is a 7 percent uncertainty before any detection error. If
   the benefit collapses at 10 percent scale error, the route is dead on arrival.

So this fits the exponent that best explains the model's behaviour under zoom, then sweeps a
detector's error to find where the benefit stops.

Usage:
    python scripts/measure_scale_recovery.py --checkpoint runs/nutrition/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from platevision import checkpoint, datasets, meta, regression, scale, transforms

# The worst row of the scale-dependence table: anywhere from twice as far to twice as close.
DEFAULT_ZOOM = 2.0

# What a detector might plausibly achieve, from "exact" through the 7 percent a dinner plate
# costs before any detection error at all.
DEFAULT_ERRORS = (0.0, 0.05, 0.10, 0.20, 0.40)


def infer(model, samples, target_transform, zoom, device, workers, seed):
    """Predictions in real units for the whole split, plus the factor each dish was given."""
    dataset = datasets.ZoomedDishes(
        samples,
        transform=transforms.eval_transform(),
        target_transform=target_transform,
        zoom=zoom,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=workers)

    predictions, targets = [], []
    with torch.no_grad():
        for images, batch_targets in loader:
            predictions.append(model(images.to(device)).float().cpu())
            targets.append(batch_targets.float())

    predicted = target_transform.inverse(regression.enforce_monotonic(torch.cat(predictions)))
    actual = target_transform.inverse(torch.cat(targets))
    return predicted, actual, torch.tensor(dataset.factors, dtype=torch.float32)


def score(predicted: torch.Tensor, actual: torch.Tensor, index: int, median: int) -> dict:
    point = predicted[:, index, median]
    truth = actual[:, index]
    return {
        "mae": float((point - truth).abs().mean()),
        "median_ape": float(((point - truth).abs() / truth.clamp(min=1)).median() * 100),
        "coverage": float(
            ((truth >= predicted[:, index, 0]) & (truth <= predicted[:, index, -1])).float().mean()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    model, target_transform, _ = checkpoint.restore_nutrition_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    samples, _ = datasets.build_nutrition5k_index(args.data_root, args.split)
    keys = meta.target_keys()
    energy, mass = keys.index("energy"), keys.index("mass")
    median = meta.quantiles().index(0.5)

    print(f"{len(samples):,} dishes, zoom +/-{args.zoom:.2f}x simulates unknown distance\n")

    fixed, actual, _ = infer(model, samples, target_transform, 1.0, device, args.workers, args.seed)
    zoomed, _, factors = infer(
        model, samples, target_transform, args.zoom, device, args.workers, args.seed
    )

    rig = score(fixed, actual, energy, median)
    handheld = score(zoomed, actual, energy, median)

    def condition(label: str, result: dict) -> str:
        return (
            f"{label:<34}{result['mae']:>11.1f}{result['median_ape']:>9.1f}%"
            f"{result['coverage'] * 100:>8.1f}%"
        )

    print(f"{'condition':<34}{'kcal MAE':>11}{'kcal APE':>10}{'cover':>9}")
    print(condition("fixed rig (the published number)", rig))
    print(condition("unknown distance, uncorrected", handheld))

    # What the model actually keys on. Near zero here would end the route: an error that does
    # not depend on scale cannot be corrected by knowing the scale.
    fitted = {
        "energy": scale.fit_scale_exponent(zoomed[:, energy, median], actual[:, energy], factors),
        "mass": scale.fit_scale_exponent(zoomed[:, mass, median], actual[:, mass], factors),
    }
    print(
        f"\nfitted exponent: energy {fitted['energy']:.2f}, mass {fitted['mass']:.2f}  "
        f"(area would be {scale.AREA_EXPONENT:.0f}, volume {scale.VOLUME_EXPONENT:.0f})"
    )

    candidates = {
        "area (k=2)": scale.AREA_EXPONENT,
        "volume (k=3)": scale.VOLUME_EXPONENT,
        # Fitted on the split it is then applied to, so it is an upper bound on what a
        # correction can do rather than an achievable result. Reported to bound the others.
        f"fitted (k={fitted['energy']:.2f})*": fitted["energy"],
    }

    print(f"\ncorrected, by exponent and detector error{'kcal APE':>16}")
    header = "".join(f"{error * 100:>8.0f}%" for error in DEFAULT_ERRORS)
    print(f"{'exponent':<22}{header}")

    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    for label, exponent in candidates.items():
        cells, row = [], {"exponent_label": label, "exponent": exponent, "results": {}}
        for error in DEFAULT_ERRORS:
            estimated = scale.perturb_factors(factors, error, generator)
            corrected = scale.correct_for_scale(zoomed, estimated, exponent)
            result = score(corrected, actual, energy, median)
            row["results"][f"{error:.2f}"] = result
            cells.append(f"{result['median_ape']:>8.1f}")
        rows.append(row)
        print(f"{label:<22}{''.join(cells)}")

    best = min((row["results"]["0.00"]["median_ape"], row["exponent_label"]) for row in rows)
    recovered = handheld["median_ape"] - best[0]
    lost = handheld["median_ape"] - rig["median_ape"]

    print(
        f"\nA perfect detector recovers {recovered:.1f} of the {lost:.1f} points "
        f"unknown distance costs, using {best[1]}."
    )
    print("* fitted on the split it corrects, so that column is an upper bound, not a result.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "zoom": args.zoom,
                    "fixed_rig": rig,
                    "uncorrected": handheld,
                    "fitted_exponent": fitted,
                    "corrections": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
