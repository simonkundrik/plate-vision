#!/usr/bin/env python
"""Evaluate a nutrition checkpoint and write the full report.

Produces the numbers the model card is built from: error in real units, calibration of the
predicted intervals, error broken down by portion size, and the specific dishes the model
gets most wrong.

Usage:
    python scripts/evaluate_nutrition.py --checkpoint runs/nutrition/best.pt \\
        --data-root data/nutrition5k --out runs/nutrition/report
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from platevision import (
    checkpoint,
    datasets,
    evaluation,
    meta,
    models,
    regression,
    routing,
    transforms,
)
from platevision.quantile import enforce_monotonic


def load_availability(path: Path | None, dish_ids: list[str]) -> list[set[str]]:
    """Which routes had the evidence they needed, per dish.

    Nutrition5k is cafeteria trays with no barcodes, no restaurant, and no reference object,
    so with no file every dish falls to the vision route. That is the honest default and it
    is worth seeing: it says the near-exact routes are measured on nothing.
    """
    if path is None:
        return [set() for _ in dish_ids]

    table = json.loads(path.read_text(encoding="utf-8"))
    return [set(table.get(dish_id, [])) for dish_id in dish_ids]


def collect(model, loader, device, target_transform):
    """Run the model over a split and return real-unit predictions, targets, and ids."""
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for images, batch_targets in loader:
            predictions.append(model(images.to(device)).float().cpu())
            targets.append(batch_targets.float().cpu())

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    crossing_rate = regression.crossings(predictions)
    predictions = enforce_monotonic(predictions)

    return (
        target_transform.inverse(predictions),
        target_transform.inverse(targets),
        crossing_rate,
    )


def plot(report, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = report.target_keys[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    nominal = report.quantiles
    empirical = report.calibration[key]
    ax1.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax1.plot(nominal, empirical, "o-", label="model")
    ax1.set_xlabel("nominal quantile")
    ax1.set_ylabel("empirical fraction below")
    ax1.set_title(f"Calibration ({key})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    centres = [(b.lower + b.upper) / 2 for b in report.buckets]
    ax2.bar(range(len(centres)), [b.mae for b in report.buckets])
    ax2.set_xticks(range(len(centres)))
    ax2.set_xticklabels([f"{b.lower:.0f}\n{b.upper:.0f}" for b in report.buckets], fontsize=8)
    ax2.set_xlabel("true kcal range")
    ax2.set_ylabel("MAE (kcal)")
    ax2.set_title("Error by portion size")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out / "report.png", dpi=140)
    print(f"  wrote {out / 'report.png'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/nutrition/report"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--routes",
        type=Path,
        help="json mapping dish id to the routes with evidence for it; "
        "without it every dish is evaluated on the vision route alone",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    device = models.resolve_device(args.device)
    model, target_transform, payload = checkpoint.restore_nutrition_model(args.checkpoint)
    model.to(device)
    print(f"checkpoint: {args.checkpoint}  (epoch {payload['epoch']}, {payload['backbone']})")

    samples, stats = datasets.build_nutrition5k_index(args.data_root, args.split)
    if args.limit:
        samples = samples[: args.limit]
    print(f"{args.split}: {len(samples):,} dishes of {stats.listed:,} listed")
    dish_ids = [s.dish_id for s in samples]

    dataset = datasets.Nutrition5kDataset(
        samples, transform=transforms.eval_transform(), target_transform=target_transform
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    predictions, targets, crossing_rate = collect(model, loader, device, target_transform)

    report = evaluation.build_report(
        predictions,
        targets,
        target_keys=meta.target_keys(),
        quantiles=meta.quantiles(),
        dish_ids=dish_ids,
        crossing_rate=crossing_rate,
        top_n=args.top_n,
    )

    print(f"\n{report.headline()}")
    print(f"  median AE          {report.median_ae['energy']:.1f} kcal")
    print(f"  mean interval      {report.mean_interval_width['energy']:.1f} kcal wide")
    print(f"  quantile crossings {report.crossing_rate * 100:.2f}%")

    print("\ncalibration (nominal -> empirical)")
    for nominal, empirical in zip(report.quantiles, report.calibration["energy"], strict=True):
        print(f"  {nominal:.2f} -> {empirical:.3f}")

    print("\nerror by portion size")
    for bucket in report.buckets:
        print(
            f"  {bucket.lower:7.0f} to {bucket.upper:7.0f} kcal  "
            f"n={bucket.count:4d}  MAE {bucket.mae:7.1f}  coverage {bucket.coverage * 100:5.1f}%"
        )

    print(f"\nworst {len(report.worst)} predictions")
    for case in report.worst:
        flag = "inside" if case.inside_interval else "OUTSIDE"
        print(
            f"  {case.dish_id:<20} truth {case.truth:7.0f}  predicted {case.predicted:7.0f}  "
            f"[{case.lower:6.0f}, {case.upper:7.0f}] {flag}"
        )

    availability = load_availability(args.routes, dish_ids)
    routed = routing.build_routed_report(
        predictions,
        targets,
        routing.assign_routes(availability),
        target_keys=meta.target_keys(),
        quantiles=meta.quantiles(),
        dish_ids=dish_ids,
        availability=availability,
        crossing_rate=crossing_rate,
        top_n=args.top_n,
    )

    print()
    for line in routed.headline():
        print(line)
    if len(routed.routes) == 1 and routed.routes[0].route == routing.FALLBACK_ROUTE:
        # Worth saying out loud rather than leaving to be inferred from a one-row table. The
        # near-exact routes are shipped and unmeasured, which is not the same as accurate.
        print("\n  every dish took the vision route; no other route is measured by this split")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    (args.out / "routed.json").write_text(json.dumps(routed.as_dict(), indent=2), encoding="utf-8")
    print(f"\n  wrote {args.out / 'report.json'}")
    print(f"  wrote {args.out / 'routed.json'}")

    if not args.no_plot:
        plot(report, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
