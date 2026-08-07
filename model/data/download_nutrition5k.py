#!/usr/bin/env python
"""Fetch the Nutrition5k subset this project actually uses.

The full Nutrition5k archive is 181 GB, almost all of which is rotating side-angle video.
This pulls only the overhead RGB frames plus metadata and splits, which is 1.35 GB.

Depth imagery is skipped on purpose. Nutrition5k ships RGB-D and depth measurably improves
nutrition estimates, but a phone camera has no depth sensor, so training on it would open a
gap between training and deployment that no amount of tuning closes.

Usage:
    python data/download_nutrition5k.py --report          # measure and audit, download nothing
    python data/download_nutrition5k.py                   # fetch into data/nutrition5k/
    python data/download_nutrition5k.py --limit 20        # small slice for a smoke test
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from platevision import download as dl
from platevision import nutrition5k as n5k

DEFAULT_OUT = Path(__file__).resolve().parent / "nutrition5k"


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:,.2f} {unit}"
        num_bytes /= 1024
    raise AssertionError("unreachable")


def load_metadata() -> dict[str, n5k.Dish]:
    dishes: dict[str, n5k.Dish] = {}
    for name in n5k.METADATA_FILES:
        text = dl.fetch(n5k.metadata_url(name)).decode("utf-8")
        parsed = n5k.parse_dish_metadata(text)
        overlap = dishes.keys() & parsed.keys()
        if overlap:
            print(f"  note: {len(overlap)} dish ids appear in more than one cafe file")
        dishes.update(parsed)
        print(f"  {name}: {len(parsed):,} dishes")
    return dishes


def load_splits() -> dict[str, list[str]]:
    splits = {}
    for name in n5k.SPLIT_FILES:
        ids = n5k.parse_split_ids(dl.fetch(n5k.split_url(name)).decode("utf-8"))
        splits[name] = ids
        print(f"  {name}: {len(ids):,} ids")
    return splits


def report() -> int:
    """Measure the subset and audit label quality without downloading imagery."""
    print("Metadata")
    dishes = load_metadata()
    print(f"  total unique dishes: {len(dishes):,}")

    print("\nSplits")
    splits = load_splits()
    split_ids = {i for ids in splits.values() for i in ids}
    print(f"  unique ids across splits: {len(split_ids):,}")

    train = set(splits["rgb_train_ids.txt"])
    test = set(splits["rgb_test_ids.txt"])
    leak = train & test
    print(f"  train/test overlap: {len(leak)}" + ("  <-- LEAKAGE" if leak else "  (clean)"))

    missing_meta = sorted(split_ids - dishes.keys())
    print(f"  split ids with no metadata row: {len(missing_meta):,}")

    print("\nImagery (overhead)")
    rgb_bytes = rgb_count = other_bytes = other_count = 0
    have_rgb: set[str] = set()
    for name, size in dl.iter_overhead_objects():
        dish_id = dl.dish_id_from_object(name)
        if dish_id:
            rgb_bytes += size
            rgb_count += 1
            have_rgb.add(dish_id)
        else:
            other_bytes += size
            other_count += 1
    print(f"  rgb.png:        {rgb_count:,} files, {human(rgb_bytes)}  <-- fetched")
    print(f"  depth_*.png:    {other_count:,} files, {human(other_bytes)}  <-- skipped")
    print(f"  mean rgb size:  {human(rgb_bytes / max(rgb_count, 1))}")

    usable = split_ids & dishes.keys() & have_rgb
    print(f"\n  usable dishes (split + metadata + rgb): {len(usable):,}")
    print(f"  split ids with no rgb frame:            {len(split_ids - have_rgb):,}")
    for name, ids in splits.items():
        kept = set(ids) & dishes.keys() & have_rgb
        pct = len(kept) / max(len(ids), 1)
        print(f"    {name:<20} {len(kept):>5,} usable of {len(ids):>5,}  ({pct:.0%})")

    print("\nLabel quality")
    audit(dishes, usable)

    print(f"\nDownload footprint: {human(rgb_bytes)} imagery plus a few MB of metadata.")
    return 0


def audit(dishes: dict[str, n5k.Dish], usable: set[str]) -> None:
    """Check stated totals against the per-ingredient sums.

    These are real labels from a real dataset, so some of them are wrong. Knowing how many
    before training beats discovering it while staring at a bad loss curve.
    """
    subset = [dishes[d] for d in sorted(usable)]
    if not subset:
        print("  no usable dishes to audit")
        return

    nonpositive = [d for d in subset if d.calories <= 0]
    no_ingredients = [d for d in subset if not d.ingredients]
    disagree_1pct = [d for d in subset if d.calorie_disagreement() > 0.01]
    disagree_10pct = [d for d in subset if d.calorie_disagreement() > 0.10]

    cals = sorted(d.calories for d in subset)
    mid = cals[len(cals) // 2]
    print(f"  dishes audited:                    {len(subset):,}")
    print(f"  calories <= 0:                     {len(nonpositive):,}")
    print(f"  no ingredients listed:             {len(no_ingredients):,}")
    print(f"  total vs ingredient sum >1% off:   {len(disagree_1pct):,}")
    print(f"  total vs ingredient sum >10% off:  {len(disagree_10pct):,}")
    print(f"  calories min/median/max:           {cals[0]:.0f} / {mid:.0f} / {cals[-1]:.0f}")


def download(out: Path, limit: int | None, workers: int, force: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(exist_ok=True)
    (out / "splits").mkdir(exist_ok=True)
    imagery = out / "imagery"
    imagery.mkdir(exist_ok=True)

    print("Metadata and splits")
    for name in n5k.METADATA_FILES:
        (out / "metadata" / name).write_bytes(dl.fetch(n5k.metadata_url(name)))
        print(f"  {name}")
    requested: list[str] = []
    for name in n5k.SPLIT_FILES:
        raw = dl.fetch(n5k.split_url(name))
        (out / "splits" / name).write_bytes(raw)
        requested.extend(n5k.parse_split_ids(raw.decode("utf-8")))
        print(f"  {name}")

    # The split files list ids the overhead camera never captured. Asking for those is a
    # guaranteed 404 each, so the bucket is listed once up front and the request set is
    # intersected with what exists rather than discovering absence one request at a time.
    print("\nChecking which dishes have an overhead frame")
    available = dl.available_dish_ids()
    requested_unique = sorted(set(requested))
    ids = [d for d in requested_unique if d in available]
    print(f"  ids in splits:               {len(requested_unique):,}")
    print(f"  with an overhead frame:      {len(ids):,}")
    print(f"  skipped, no overhead frame:  {len(requested_unique) - len(ids):,}")

    if limit:
        ids = ids[:limit]
    print(f"\nImagery: {len(ids):,} dishes into {imagery}")

    pending = [d for d in ids if force or not (imagery / d / "rgb.png").exists()]
    print(f"  already present: {len(ids) - len(pending):,}, to fetch: {len(pending):,}")

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, imagery, d): d for d in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            dish_id = futures[fut]
            try:
                fut.result()
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAILED {dish_id}: {exc}")
            if i % 250 == 0 or i == len(pending):
                print(f"  {i:,}/{len(pending):,}")

    print(f"\nDone. fetched {ok:,}, failed {failed:,}")
    return 1 if failed else 0


def _fetch_one(imagery: Path, dish_id: str) -> None:
    # Fetch before creating anything on disk, so a failure leaves no empty directory
    # behind. Write through a .part name so an interrupted run cannot leave a truncated
    # PNG that a later resume would treat as already downloaded.
    data = dl.fetch(n5k.rgb_url(dish_id))
    dest = imagery / dish_id / "rgb.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".png.part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", action="store_true", help="measure and audit, download nothing")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, help="only fetch the first N dishes")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="re-fetch files already on disk")
    args = parser.parse_args(argv)

    if args.report:
        return report()
    return download(args.out, args.limit, args.workers, args.force)


if __name__ == "__main__":
    sys.exit(main())
