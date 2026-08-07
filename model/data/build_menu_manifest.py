#!/usr/bin/env python
"""Build the chain-menu manifest: named items, published calories, and CC-licensed photos.

Two sources. USDA FoodData Central supplies a per-item calorie figure for each menu item;
Openverse supplies photos of it. Pairing them gives images with approximate nutrition
ground truth on photos that are not cafeteria trays, which is the gap Nutrition5k leaves.

The calorie figures are approximate and the manifest says so in its own metadata. FNDDS
values are survey estimates rather than the chain's published numbers, and a photo of a
named item may show a partial portion or a combo.

The API key is read from the environment, never from the repository. See
platevision.secrets.

Usage:
    python data/build_menu_manifest.py
    python data/build_menu_manifest.py --items 5 --out data/menu/sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

from platevision import fdc, openverse, secrets
from platevision.download import fetch

DEFAULT_OUT = Path(__file__).resolve().parent / "menu" / "manifest.json"

# FDC with a real key allows far more, but Openverse still caps at 20 requests a minute.
SECONDS_BETWEEN_REQUESTS = 3.2
PHOTOS_PER_ITEM = 12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--items", type=int, help="only the first N menu items, for a dry run")
    parser.add_argument("--photos-per-item", type=int, default=PHOTOS_PER_ITEM)
    parser.add_argument("--delay", type=float, default=SECONDS_BETWEEN_REQUESTS)
    parser.add_argument("--api-key", help="overrides the environment; avoid on shared machines")
    args = parser.parse_args(argv)

    api_key, source = secrets.resolve_fdc_key(args.api_key)
    print(f"FDC key from: {source}")
    if api_key == fdc.DEMO_KEY:
        print(
            f"  warning: the shared demo key is throttled to about "
            f"{fdc.DEMO_KEY_PER_HOUR} requests an hour. A free key from "
            f"fdc.nal.usda.gov/api-key-signup lifts this."
        )

    seeds = fdc.SEED_ITEMS[: args.items] if args.items else fdc.SEED_ITEMS
    print(f"\nresolving calories for {len(seeds)} menu items")

    resolved: list[tuple[str, str, fdc.MenuItem]] = []
    unmatched: list[str] = []

    for index, (brand, item) in enumerate(seeds, 1):
        query = fdc.query_for(brand, item)
        try:
            payload = json.loads(fetch(fdc.build_search_url(query, api_key), retries=2))
        except (urllib.error.HTTPError, RuntimeError) as exc:
            print(
                f"  [{index}/{len(seeds)}] {query[:38]:40} FAILED "
                f"{secrets.redact(str(exc), api_key)[:60]}"
            )
            unmatched.append(query)
            time.sleep(args.delay)
            continue

        record = fdc.parse_search(payload, query, item)
        if record is None:
            # No FNDDS record matched by name. Dropping it is the point: the top Survey
            # result for an unmatched item is a different food with real calories.
            print(f"  [{index}/{len(seeds)}] {query[:38]:40} no matching record")
            unmatched.append(query)
        else:
            resolved.append((brand, item, record))
            print(
                f"  [{index}/{len(seeds)}] {query[:38]:40} {record.kcal_per_item:6.0f} kcal "
                f"({record.gram_weight:.0f}g)  {record.description[:34]}"
            )
        time.sleep(args.delay)

    print(f"\nresolved {len(resolved)} of {len(seeds)} items")
    if not resolved:
        raise SystemExit("no items resolved; nothing to pair photos with")

    print(f"\nsearching photos, {args.photos_per_item} per item")
    entries: list[dict] = []
    for index, (brand, item, record) in enumerate(resolved, 1):
        term = f"{brand} {item}"
        try:
            payload = json.loads(
                fetch(openverse.build_search_url(term, page_size=args.photos_per_item))
            )
        except (urllib.error.HTTPError, RuntimeError) as exc:
            print(f"  [{index}/{len(resolved)}] {term[:38]:40} FAILED {str(exc)[:50]}")
            time.sleep(args.delay)
            continue

        photos = openverse.parse_results(payload, item, index - 1)
        for photo in photos:
            entries.append(
                {
                    **photo.as_dict(),
                    "brand": brand,
                    "item": item,
                    "kcal_per_item": record.kcal_per_item,
                    "gram_weight": record.gram_weight,
                    "fdc_id": record.fdc_id,
                    "fdc_description": record.description,
                }
            )
        print(f"  [{index}/{len(resolved)}] {term[:38]:40} {len(photos)} photos")
        time.sleep(args.delay)

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "calorie_source": "USDA FoodData Central, Survey (FNDDS)",
        "photo_source": "openverse",
        "license_filter": openverse.LICENSE_FILTER,
        "ground_truth_quality": (
            "approximate, in two independent ways. FNDDS figures are survey estimates "
            "rather than the chain's published values: a Big Mac resolves to about 535 "
            "kcal against McDonald's published 563, roughly 5 percent apart, which is a "
            "floor on achievable error here. Separately, a photo of a named item may show "
            "a partial portion, a combo, or the wrong thing, and that error is unbounded. "
            "Review before quoting any figure computed from this set."
        ),
        "items_resolved": len(resolved),
        "items_unmatched": unmatched,
        "images": entries,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    calories = sorted(record.kcal_per_item for _, _, record in resolved)
    print(f"\n{len(entries):,} photos across {len(resolved)} items")
    print(
        f"calories min/median/max: {calories[0]:.0f} / "
        f"{calories[len(calories) // 2]:.0f} / {calories[-1]:.0f}"
    )
    if unmatched:
        print(f"unmatched items ({len(unmatched)}): {unmatched[:6]}")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
