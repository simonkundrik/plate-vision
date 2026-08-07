#!/usr/bin/env python
"""Build a manifest of CC-licensed food images for out-of-distribution testing.

Writes URLs and attribution, not images. The repository is public, so redistributing the
photos would attach a licensing question to a portfolio piece for no benefit; a manifest
plus a download script is what ImageNet and similar datasets do.

Respects the anonymous Openverse limits measured against the live API: 20 requests per
minute and 200 per day, with page_size capped at 20. One request per class is 101 requests,
which fits.

Usage:
    python data/build_ood_manifest.py --out data/ood/manifest.json
    python data/build_ood_manifest.py --classes 5 --out data/ood/sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

from platevision import food101, openverse
from platevision.download import fetch

DEFAULT_OUT = Path(__file__).resolve().parent / "ood" / "manifest.json"

# 20 per minute means one every three seconds. Sleeping slightly longer than the minimum
# keeps a retry from tipping the window over.
SECONDS_BETWEEN_REQUESTS = 3.2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-class", type=int, default=openverse.MAX_ANONYMOUS_PAGE_SIZE)
    parser.add_argument("--classes", type=int, help="only the first N classes, for a dry run")
    parser.add_argument("--delay", type=float, default=SECONDS_BETWEEN_REQUESTS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the existing manifest and only query classes missing from it",
    )
    args = parser.parse_args(argv)

    class_keys = food101.class_keys()
    if args.classes:
        class_keys = class_keys[: args.classes]

    # Resume matters because of the budget, not convenience. A transient failure on one
    # class out of 101 should cost one request to repair, not another full pass against a
    # 200-request daily allowance.
    existing: list[openverse.OpenverseImage] = []
    if args.resume and args.out.is_file():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        existing = [openverse.OpenverseImage(**row) for row in previous["images"]]
        have = {image.class_key for image in existing}
        missing = [key for key in class_keys if key not in have]
        print(f"resuming: {len(existing):,} images kept, {len(missing)} classes to query")
        class_keys = missing

    budget = len(class_keys)
    print(f"querying {budget} classes, {args.per_class} images each")
    if budget > openverse.SUSTAINED_PER_DAY:
        print(
            f"  warning: {budget} requests exceeds the anonymous daily limit of "
            f"{openverse.SUSTAINED_PER_DAY}"
        )

    collected: list[openverse.OpenverseImage] = list(existing)
    failures: list[str] = []
    all_keys = food101.class_keys()

    for index, class_key in enumerate(class_keys):
        term = openverse.search_term(class_key)
        url = openverse.build_search_url(term, page_size=args.per_class)
        try:
            payload = json.loads(fetch(url))
        except (urllib.error.HTTPError, RuntimeError) as exc:
            failures.append(class_key)
            print(f"  [{index + 1}/{budget}] {class_key}: FAILED {exc}")
            time.sleep(args.delay)
            continue

        # The label is the class's index in the committed ordering, not its position in
        # this loop. On a resume run the loop only covers the missing classes, and using
        # the loop index would relabel every one of them.
        found = openverse.parse_results(payload, class_key, all_keys.index(class_key))
        collected.extend(found)
        print(f"  [{index + 1}/{budget}] {class_key}: {len(found)}")
        time.sleep(args.delay)

    before = len(collected)
    collected = openverse.deduplicate(collected)
    deduped = before - len(collected)

    before = len(collected)
    collected = openverse.drop_cross_labelled(collected)
    cross = before - len(collected)

    counts = openverse.per_class_counts(collected)
    manifest = {
        "schema_version": 1,
        "source": "openverse",
        "generated_utc": datetime.now(UTC).isoformat(),
        "license_filter": openverse.LICENSE_FILTER,
        "label_quality": (
            "weak. Labels come from the search term, not from anyone looking at the image. "
            "Any accuracy computed on this set has a noise floor; estimate it by reviewing "
            "the sample written alongside this manifest before quoting a number."
        ),
        "classes_queried": len(class_keys),
        "images": [image.as_dict() for image in collected],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    review = openverse.sample_for_review(collected)
    review_path = args.out.with_name(args.out.stem + "_review_sample.json")
    review_path.write_text(
        json.dumps([image.as_dict() for image in review], indent=2), encoding="utf-8"
    )

    print(f"\ncollected {len(collected):,} images across {len(counts)} classes")
    print(f"  dropped {deduped:,} duplicate URLs")
    print(f"  dropped {cross:,} appearing under more than one class")
    if counts:
        thinnest = list(counts.items())[:3]
        print(f"  thinnest classes: {thinnest}")
    if failures:
        print(f"  failed classes ({len(failures)}): {failures[:8]}")
    print(f"\n  wrote {args.out}")
    print(f"  wrote {review_path}  <- hand-check these to estimate the label noise rate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
