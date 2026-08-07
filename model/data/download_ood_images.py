#!/usr/bin/env python
"""Download the images a manifest points at.

Kept separate from manifest building so the set is reproducible from the tracked JSON
without re-querying the API, and so a failed download run does not cost part of the
200-request daily search budget.

Images land in data/ood/images/<class_key>/ and are gitignored. The manifest and its
attribution are what the repository carries.

Usage:
    python data/download_ood_images.py
    python data/download_ood_images.py --limit 50 --workers 8
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from platevision.download import fetch

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "ood" / "manifest.json"
DEFAULT_IMAGES = Path(__file__).resolve().parent / "ood" / "images"

# Below this a file is a placeholder, an error page, or a thumbnail too small to evaluate.
MIN_BYTES = 4096


def is_decodable(data: bytes) -> bool:
    """Whether the bytes are an image PIL can actually open.

    A size check is not enough. Some hosts answer with an SVG, an HTML error page, or a
    truncated file, all of which land on disk with a .jpg name and a plausible length. The
    failure then surfaces mid-evaluation as an UnidentifiedImageError several hundred
    images in, which is a slow and confusing way to learn about it.
    """
    from PIL import Image

    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:  # noqa: BLE001 - any decode failure disqualifies the file
        return False
    return True


def prune_unreadable(images_root: Path) -> int:
    """Delete already-downloaded files that cannot be decoded."""
    from PIL import Image

    removed = 0
    for path in images_root.rglob("*.jpg"):
        try:
            Image.open(path).verify()
        except Exception:  # noqa: BLE001
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def destination_for(images_root: Path, entry: dict) -> Path:
    stem = entry["identifier"] or str(abs(hash(entry["url"])))
    return images_root / entry["class_key"] / f"{stem}.jpg"


def fetch_one(images_root: Path, entry: dict) -> tuple[str, str | None]:
    """Return (url, error). Errors are collected rather than raised.

    Link rot is expected on a set assembled from third-party hosts. One dead photo should
    not end a run over two thousand images.
    """
    destination = destination_for(images_root, entry)
    try:
        data = fetch(entry["url"], retries=2)
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as exc:
        return entry["url"], str(exc)[:80]

    if len(data) < MIN_BYTES:
        return entry["url"], f"too small ({len(data)} bytes)"
    if not is_decodable(data):
        return entry["url"], "not a decodable image"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jpg.part")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return entry["url"], None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, help="only the first N entries, for a smoke run")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prune", action="store_true", help="delete undecodable files already on disk first"
    )
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}. Run build_ood_manifest.py first.")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["images"]
    if args.limit:
        entries = entries[: args.limit]

    print(f"manifest: {len(manifest['images']):,} images, {manifest['classes_queried']} classes")
    print(f"label quality: {manifest['label_quality'][:70]}...")

    if args.prune and args.out.is_dir():
        removed = prune_unreadable(args.out)
        print(f"\npruned {removed:,} undecodable files already on disk")

    pending = [
        entry for entry in entries if args.force or not destination_for(args.out, entry).exists()
    ]
    print(f"\nalready present: {len(entries) - len(pending):,}, to fetch: {len(pending):,}")

    downloaded = 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, args.out, entry) for entry in pending]
        for index, future in enumerate(as_completed(futures), 1):
            url, error = future.result()
            if error:
                failures.append((url, error))
            else:
                downloaded += 1
            if index % 200 == 0 or index == len(pending):
                print(f"  {index:,}/{len(pending):,}  ok {downloaded:,}  failed {len(failures):,}")

    on_disk = sum(1 for entry in entries if destination_for(args.out, entry).exists())
    print(f"\ndownloaded {downloaded:,}, failed {len(failures):,}")
    print(f"usable on disk: {on_disk:,} of {len(entries):,} manifest entries")
    if failures:
        print("\nfirst few failures (link rot is expected on third-party hosts):")
        for url, error in failures[:5]:
            print(f"  {error:<40} {url[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
