#!/usr/bin/env python
"""Fetch and verify Food-101.

The tarball is roughly 5 GB. Training runs on Kaggle, where Food-101 is already available
as a mounted dataset, so a local copy is only needed for local smoke tests and for
regenerating the label contract. Use --verify-only against a copy you already have.

The important part of this script is not the download, it is the verification: the label
ordering in shared/food101_labels.json was generated from dataset metadata, and this proves
it matches the tarball's own meta/classes.txt. Index N in that file is position N on the
logits axis, so a mismatch would silently relabel every prediction the model makes.

Usage:
    python data/download_food101.py                        # download, extract, verify
    python data/download_food101.py --verify-only --out /path/to/food-101
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from platevision import food101

DEFAULT_OUT = Path(__file__).resolve().parent / "food101"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        step = 100 * 1024 * 1024
        next_mark = step
        with tmp.open("wb") as fh:
            while chunk := resp.read(1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if done >= next_mark:
                    pct = f" ({done / total:.0%})" if total else ""
                    print(f"  {done / 1024**3:.2f} GB{pct}")
                    next_mark += step
    tmp.replace(dest)
    print(f"  done, {dest.stat().st_size / 1024**3:.2f} GB")


def extract(archive: Path, out: Path) -> Path:
    print(f"Extracting {archive.name}")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        # filter="data" refuses absolute paths and traversal entries. Required on 3.14+
        # and a good idea regardless when unpacking a third-party archive.
        tar.extractall(out, filter="data")
    root = out / "food-101"
    if not root.is_dir():
        raise SystemExit(f"expected {root} after extraction, not found")
    print(f"  -> {root}")
    return root


def verify(root: Path) -> int:
    classes_txt = root / "meta" / "classes.txt"
    if not classes_txt.is_file():
        raise SystemExit(f"missing {classes_txt}")

    food101.verify_against_classes_txt(classes_txt)
    labels = food101.load_labels()
    print(f"  meta/classes.txt matches shared/food101_labels.json ({labels['count']} classes)")

    images = root / "images"
    if images.is_dir():
        dirs = sorted(p.name for p in images.iterdir() if p.is_dir())
        expected = food101.class_keys()
        if dirs != sorted(expected):
            missing = sorted(set(expected) - set(dirs))
            extra = sorted(set(dirs) - set(expected))
            raise SystemExit(f"image dirs disagree. missing={missing[:5]} extra={extra[:5]}")
        print(f"  {len(dirs)} image class directories present")
    else:
        print("  images/ not present, skipped directory check")

    for split in ("train.txt", "test.txt"):
        path = root / "meta" / split
        if path.is_file():
            n = sum(1 for line in path.read_text().splitlines() if line.strip())
            print(f"  meta/{split}: {n:,} entries")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--keep-archive", action="store_true", help="do not delete the tarball")
    args = parser.parse_args(argv)

    if args.verify_only:
        root = args.out if (args.out / "meta").is_dir() else args.out / "food-101"
        print(f"Verifying {root}")
        return verify(root)

    archive = args.out / "food-101.tar.gz"
    if archive.is_file():
        print(f"Archive already present, skipping download: {archive}")
    else:
        download(food101.DOWNLOAD_URL, archive)

    root = extract(archive, args.out)
    print("Verifying")
    code = verify(root)

    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"  removed {archive.name} (pass --keep-archive to retain it)")
    if shutil.disk_usage(args.out).free < 5 * 1024**3:
        print("  warning: under 5 GB free on this volume")
    return code


if __name__ == "__main__":
    sys.exit(main())
