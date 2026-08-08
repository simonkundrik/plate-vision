"""Food-101 label handling.

The label ordering is a hard contract: index N here is position N on the logits axis of
the exported model. Reordering the list does not break anything loudly, it just relabels
every prediction, so the order is checksummed and asserted in tests.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from platevision._contract import contract_path

LABELS_PATH = contract_path("food101_labels.json")

DOWNLOAD_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"


@lru_cache(maxsize=1)
def load_labels(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the Food-101 label contract."""
    labels_path = path or LABELS_PATH
    with labels_path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    labels = data["labels"]
    if len(labels) != data["count"]:
        raise ValueError(f"count is {data['count']} but {len(labels)} labels are listed")

    keys = [entry["key"] for entry in labels]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate class keys")

    indices = [entry["index"] for entry in labels]
    if indices != list(range(len(labels))):
        raise ValueError("label indices must be contiguous and start at 0")

    actual = order_digest(keys)
    if actual != data["order_sha256"]:
        raise ValueError(
            f"label order digest mismatch: file says {data['order_sha256']}, "
            f"computed {actual}. Reordering labels relabels every prediction."
        )

    return data


def order_digest(keys: list[str]) -> str:
    """Stable digest over the ordered class keys."""
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def class_keys() -> list[str]:
    """Class keys in logits-axis order."""
    return [entry["key"] for entry in load_labels()["labels"]]


def display_names() -> list[str]:
    """Human-readable class names in logits-axis order."""
    return [entry["display"] for entry in load_labels()["labels"]]


def verify_against_classes_txt(classes_txt: Path) -> None:
    """Check a downloaded Food-101 ``meta/classes.txt`` against the committed contract.

    The committed label file was generated from dataset metadata rather than from the
    tarball, so this is what proves the two actually agree.
    """
    downloaded = [line.strip() for line in classes_txt.read_text().splitlines() if line.strip()]
    expected = class_keys()
    if downloaded != expected:
        first = next(
            (i for i, (a, b) in enumerate(zip(downloaded, expected, strict=False)) if a != b),
            min(len(downloaded), len(expected)),
        )
        raise ValueError(
            f"{classes_txt} disagrees with shared/food101_labels.json at index {first}: "
            f"downloaded {downloaded[first : first + 1]}, expected {expected[first : first + 1]}"
        )
