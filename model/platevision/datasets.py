"""Dataset index construction and torch Dataset wrappers.

Index building is deliberately torch-free and returns plain dataclasses, so the filtering
and label-assembly logic (the part that can silently corrupt training) is testable without
importing torch. The Dataset classes on top are thin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from platevision import food101, meta
from platevision import nutrition5k as n5k

SPLITS = ("train", "test")

# Maps a target key from shared/model_meta.json onto the Dish attribute that supplies it.
# Driven by the contract rather than positional order, so adding or reordering targets in
# the contract cannot silently shuffle the regression labels.
_TARGET_SOURCES = {
    "energy": lambda d: d.calories,
    "protein": lambda d: d.protein_g,
    "fat": lambda d: d.fat_g,
    "carbohydrate": lambda d: d.carb_g,
    "mass": lambda d: d.mass_g,
}


@dataclass(frozen=True, slots=True)
class NutritionSample:
    dish_id: str
    image_path: Path
    targets: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Food101Sample:
    image_path: Path
    label: int


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Why samples were dropped. Printed at build time so shrinkage is never a surprise."""

    listed: int
    missing_metadata: int
    missing_image: int
    nonpositive_calories: int
    kept: int


def split_filename(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return f"rgb_{split}_ids.txt"


def targets_for(dish: n5k.Dish) -> tuple[float, ...]:
    """Assemble regression targets in the order declared by the contract."""
    keys = meta.target_keys()
    unknown = [k for k in keys if k not in _TARGET_SOURCES]
    if unknown:
        raise ValueError(f"contract declares targets with no known source: {unknown}")
    return tuple(float(_TARGET_SOURCES[k](dish)) for k in keys)


def build_nutrition5k_index(
    root: Path,
    split: str,
    *,
    drop_nonpositive_calories: bool = True,
) -> tuple[list[NutritionSample], IndexStats]:
    """Build the sample list for one Nutrition5k split.

    Three filters, applied in order:

    1. The dish must have a metadata row.
    2. The overhead RGB frame must exist on disk. Roughly a third of the ids in the split
       files were never captured by the overhead camera, so trusting the split file means
       training on fewer examples than reported.
    3. Calories must be positive. Two dishes in the usable set are labelled zero, which
       is a broken label rather than a genuinely calorie-free plate.
    """
    dishes: dict[str, n5k.Dish] = {}
    for name in n5k.METADATA_FILES:
        path = root / "metadata" / name
        dishes.update(n5k.parse_dish_metadata(path.read_text(encoding="utf-8")))

    split_path = root / "splits" / split_filename(split)
    ids = n5k.parse_split_ids(split_path.read_text(encoding="utf-8"))

    samples: list[NutritionSample] = []
    missing_metadata = missing_image = nonpositive = 0

    for dish_id in ids:
        dish = dishes.get(dish_id)
        if dish is None:
            missing_metadata += 1
            continue

        image_path = root / "imagery" / dish_id / "rgb.png"
        if not image_path.is_file():
            missing_image += 1
            continue

        if drop_nonpositive_calories and dish.calories <= 0:
            nonpositive += 1
            continue

        samples.append(
            NutritionSample(dish_id=dish_id, image_path=image_path, targets=targets_for(dish))
        )

    stats = IndexStats(
        listed=len(ids),
        missing_metadata=missing_metadata,
        missing_image=missing_image,
        nonpositive_calories=nonpositive,
        kept=len(samples),
    )
    return samples, stats


def build_food101_index(root: Path, split: str) -> list[Food101Sample]:
    """Build the sample list for one Food-101 split from ``meta/{split}.txt``.

    Labels come from the committed class ordering, not from directory iteration order,
    which varies by filesystem.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    label_of = {key: i for i, key in enumerate(food101.class_keys())}
    listing = (root / "meta" / f"{split}.txt").read_text(encoding="utf-8")

    samples: list[Food101Sample] = []
    for line in listing.splitlines():
        entry = line.strip()
        if not entry:
            continue
        class_key, _, stem = entry.partition("/")
        if not stem:
            raise ValueError(f"malformed entry in {split}.txt: {entry!r}")
        if class_key not in label_of:
            raise ValueError(f"unknown class {class_key!r} in {split}.txt")
        samples.append(
            Food101Sample(
                image_path=root / "images" / f"{entry}.jpg",
                label=label_of[class_key],
            )
        )
    return samples


def build_ood_index(manifest_path: Path, images_root: Path) -> tuple[list[Food101Sample], int]:
    """Build the out-of-distribution index from a manifest and downloaded images.

    Returns (samples, missing). Entries whose image is not on disk are skipped rather than
    silently producing load errors mid-epoch; link rot is expected on a set assembled from
    third-party hosts, so the count is returned instead of being swallowed.

    Labels here are weak: they come from the search term used to find each image, not from
    anyone checking it. Accuracy measured on this set has a noise floor and should be
    reported alongside an estimate of it.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    samples: list[Food101Sample] = []
    missing = 0
    for entry in manifest["images"]:
        stem = entry["identifier"] or str(abs(hash(entry["url"])))
        path = images_root / entry["class_key"] / f"{stem}.jpg"
        if not path.is_file():
            missing += 1
            continue
        samples.append(Food101Sample(image_path=path, label=entry["label"]))
    return samples, missing


def _load_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


class Nutrition5kDataset:
    """Overhead RGB frame to nutrition targets.

    Implements the torch Dataset protocol without subclassing it, so importing this module
    does not require torch.
    """

    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = list(samples)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        image = _load_rgb(sample.image_path)
        if self.transform is not None:
            image = self.transform(image)

        targets = torch.tensor(sample.targets, dtype=torch.float32)
        if self.target_transform is not None:
            targets = self.target_transform.forward(targets)
        return image, targets


class Food101Dataset:
    """Food photo to class index."""

    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = _load_rgb(sample.image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label
