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
    # Names rather than indices: the vocabulary is built from the training split alone, so
    # a sample cannot know its own encoding without knowing which split it landed in.
    ingredients: tuple[str, ...] = ()


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
    # Defaulted so every existing construction still type-checks. Only non-zero when depth
    # was required: dish_1564159636 has a 0-byte depth_raw.png upstream, and one broken file
    # in 3,262 should cost one dish rather than the run.
    missing_depth: int = 0


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
    require_depth: bool = False,
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
    missing_metadata = missing_image = nonpositive = missing_depth = 0

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

        # Checked here rather than at load time. A dish without a depth map would otherwise
        # raise several hundred steps into an epoch, after the GPU time is spent, and the
        # dataset genuinely ships one broken file.
        if require_depth and not (image_path.parent / "depth_raw.png").is_file():
            missing_depth += 1
            continue

        samples.append(
            NutritionSample(
                dish_id=dish_id,
                image_path=image_path,
                targets=targets_for(dish),
                ingredients=tuple(sorted({i.name for i in dish.ingredients})),
            )
        )

    stats = IndexStats(
        listed=len(ids),
        missing_metadata=missing_metadata,
        missing_image=missing_image,
        nonpositive_calories=nonpositive,
        kept=len(samples),
        missing_depth=missing_depth,
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


def _stack_depth(image, rgb_path: Path):
    """RGB plus the dish's depth map as a fourth channel, as a uint8 tensor.

    Returned as a tensor rather than a PIL image because PIL has no four-channel mode that
    is not RGBA, and an alpha channel is silently dropped or composited by half the
    transforms it would then pass through.
    """
    import numpy as np
    import torch
    from PIL import Image

    from platevision import depth as depth_lib

    depth_path = rgb_path.parent / "depth_raw.png"
    if not depth_path.is_file():
        raise FileNotFoundError(
            f"{depth_path} is missing. Fetch depth with "
            "`python data/download_nutrition5k.py --depth`, or train without --depth."
        )

    with Image.open(depth_path) as raw:
        normalised = depth_lib.normalise_depth(np.asarray(raw))

    rgb = torch.from_numpy(np.asarray(image, dtype=np.uint8)).permute(2, 0, 1)
    channel = torch.from_numpy((normalised * 255.0).astype(np.uint8)).unsqueeze(0)
    if channel.shape[-2:] != rgb.shape[-2:]:
        raise ValueError(
            f"depth {tuple(channel.shape[-2:])} does not match rgb {tuple(rgb.shape[-2:])} "
            f"for {rgb_path.parent.name}"
        )
    return torch.cat([rgb, channel], dim=0)


class Nutrition5kDataset:
    """Overhead RGB frame to nutrition targets.

    Implements the torch Dataset protocol without subclassing it, so importing this module
    does not require torch.
    """

    def __init__(
        self,
        samples,
        transform=None,
        target_transform=None,
        ingredient_vocab=None,
        with_depth: bool = False,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.target_transform = target_transform
        # Appends the depth map as a fourth channel. The published gain on this dataset is
        # 70.6 to 47.6 kcal MAE, and it is the one input change with evidence behind it.
        self.with_depth = with_depth
        # When set, each item gains a multi-hot ingredient vector for the auxiliary head.
        # Left unset the dataset behaves exactly as before, so nothing that does not ask
        # for ingredients has to know they exist.
        self.ingredient_vocab = list(ingredient_vocab) if ingredient_vocab else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        image = _load_rgb(sample.image_path)

        if self.with_depth:
            # Stacked before the transform, not after, so every geometric augmentation moves
            # the depth map with the pixels it describes. Cropping RGB and leaving depth
            # untouched would pair each plate with another plate's geometry.
            image = _stack_depth(image, sample.image_path)

        if self.transform is not None:
            image = self.transform(image)

        targets = torch.tensor(sample.targets, dtype=torch.float32)
        if self.target_transform is not None:
            targets = self.target_transform.forward(targets)

        if self.ingredient_vocab is None:
            return image, targets

        from platevision.ingredients import multi_hot

        return image, targets, multi_hot(sample.ingredients, self.ingredient_vocab)


class ZoomedDishes:
    """Nutrition5k with each dish re-framed as if the camera sat at a different distance.

    Nutrition5k is shot from a fixed overhead rig, so apparent size in pixels maps directly
    to real size and the model can read scale straight off the image. A phone held at an
    unknown distance destroys that mapping, and zooming the test images is the closest
    simulation of the loss obtainable without weighed photographs.

    Two details decide whether this measures scale or something else:

    **The crop keeps the source aspect ratio.** Nutrition5k frames are 640x480 and the eval
    transform squashes them to a square, so a square crop would change the aspect distortion
    the model was trained under at the same time as the scale. The degradation would then be
    partly a shape change wearing a scale label.

    **Padding replicates the edge**, for the reason ``transforms.RandomZoomOut`` gives: a flat
    black border is a cue no camera produces, and a factor below 1 needs pixels from outside
    the frame.

    Factors are drawn once per dish rather than per epoch, so every zoom level sees the same
    assignment and comparisons are not confounded by which dish got which factor.
    """

    def __init__(self, samples, transform=None, target_transform=None, zoom=1.0, seed=0):
        if zoom < 1.0:
            raise ValueError(f"zoom is a half-range and must be at least 1.0, got {zoom}")

        import numpy as np

        self.samples = list(samples)
        self.transform = transform
        self.target_transform = target_transform
        rng = np.random.default_rng(seed)
        self.factors = (
            rng.uniform(1 / zoom, zoom, size=len(self.samples))
            if zoom > 1.0
            else np.ones(len(self.samples))
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        image = zoom_image(_load_rgb(sample.image_path), float(self.factors[index]))
        if self.transform is not None:
            image = self.transform(image)

        targets = torch.tensor(sample.targets, dtype=torch.float32)
        if self.target_transform is not None:
            targets = self.target_transform.forward(targets)
        return image, targets


def zoom_image(image, factor: float):
    """Re-frame a PIL image as a camera ``factor`` times closer, keeping the aspect ratio.

    Above 1 the frame tightens and the food fills more of it. Below 1 the frame widens past
    the original photograph, and the pixels that were never captured are filled by
    replicating the edge.
    """
    if factor <= 0:
        raise ValueError(f"zoom factor must be positive, got {factor}")
    if factor == 1.0:
        return image

    import numpy as np
    from PIL import Image

    width, height = image.size
    crop_width = max(1, int(round(width / factor)))
    crop_height = max(1, int(round(height / factor)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    if left >= 0 and top >= 0:
        return image.crop((left, top, left + crop_width, top + crop_height))

    # Zoomed out past the frame. Pad first so the crop lands inside real pixels, rather than
    # letting PIL fill the overhang with black.
    pad_x, pad_y = max(0, -left), max(0, -top)
    padded = np.pad(np.asarray(image), ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
    return Image.fromarray(padded).crop(
        (left + pad_x, top + pad_y, left + pad_x + crop_width, top + pad_y + crop_height)
    )


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
