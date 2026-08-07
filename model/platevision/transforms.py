"""Image transforms.

The eval transform is not a stylistic choice. It has to reproduce, step for step, what the
exported ONNX graph does, or the reported metrics describe a model that never ships. Both
read the sequence from ``shared/model_meta.json`` so neither can drift alone.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode

from platevision import meta

_INTERPOLATION = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


def _resize_spec() -> tuple[tuple[int, int], InterpolationMode, bool]:
    spec = meta.load_meta()["preprocessing"]["resize"]
    mode = _INTERPOLATION[spec["mode"]]
    return (spec["height"], spec["width"]), mode, bool(spec["antialias"])


def eval_transform() -> v2.Compose:
    """The exact preprocessing the exported graph performs.

    Steps follow ``preprocessing.order`` from the contract: convert to float in the unit
    range first, then resize, then normalise. Resizing uint8 and converting afterwards
    rounds at a different point and produces measurably different pixels.
    """
    order = meta.preprocessing_order()
    if order != meta.SUPPORTED_PREPROCESSING_ORDER:
        raise ValueError(f"unsupported preprocessing order {order}")

    size, mode, antialias = _resize_spec()
    mean, std = meta.normalization()
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size, interpolation=mode, antialias=antialias),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def classification_train_transform(*, scale: tuple[float, float] = (0.35, 1.0)) -> v2.Compose:
    """Augmentation for Food-101 classification.

    Aggressive cropping is fine here. A crop of a plate of carbonara is still carbonara,
    so the label survives the augmentation intact.
    """
    size, mode, antialias = _resize_spec()
    mean, std = meta.normalization()
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(size, scale=scale, interpolation=mode, antialias=antialias),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def nutrition_train_transform(*, scale: tuple[float, float] = (0.85, 1.0)) -> v2.Compose:
    """Augmentation for Nutrition5k regression.

    Two differences from the classification recipe, both deliberate.

    Cropping is kept mild. Aggressive random cropping is safe for a class label but not
    for a calorie label: cropping away half the plate removes food from the image while
    the target still says the original number of calories. That is label noise
    manufactured by the augmentation, and on 2,755 training examples there is no budget
    for it.

    Perspective and rotation are added instead. Nutrition5k was captured by a fixed
    overhead rig, and the deployed model sees handheld phone photos taken from whatever
    angle the user happened to hold. These augmentations attack that domain gap directly,
    and unlike cropping they leave the amount of visible food unchanged.
    """
    size, mode, antialias = _resize_spec()
    mean, std = meta.normalization()
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(size, scale=scale, interpolation=mode, antialias=antialias),
            v2.RandomHorizontalFlip(),
            v2.RandomPerspective(distortion_scale=0.3, p=0.5),
            v2.RandomRotation(degrees=15, interpolation=mode),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
            v2.Normalize(mean=mean, std=std),
        ]
    )
