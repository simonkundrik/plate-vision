"""Image transforms.

The eval transform is not a stylistic choice. It has to reproduce, step for step, what the
exported ONNX graph does, or the reported metrics describe a model that never ships. Both
read the sequence from ``shared/model_meta.json`` so neither can drift alone.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as F

from platevision import meta

_INTERPOLATION = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


def _resize_spec(size: int | None = None) -> tuple[tuple[int, int], InterpolationMode, bool]:
    spec = meta.load_meta()["preprocessing"]["resize"]
    mode = _INTERPOLATION[spec["mode"]]
    resolved = (size, size) if size else (spec["height"], spec["width"])
    return resolved, mode, bool(spec["antialias"])


def resolution_schedule(
    epochs: int,
    *,
    start: int = 160,
    end: int | None = None,
    ramp_fraction: float = 0.7,
    multiple: int = 32,
) -> list[int]:
    """Per-epoch training resolution, ramping from ``start`` up to the contract size.

    Small images early are cheap, so more epochs fit in the same GPU budget; the last
    stretch runs at full resolution so the weights are tuned for the size that ships.

    The schedule is forced to end at the contract resolution. Finishing at anything else
    means the exported model is evaluated at a resolution it was never fine-tuned on,
    which costs accuracy that looks like a bad training recipe rather than a bad schedule.
    Resolutions are rounded to a multiple of 32 to keep the downsampling stages whole.
    """
    final = end or meta.input_size()[0]
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if start > final:
        raise ValueError(f"start resolution {start} exceeds the final resolution {final}")
    if not 0.0 < ramp_fraction <= 1.0:
        raise ValueError(f"ramp_fraction must be in (0, 1], got {ramp_fraction}")

    ramp_epochs = max(1, int(round(epochs * ramp_fraction)))
    sizes: list[int] = []
    for epoch in range(epochs):
        if epoch >= ramp_epochs - 1:
            sizes.append(final)
            continue
        progress = epoch / (ramp_epochs - 1) if ramp_epochs > 1 else 1.0
        raw = start + (final - start) * progress
        sizes.append(int(round(raw / multiple) * multiple))

    sizes[-1] = final
    return sizes


# Measured over 300 random Nutrition5k depth maps after `depth.normalise_depth`. The mean is
# high and the spread is tiny because the rig never moves: almost every pixel is the table at
# a constant distance, and the food is a thin band above it. Normalising with the ImageNet
# std would leave this channel nearly flat, which is the same as not supplying it.
DEPTH_MEAN = 0.612
DEPTH_STD = 0.052


class RgbOnly(v2.Transform):
    """Apply a colour transform to the first three channels and leave the rest alone.

    ColorJitter on a four-channel tensor treats depth as colour: brightness and saturation
    are meaningless for a distance map, and hue rotation mixes it into the red channel. The
    augmentation would be silently corrupting the one signal the channel exists to carry.
    """

    def __init__(self, wrapped: v2.Transform) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *inputs):  # noqa: D102
        image = inputs[0] if len(inputs) == 1 else inputs
        if not isinstance(image, torch.Tensor) or image.shape[-3] <= 3:
            return self.wrapped(image)

        colour, rest = image[..., :3, :, :], image[..., 3:, :, :]
        return torch.cat([self.wrapped(colour), rest], dim=-3)


def _normalization(channels: int) -> tuple[list[float], list[float]]:
    """Contract mean/std, extended for a depth channel when one is present."""
    mean, std = meta.normalization()
    if channels == 3:
        return mean, std
    if channels != 4:
        raise ValueError(f"expected 3 or 4 channels, got {channels}")
    return [*mean, DEPTH_MEAN], [*std, DEPTH_STD]


def eval_transform(channels: int = 3) -> v2.Compose:
    """The exact preprocessing the exported graph performs.

    Steps follow ``preprocessing.order`` from the contract: convert to float in the unit
    range first, then resize, then normalise. Resizing uint8 and converting afterwards
    rounds at a different point and produces measurably different pixels.
    """
    order = meta.preprocessing_order()
    if order != meta.SUPPORTED_PREPROCESSING_ORDER:
        raise ValueError(f"unsupported preprocessing order {order}")

    size, mode, antialias = _resize_spec()
    mean, std = _normalization(channels)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size, interpolation=mode, antialias=antialias),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def classification_train_transform(
    *, scale: tuple[float, float] = (0.35, 1.0), size: int | None = None
) -> v2.Compose:
    """Augmentation for Food-101 classification.

    Aggressive cropping is fine here. A crop of a plate of carbonara is still carbonara,
    so the label survives the augmentation intact.

    ``size`` overrides the contract resolution for progressive resizing. Only training
    transforms accept it: evaluation always runs at the contract size, or the reported
    metric would not describe the deployed model.
    """
    size, mode, antialias = _resize_spec(size)
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


class RandomZoomOut(v2.Transform):
    """Shrink the image within the frame, as a more distant camera would.

    Cropping cannot simulate a further camera: it removes food while the calorie label
    still claims all of it. Shrinking removes nothing, so the label stays true, which is
    what makes this the one scale augmentation a regression target tolerates.

    Padding replicates the edge rather than filling with a constant. A flat border is a cue
    no camera produces, and the model would learn to read it as "this food is smaller than
    it looks" rather than learning to judge size without the cue at all.
    """

    def __init__(self, max_factor: float = 1.6, p: float = 0.5, interpolation=None) -> None:
        super().__init__()
        if max_factor < 1.0:
            raise ValueError(f"max_factor must be at least 1.0, got {max_factor}")
        self.max_factor = max_factor
        self.p = p
        self.interpolation = interpolation or InterpolationMode.BILINEAR

    def transform(self, inpt, params):  # noqa: D102
        if float(torch.rand(())) >= self.p or self.max_factor <= 1.0:
            return inpt

        height, width = F.get_size(inpt)
        factor = float(torch.empty(()).uniform_(1.0, self.max_factor))
        shrunk = F.resize(
            inpt,
            [max(1, int(round(height / factor))), max(1, int(round(width / factor)))],
            interpolation=self.interpolation,
            antialias=True,
        )

        new_height, new_width = F.get_size(shrunk)
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        return F.pad(
            shrunk,
            [left, top, width - new_width - left, height - new_height - top],
            padding_mode="edge",
        )


def nutrition_train_transform(
    *,
    scale: tuple[float, float] = (0.85, 1.0),
    size: int | None = None,
    zoom_out: float = 1.6,
    channels: int = 3,
) -> v2.Compose:
    """Augmentation for Nutrition5k regression.

    Three differences from the classification recipe, all deliberate.

    Cropping is kept mild. Aggressive random cropping is safe for a class label but not
    for a calorie label: cropping away half the plate removes food from the image while
    the target still says the original number of calories. That is label noise
    manufactured by the augmentation, and on 2,755 training examples there is no budget
    for it.

    Perspective and rotation are added instead. Nutrition5k was captured by a fixed
    overhead rig, and the deployed model sees handheld phone photos taken from whatever
    angle the user happened to hold. These augmentations attack that domain gap directly,
    and unlike cropping they leave the amount of visible food unchanged.

    Zooming *out* is added for the same reason, and it is free of the objection above.
    Shrinking the plate within the frame is exactly what a more distant camera does, and
    it removes no food at all, so the calorie label stays true. That asymmetry matters:
    a fixed overhead rig at constant height means apparent size in pixels *is* real size,
    and a model trained only on it learns to read scale off the image. Measured on this
    model, simulating unknown camera distance costs 12.5 points of calorie error, 18.1%
    to 30.6%, and interval coverage falls from 64.6% to 42.9%. See
    ``scripts/measure_scale_dependence.py``. ``zoom_out=1.0`` disables it.
    """
    size, mode, antialias = _resize_spec(size)
    mean, std = _normalization(channels)

    steps: list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if zoom_out > 1.0:
        steps.append(RandomZoomOut(max_factor=zoom_out, p=0.5, interpolation=mode))

    steps += [
        v2.RandomResizedCrop(size, scale=scale, interpolation=mode, antialias=antialias),
        v2.RandomHorizontalFlip(),
        v2.RandomPerspective(distortion_scale=0.3, p=0.5),
        v2.RandomRotation(degrees=15, interpolation=mode),
        # Wrapped, because brightness and saturation are meaningless for a distance map and
        # hue rotation would mix it into the red channel. Geometric transforms above are
        # applied to every channel on purpose: they must move depth with the pixels it
        # describes, or each plate is paired with another plate's geometry.
        RgbOnly(v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03)),
        v2.Normalize(mean=mean, std=std),
    ]
    return v2.Compose(steps)
