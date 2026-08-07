"""Tests for image transforms.

The eval transform has one job: be indistinguishable from what the exported ONNX graph
does. Everything here exists to pin that down, because a mismatch produces metrics that
describe a model nobody ships.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torchvision.transforms.v2 import InterpolationMode

from platevision import meta, transforms


@pytest.fixture
def sample_image():
    """Deterministic non-uniform image. A flat colour would hide interpolation differences."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (137, 211, 3), dtype=np.uint8), mode="RGB")


def test_eval_transform_output_shape_and_dtype(sample_image):
    out = transforms.eval_transform()(sample_image)
    height, width = meta.input_size()
    assert out.shape == (3, height, width)
    assert out.dtype == torch.float32


def test_eval_transform_is_deterministic(sample_image):
    t = transforms.eval_transform()
    assert torch.equal(t(sample_image), t(sample_image))


def test_eval_transform_matches_the_documented_order(sample_image):
    """Float conversion, then resize, then normalise, exactly as the contract states."""
    size = list(meta.input_size())
    mean, std = meta.normalization()

    expected = TF.to_image(sample_image)
    expected = TF.to_dtype(expected, torch.float32, scale=True)
    expected = TF.resize(expected, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
    expected = TF.normalize(expected, mean=mean, std=std)

    got = transforms.eval_transform()(sample_image)
    assert torch.allclose(got, expected, atol=1e-6)


def test_resizing_before_the_float_conversion_gives_different_pixels(sample_image):
    """The reason preprocessing order is part of the contract rather than a code detail.

    Swapping the first two steps is the kind of change that looks like a harmless
    refactor, passes every shape assertion, and shifts every pixel.
    """
    size = list(meta.input_size())
    mean, std = meta.normalization()

    wrong = TF.to_image(sample_image)
    wrong = TF.resize(wrong, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
    wrong = TF.to_dtype(wrong, torch.float32, scale=True)
    wrong = TF.normalize(wrong, mean=mean, std=std)

    got = transforms.eval_transform()(sample_image)
    assert not torch.allclose(got, wrong, atol=1e-6)
    assert (got - wrong).abs().max() > 1e-4


def test_eval_transform_rejects_an_unimplemented_order(monkeypatch):
    monkeypatch.setattr(meta, "preprocessing_order", lambda: ("resize", "normalize"))
    with pytest.raises(ValueError, match="unsupported preprocessing order"):
        transforms.eval_transform()


@pytest.mark.parametrize(
    "factory",
    [transforms.classification_train_transform, transforms.nutrition_train_transform],
    ids=["classification", "nutrition"],
)
def test_train_transforms_produce_the_contract_shape(factory, sample_image):
    out = factory()(sample_image)
    height, width = meta.input_size()
    assert out.shape == (3, height, width)
    assert out.dtype == torch.float32


def test_train_transforms_actually_randomise(sample_image):
    t = transforms.classification_train_transform()
    torch.manual_seed(0)
    first = t(sample_image)
    torch.manual_seed(1)
    second = t(sample_image)
    assert not torch.equal(first, second)


def test_nutrition_crop_is_milder_than_classification_crop():
    """Cropping away food while keeping the calorie label is manufactured label noise.

    On 2,755 training examples there is no budget for it, so the nutrition recipe keeps
    far more of the plate in frame than the classification recipe does.
    """
    crop_of = {}
    for name, factory in (
        ("classification", transforms.classification_train_transform),
        ("nutrition", transforms.nutrition_train_transform),
    ):
        crops = [t for t in factory().transforms if type(t).__name__ == "RandomResizedCrop"]
        assert len(crops) == 1, f"expected exactly one cropping transform in {name}"
        crop_of[name] = crops[0].scale

    assert crop_of["nutrition"][0] > crop_of["classification"][0]


def test_only_the_nutrition_recipe_corrects_for_viewing_angle():
    """Nutrition5k is a fixed overhead rig; the phone is not. Perspective and rotation
    attack that gap without changing how much food is visible."""
    names = {type(t).__name__ for t in transforms.nutrition_train_transform().transforms}
    assert {"RandomPerspective", "RandomRotation"} <= names

    baseline = {type(t).__name__ for t in transforms.classification_train_transform().transforms}
    assert "RandomPerspective" not in baseline
