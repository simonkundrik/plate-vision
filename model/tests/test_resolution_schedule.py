"""Tests for the progressive-resize schedule."""

from __future__ import annotations

import pytest
from PIL import Image

from platevision import meta, transforms


def test_schedule_has_one_entry_per_epoch():
    assert len(transforms.resolution_schedule(30, start=160)) == 30


def test_schedule_ends_at_the_contract_resolution():
    """Finishing anywhere else means the exported model is evaluated at a size it was
    never fine-tuned on, which presents as a bad recipe rather than a bad schedule."""
    final = meta.input_size()[0]
    for epochs in (1, 2, 5, 30, 100):
        assert transforms.resolution_schedule(epochs, start=160)[-1] == final


def test_schedule_starts_at_the_requested_resolution():
    assert transforms.resolution_schedule(30, start=128)[0] == 128


def test_schedule_is_non_decreasing():
    sizes = transforms.resolution_schedule(30, start=128)
    assert all(b >= a for a, b in zip(sizes, sizes[1:], strict=False))


def test_every_resolution_is_a_multiple_of_32():
    """Odd sizes leave ragged feature maps at the deeper downsampling stages."""
    assert all(s % 32 == 0 for s in transforms.resolution_schedule(30, start=128))


def test_the_tail_runs_at_full_resolution():
    """The point of the ramp: cheap epochs early, full resolution for the final stretch."""
    sizes = transforms.resolution_schedule(20, start=128, ramp_fraction=0.5)
    final = meta.input_size()[0]
    assert sizes.count(final) >= 10


def test_single_epoch_runs_at_full_resolution():
    assert transforms.resolution_schedule(1, start=128) == [meta.input_size()[0]]


def test_ramp_fraction_of_one_still_ends_at_full_resolution():
    sizes = transforms.resolution_schedule(10, start=128, ramp_fraction=1.0)
    assert sizes[-1] == meta.input_size()[0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epochs": 0}, "at least 1"),
        ({"epochs": 10, "start": 512}, "exceeds the final"),
        ({"epochs": 10, "ramp_fraction": 0.0}, "ramp_fraction"),
        ({"epochs": 10, "ramp_fraction": 1.5}, "ramp_fraction"),
    ],
)
def test_invalid_schedules_are_rejected(kwargs, match):
    epochs = kwargs.pop("epochs")
    with pytest.raises(ValueError, match=match):
        transforms.resolution_schedule(epochs, **{"start": 128, **kwargs})


def test_train_transform_honours_a_size_override():
    out = transforms.classification_train_transform(size=128)(Image.new("RGB", (200, 300)))
    assert out.shape == (3, 128, 128)


def test_eval_transform_ignores_training_resolution():
    """Validation always runs at the contract size, whatever training is doing."""
    out = transforms.eval_transform()(Image.new("RGB", (200, 300)))
    height, width = meta.input_size()
    assert out.shape == (3, height, width)
