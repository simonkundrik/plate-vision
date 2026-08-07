"""Tests for the nutrition target transform."""

from __future__ import annotations

import math

import pytest
import torch

from platevision import meta
from platevision.targets import TargetTransform

WIDTH = 5


def rows(n=8):
    """Target rows spanning the real calorie range, which is skewed and wide."""
    return [
        (float(100 * (i + 1)), float(10 + i), float(5 + i), float(20 + i), float(150 + 10 * i))
        for i in range(n)
    ]


def test_fit_uses_contract_keys_by_default():
    t = TargetTransform.fit(rows())
    assert t.keys == tuple(meta.target_keys())
    assert len(t.mean) == len(t.std) == WIDTH


def test_round_trip_recovers_the_original_values():
    t = TargetTransform.fit(rows())
    raw = torch.tensor(rows(), dtype=torch.float32)
    assert torch.allclose(t.inverse(t.forward(raw)), raw, rtol=1e-4, atol=1e-3)


def test_forward_is_strictly_monotonic():
    """The property quantile regression depends on.

    Quantiles are equivariant under a monotonically increasing transform, so a head
    trained in log space can be inverted afterwards and the interval still means what it
    claims. If this ordering were ever broken, the 5th and 95th percentile predictions
    could swap places on inversion.
    """
    t = TargetTransform.fit(rows())
    ascending = torch.tensor(
        [[0.0] * WIDTH, [50.0] * WIDTH, [500.0] * WIDTH, [3943.0] * WIDTH],
        dtype=torch.float32,
    )
    transformed = t.forward(ascending)
    deltas = transformed[1:] - transformed[:-1]
    assert (deltas > 0).all()


def test_inverse_preserves_quantile_ordering():
    t = TargetTransform.fit(rows())
    q05, q50, q95 = -1.5, 0.0, 1.5
    stacked = torch.tensor([[q05] * WIDTH, [q50] * WIDTH, [q95] * WIDTH], dtype=torch.float32)
    inverted = t.inverse(stacked)
    assert (inverted[1] > inverted[0]).all()
    assert (inverted[2] > inverted[1]).all()


def test_log_space_compresses_the_skew():
    """A 40x spread in raw calories should not remain a 40x spread after transformation."""
    t = TargetTransform.fit(rows())
    small = t.forward(torch.tensor([[100.0] * WIDTH]))
    large = t.forward(torch.tensor([[3943.0] * WIDTH]))
    raw_ratio = 3943.0 / 100.0
    transformed_gap = (large - small).abs().max().item()
    assert transformed_gap < raw_ratio


def test_zero_is_representable():
    """Calories of zero must survive the transform; log1p(0) is 0, log(0) is not."""
    t = TargetTransform.fit(rows())
    zeros = torch.zeros(1, WIDTH)
    assert torch.isfinite(t.forward(zeros)).all()
    assert torch.allclose(t.inverse(t.forward(zeros)), zeros, atol=1e-4)


def test_inverse_handles_quantile_predictions():
    """Predictions are (batch, targets, quantiles), labels are (batch, targets). The
    per-target statistics have to align to different axes for each."""
    t = TargetTransform.fit(rows())
    predictions = torch.zeros(4, WIDTH, 3)

    inverted = t.inverse(predictions)

    assert inverted.shape == (4, WIDTH, 3)
    # A standardised zero maps back to expm1(mean) for that target, identically across
    # the quantile axis.
    for target in range(WIDTH):
        expected = math.expm1(t.mean[target])
        assert inverted[0, target, 0].item() == pytest.approx(expected, rel=1e-4)
        assert inverted[0, target, 2].item() == pytest.approx(expected, rel=1e-4)


def test_inverse_applies_per_target_statistics_along_the_target_axis():
    """The bug this guards: broadcasting the statistics onto the quantile axis instead.
    With equal target and quantile counts it would not even raise."""
    t = TargetTransform(mean=(0.0, 5.0, 0.0), std=(1.0, 1.0, 1.0), keys=("a", "b", "c"))
    predictions = torch.zeros(1, 3, 3)

    inverted = t.inverse(predictions)

    assert inverted[0, 0, 0].item() == pytest.approx(math.expm1(0.0), abs=1e-5)
    assert inverted[0, 1, 0].item() == pytest.approx(math.expm1(5.0), rel=1e-4)
    assert inverted[0, 1, 2].item() == pytest.approx(math.expm1(5.0), rel=1e-4)


def test_round_trip_holds_for_quantile_shaped_tensors():
    t = TargetTransform.fit(rows())
    raw = torch.rand(4, WIDTH, 3) * 500
    assert torch.allclose(t.inverse(t.forward(raw)), raw, rtol=1e-3, atol=1e-2)


def test_wrong_target_count_is_rejected():
    t = TargetTransform.fit(rows())
    with pytest.raises(ValueError, match="expected 5 targets"):
        t.inverse(torch.zeros(4, 2))


def test_single_sample_targets_are_supported():
    """The Dataset applies the transform per item, so it passes a rank-1 tensor."""
    t = TargetTransform.fit(rows())
    single = torch.tensor(rows(1)[0], dtype=torch.float32)

    transformed = t.forward(single)

    assert transformed.shape == (WIDTH,)
    assert torch.allclose(t.inverse(transformed), single, rtol=1e-3, atol=1e-2)


def test_unsupported_rank_is_rejected():
    t = TargetTransform.fit(rows())
    with pytest.raises(ValueError, match="rank 1, 2, or 3"):
        t.inverse(torch.zeros(2, WIDTH, 3, 1))


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError, match="zero samples"):
        TargetTransform.fit([])


def test_fit_rejects_negative_targets():
    with pytest.raises(ValueError, match="log1p is undefined"):
        TargetTransform.fit([(-1.0, 1.0, 1.0, 1.0, 1.0)])


def test_fit_rejects_wrong_width():
    with pytest.raises(ValueError, match="must have 5 values"):
        TargetTransform.fit([(1.0, 2.0)])


def test_constant_column_does_not_divide_by_zero():
    constant = [(200.0, 10.0, 5.0, 20.0, 150.0)] * 4
    t = TargetTransform.fit(constant)
    assert all(s > 0 for s in t.std)
    raw = torch.tensor(constant, dtype=torch.float32)
    assert torch.isfinite(t.forward(raw)).all()


def test_rejects_nonpositive_std_on_construction():
    with pytest.raises(ValueError, match="positive"):
        TargetTransform(mean=(0.0,), std=(0.0,), keys=("energy",))


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        TargetTransform(mean=(0.0, 1.0), std=(1.0,), keys=("energy",))


def test_serialisation_round_trip():
    t = TargetTransform.fit(rows())
    assert TargetTransform.from_dict(t.to_dict()) == t


def test_fitted_statistics_are_log_space_statistics():
    single = [(float(v), 1.0, 1.0, 1.0, 1.0) for v in (0.0, 9.0, 99.0)]
    t = TargetTransform.fit(single)
    expected_mean = sum(math.log1p(v) for v in (0.0, 9.0, 99.0)) / 3
    assert t.mean[0] == pytest.approx(expected_mean)
