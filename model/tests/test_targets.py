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
