"""Tests for quantile regression.

Pinball loss is asymmetric, and that asymmetry is the entire mechanism. A symmetric loss
would drive all three heads to the conditional mean and the interval would collapse to a
point while every shape assertion still passed.
"""

from __future__ import annotations

import pytest
import torch

from platevision.quantile import (
    PinballLoss,
    enforce_monotonic,
    interval_coverage,
    pinball_loss,
)

QUANTILES = torch.tensor([0.05, 0.5, 0.95])


def preds(batch=4, targets=2, quantiles=3, value=0.0):
    return torch.full((batch, targets, quantiles), value)


def test_zero_error_gives_zero_loss():
    target = torch.zeros(4, 2)
    assert pinball_loss(preds(), target, QUANTILES).item() == pytest.approx(0.0)


def test_loss_is_non_negative():
    torch.manual_seed(0)
    p = torch.randn(8, 2, 3)
    t = torch.randn(8, 2)
    assert pinball_loss(p, t, QUANTILES, reduction="none").min().item() >= 0.0


def test_high_quantile_penalises_under_prediction_more():
    """The asymmetry that makes the 0.95 head sit high rather than at the mean."""
    target = torch.zeros(1, 1)
    q = torch.tensor([0.95])

    under = pinball_loss(torch.full((1, 1, 1), -1.0), target, q).item()
    over = pinball_loss(torch.full((1, 1, 1), 1.0), target, q).item()

    assert under == pytest.approx(0.95)
    assert over == pytest.approx(0.05)
    assert under > over


def test_low_quantile_penalises_over_prediction_more():
    target = torch.zeros(1, 1)
    q = torch.tensor([0.05])

    under = pinball_loss(torch.full((1, 1, 1), -1.0), target, q).item()
    over = pinball_loss(torch.full((1, 1, 1), 1.0), target, q).item()

    assert under == pytest.approx(0.05)
    assert over == pytest.approx(0.95)
    assert over > under


def test_median_quantile_is_symmetric():
    target = torch.zeros(1, 1)
    q = torch.tensor([0.5])
    under = pinball_loss(torch.full((1, 1, 1), -1.0), target, q).item()
    over = pinball_loss(torch.full((1, 1, 1), 1.0), target, q).item()
    assert under == pytest.approx(over)


def test_optimum_recovers_the_empirical_quantile():
    """The property the whole approach rests on: minimising pinball loss at level q lands
    on the q-th quantile of the data, not on its mean."""
    torch.manual_seed(0)
    samples = torch.distributions.LogNormal(0.0, 1.0).sample((4000,))
    q = torch.tensor([0.9])

    prediction = torch.zeros(1, 1, 1, requires_grad=True)
    optimizer = torch.optim.Adam([prediction], lr=0.05)
    for _ in range(600):
        optimizer.zero_grad()
        loss = pinball_loss(prediction.expand(samples.numel(), 1, 1), samples.unsqueeze(-1), q)
        loss.backward()
        optimizer.step()

    expected = torch.quantile(samples, 0.9).item()
    assert prediction.item() == pytest.approx(expected, rel=0.1)
    assert abs(prediction.item() - samples.mean().item()) > abs(prediction.item() - expected)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="disagree"):
        pinball_loss(torch.zeros(4, 2, 3), torch.zeros(4, 5), QUANTILES)


def test_quantile_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="quantile outputs"):
        pinball_loss(torch.zeros(4, 2, 3), torch.zeros(4, 2), torch.tensor([0.5]))


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_reductions_are_supported(reduction):
    out = pinball_loss(torch.randn(4, 2, 3), torch.randn(4, 2), QUANTILES, reduction)
    assert out.ndim == (3 if reduction == "none" else 0)


def test_unknown_reduction_is_rejected():
    with pytest.raises(ValueError, match="unknown reduction"):
        pinball_loss(torch.randn(4, 2, 3), torch.randn(4, 2), QUANTILES, "median")


# --- module wrapper ---------------------------------------------------------------


def test_module_matches_the_function():
    torch.manual_seed(0)
    p, t = torch.randn(4, 2, 3), torch.randn(4, 2)
    criterion = PinballLoss([0.05, 0.5, 0.95])
    assert criterion(p, t).item() == pytest.approx(pinball_loss(p, t, QUANTILES).item())


def test_quantiles_are_registered_as_a_buffer():
    """A buffer, so .to(device) moves them and they land in the state dict."""
    criterion = PinballLoss([0.05, 0.5, 0.95])
    assert "quantiles" in criterion.state_dict()


@pytest.mark.parametrize(
    ("levels", "match"),
    [
        ([], "at least one"),
        ([0.0, 0.5], "strictly in"),
        ([0.5, 1.0], "strictly in"),
        ([0.95, 0.5, 0.05], "ascending"),
    ],
)
def test_invalid_quantile_levels_are_rejected(levels, match):
    with pytest.raises(ValueError, match=match):
        PinballLoss(levels)


# --- monotonicity and coverage -----------------------------------------------------


def test_crossed_quantiles_are_sorted():
    """Nothing in the loss couples the heads, so a model can predict its 5th percentile
    above its 95th. That is a negative-width interval, and coverage computed from it is
    meaningless."""
    crossed = torch.tensor([[[9.0, 5.0, 1.0]]])
    fixed = enforce_monotonic(crossed)
    assert torch.equal(fixed, torch.tensor([[[1.0, 5.0, 9.0]]]))


def test_already_ordered_predictions_are_untouched():
    ordered = torch.tensor([[[1.0, 5.0, 9.0]]])
    assert torch.equal(enforce_monotonic(ordered), ordered)


def test_coverage_is_one_when_every_target_is_inside():
    predictions = torch.tensor([[[0.0, 5.0, 10.0]], [[0.0, 5.0, 10.0]]])
    targets = torch.tensor([[5.0], [7.0]])
    assert interval_coverage(predictions, targets).item() == pytest.approx(1.0)


def test_coverage_is_zero_when_every_target_is_outside():
    predictions = torch.tensor([[[0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0]]])
    targets = torch.tensor([[50.0], [-50.0]])
    assert interval_coverage(predictions, targets).item() == pytest.approx(0.0)


def test_coverage_is_reported_per_target():
    predictions = torch.tensor([[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]]])
    targets = torch.tensor([[5.0, 99.0]])
    coverage = interval_coverage(predictions, targets)
    assert coverage.shape == (2,)
    assert coverage[0].item() == pytest.approx(1.0)
    assert coverage[1].item() == pytest.approx(0.0)


def test_coverage_includes_the_boundaries():
    predictions = torch.tensor([[[0.0, 5.0, 10.0]]])
    assert interval_coverage(predictions, torch.tensor([[10.0]])).item() == pytest.approx(1.0)
