"""Tests for nutrition metrics.

Everything here reports in real units. The model trains in standardised log space, where a
loss of 0.31 tells nobody whether the thing is usable, so the inversion happening before
measurement is load-bearing rather than cosmetic.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from platevision import regression

KEYS = ["energy", "mass"]


def test_mae_uses_the_median_quantile():
    """MAE against the 5th or 95th percentile would be measuring the wrong output."""
    predictions = torch.tensor([[[0.0, 100.0, 500.0], [0.0, 50.0, 200.0]]])
    targets = torch.tensor([[110.0, 45.0]])

    mae = regression.per_target_mae(predictions, targets, KEYS, median_index=1)

    assert mae["energy"] == pytest.approx(10.0)
    assert mae["mass"] == pytest.approx(5.0)


def test_mae_is_zero_for_perfect_predictions():
    predictions = torch.tensor([[[0.0, 200.0, 400.0]]])
    targets = torch.tensor([[200.0]])
    assert regression.per_target_mae(predictions, targets, ["energy"], 1)["energy"] == 0.0


def test_mape_is_relative():
    predictions = torch.tensor([[[0.0, 110.0, 500.0]]])
    targets = torch.tensor([[100.0]])
    assert regression.per_target_mape(predictions, targets, ["energy"], 1)[
        "energy"
    ] == pytest.approx(0.1)


def test_mape_skips_near_zero_targets():
    """A dish labelled near-zero calories would contribute an unbounded term and make the
    average meaningless."""
    predictions = torch.tensor([[[0.0, 100.0, 200.0]], [[0.0, 5.0, 10.0]]])
    targets = torch.tensor([[100.0], [0.0]])

    mape = regression.per_target_mape(predictions, targets, ["energy"], 1)

    assert mape["energy"] == pytest.approx(0.0)


def test_mape_is_nan_when_every_target_is_unusable():
    predictions = torch.tensor([[[0.0, 5.0, 10.0]]])
    targets = torch.tensor([[0.0]])
    assert math.isnan(regression.per_target_mape(predictions, targets, ["energy"], 1)["energy"])


def test_coverage_uses_the_outer_quantiles():
    predictions = torch.tensor([[[10.0, 100.0, 200.0]], [[10.0, 100.0, 200.0]]])
    targets = torch.tensor([[50.0], [500.0]])

    coverage = regression.per_target_coverage(predictions, targets, ["energy"])

    assert coverage["energy"] == pytest.approx(0.5)


def test_coverage_is_reported_per_target():
    predictions = torch.tensor([[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]]])
    targets = torch.tensor([[5.0, 99.0]])

    coverage = regression.per_target_coverage(predictions, targets, KEYS)

    assert coverage["energy"] == pytest.approx(1.0)
    assert coverage["mass"] == pytest.approx(0.0)


def test_crossing_rate_counts_disordered_predictions():
    """Measured before sorting. Sorting first would erase the diagnostic completely."""
    predictions = torch.tensor(
        [
            [[1.0, 5.0, 9.0]],  # ordered
            [[9.0, 5.0, 1.0]],  # reversed
            [[1.0, 9.0, 5.0]],  # partially crossed
            [[0.0, 1.0, 2.0]],  # ordered
        ]
    )
    assert regression.crossings(predictions) == pytest.approx(0.5)


def test_crossing_rate_is_zero_when_all_ordered():
    predictions = torch.tensor([[[1.0, 2.0, 3.0]], [[0.0, 5.0, 10.0]]])
    assert regression.crossings(predictions) == 0.0


def test_crossing_rate_is_one_when_all_disordered():
    predictions = torch.tensor([[[3.0, 2.0, 1.0]], [[10.0, 5.0, 0.0]]])
    assert regression.crossings(predictions) == pytest.approx(1.0)


def test_equal_quantiles_do_not_count_as_crossed():
    """A collapsed but ordered interval is a modelling problem, not a sorting bug."""
    predictions = torch.tensor([[[5.0, 5.0, 5.0]]])
    assert regression.crossings(predictions) == 0.0


# --- result formatting --------------------------------------------------------------


def test_result_reports_real_units_in_its_summary():
    result = regression.RegressionResult(
        epoch=3,
        split="val",
        loss=0.31,
        seconds=1.0,
        mae={"energy": 84.2},
        mape={"energy": 0.27},
        coverage={"energy": 0.89},
        crossing_rate=0.004,
    )
    line = result.format()

    assert "84.2" in line
    assert "27.0%" in line
    assert "89.0%" in line
    assert "0.40%" in line


def test_train_results_show_the_learning_rate_not_coverage():
    result = regression.RegressionResult(epoch=0, split="train", loss=0.5, seconds=1.0, lr=3e-4)
    line = result.format()
    assert "lr" in line
    assert "cover" not in line


def test_result_serialises_for_history():
    result = regression.RegressionResult(
        epoch=0, split="val", loss=0.5, seconds=1.0, mae={"energy": 10.0}
    )
    payload = result.as_dict()
    assert payload["mae"]["energy"] == 10.0
    assert payload["split"] == "val"


class TestEndless:
    """Food-101 has 75,750 training images against Nutrition5k's 2,424, so the two loaders
    cannot be zipped: one epoch of the smaller set is a rounding error of the larger."""

    def test_it_restarts_rather_than_stopping(self):
        batches = list(itertools.islice(regression.endless([1, 2, 3]), 7))
        assert batches == [1, 2, 3, 1, 2, 3, 1]

    def test_it_keeps_drawing_new_data_across_restarts(self):
        # The point of cycling rather than truncating: over many nutrition epochs the
        # classification set is sampled through rather than the first N images every time.
        seen = list(itertools.islice(regression.endless(range(5)), 12))
        assert set(seen) == set(range(5))

    def test_an_empty_loader_raises_instead_of_spinning(self):
        # Without this the generator loops at full speed producing nothing, which presents
        # as a hang rather than as an empty dataset.
        with pytest.raises(ValueError, match="no batches"):
            next(regression.endless([]))
