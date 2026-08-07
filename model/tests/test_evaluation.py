"""Tests for the nutrition evaluation metrics.

The reason this module exists: MAPE read 127 percent on a real run whose typical error was
about 100 kcal against a 208 kcal median. That number describes the metric, not the model,
and quoting it would misrepresent the work in both directions depending on the audience.
"""

from __future__ import annotations

import pytest
import torch

from platevision import evaluation

QUANTILES = [0.05, 0.5, 0.95]
KEYS = ["energy"]


def make(median, truth, lower=None, upper=None):
    """One-target predictions with an explicit interval."""
    median = torch.as_tensor(median, dtype=torch.float32)
    lower = torch.as_tensor(lower if lower is not None else median - 50.0, dtype=torch.float32)
    upper = torch.as_tensor(upper if upper is not None else median + 50.0, dtype=torch.float32)
    predictions = torch.stack([lower, median, upper], dim=-1).unsqueeze(1)
    return predictions, torch.as_tensor(truth, dtype=torch.float32).unsqueeze(1)


# --- the metric problem this module was built for -----------------------------------


def test_mape_is_destroyed_by_one_tiny_target():
    """The observed failure: nine good predictions and one 2 kcal dish."""
    median = [200.0] * 9 + [100.0]
    truth = [210.0] * 9 + [2.0]
    predictions, targets = make(median, truth)
    index = QUANTILES.index(0.5)

    mape = evaluation.mean_absolute_percentage_error(predictions, targets, index)[0].item()
    assert mape > 4.0  # over 400 percent, from a single sample


def test_median_ape_is_not():
    median = [200.0] * 9 + [100.0]
    truth = [210.0] * 9 + [2.0]
    predictions, targets = make(median, truth)
    index = QUANTILES.index(0.5)

    median_ape = evaluation.median_absolute_percentage_error(predictions, targets, index)[0]
    assert median_ape.item() == pytest.approx(10.0 / 210.0, rel=1e-3)


def test_smape_is_bounded_even_against_a_zero_target():
    """Bounded at 2 by construction, because the prediction is in the denominator too."""
    predictions, targets = make([100.0], [0.0])
    index = QUANTILES.index(0.5)
    assert evaluation.symmetric_mape(predictions, targets, index)[0].item() == pytest.approx(2.0)


def test_smape_never_exceeds_two():
    torch.manual_seed(0)
    predictions, targets = make(torch.rand(200) * 1000, torch.rand(200) * 1000)
    index = QUANTILES.index(0.5)
    assert evaluation.symmetric_mape(predictions, targets, index)[0].item() <= 2.0


def test_all_three_percentage_metrics_agree_when_targets_are_well_scaled():
    """They only diverge because of small denominators, which is the point."""
    predictions, targets = make([200.0] * 20, [220.0] * 20)
    index = QUANTILES.index(0.5)

    mape = evaluation.mean_absolute_percentage_error(predictions, targets, index)[0].item()
    med = evaluation.median_absolute_percentage_error(predictions, targets, index)[0].item()

    assert mape == pytest.approx(med, rel=1e-4)


# --- basic errors ------------------------------------------------------------------


def test_mae_and_median_ae_differ_under_skew():
    predictions, targets = make([100.0] * 9 + [100.0], [100.0] * 9 + [1000.0])
    errors = evaluation.absolute_errors(predictions, targets, QUANTILES.index(0.5))

    assert evaluation.mean_absolute_error(errors)[0].item() == pytest.approx(90.0)
    assert evaluation.median_absolute_error(errors)[0].item() == pytest.approx(0.0)


# --- calibration --------------------------------------------------------------------


def test_perfect_calibration_recovers_the_nominal_levels():
    """For a calibrated model the fraction of truths below the q-th quantile equals q."""
    torch.manual_seed(0)
    n = 20000
    truth = torch.randn(n)
    lower = torch.full((n,), torch.distributions.Normal(0, 1).icdf(torch.tensor(0.05)).item())
    median = torch.zeros(n)
    upper = torch.full((n,), torch.distributions.Normal(0, 1).icdf(torch.tensor(0.95)).item())

    predictions = torch.stack([lower, median, upper], dim=-1).unsqueeze(1)
    targets = truth.unsqueeze(1)

    calibration = evaluation.quantile_calibration(predictions, targets, QUANTILES)[0]
    assert calibration[0].item() == pytest.approx(0.05, abs=0.02)
    assert calibration[1].item() == pytest.approx(0.50, abs=0.02)
    assert calibration[2].item() == pytest.approx(0.95, abs=0.02)


def test_calibration_reveals_which_bound_is_wrong():
    """Coverage alone cannot: two bounds shifted the same way keep coverage intact while
    both quantiles are badly wrong."""
    truth = torch.linspace(0.0, 100.0, 1000)
    predictions = torch.stack(
        [torch.full_like(truth, -1000.0), truth, torch.full_like(truth, 50.0)], dim=-1
    ).unsqueeze(1)
    targets = truth.unsqueeze(1)

    calibration = evaluation.quantile_calibration(predictions, targets, QUANTILES)[0]

    assert calibration[0].item() == pytest.approx(0.0, abs=0.01)  # lower bound far too low
    assert calibration[2].item() < 0.6  # upper bound far too low


def test_calibration_rejects_a_level_count_mismatch():
    predictions, targets = make([100.0], [100.0])
    with pytest.raises(ValueError, match="quantile outputs"):
        evaluation.quantile_calibration(predictions, targets, [0.5])


def test_interval_width_is_reported():
    """Coverage alone is trivially gamed: zero to infinity covers everything."""
    predictions, targets = make([100.0] * 4, [100.0] * 4, lower=[0.0] * 4, upper=[300.0] * 4)
    assert evaluation.interval_width(predictions)[0].item() == pytest.approx(300.0)


# --- buckets and failures -----------------------------------------------------------


def test_buckets_partition_every_sample():
    torch.manual_seed(0)
    truth = torch.rand(200) * 1000
    predictions, targets = make(truth + 20.0, truth)

    buckets = evaluation.error_buckets(predictions, targets, QUANTILES.index(0.5), n_buckets=5)

    assert sum(b.count for b in buckets) == 200


def test_buckets_are_ordered_and_non_overlapping():
    torch.manual_seed(0)
    truth = torch.rand(200) * 1000
    predictions, targets = make(truth, truth)
    buckets = evaluation.error_buckets(predictions, targets, QUANTILES.index(0.5))
    assert all(a.upper <= b.lower + 1e-6 for a, b in zip(buckets, buckets[1:], strict=False))


def test_buckets_expose_error_growing_with_portion_size():
    """A single MAE hides this. It changes what you do about the problem."""
    truth = torch.tensor([10.0, 20.0, 30.0, 1000.0, 2000.0, 3000.0])
    median = torch.tensor([11.0, 21.0, 31.0, 1500.0, 2500.0, 3500.0])
    predictions, targets = make(median, truth)

    buckets = evaluation.error_buckets(predictions, targets, QUANTILES.index(0.5), n_buckets=2)

    assert buckets[0].mae < buckets[-1].mae


def test_empty_input_produces_no_buckets():
    predictions = torch.zeros(0, 1, 3)
    targets = torch.zeros(0, 1)
    assert evaluation.error_buckets(predictions, targets, 1) == []


def test_worst_cases_are_sorted_by_error():
    predictions, targets = make([100.0, 100.0, 100.0], [105.0, 300.0, 150.0])
    cases = evaluation.worst_cases(
        predictions, targets, QUANTILES.index(0.5), ["a", "b", "c"], top_n=3
    )

    assert [c.dish_id for c in cases] == ["b", "c", "a"]
    assert cases[0].absolute_error > cases[-1].absolute_error


def test_worst_cases_record_whether_the_interval_caught_it():
    """A large error inside the interval is a different failure from one outside it: the
    first is honest uncertainty, the second is a miscalibrated model."""
    predictions, targets = make([100.0], [140.0], lower=[0.0], upper=[200.0])
    case = evaluation.worst_cases(predictions, targets, QUANTILES.index(0.5), ["a"])[0]
    assert case.inside_interval


def test_worst_cases_handles_fewer_samples_than_requested():
    predictions, targets = make([100.0], [200.0])
    assert len(evaluation.worst_cases(predictions, targets, 1, ["a"], top_n=50)) == 1


def test_worst_cases_tolerates_missing_dish_ids():
    predictions, targets = make([100.0, 100.0], [200.0, 300.0])
    cases = evaluation.worst_cases(predictions, targets, 1, [], top_n=2)
    assert all(c.dish_id == "" for c in cases)


# --- the report ---------------------------------------------------------------------


def test_report_collects_every_metric():
    torch.manual_seed(0)
    truth = torch.rand(100) * 500 + 50
    predictions, targets = make(truth + 30.0, truth)

    report = evaluation.build_report(
        predictions, targets, target_keys=KEYS, quantiles=QUANTILES, dish_ids=["x"] * 100
    )

    assert report.count == 100
    assert set(report.mae) == {"energy"}
    assert len(report.calibration["energy"]) == 3
    assert report.buckets
    assert report.worst


def test_report_headline_names_the_metric_to_quote():
    torch.manual_seed(0)
    truth = torch.rand(50) * 500 + 50
    predictions, targets = make(truth + 30.0, truth)

    line = evaluation.build_report(
        predictions, targets, target_keys=KEYS, quantiles=QUANTILES
    ).headline()

    assert "median APE" in line
    assert "distorted by small denominators" in line


def test_report_serialises():
    predictions, targets = make([100.0] * 10, [110.0] * 10)
    report = evaluation.build_report(predictions, targets, target_keys=KEYS, quantiles=QUANTILES)
    payload = report.as_dict()
    assert payload["count"] == 10
    assert "median_ape" in payload
