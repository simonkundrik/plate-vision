"""Conformal calibration: the fix for intervals that do not cover what they claim."""

from __future__ import annotations

import math

import pytest
import torch

from platevision.conformal import (
    ConformalCalibration,
    apply_offset,
    conformal_offset,
    conformity_scores,
)


def intervals(low, median, high) -> torch.Tensor:
    """Shape (n, 1, 3): one target, three quantiles."""
    return torch.tensor([[[lo, mid, hi]] for lo, mid, hi in zip(low, median, high, strict=True)])


class TestConformityScores:
    def test_negative_when_the_truth_sits_inside(self):
        # Negative scores are load-bearing, not a curiosity: they are what lets a
        # well-calibrated model's intervals be narrowed rather than only widened.
        preds = intervals([10.0], [20.0], [30.0])
        scores = conformity_scores(preds, torch.tensor([[20.0]]))
        assert scores.item() == pytest.approx(-10.0)

    def test_measures_the_miss_below_the_lower_bound(self):
        preds = intervals([10.0], [20.0], [30.0])
        assert conformity_scores(preds, torch.tensor([[4.0]])).item() == pytest.approx(6.0)

    def test_measures_the_miss_above_the_upper_bound(self):
        preds = intervals([10.0], [20.0], [30.0])
        assert conformity_scores(preds, torch.tensor([[38.0]])).item() == pytest.approx(8.0)

    def test_zero_exactly_on_a_bound(self):
        preds = intervals([10.0], [20.0], [30.0])
        assert conformity_scores(preds, torch.tensor([[30.0]])).item() == pytest.approx(0.0)

    def test_rejects_predictions_of_the_wrong_rank(self):
        with pytest.raises(ValueError, match="targets, quantiles"):
            conformity_scores(torch.zeros(4, 3), torch.zeros(4, 3))

    def test_rejects_mismatched_targets(self):
        with pytest.raises(ValueError, match="do not match"):
            conformity_scores(torch.zeros(4, 5, 3), torch.zeros(4, 2))


class TestConformalOffset:
    def test_uses_the_finite_sample_rank(self):
        # ceil((n+1)(1-alpha)) rather than the plain quantile. The +1 is what turns an
        # asymptotic statement into a guarantee, and at n=99 it is a whole index.
        scores = torch.arange(99, dtype=torch.float32).unsqueeze(1)
        offset = conformal_offset(scores, alpha=0.10)
        assert offset.item() == pytest.approx(math.ceil(100 * 0.9) - 1)

    def test_is_per_target(self):
        scores = torch.stack([torch.arange(20.0), torch.arange(20.0) * 10], dim=1)
        offsets = conformal_offset(scores, alpha=0.10)
        assert offsets.shape == (2,)
        assert offsets[1] > offsets[0]

    def test_clamps_when_the_sample_cannot_certify_the_level(self):
        # With 5 points a 99% level would need rank 6. Falling back to the largest observed
        # miss is the most conservative honest answer available.
        scores = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        assert conformal_offset(scores, alpha=0.01).item() == pytest.approx(5.0)

    def test_rejects_a_degenerate_level(self):
        with pytest.raises(ValueError, match="alpha"):
            conformal_offset(torch.zeros(10, 1), alpha=0.0)

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            conformal_offset(torch.zeros(1, 1))


class TestApplyOffset:
    def test_widens_both_bounds(self):
        preds = intervals([10.0], [20.0], [30.0])
        widened = apply_offset(preds, torch.tensor([5.0]))
        assert widened[0, 0, 0].item() == pytest.approx(5.0)
        assert widened[0, 0, -1].item() == pytest.approx(35.0)

    def test_leaves_the_median_alone(self):
        # Conformal prediction says nothing about the point estimate, and the median was the
        # one output already well calibrated. Shifting it would degrade it for nothing.
        preds = intervals([10.0], [20.0], [30.0])
        assert apply_offset(preds, torch.tensor([5.0]))[0, 0, 1].item() == pytest.approx(20.0)

    def test_does_not_mutate_its_input(self):
        preds = intervals([10.0], [20.0], [30.0])
        apply_offset(preds, torch.tensor([5.0]))
        assert preds[0, 0, 0].item() == pytest.approx(10.0)


class TestCoverageGuarantee:
    """The property the module exists for, measured rather than asserted from theory."""

    @staticmethod
    def sample(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """An overconfident model: intervals far too narrow for the real noise."""
        truth = torch.randn(n, 1, generator=generator) * 100 + 400
        centre = truth + torch.randn(n, 1, generator=generator) * 40
        preds = torch.stack([centre - 15, centre, centre + 15], dim=-1)
        return preds, truth

    def test_raises_coverage_to_the_requested_level(self):
        generator = torch.Generator().manual_seed(0)

        covered = []
        for _ in range(30):
            cal_preds, cal_truth = self.sample(400, generator)
            ev_preds, ev_truth = self.sample(400, generator)

            calibration = ConformalCalibration.fit(cal_preds, cal_truth, ["energy"], alpha=0.10)
            adjusted = calibration.apply(ev_preds)

            inside = (ev_truth >= adjusted[..., 0]) & (ev_truth <= adjusted[..., -1])
            covered.append(inside.float().mean().item())

        # Marginal over calibration draws, so a single split varies. The mean is the claim.
        assert sum(covered) / len(covered) == pytest.approx(0.90, abs=0.03)

    def test_the_uncalibrated_model_falls_far_short(self):
        # Without this the test above could pass on a model that was already fine.
        generator = torch.Generator().manual_seed(0)
        preds, truth = self.sample(2000, generator)
        inside = (truth >= preds[..., 0]) & (truth <= preds[..., -1])
        assert inside.float().mean().item() < 0.5


class TestSerialisation:
    def test_round_trips(self):
        original = ConformalCalibration(
            keys=["energy", "mass"], offsets=[31.7, 28.1], alpha=0.1, calibration_size=253
        )
        assert ConformalCalibration.from_dict(original.to_dict()) == original

    def test_refuses_predictions_with_the_wrong_target_count(self):
        # Applying a five-target calibration to a two-target model would widen the wrong
        # quantities and produce intervals in the wrong units.
        calibration = ConformalCalibration(["energy"], [10.0], 0.1, 100)
        with pytest.raises(ValueError, match="covers 1"):
            calibration.apply(torch.zeros(4, 5, 3))

    def test_records_how_much_data_backed_it(self):
        preds = intervals([1.0] * 50, [2.0] * 50, [3.0] * 50)
        calibration = ConformalCalibration.fit(preds, torch.full((50, 1), 2.0), ["energy"])
        assert calibration.calibration_size == 50
