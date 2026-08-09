from __future__ import annotations

import pytest
import torch

from platevision import scale


class TestCorrectForScale:
    def test_a_zoomed_in_photo_has_its_prediction_brought_down(self):
        # The model saw more food than was there.
        predictions = torch.full((4, 1, 3), 800.0)
        corrected = scale.correct_for_scale(
            predictions, torch.full((4,), 2.0), scale.VOLUME_EXPONENT
        )
        assert corrected.max().item() == pytest.approx(100.0)

    def test_a_zoomed_out_photo_has_its_prediction_raised(self):
        predictions = torch.full((2, 1, 3), 100.0)
        corrected = scale.correct_for_scale(
            predictions, torch.full((2,), 0.5), scale.VOLUME_EXPONENT
        )
        assert corrected.min().item() == pytest.approx(800.0)

    def test_the_reference_distance_changes_nothing(self):
        predictions = torch.rand(5, 2, 3) * 100
        corrected = scale.correct_for_scale(predictions, torch.ones(5), scale.VOLUME_EXPONENT)
        torch.testing.assert_close(corrected, predictions)

    def test_the_whole_interval_moves_together(self):
        # Correcting the median alone would leave an interval that no longer contains its own
        # point estimate, which is worse than not correcting at all.
        predictions = torch.tensor([[[50.0, 100.0, 200.0]]])
        corrected = scale.correct_for_scale(predictions, torch.tensor([2.0]), 2.0)

        assert corrected[0, 0, 0] < corrected[0, 0, 1] < corrected[0, 0, 2]
        ratios = corrected[0, 0] / predictions[0, 0]
        torch.testing.assert_close(ratios, torch.full((3,), 0.25))

    def test_each_sample_gets_its_own_factor(self):
        predictions = torch.full((2, 1, 1), 100.0)
        corrected = scale.correct_for_scale(predictions, torch.tensor([1.0, 2.0]), 2.0)
        assert corrected[0].item() == pytest.approx(100.0)
        assert corrected[1].item() == pytest.approx(25.0)

    def test_area_and_volume_exponents_differ(self):
        predictions = torch.full((1, 1, 1), 1000.0)
        factors = torch.tensor([2.0])
        area = scale.correct_for_scale(predictions, factors, scale.AREA_EXPONENT)
        volume = scale.correct_for_scale(predictions, factors, scale.VOLUME_EXPONENT)
        assert volume.item() < area.item()

    def test_rejects_a_factor_per_target_instead_of_per_sample(self):
        with pytest.raises(ValueError, match="one factor per sample"):
            scale.correct_for_scale(torch.ones(3, 1, 3), torch.ones(3, 1), 2.0)

    def test_rejects_the_wrong_number_of_factors(self):
        with pytest.raises(ValueError, match="factors for"):
            scale.correct_for_scale(torch.ones(3, 1, 3), torch.ones(2), 2.0)

    def test_rejects_a_non_positive_factor(self):
        # A detector reporting zero scale would otherwise divide the prediction to infinity.
        with pytest.raises(ValueError, match="must be positive"):
            scale.correct_for_scale(torch.ones(2, 1, 3), torch.tensor([1.0, 0.0]), 2.0)


class TestFitScaleExponent:
    def test_recovers_the_exponent_it_was_generated_with(self):
        factors = torch.linspace(0.5, 2.0, 50)
        actual = torch.full((50,), 200.0)
        predicted = actual * factors.pow(3.0)
        assert scale.fit_scale_exponent(predicted, actual, factors) == pytest.approx(3.0)

    def test_recovers_an_area_exponent(self):
        factors = torch.linspace(0.5, 2.0, 50)
        actual = torch.full((50,), 200.0)
        assert scale.fit_scale_exponent(
            actual * factors.pow(2.0), actual, factors
        ) == pytest.approx(2.0)

    def test_a_model_indifferent_to_zoom_fits_zero(self):
        # The result that would cancel the route: knowing the scale cannot correct an error
        # that does not depend on scale.
        factors = torch.linspace(0.5, 2.0, 50)
        actual = torch.full((50,), 200.0)
        assert scale.fit_scale_exponent(actual, actual, factors) == pytest.approx(0.0)

    def test_unbiased_noise_does_not_move_the_exponent_much(self):
        generator = torch.Generator().manual_seed(0)
        factors = torch.linspace(0.5, 2.0, 400)
        actual = torch.full((400,), 200.0)
        noisy = actual * factors.pow(3.0) * (torch.randn(400, generator=generator) * 0.1).exp()
        assert scale.fit_scale_exponent(noisy, actual, factors) == pytest.approx(3.0, abs=0.15)

    def test_refuses_a_set_with_no_zoom_in_it(self):
        ones = torch.ones(10)
        with pytest.raises(ValueError, match="no zoom"):
            scale.fit_scale_exponent(ones * 200, ones * 200, ones)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            scale.fit_scale_exponent(torch.ones(5), torch.ones(5), torch.ones(4))


class TestPerturbFactors:
    def test_a_perfect_detector_returns_the_truth(self):
        factors = torch.tensor([0.7, 1.0, 1.6])
        torch.testing.assert_close(scale.perturb_factors(factors, 0.0), factors)

    def test_noise_is_multiplicative_not_additive(self):
        # A 10 percent error on a 2.0x scale is a bigger absolute move than on a 0.5x scale,
        # because scale is a ratio. Additive noise would make one of them nearly exact.
        generator = torch.Generator().manual_seed(3)
        small = scale.perturb_factors(torch.full((2000,), 0.5), 0.1, generator)
        generator = torch.Generator().manual_seed(3)
        large = scale.perturb_factors(torch.full((2000,), 2.0), 0.1, generator)

        assert (large / 2.0).std().item() == pytest.approx((small / 0.5).std().item(), rel=1e-4)
        assert large.std().item() > small.std().item()

    def test_the_error_is_the_size_it_claims(self):
        generator = torch.Generator().manual_seed(1)
        perturbed = scale.perturb_factors(torch.full((20000,), 1.3), 0.1, generator)
        assert (perturbed / 1.3).log().std().item() == pytest.approx(0.1, abs=0.005)

    def test_perturbed_factors_stay_positive(self):
        # correct_for_scale rejects a non-positive factor, so a noise model that could
        # produce one would turn a plausible detector error into a crash.
        generator = torch.Generator().manual_seed(2)
        perturbed = scale.perturb_factors(torch.full((5000,), 1.0), 0.4, generator)
        assert (perturbed > 0).all()

    def test_rejects_a_negative_error(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            scale.perturb_factors(torch.ones(3), -0.1)
