"""Ingredient supervision.

Aimed at the density half of the calorie error, which the oracle decomposition put at 12.8%
and which geometry cannot fix: a plate of arugula and the same volume dressed in olive oil
look different and are nothing alike calorically.
"""

from __future__ import annotations

import pytest
import torch

from platevision import ingredients


class TestBuildVocabulary:
    def test_keeps_ingredients_with_enough_support(self):
        dishes = [["olive oil"], ["olive oil"], ["saffron"]]
        assert ingredients.build_vocabulary(dishes, min_count=2) == ["olive oil"]

    def test_counts_per_dish_not_per_mention(self):
        # A recipe listing salt twice does not make salt twice as common.
        assert ingredients.build_vocabulary([["salt", "salt"]], min_count=2) == []

    def test_is_alphabetical_not_frequency_ordered(self):
        # Frequency order is not stable: more training data reshuffles it, and a reshuffled
        # vocabulary silently relabels every dimension of a saved head.
        dishes = [["zucchini"] * 1, ["apple"], ["apple"], ["apple"]]
        assert ingredients.build_vocabulary(dishes, min_count=1) == ["apple", "zucchini"]

    def test_returns_empty_when_nothing_clears_the_threshold(self):
        assert ingredients.build_vocabulary([["rare"]], min_count=5) == []


class TestMultiHot:
    def test_marks_present_ingredients(self):
        vocab = ["garlic", "olive oil", "salt"]
        assert ingredients.multi_hot(["olive oil"], vocab).tolist() == [0.0, 1.0, 0.0]

    def test_drops_ingredients_outside_the_vocabulary(self):
        # A shared "other" dimension would ask the head to predict the presence of an
        # arbitrary union of unrelated foods, which is not a coherent target.
        vocab = ["garlic"]
        assert ingredients.multi_hot(["saffron", "garlic"], vocab).tolist() == [1.0]

    def test_empty_dish_encodes_to_zeros(self):
        assert ingredients.multi_hot([], ["a", "b"]).sum().item() == 0.0

    def test_length_always_matches_the_vocabulary(self):
        assert ingredients.multi_hot(["x"], ["a", "b", "c"]).shape == (3,)


class TestPositiveWeight:
    def test_rare_ingredients_weigh_more(self):
        dishes = [["common"]] * 9 + [["common", "rare"]]
        vocab = ["common", "rare"]
        weights = ingredients.positive_weight(dishes, vocab)
        assert weights[1] > weights[0]

    def test_is_capped(self):
        # An ingredient in 1% of dishes would otherwise get a weight near 100 and dominate
        # the gradient on its own.
        dishes = [["common"]] * 999 + [["rare"]]
        weights = ingredients.positive_weight(dishes, ["rare"], cap=20.0)
        assert weights[0] == pytest.approx(20.0)

    def test_absent_ingredient_does_not_divide_by_zero(self):
        weights = ingredients.positive_weight([["a"]], ["never_seen"])
        assert torch.isfinite(weights).all()


class TestAuxiliaryHeadStaysOffTheExportedGraph:
    """The contract declares two outputs. The ingredient head must not become a third."""

    @staticmethod
    def model(num_ingredients: int):
        from platevision.models import NutritionModel

        return NutritionModel(
            "efficientnet_b0",
            num_targets=5,
            num_quantiles=3,
            pretrained=False,
            num_ingredients=num_ingredients,
        )

    def test_forward_returns_only_quantiles(self):
        out = self.model(7)(torch.zeros(2, 3, 224, 224))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 5, 3)

    def test_forward_with_ingredients_returns_both(self):
        quantiles, logits = self.model(7).forward_with_ingredients(torch.zeros(2, 3, 224, 224))
        assert quantiles.shape == (2, 5, 3)
        assert logits.shape == (2, 7)

    def test_no_head_means_no_logits(self):
        quantiles, logits = self.model(0).forward_with_ingredients(torch.zeros(1, 3, 224, 224))
        assert quantiles.shape == (1, 5, 3)
        assert logits is None

    def test_the_head_does_not_change_the_quantile_output(self):
        # If adding auxiliary supervision changed the shape or meaning of the shipped
        # output, every downstream consumer would break silently.
        torch.manual_seed(0)
        with_head = self.model(7)
        torch.manual_seed(0)
        without = self.model(0)

        x = torch.zeros(1, 3, 224, 224)
        assert with_head(x).shape == without(x).shape
