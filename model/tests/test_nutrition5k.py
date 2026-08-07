"""Tests for Nutrition5k row parsing.

The row layout is the thing worth testing. A dish row is 6 header fields followed by 7
fields per ingredient, with no count field, which is not what the dataset README describes.
If that assumption is wrong, parsing does not crash: every nutrition label silently shifts.
"""

from __future__ import annotations

import pytest

from platevision import nutrition5k as n5k


def make_row(dish_id="dish_1", totals=(300.0, 193.0, 12.0, 28.0, 18.0), n_ingredients=1):
    row = [dish_id, *[f"{v:.6f}" for v in totals]]
    for i in range(n_ingredients):
        row += [f"ingr_{i:010d}", f"ingredient {i}", "10.0", "50.0", "1.0", "2.0", "3.0"]
    return row


def test_parses_single_ingredient_row():
    dish = n5k.parse_dish_row(make_row())
    assert dish.dish_id == "dish_1"
    assert dish.calories == pytest.approx(300.0)
    assert dish.mass_g == pytest.approx(193.0)
    assert len(dish.ingredients) == 1
    assert dish.ingredients[0].name == "ingredient 0"


def test_parses_the_widest_real_row_shape():
    """A 125-field row is 6 + 7*17. This shape appears in the real cafe1 metadata."""
    row = make_row(n_ingredients=17)
    assert len(row) == 125
    dish = n5k.parse_dish_row(row)
    assert len(dish.ingredients) == 17


@pytest.mark.parametrize("n_ingredients", [0, 1, 2, 3, 13, 17])
def test_field_count_arithmetic_holds(n_ingredients):
    row = make_row(n_ingredients=n_ingredients)
    assert len(row) == n5k.DISH_HEADER_FIELDS + n5k.INGREDIENT_FIELDS * n_ingredients
    assert len(n5k.parse_dish_row(row).ingredients) == n_ingredients


@pytest.mark.parametrize("extra", [1, 2, 3, 4, 5, 6])
def test_partial_ingredient_group_is_rejected(extra):
    """A truncated group means the layout assumption is wrong. It must not parse."""
    row = make_row(n_ingredients=2) + ["x"] * extra
    with pytest.raises(ValueError, match="not a multiple"):
        n5k.parse_dish_row(row)


def test_short_row_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        n5k.parse_dish_row(["dish_1", "1.0", "2.0"])


def test_non_numeric_total_is_rejected():
    row = make_row()
    row[1] = "not-a-number"
    with pytest.raises(ValueError, match="non-numeric"):
        n5k.parse_dish_row(row)


def test_non_numeric_ingredient_is_rejected():
    row = make_row()
    row[8] = "not-a-number"
    with pytest.raises(ValueError, match="bad ingredient"):
        n5k.parse_dish_row(row)


def test_calorie_disagreement_is_relative():
    row = make_row(totals=(100.0, 1.0, 1.0, 1.0, 1.0), n_ingredients=1)
    dish = n5k.parse_dish_row(row)
    assert dish.ingredient_calorie_sum == pytest.approx(50.0)
    assert dish.calorie_disagreement() == pytest.approx(0.5)


def test_calorie_disagreement_handles_zero_total():
    """Zero-calorie dishes are reported separately, not divided by."""
    row = make_row(totals=(0.0, 1.0, 1.0, 1.0, 1.0), n_ingredients=1)
    assert n5k.parse_dish_row(row).calorie_disagreement() == 0.0


def test_metadata_parsing_skips_blank_lines():
    text = "\n".join([",".join(make_row("dish_a")), "", ",".join(make_row("dish_b")), ""])
    dishes = n5k.parse_dish_metadata(text)
    assert set(dishes) == {"dish_a", "dish_b"}


def test_split_parsing_preserves_order_and_drops_blanks():
    assert n5k.parse_split_ids("dish_b\n\n  dish_a  \n\n") == ["dish_b", "dish_a"]


def test_urls_point_at_the_overhead_rgb_frame():
    url = n5k.rgb_url("dish_1556572657")
    assert url.startswith(n5k.GCS_BASE)
    assert url.endswith("/imagery/realsense_overhead/dish_1556572657/rgb.png")


def test_rgb_splits_are_used_not_depth_splits():
    """RGB-only training means the RGB splits, or evaluation runs on the wrong dishes."""
    assert all("rgb" in name for name in n5k.SPLIT_FILES)
    assert not any("depth" in name for name in n5k.SPLIT_FILES)
