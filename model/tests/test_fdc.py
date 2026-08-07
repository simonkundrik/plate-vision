"""Tests for USDA FoodData Central parsing.

The name match is the load-bearing part. FNDDS has no McChicken record, so a search for one
returns "Cheeseburger (McDonalds)" as its best Survey match. Taking the top result labels
McChicken photos with cheeseburger calories: a real, plausible number attached to the wrong
food, which nothing downstream can detect.
"""

from __future__ import annotations

import pytest

from platevision import fdc


def food(
    description="Big Mac (McDonalds)",
    kcal=261.0,
    data_type=fdc.PREFERRED_DATA_TYPE,
    measures=None,
    fdc_id=1,
):
    return {
        "fdcId": fdc_id,
        "description": description,
        "dataType": data_type,
        "foodNutrients": [
            {"nutrientName": "Energy", "unitName": "KCAL", "value": kcal},
            {"nutrientName": "Energy", "unitName": "kJ", "value": kcal * 4.184},
        ],
        "foodMeasures": measures
        if measures is not None
        else [
            {"disseminationText": "Quantity not specified", "gramWeight": 205},
            {"disseminationText": "1 McDonald's Big Mac", "gramWeight": 205},
        ],
    }


# --- energy extraction --------------------------------------------------------------


def test_energy_prefers_kcal_over_kilojoules():
    """Both nutrients are named "Energy". Taking the first match would return kJ."""
    assert fdc.energy_per_100g(food(kcal=261.0)) == pytest.approx(261.0)


def test_energy_is_none_when_absent():
    assert fdc.energy_per_100g({"foodNutrients": []}) is None


# --- name matching ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "item", "expected"),
    [
        ("Big Mac (McDonalds)", "Big Mac", True),
        ("Whopper (Burger King)", "Whopper", True),
        ("Cheeseburger (McDonalds)", "cheeseburger", True),
        # The real failures observed against the live API.
        ("Cheeseburger (McDonalds)", "McChicken sandwich", False),
        ("Hamburger (McDonalds)", "Filet-O-Fish", False),
        ("Cheeseburger (McDonalds)", "Quarter Pounder with cheese", False),
    ],
)
def test_name_matching(description, item, expected):
    assert fdc.matches_item(description, item) is expected


def test_a_shared_brand_does_not_rescue_a_mismatched_item():
    """Both strings contain "McDonalds", and the match still has to fail on the item.

    The protection lives in what callers pass, not inside the matcher: SEED_ITEMS keeps
    brand and item separate so the brand is never handed in as the thing to match. If a
    query string were passed whole, every McDonald's record would match every McDonald's
    item.
    """
    assert not fdc.matches_item("Cheeseburger (McDonalds)", "McChicken sandwich")

    # A two-token query is where the brand can tip the balance on its own: one match out
    # of two clears a 0.5 threshold with no agreement on the food at all. This is the
    # concrete reason brand and item are stored separately rather than as one string.
    assert fdc.matches_item("Cheeseburger (McDonalds)", "fries McDonalds")


def test_matching_ignores_case_and_punctuation():
    assert fdc.matches_item("MCDONALD'S, BIG MAC", "big mac")


def test_empty_item_never_matches():
    assert not fdc.matches_item("Big Mac (McDonalds)", "")


# --- portion selection --------------------------------------------------------------


def test_portion_prefers_the_measure_naming_the_item():
    """The Big Mac record also lists a Grand Mac at 315g. Taking the first or the largest
    would label photos with a different product's calories."""
    measures = [
        {"disseminationText": "Quantity not specified", "gramWeight": 205},
        {"disseminationText": "1 McDonald's Grand Mac", "gramWeight": 315},
        {"disseminationText": "1 McDonald's Big Mac", "gramWeight": 205},
    ]
    grams, label = fdc.choose_portion(food(measures=measures), "Big Mac")
    assert grams == 205
    assert "Big Mac" in label


def test_portion_falls_back_to_the_standard_serving():
    measures = [{"disseminationText": "Quantity not specified", "gramWeight": 180}]
    grams, label = fdc.choose_portion(food(measures=measures), "something unrelated")
    assert grams == 180
    assert label == fdc.FALLBACK_MEASURE


def test_portion_is_none_without_usable_measures():
    assert fdc.choose_portion(food(measures=[]), "Big Mac") is None


def test_zero_weight_measures_are_ignored():
    measures = [
        {"disseminationText": "1 Big Mac", "gramWeight": 0},
        {"disseminationText": "Quantity not specified", "gramWeight": 205},
    ]
    grams, _ = fdc.choose_portion(food(measures=measures), "Big Mac")
    assert grams == 205


# --- end to end ---------------------------------------------------------------------


def test_parse_computes_calories_per_item():
    """261 kcal per 100 g against a 205 g portion is 535 kcal."""
    item = fdc.parse_search({"foods": [food()]}, "Big Mac McDonalds", "Big Mac")

    assert item is not None
    assert item.kcal_per_item == pytest.approx(535.0, abs=0.5)
    assert item.gram_weight == 205


def test_parse_rejects_a_mismatched_record():
    """The regression this file exists for."""
    payload = {"foods": [food(description="Cheeseburger (McDonalds)")]}
    assert fdc.parse_search(payload, "McChicken sandwich McDonalds", "McChicken sandwich") is None


def test_parse_skips_records_without_portions():
    """SR Legacy carries the nutrients but no gram weights, so per-item calories are not
    derivable from it no matter how well the name matches."""
    payload = {
        "foods": [
            food(description="MCDONALD'S, BIG MAC", data_type="SR Legacy", measures=[]),
        ]
    }
    assert fdc.parse_search(payload, "Big Mac McDonalds", "Big Mac") is None


@pytest.mark.parametrize("kcal_per_100g", [1.0, 5000.0])
def test_implausible_totals_are_rejected(kcal_per_100g):
    """Below the floor is a condiment packet or a parse failure; above the ceiling is not a
    plated dish. Either would distort a benchmark rather than inform it."""
    payload = {"foods": [food(kcal=kcal_per_100g)]}
    assert fdc.parse_search(payload, "Big Mac McDonalds", "Big Mac") is None


def test_parse_returns_none_on_an_empty_response():
    assert fdc.parse_search({}, "Big Mac McDonalds", "Big Mac") is None


def test_parse_records_provenance():
    """Every figure needs to be traceable to the record it came from."""
    item = fdc.parse_search({"foods": [food(fdc_id=2706916)]}, "Big Mac McDonalds", "Big Mac")
    assert item.fdc_id == 2706916
    assert item.data_type == fdc.PREFERRED_DATA_TYPE
    assert item.description


# --- brand verification --------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "brand", "expected"),
    [
        ("Big Mac (McDonalds)", "McDonalds", True),
        ("Whopper (Burger King)", "Burger King", True),
        # The generic records that were silently passing as branded ground truth.
        ("Chicken nuggets, NFS", "Wendys", False),
        ("Chicken fillet sandwich, NFS", "Chick-fil-A", False),
        ("Coleslaw", "KFC", False),
        ("Croissant", "Starbucks", False),
        ("Soup, broccoli cheese", "Panera", False),
        ("Burrito bowl, chicken", "Chipotle", False),
    ],
)
def test_brand_verification(description, brand, expected):
    assert fdc.mentions_brand(description, brand) is expected


def test_nfs_marker_disqualifies_even_with_a_brand_match():
    """ "Not Further Specified" means a category average. A branded name appearing in one
    would still not make it that chain's product."""
    assert not fdc.mentions_brand("McDonalds hamburger, NFS", "McDonalds")


def test_generic_records_are_rejected_when_the_brand_is_required():
    """A generic croissant is about 170 kcal against a Starbucks one nearer 260. Labelling
    Starbucks photos with the generic figure is roughly a 50 percent error presented as
    ground truth."""
    payload = {"foods": [food(description="Croissant")]}
    assert fdc.parse_search(payload, "croissant Starbucks", "croissant", "Starbucks", True) is None


def test_generic_records_are_kept_but_flagged_when_allowed():
    payload = {"foods": [food(description="Croissant")]}
    item = fdc.parse_search(payload, "croissant Starbucks", "croissant", "Starbucks", False)

    assert item is not None
    assert item.brand_verified is False


def test_brand_specific_records_are_flagged_verified():
    item = fdc.parse_search({"foods": [food()]}, "Big Mac McDonalds", "Big Mac", "McDonalds", True)
    assert item.brand_verified is True


# --- the seed list ------------------------------------------------------------------


def test_seed_items_separate_brand_from_item():
    assert all(isinstance(entry, tuple) and len(entry) == 2 for entry in fdc.SEED_ITEMS)


def test_seed_items_encode_no_nutrition_claims():
    """Item names only. A remembered calorie figure written here would be inventing the
    ground truth the set exists to provide."""
    for brand, item in fdc.SEED_ITEMS:
        assert not any(character.isdigit() for character in f"{brand}{item}")


def test_query_puts_the_item_first():
    assert fdc.query_for("McDonalds", "Big Mac") == "Big Mac McDonalds"
