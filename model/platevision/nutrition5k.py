"""Nutrition5k parsing and URL helpers.

Pure functions only, so the row-layout logic is testable without touching the network.
Downloading lives in ``model/data/download_nutrition5k.py``.

Dataset: https://github.com/google-research-datasets/Nutrition5k (CC BY 4.0)
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

GCS_BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"

# Row layout verified against the real CSVs, not the dataset README. The README
# documents a num_ingrs field between total_protein and the ingredient groups; that
# field does not exist in the published data. A row is:
#
#   [dish_id, calories, mass, fat, carb, protein] + 7 fields per ingredient
#
# Cross-check: a 125-field row is 6 + 7*17, i.e. seventeen ingredients.
DISH_HEADER_FIELDS = 6
INGREDIENT_FIELDS = 7

METADATA_FILES = ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv")

# RGB splits, not depth splits. This project trains on RGB only because a phone camera
# has no depth sensor, so using the depth splits would evaluate on a different set of
# dishes than the ones the model can actually be deployed against.
SPLIT_FILES = ("rgb_train_ids.txt", "rgb_test_ids.txt")


@dataclass(frozen=True, slots=True)
class Ingredient:
    ingredient_id: str
    name: str
    grams: float
    calories: float
    fat_g: float
    carb_g: float
    protein_g: float


@dataclass(frozen=True, slots=True)
class Dish:
    dish_id: str
    calories: float
    mass_g: float
    fat_g: float
    carb_g: float
    protein_g: float
    ingredients: tuple[Ingredient, ...]

    @property
    def ingredient_calorie_sum(self) -> float:
        return sum(i.calories for i in self.ingredients)

    def calorie_disagreement(self) -> float:
        """Relative gap between the stated total and the sum over ingredients.

        Returns 0.0 for a dish with no stated calories, which is itself reported
        separately rather than being silently treated as agreement.
        """
        if self.calories <= 0:
            return 0.0
        return abs(self.calories - self.ingredient_calorie_sum) / self.calories


def metadata_url(filename: str) -> str:
    return f"{GCS_BASE}/metadata/{filename}"


def split_url(filename: str) -> str:
    return f"{GCS_BASE}/dish_ids/splits/{filename}"


def rgb_url(dish_id: str) -> str:
    """Overhead RGB frame for one dish."""
    return f"{GCS_BASE}/imagery/realsense_overhead/{dish_id}/rgb.png"


def depth_url(dish_id: str) -> str:
    """Overhead raw depth frame for one dish, 16-bit millimetres.

    Not an inference input. A phone has no depth sensor, so feeding depth to the model
    would open a gap between training and deployment that no tuning closes.

    It is fetched because it makes two things possible that RGB alone does not. Integrated
    against the table plane it yields each dish's *volume*, which turns calorie density into
    a supervisable target: density is intensive and readable from appearance, while absolute
    mass is not observable from a single photograph at all. And it can be predicted as an
    auxiliary target, so the backbone is pushed to learn geometry while inference stays RGB.

    depth_color.png is the same data rendered for human eyes and is deliberately skipped.
    """
    return f"{GCS_BASE}/imagery/realsense_overhead/{dish_id}/depth_raw.png"


def parse_dish_row(row: list[str]) -> Dish:
    """Parse one metadata row into a :class:`Dish`.

    Raises ValueError on a row whose ingredient fields are not a whole multiple of
    :data:`INGREDIENT_FIELDS`, since a partial group means the layout assumption is
    wrong and every downstream nutrition label would be shifted.
    """
    if len(row) < DISH_HEADER_FIELDS:
        raise ValueError(f"row has {len(row)} fields, need at least {DISH_HEADER_FIELDS}")

    trailing = len(row) - DISH_HEADER_FIELDS
    if trailing % INGREDIENT_FIELDS:
        raise ValueError(
            f"dish {row[0]!r}: {trailing} ingredient fields is not a multiple of "
            f"{INGREDIENT_FIELDS}, so the row layout assumption is wrong"
        )

    dish_id = row[0]
    try:
        calories, mass, fat, carb, protein = (float(v) for v in row[1:DISH_HEADER_FIELDS])
    except ValueError as exc:
        raise ValueError(f"dish {dish_id!r}: non-numeric total: {exc}") from exc

    ingredients = []
    for start in range(DISH_HEADER_FIELDS, len(row), INGREDIENT_FIELDS):
        chunk = row[start : start + INGREDIENT_FIELDS]
        try:
            ingredients.append(
                Ingredient(
                    ingredient_id=chunk[0],
                    name=chunk[1],
                    grams=float(chunk[2]),
                    calories=float(chunk[3]),
                    fat_g=float(chunk[4]),
                    carb_g=float(chunk[5]),
                    protein_g=float(chunk[6]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"dish {dish_id!r}: bad ingredient at field {start}: {exc}") from exc

    return Dish(
        dish_id=dish_id,
        calories=calories,
        mass_g=mass,
        fat_g=fat,
        carb_g=carb,
        protein_g=protein,
        ingredients=tuple(ingredients),
    )


def parse_dish_metadata(text: str) -> dict[str, Dish]:
    """Parse a whole metadata CSV. Blank lines are skipped."""
    dishes: dict[str, Dish] = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].strip():
            continue
        dish = parse_dish_row(row)
        dishes[dish.dish_id] = dish
    return dishes


def parse_split_ids(text: str) -> list[str]:
    """Parse a split file into dish ids, preserving order and dropping blanks."""
    return [line.strip() for line in text.splitlines() if line.strip()]
