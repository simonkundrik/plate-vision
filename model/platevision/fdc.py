"""USDA FoodData Central parsing for the chain-menu nutrition set.

Nutrition5k is cafeteria trays on a fixed overhead rig. The deployed model sees handheld
photos. Measuring calorie error outside that rig needs images with known calories, and a
kitchen scale is the only exact way to get them.

Chain restaurant items are the closest approximation available without one, because they
are standardised: a Big Mac is a Big Mac. USDA FoodData Central publishes per-item figures
for many of them.

**The ground truth here is approximate, in two independent ways.**

FDC's Survey (FNDDS) values are survey estimates, not the chain's published numbers. For a
Big Mac, FNDDS gives 261 kcal per 100 g against a 205 g portion, so 535 kcal, where
McDonald's publishes 563. Roughly 5 percent apart.

And a photograph of a named item may show a partial portion, a combo, or the wrong thing
entirely. That error is unbounded and is why the set is filtered and hand-reviewable rather
than trusted wholesale.

Both are far better than no label at all, and both belong in any number computed from this
set. Pure functions only; network calls live in the data script.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Only Survey (FNDDS) records carry foodMeasures with gram weights, which is what turns a
# per-100g figure into a per-item one. SR Legacy has the nutrients but no portions.
PREFERRED_DATA_TYPE = "Survey (FNDDS)"

# api.data.gov throttles DEMO_KEY hard: roughly 30 requests an hour. A free key is instant
# to obtain and lifts this substantially.
DEMO_KEY = "DEMO_KEY"
DEMO_KEY_PER_HOUR = 30

# Used when no portion mentions the item by name. FNDDS sets it to the standard serving.
FALLBACK_MEASURE = "Quantity not specified"

# Outside this range a "per item" figure is not a plated dish: it is a condiment packet or
# a parsing failure. Both should be dropped rather than averaged into a benchmark.
MIN_PLAUSIBLE_KCAL = 40.0
MAX_PLAUSIBLE_KCAL = 2000.0


# FDC marks generic entries "NFS" (Not Further Specified) or "NS as to". A record carrying
# one of these is a category average, not a particular restaurant's product.
GENERIC_MARKERS = ("nfs", "ns as to")


@dataclass(frozen=True, slots=True)
class MenuItem:
    query: str
    fdc_id: int
    description: str
    data_type: str
    kcal_per_100g: float
    gram_weight: float
    portion_label: str
    kcal_per_item: float
    brand_verified: bool

    def as_dict(self) -> dict:
        return asdict(self)


def mentions_brand(description: str, brand: str) -> bool:
    """Whether a record is the chain's own product rather than a generic category average.

    This is what the whole approach rests on. The premise is that chain items are
    standardised, so a photo of a Big Mac has known calories. That holds for
    "Big Mac (McDonalds)" and collapses for "Croissant": a generic croissant is about
    170 kcal where a Starbucks butter croissant is nearer 260, so labelling Starbucks
    photos with the generic figure is a roughly 50 percent error presented as ground truth.

    Note this is the opposite requirement from :func:`matches_item`, and both are needed.
    Ignoring the brand while matching the item prevents a shared brand from rescuing a
    mismatched food; requiring the brand separately prevents a generic food from
    impersonating a branded one.
    """
    lowered = description.lower()
    if any(marker in lowered for marker in GENERIC_MARKERS):
        return False
    brand_tokens = tokens_of(brand)
    return bool(brand_tokens) and bool(brand_tokens & tokens_of(description))


def energy_per_100g(food: dict) -> float | None:
    """Pull the kcal figure out of a search result's nutrient list.

    FDC reports energy in both kcal and kJ under names that both begin with "Energy", so
    the unit is checked rather than taking the first match.
    """
    for nutrient in food.get("foodNutrients", []):
        name = (nutrient.get("nutrientName") or "").lower()
        unit = (nutrient.get("unitName") or "").lower()
        if name.startswith("energy") and unit in ("kcal", "kilocalorie"):
            value = nutrient.get("value")
            if value is not None:
                return float(value)
    return None


def choose_portion(food: dict, query: str) -> tuple[float, str] | None:
    """Pick the gram weight for one serving of the queried item.

    A single record can list several portions. The Big Mac record carries both
    "1 McDonald's Big Mac" at 205 g and "1 McDonald's Grand Mac" at 315 g; taking the first
    or the largest would silently label photos with a different product's calories.
    Preference goes to a portion whose text overlaps the query.
    """
    measures = [
        m for m in food.get("foodMeasures", []) if m.get("gramWeight") and m.get("gramWeight") > 0
    ]
    if not measures:
        return None

    tokens = {word for word in query.lower().split() if len(word) > 2}

    best = None
    best_overlap = 0
    for measure in measures:
        text = (measure.get("disseminationText") or "").lower()
        if text == FALLBACK_MEASURE.lower():
            continue
        overlap = sum(1 for token in tokens if token in text)
        if overlap > best_overlap:
            best, best_overlap = measure, overlap

    if best is None:
        for measure in measures:
            if (measure.get("disseminationText") or "") == FALLBACK_MEASURE:
                best = measure
                break
    if best is None:
        best = measures[0]

    return float(best["gramWeight"]), str(best.get("disseminationText") or "")


def tokens_of(text: str) -> set[str]:
    """Comparable words, ignoring punctuation, case, and anything too short to be a name."""
    cleaned = "".join(character if character.isalnum() else " " for character in text.lower())
    return {word for word in cleaned.split() if len(word) >= 3}


def matches_item(description: str, item_name: str, threshold: float = 0.5) -> bool:
    """Whether an FDC record is plausibly the item that was searched for.

    Without this the set silently acquires wrong ground truth. FNDDS has no McChicken
    record, so a search for one returns "Cheeseburger (McDonalds)" as its best Survey
    match, and taking the top result would label McChicken photos with cheeseburger
    calories. Nothing downstream would notice: the number is real, plausible, and attached
    to the wrong food.

    Matching is on the item name only, deliberately excluding the brand. Every McDonald's
    record matches "McDonalds", so allowing the brand to carry a match makes the check
    useless.
    """
    wanted = tokens_of(item_name)
    if not wanted:
        return False
    found = tokens_of(description)
    return len(wanted & found) / len(wanted) >= threshold


def parse_search(
    payload: dict, query: str, item_name: str = "", brand: str = "", require_brand: bool = False
) -> MenuItem | None:
    """Turn one FDC search response into a per-item calorie record, or None.

    Returns None rather than a partial record whenever the energy value, the portion, the
    name match, or the resulting figure is unusable. A benchmark built from half-parsed
    rows is worse than a smaller one, and a benchmark built from *confidently mismatched*
    rows is worse than having none.
    """
    candidates = [
        food
        for food in payload.get("foods", [])
        if food.get("dataType") == PREFERRED_DATA_TYPE
        and energy_per_100g(food) is not None
        and matches_item(food.get("description", ""), item_name or query)
    ]
    if not candidates:
        return None

    food = candidates[0]
    kcal_100g = energy_per_100g(food)
    portion = choose_portion(food, query)
    if portion is None:
        return None

    grams, label = portion
    kcal_item = kcal_100g * grams / 100.0
    if not MIN_PLAUSIBLE_KCAL <= kcal_item <= MAX_PLAUSIBLE_KCAL:
        return None

    verified = mentions_brand(food.get("description", ""), brand) if brand else False
    if require_brand and not verified:
        return None

    return MenuItem(
        query=query,
        fdc_id=int(food["fdcId"]),
        description=str(food.get("description", "")),
        data_type=str(food.get("dataType", "")),
        kcal_per_100g=kcal_100g,
        gram_weight=grams,
        portion_label=label,
        kcal_per_item=round(kcal_item, 1),
        brand_verified=verified,
    )


def build_search_url(query: str, api_key: str = DEMO_KEY, page_size: int = 5) -> str:
    import urllib.parse

    params = urllib.parse.urlencode({"query": query, "api_key": api_key, "pageSize": page_size})
    return f"{SEARCH_URL}?{params}"


# (brand, item) rather than one string, so name matching can ignore the brand. Every
# McDonald's record matches "McDonalds", so a brand token carrying a match makes the check
# useless.
#
# Item names only. Every calorie figure is fetched from FDC at build time. Nothing here
# encodes a nutrition claim, because writing remembered numbers into a benchmark would be
# inventing the ground truth it exists to measure against.
SEED_ITEMS: list[tuple[str, str]] = [
    ("McDonalds", "Big Mac"),
    ("McDonalds", "Quarter Pounder with cheese"),
    ("McDonalds", "McChicken sandwich"),
    ("McDonalds", "Filet-O-Fish"),
    ("McDonalds", "Chicken McNuggets"),
    ("McDonalds", "french fries"),
    ("McDonalds", "Egg McMuffin"),
    ("McDonalds", "hamburger"),
    ("McDonalds", "cheeseburger"),
    ("Burger King", "Whopper"),
    ("Burger King", "chicken sandwich"),
    ("Burger King", "onion rings"),
    ("Wendys", "hamburger"),
    ("Wendys", "chicken nuggets"),
    ("Wendys", "chili"),
    ("Wendys", "Frosty"),
    ("Taco Bell", "crunchy taco"),
    ("Taco Bell", "burrito"),
    ("Taco Bell", "quesadilla"),
    ("Taco Bell", "nachos"),
    ("Subway", "turkey breast sandwich"),
    ("Subway", "meatball marinara sandwich"),
    ("Chick-fil-A", "chicken sandwich"),
    ("Chick-fil-A", "waffle fries"),
    ("KFC", "fried chicken breast"),
    ("KFC", "coleslaw"),
    ("KFC", "mashed potatoes with gravy"),
    ("Popeyes", "fried chicken"),
    ("Dominos", "cheese pizza"),
    ("Pizza Hut", "pepperoni pizza"),
    ("Papa Johns", "cheese pizza"),
    ("Starbucks", "blueberry muffin"),
    ("Starbucks", "croissant"),
    ("Dunkin Donuts", "glazed donut"),
    ("Chipotle", "chicken burrito"),
    ("Chipotle", "chicken burrito bowl"),
    ("Panera", "broccoli cheddar soup"),
    ("Olive Garden", "breadstick"),
    ("Applebees", "mozzarella sticks"),
    ("IHOP", "buttermilk pancakes"),
]


def query_for(brand: str, item: str) -> str:
    return f"{item} {brand}"
