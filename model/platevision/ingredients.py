"""Ingredient supervision for the nutrition head.

Calorie error splits almost evenly between mass and density, and they combine in quadrature,
so the density half has to be attacked separately. Density is what the pixels can actually
tell you: a plate of arugula and a plate of the same volume dressed in olive oil look
different and are nothing alike calorically.

Nutrition5k lists the ingredients of every dish and this project has been ignoring them,
predicting five scalars directly from an image. Adding a multi-label ingredient head gives
the backbone semantic pressure aimed at exactly the half that geometry cannot fix.

The head is auxiliary. It is trained, and it is never exported: the contract declares two
outputs and the app has no use for a 133-way ingredient vector. Its job is to shape the
features that the quantile head reads.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import torch

# An ingredient seen a handful of times cannot be learned, and a head full of such classes
# spends capacity on noise. 20 keeps 133 of 196 on the training split, including olive oil,
# which appears in about half of all dishes and is 884 kcal per 100g.
DEFAULT_MIN_COUNT = 20


def build_vocabulary(
    dish_ingredients: Iterable[Iterable[str]], min_count: int = DEFAULT_MIN_COUNT
) -> list[str]:
    """Ingredient names with enough support to learn, in a stable order.

    Takes one iterable of names per dish. Counted per dish rather than per mention, so a
    recipe listing salt twice does not make salt look twice as common.

    Sorted alphabetically rather than by frequency. Frequency order is not stable: adding
    training data reshuffles it, and a reshuffled vocabulary silently relabels every
    dimension of a saved head.
    """
    counts: Counter[str] = Counter()
    for names in dish_ingredients:
        for name in set(names):
            counts[name] += 1

    return sorted(name for name, count in counts.items() if count >= min_count)


def multi_hot(names: Iterable[str], vocabulary: list[str]) -> torch.Tensor:
    """Encode a dish's ingredients against a fixed vocabulary.

    Ingredients outside the vocabulary are dropped rather than mapped to a catch-all. A
    shared "other" dimension would ask the head to predict the presence of an arbitrary
    union of unrelated foods, which is not a coherent target.
    """
    index = {name: position for position, name in enumerate(vocabulary)}
    encoded = torch.zeros(len(vocabulary), dtype=torch.float32)
    for name in names:
        position = index.get(name)
        if position is not None:
            encoded[position] = 1.0
    return encoded


def positive_weight(
    dish_ingredients: Iterable[Iterable[str]], vocabulary: list[str], cap: float = 20.0
) -> torch.Tensor:
    """Per-ingredient weight for the positive class in BCE.

    Most ingredients are absent from most dishes, so an unweighted head reaches a low loss
    by predicting "not present" everywhere and learns nothing. The weight is
    negatives/positives, capped: an ingredient in 2% of dishes would otherwise get a weight
    near 50 and dominate the gradient on its own.
    """
    per_dish = [set(names) for names in dish_ingredients]
    total = len(per_dish)
    counts: Counter[str] = Counter()
    for names in per_dish:
        for name in names:
            counts[name] += 1

    weights = []
    for name in vocabulary:
        positives = counts.get(name, 0)
        if positives == 0:
            weights.append(1.0)
            continue
        weights.append(min(cap, (total - positives) / positives))

    return torch.tensor(weights, dtype=torch.float32)
