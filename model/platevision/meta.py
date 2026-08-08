"""Loader for the shared model contract.

``shared/model_meta.json`` is the single source of truth for input shape, output
names, and preprocessing constants. The exporter, the Android app, and the web demo
all read it. This module is the Python side of that contract.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from platevision._contract import contract_path

META_PATH = contract_path("model_meta.json")

# The only preprocessing order this codebase implements. Both the training eval transform
# and the ONNX exporter are written against it, so an edit to the contract that nothing
# implements should fail loudly rather than silently disagree with the deployed graph.
SUPPORTED_PREPROCESSING_ORDER = ("to_float_unit_range", "resize", "normalize")


@lru_cache(maxsize=1)
def load_meta(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the shared model contract.

    Validation is deliberately strict. A malformed contract silently produces a model
    that scores well in Python and garbage on the phone, so it fails loudly here instead.
    """
    meta_path = path or META_PATH
    with meta_path.open(encoding="utf-8") as fh:
        meta: dict[str, Any] = json.load(fh)

    _validate(meta)
    return meta


def _validate(meta: dict[str, Any]) -> None:
    if meta.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version: {meta.get('schema_version')!r}")

    order = tuple(meta["preprocessing"]["order"])
    if order != SUPPORTED_PREPROCESSING_ORDER:
        raise ValueError(
            f"preprocessing order {order} is not implemented; "
            f"only {SUPPORTED_PREPROCESSING_ORDER} is"
        )

    norm = meta["preprocessing"]["normalize"]
    if len(norm["mean"]) != 3 or len(norm["std"]) != 3:
        raise ValueError("normalize mean and std must each have 3 channel values")
    if any(s <= 0 for s in norm["std"]):
        raise ValueError("normalize std values must be positive")

    quantiles = meta["outputs"]["nutrition_quantiles"]["quantiles"]
    if not all(0.0 < q < 1.0 for q in quantiles):
        raise ValueError("quantiles must lie strictly between 0 and 1")
    if list(quantiles) != sorted(quantiles):
        raise ValueError("quantiles must be listed in ascending order")

    targets = meta["outputs"]["nutrition_quantiles"]["targets"]
    shape = meta["outputs"]["nutrition_quantiles"]["shape"]
    if shape[1] != len(targets):
        raise ValueError(f"output axis 1 is {shape[1]} but {len(targets)} targets are declared")
    if shape[2] != len(quantiles):
        raise ValueError(f"output axis 2 is {shape[2]} but {len(quantiles)} quantiles are declared")


def target_keys() -> list[str]:
    """Nutrition target keys, in the order they appear on output axis 1."""
    return [t["key"] for t in load_meta()["outputs"]["nutrition_quantiles"]["targets"]]


def quantiles() -> list[float]:
    """Quantile levels, in the order they appear on output axis 2."""
    return list(load_meta()["outputs"]["nutrition_quantiles"]["quantiles"])


def input_size() -> tuple[int, int]:
    """Target (height, width) the in-graph resize produces."""
    resize = load_meta()["preprocessing"]["resize"]
    return resize["height"], resize["width"]


def normalization() -> tuple[list[float], list[float]]:
    """ImageNet (mean, std). Baked into the exported graph; clients never apply these."""
    norm = load_meta()["preprocessing"]["normalize"]
    return list(norm["mean"]), list(norm["std"])


def preprocessing_order() -> tuple[str, ...]:
    """The sequence the eval transform and the exported graph must both follow."""
    return tuple(load_meta()["preprocessing"]["order"])
