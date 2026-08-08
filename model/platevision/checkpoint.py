"""Checkpoint saving and loading.

Checkpoints carry the label-order digest alongside the weights. A model trained against
one class ordering and later loaded against another produces confident, plausible, wrong
predictions, and nothing in the loss or the accuracy would reveal it. The digest turns
that into a load-time error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from platevision import food101

FORMAT_VERSION = 1


def _output_width(model: torch.nn.Module) -> int:
    """How many classes the model actually predicts.

    timm sets ``num_classes``; anything else is inspected by taking the leading dimension
    of the last parameter with more than one element, which is the classifier weight or
    bias for every architecture used here.
    """
    declared = getattr(model, "num_classes", None)
    if isinstance(declared, int) and declared > 0:
        return declared

    for param in reversed(list(model.parameters())):
        if param.numel() > 1:
            return int(param.shape[0])
    raise ValueError("cannot determine the model's output width")


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    backbone: str | None = None,
    config: dict[str, Any] | None = None,
    history: list[dict] | None = None,
    best_metric: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> None:
    """Write a checkpoint atomically.

    Written to a temporary name and renamed, so an interrupted save cannot leave a
    truncated file that later loads as a corrupt model rather than failing outright.
    """
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "epoch": epoch,
        "model": model.state_dict(),
        # Recorded as a first-class field, not left to be dug out of the config blob.
        # A state_dict alone does not say which architecture it belongs to, and loading
        # a teacher requires constructing the right one before the weights will fit.
        "backbone": backbone or (config or {}).get("backbone"),
        "config": config or {},
        "history": history or [],
        "best_metric": best_metric,
        "label_order_sha256": food101.load_labels()["order_sha256"],
        # The model's real output width, not the contract's class count. A run with
        # --subset-classes has a narrower head, and recording 101 regardless produces a
        # checkpoint that cannot be reconstructed: load_state_dict fails on a shape
        # mismatch that points at the head rather than at this line.
        "num_classes": _output_width(model),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint and verify it belongs to the current label contract."""
    payload = torch.load(path, map_location=map_location, weights_only=False)

    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"checkpoint format version {version!r}, expected {FORMAT_VERSION}")

    expected = food101.load_labels()["order_sha256"]
    found = payload.get("label_order_sha256")
    if found != expected:
        raise ValueError(
            "checkpoint was trained against a different label ordering "
            f"({found} vs {expected}). Loading it would relabel every prediction."
        )

    return payload


def load_model_weights(
    model: torch.nn.Module, path: Path, *, map_location: str = "cpu", strict: bool = True
) -> dict[str, Any]:
    """Restore weights into ``model`` and return the rest of the checkpoint."""
    payload = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=strict)
    return payload


def backbone_of(payload: dict[str, Any]) -> str | None:
    """Which architecture a checkpoint belongs to.

    Prefers the first-class field, falling back to the config blob. Checkpoints written
    before the field existed still carry the backbone in their saved arguments, and the
    Food-101 baseline is one of them: refusing to load a genuinely valid four-hour
    training run over a missing key would make the guard an obstacle rather than a check.
    """
    return payload.get("backbone") or payload.get("config", {}).get("backbone")


def restore_nutrition_model(path: Path, *, map_location: str = "cpu"):
    """Rebuild a nutrition model and the target transform it was trained with.

    The transform is not optional extra metadata. The model predicts in standardised log
    space, so without the exact statistics it was fitted with, its outputs are unitless
    numbers rather than kilocalories. Storing them in the checkpoint is what makes the
    weights usable on their own.
    """
    from platevision.meta import quantiles, target_keys
    from platevision.models import NutritionModel
    from platevision.targets import TargetTransform

    payload = load_checkpoint(path, map_location=map_location)

    backbone = backbone_of(payload)
    if not backbone:
        raise ValueError(f"{path} does not record which backbone it was trained with")

    stored = payload.get("config", {}).get("target_transform")
    if not stored:
        raise ValueError(
            f"{path} has no target transform, so its predictions cannot be converted "
            "back into kilocalories"
        )

    # The auxiliary ingredient head is trained but never exported, so its width is not in
    # the contract. Reading it back from the weights lets a checkpoint round-trip; hardcoding
    # zero here would make every run that used ingredient supervision unloadable.
    ingredient_weight = payload["model"].get("ingredient_head.weight")
    num_ingredients = int(ingredient_weight.shape[0]) if ingredient_weight is not None else 0

    model = NutritionModel(
        backbone,
        num_targets=len(target_keys()),
        num_quantiles=len(quantiles()),
        pretrained=False,
        num_ingredients=num_ingredients,
    )
    model.load_state_dict(payload["model"])
    return model, TargetTransform.from_dict(stored), payload


def restore_combined_model(path: Path, *, map_location: str = "cpu"):
    """Rebuild the two-head model that gets exported, and its target transform.

    Both heads must have been fitted against the same backbone. That is what the linear
    probe step guarantees, and it is why the combined checkpoint is written by that script
    rather than assembled ad hoc at export time.
    """
    import timm
    from torch import nn

    from platevision.models import CombinedModel
    from platevision.targets import TargetTransform

    payload = load_checkpoint(path, map_location=map_location)
    config = payload.get("config", {})

    backbone_name = backbone_of(payload)
    if not backbone_name:
        raise ValueError(f"{path} does not record which backbone it was trained with")
    stored = config.get("target_transform")
    if not stored:
        raise ValueError(
            f"{path} has no target transform, so its nutrition outputs cannot be "
            "converted back into kilocalories"
        )

    num_targets = config["num_targets"]
    num_quantiles = config["num_quantiles"]

    backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
    feature_dim = backbone(torch.zeros(1, 3, 224, 224)).shape[1]

    model = CombinedModel(
        backbone,
        nn.Linear(feature_dim, payload["num_classes"]),
        nn.Linear(feature_dim, num_targets * num_quantiles),
        num_targets=num_targets,
        num_quantiles=num_quantiles,
    )
    model.load_state_dict(payload["model"])
    return model, TargetTransform.from_dict(stored), payload


def restore_classifier(
    path: Path, *, map_location: str = "cpu", pretrained: bool = False
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the architecture a checkpoint describes and load its weights into it.

    Needed for distillation: the teacher is a different architecture from the student, so
    the training script cannot assume which model to construct before loading.

    ``pretrained`` is False because the weights are about to be overwritten. Downloading
    ImageNet weights only to replace them wastes time and, on Kaggle, needs the network.
    """
    from platevision.models import create_classifier

    payload = load_checkpoint(path, map_location=map_location)

    backbone = backbone_of(payload)
    if not backbone:
        raise ValueError(
            f"{path} does not record which backbone it was trained with, so the "
            "architecture cannot be reconstructed. Retrain, or pass the backbone explicitly."
        )

    model = create_classifier(
        backbone,
        num_classes=payload["num_classes"],
        pretrained=pretrained,
    )
    model.load_state_dict(payload["model"])
    return model, payload
