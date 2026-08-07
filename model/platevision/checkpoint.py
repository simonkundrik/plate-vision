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


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    epoch: int,
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
        "config": config or {},
        "history": history or [],
        "best_metric": best_metric,
        "label_order_sha256": food101.load_labels()["order_sha256"],
        "num_classes": len(food101.class_keys()),
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
