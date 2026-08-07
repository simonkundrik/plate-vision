"""Model construction.

EfficientNet-B0 is the student. It is the smallest backbone that still reaches useful
Food-101 accuracy, and it quantises cleanly to int8, which matters because the whole
project is pointed at a model that has to run inside a phone and a browser tab.
"""

from __future__ import annotations

import torch
from torch import nn

from platevision import food101

STUDENT_BACKBONE = "efficientnet_b0"

# Candidate teachers for the distillation stage (PR #8). Listed here so the choice is
# recorded next to the student rather than buried in a notebook.
TEACHER_BACKBONES = ("efficientnet_b3", "convnext_small")


def create_classifier(
    backbone: str = STUDENT_BACKBONE,
    *,
    num_classes: int | None = None,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.2,
) -> nn.Module:
    """Build a timm classification backbone.

    ``num_classes`` defaults to the committed Food-101 class count, so the head width
    cannot drift from the label contract.
    """
    import timm

    if num_classes is None:
        num_classes = len(food101.class_keys())

    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters so norms and biases are excluded from weight decay.

    Decaying bias and normalisation terms is a small but consistent accuracy loss, and it
    is the kind of default that quietly costs a point without ever looking wrong.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
