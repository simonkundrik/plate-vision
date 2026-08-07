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


class NutritionModel(nn.Module):
    """Backbone plus a quantile-regression head.

    Outputs (batch, targets, quantiles). The head is deliberately a single linear layer
    with dropout rather than an MLP: there are 2,755 training dishes, and extra head
    capacity on that little data buys overfitting, not accuracy.
    """

    def __init__(
        self,
        backbone: str = STUDENT_BACKBONE,
        *,
        num_targets: int,
        num_quantiles: int,
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ) -> None:
        super().__init__()
        import timm

        # num_classes=0 makes timm return pooled features instead of logits.
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        # Probed rather than read from backbone.num_features, which is not always the
        # width that comes out. MobileNetV3 reports 576 there but emits 1024, because it
        # has an extra head convolution ahead of the classifier. Measuring is correct for
        # every architecture; trusting the attribute is correct for some of them.
        self.feature_dim = self._probe_feature_dim()

        self.num_targets = num_targets
        self.num_quantiles = num_quantiles
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.feature_dim, num_targets * num_quantiles)

    @torch.no_grad()
    def _probe_feature_dim(self) -> int:
        from platevision.meta import input_size

        was_training = self.backbone.training
        self.backbone.eval()
        height, width = input_size()
        features = self.backbone(torch.zeros(1, 3, height, width))
        self.backbone.train(was_training)
        return int(features.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.backbone(x))
        return self.head(features).view(-1, self.num_targets, self.num_quantiles)


class CombinedModel(nn.Module):
    """One backbone, two heads: Food-101 logits and nutrition quantiles.

    The contract declares a single model with both outputs, so the app downloads and runs
    one artifact rather than two.

    This is only valid if both heads were fitted against *this* backbone. Nutrition
    training fine-tunes the backbone, which invalidates the classification head from stage
    one, so that head is re-fitted as a linear probe on the final frozen backbone before
    this is assembled. Skipping that step produces a model whose logits are confidently
    wrong while its nutrition outputs are fine, and nothing about the export would say so.
    """

    def __init__(
        self,
        backbone: nn.Module,
        classifier_head: nn.Module,
        nutrition_head: nn.Module,
        *,
        num_targets: int,
        num_quantiles: int,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_head = classifier_head
        self.nutrition_head = nutrition_head
        self.num_targets = num_targets
        self.num_quantiles = num_quantiles
        self.dropout = nn.Dropout(drop_rate)
        # Declared explicitly. Checkpoint saving infers output width from the last
        # parameter when a model does not say, and the last parameter here belongs to the
        # nutrition head, so it would record 15 classes instead of 101.
        self.num_classes = classifier_head.out_features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.dropout(self.backbone(x))
        logits = self.classifier_head(features)
        nutrition = self.nutrition_head(features).view(-1, self.num_targets, self.num_quantiles)
        return logits, nutrition


def load_backbone_weights(
    model: NutritionModel, classifier_state: dict[str, torch.Tensor]
) -> tuple[int, int]:
    """Transfer a trained classifier's backbone into a nutrition model.

    Returns (copied, skipped). The classifier head has no counterpart here and is expected
    to be skipped; anything else being skipped means the two backbones are not the same
    architecture, which would otherwise present as a nutrition model that trains from
    scratch while appearing to be a fine-tune.
    """
    own = model.state_dict()
    matched = {}
    for key, value in classifier_state.items():
        prefixed = key if key in own else f"backbone.{key}"
        if prefixed in own and own[prefixed].shape == value.shape:
            matched[prefixed] = value

    model.load_state_dict(matched, strict=False)
    return len(matched), len(classifier_state) - len(matched)


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
