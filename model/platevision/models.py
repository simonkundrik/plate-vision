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
        num_ingredients: int = 0,
        num_classes: int = 0,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        import timm

        # num_classes=0 makes timm return pooled features instead of logits.
        #
        # in_chans=4 adds a depth channel. timm adapts the stem's pretrained weights rather
        # than discarding them, which matters: reinitialising the first convolution would
        # throw away the ImageNet features that make training on 2,424 dishes viable at all.
        self.in_chans = in_chans
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=in_chans
        )

        # Probed rather than read from backbone.num_features, which is not always the
        # width that comes out. MobileNetV3 reports 576 there but emits 1024, because it
        # has an extra head convolution ahead of the classifier. Measuring is correct for
        # every architecture; trusting the attribute is correct for some of them.
        self.feature_dim = self._probe_feature_dim()

        self.num_targets = num_targets
        self.num_quantiles = num_quantiles
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.feature_dim, num_targets * num_quantiles)

        # Auxiliary, and deliberately not part of forward(). The contract declares two
        # outputs and the app has no use for an ingredient vector, so exporting one would
        # change the graph for a signal only training consumes. Its job is to shape the
        # features the quantile head reads, not to be shipped.
        self.num_ingredients = num_ingredients
        self.ingredient_head = (
            nn.Linear(self.feature_dim, num_ingredients) if num_ingredients else None
        )

        # Also auxiliary, and also off forward(). Trained by distilling the frozen Food-101
        # classifier, which does two things: it stops fine-tuning on 2,424 cafeteria trays
        # from destroying the dish semantics the backbone arrived with, and it keeps this
        # head valid against the fine-tuned backbone. CombinedModel otherwise needs the
        # stage-one head re-fitted as a linear probe before export, because nutrition
        # training invalidates it.
        self.num_classes = num_classes
        self.classifier_head = nn.Linear(self.feature_dim, num_classes) if num_classes else None

        # Set by freeze_backbone(). Declared here so `train()` can read it before anything
        # has frozen anything, and so it travels in the state dict's buffers-free metadata
        # rather than being an attribute that only sometimes exists.
        self.backbone_frozen = False

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode through every ``model.train()`` call.

        Setting ``requires_grad = False`` stops the weights moving and does nothing about
        BatchNorm running statistics, which keep updating from cafeteria trays in train mode
        and drift the features anyway. Freezing that way looks frozen and is not, and the
        classifier head that depends on those exact features would degrade for reasons no
        gradient explains.

        Overriding here rather than calling ``backbone.eval()`` in the training loop, because
        the loop calls ``model.train()`` once per epoch and would silently undo it.
        """
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def _probe_feature_dim(self) -> int:
        from platevision.meta import input_size

        was_training = self.backbone.training
        self.backbone.eval()
        height, width = input_size()
        # self.in_chans, not a hardcoded 3: a four-channel backbone rejects a three-channel
        # probe, and the failure is a shape error at construction rather than anything that
        # explains itself.
        features = self.backbone(torch.zeros(1, self.in_chans, height, width))
        self.backbone.train(was_training)
        return int(features.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.backbone(x))
        return self.head(features).view(-1, self.num_targets, self.num_quantiles)

    def forward_with_aux(self, x: torch.Tensor):
        """Quantiles plus any auxiliary heads, sharing one backbone pass.

        Separate from forward() so the exported graph keeps the shape the contract
        declares. Calling forward() and then each head would run the backbone once per
        head, which is the expensive part.

        Returns (quantiles, ingredient_logits, class_logits); either auxiliary output is
        None when its head is absent.
        """
        features = self.dropout(self.backbone(x))
        quantiles = self.head(features).view(-1, self.num_targets, self.num_quantiles)
        ingredients = self.ingredient_head(features) if self.ingredient_head is not None else None
        classes = self.classifier_head(features) if self.classifier_head is not None else None
        return quantiles, ingredients, classes


class CombinedModel(nn.Module):
    """Two heads in one artifact: Food-101 logits and nutrition quantiles.

    The contract declares a single model with both outputs, so the app downloads and runs
    one artifact rather than two. It says nothing about how many backbones are inside, and
    that distinction turned out to matter.

    **Shared backbone**, the default. Valid only if both heads were fitted against it.
    Nutrition training fine-tunes the backbone, which invalidates the stage-one
    classification head, so that head has to be re-fitted as a linear probe first. Skipping
    that produces logits that are confidently wrong while the nutrition outputs are fine,
    and nothing about the export would say so.

    **Separate backbones**, by passing ``nutrition_backbone``. Seven training runs measured
    a monotonic trade between the two jobs on one backbone: classifier accuracy from 25.9%
    to 86.1% bought calorie error from 54.7 to 86.3 kcal, and no configuration escaped it,
    including joint training on both datasets. Giving each head the weights it was actually
    fitted against costs size and a second forward pass, and removes the trade entirely.
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
        nutrition_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_head = classifier_head
        self.nutrition_head = nutrition_head
        # Registered as None when shared, so state dict keys stay identical to the
        # single-backbone artifacts already published and old checkpoints still load.
        self.nutrition_backbone = nutrition_backbone
        self.num_targets = num_targets
        self.num_quantiles = num_quantiles
        self.dropout = nn.Dropout(drop_rate)
        # Declared explicitly. Checkpoint saving infers output width from the last
        # parameter when a model does not say, and the last parameter here belongs to the
        # nutrition head, so it would record 15 classes instead of 101.
        self.num_classes = classifier_head.out_features

    @property
    def shares_backbone(self) -> bool:
        return self.nutrition_backbone is None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.dropout(self.backbone(x))
        logits = self.classifier_head(features)

        # A second pass when the backbones differ. Reusing the classifier's features would
        # silently produce the shared-backbone model this exists to avoid, and the outputs
        # would look entirely reasonable.
        nutrition_features = (
            features if self.shares_backbone else self.dropout(self.nutrition_backbone(x))
        )
        nutrition = self.nutrition_head(nutrition_features).view(
            -1, self.num_targets, self.num_quantiles
        )
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
        if prefixed not in own:
            continue

        target = own[prefixed]
        if target.shape == value.shape:
            matched[prefixed] = value
        elif (adapted := _adapt_stem(value, target)) is not None:
            # The one shape mismatch that is not a disagreement: a four-channel stem being
            # loaded from three-channel weights. Skipping it silently leaves the first
            # convolution random while every other layer is pretrained, which presents as a
            # depth experiment that failed rather than as a backbone that was never loaded.
            matched[prefixed] = adapted

    model.load_state_dict(matched, strict=False)
    return len(matched), len(classifier_state) - len(matched)


def _adapt_stem(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """Widen a 3-channel convolution stem to accept extra channels, or return None.

    The colour filters are kept exactly and each new channel starts as their mean, which is
    what a greyscale copy of the image would produce. That is a neutral starting point: the
    channel contributes the same as the average of the ones already there, and training moves
    it from a working model rather than from noise.
    """
    if source.ndim != 4 or target.ndim != 4:
        return None
    if source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
        return None
    if target.shape[1] <= source.shape[1]:
        return None

    widened = target.clone()
    widened[:, : source.shape[1]] = source
    widened[:, source.shape[1] :] = source.mean(dim=1, keepdim=True)
    return widened


def load_classifier_head(model: NutritionModel, classifier_state: dict[str, torch.Tensor]) -> bool:
    """Start the auxiliary classifier head from the trained one rather than from noise.

    Only useful alongside the backbone it was fitted against, and then it matters: a random
    101-class head begins at chance and spends the run climbing back to a number the project
    already has, while the gradient it produces on the way there is pulling the shared
    backbone somewhere nobody chose.

    Returns whether a head was found. Timm keeps it under ``classifier`` for EfficientNet and
    ``head``/``fc`` elsewhere, so the shape is what identifies it rather than the name.
    """
    head = model.classifier_head
    if head is None:
        return False

    for key, value in classifier_state.items():
        if not key.endswith(".weight") or value.shape != head.weight.shape:
            continue
        bias = classifier_state.get(key[: -len(".weight")] + ".bias")
        if bias is None or bias.shape != head.bias.shape:
            continue
        with torch.no_grad():
            head.weight.copy_(value)
            head.bias.copy_(bias)
        return True
    return False


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def backbone_matches(backbone: nn.Module, classifier_state: dict) -> bool:
    """Whether a backbone is bit-identical to the one inside a classifier checkpoint.

    This is the difference between a classifier head that is still valid and one that
    describes features which no longer exist. Nutrition training normally fine-tunes the
    backbone, which invalidates the stage-one head and produces confidently wrong logits
    that nothing about the export reveals. Freezing the backbone leaves it valid.

    Both cases now exist in this project, so the exporter can check which one it has instead
    of trusting whoever invoked it to remember.
    """
    own = backbone.state_dict()
    compared = 0
    for key, value in own.items():
        other = classifier_state.get(key)
        if other is None or other.shape != value.shape:
            continue
        compared += 1

        reference = other.to(value.dtype)
        if value.is_floating_point():
            # Tolerant, not exact. Tracking an EMA of frozen weights recomputes each tensor
            # as decay * w + (1 - decay) * w every step, which is w in exact arithmetic and
            # a few ulps away in float32. Measured on a frozen 60-epoch run that residue
            # reached 5.8e-06 relative, against weight changes four orders of magnitude
            # larger for a backbone that genuinely trained. Comparing with torch.equal
            # reported a frozen run as fine-tuned.
            if not torch.allclose(reference, value, rtol=1e-4, atol=1e-6):
                return False
        elif not torch.equal(reference, value):
            return False

    # No overlap at all means a different architecture, not an untouched backbone.
    return compared > 0


def freeze_backbone(model: nn.Module) -> int:
    """Stop the backbone moving at all. Returns the number of parameters frozen.

    The point of this is what it guarantees rather than what it saves. With the backbone
    held at the weights the Food-101 classifier was fitted against, that classifier head
    stays exactly as accurate as it was measured to be: no distillation defending it, no
    linear probe re-fitting it, nothing to verify afterwards. Four runs that fine-tuned the
    backbone produced classifiers at 76.9%, 74.7%, 64.3% and 25.9% against a baseline of
    85.9%, and none of the levers that were supposed to protect it reached 80%.

    What it costs is unknown until measured: the nutrition head has to read features chosen
    for telling pizza from ramen, not for judging how much of it is on the plate.
    """
    if not hasattr(model, "backbone"):
        raise TypeError(f"{type(model).__name__} has no backbone to freeze")

    frozen = 0
    for param in model.backbone.parameters():
        param.requires_grad = False
        frozen += param.numel()

    # Read by NutritionModel.train(), which is what keeps BatchNorm's running statistics
    # from drifting even though no gradient reaches the weights.
    model.backbone_frozen = True
    model.backbone.eval()
    return frozen


def parameter_groups(
    model: nn.Module, weight_decay: float, *, backbone_lr: float | None = None
) -> list[dict]:
    """Split parameters so norms and biases are excluded from weight decay.

    Decaying bias and normalisation terms is a small but consistent accuracy loss, and it
    is the kind of default that quietly costs a point without ever looking wrong.

    ``backbone_lr`` gives the backbone its own learning rate, which is the lever the
    auxiliary-loss weights turned out not to be. Nutrition training fine-tunes the backbone
    on 2,424 cafeteria dishes and that erases Food-101 semantics: measured, a classifier head
    co-trained at ``--kd-weight 0.010`` fell to 25.9% top-1, and raising the weight to 0.5 to
    defend it took calorie error from 54.7 to 68.8 kcal. One knob cannot hold both, because
    it is asking a loss term to undo damage the optimiser is doing at full speed. Slowing the
    backbone instead preserves the features rather than fighting for them.

    Left unset the groups are exactly as before, so nothing that does not ask for this has
    to know it exists.
    """
    groups: dict[tuple[bool, bool], list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = backbone_lr is not None and name.startswith("backbone.")
        skip_decay = param.ndim <= 1 or name.endswith(".bias")
        groups.setdefault((is_backbone, skip_decay), []).append(param)

    built: list[dict] = []
    for (is_backbone, skip_decay), params in sorted(groups.items()):
        group = {"params": params, "weight_decay": 0.0 if skip_decay else weight_decay}
        if is_backbone:
            group["lr"] = backbone_lr
        built.append(group)
    return built


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
