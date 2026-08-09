"""Tests for model construction."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from platevision import food101, models

TINY_BACKBONE = "mobilenetv3_small_100"


def test_head_width_defaults_to_the_label_contract():
    """The head cannot drift from the committed class list."""
    model = models.create_classifier(TINY_BACKBONE, pretrained=False)
    logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape[1] == len(food101.class_keys()) == 101


def test_explicit_class_count_is_respected():
    model = models.create_classifier(TINY_BACKBONE, num_classes=5, pretrained=False)
    assert model(torch.zeros(1, 3, 224, 224)).shape[1] == 5


def test_parameter_groups_exclude_norms_and_biases_from_decay():
    """Decaying bias and normalisation terms costs accuracy and never looks wrong."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Linear(4, 2))
    decay, no_decay = models.parameter_groups(model, weight_decay=0.05)

    assert decay["weight_decay"] == 0.05
    assert no_decay["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in decay["params"])
    assert all(p.ndim <= 1 for p in no_decay["params"])


def test_parameter_groups_cover_every_trainable_parameter():
    model = models.create_classifier(TINY_BACKBONE, num_classes=5, pretrained=False)
    decay, no_decay = models.parameter_groups(model, weight_decay=0.05)
    grouped = sum(p.numel() for p in decay["params"]) + sum(p.numel() for p in no_decay["params"])
    assert grouped == models.count_parameters(model)


def test_backbone_lr_is_absent_unless_asked_for():
    """Every caller that does not want discriminative rates must see the groups it always
    saw, including their order, since two of them unpack the result positionally."""
    model = models.create_classifier(TINY_BACKBONE, num_classes=5, pretrained=False)
    groups = models.parameter_groups(model, weight_decay=0.05)

    assert len(groups) == 2
    assert not any("lr" in group for group in groups)


def test_backbone_gets_its_own_learning_rate():
    # The lever the auxiliary weights were not. A classifier head co-trained at kd-weight
    # 0.010 fell to 25.9% top-1 because the optimiser rewrote the backbone at full speed,
    # and no loss term undoes that as fast as it happens.
    model = models.NutritionModel(TINY_BACKBONE, num_targets=5, num_quantiles=3, pretrained=False)
    groups = models.parameter_groups(model, weight_decay=0.05, backbone_lr=1e-5)

    with_lr = [g for g in groups if "lr" in g]
    without = [g for g in groups if "lr" not in g]

    assert with_lr and without
    assert all(g["lr"] == 1e-5 for g in with_lr)


def test_discriminative_groups_still_cover_every_trainable_parameter():
    # Splitting on a name prefix is exactly how parameters get silently dropped from the
    # optimiser, and a dropped group trains at zero without ever raising.
    model = models.NutritionModel(TINY_BACKBONE, num_targets=5, num_quantiles=3, pretrained=False)
    groups = models.parameter_groups(model, weight_decay=0.05, backbone_lr=1e-5)

    grouped = sum(p.numel() for g in groups for p in g["params"])
    assert grouped == models.count_parameters(model)


def test_the_head_is_not_slowed_with_the_backbone():
    model = models.NutritionModel(TINY_BACKBONE, num_targets=5, num_quantiles=3, pretrained=False)
    groups = models.parameter_groups(model, weight_decay=0.05, backbone_lr=1e-5)

    head_params = {id(p) for p in model.head.parameters()}
    slowed = {id(p) for g in groups if "lr" in g for p in g["params"]}
    assert not (head_params & slowed)


def test_weight_decay_still_skips_norms_under_discriminative_rates():
    model = models.NutritionModel(TINY_BACKBONE, num_targets=5, num_quantiles=3, pretrained=False)
    groups = models.parameter_groups(model, weight_decay=0.05, backbone_lr=1e-5)

    for group in groups:
        expected = 0.0 if all(p.ndim <= 1 for p in group["params"]) else 0.05
        assert group["weight_decay"] == expected


def _nutrition():
    return models.NutritionModel(TINY_BACKBONE, num_targets=5, num_quantiles=3, pretrained=False)


def test_an_untouched_backbone_matches_its_classifier():
    classifier = models.create_classifier(TINY_BACKBONE, num_classes=5, pretrained=False)
    model = _nutrition()
    models.load_backbone_weights(model, classifier.state_dict())

    assert models.backbone_matches(model.backbone, classifier.state_dict())


def test_a_moved_backbone_does_not_match():
    # The case the exporter has to catch. A stage-one classifier head on a fine-tuned
    # backbone stays confident while describing features that no longer exist, and every
    # parity check passes because PyTorch and ONNX agree about the same wrong answer.
    classifier = models.create_classifier(TINY_BACKBONE, num_classes=5, pretrained=False)
    model = _nutrition()
    models.load_backbone_weights(model, classifier.state_dict())

    with torch.no_grad():
        next(iter(model.backbone.parameters())).add_(0.01)

    assert not models.backbone_matches(model.backbone, classifier.state_dict())


def test_a_drifted_running_statistic_counts_as_moved():
    # Nothing here is a gradient, and the features change anyway. This is the half that
    # freezing by requires_grad alone would miss.
    classifier = models.create_classifier(TINY_BACKBONE, num_classes=5, pretrained=False)
    state = classifier.state_dict()
    model = _nutrition()
    models.load_backbone_weights(model, state)

    running = [k for k in model.backbone.state_dict() if "running_mean" in k]
    if not running:
        pytest.skip("backbone has no normalisation running statistics")
    with torch.no_grad():
        model.backbone.state_dict()[running[0]].add_(0.5)

    assert not models.backbone_matches(model.backbone, state)


def test_an_unrelated_architecture_does_not_count_as_untouched():
    # Zero overlapping tensors must not vacuously pass, which is how "nothing to compare"
    # becomes "everything matches".
    model = _nutrition()
    assert not models.backbone_matches(model.backbone, {"totally.different.key": torch.zeros(3)})


def test_freezing_stops_every_backbone_gradient():
    model = _nutrition()
    frozen = models.freeze_backbone(model)

    assert frozen > 0
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_a_frozen_backbone_stays_in_eval_through_model_train():
    """The half that `requires_grad = False` does not cover. BatchNorm keeps updating its
    running statistics in train mode, so the features drift even though no gradient reaches
    the weights, and the classifier head that depends on them degrades for reasons no
    gradient explains. The training loop calls model.train() once per epoch."""
    model = _nutrition()
    models.freeze_backbone(model)

    model.train()
    assert not model.backbone.training
    assert model.head.training


def test_freezing_is_off_by_default():
    model = _nutrition()
    model.train()
    assert model.backbone.training
    assert not model.backbone_frozen


def test_running_statistics_do_not_move_while_frozen():
    # The property stated rather than the mechanism: run data through a frozen backbone in
    # training mode and its normalisation state must be byte-identical afterwards.
    model = _nutrition()
    models.freeze_backbone(model)
    model.train()

    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    model(torch.randn(4, 3, 224, 224))
    after = model.backbone.state_dict()

    for key, value in before.items():
        assert torch.equal(value, after[key]), f"{key} moved while frozen"


def test_a_frozen_backbone_contributes_no_optimiser_groups():
    model = _nutrition()
    models.freeze_backbone(model)
    groups = models.parameter_groups(model, weight_decay=0.05)

    grouped = sum(p.numel() for g in groups for p in g["params"])
    assert grouped == models.count_parameters(model, trainable_only=True)
    assert grouped < models.count_parameters(model, trainable_only=False)


def test_freezing_something_with_no_backbone_is_an_error():
    with pytest.raises(TypeError, match="no backbone"):
        models.freeze_backbone(nn.Linear(4, 2))


def test_frozen_parameters_are_skipped():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    model[0].weight.requires_grad_(False)
    decay, no_decay = models.parameter_groups(model, weight_decay=0.01)
    all_params = decay["params"] + no_decay["params"]
    assert all(p.requires_grad for p in all_params)


def test_count_parameters_respects_trainable_only():
    model = nn.Linear(4, 2)
    total = models.count_parameters(model, trainable_only=False)
    model.weight.requires_grad_(False)
    assert models.count_parameters(model, trainable_only=True) < total


def test_resolve_device_honours_an_explicit_request():
    assert models.resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_falls_back_to_cpu_without_cuda():
    device = models.resolve_device(None)
    assert device.type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_teacher_candidates_do_not_include_the_student():
    """Distilling a model into itself would be a no-op worth catching early."""
    assert models.STUDENT_BACKBONE not in models.TEACHER_BACKBONES
