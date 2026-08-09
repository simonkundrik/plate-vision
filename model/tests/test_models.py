"""Tests for model construction."""

from __future__ import annotations

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
