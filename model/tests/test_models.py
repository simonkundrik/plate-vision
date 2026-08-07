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
