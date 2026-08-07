"""Tests for the nutrition model and backbone transfer."""

from __future__ import annotations

import pytest
import torch

from platevision import meta, models

TINY = "mobilenetv3_small_100"


def build(num_targets=5, num_quantiles=3):
    return models.NutritionModel(
        TINY, num_targets=num_targets, num_quantiles=num_quantiles, pretrained=False
    )


def test_output_shape_is_targets_by_quantiles():
    model = build()
    out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 5, 3)


def test_output_shape_follows_the_contract():
    """Head width comes from the contract, so it cannot drift from what the app expects."""
    declared = meta.load_meta()["outputs"]["nutrition_quantiles"]["shape"]
    model = build(num_targets=declared[1], num_quantiles=declared[2])
    out = model(torch.zeros(1, 3, 224, 224))
    assert list(out.shape[1:]) == declared[1:]


def test_backbone_produces_pooled_features_not_logits():
    model = build()
    features = model.backbone(torch.zeros(1, 3, 224, 224))
    assert features.ndim == 2
    assert features.shape[1] == model.feature_dim


def test_feature_dim_is_measured_not_assumed():
    """MobileNetV3 reports num_features=576 but emits 1024, because of an extra head
    convolution. Trusting the attribute builds a head of the wrong width."""
    model = build()
    assert model.feature_dim == model.head.in_features
    assert model.feature_dim != model.backbone.num_features


def test_head_is_a_single_linear_layer():
    """2,755 training dishes will not support a deeper head; extra capacity buys
    overfitting rather than accuracy."""
    assert isinstance(build().head, torch.nn.Linear)


def test_gradients_reach_the_backbone():
    model = build()
    model(torch.zeros(1, 3, 224, 224)).sum().backward()
    grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_dropout_is_active_in_train_mode_only():
    model = build()
    x = torch.randn(8, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))

    model.train()
    torch.manual_seed(0)
    first = model(x)
    torch.manual_seed(1)
    assert not torch.allclose(first, model(x))


# --- backbone transfer -------------------------------------------------------------


def test_transfer_copies_backbone_weights_from_a_classifier():
    """The two-stage story: Food-101 teaches the backbone what food looks like, and
    Nutrition5k only has to fit a head on top of it."""
    classifier = models.create_classifier(TINY, num_classes=101, pretrained=False)
    model = build()

    copied, skipped = models.load_backbone_weights(model, classifier.state_dict())

    assert copied > 0
    for name, param in classifier.named_parameters():
        target = dict(model.named_parameters()).get(f"backbone.{name}")
        if target is not None and target.shape == param.shape:
            assert torch.equal(target, param)
    # The classifier head has no counterpart in a regression model.
    assert skipped > 0


def test_transfer_actually_changes_the_weights():
    classifier = models.create_classifier(TINY, num_classes=101, pretrained=False)
    model = build()
    before = next(iter(model.backbone.parameters())).clone()

    models.load_backbone_weights(model, classifier.state_dict())

    assert not torch.equal(next(iter(model.backbone.parameters())), before)


def test_transfer_from_a_mismatched_architecture_copies_almost_nothing():
    """A silent near-total skip would look like a fine-tune while training from scratch,
    so the copied count is returned rather than discarded."""
    other = models.create_classifier("resnet18", num_classes=101, pretrained=False)
    model = build()

    copied, skipped = models.load_backbone_weights(model, other.state_dict())

    assert copied < skipped


def test_transfer_leaves_the_head_untouched():
    classifier = models.create_classifier(TINY, num_classes=101, pretrained=False)
    model = build()
    head_before = model.head.weight.clone()

    models.load_backbone_weights(model, classifier.state_dict())

    assert torch.equal(model.head.weight, head_before)


def test_parameter_groups_cover_the_nutrition_model():
    model = build()
    decay, no_decay = models.parameter_groups(model, weight_decay=0.05)
    grouped = sum(p.numel() for p in decay["params"]) + sum(p.numel() for p in no_decay["params"])
    assert grouped == models.count_parameters(model)


@pytest.mark.parametrize(("targets", "quantiles"), [(1, 1), (5, 3), (3, 5)])
def test_head_width_is_targets_times_quantiles(targets, quantiles):
    model = build(num_targets=targets, num_quantiles=quantiles)
    assert model.head.out_features == targets * quantiles
