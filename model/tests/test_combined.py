"""Tests for the two-head combined model and its checkpoint round trip."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from platevision import checkpoint, meta, models

TINY = "mobilenetv3_small_100"


def build(num_classes=101):
    import timm

    backbone = timm.create_model(TINY, pretrained=False, num_classes=0)
    feature_dim = backbone(torch.zeros(1, 3, 224, 224)).shape[1]
    targets, quantiles = len(meta.target_keys()), len(meta.quantiles())
    return models.CombinedModel(
        backbone,
        nn.Linear(feature_dim, num_classes),
        nn.Linear(feature_dim, targets * quantiles),
        num_targets=targets,
        num_quantiles=quantiles,
    )


def test_returns_both_outputs_with_contract_shapes():
    model = build().eval()
    logits, nutrition = model(torch.zeros(2, 3, 224, 224))

    declared = meta.load_meta()["outputs"]
    assert list(logits.shape[1:]) == declared["logits"]["shape"][1:]
    assert list(nutrition.shape[1:]) == declared["nutrition_quantiles"]["shape"][1:]


def test_both_heads_read_the_same_features():
    """One backbone, two heads. That is what makes a single artifact possible."""
    model = build().eval()
    assert model.classifier_head.in_features == model.nutrition_head.in_features


def test_num_classes_is_declared_not_inferred():
    """Inference takes the last parameter's leading dimension, and the last parameter
    here belongs to the nutrition head, so a checkpoint would record 15 rather than 101."""
    model = build()
    assert model.num_classes == 101
    assert checkpoint._output_width(model) == 101


def test_checkpoint_records_the_classifier_width(tmp_path):
    model = build()
    path = tmp_path / "combined.pt"
    checkpoint.save_checkpoint(path, model=model, epoch=0, backbone=TINY)
    assert checkpoint.load_checkpoint(path)["num_classes"] == 101


def test_round_trip_restores_a_working_model(tmp_path):
    from platevision.targets import TargetTransform

    model = build().eval()
    transform = TargetTransform.fit([(200.0, 10.0, 5.0, 20.0, 150.0)] * 4)
    path = tmp_path / "combined.pt"

    checkpoint.save_checkpoint(
        path,
        model=model,
        epoch=0,
        backbone=TINY,
        config={
            "target_transform": transform.to_dict(),
            "num_targets": model.num_targets,
            "num_quantiles": model.num_quantiles,
        },
    )
    restored, restored_transform, _ = checkpoint.restore_combined_model(path)
    restored.eval()

    with torch.no_grad():
        expected = model(torch.zeros(1, 3, 224, 224))
        actual = restored(torch.zeros(1, 3, 224, 224))

    assert torch.allclose(expected[0], actual[0])
    assert torch.allclose(expected[1], actual[1])
    assert restored_transform == transform


def test_restore_requires_a_target_transform(tmp_path):
    """Without it the nutrition outputs are unitless numbers rather than kilocalories."""
    model = build()
    path = tmp_path / "combined.pt"
    checkpoint.save_checkpoint(
        path,
        model=model,
        epoch=0,
        backbone=TINY,
        config={"num_targets": 5, "num_quantiles": 3},
    )
    with pytest.raises(ValueError, match="no target transform"):
        checkpoint.restore_combined_model(path)


def test_combined_model_exports_and_matches_pytorch(tmp_path):
    """The end of the chain: two heads, one graph, in-graph preprocessing, parity."""
    import warnings

    from platevision import export

    model = build().eval()
    path = tmp_path / "combined.onnx"
    names = ["logits", "nutrition_quantiles"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export.export_onnx(model, path, output_names=names)

    assert not list(tmp_path.glob("*.data"))
    for result in export.check_parity(model, path, output_names=names):
        assert result.within_tolerance, (
            f"{result.output_name} differs by {result.max_absolute_difference}"
        )


def build_dual(num_classes=101):
    """Two backbones, deliberately given different weights so a shared one is detectable."""
    import timm

    classifier_backbone = timm.create_model(TINY, pretrained=False, num_classes=0)
    nutrition_backbone = timm.create_model(TINY, pretrained=False, num_classes=0)
    with torch.no_grad():
        for param in nutrition_backbone.parameters():
            param.add_(0.05)

    feature_dim = classifier_backbone(torch.zeros(1, 3, 224, 224)).shape[1]
    targets, quantiles = len(meta.target_keys()), len(meta.quantiles())
    return models.CombinedModel(
        classifier_backbone,
        nn.Linear(feature_dim, num_classes),
        nn.Linear(feature_dim, targets * quantiles),
        num_targets=targets,
        num_quantiles=quantiles,
        nutrition_backbone=nutrition_backbone,
    )


class TestDualBackbone:
    """Seven runs measured a monotonic trade between the two jobs on one backbone. Giving
    each head the weights it was fitted against is the only configuration where neither is
    compromised, and it costs a second forward pass."""

    def test_output_shapes_are_unchanged(self):
        model = build_dual().eval()
        logits, nutrition = model(torch.zeros(2, 3, 224, 224))

        declared = meta.load_meta()["outputs"]
        assert list(logits.shape[1:]) == declared["logits"]["shape"][1:]
        assert list(nutrition.shape[1:]) == declared["nutrition_quantiles"]["shape"][1:]

    def test_the_nutrition_head_does_not_read_the_classifier_features(self):
        # The failure this guards: reusing the classifier's features silently rebuilds the
        # shared-backbone model, and the outputs look entirely reasonable.
        model = build_dual().eval()
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            shared_features = model.backbone(x)
            from_shared = model.nutrition_head(shared_features)
            _, actual = model(x)

        assert not torch.allclose(
            actual.flatten(1), from_shared.view(actual.shape).flatten(1), atol=1e-5
        )

    def test_sharing_is_still_the_default(self):
        model = build()
        assert model.shares_backbone
        assert model.nutrition_backbone is None

    def test_a_shared_model_keeps_its_state_dict_keys(self):
        # Published artifacts and every existing checkpoint use these keys. A key that only
        # appears in the dual case must not appear in the shared one.
        keys = set(build().state_dict())
        assert not any(k.startswith("nutrition_backbone.") for k in keys)

    def test_a_dual_model_carries_both_backbones_in_its_checkpoint(self, tmp_path):
        model = build_dual()
        path = tmp_path / "dual.pt"
        checkpoint.save_checkpoint(path, model=model, epoch=0, backbone=TINY)
        saved = checkpoint.load_checkpoint(path)["model"]

        assert any(k.startswith("backbone.") for k in saved)
        assert any(k.startswith("nutrition_backbone.") for k in saved)
        assert checkpoint.load_checkpoint(path)["num_classes"] == 101
