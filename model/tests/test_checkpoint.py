"""Tests for checkpoint saving and loading.

The label-order guard is the point. A checkpoint trained under one class ordering and
loaded under another gives confident, plausible, wrong answers, and no metric reveals it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from platevision import checkpoint, food101


def tiny_model():
    return nn.Sequential(nn.Linear(4, 3))


def test_round_trip_restores_weights(tmp_path):
    model = tiny_model()
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=model, epoch=3)

    restored = tiny_model()
    payload = checkpoint.load_model_weights(restored, path)

    assert payload["epoch"] == 3
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_checkpoint_records_the_label_contract(tmp_path):
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=tiny_model(), epoch=0)
    payload = checkpoint.load_checkpoint(path)

    assert payload["label_order_sha256"] == food101.load_labels()["order_sha256"]
    # The model's own head width, which for this three-output stand-in is 3. This
    # previously asserted 101 and passed only because the value was hardcoded, so the
    # assertion was encoding the bug rather than catching it.
    assert payload["num_classes"] == 3


def test_a_full_width_model_records_the_contract_class_count(tmp_path):
    from platevision import models

    model = models.create_classifier("mobilenetv3_small_100", pretrained=False)
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=model, epoch=0, backbone="mobilenetv3_small_100")

    assert checkpoint.load_checkpoint(path)["num_classes"] == len(food101.class_keys()) == 101


def test_load_rejects_a_different_label_ordering(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=tiny_model(), epoch=0)

    monkeypatch.setattr(food101, "load_labels", lambda *a, **k: {"order_sha256": "deadbeef"})
    with pytest.raises(ValueError, match="different label ordering"):
        checkpoint.load_checkpoint(path)


def test_load_rejects_an_unknown_format_version(tmp_path):
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=tiny_model(), epoch=0)

    payload = torch.load(path, weights_only=False)
    payload["format_version"] = 99
    torch.save(payload, path)

    with pytest.raises(ValueError, match="format version"):
        checkpoint.load_checkpoint(path)


def test_optimizer_and_scheduler_state_round_trip(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _s: 1.0)
    path = tmp_path / "ckpt.pt"

    checkpoint.save_checkpoint(path, model=model, epoch=1, optimizer=optimizer, scheduler=scheduler)
    payload = checkpoint.load_checkpoint(path)

    assert "optimizer" in payload
    assert "scheduler" in payload


def test_optimizer_state_is_absent_when_not_supplied(tmp_path):
    """best.pt does not need optimiser state; carrying it would double the file size."""
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=tiny_model(), epoch=0)
    assert "optimizer" not in checkpoint.load_checkpoint(path)


def test_save_is_atomic_and_leaves_no_partial_file(tmp_path):
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=tiny_model(), epoch=0)

    assert path.is_file()
    assert not list(tmp_path.glob("*.part"))


def test_save_creates_missing_directories(tmp_path):
    path = tmp_path / "runs" / "baseline" / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=tiny_model(), epoch=0)
    assert path.is_file()


def test_output_width_is_the_models_real_head_not_the_contract_count(tmp_path):
    """A --subset-classes run has a narrower head. Recording 101 regardless produces a
    checkpoint that cannot be reconstructed, and the shape mismatch at load time points
    at the classifier rather than at the line that wrote the wrong number."""
    narrow = nn.Sequential(nn.Linear(4, 7))
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=narrow, epoch=0, backbone="tiny")

    assert checkpoint.load_checkpoint(path)["num_classes"] == 7


def test_output_width_prefers_the_declared_attribute():
    class Declared(nn.Module):
        num_classes = 42

        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 9)

    assert checkpoint._output_width(Declared()) == 42


def test_backbone_is_recorded_as_a_first_class_field(tmp_path):
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=tiny_model(), epoch=0, backbone="efficientnet_b3")
    assert checkpoint.load_checkpoint(path)["backbone"] == "efficientnet_b3"


def test_backbone_falls_back_to_the_config_blob(tmp_path):
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(
        path, model=tiny_model(), epoch=0, config={"backbone": "convnext_small"}
    )
    assert checkpoint.load_checkpoint(path)["backbone"] == "convnext_small"


def test_restore_rebuilds_the_architecture_and_loads_weights(tmp_path):
    """What distillation needs: the teacher is a different architecture from the student,
    so the loader cannot assume which model to construct."""
    from platevision import models

    original = models.create_classifier("mobilenetv3_small_100", num_classes=5, pretrained=False)
    path = tmp_path / "teacher.pt"
    checkpoint.save_checkpoint(path, model=original, epoch=2, backbone="mobilenetv3_small_100")

    restored, payload = checkpoint.restore_classifier(path)

    assert payload["epoch"] == 2
    assert restored(torch.zeros(1, 3, 224, 224)).shape[1] == 5
    for a, b in zip(original.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_restore_rejects_a_checkpoint_with_no_recorded_backbone(tmp_path):
    path = tmp_path / "ckpt.pt"
    checkpoint.save_checkpoint(path, model=tiny_model(), epoch=0)
    with pytest.raises(ValueError, match="does not record which backbone"):
        checkpoint.restore_classifier(path)


def test_history_and_config_survive_the_round_trip(tmp_path):
    path = tmp_path / "ckpt.pt"
    history = [{"epoch": 0, "split": "val", "top1": 42.0}]
    checkpoint.save_checkpoint(
        path, model=tiny_model(), epoch=0, config={"lr": 0.001}, history=history, best_metric=42.0
    )
    payload = checkpoint.load_checkpoint(path)

    assert payload["config"]["lr"] == 0.001
    assert payload["history"] == history
    assert payload["best_metric"] == 42.0
