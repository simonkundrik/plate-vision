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
    assert payload["num_classes"] == 101


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
