"""Tests for the training loop.

Targeted at the failure modes that do not raise: a scheduler stepped once per epoch
instead of once per batch, evaluation running with dropout active, weights drifting
during a no-grad pass. Each of those produces a model that trains and reports numbers,
just the wrong ones.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from platevision import engine


class TinyNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ModeSpy(TinyNet):
    """Records whether it was in training mode on each forward pass."""

    def __init__(self):
        super().__init__()
        self.observed_modes: list[bool] = []

    def forward(self, x):
        self.observed_modes.append(self.training)
        return super().forward(x)


def separable_loader(n: int = 64, batch_size: int = 8) -> DataLoader:
    """Two trivially separable classes. A loop that cannot fit this is broken."""
    half = n // 2
    x = torch.cat([torch.zeros(half, 3, 8, 8), torch.ones(half, 3, 8, 8)])
    y = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


# --- meters and metrics ---------------------------------------------------------


def test_average_meter_weights_by_batch_size():
    """A short final batch must not count as much as a full one."""
    meter = engine.AverageMeter()
    meter.update(1.0, n=10)
    meter.update(0.0, n=90)
    assert meter.mean == pytest.approx(0.1)


def test_average_meter_is_zero_when_empty():
    assert engine.AverageMeter().mean == 0.0


def test_accuracy_top1():
    output = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    target = torch.tensor([0, 0])
    assert engine.accuracy(output, target, topk=(1,)) == [50.0]


def test_accuracy_top5_counts_a_hit_anywhere_in_the_top_k():
    output = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    target = torch.tensor([1])
    top1, top5 = engine.accuracy(output, target, topk=(1, 5))
    assert top1 == 0.0
    assert top5 == 100.0


def test_accuracy_handles_k_larger_than_class_count():
    """top-5 on a 2-class subset must not crash, which is what --subset-classes produces."""
    output = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([0, 1])
    top1, top5 = engine.accuracy(output, target, topk=(1, 5))
    assert top1 == 100.0
    assert top5 == 100.0


def test_accuracy_is_perfect_when_predictions_are_right():
    output = torch.eye(4) * 10
    assert engine.accuracy(output, torch.arange(4), topk=(1,)) == [100.0]


# --- schedule -------------------------------------------------------------------


def test_warmup_ramps_from_near_zero_to_one():
    assert engine.cosine_warmup_factor(0, 10, 100) == pytest.approx(0.1)
    assert engine.cosine_warmup_factor(9, 10, 100) == pytest.approx(1.0)


def test_cosine_decays_to_the_floor_at_the_end():
    assert engine.cosine_warmup_factor(100, 10, 100) == pytest.approx(0.0, abs=1e-9)
    assert engine.cosine_warmup_factor(100, 10, 100, min_ratio=0.1) == pytest.approx(0.1)


def test_cosine_is_halfway_at_the_midpoint():
    mid = engine.cosine_warmup_factor(55, 10, 100)
    assert mid == pytest.approx(0.5, abs=1e-6)


def test_schedule_is_monotonically_decreasing_after_warmup():
    values = [engine.cosine_warmup_factor(s, 10, 100) for s in range(10, 101)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:], strict=False))


def test_schedule_never_exceeds_one():
    assert all(engine.cosine_warmup_factor(s, 10, 100) <= 1.0 + 1e-12 for s in range(0, 120))


def test_factor_stays_at_the_floor_past_total_steps():
    """A run that overshoots its step budget must not see the cosine turn back upward."""
    assert engine.cosine_warmup_factor(150, 10, 100) == pytest.approx(0.0, abs=1e-9)


def test_scheduler_steps_once_per_batch_not_once_per_epoch():
    """The silent bug this guards: a per-epoch step leaves the run at nearly peak LR."""
    model = TinyNet()
    loader = separable_loader(n=64, batch_size=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = engine.build_scheduler(optimizer, warmup_steps=2, total_steps=8)

    engine.train_one_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"), scheduler=scheduler
    )

    assert len(loader) == 8
    assert scheduler.last_epoch == 8


def test_learning_rate_actually_changes_during_an_epoch():
    model = TinyNet()
    loader = separable_loader(n=64, batch_size=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = engine.build_scheduler(optimizer, warmup_steps=2, total_steps=8)
    start = optimizer.param_groups[0]["lr"]

    engine.train_one_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"), scheduler=scheduler
    )

    assert not math.isclose(optimizer.param_groups[0]["lr"], start)


# --- the loop itself ------------------------------------------------------------


def test_training_reduces_loss_on_separable_data():
    """The end-to-end check that the loop learns at all."""
    engine.seed_everything(0)
    model = TinyNet()
    loader = separable_loader(n=128, batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    first = engine.train_one_epoch(model, loader, criterion, optimizer, device, epoch=0)
    for epoch in range(1, 6):
        last = engine.train_one_epoch(model, loader, criterion, optimizer, device, epoch=epoch)

    assert last.loss < first.loss
    result = engine.evaluate(model, loader, criterion, device)
    assert result.top1 > 90.0


def test_weights_change_during_training():
    model = TinyNet()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    engine.train_one_epoch(
        model, separable_loader(), nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
    )

    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_evaluate_does_not_change_weights():
    model = TinyNet()
    before = copy.deepcopy(model.state_dict())

    engine.evaluate(model, separable_loader(), nn.CrossEntropyLoss(), torch.device("cpu"))

    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_train_runs_in_train_mode_and_evaluate_in_eval_mode():
    """Dropout and batch norm behave differently. Evaluating in train mode silently
    reports a number for a model that is not the one being deployed."""
    model = ModeSpy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    engine.train_one_epoch(
        model, separable_loader(), nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
    )
    assert model.observed_modes and all(model.observed_modes)

    model.observed_modes.clear()
    engine.evaluate(model, separable_loader(), nn.CrossEntropyLoss(), torch.device("cpu"))
    assert model.observed_modes and not any(model.observed_modes)


def test_evaluate_leaves_no_gradients_behind():
    model = TinyNet()
    engine.evaluate(model, separable_loader(), nn.CrossEntropyLoss(), torch.device("cpu"))
    assert all(p.grad is None for p in model.parameters())


def test_epoch_result_reports_the_split_and_epoch():
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = engine.train_one_epoch(
        model, separable_loader(), nn.CrossEntropyLoss(), optimizer, torch.device("cpu"), epoch=7
    )
    assert result.epoch == 7
    assert result.split == "train"
    assert result.seconds > 0
    assert "top1" in result.format()


# --- history --------------------------------------------------------------------


def test_history_best_picks_the_highest_val_top1():
    history = engine.History()
    history.add(engine.EpochResult(0, "val", 1.0, 40.0, 70.0, 1.0))
    history.add(engine.EpochResult(1, "val", 0.9, 55.0, 80.0, 1.0))
    history.add(engine.EpochResult(2, "val", 0.8, 50.0, 78.0, 1.0))
    history.add(engine.EpochResult(1, "train", 0.1, 99.0, 99.0, 1.0))

    best = history.best("val")
    assert best.epoch == 1
    assert best.top1 == 55.0


def test_history_best_is_none_without_entries():
    assert engine.History().best("val") is None


def test_seeding_makes_runs_comparable():
    engine.seed_everything(123)
    a = torch.randn(5)
    engine.seed_everything(123)
    assert torch.equal(a, torch.randn(5))
