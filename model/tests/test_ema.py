"""Tests for the weight EMA.

Two silent failure modes drive these: an average that skips batch-norm buffers, and one
that tries to average an integer counter.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from platevision.ema import ModelEma


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))


def test_shadow_starts_equal_to_the_model():
    model = make_model()
    ema = ModelEma(model)
    for a, b in zip(model.parameters(), ema.module.parameters(), strict=True):
        assert torch.equal(a, b)


def test_shadow_is_detached_from_autograd():
    ema = ModelEma(make_model())
    assert all(not p.requires_grad for p in ema.module.parameters())


def test_shadow_does_not_alias_the_live_model():
    """A deepcopy, not a reference. Aliasing would make the EMA a no-op that looks fine."""
    model = make_model()
    ema = ModelEma(model)
    with torch.no_grad():
        model[0].weight.add_(1.0)
    assert not torch.equal(model[0].weight, ema.module[0].weight)


def test_update_moves_the_shadow_toward_the_live_model():
    model = make_model()
    ema = ModelEma(model, decay=0.5, warmup=False)
    before = ema.module[0].weight.clone()

    with torch.no_grad():
        model[0].weight.add_(2.0)
    ema.update(model)

    after = ema.module[0].weight
    assert not torch.equal(after, before)
    # Halfway, since decay is 0.5.
    assert torch.allclose(after, before + 1.0, atol=1e-6)


def test_update_never_reaches_the_live_model_in_one_step():
    model = make_model()
    ema = ModelEma(model, decay=0.9, warmup=False)
    with torch.no_grad():
        model[0].weight.add_(10.0)
    ema.update(model)
    assert not torch.allclose(ema.module[0].weight, model[0].weight)


def test_repeated_updates_converge_on_the_live_model():
    model = make_model()
    ema = ModelEma(model, decay=0.5, warmup=False)
    with torch.no_grad():
        model[0].weight.add_(1.0)
    for _ in range(60):
        ema.update(model)
    assert torch.allclose(ema.module[0].weight, model[0].weight, atol=1e-6)


def test_batch_norm_buffers_are_averaged_too():
    """Skipping buffers ships averaged weights with unaveraged statistics."""
    model = make_model()
    ema = ModelEma(model, decay=0.5, warmup=False)
    before = ema.module[1].running_mean.clone()

    with torch.no_grad():
        model[1].running_mean.add_(4.0)
    ema.update(model)

    assert not torch.equal(ema.module[1].running_mean, before)


def test_integer_buffers_are_copied_not_averaged():
    """num_batches_tracked is a counter. Averaging it truncates toward zero on an int
    tensor, so the EMA copy would report a batch count it never saw."""
    model = make_model()
    ema = ModelEma(model, decay=0.5, warmup=False)

    model[1].num_batches_tracked.fill_(100)
    ema.update(model)

    assert ema.module[1].num_batches_tracked.item() == 100


def test_warmup_starts_far_below_the_target_decay():
    """Without warmup the average stays anchored to random initialisation for thousands
    of steps and evaluates worse than the live model, which reads like a broken EMA."""
    ema = ModelEma(make_model(), decay=0.999, warmup=True)
    assert ema.effective_decay() == pytest.approx(1.0 / 10.0)


def test_warmup_climbs_toward_the_target_decay():
    ema = ModelEma(make_model(), decay=0.999, warmup=True)
    early = ema.effective_decay()
    ema.updates = 5000
    assert early < ema.effective_decay() <= 0.999


def test_warmup_never_exceeds_the_configured_decay():
    ema = ModelEma(make_model(), decay=0.9, warmup=True)
    ema.updates = 10_000_000
    assert ema.effective_decay() == pytest.approx(0.9)


def test_warmup_can_be_disabled():
    ema = ModelEma(make_model(), decay=0.9, warmup=False)
    assert ema.effective_decay() == pytest.approx(0.9)


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.5, 1.5])
def test_invalid_decay_is_rejected(decay):
    with pytest.raises(ValueError, match="decay must be"):
        ModelEma(make_model(), decay=decay)


def test_update_counter_increments():
    model = make_model()
    ema = ModelEma(model)
    for _ in range(3):
        ema.update(model)
    assert ema.updates == 3


def test_state_round_trips():
    model = make_model()
    ema = ModelEma(model, decay=0.7, warmup=False)
    with torch.no_grad():
        model[0].weight.add_(1.0)
    ema.update(model)

    restored = ModelEma(make_model(), decay=0.9)
    restored.load_state_dict(ema.state_dict())

    assert restored.updates == ema.updates
    assert restored.decay == pytest.approx(0.7)
    assert torch.equal(restored.module[0].weight, ema.module[0].weight)


def test_set_to_resets_the_shadow():
    model = make_model()
    ema = ModelEma(model, decay=0.5, warmup=False)
    with torch.no_grad():
        model[0].weight.add_(3.0)

    ema.set_to(model)
    assert torch.equal(ema.module[0].weight, model[0].weight)


def test_loading_state_keeps_future_updates_working():
    """The shadow must be rebound after load, or updates silently write to a stale dict."""
    model = make_model()
    ema = ModelEma(model, decay=0.5, warmup=False)
    restored = ModelEma(make_model(), decay=0.5, warmup=False)
    restored.load_state_dict(ema.state_dict())

    before = restored.module[0].weight.clone()
    with torch.no_grad():
        model[0].weight.add_(2.0)
    restored.update(model)

    assert not torch.equal(restored.module[0].weight, before)
