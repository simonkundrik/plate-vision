"""Tests for mixup and cutmix.

The load-bearing case is CutMix lambda correction. The sampled lambda describes the box
you meant to paste; the box actually pasted is clipped at the image border. Using the
sampled value mislabels those batches, and it does so quietly: the loss still falls.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from platevision import mixing


@pytest.fixture
def batch():
    images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 3])
    return images, targets


def rng(seed=0):
    return np.random.default_rng(seed)


# --- mixup ----------------------------------------------------------------------


def test_mixup_preserves_shape_and_dtype(batch):
    images, targets = batch
    result = mixing.mixup_batch(images, targets, alpha=0.2, rng=rng())
    assert result.images.shape == images.shape
    assert result.images.dtype == images.dtype


def test_mixup_lambda_is_a_probability(batch):
    images, targets = batch
    for seed in range(20):
        result = mixing.mixup_batch(images, targets, alpha=0.2, rng=rng(seed))
        assert 0.0 <= result.lam <= 1.0


def test_mixup_with_zero_alpha_is_a_no_op(batch):
    """alpha=0 must disable mixing rather than produce a degenerate distribution."""
    images, targets = batch
    result = mixing.mixup_batch(images, targets, alpha=0.0, rng=rng())
    assert result.lam == 1.0
    assert torch.equal(result.images, images)


def test_mixup_output_is_a_convex_combination(batch):
    """Every mixed pixel must lie between the two source pixels."""
    images, targets = batch
    result = mixing.mixup_batch(images, targets, alpha=0.5, rng=rng(3))
    lower = torch.minimum(images.min(), images.min())
    assert (result.images >= lower).all()
    assert (result.images <= images.max()).all()


# --- cutmix ---------------------------------------------------------------------


def test_cutmix_lambda_matches_the_pasted_area():
    """The property the correction exists for, checked directly against the pixels.

    Measured as the maximum over the batch, not over one image. `torch.randperm` has
    fixed points, so an image can be paired with itself and show no changed pixels at all
    while lambda still correctly describes the box. That is harmless for the loss, since
    target_a and target_b are then identical and the weighted sum collapses back to the
    plain loss, but it makes a single-image assertion flaky.
    """
    # Sixteen images, not four: with a small batch the permutation is occasionally the
    # identity, every image is paired with itself, and nothing changes anywhere.
    torch.manual_seed(0)
    images = torch.randn(16, 3, 32, 32)
    targets = torch.arange(16)

    checked = 0
    for seed in range(40):
        result = mixing.cutmix_batch(images, targets, alpha=1.0, rng=rng(seed))
        pasted = 1.0 - result.lam
        if pasted < 0.05:
            continue
        changed = (result.images != images).any(dim=1).flatten(1).float().mean(dim=1)
        assert changed.max().item() == pytest.approx(pasted, abs=0.02)
        checked += 1
    assert checked > 5, "no seed produced a box large enough to measure"


def test_clipping_only_ever_reduces_the_pasted_area():
    """Corrected lambda must never claim more was pasted than actually was.

    A box centred near a border is clipped, so it covers less than the sampled lambda
    intended. Correction can therefore only move lambda upward. Reporting the sampled
    value would systematically overstate the paste on every edge-centred box, and nothing
    downstream would notice: the loss still decreases, against slightly wrong targets.
    """
    images = torch.zeros(2, 3, 32, 32)
    targets = torch.tensor([0, 1])

    strictly_corrected = 0
    for seed in range(200):
        # cutmix_batch draws the beta variate first, so replaying the same seed
        # reproduces the lambda it sampled before clipping.
        sampled = float(np.random.default_rng(seed).beta(1.0, 1.0))
        result = mixing.cutmix_batch(images, targets, alpha=1.0, rng=rng(seed))

        assert result.lam >= sampled - 1e-9
        if result.lam > sampled + 1e-6:
            strictly_corrected += 1

    assert strictly_corrected > 0, "clipping never occurred, so the correction is untested"


def test_cutmix_box_is_always_inside_the_image():
    for seed in range(50):
        y1, y2, x1, x2 = mixing.rand_bbox(64, 48, 0.3, rng(seed))
        assert 0 <= y1 <= y2 <= 64
        assert 0 <= x1 <= x2 <= 48


def test_cutmix_preserves_pixels_outside_the_box(batch):
    images, targets = batch
    result = mixing.cutmix_batch(images, targets, alpha=1.0, rng=rng(7))
    unchanged = result.images == images
    assert unchanged.any(), "cutmix replaced the entire image"


def test_cutmix_with_zero_alpha_pastes_nothing(batch):
    images, targets = batch
    result = mixing.cutmix_batch(images, targets, alpha=0.0, rng=rng())
    assert result.lam == pytest.approx(1.0)
    assert torch.equal(result.images, images)


# --- policy ---------------------------------------------------------------------


def test_policy_returns_none_when_disabled(batch):
    images, targets = batch
    policy = mixing.MixingPolicy(mixup_alpha=0.0, cutmix_alpha=0.0)
    assert not policy.enabled
    assert policy(images, targets) is None


def test_policy_returns_none_when_probability_is_zero(batch):
    images, targets = batch
    policy = mixing.MixingPolicy(prob=0.0)
    assert policy(images, targets) is None


def test_policy_mixes_every_batch_at_probability_one(batch):
    images, targets = batch
    policy = mixing.MixingPolicy(prob=1.0, seed=0)
    assert all(policy(images, targets) is not None for _ in range(10))


def test_policy_uses_both_strategies_over_many_batches(batch):
    """switch_prob must actually switch, or one of the two is dead code."""
    images, targets = batch
    policy = mixing.MixingPolicy(mixup_alpha=0.2, cutmix_alpha=1.0, switch_prob=0.5, seed=1)

    # CutMix leaves a rectangular block identical to another image; mixup blends
    # everything, so no pixel survives untouched. That distinguishes them.
    fully_blended = 0
    for _ in range(40):
        result = policy(images, targets)
        if result is not None and not (result.images == images).any():
            fully_blended += 1
    assert 0 < fully_blended < 40


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prob": 1.5}, "prob must be"),
        ({"switch_prob": -0.1}, "switch_prob must be"),
        ({"mixup_alpha": -1.0}, "non-negative"),
    ],
)
def test_policy_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mixing.MixingPolicy(**kwargs)


def test_policy_is_reproducible_from_a_seed(batch):
    images, targets = batch
    a = mixing.MixingPolicy(seed=42)(images, targets)
    b = mixing.MixingPolicy(seed=42)(images, targets)
    assert a is not None and b is not None
    assert a.lam == pytest.approx(b.lam)


# --- loss and metrics ------------------------------------------------------------


def test_mixed_criterion_weights_both_label_sets(batch):
    images, targets = batch
    criterion = nn.CrossEntropyLoss()
    outputs = torch.randn(4, 5)
    result = mixing.MixResult(images, targets, targets.flip(0), lam=0.25)

    expected = 0.25 * criterion(outputs, targets) + 0.75 * criterion(outputs, targets.flip(0))
    assert mixing.mixed_criterion(criterion, outputs, result) == pytest.approx(expected.item())


def test_mixed_criterion_reduces_to_plain_loss_when_lambda_is_one(batch):
    images, targets = batch
    criterion = nn.CrossEntropyLoss()
    outputs = torch.randn(4, 5)
    result = mixing.MixResult(images, targets, targets.flip(0), lam=1.0)

    assert mixing.mixed_criterion(criterion, outputs, result) == pytest.approx(
        criterion(outputs, targets).item()
    )


def test_mixed_criterion_composes_with_label_smoothing(batch):
    """Expressing the loss in terms of the criterion is what keeps smoothing working."""
    images, targets = batch
    smoothed = nn.CrossEntropyLoss(label_smoothing=0.1)
    plain = nn.CrossEntropyLoss()
    outputs = torch.randn(4, 5)
    result = mixing.MixResult(images, targets, targets.flip(0), lam=0.5)

    assert mixing.mixed_criterion(smoothed, outputs, result) != pytest.approx(
        mixing.mixed_criterion(plain, outputs, result)
    )


@pytest.mark.parametrize(("lam", "expect_a"), [(0.9, True), (0.5, True), (0.1, False)])
def test_dominant_target_follows_lambda(batch, lam, expect_a):
    images, targets = batch
    flipped = targets.flip(0)
    result = mixing.MixResult(images, targets, flipped, lam=lam)
    expected = targets if expect_a else flipped
    assert torch.equal(result.dominant_target, expected)
