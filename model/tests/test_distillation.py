"""Tests for knowledge distillation.

Both of the interesting bugs here are silent. A missing T-squared factor makes temperature
secretly control the loss balance; the wrong KL reduction divides the soft loss by the
class count. Neither raises, and neither shows up as anything except a distillation that
mysteriously fails to help.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from platevision.distillation import DistillationConfig, Distiller, kd_loss


def logits(batch=4, classes=6, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, classes)


# --- the soft loss ---------------------------------------------------------------


def test_identical_distributions_give_zero_divergence():
    x = logits()
    assert kd_loss(x, x, temperature=4.0).item() == pytest.approx(0.0, abs=1e-6)


def test_different_distributions_give_positive_divergence():
    assert kd_loss(logits(seed=0), logits(seed=1), temperature=4.0).item() > 0


def test_divergence_is_never_negative():
    for seed in range(20):
        value = kd_loss(logits(seed=seed), logits(seed=seed + 100), temperature=3.0).item()
        assert value >= -1e-6


def test_temperature_squared_scaling_is_applied():
    """Without it, raising the temperature silently shrinks the soft term's contribution
    and alpha stops describing the actual mix."""
    student, teacher = logits(seed=0), logits(seed=1)
    temperature = 5.0

    raw = F.kl_div(
        F.log_softmax(student / temperature, dim=1),
        F.softmax(teacher / temperature, dim=1),
        reduction="batchmean",
    )
    scaled = kd_loss(student, teacher, temperature)

    assert scaled.item() == pytest.approx(raw.item() * temperature**2, rel=1e-5)


def test_reduction_averages_over_the_batch_not_over_every_element():
    """`mean` would divide by the class count too. On 101 classes that understates the
    soft loss a hundredfold, and the soft term simply stops mattering."""
    student, teacher = logits(batch=4, classes=6, seed=0), logits(batch=4, classes=6, seed=1)
    temperature = 2.0

    per_sample = F.kl_div(
        F.log_softmax(student / temperature, dim=1),
        F.softmax(teacher / temperature, dim=1),
        reduction="none",
    ).sum(dim=1)
    expected = per_sample.mean() * temperature**2

    assert kd_loss(student, teacher, temperature).item() == pytest.approx(expected.item(), rel=1e-5)


def test_loss_does_not_shrink_as_the_class_count_grows():
    """The direct symptom of the `mean` bug: a wider head should not weaken the soft loss."""
    torch.manual_seed(0)
    narrow = kd_loss(logits(classes=5, seed=0), logits(classes=5, seed=1), 2.0).item()
    wide = kd_loss(logits(classes=500, seed=0), logits(classes=500, seed=1), 2.0).item()
    assert wide > narrow / 10


def test_gradients_flow_to_the_student_only():
    student = logits().requires_grad_(True)
    teacher = logits(seed=1)
    kd_loss(student, teacher, 3.0).backward()

    assert student.grad is not None
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_non_positive_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature must be positive"):
        kd_loss(logits(), logits(seed=1), temperature)


# --- config ----------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha must be"):
        DistillationConfig(alpha=alpha)


def test_invalid_temperature_in_config_is_rejected():
    with pytest.raises(ValueError, match="temperature must be positive"):
        DistillationConfig(temperature=0.0)


# --- the distiller ---------------------------------------------------------------


def teacher_model(classes=6):
    torch.manual_seed(7)
    return nn.Sequential(nn.Flatten(), nn.Linear(12, classes))


def test_teacher_is_frozen_and_in_eval_mode():
    """Left in train mode the teacher keeps updating its own batch-norm statistics from
    the student's augmented batches, drifting the reference being learned from."""
    distiller = Distiller(teacher_model())
    assert not distiller.teacher.training
    assert all(not p.requires_grad for p in distiller.teacher.parameters())


def test_teacher_logits_carry_no_gradient():
    distiller = Distiller(teacher_model())
    out = distiller.teacher_logits(torch.randn(4, 12))
    assert not out.requires_grad


def test_teacher_weights_do_not_change_when_the_student_trains():
    distiller = Distiller(teacher_model())
    before = [p.clone() for p in distiller.teacher.parameters()]

    student = nn.Sequential(nn.Flatten(), nn.Linear(12, 6))
    optimizer = torch.optim.SGD(student.parameters(), lr=0.5)
    images = torch.randn(8, 12)

    for _ in range(5):
        optimizer.zero_grad()
        loss = distiller.combine(
            student(images),
            distiller.teacher_logits(images),
            torch.tensor(0.0, requires_grad=True),
        )
        loss.backward()
        optimizer.step()

    for a, b in zip(before, distiller.teacher.parameters(), strict=True):
        assert torch.equal(a, b)


def test_alpha_zero_returns_the_hard_loss_untouched():
    distiller = Distiller(teacher_model(), DistillationConfig(alpha=0.0))
    hard = torch.tensor(1.234)
    assert distiller.combine(logits(), logits(seed=1), hard) is hard


def test_alpha_one_ignores_the_hard_loss():
    distiller = Distiller(teacher_model(), DistillationConfig(alpha=1.0, temperature=3.0))
    student, teacher = logits(seed=0), logits(seed=1)

    combined = distiller.combine(student, teacher, torch.tensor(99.0))
    assert combined.item() == pytest.approx(kd_loss(student, teacher, 3.0).item())


def test_alpha_blends_the_two_losses():
    config = DistillationConfig(alpha=0.25, temperature=2.0)
    distiller = Distiller(teacher_model(), config)
    student, teacher = logits(seed=0), logits(seed=1)
    hard = torch.tensor(2.0)

    expected = 0.25 * kd_loss(student, teacher, 2.0).item() + 0.75 * 2.0
    assert distiller.combine(student, teacher, hard).item() == pytest.approx(expected, rel=1e-5)


def test_student_learns_to_imitate_the_teacher():
    """End-to-end: pure soft loss should pull the student's outputs toward the teacher's."""
    torch.manual_seed(0)
    teacher = teacher_model()
    distiller = Distiller(teacher, DistillationConfig(alpha=1.0, temperature=2.0))

    student = nn.Sequential(nn.Flatten(), nn.Linear(12, 6))
    optimizer = torch.optim.Adam(student.parameters(), lr=0.1)
    images = torch.randn(32, 12)

    with torch.no_grad():
        target = teacher(images)
    before = F.mse_loss(student(images), target).item()

    for _ in range(150):
        optimizer.zero_grad()
        distiller.combine(
            student(images), distiller.teacher_logits(images), torch.tensor(0.0)
        ).backward()
        optimizer.step()

    after = F.mse_loss(student(images), target).item()
    assert after < before
