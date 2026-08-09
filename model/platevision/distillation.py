"""Knowledge distillation.

The student learns from the teacher's full output distribution rather than only the hard
label. The teacher's relative confidence across wrong classes carries information a
one-hot target does not: that a tiramisu photo looks somewhat like chocolate cake and
nothing like a caesar salad.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def kd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Soft-target loss: KL(teacher || student), softened by ``temperature``.

    Two details, both of which fail quietly.

    **The T-squared factor.** Softening logits by T shrinks the gradients of the soft loss
    by roughly 1/T-squared. Without multiplying back, raising the temperature silently
    reduces how much the soft term contributes, so ``alpha`` no longer means the mix it
    claims and tuning temperature secretly tunes the balance too.

    **``batchmean``, not ``mean``.** PyTorch's ``kl_div`` with ``mean`` averages over every
    element, which divides by the class count as well as the batch. On 101 classes that is
    a hundredfold understatement of the soft loss. Nothing errors; the soft term just
    stops mattering.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    divergence = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return divergence * (temperature**2)


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    alpha: float = 0.5
    temperature: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


class Distiller:
    """Holds the frozen teacher and combines the soft and hard losses.

    The teacher is put in eval mode and has gradients disabled. Left in train mode it
    would keep updating its own batch-norm statistics from the student's augmented
    batches, drifting the very reference the student is being measured against.
    """

    def __init__(self, teacher: nn.Module, config: DistillationConfig | None = None) -> None:
        self.config = config or DistillationConfig()
        self.teacher = teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        # Read off the stem rather than assumed, so a teacher that is not three-channel is
        # not silently handed a truncated batch.
        self.in_chans = next((int(p.shape[1]) for p in self.teacher.parameters() if p.ndim == 4), 3)

    def to(self, device: torch.device) -> Distiller:
        self.teacher.to(device)
        return self

    @torch.no_grad()
    def teacher_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Score a batch with the teacher.

        Called on the same augmented images the student sees. Precomputing these once and
        caching them does not work under random augmentation: the cache would hold the
        teacher's opinion of a crop the student never trains on, and the resulting targets
        are wrong in a way the loss curve does not reveal.

        Extra channels are dropped. The teacher is a Food-101 classifier and its stem takes
        three, so a depth-augmented batch kills the run on the first step with a shape error
        several frames deep in timm. Depth is meaningless to it in any case: it was trained
        to tell carbonara from ramen, not to judge how far away the plate is.
        """
        return self.teacher(self._colour_only(images))

    def _colour_only(self, images: torch.Tensor) -> torch.Tensor:
        # Only images have channels. A teacher taking flat feature vectors is a legitimate
        # thing to distil from, and indexing dimension -3 of a 2-D tensor is an IndexError
        # rather than a no-op.
        expected = getattr(self, "in_chans", 3)
        if images.ndim < 3 or images.shape[-3] == expected:
            return images
        return images[..., :expected, :, :]

    def combine(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Blend the soft and hard losses.

        ``hard_loss`` is passed in already computed rather than derived here, so that
        mixup's two-label weighting composes without this class knowing about mixing.
        The teacher is scored on the mixed image too, so its soft target is its genuine
        opinion of the blend and needs no lambda weighting of its own.
        """
        alpha = self.config.alpha
        if alpha == 0.0:
            return hard_loss

        soft = kd_loss(student_logits, teacher_logits, self.config.temperature)
        if alpha == 1.0:
            return soft
        return alpha * soft + (1.0 - alpha) * hard_loss
