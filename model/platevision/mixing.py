"""Mixup and CutMix.

Written out rather than imported from timm. Both are short, and the details that matter
are exactly the ones a wrapper hides: CutMix has to recompute lambda from the box it
actually cut, and training accuracy stops meaning what it usually means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class MixResult:
    """A mixed batch and the two label sets it was built from.

    Soft targets are represented as a pair plus a weight rather than as a dense
    probability matrix, so the loss composes with label smoothing without either feature
    having to know about the other.
    """

    images: torch.Tensor
    target_a: torch.Tensor
    target_b: torch.Tensor
    lam: float

    @property
    def dominant_target(self) -> torch.Tensor:
        """The label set contributing more than half of each image.

        Training accuracy under mixing is not comparable to evaluation accuracy: the input
        is a blend, so there is no single correct answer. Scoring against the dominant
        label keeps the number interpretable as a trend rather than pretending it is
        accuracy.
        """
        return self.target_a if self.lam >= 0.5 else self.target_b


def rand_bbox(
    height: int, width: int, lam: float, rng: np.random.Generator
) -> tuple[int, int, int, int]:
    """Sample a CutMix box covering ``1 - lam`` of the image area.

    Returns (y1, y2, x1, x2). The box is centred uniformly and clipped to the image, which
    is why the caller must recompute lambda from the clipped result.
    """
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)

    center_y = int(rng.integers(0, height))
    center_x = int(rng.integers(0, width))

    y1 = int(np.clip(center_y - cut_h // 2, 0, height))
    y2 = int(np.clip(center_y + cut_h // 2, 0, height))
    x1 = int(np.clip(center_x - cut_w // 2, 0, width))
    x2 = int(np.clip(center_x + cut_w // 2, 0, width))
    return y1, y2, x1, x2


def mixup_batch(
    images: torch.Tensor, targets: torch.Tensor, alpha: float, rng: np.random.Generator
) -> MixResult:
    """Blend each image with another from the same batch."""
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return MixResult(images=mixed, target_a=targets, target_b=targets[index], lam=lam)


def cutmix_batch(
    images: torch.Tensor, targets: torch.Tensor, alpha: float, rng: np.random.Generator
) -> MixResult:
    """Paste a rectangle from another image in the batch.

    Lambda is recomputed from the box that was actually pasted. The sampled lambda
    describes the intended area, but the box is clipped at the image border, so a box
    centred near an edge covers less than intended. Using the sampled value instead
    mislabels those batches, and it does so silently: the loss still decreases, just
    against slightly wrong targets.
    """
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=images.device)

    height, width = images.shape[-2:]
    y1, y2, x1, x2 = rand_bbox(height, width, lam, rng)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index][:, :, y1:y2, x1:x2]

    pasted_area = (y2 - y1) * (x2 - x1)
    corrected_lam = 1.0 - pasted_area / (height * width)
    return MixResult(images=mixed, target_a=targets, target_b=targets[index], lam=corrected_lam)


class MixingPolicy:
    """Applies mixup or cutmix to a batch, or neither.

    ``prob`` is the chance any mixing happens at all; ``switch_prob`` chooses cutmix over
    mixup when it does. Returning ``None`` for an unmixed batch keeps the caller's fast
    path free of a degenerate ``lam == 1.0`` case.
    """

    def __init__(
        self,
        *,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        if not 0.0 <= switch_prob <= 1.0:
            raise ValueError(f"switch_prob must be in [0, 1], got {switch_prob}")
        if mixup_alpha < 0 or cutmix_alpha < 0:
            raise ValueError("alphas must be non-negative")

        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.rng = np.random.default_rng(seed)

    @property
    def enabled(self) -> bool:
        return self.prob > 0 and (self.mixup_alpha > 0 or self.cutmix_alpha > 0)

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> MixResult | None:
        if not self.enabled or self.rng.random() >= self.prob:
            return None

        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or self.rng.random() < self.switch_prob
        )
        if use_cutmix:
            return cutmix_batch(images, targets, self.cutmix_alpha, self.rng)
        return mixup_batch(images, targets, self.mixup_alpha, self.rng)


def mixed_criterion(criterion: nn.Module, outputs: torch.Tensor, mix: MixResult) -> torch.Tensor:
    """Weighted loss against both label sets.

    Deliberately expressed in terms of the existing criterion rather than a bespoke
    soft-target loss, so label smoothing keeps working unchanged.
    """
    return mix.lam * criterion(outputs, mix.target_a) + (1.0 - mix.lam) * criterion(
        outputs, mix.target_b
    )
