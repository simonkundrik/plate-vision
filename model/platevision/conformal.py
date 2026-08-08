"""Conformalized quantile regression: intervals that cover what they claim to cover.

A quantile head trained with pinball loss produces intervals whose stated level is an
aspiration, not a guarantee. This model's 90% interval covered 64.6% of the test set: the
median was well calibrated and both bounds were pulled inward, which is what overfitting
looks like when you are predicting quantiles rather than points.

Split conformal prediction (Romano, Patterson, Candès 2019) fixes that after training. On a
calibration set the model has never been fitted to, measure how far outside its own
interval the truth actually falls, then widen every interval by the appropriate quantile of
those misses. The result has a finite-sample marginal coverage guarantee that holds for any
model, however badly calibrated, and assumes only that calibration and test data are
exchangeable.

The cost is sharpness. A conformalized interval is wider, and honestly so: the width is the
model's uncertainty as measured rather than as hoped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def conformity_scores(
    predictions: torch.Tensor, targets: torch.Tensor, lower: int = 0, upper: int = -1
) -> torch.Tensor:
    """How far outside its interval each truth fell, in real units.

    Negative when the truth sat comfortably inside, which matters: those points are what
    allow a well-calibrated model's intervals to be *narrowed* rather than only widened.

    ``predictions`` is (n, targets, quantiles) in real units, ``targets`` is (n, targets).
    """
    if predictions.ndim != 3:
        raise ValueError(
            f"expected (n, targets, quantiles) predictions, got {tuple(predictions.shape)}"
        )
    if targets.shape != predictions.shape[:2]:
        raise ValueError(
            f"targets {tuple(targets.shape)} do not match "
            f"predictions {tuple(predictions.shape[:2])}"
        )

    low = predictions[..., lower]
    high = predictions[..., upper]
    return torch.maximum(low - targets, targets - high)


def conformal_offset(scores: torch.Tensor, alpha: float = 0.10) -> torch.Tensor:
    """The amount to widen each target's interval by, one value per target.

    Uses the ceil((n+1)(1-alpha))/n empirical quantile rather than the plain (1-alpha)
    quantile. The +1 is what turns an asymptotic statement into a finite-sample guarantee,
    and with a few hundred calibration points the difference is not academic.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = scores.shape[0]
    if n < 2:
        raise ValueError("conformal calibration needs at least 2 points")

    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        # Too few points to certify this level. Falling back to the largest observed miss
        # is the most conservative honest answer; pretending otherwise would state a
        # guarantee the sample cannot support.
        rank = n

    sorted_scores, _ = torch.sort(scores, dim=0)
    return sorted_scores[rank - 1]


def apply_offset(predictions: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Widen the outer quantiles by the offset, leaving the median alone.

    The median is a point estimate and conformal prediction says nothing about it. Shifting
    it would degrade the one number that was already well calibrated.
    """
    widened = predictions.clone()
    widened[..., 0] = widened[..., 0] - offset
    widened[..., -1] = widened[..., -1] + offset
    return widened


@dataclass(frozen=True, slots=True)
class ConformalCalibration:
    """Per-target interval adjustments, fitted on held-out data.

    Travels in the model bundle so every client widens identically. A client that applied
    the raw model output would report the same overconfident intervals this exists to fix.
    """

    keys: list[str]
    offsets: list[float]
    alpha: float
    calibration_size: int

    @classmethod
    def fit(
        cls,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        keys: list[str],
        alpha: float = 0.10,
    ) -> ConformalCalibration:
        scores = conformity_scores(predictions, targets)
        offsets = conformal_offset(scores, alpha)
        return cls(
            keys=list(keys),
            offsets=[float(v) for v in offsets],
            alpha=float(alpha),
            calibration_size=int(predictions.shape[0]),
        )

    def as_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(self.offsets, dtype=torch.float32, device=device)

    def apply(self, predictions: torch.Tensor) -> torch.Tensor:
        if predictions.shape[1] != len(self.offsets):
            raise ValueError(
                f"predictions carry {predictions.shape[1]} targets but this calibration "
                f"covers {len(self.offsets)} ({', '.join(self.keys)})"
            )
        return apply_offset(predictions, self.as_tensor(predictions.device))

    def to_dict(self) -> dict[str, Any]:
        return {
            "keys": self.keys,
            "offsets": self.offsets,
            "alpha": self.alpha,
            "calibration_size": self.calibration_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConformalCalibration:
        return cls(
            keys=list(payload["keys"]),
            offsets=[float(v) for v in payload["offsets"]],
            alpha=float(payload["alpha"]),
            calibration_size=int(payload["calibration_size"]),
        )
