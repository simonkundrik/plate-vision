"""Quantile regression for the nutrition head.

Predicting a single calorie number would be false precision. A photo carries no scale
reference, so portion size is genuinely uncertain, and the honest output is an interval.

Pinball loss gives that directly: train one output per quantile level, and the model learns
where the 5th, 50th, and 95th percentiles of the conditional distribution sit. Whether those
intervals are actually calibrated is then a measurable claim rather than a hope, which is
what the evaluation stage checks.
"""

from __future__ import annotations

import torch
from torch import nn


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pinball (quantile) loss.

    ``predictions`` is (batch, targets, quantiles), ``targets`` is (batch, targets), and
    ``quantiles`` is (quantiles,).

    The loss is asymmetric by design: for the 0.95 quantile, under-predicting costs 0.95
    per unit and over-predicting costs 0.05, so the minimiser sits where 95 percent of the
    mass falls below. A symmetric loss such as MSE would drive every output to the
    conditional mean and the three heads would collapse onto each other.
    """
    if predictions.shape[:-1] != targets.shape:
        raise ValueError(
            f"predictions {tuple(predictions.shape)} and targets {tuple(targets.shape)} "
            "disagree; expected predictions to be targets plus a trailing quantile axis"
        )
    if predictions.shape[-1] != quantiles.numel():
        raise ValueError(
            f"predictions have {predictions.shape[-1]} quantile outputs but "
            f"{quantiles.numel()} quantile levels were given"
        )

    errors = targets.unsqueeze(-1) - predictions
    quantiles = quantiles.to(dtype=predictions.dtype, device=predictions.device)
    losses = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


class PinballLoss(nn.Module):
    """Module wrapper so the loss can be swapped like any other criterion."""

    def __init__(self, quantiles: list[float], reduction: str = "mean") -> None:
        super().__init__()
        if not quantiles:
            raise ValueError("at least one quantile level is required")
        if any(not 0.0 < q < 1.0 for q in quantiles):
            raise ValueError(f"quantiles must lie strictly in (0, 1), got {quantiles}")
        if list(quantiles) != sorted(quantiles):
            raise ValueError("quantile levels must be given in ascending order")

        self.reduction = reduction
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return pinball_loss(predictions, targets, self.quantiles, self.reduction)


def enforce_monotonic(predictions: torch.Tensor) -> torch.Tensor:
    """Sort the quantile axis so the interval bounds cannot cross.

    Nothing in pinball loss couples the three outputs, so a model can predict a 5th
    percentile above its 95th, especially early in training or on inputs unlike anything it
    saw. That produces a negative-width interval, which is not merely ugly: coverage
    statistics computed from it are meaningless.

    Sorting is the standard fix and costs nothing at inference. It is applied for reporting
    and export, not inside the loss, because clamping during training would hide the
    crossings rather than let them be measured.
    """
    return predictions.sort(dim=-1).values


def interval_coverage(
    predictions: torch.Tensor, targets: torch.Tensor, lower: int = 0, upper: int = -1
) -> torch.Tensor:
    """Fraction of targets falling inside the predicted interval, per target dimension.

    This is the number that makes the uncertainty claim checkable. A 90 percent interval
    that contains the truth 60 percent of the time is not a 90 percent interval, and no
    amount of good MAE excuses it.
    """
    low = predictions[..., lower]
    high = predictions[..., upper]
    inside = (targets >= low) & (targets <= high)
    return inside.float().mean(dim=0)
