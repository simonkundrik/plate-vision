"""Evaluation metrics and failure analysis for the nutrition model.

Built around one claim that has to survive scrutiny: the model predicts an interval, and
that interval is calibrated. Everything here exists to check that rather than assert it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

# Percentage errors need a denominator floor regardless of the metric, but the floor is
# not the real fix; see smape and median_ape below.
EPS = 1e-6


def absolute_errors(
    predictions: torch.Tensor, targets: torch.Tensor, median_index: int
) -> torch.Tensor:
    """Per-sample absolute error of the median prediction, shape (batch, targets)."""
    return (predictions[..., median_index] - targets).abs()


def mean_absolute_error(errors: torch.Tensor) -> torch.Tensor:
    return errors.mean(dim=0)


def median_absolute_error(errors: torch.Tensor) -> torch.Tensor:
    return errors.median(dim=0).values


def mean_absolute_percentage_error(
    predictions: torch.Tensor, targets: torch.Tensor, median_index: int
) -> torch.Tensor:
    """Classic MAPE. Reported for comparability, but it is the wrong headline here.

    Nutrition5k calories span 2 to 3,943 with a median near 208. A dish labelled 2 kcal
    and predicted at 100 contributes a 4,900 percent error on its own, and a handful of
    those dominate the mean. MAPE can read above 100 percent while the typical error is
    half the median target, which describes the metric rather than the model.
    """
    medians = predictions[..., median_index]
    denominator = targets.abs().clamp(min=EPS)
    return ((medians - targets).abs() / denominator).mean(dim=0)


def median_absolute_percentage_error(
    predictions: torch.Tensor, targets: torch.Tensor, median_index: int
) -> torch.Tensor:
    """Median of the per-sample percentage errors.

    The honest headline for a skewed target: it describes the typical dish rather than
    being dragged around by a handful of tiny-denominator ones. Insensitive to exactly the
    outliers that make MAPE unusable here.
    """
    medians = predictions[..., median_index]
    denominator = targets.abs().clamp(min=EPS)
    relative = (medians - targets).abs() / denominator
    return relative.median(dim=0).values


def symmetric_mape(
    predictions: torch.Tensor, targets: torch.Tensor, median_index: int
) -> torch.Tensor:
    """SMAPE: 2|p - t| / (|p| + |t|).

    Bounded in [0, 2] no matter how small the target, because the prediction is in the
    denominator too. A tiny target can no longer produce an unbounded term, so no
    arbitrary floor is needed to keep the average finite.
    """
    medians = predictions[..., median_index]
    denominator = (medians.abs() + targets.abs()).clamp(min=EPS)
    return (2.0 * (medians - targets).abs() / denominator).mean(dim=0)


def quantile_calibration(
    predictions: torch.Tensor, targets: torch.Tensor, quantiles: list[float]
) -> torch.Tensor:
    """Empirical fraction of targets at or below each predicted quantile.

    The check that makes the uncertainty claim falsifiable. For a well-calibrated model the
    empirical fraction at level q equals q: 5 percent of truths below the 5th percentile
    prediction, 50 percent below the median, 95 percent below the 95th.

    More informative than a single coverage number, because it shows *which* bound is
    wrong. A model can hit 90 percent coverage while its lower bound is far too high and
    its upper bound far too high as well, and the two errors cancel in the coverage figure
    but not here.

    Returns (targets, quantiles), so row i is target i's calibration curve.
    """
    if predictions.shape[-1] != len(quantiles):
        raise ValueError(
            f"predictions have {predictions.shape[-1]} quantile outputs but "
            f"{len(quantiles)} levels were given"
        )
    below = targets.unsqueeze(-1) <= predictions
    return below.float().mean(dim=0)


def interval_width(predictions: torch.Tensor) -> torch.Tensor:
    """Mean width of the predicted interval, per target.

    Reported next to coverage because coverage alone is trivially gamed. A model
    predicting zero to infinity has perfect coverage and no value.
    """
    return (predictions[..., -1] - predictions[..., 0]).mean(dim=0)


@dataclass(slots=True)
class Bucket:
    lower: float
    upper: float
    count: int
    mae: float
    coverage: float


def error_buckets(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    median_index: int,
    target_index: int = 0,
    n_buckets: int = 5,
) -> list[Bucket]:
    """Break error down by the size of the true value.

    A single MAE hides the shape of the failure. Large portions almost always carry larger
    absolute error, and knowing whether the model is uniformly mediocre or specifically bad
    at big plates changes what to do about it.
    """
    truth = targets[:, target_index]
    if truth.numel() == 0:
        return []

    errors = absolute_errors(predictions, targets, median_index)[:, target_index]
    inside = (truth >= predictions[:, target_index, 0]) & (
        truth <= predictions[:, target_index, -1]
    )

    edges = torch.quantile(truth, torch.linspace(0, 1, n_buckets + 1))
    buckets: list[Bucket] = []
    for i in range(n_buckets):
        low, high = edges[i].item(), edges[i + 1].item()
        # Last bucket is closed so the maximum value is not dropped.
        selected = (truth >= low) & (truth <= high if i == n_buckets - 1 else truth < high)
        if not selected.any():
            continue
        buckets.append(
            Bucket(
                lower=low,
                upper=high,
                count=int(selected.sum().item()),
                mae=errors[selected].mean().item(),
                coverage=inside[selected].float().mean().item(),
            )
        )
    return buckets


@dataclass(slots=True)
class FailureCase:
    index: int
    dish_id: str
    truth: float
    predicted: float
    lower: float
    upper: float
    absolute_error: float
    inside_interval: bool


def worst_cases(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    median_index: int,
    dish_ids: list[str],
    target_index: int = 0,
    top_n: int = 10,
) -> list[FailureCase]:
    """The largest absolute errors, with their dish ids so they can be looked at.

    A model card that lists failure modes needs actual failures to look at, not a summary
    statistic. These are the images to open.
    """
    errors = absolute_errors(predictions, targets, median_index)[:, target_index]
    top_n = min(top_n, errors.numel())
    if top_n == 0:
        return []

    order = errors.argsort(descending=True)[:top_n]
    cases: list[FailureCase] = []
    for position in order.tolist():
        low = predictions[position, target_index, 0].item()
        high = predictions[position, target_index, -1].item()
        truth = targets[position, target_index].item()
        cases.append(
            FailureCase(
                index=position,
                dish_id=dish_ids[position] if position < len(dish_ids) else "",
                truth=truth,
                predicted=predictions[position, target_index, median_index].item(),
                lower=low,
                upper=high,
                absolute_error=errors[position].item(),
                inside_interval=bool(low <= truth <= high),
            )
        )
    return cases


@dataclass(slots=True)
class NutritionReport:
    count: int
    target_keys: list[str]
    quantiles: list[float]
    mae: dict[str, float] = field(default_factory=dict)
    median_ae: dict[str, float] = field(default_factory=dict)
    mape: dict[str, float] = field(default_factory=dict)
    median_ape: dict[str, float] = field(default_factory=dict)
    smape: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    mean_interval_width: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, list[float]] = field(default_factory=dict)
    crossing_rate: float = 0.0
    buckets: list[Bucket] = field(default_factory=list)
    worst: list[FailureCase] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def headline(self, key: str = "energy") -> str:
        """One line stating the metric that should be quoted, and why."""
        return (
            f"{key}: MAE {self.mae[key]:.1f}  median APE {self.median_ape[key] * 100:.1f}%  "
            f"SMAPE {self.smape[key] * 100:.1f}%  coverage {self.coverage[key] * 100:.1f}%  "
            f"(MAPE {self.mape[key] * 100:.1f}%, distorted by small denominators)"
        )


def build_report(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_keys: list[str],
    quantiles: list[float],
    dish_ids: list[str] | None = None,
    crossing_rate: float = 0.0,
    top_n: int = 10,
) -> NutritionReport:
    """Assemble every metric from already-inverted, real-unit predictions."""
    median_index = quantiles.index(0.5)
    errors = absolute_errors(predictions, targets, median_index)

    def per_key(values: torch.Tensor) -> dict[str, float]:
        return {key: values[i].item() for i, key in enumerate(target_keys)}

    inside = (targets >= predictions[..., 0]) & (targets <= predictions[..., -1])
    calibration = quantile_calibration(predictions, targets, quantiles)

    return NutritionReport(
        count=int(targets.shape[0]),
        target_keys=list(target_keys),
        quantiles=list(quantiles),
        mae=per_key(mean_absolute_error(errors)),
        median_ae=per_key(median_absolute_error(errors)),
        mape=per_key(mean_absolute_percentage_error(predictions, targets, median_index)),
        median_ape=per_key(median_absolute_percentage_error(predictions, targets, median_index)),
        smape=per_key(symmetric_mape(predictions, targets, median_index)),
        coverage=per_key(inside.float().mean(dim=0)),
        mean_interval_width=per_key(interval_width(predictions)),
        calibration={key: calibration[i].tolist() for i, key in enumerate(target_keys)},
        crossing_rate=crossing_rate,
        buckets=error_buckets(predictions, targets, median_index),
        worst=worst_cases(predictions, targets, median_index, dish_ids or [], top_n=top_n),
    )
