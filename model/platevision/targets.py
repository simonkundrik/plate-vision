"""Nutrition target transformation.

Calories in this dataset run from 0 to 3,943 with a median of 209, so the distribution is
positive and heavily right-skewed. Regressing on the raw scale lets a handful of large
plates dominate the loss.

The transform is log1p followed by standardisation, fitted on the training split only.

Why log1p specifically, given the head predicts quantiles: quantiles are equivariant under
any monotonically increasing transform. The q-th quantile of log1p(Y) is exactly
log1p of the q-th quantile of Y. So the head can be trained entirely in log space and the
predictions inverted afterwards without the interval losing its meaning. That is not true
of the mean, which is why this trick works for quantile regression and would quietly bias
an ordinary least-squares head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from platevision import meta


@dataclass(frozen=True, slots=True)
class TargetTransform:
    """log1p then standardise, per target.

    Fitted on the training split only. Fitting on train plus test leaks the test
    distribution into training, which inflates results in a way that is invisible in the
    loss curve.

    The inverse must eventually be baked into the exported ONNX graph, for the same reason
    preprocessing is: clients should receive kilocalories, not standardised log-space
    values that they have to remember to un-transform.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (len(self.mean) == len(self.std) == len(self.keys)):
            raise ValueError("mean, std, and keys must have equal length")
        if any(s <= 0 for s in self.std):
            raise ValueError("standard deviations must be positive")

    @classmethod
    def fit(cls, target_rows, keys: tuple[str, ...] | None = None) -> TargetTransform:
        """Fit from an iterable of target tuples, one per training sample."""
        rows = [tuple(float(v) for v in row) for row in target_rows]
        if not rows:
            raise ValueError("cannot fit a target transform on zero samples")

        keys = keys or tuple(meta.target_keys())
        width = len(keys)
        if any(len(row) != width for row in rows):
            raise ValueError(f"every target row must have {width} values")
        if any(v < 0 for row in rows for v in row):
            raise ValueError("negative target encountered; log1p is undefined below -1")

        means: list[float] = []
        stds: list[float] = []
        for col in range(width):
            logged = [math.log1p(row[col]) for row in rows]
            mu = sum(logged) / len(logged)
            var = sum((v - mu) ** 2 for v in logged) / len(logged)
            sigma = math.sqrt(var)
            # A constant column would give sigma 0 and divide by zero. Fall back to 1.0,
            # which leaves the column centred and harmlessly unscaled.
            means.append(mu)
            stds.append(sigma if sigma > 1e-8 else 1.0)

        return cls(mean=tuple(means), std=tuple(stds), keys=tuple(keys))

    def forward(self, targets):
        """Raw units to standardised log space."""
        import torch

        mean = torch.as_tensor(self.mean, dtype=targets.dtype, device=targets.device)
        std = torch.as_tensor(self.std, dtype=targets.dtype, device=targets.device)
        return (torch.log1p(targets) - mean) / std

    def inverse(self, values):
        """Standardised log space back to raw units."""
        import torch

        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return torch.expm1(values * std + mean)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": list(self.mean), "std": list(self.std), "keys": list(self.keys)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetTransform:
        return cls(
            mean=tuple(data["mean"]),
            std=tuple(data["std"]),
            keys=tuple(data["keys"]),
        )
