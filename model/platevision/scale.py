"""Correcting predictions for a known camera scale, and finding out whether that works.

The scale-reference route rests on an assumption worth testing before anything is built to
serve it: that the damage unknown camera distance does to this model is a *systematic* error
in apparent size, and therefore correctable once the true scale is known.

The alternative is that zooming breaks the representation rather than merely rescaling it,
in which case no plate detector helps however accurate it is, because there is nothing for
its measurement to correct.

The functions here are pure so the arithmetic can be tested without a GPU or a dataset. What
they cannot answer on their own is whether the exponent that best explains the model's
behaviour is close to a physical one, which is what ``scripts/measure_scale_recovery.py``
uses them to find out.
"""

from __future__ import annotations

import torch

# Mass scales with volume, so a linear zoom of f inflates apparent mass by f^3 if the model
# reads the food as a solid, and by f^2 if it reads the area of the plate it covers. Both are
# defensible priors, which is exactly why the exponent is measured rather than assumed.
VOLUME_EXPONENT = 3.0
AREA_EXPONENT = 2.0


def correct_for_scale(
    predictions: torch.Tensor, factors: torch.Tensor, exponent: float
) -> torch.Tensor:
    """Divide real-unit predictions by the scale factor raised to ``exponent``.

    ``factors`` is one linear zoom per sample: greater than one means the food filled more of
    the frame than it would have at the reference distance, so the model saw more food than
    was there and the prediction comes down.

    Broadcasts across targets and quantiles, so the whole interval moves together. Shifting
    the median without the bounds would produce an interval that no longer contains its own
    point estimate.
    """
    if factors.ndim != 1:
        raise ValueError(f"expected one factor per sample, got shape {tuple(factors.shape)}")
    if factors.shape[0] != predictions.shape[0]:
        raise ValueError(f"{factors.shape[0]} factors for {predictions.shape[0]} predictions")
    if (factors <= 0).any():
        raise ValueError("scale factors must be positive")

    shape = (-1,) + (1,) * (predictions.ndim - 1)
    return predictions / factors.reshape(shape).pow(exponent)


def fit_scale_exponent(
    predicted: torch.Tensor, actual: torch.Tensor, factors: torch.Tensor
) -> float:
    """The exponent that best explains the model's error as a function of zoom.

    Least squares in log space on ``log(predicted / actual) = k * log(factor)``, which has a
    closed form. The value is a diagnostic, not a tuned parameter:

    - near 3, the model reads apparent volume and scale correction should work
    - near 2, it reads apparent area
    - **near 0, zoom is not producing a systematic size error at all**, and the accuracy lost
      to unknown camera distance is not recoverable by knowing the distance

    The last case is the one that would cancel the scale-reference route, so it is worth
    being able to see rather than inferring from a correction that fails to help.
    """
    if not (predicted.shape == actual.shape == factors.shape):
        raise ValueError("predicted, actual, and factors must have the same shape")

    # A factor of exactly one carries no information about the exponent and contributes a
    # zero row to both sums, so it is dropped rather than relied on to cancel.
    log_factor = factors.double().clamp(min=1e-12).log()
    usable = log_factor.abs() > 1e-9
    if not usable.any():
        raise ValueError("every factor is 1; there is no zoom to fit an exponent to")

    ratio = (predicted.double().clamp(min=1e-6) / actual.double().clamp(min=1e-6)).log()
    return float((ratio[usable] * log_factor[usable]).sum() / log_factor[usable].pow(2).sum())


def perturb_factors(
    factors: torch.Tensor, relative_error: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """What a real detector would report instead of the true scale.

    Multiplicative rather than additive, because a scale estimate is a ratio: a detector that
    is "10 percent out" is equally likely to say 1.1x or 1/1.1x, and additive noise would make
    overestimates and underestimates different sizes of mistake.
    """
    if relative_error < 0:
        raise ValueError("relative error cannot be negative")
    if relative_error == 0:
        return factors.clone()

    noise = torch.randn(factors.shape, generator=generator, dtype=factors.dtype)
    return factors * (noise * relative_error).exp()
