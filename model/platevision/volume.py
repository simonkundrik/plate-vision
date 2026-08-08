"""Volume from an overhead depth map.

A single RGB photograph carries no scale, which is why absolute mass resists every purely
visual fix: it is not observable from the input. Depth makes it observable. Integrated
against the table plane, an overhead depth map gives the volume of whatever sits on the
tray, and volume times density is mass.

That reframes what the model should predict. Density is *intensive* and genuinely readable
from appearance; mass is *extensive* and needs geometry. Measuring one and predicting the
other plays to the strengths of each.

Nutrition5k publishes no camera intrinsics, and none are needed to answer whether this
helps. For a pinhole camera the area a pixel covers at depth d is d^2 / (fx*fy), so

    volume = sum(height * d^2) / (fx * fy)

and every unknown collapses into one global constant. :func:`volume_index` returns the sum;
:func:`fit_scale` recovers the constant from labelled masses. Absolute calibration only
becomes necessary when a phone, with its own intrinsics, has to produce a number in cm3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Depth is 16-bit millimetres, and zero means the sensor returned nothing. Roughly a fifth
# of pixels are holes on this rig, so treating zero as "at the camera" would invent a tall
# column of food at every dropout.
INVALID_DEPTH = 0

# The tray occupies the middle of the frame; the border is table. Estimating the plane from
# the border rather than the whole frame stops a large dish from dragging the plane towards
# itself and erasing its own height.
BORDER_FRACTION = 0.12


@dataclass(frozen=True, slots=True)
class VolumeMeasurement:
    """One dish's geometry, in uncalibrated units."""

    index: float
    """Sum of height * depth^2 over the dish, proportional to true volume."""

    plane_mm: float
    """Estimated table depth in millimetres."""

    valid_fraction: float
    """Share of pixels the depth sensor actually returned."""

    height_px: int
    """Pixels standing above the plane by more than the noise floor."""


def table_plane(depth: np.ndarray, border_fraction: float = BORDER_FRACTION) -> float:
    """Depth of the table surface in millimetres, estimated from the frame border.

    The rig is fixed and overhead, so the table is a roughly constant depth and the food is
    strictly nearer the camera. The median of the border is used rather than the mean or a
    high percentile: a few dropout pixels or a stray tray edge should not move it.
    """
    height, width = depth.shape
    band_h = max(1, int(height * border_fraction))
    band_w = max(1, int(width * border_fraction))

    border = np.concatenate(
        [
            depth[:band_h, :].ravel(),
            depth[-band_h:, :].ravel(),
            depth[:, :band_w].ravel(),
            depth[:, -band_w:].ravel(),
        ]
    )
    valid = border[border != INVALID_DEPTH]
    if valid.size == 0:
        raise ValueError("no valid depth pixels on the frame border; cannot locate the table")

    return float(np.median(valid))


def height_map(depth: np.ndarray, plane_mm: float, noise_mm: float = 4.0) -> np.ndarray:
    """Millimetres each pixel stands above the table, zero where it does not.

    ``noise_mm`` suppresses sensor jitter on the bare table. Without it the whole tray
    contributes a thin, ever-present slab of volume that scales with the tray's area rather
    than with anything on it.
    """
    heights = plane_mm - depth.astype(np.float64)
    heights[depth == INVALID_DEPTH] = 0.0
    heights[heights < noise_mm] = 0.0
    return heights


def volume_index(depth: np.ndarray, noise_mm: float = 4.0) -> VolumeMeasurement:
    """Uncalibrated volume: sum of height * depth^2 over the dish.

    Weighting by depth squared is not decoration. A pixel further from the camera covers
    more of the world, and summing raw heights would count a distant dish as smaller than
    an identical near one.
    """
    if depth.ndim != 2:
        raise ValueError(f"expected a single-channel depth map, got shape {depth.shape}")

    plane = table_plane(depth)
    heights = height_map(depth, plane, noise_mm)

    # Depth at the food surface, not at the table: the pixel footprint is set by how far
    # away the thing being measured actually is.
    surface = np.where(depth == INVALID_DEPTH, plane, depth).astype(np.float64)
    contributions = heights * surface**2

    return VolumeMeasurement(
        index=float(contributions.sum()),
        plane_mm=plane,
        valid_fraction=float((depth != INVALID_DEPTH).mean()),
        height_px=int((heights > 0).sum()),
    )


def fit_scale(indices: np.ndarray, volumes_cm3: np.ndarray) -> float:
    """Recover the single constant relating the index to real volume.

    Least squares through the origin, because the relationship is a pure scaling: zero
    height is zero volume, and an intercept would let the fit invent food that is not there.
    """
    indices = np.asarray(indices, dtype=np.float64)
    volumes_cm3 = np.asarray(volumes_cm3, dtype=np.float64)
    if indices.shape != volumes_cm3.shape:
        raise ValueError("indices and volumes must have the same shape")

    denominator = float((indices**2).sum())
    if denominator == 0:
        raise ValueError("all volume indices are zero; nothing to fit")

    return float((indices * volumes_cm3).sum() / denominator)


def implied_density(mass_g: np.ndarray, indices: np.ndarray, scale: float) -> np.ndarray:
    """Grams per unit volume, given a fitted scale. Used to sanity-check the geometry.

    Food density clusters near water, so a plausible distribution here is evidence the
    plane estimate and the depth weighting are behaving. A median far from about 1 g/cm3
    means the geometry is wrong, not that the food is exotic.
    """
    scaled = np.asarray(indices, dtype=np.float64) * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scaled > 0, np.asarray(mass_g, dtype=np.float64) / scaled, np.nan)
