"""Turning Nutrition5k's raw depth maps into a model input.

The published result this exists to chase: feeding depth to the network as a fourth channel
takes calorie MAE from 70.6 to 47.6 kcal on this dataset, a 33% reduction, and it is what
the market leader ships via iPhone LiDAR.

An earlier experiment here concluded depth does not help. That experiment integrated the
depth map into a volume by hand and used it as a **direct mass predictor**, scoring 104.9 g
against RGB's 71.6 g. Handing the map to the network and letting it learn is a different
proposition and has never been tried.

Three properties of this data shape everything below, measured over 250 random maps:

- **Values are millimetres from the camera**, median 3,562, tightly spread from 3,415 at p5
  to 4,054 at p95. That tightness is the fixed overhead rig, not the food.
- **Roughly 16% of pixels are dropouts**, encoded as zero, reaching 39% on the worst map. A
  zero means "the sensor returned nothing", and it is the one value that must not be read
  literally: as a distance it says the surface is touching the lens.
- **Some pixels saturate to 65535**, which is the same failure at the other end.

Pure functions, so the handling is testable without a GPU or a dataset.
"""

from __future__ import annotations

import numpy as np

# Nothing in a Nutrition5k frame is further than this, and the rig sits near 3.5 m. Clipping
# here turns saturated pixels into "far" rather than letting a single 65535 dominate the
# normalisation of an entire image.
MAX_DEPTH_MM = 6000.0

# Sensor dropouts, encoded as zero.
DROPOUT = 0


def normalise_depth(raw: np.ndarray) -> np.ndarray:
    """A raw uint16 depth map as float32 in [0, 1], dropouts filled.

    Dropouts are replaced with the median of the valid pixels rather than zero or the
    maximum. Zero reads as "against the lens" and the maximum as "infinitely far", and both
    are strong, wrong signals in a region where the sensor simply had nothing to say. The
    median is the least informative value available, which is the point.

    A map with no valid pixels at all returns the midpoint, since there is nothing to take a
    median of and the alternative is a division by zero deep in a training loop.
    """
    depth = raw.astype(np.float32)
    valid = depth > DROPOUT

    if not valid.any():
        return np.full(depth.shape, 0.5, dtype=np.float32)

    filled = np.where(valid, depth, np.median(depth[valid]))
    return np.clip(filled, 0.0, MAX_DEPTH_MM).astype(np.float32) / MAX_DEPTH_MM


def dropout_fraction(raw: np.ndarray) -> float:
    """Share of pixels the sensor did not report. Worth logging: it reaches 39% here."""
    return float((raw == DROPOUT).mean())


def height_above(raw: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Depth re-expressed as height above the surface behind the food, in [0, 1].

    Not used by the four-channel path, which follows the paper and hands over the raw map.
    Offered because the absolute distance on this dataset is nearly constant — the rig never
    moves — so a model can read the table's distance as a free scale cue that a handheld
    phone will not reproduce. Height above the table removes that cue and keeps the shape.

    The reference surface is a high percentile of the valid depths rather than the maximum,
    which would be a saturated pixel on most frames.
    """
    depth = raw.astype(np.float32)
    valid = depth > DROPOUT
    if not valid.any():
        return np.zeros(depth.shape, dtype=np.float32)

    surface = float(np.percentile(depth[valid], percentile))
    height = np.where(valid, surface - depth, 0.0)
    return np.clip(height, 0.0, MAX_DEPTH_MM).astype(np.float32) / MAX_DEPTH_MM
