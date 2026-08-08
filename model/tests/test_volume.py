"""Volume from an overhead depth map.

The module did not deliver what it was written for: measured volume turned out to be a
worse mass estimator than RGB. It is kept because the measurement is the deliverable, and
because the geometry itself is correct even though it is not competitive. These tests pin
the geometry so the negative result stays reproducible rather than becoming folklore.
"""

from __future__ import annotations

import numpy as np
import pytest

from platevision import volume


def flat(depth_value: float, shape=(64, 64)) -> np.ndarray:
    return np.full(shape, depth_value, dtype=np.uint16)


def with_block(plane: float, height: float, size: int = 16, shape=(64, 64)) -> np.ndarray:
    """A plane with a rectangular block standing on it, nearer the camera."""
    depth = flat(plane, shape)
    top, left = (shape[0] - size) // 2, (shape[1] - size) // 2
    depth[top : top + size, left : left + size] = int(plane - height)
    return depth


class TestTablePlane:
    def test_finds_a_flat_table(self):
        assert volume.table_plane(flat(4000)) == pytest.approx(4000)

    def test_ignores_a_block_in_the_middle(self):
        # The plane comes from the border precisely so a large dish cannot drag the estimate
        # towards itself and erase its own height.
        assert volume.table_plane(with_block(4000, 300, size=40)) == pytest.approx(4000)

    def test_ignores_dropout_pixels(self):
        depth = flat(4000)
        depth[:8, :] = volume.INVALID_DEPTH
        assert volume.table_plane(depth) == pytest.approx(4000)

    def test_refuses_a_border_with_no_valid_depth(self):
        depth = flat(4000)
        depth[:8, :] = depth[-8:, :] = depth[:, :8] = depth[:, -8:] = volume.INVALID_DEPTH
        with pytest.raises(ValueError, match="border"):
            volume.table_plane(depth)


class TestHeightMap:
    def test_measures_height_above_the_plane(self):
        heights = volume.height_map(with_block(4000, 300), 4000.0)
        assert heights.max() == pytest.approx(300)

    def test_bare_table_contributes_nothing(self):
        # Without the noise floor the whole tray contributes a thin ever-present slab that
        # scales with the tray's area rather than with anything on it.
        assert volume.height_map(flat(4000), 4000.0).sum() == 0

    def test_suppresses_jitter_below_the_noise_floor(self):
        depth = flat(4000)
        depth[10:20, 10:20] = 3998  # 2mm of sensor noise
        assert volume.height_map(depth, 4000.0, noise_mm=4.0).sum() == 0

    def test_dropouts_are_not_treated_as_tall_food(self):
        # Zero means "the sensor returned nothing". Read literally it is a surface at the
        # camera, which would invent a towering column at every hole.
        depth = with_block(4000, 300)
        depth[0:5, 0:5] = volume.INVALID_DEPTH
        heights = volume.height_map(depth, 4000.0)
        assert heights[0:5, 0:5].sum() == 0


class TestVolumeIndex:
    def test_scales_with_height(self):
        # Heights kept small against the plane on purpose. A tall block's top surface is
        # measurably nearer the camera than a short one's, so its pixels cover less world
        # and the ratio is not exactly two. That is correct physics, not an error, and this
        # test is about the scaling rather than about that second-order effect.
        thin = volume.volume_index(with_block(40000, 50)).index
        thick = volume.volume_index(with_block(40000, 100)).index
        assert thick == pytest.approx(2 * thin, rel=0.01)

    def test_scales_with_area(self):
        small = volume.volume_index(with_block(4000, 200, size=10)).index
        large = volume.volume_index(with_block(4000, 200, size=20)).index
        assert large == pytest.approx(4 * small, rel=0.05)

    def test_a_distant_dish_is_not_counted_as_smaller(self):
        # The depth-squared weighting exists for this. A pixel further from the camera covers
        # more of the world, and summing raw heights would shrink identical food with range.
        near = volume.volume_index(with_block(2000, 20)).index
        far = volume.volume_index(with_block(4000, 20)).index
        assert far == pytest.approx(4 * near, rel=0.05)

    def test_empty_table_measures_nothing(self):
        assert volume.volume_index(flat(4000)).index == 0

    def test_reports_sensor_coverage(self):
        depth = with_block(4000, 200)
        depth[:16, :] = volume.INVALID_DEPTH
        assert volume.volume_index(depth).valid_fraction == pytest.approx(0.75)

    def test_rejects_a_multichannel_image(self):
        with pytest.raises(ValueError, match="single-channel"):
            volume.volume_index(np.zeros((8, 8, 3), dtype=np.uint16))


class TestFitScale:
    def test_recovers_a_known_constant(self):
        indices = np.array([1.0, 2.0, 4.0, 8.0])
        assert volume.fit_scale(indices, indices * 0.25) == pytest.approx(0.25)

    def test_fits_through_the_origin(self):
        # An intercept would let the fit invent food that is not there: zero height has to
        # mean zero volume.
        indices = np.array([1.0, 2.0, 3.0])
        assert volume.fit_scale(indices, indices * 2 + 10) > 2

    def test_refuses_an_all_zero_input(self):
        with pytest.raises(ValueError, match="nothing to fit"):
            volume.fit_scale(np.zeros(4), np.ones(4))

    def test_refuses_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            volume.fit_scale(np.ones(4), np.ones(3))


class TestImpliedDensity:
    def test_computes_grams_per_unit_volume(self):
        density = volume.implied_density(np.array([100.0]), np.array([50.0]), scale=2.0)
        assert density[0] == pytest.approx(1.0)

    def test_returns_nan_for_a_dish_with_no_measured_volume(self):
        # Better than an infinity that quietly poisons a median downstream.
        assert np.isnan(volume.implied_density(np.array([100.0]), np.array([0.0]), 1.0)[0])
