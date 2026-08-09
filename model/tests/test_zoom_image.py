from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from platevision.datasets import ZoomedDishes, zoom_image


def frame(width: int = 640, height: int = 480) -> Image.Image:
    """A Nutrition5k-shaped frame with a distinct border, so cropping is visible."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    array[8:-8, 8:-8, 1] = 200
    # A non-zero floor everywhere, so "is this pixel black" is a question about the padding
    # rather than about the corner of the fixture, whose red channel starts at zero.
    array[:, :, 2] = 40
    return Image.fromarray(array)


class TestZoomImage:
    def test_the_reference_distance_returns_the_image_untouched(self):
        source = frame()
        assert zoom_image(source, 1.0) is source

    def test_aspect_ratio_survives_the_zoom(self):
        # The confound this replaced: a square crop of a 640x480 frame changes the aspect
        # distortion the eval transform applies at the same time as the scale, and the
        # degradation would then be partly a shape change wearing a scale label.
        for factor in (0.5, 0.8, 1.3, 2.0):
            width, height = zoom_image(frame(), factor).size
            assert width / height == pytest.approx(640 / 480, rel=0.01)

    def test_zooming_in_tightens_the_frame(self):
        assert zoom_image(frame(), 2.0).size == (320, 240)

    def test_zooming_out_widens_past_the_original(self):
        assert zoom_image(frame(), 0.5).size == (1280, 960)

    def test_the_crop_stays_centred(self):
        # Off-centre cropping would move the food as well as resize it, and the model would
        # be penalised for a translation nobody meant to test.
        zoomed = np.asarray(zoom_image(frame(), 2.0))
        source = np.asarray(frame())
        np.testing.assert_array_equal(zoomed, source[120:360, 160:480])

    def test_pixels_outside_the_photograph_replicate_the_edge(self):
        # A flat black border is a cue no camera produces, and the model would learn to read
        # it rather than learning to judge size without it.
        zoomed = np.asarray(zoom_image(frame(), 0.5))
        assert zoomed[0, 0].tolist() == np.asarray(frame())[0, 0].tolist()
        assert zoomed[:, :, 0].max() > 0

    def test_no_black_border_appears_anywhere(self):
        zoomed = np.asarray(zoom_image(frame(), 0.6))
        corners = [zoomed[0, 0], zoomed[0, -1], zoomed[-1, 0], zoomed[-1, -1]]
        assert not any(corner.sum() == 0 for corner in corners)

    def test_rejects_a_non_positive_factor(self):
        with pytest.raises(ValueError, match="must be positive"):
            zoom_image(frame(), 0.0)


class TestZoomedDishes:
    def test_the_reference_level_leaves_every_dish_alone(self):
        dataset = ZoomedDishes([], zoom=1.0)
        assert dataset.factors.size == 0

    def test_factors_straddle_the_reference_distance(self):
        # A half-range: 2.0 means anywhere from twice as far to twice as close, so the
        # simulation does not systematically shrink or enlarge every dish.
        dataset = ZoomedDishes([None] * 500, zoom=2.0)
        assert dataset.factors.min() >= 0.5
        assert dataset.factors.max() <= 2.0
        assert dataset.factors.min() < 1.0 < dataset.factors.max()

    def test_the_same_dish_gets_the_same_factor_at_every_level(self):
        # Comparisons across zoom levels would otherwise be confounded by which dish drew
        # which factor rather than by the zoom itself.
        first = ZoomedDishes([None] * 100, zoom=2.0, seed=7).factors
        second = ZoomedDishes([None] * 100, zoom=2.0, seed=7).factors
        np.testing.assert_array_equal(first, second)

    def test_rejects_a_zoom_below_the_reference(self):
        with pytest.raises(ValueError, match="half-range"):
            ZoomedDishes([], zoom=0.5)
