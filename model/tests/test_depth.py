from __future__ import annotations

import numpy as np
import pytest

from platevision import depth


def frame(values: list[list[int]]) -> np.ndarray:
    return np.array(values, dtype=np.uint16)


class TestNormaliseDepth:
    def test_output_is_float_in_the_unit_range(self):
        raw = frame([[1000, 2000], [3000, 4000]])
        out = depth.normalise_depth(raw)

        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_millimetres_map_onto_the_declared_ceiling(self):
        raw = frame([[3000, 6000]])
        out = depth.normalise_depth(raw)
        assert out[0, 0] == pytest.approx(0.5)
        assert out[0, 1] == pytest.approx(1.0)

    def test_dropouts_become_the_median_not_zero(self):
        # A zero means the sensor returned nothing. Read literally it says the surface is
        # touching the lens, which is the strongest possible wrong signal.
        raw = frame([[0, 3000], [3000, 3000]])
        out = depth.normalise_depth(raw)
        assert out[0, 0] == pytest.approx(out[0, 1])

    def test_a_saturated_pixel_does_not_drag_the_rest(self):
        # 65535 appears in real maps. Without clipping it would compress everything else
        # into a narrow band near zero.
        raw = frame([[65535, 3000], [3000, 3000]])
        out = depth.normalise_depth(raw)

        assert out[0, 0] == pytest.approx(1.0)
        assert out[0, 1] == pytest.approx(0.5)

    def test_a_map_with_no_valid_pixels_returns_the_midpoint(self):
        # The alternative is a median of an empty array, which is a nan that surfaces as a
        # loss going to nan several hundred steps into a run.
        out = depth.normalise_depth(np.zeros((4, 4), dtype=np.uint16))
        assert np.all(out == 0.5)
        assert np.isfinite(out).all()

    def test_shape_survives(self):
        assert depth.normalise_depth(np.zeros((480, 640), dtype=np.uint16)).shape == (480, 640)

    def test_real_maps_normalise_into_a_usable_band(self):
        # Guards the constant against the data: a MAX_DEPTH_MM far above the rig would push
        # every real frame into a sliver of the range and waste most of the channel.
        rig = np.full((8, 8), 3562, dtype=np.uint16)
        assert 0.4 < float(depth.normalise_depth(rig).mean()) < 0.8


class TestDropoutFraction:
    def test_counts_only_zeros(self):
        raw = frame([[0, 0], [1000, 2000]])
        assert depth.dropout_fraction(raw) == pytest.approx(0.5)

    def test_a_clean_map_reports_none(self):
        assert depth.dropout_fraction(frame([[1000, 2000]])) == 0.0


class TestHeightAbove:
    def test_food_nearer_the_camera_becomes_taller(self):
        # The whole point of the transform: a smaller distance is a higher surface.
        raw = frame([[3500, 3500], [3500, 3000]])
        out = depth.height_above(raw)
        assert out[1, 1] > out[0, 0]

    def test_the_reference_surface_reads_as_zero_height(self):
        raw = frame([[3500, 3500], [3500, 3500]])
        assert np.allclose(depth.height_above(raw), 0.0)

    def test_dropouts_contribute_no_height(self):
        raw = frame([[0, 3500], [3500, 3000]])
        assert depth.height_above(raw)[0, 0] == 0.0

    def test_a_saturated_pixel_does_not_define_the_surface(self):
        # Using the maximum rather than a percentile would make one bad pixel the table,
        # and every real height would be measured against it.
        raw = np.full((10, 10), 3500, dtype=np.uint16)
        raw[0, 0] = 65535
        raw[5, 5] = 3000

        out = depth.height_above(raw)
        assert out[5, 5] == pytest.approx(500 / depth.MAX_DEPTH_MM, abs=1e-3)

    def test_no_valid_pixels_gives_no_height(self):
        out = depth.height_above(np.zeros((4, 4), dtype=np.uint16))
        assert np.all(out == 0.0)
        assert np.isfinite(out).all()


class TestFourChannelPath:
    """The plumbing between a depth map and the model, checked where it silently breaks."""

    def test_geometric_augmentation_moves_depth_with_the_pixels(self):
        # If depth were transformed separately, or not at all, each plate would be paired
        # with another plate's geometry and the channel would be worse than absent.
        import torch

        from platevision import transforms

        image = torch.zeros(4, 64, 64, dtype=torch.uint8)
        image[:, :32, :] = 200  # a horizontal edge in every channel, including depth

        flipped = transforms.v2.RandomVerticalFlip(p=1.0)(image)
        assert torch.equal(flipped[3], flipped[0]), "depth parted company with the colour"

    def test_colour_jitter_leaves_the_depth_channel_untouched(self):
        # Brightness and saturation are meaningless for a distance map, and hue rotation
        # would mix it into red.
        import torch

        from platevision import transforms

        jitter = transforms.RgbOnly(
            transforms.v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)
        )
        image = torch.randint(0, 255, (4, 32, 32), dtype=torch.uint8)
        out = jitter(image)

        assert torch.equal(out[3], image[3])
        assert not torch.equal(out[:3], image[:3])

    def test_the_wrapper_passes_three_channel_input_straight_through(self):
        import torch

        from platevision import transforms

        jitter = transforms.RgbOnly(transforms.v2.ColorJitter(brightness=0.5))
        out = jitter(torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8))
        assert out.shape == (3, 16, 16)

    def test_normalisation_gains_a_fourth_pair(self):
        from platevision import transforms

        mean3, std3 = transforms._normalization(3)
        mean4, std4 = transforms._normalization(4)

        assert len(mean3) == 3 and len(mean4) == 4
        assert mean4[:3] == mean3 and std4[:3] == std3
        assert mean4[3] == transforms.DEPTH_MEAN

    def test_an_unsupported_channel_count_raises(self):
        from platevision import transforms

        with pytest.raises(ValueError, match="3 or 4 channels"):
            transforms._normalization(5)

    def test_a_four_channel_model_accepts_four_channels(self):
        import torch

        from platevision import models

        model = models.NutritionModel(
            "mobilenetv3_small_100",
            num_targets=5,
            num_quantiles=3,
            pretrained=False,
            in_chans=4,
        ).eval()

        out = model(torch.zeros(2, 4, 224, 224))
        assert out.shape == (2, 5, 3)

    def test_three_channels_stays_the_default(self):
        from platevision import models

        model = models.NutritionModel(
            "mobilenetv3_small_100", num_targets=5, num_quantiles=3, pretrained=False
        )
        assert model.in_chans == 3


class TestFourChannelWeightTransfer:
    def test_a_three_channel_stem_widens_instead_of_being_skipped(self):
        # Skipping it leaves the first convolution random while every other layer is
        # pretrained, which presents as a depth experiment that failed rather than as a
        # backbone that was never loaded. Measured: 357 tensors transferred instead of 358.
        from platevision import models

        classifier = models.create_classifier(
            "mobilenetv3_small_100", num_classes=5, pretrained=False
        )
        model = models.NutritionModel(
            "mobilenetv3_small_100",
            num_targets=5,
            num_quantiles=3,
            pretrained=False,
            in_chans=4,
        )
        copied, _ = models.load_backbone_weights(model, classifier.state_dict())

        stems = [(k, v) for k, v in model.backbone.state_dict().items() if v.ndim == 4]
        assert stems[0][1].shape[1] == 4
        assert (
            copied
            >= len([v for v in classifier.state_dict().values() if getattr(v, "ndim", 0) > 0]) - 2
        )

    def test_the_colour_filters_survive_widening_exactly(self):
        import torch

        from platevision import models

        source = torch.randn(8, 3, 3, 3)
        target = torch.zeros(8, 4, 3, 3)
        widened = models._adapt_stem(source, target)

        assert torch.equal(widened[:, :3], source)

    def test_the_new_channel_starts_as_the_colour_mean(self):
        # A neutral start: the channel contributes what a greyscale copy would, so training
        # moves it from a working model rather than from noise.
        import torch

        from platevision import models

        source = torch.randn(8, 3, 3, 3)
        widened = models._adapt_stem(source, torch.zeros(8, 4, 3, 3))
        assert torch.allclose(widened[:, 3], source.mean(dim=1))

    def test_a_genuine_shape_disagreement_is_still_refused(self):
        import torch

        from platevision import models

        assert models._adapt_stem(torch.randn(8, 3, 3, 3), torch.zeros(16, 4, 3, 3)) is None
        assert models._adapt_stem(torch.randn(8, 3, 3, 3), torch.zeros(8, 2, 3, 3)) is None

    def test_a_four_channel_checkpoint_restores_as_four_channel(self, tmp_path):
        # Rebuilding a three-channel model for a four-channel checkpoint fails as an opaque
        # state dict error at the very end of a training run, after the GPU time is spent.
        import torch

        from platevision import checkpoint, models
        from platevision.targets import TargetTransform

        model = models.NutritionModel(
            "mobilenetv3_small_100",
            num_targets=5,
            num_quantiles=3,
            pretrained=False,
            in_chans=4,
        )
        path = tmp_path / "depth.pt"
        checkpoint.save_checkpoint(
            path,
            model=model,
            epoch=0,
            backbone="mobilenetv3_small_100",
            config={"target_transform": TargetTransform.fit([(1.0,) * 5, (2.0,) * 5]).to_dict()},
        )

        restored, _, _ = checkpoint.restore_nutrition_model(path)
        assert restored.in_chans == 4
        assert restored(torch.zeros(1, 4, 224, 224)).shape == (1, 5, 3)


class TestDistillationWithDepth:
    def test_the_teacher_is_not_handed_a_depth_channel(self):
        """Phase H died on its first batch here. The teacher is a Food-101 classifier whose
        stem takes three channels, and distillation feeds it the student's batch, so
        --depth killed the run with a shape error several frames deep in timm."""
        import torch

        from platevision import distillation, models

        teacher = models.create_classifier("mobilenetv3_small_100", num_classes=5, pretrained=False)
        distiller = distillation.Distiller(teacher)

        logits = distiller.teacher_logits(torch.zeros(2, 4, 224, 224))
        assert logits.shape == (2, 5)

    def test_a_three_channel_batch_is_untouched(self):
        import torch

        from platevision import distillation, models

        teacher = models.create_classifier("mobilenetv3_small_100", num_classes=5, pretrained=False)
        distiller = distillation.Distiller(teacher)
        assert distiller.in_chans == 3
        assert distiller.teacher_logits(torch.zeros(2, 3, 224, 224)).shape == (2, 5)

    def test_the_channel_count_is_read_off_the_teacher(self):
        # Assuming three would silently truncate a batch for a teacher that wanted four.
        from platevision import distillation, models

        teacher = models.NutritionModel(
            "mobilenetv3_small_100",
            num_targets=5,
            num_quantiles=3,
            pretrained=False,
            in_chans=4,
        )
        assert distillation.Distiller(teacher).in_chans == 4


class TestDepthAwareIndex:
    """dish_1564159636 has a 0-byte depth_raw.png upstream. One broken file in 3,262 should
    cost one dish, not the run: the download exited 1 and the notebook, now that it
    propagates failures, killed phase H2 on the first cell."""

    def _tree(self, tmp_path, dishes, with_depth):
        (tmp_path / "metadata").mkdir(parents=True)
        (tmp_path / "splits").mkdir()
        rows = []  # the real metadata csv has no header row
        for dish in dishes:
            rows.append(f"{dish},100,200,5,10,8")
            imagery = tmp_path / "imagery" / dish
            imagery.mkdir(parents=True)
            (imagery / "rgb.png").write_bytes(b"x")
            if dish in with_depth:
                (imagery / "depth_raw.png").write_bytes(b"x")
        return rows

    def test_a_dish_without_depth_is_dropped_only_when_depth_is_required(self, tmp_path):
        from platevision import datasets, nutrition5k

        dishes = ["dish_a", "dish_b"]
        rows = self._tree(tmp_path, dishes, with_depth={"dish_a"})
        for name in nutrition5k.METADATA_FILES:
            (tmp_path / "metadata" / name).write_text("\n".join(rows), encoding="utf-8")
        (tmp_path / "splits" / "rgb_test_ids.txt").write_text("\n".join(dishes), encoding="utf-8")
        (tmp_path / "splits" / "rgb_train_ids.txt").write_text("", encoding="utf-8")

        without, stats_without = datasets.build_nutrition5k_index(tmp_path, "test")
        with_req, stats_with = datasets.build_nutrition5k_index(
            tmp_path, "test", require_depth=True
        )

        assert len(without) == 2 and stats_without.missing_depth == 0
        assert len(with_req) == 1 and stats_with.missing_depth == 1
        assert with_req[0].dish_id == "dish_a"

    def test_the_count_is_reported_rather_than_swallowed(self, tmp_path):
        # Shrinkage that nobody sees is how a training set quietly loses a third of itself.
        from platevision import datasets

        stats = datasets.IndexStats(
            listed=10, missing_metadata=0, missing_image=0, nonpositive_calories=0, kept=9
        )
        assert stats.missing_depth == 0  # defaulted, so existing constructions still work
