"""Tests for dataset index construction.

The filters here decide what the model trains on. When they are wrong, nothing raises:
training just runs on a different set of examples than the logs claim.
"""

from __future__ import annotations

import pytest
from PIL import Image

from platevision import datasets, meta

CAFE1 = "dish_metadata_cafe1.csv"
CAFE2 = "dish_metadata_cafe2.csv"


def dish_row(dish_id, calories=300.0, mass=200.0, fat=10.0, carb=20.0, protein=30.0):
    """One metadata row: 6 totals then a single 7-field ingredient group."""
    totals = [f"{v:.6f}" for v in (calories, mass, fat, carb, protein)]
    ingredient = ["ingr_0000000001", "rice", "50.0", "150.0", "1.0", "2.0", "3.0"]
    return ",".join([dish_id, *totals, *ingredient])


@pytest.fixture
def n5k_root(tmp_path):
    """A miniature Nutrition5k tree exercising every filter."""
    root = tmp_path / "nutrition5k"
    (root / "metadata").mkdir(parents=True)
    (root / "splits").mkdir()
    (root / "imagery").mkdir()

    rows = [
        dish_row("dish_ok_1", calories=300.0),
        dish_row("dish_ok_2", calories=500.0),
        dish_row("dish_no_image", calories=400.0),
        dish_row("dish_zero_cal", calories=0.0),
    ]
    (root / "metadata" / CAFE1).write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "metadata" / CAFE2).write_text("", encoding="utf-8")

    # dish_no_metadata is listed in the split but absent from both metadata files.
    listed = ["dish_ok_1", "dish_ok_2", "dish_no_image", "dish_zero_cal", "dish_no_metadata"]
    (root / "splits" / "rgb_train_ids.txt").write_text("\n".join(listed) + "\n", encoding="utf-8")
    (root / "splits" / "rgb_test_ids.txt").write_text("dish_ok_1\n", encoding="utf-8")

    for dish_id in ("dish_ok_1", "dish_ok_2", "dish_zero_cal"):
        d = root / "imagery" / dish_id
        d.mkdir()
        Image.new("RGB", (8, 8), (120, 130, 140)).save(d / "rgb.png")

    return root


def test_index_applies_all_three_filters(n5k_root):
    samples, stats = datasets.build_nutrition5k_index(n5k_root, "train")

    assert [s.dish_id for s in samples] == ["dish_ok_1", "dish_ok_2"]
    assert stats.listed == 5
    assert stats.missing_metadata == 1
    assert stats.missing_image == 1
    assert stats.nonpositive_calories == 1
    assert stats.kept == 2


def test_dropped_counts_account_for_every_listed_id(n5k_root):
    samples, stats = datasets.build_nutrition5k_index(n5k_root, "train")
    accounted = stats.kept + stats.missing_metadata + stats.missing_image
    accounted += stats.nonpositive_calories
    assert accounted == stats.listed
    assert len(samples) == stats.kept


def test_zero_calorie_dishes_can_be_kept_explicitly(n5k_root):
    samples, stats = datasets.build_nutrition5k_index(
        n5k_root, "train", drop_nonpositive_calories=False
    )
    assert "dish_zero_cal" in {s.dish_id for s in samples}
    assert stats.nonpositive_calories == 0


def test_targets_follow_contract_order(n5k_root):
    """Targets are assembled by contract key, not positional order."""
    samples, _ = datasets.build_nutrition5k_index(n5k_root, "train")
    by_key = dict(zip(meta.target_keys(), samples[0].targets, strict=True))

    assert by_key["energy"] == pytest.approx(300.0)
    assert by_key["mass"] == pytest.approx(200.0)
    assert by_key["fat"] == pytest.approx(10.0)
    assert by_key["carbohydrate"] == pytest.approx(20.0)
    assert by_key["protein"] == pytest.approx(30.0)


def test_target_width_matches_the_declared_output_shape(n5k_root):
    samples, _ = datasets.build_nutrition5k_index(n5k_root, "train")
    declared = meta.load_meta()["outputs"]["nutrition_quantiles"]["shape"][1]
    assert len(samples[0].targets) == declared


def test_test_split_reads_the_test_file(n5k_root):
    samples, stats = datasets.build_nutrition5k_index(n5k_root, "test")
    assert [s.dish_id for s in samples] == ["dish_ok_1"]
    assert stats.listed == 1


def test_unknown_split_is_rejected(n5k_root):
    with pytest.raises(ValueError, match="split must be one of"):
        datasets.build_nutrition5k_index(n5k_root, "validation")


def test_image_paths_point_at_files_that_exist(n5k_root):
    samples, _ = datasets.build_nutrition5k_index(n5k_root, "train")
    assert all(s.image_path.is_file() for s in samples)


# --- Food-101 -------------------------------------------------------------------


@pytest.fixture
def food101_root(tmp_path):
    root = tmp_path / "food-101"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "train.txt").write_text(
        "apple_pie/1005649\ncheesecake/1234\ncheese_plate/5678\n", encoding="utf-8"
    )
    (root / "meta" / "test.txt").write_text("waffles/999\n", encoding="utf-8")
    return root


def test_food101_labels_come_from_the_committed_ordering(food101_root):
    from platevision import food101

    samples = datasets.build_food101_index(food101_root, "train")
    keys = food101.class_keys()

    assert [s.label for s in samples] == [
        keys.index("apple_pie"),
        keys.index("cheesecake"),
        keys.index("cheese_plate"),
    ]


def test_food101_paths_get_the_jpg_suffix(food101_root):
    samples = datasets.build_food101_index(food101_root, "train")
    assert samples[0].image_path.name == "1005649.jpg"
    assert samples[0].image_path.parent.name == "apple_pie"


def test_food101_rejects_unknown_class(food101_root):
    (food101_root / "meta" / "train.txt").write_text("not_a_dish/1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown class"):
        datasets.build_food101_index(food101_root, "train")


def test_food101_rejects_malformed_entry(food101_root):
    (food101_root / "meta" / "train.txt").write_text("apple_pie\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed entry"):
        datasets.build_food101_index(food101_root, "train")


# --- out-of-distribution set ------------------------------------------------------


@pytest.fixture
def ood_tree(tmp_path):
    import json

    root = tmp_path / "ood"
    images = root / "images"
    (images / "pizza").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(images / "pizza" / "aaa.jpg")

    manifest = {
        "schema_version": 1,
        "classes_queried": 101,
        "label_quality": "weak",
        "images": [
            {"class_key": "pizza", "label": 76, "identifier": "aaa", "url": "https://x/a.jpg"},
            {"class_key": "pizza", "label": 76, "identifier": "bbb", "url": "https://x/b.jpg"},
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, images


def test_ood_index_skips_entries_not_on_disk(ood_tree):
    """Link rot is expected on third-party hosts, so the count is returned rather than
    surfacing later as an image-load crash partway through evaluation."""
    manifest, images = ood_tree
    samples, missing = datasets.build_ood_index(manifest, images)

    assert len(samples) == 1
    assert missing == 1
    assert samples[0].label == 76


def test_ood_index_label_matches_the_committed_class_ordering(ood_tree):
    """The manifest stores the index from shared/food101_labels.json. If it drifted from
    the ordering the model was trained under, every OOD prediction would be scored against
    the wrong class and the measured drop would be meaningless."""
    from platevision import food101

    manifest, images = ood_tree
    samples, _ = datasets.build_ood_index(manifest, images)

    assert samples[0].label == food101.class_keys().index("pizza")


def test_food101_skips_blank_lines(food101_root):
    (food101_root / "meta" / "train.txt").write_text(
        "apple_pie/1\n\n  \nwaffles/2\n", encoding="utf-8"
    )
    assert len(datasets.build_food101_index(food101_root, "train")) == 2
