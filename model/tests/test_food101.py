"""Tests for the Food-101 label contract.

Label order is silent when wrong. A reordered list trains fine, evaluates fine, and
attaches the wrong dish name to every prediction, so the order is pinned by checksum.
"""

from __future__ import annotations

import json

import pytest

from platevision import food101


def test_labels_load_and_validate():
    data = food101.load_labels()
    assert data["count"] == 101
    assert len(data["labels"]) == 101


def test_indices_are_contiguous_from_zero():
    assert [e["index"] for e in food101.load_labels()["labels"]] == list(range(101))


def test_keys_are_unique():
    keys = food101.class_keys()
    assert len(set(keys)) == len(keys)


def test_every_label_has_a_display_name():
    assert all(e["display"] for e in food101.load_labels()["labels"])


def test_canonical_order_is_not_python_sorted_order():
    """The specific trap: `_` sorts below letters in ASCII, but Food-101 disagrees.

    Regenerating this list with sorted() would put cheese_plate first and silently
    relabel a chunk of the class space.
    """
    keys = food101.class_keys()
    assert keys.index("cheesecake") < keys.index("cheese_plate")
    assert keys != sorted(keys)


def test_first_and_last_classes_are_stable():
    keys = food101.class_keys()
    assert keys[0] == "apple_pie"
    assert keys[-1] == "waffles"


def test_order_digest_matches_the_committed_list():
    data = food101.load_labels()
    assert food101.order_digest(food101.class_keys()) == data["order_sha256"]


def test_reordered_labels_are_rejected(tmp_path):
    data = json.loads(food101.LABELS_PATH.read_text(encoding="utf-8"))
    data["labels"][16], data["labels"][17] = data["labels"][17], data["labels"][16]
    for i, entry in enumerate(data["labels"]):
        entry["index"] = i

    bad = tmp_path / "food101_labels.json"
    bad.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="order digest mismatch"):
        food101.load_labels(bad)


def test_wrong_count_is_rejected(tmp_path):
    data = json.loads(food101.LABELS_PATH.read_text(encoding="utf-8"))
    data["count"] = 100

    bad = tmp_path / "food101_labels.json"
    bad.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="count is"):
        food101.load_labels(bad)


def test_classes_txt_verification_detects_a_mismatch(tmp_path):
    classes = tmp_path / "classes.txt"
    keys = food101.class_keys()
    keys[16], keys[17] = keys[17], keys[16]
    classes.write_text("\n".join(keys), encoding="utf-8")

    with pytest.raises(ValueError, match="index 16"):
        food101.verify_against_classes_txt(classes)


def test_classes_txt_verification_accepts_the_real_order(tmp_path):
    classes = tmp_path / "classes.txt"
    classes.write_text("\n".join(food101.class_keys()) + "\n", encoding="utf-8")
    food101.verify_against_classes_txt(classes)
