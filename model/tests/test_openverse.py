"""Tests for the out-of-distribution manifest builder.

The dedup and cross-label rules matter more than they look. This set exists to produce an
accuracy number on non-Food-101 photos, and a photo carrying two different labels is
guaranteed-wrong ground truth that would silently depress that number.
"""

from __future__ import annotations

import pytest

from platevision import openverse


def image(class_key="apple_pie", label=0, url="https://example.test/a.jpg"):
    return openverse.OpenverseImage(
        class_key=class_key,
        label=label,
        identifier="id",
        url=url,
        license="by",
        license_url="https://creativecommons.org/licenses/by/2.0/",
        attribution="Someone (CC BY 2.0)",
        source="flickr",
        foreign_landing_url="https://flickr.test/photo",
    )


def test_search_term_converts_snake_case():
    assert openverse.search_term("apple_pie") == "apple pie"
    assert openverse.search_term("hot_and_sour_soup") == "hot and sour soup"


def test_search_url_carries_the_licence_filter():
    url = openverse.build_search_url("pizza")
    assert "license_type=commercial%2Cmodification" in url
    assert "q=pizza" in url


def test_search_url_requests_the_anonymous_maximum_by_default():
    assert f"page_size={openverse.MAX_ANONYMOUS_PAGE_SIZE}" in openverse.build_search_url("pizza")


@pytest.mark.parametrize("size", [0, 21, 100])
def test_page_sizes_beyond_the_anonymous_cap_are_rejected(size):
    """The API answers 401 rather than clamping, so this fails early instead of
    consuming part of a 200-request daily budget on a guaranteed rejection."""
    with pytest.raises(ValueError, match="page_size must be"):
        openverse.build_search_url("pizza", page_size=size)


def test_parse_extracts_licence_and_attribution():
    payload = {
        "results": [
            {
                "id": "abc",
                "url": "https://example.test/x.jpg",
                "license": "by-sa",
                "license_url": "https://creativecommons.org/licenses/by-sa/2.0/",
                "attribution": "Photographer (CC BY-SA 2.0)",
                "source": "flickr",
                "foreign_landing_url": "https://flickr.test/x",
            }
        ]
    }
    parsed = openverse.parse_results(payload, "pizza", 76)

    assert len(parsed) == 1
    assert parsed[0].class_key == "pizza"
    assert parsed[0].label == 76
    assert parsed[0].attribution.startswith("Photographer")


def test_parse_skips_entries_without_a_url_or_licence():
    """A row that cannot be downloaded or attributed is worse than a missing one."""
    payload = {
        "results": [
            {"id": "a", "license": "by"},
            {"id": "b", "url": "https://example.test/b.jpg"},
            {"id": "c", "url": "https://example.test/c.jpg", "license": "by"},
        ]
    }
    assert len(openverse.parse_results(payload, "pizza", 0)) == 1


def test_parse_handles_an_empty_response():
    assert openverse.parse_results({}, "pizza", 0) == []


def test_deduplicate_keeps_first_occurrence():
    images = [
        image(url="https://example.test/1.jpg"),
        image(url="https://example.test/1.jpg"),
        image(url="https://example.test/2.jpg"),
    ]
    assert len(openverse.deduplicate(images)) == 2


def test_cross_labelled_images_are_dropped_entirely():
    """A photo matching two dish searches has an unresolvable label. Keeping either copy
    injects ground truth that is known to be wrong for at least one of them."""
    shared = "https://example.test/ambiguous.jpg"
    images = [
        image(class_key="apple_pie", url=shared),
        image(class_key="bread_pudding", url=shared),
        image(class_key="pizza", url="https://example.test/clean.jpg"),
    ]
    kept = openverse.drop_cross_labelled(images)

    assert [i.class_key for i in kept] == ["pizza"]


def test_cross_label_filter_leaves_unique_images_alone():
    images = [
        image(class_key="apple_pie", url="https://example.test/1.jpg"),
        image(class_key="pizza", url="https://example.test/2.jpg"),
    ]
    assert len(openverse.drop_cross_labelled(images)) == 2


def test_review_sample_is_reproducible():
    images = [image(url=f"https://example.test/{i}.jpg") for i in range(50)]
    first = openverse.sample_for_review(images, size=10, seed=1)
    second = openverse.sample_for_review(images, size=10, seed=1)
    assert [i.url for i in first] == [i.url for i in second]


def test_review_sample_does_not_exceed_the_set():
    images = [image(url=f"https://example.test/{i}.jpg") for i in range(5)]
    assert len(openverse.sample_for_review(images, size=100)) == 5


def test_per_class_counts_surface_the_thinnest_classes():
    """Openverse has far more pizza than beet salad. A class with three usable images
    contributes almost nothing and should be visible rather than averaged away."""
    images = [image(class_key="pizza", url=f"https://example.test/p{i}.jpg") for i in range(5)]
    images += [image(class_key="beet_salad", url="https://example.test/b.jpg")]

    counts = openverse.per_class_counts(images)

    assert next(iter(counts)) == "beet_salad"
    assert counts["pizza"] == 5


def test_licence_filter_permits_commercial_use_and_modification():
    """The manifest ships in a public repo and images get resized before inference."""
    assert "commercial" in openverse.LICENSE_FILTER
    assert "modification" in openverse.LICENSE_FILTER
