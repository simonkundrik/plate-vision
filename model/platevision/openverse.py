"""Openverse search for a CC-licensed out-of-distribution test set.

Food-101 images are curated: mostly well-lit, centred, single-dish. The deployed model sees
handheld photos of home cooking. This builds a second Food-101-labelled test set from
Creative Commons images so that gap can be measured rather than assumed.

**The labels are weak.** They come from search terms, not from anyone looking at the
images. A search for "apple pie" returns mostly apple pie and also illustrations,
close-ups of a single slice, and the occasional unrelated photo whose title happens to
match. Any accuracy computed on this set has a noise floor, and reporting a number without
an estimate of that floor would be misleading. ``sample_for_review`` exists to measure it.

Pure functions only. Network calls live in ``model/data/build_ood_manifest.py``.
"""

from __future__ import annotations

import random
import urllib.parse
from dataclasses import asdict, dataclass

SEARCH_URL = "https://api.openverse.org/v1/images/"

# Anonymous limits, measured against the live API rather than read from docs:
#   20 requests per minute burst, 200 per day sustained, and page_size above 20 is 401.
# One request per class at the maximum page size is 101 requests, which fits the daily
# budget with room for retries. Larger pulls need an API key.
MAX_ANONYMOUS_PAGE_SIZE = 20
BURST_PER_MINUTE = 20
SUSTAINED_PER_DAY = 200

# Commercial use and modification, so the manifest can live in a public repository and the
# images can be resized and re-encoded without a licensing question.
LICENSE_FILTER = "commercial,modification"

USER_AGENT = "plate-vision/0.1 (https://github.com/simonkundrik/plate-vision)"


@dataclass(frozen=True, slots=True)
class OpenverseImage:
    class_key: str
    label: int
    identifier: str
    url: str
    license: str
    license_url: str
    attribution: str
    source: str
    foreign_landing_url: str

    def as_dict(self) -> dict:
        return asdict(self)


def search_term(class_key: str) -> str:
    """Food-101 class keys are snake_case; the search API wants words."""
    return class_key.replace("_", " ")


def build_search_url(
    term: str,
    *,
    page_size: int = MAX_ANONYMOUS_PAGE_SIZE,
    page: int = 1,
    license_type: str = LICENSE_FILTER,
) -> str:
    if not 1 <= page_size <= MAX_ANONYMOUS_PAGE_SIZE:
        raise ValueError(
            f"page_size must be between 1 and {MAX_ANONYMOUS_PAGE_SIZE} without an API key, "
            f"got {page_size}"
        )
    query = urllib.parse.urlencode(
        {
            "q": term,
            "license_type": license_type,
            "page_size": page_size,
            "page": page,
            "mature": "false",
        }
    )
    return f"{SEARCH_URL}?{query}"


def parse_results(payload: dict, class_key: str, label: int) -> list[OpenverseImage]:
    """Turn one API response into image records, skipping anything unusable.

    Entries without a direct URL or a license are dropped rather than carried forward with
    empty fields, because a manifest row that cannot be downloaded or attributed is worse
    than a missing one.
    """
    images: list[OpenverseImage] = []
    for result in payload.get("results", []):
        url = result.get("url")
        license_name = result.get("license")
        if not url or not license_name:
            continue
        images.append(
            OpenverseImage(
                class_key=class_key,
                label=label,
                identifier=str(result.get("id", "")),
                url=url,
                license=license_name,
                license_url=result.get("license_url") or "",
                attribution=result.get("attribution") or "",
                source=result.get("source") or "",
                foreign_landing_url=result.get("foreign_landing_url") or "",
            )
        )
    return images


def deduplicate(images: list[OpenverseImage]) -> list[OpenverseImage]:
    """Drop repeated URLs, keeping first occurrence.

    Searches for neighbouring classes overlap, and the same photo appearing under two
    labels would be scored as wrong at least once no matter what the model predicts.
    """
    seen: set[str] = set()
    unique: list[OpenverseImage] = []
    for image in images:
        if image.url in seen:
            continue
        seen.add(image.url)
        unique.append(image)
    return unique


def drop_cross_labelled(images: list[OpenverseImage]) -> list[OpenverseImage]:
    """Remove any URL that appears under more than one class.

    Stronger than deduplication: if a photo matched two different dish searches, its label
    is unresolvable and keeping either version injects known-bad ground truth.
    """
    counts: dict[str, set[str]] = {}
    for image in images:
        counts.setdefault(image.url, set()).add(image.class_key)
    return [image for image in images if len(counts[image.url]) == 1]


def sample_for_review(
    images: list[OpenverseImage], size: int = 100, seed: int = 0
) -> list[OpenverseImage]:
    """A random sample to hand-check, for estimating the label noise rate.

    Without this the set produces an accuracy figure with no error bar. Checking 100 images
    takes a few minutes and turns "top-1 was 62%" into "top-1 was 62% on a set with roughly
    N% wrong labels", which is a claim that survives being questioned.
    """
    return random.Random(seed).sample(images, min(size, len(images)))


def per_class_counts(images: list[OpenverseImage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image in images:
        counts[image.class_key] = counts.get(image.class_key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1]))
