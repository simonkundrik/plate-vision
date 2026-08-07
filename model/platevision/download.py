"""HTTP helpers for dataset acquisition.

Kept out of ``platevision.nutrition5k`` so that module stays pure and network-free.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

BUCKET_API = "https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o"
OVERHEAD_PREFIX = "nutrition5k_dataset/imagery/realsense_overhead/"

RETRIES = 3
RETRY_BACKOFF_S = 2.0

# Sent on every request. Some APIs reject urllib's default agent outright: Openverse
# answers 403, which looks like an auth or licensing problem rather than a missing header.
# Identifying the client is also simply the polite thing to do when hitting a free service
# a hundred times.
DEFAULT_USER_AGENT = "plate-vision/0.1 (https://github.com/simonkundrik/plate-vision)"


def fetch(url: str, retries: int = RETRIES, headers: dict[str, str] | None = None) -> bytes:
    """GET with a bounded retry on transient failures.

    A 4xx is not retried. The server has given a definitive answer, and retrying a 404
    three times with backoff turns a fast negative into a slow one. That distinction
    matters here: roughly a third of the dish ids in Nutrition5k's own split files have
    no overhead frame, so 404 is an expected outcome rather than an anomaly.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": DEFAULT_USER_AGENT, **(headers or {})}
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:  # noqa: S310
                return resp.read()
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"failed after {retries} attempts: {url}") from last


def iter_overhead_objects() -> Iterator[tuple[str, int]]:
    """Yield (object_name, size_bytes) for everything under the overhead imagery prefix."""
    token: str | None = None
    while True:
        params = {
            "prefix": OVERHEAD_PREFIX,
            "maxResults": "1000",
            "fields": "items(name,size),nextPageToken",
        }
        if token:
            params["pageToken"] = token
        payload = json.loads(fetch(f"{BUCKET_API}?{urllib.parse.urlencode(params)}"))
        for item in payload.get("items", []):
            yield item["name"], int(item["size"])
        token = payload.get("nextPageToken")
        if not token:
            return


def dish_id_from_object(name: str) -> str | None:
    """Extract a dish id from an overhead object name, or None if it is not an RGB frame."""
    if not name.endswith("/rgb.png"):
        return None
    return name.removeprefix(OVERHEAD_PREFIX).removesuffix("/rgb.png")


def available_dish_ids() -> set[str]:
    """Dish ids that actually have an overhead rgb.png in the bucket.

    The split files list ids the overhead camera never captured, so this is the set that
    matters for both downloading and dataset construction.
    """
    ids = set()
    for name, _size in iter_overhead_objects():
        dish_id = dish_id_from_object(name)
        if dish_id:
            ids.add(dish_id)
    return ids
