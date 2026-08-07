"""Tests for the acquisition HTTP layer.

The retry policy is the interesting part. Roughly a third of the dish ids in Nutrition5k's
split files have no overhead frame, so 404 is an expected outcome, not an anomaly. Treating
it as transient turns a fast negative into a slow one and stalls the whole download behind
a contiguous block of missing ids.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

from platevision import download


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda _s: None)


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "boom", {}, None)


def test_fetch_returns_body(monkeypatch):
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"hi"))
    assert download.fetch("http://x") == b"hi"


def test_fetch_retries_transient_failures(monkeypatch):
    calls = []

    def flaky(*_a, **_k):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("connection reset")
        return FakeResponse(b"ok")

    monkeypatch.setattr(download.urllib.request, "urlopen", flaky)
    assert download.fetch("http://x") == b"ok"
    assert len(calls) == 3


def test_fetch_gives_up_after_the_retry_budget(monkeypatch):
    calls = []

    def always_fails(*_a, **_k):
        calls.append(1)
        raise urllib.error.URLError("down")

    monkeypatch.setattr(download.urllib.request, "urlopen", always_fails)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        download.fetch("http://x")
    assert len(calls) == 3


@pytest.mark.parametrize("code", [400, 403, 404, 410])
def test_fetch_does_not_retry_client_errors(monkeypatch, code):
    """The regression this file exists for: a 404 must cost one request, not three."""
    calls = []

    def not_found(*_a, **_k):
        calls.append(1)
        raise http_error(code)

    monkeypatch.setattr(download.urllib.request, "urlopen", not_found)
    with pytest.raises(urllib.error.HTTPError):
        download.fetch("http://x")
    assert len(calls) == 1


@pytest.mark.parametrize("code", [500, 502, 503])
def test_fetch_does_retry_server_errors(monkeypatch, code):
    calls = []

    def flaky(*_a, **_k):
        calls.append(1)
        raise http_error(code)

    monkeypatch.setattr(download.urllib.request, "urlopen", flaky)
    with pytest.raises(RuntimeError):
        download.fetch("http://x")
    assert len(calls) == 3


def test_dish_id_extraction():
    name = download.OVERHEAD_PREFIX + "dish_1556572657/rgb.png"
    assert download.dish_id_from_object(name) == "dish_1556572657"


@pytest.mark.parametrize("suffix", ["depth_raw.png", "depth_color.png"])
def test_depth_objects_are_not_treated_as_rgb(suffix):
    name = download.OVERHEAD_PREFIX + f"dish_1556572657/{suffix}"
    assert download.dish_id_from_object(name) is None


def test_available_dish_ids_keeps_only_rgb(monkeypatch):
    objects = [
        (download.OVERHEAD_PREFIX + "dish_a/rgb.png", 1),
        (download.OVERHEAD_PREFIX + "dish_a/depth_raw.png", 1),
        (download.OVERHEAD_PREFIX + "dish_b/depth_color.png", 1),
        (download.OVERHEAD_PREFIX + "dish_c/rgb.png", 1),
    ]
    monkeypatch.setattr(download, "iter_overhead_objects", lambda: iter(objects))
    assert download.available_dish_ids() == {"dish_a", "dish_c"}
