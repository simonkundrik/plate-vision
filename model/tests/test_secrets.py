"""Tests for API key resolution.

Nothing here should make it easy to get a key into the repository or into a log.
"""

from __future__ import annotations

import pytest

from platevision import secrets


def test_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv(secrets.FDC_ENV_VAR, "from-env")
    key, source = secrets.resolve_fdc_key("explicit")
    assert key == "explicit"
    assert "argument" in source


def test_environment_variable_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv(secrets.FDC_ENV_VAR, "from-env")
    monkeypatch.setattr(secrets, "FDC_KEY_FILE", tmp_path / "absent")
    key, source = secrets.resolve_fdc_key()
    assert key == "from-env"
    assert secrets.FDC_ENV_VAR in source


def test_home_file_is_the_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv(secrets.FDC_ENV_VAR, raising=False)
    path = tmp_path / ".fdc_api_key"
    path.write_text("from-file\n", encoding="utf-8")
    monkeypatch.setattr(secrets, "FDC_KEY_FILE", path)

    key, source = secrets.resolve_fdc_key()
    assert key == "from-file"
    assert str(path) in source


def test_whitespace_is_stripped(monkeypatch, tmp_path):
    """A key file written by an editor usually ends with a newline, and a trailing newline
    in a query parameter is a 403 that looks like an invalid key."""
    monkeypatch.delenv(secrets.FDC_ENV_VAR, raising=False)
    path = tmp_path / ".fdc_api_key"
    path.write_text("  spaced-key  \n", encoding="utf-8")
    monkeypatch.setattr(secrets, "FDC_KEY_FILE", path)

    assert secrets.resolve_fdc_key()[0] == "spaced-key"


def test_empty_key_file_is_ignored(monkeypatch, tmp_path):
    monkeypatch.delenv(secrets.FDC_ENV_VAR, raising=False)
    path = tmp_path / ".fdc_api_key"
    path.write_text("\n", encoding="utf-8")
    monkeypatch.setattr(secrets, "FDC_KEY_FILE", path)

    key, source = secrets.resolve_fdc_key()
    assert key == "DEMO_KEY"
    assert "DEMO_KEY" in source


def test_demo_key_is_the_last_resort(monkeypatch, tmp_path):
    monkeypatch.delenv(secrets.FDC_ENV_VAR, raising=False)
    monkeypatch.setattr(secrets, "FDC_KEY_FILE", tmp_path / "absent")
    assert secrets.resolve_fdc_key()[0] == "DEMO_KEY"


def test_key_file_lives_outside_the_repository():
    """In the home directory, not the working tree. A dotfile inside the repo is one
    careless `git add -A` from being published, and .gitignore only helps if the pattern
    was written before the mistake."""
    from pathlib import Path

    assert secrets.FDC_KEY_FILE.parent == Path.home()


def test_redact_removes_the_key_from_text():
    """Request URLs carry the key as a query parameter, so an error quoting the URL would
    leak it into a terminal or a pasted bug report."""
    message = "HTTP 429 for https://api.nal.usda.gov/fdc/v1/foods/search?api_key=abc123&q=x"
    assert "abc123" not in secrets.redact(message, "abc123")
    assert "***" in secrets.redact(message, "abc123")


def test_redact_leaves_the_demo_key_visible():
    """DEMO_KEY is public and shared. Masking it would hide which key was in use."""
    message = "using DEMO_KEY"
    assert secrets.redact(message, "DEMO_KEY") == message


@pytest.mark.parametrize("key", ["", None])
def test_redact_tolerates_no_key(key):
    assert secrets.redact("some text", key or "") == "some text"
