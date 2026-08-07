"""Locating API keys without ever putting one in the repository.

Keys are read from the environment or from a file in the user's home directory. Nothing is
read from inside the working tree, so there is no path by which a key becomes a commit.

Nothing here logs or returns a key for display. ``describe_source`` exists so a script can
say where the key came from without printing what it is.
"""

from __future__ import annotations

import os
from pathlib import Path

FDC_ENV_VAR = "FDC_API_KEY"

# Deliberately in the home directory, not the repo. A dotfile inside the working tree is
# one careless `git add -A` away from being published, and .gitignore only helps if the
# pattern was written before the mistake.
FDC_KEY_FILE = Path.home() / ".fdc_api_key"


def read_key_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def resolve_fdc_key(explicit: str | None = None, fallback: str = "DEMO_KEY") -> tuple[str, str]:
    """Return (key, where_it_came_from).

    Order is explicit argument, environment variable, home-directory file, then the shared
    demo key. The demo key works but is throttled to roughly 30 requests an hour, so the
    caller is expected to warn rather than silently take four times as long.
    """
    if explicit:
        return explicit, "--api-key argument"
    if value := os.environ.get(FDC_ENV_VAR):
        return value, f"{FDC_ENV_VAR} environment variable"
    if value := read_key_file(FDC_KEY_FILE):
        return value, str(FDC_KEY_FILE)
    return fallback, "DEMO_KEY fallback"


def redact(text: str, key: str) -> str:
    """Remove a key from text before it is printed or written to a log.

    Request URLs carry the key as a query parameter, so an error message quoting the URL
    would leak it into a terminal, a log file, or a pasted bug report.
    """
    if not key or key == "DEMO_KEY":
        return text
    return text.replace(key, "***")
