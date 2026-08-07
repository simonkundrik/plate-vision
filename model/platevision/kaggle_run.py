"""Kernel metadata construction for headless Kaggle runs.

Deliberately free of any ``import kaggle``. The Kaggle python package authenticates at
import time and raises when no credentials are present, which would break CI on every
machine that has the package installed but no token. The CLI is driven by subprocess in
``scripts/run_on_kaggle.py`` instead, and everything testable lives here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
CONFIG_USERNAME_PATTERN = re.compile(r"^\s*-\s*username:\s*(\S+)\s*$", re.MULTILINE)

# P100 is offered by Kaggle but does not work with the default image: the preinstalled
# torch build is not compatible with it. T4 is also the better choice here regardless,
# because the training script uses mixed precision and T4 has tensor cores for it.
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
ACCELERATORS = ("NvidiaTeslaT4", "NvidiaTeslaP100", "NvidiaL4x4", "TpuVmV38")

# Kaggle rejects titles shorter than this.
MIN_TITLE_LENGTH = 5


def validate_slug(slug: str) -> str:
    if not SLUG_PATTERN.match(slug):
        raise ValueError(
            f"invalid kernel slug {slug!r}: use lowercase letters, digits, and single hyphens"
        )
    return slug


def build_kernel_metadata(
    *,
    username: str,
    slug: str,
    code_file: str,
    title: str | None = None,
    enable_gpu: bool = True,
    enable_internet: bool = True,
    is_private: bool = True,
    dataset_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Build the ``kernel-metadata.json`` payload for one notebook run.

    Internet defaults on because the notebook clones this repository and downloads
    Food-101. A run with internet disabled fails in the first cell.
    """
    validate_slug(slug)
    if not username:
        raise ValueError("username is required to build a kernel id")

    resolved_title = title or slug.replace("-", " ")
    if len(resolved_title) < MIN_TITLE_LENGTH:
        raise ValueError(f"title {resolved_title!r} is shorter than {MIN_TITLE_LENGTH} characters")

    return {
        "id": f"{username}/{slug}",
        "title": resolved_title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": is_private,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": enable_internet,
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def write_kernel_metadata(directory: Path, metadata: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "kernel-metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def parse_config_username(raw: str) -> str | None:
    """Pull the username out of ``kaggle config view`` output.

    The OAuth flow writes ``credentials.json`` rather than the legacy ``kaggle.json``, so
    there is no username field to read off disk. ``config view`` prints configuration
    values only and contains no token, which is why it is safe to parse and safe to log.
    """
    match = CONFIG_USERNAME_PATTERN.search(raw)
    if not match:
        return None
    value = match.group(1)
    return None if value.lower() == "none" else value


def parse_status(raw: str) -> str:
    """Pull the status word out of ``kaggle kernels status`` output.

    The CLI prints a sentence rather than a machine-readable field, so this is
    intentionally forgiving about the surrounding wording.
    """
    lowered = raw.lower()
    for state in ("complete", "error", "cancelled", "running", "queued"):
        if state in lowered:
            return state
    return "unknown"


def is_terminal(status: str) -> bool:
    return status in {"complete", "error", "cancelled"}
