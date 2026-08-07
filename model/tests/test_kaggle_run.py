"""Tests for Kaggle kernel metadata.

Every field here is a toggle that, when wrong, wastes a GPU run and part of a weekly
quota before failing. Internet off means the notebook dies in its first cell, an hour
after being queued.
"""

from __future__ import annotations

import json

import pytest

from platevision import kaggle_run


def test_builds_a_qualified_kernel_id():
    meta = kaggle_run.build_kernel_metadata(
        username="someone", slug="plate-vision-baseline", code_file="nb.ipynb"
    )
    assert meta["id"] == "someone/plate-vision-baseline"


def test_gpu_and_internet_default_on():
    """Internet off fails the first cell, which clones the repo and fetches Food-101."""
    meta = kaggle_run.build_kernel_metadata(
        username="someone", slug="run-one", code_file="nb.ipynb"
    )
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["enable_tpu"] is False


def test_private_by_default():
    meta = kaggle_run.build_kernel_metadata(
        username="someone", slug="run-one", code_file="nb.ipynb"
    )
    assert meta["is_private"] is True


def test_gpu_can_be_disabled():
    meta = kaggle_run.build_kernel_metadata(
        username="someone", slug="run-one", code_file="nb.ipynb", enable_gpu=False
    )
    assert meta["enable_gpu"] is False


def test_title_defaults_from_the_slug():
    meta = kaggle_run.build_kernel_metadata(
        username="someone", slug="plate-vision-baseline", code_file="nb.ipynb"
    )
    assert meta["title"] == "plate vision baseline"


def test_short_titles_are_rejected():
    """Kaggle rejects these server-side, after the upload."""
    with pytest.raises(ValueError, match="shorter than"):
        kaggle_run.build_kernel_metadata(
            username="someone", slug="run-one", code_file="nb.ipynb", title="ab"
        )


@pytest.mark.parametrize(
    "slug",
    ["Plate-Vision", "plate_vision", "plate--vision", "-plate", "plate-", "plate vision", ""],
)
def test_invalid_slugs_are_rejected(slug):
    with pytest.raises(ValueError, match="invalid kernel slug"):
        kaggle_run.validate_slug(slug)


@pytest.mark.parametrize("slug", ["plate-vision", "run1", "plate-vision-baseline-v2"])
def test_valid_slugs_are_accepted(slug):
    assert kaggle_run.validate_slug(slug) == slug


def test_missing_username_is_rejected():
    with pytest.raises(ValueError, match="username is required"):
        kaggle_run.build_kernel_metadata(username="", slug="run-one", code_file="nb.ipynb")


def test_metadata_is_written_as_valid_json(tmp_path):
    meta = kaggle_run.build_kernel_metadata(
        username="someone", slug="run-one", code_file="nb.ipynb"
    )
    path = kaggle_run.write_kernel_metadata(tmp_path / "staging", meta)

    assert path.name == "kernel-metadata.json"
    assert json.loads(path.read_text(encoding="utf-8")) == meta


def test_default_accelerator_is_t4_not_p100():
    """P100 is offered but does not work with Kaggle's default image: the preinstalled
    torch build is incompatible. T4 also has tensor cores, which the --amp path needs."""
    assert kaggle_run.DEFAULT_ACCELERATOR == "NvidiaTeslaT4"
    assert kaggle_run.DEFAULT_ACCELERATOR in kaggle_run.ACCELERATORS


CONFIG_VIEW_OUTPUT = """Configuration values from C:\\Users\\someone\\.kaggle
- username: sk21832
- auth_method: OAUTH
- path: None
- proxy: None
- competition: None
"""


def test_username_is_read_from_config_view():
    """OAuth writes credentials.json, not kaggle.json, so there is no username on disk
    to read. `kaggle config view` reports it and contains no token."""
    assert kaggle_run.parse_config_username(CONFIG_VIEW_OUTPUT) == "sk21832"


def test_config_view_username_of_none_is_treated_as_absent():
    raw = "Configuration values from ~/.kaggle\n- username: None\n- path: None\n"
    assert kaggle_run.parse_config_username(raw) is None


def test_config_view_without_a_username_returns_none():
    assert kaggle_run.parse_config_username("Configuration values\n- path: None\n") is None


def test_config_view_parsing_survives_an_unauthenticated_error():
    assert kaggle_run.parse_config_username("Authentication required to call the API") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('Kernel is complete: "someone/run"', "complete"),
        ("Kernel has status running", "running"),
        ("status: queued", "queued"),
        ("Kernel error encountered", "error"),
        ("The kernel was cancelled", "cancelled"),
        ("something unexpected", "unknown"),
    ],
)
def test_status_parsing(raw, expected):
    assert kaggle_run.parse_status(raw) == expected


@pytest.mark.parametrize("status", ["complete", "error", "cancelled"])
def test_terminal_states_stop_the_poll_loop(status):
    assert kaggle_run.is_terminal(status)


@pytest.mark.parametrize("status", ["running", "queued", "unknown"])
def test_non_terminal_states_keep_polling(status):
    """'unknown' must not be terminal, or a wording change would end the poll early
    and report success for a run that never finished."""
    assert not kaggle_run.is_terminal(status)


def test_module_does_not_import_the_kaggle_package():
    """`import kaggle` authenticates at import time and raises without credentials.
    Importing it here would make the package unimportable in CI."""
    source = (kaggle_run.__file__).replace("\\", "/")
    text = open(source, encoding="utf-8").read()  # noqa: SIM115
    assert "import kaggle\n" not in text
