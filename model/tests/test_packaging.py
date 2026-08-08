"""The package has to work when installed, not only from a checkout.

`shared/` sits above this package in the repository. Every contract-touching function read
it by walking up three directories, which is correct from a checkout and wrong the moment
the package is installed: nothing above site-packages is a repository. `pip install
platevision` produced a package where `load_meta()` raised FileNotFoundError, and no test
noticed because every test runs from the checkout.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from platevision import _contract, food101, meta

MODEL_DIR = Path(__file__).resolve().parents[1]
SHARED_DIR = MODEL_DIR.parent / "shared"
PACKAGED_DIR = MODEL_DIR / "platevision" / "_data"

CONTRACT_FILES = ("model_meta.json", "food101_labels.json")


class TestPackagedContract:
    @pytest.mark.parametrize("name", CONTRACT_FILES)
    def test_the_copy_exists(self, name: str):
        assert (PACKAGED_DIR / name).exists(), "run python scripts/sync_contract.py"

    @pytest.mark.parametrize("name", CONTRACT_FILES)
    def test_the_copy_matches_shared(self, name: str):
        # Drift does not crash anything. A moved label ordering renames every prediction,
        # and a stale preprocessing constant produces a model that scores well in Python
        # and returns wrong calories on a phone.
        packaged = json.loads((PACKAGED_DIR / name).read_text(encoding="utf-8"))
        canonical = json.loads((SHARED_DIR / name).read_text(encoding="utf-8"))
        assert packaged == canonical, f"{name} is stale; run python scripts/sync_contract.py"

    def test_sync_script_reports_them_as_current(self):
        result = subprocess.run(
            [sys.executable, str(MODEL_DIR / "scripts" / "sync_contract.py"), "--check"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestResolution:
    def test_prefers_the_repository_copy_from_a_checkout(self):
        # Deliberate: editing shared/ and having an editable install keep serving a stale
        # build-time snapshot is exactly the confusion this project spends its time avoiding.
        assert _contract.contract_path("model_meta.json") == SHARED_DIR / "model_meta.json"

    def test_falls_back_to_the_packaged_copy(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_contract, "REPO_SHARED_DIR", Path("/nonexistent/shared"))
        assert _contract.contract_path("model_meta.json") == PACKAGED_DIR / "model_meta.json"

    def test_never_raises_at_import_time(self, monkeypatch: pytest.MonkeyPatch):
        # A package that cannot be imported at all is a worse failure than one whose loader
        # reports a missing file, and these paths are resolved at module scope.
        monkeypatch.setattr(_contract, "REPO_SHARED_DIR", Path("/nonexistent/shared"))
        monkeypatch.setattr(_contract, "PACKAGED_DIR", Path("/nonexistent/_data"))
        assert _contract.contract_path("model_meta.json").name == "model_meta.json"


class TestInstalledLayout:
    """Copy the package somewhere with no repository above it and use it there."""

    @staticmethod
    @pytest.fixture(scope="class")
    def installed(tmp_path_factory: pytest.TempPathFactory) -> Path:
        root = tmp_path_factory.mktemp("site-packages")
        shutil.copytree(
            MODEL_DIR / "platevision",
            root / "platevision",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        return root

    def _run(self, root: Path, code: str) -> subprocess.CompletedProcess:
        # A subprocess, not an import: this module already imported platevision from the
        # checkout, and sys.modules would serve that copy no matter what sys.path said.
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=tempfile.gettempdir(),
            env={"PYTHONPATH": str(root), "PATH": ""},
            capture_output=True,
            text=True,
        )

    def test_loads_the_contract_with_no_repository_above_it(self, installed: Path):
        result = self._run(
            installed,
            "from platevision import meta;"
            "m = meta.load_meta();"
            "print(m['input']['name'], m['runtime']['onnx_opset'])",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "image 18"

    def test_loads_the_label_list(self, installed: Path):
        result = self._run(
            installed,
            "from platevision import food101;"
            "labels = food101.load_labels()['labels'];"
            "print(len(labels), labels[0]['key'], labels[-1]['key'])",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "101 apple_pie waffles"

    def test_label_checksum_survives_the_copy(self, installed: Path):
        # The ordering checksum is the thing that would catch a re-sorted label list, so a
        # packaged copy that fails it is worse than one that is simply missing.
        result = self._run(
            installed,
            "from platevision import food101; food101.load_labels(); print('ok')",
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


class TestMetadata:
    def test_version_has_one_source(self):
        pyproject = (MODEL_DIR / "pyproject.toml").read_text(encoding="utf-8")
        # Declaring the version in pyproject as well is how a package reports 0.0.0 from
        # __version__ while publishing something else to an index.
        assert 'dynamic = ["version"]' in pyproject
        assert 'path = "platevision/__init__.py"' in pyproject

    def test_version_is_not_a_placeholder(self):
        import platevision

        assert platevision.__version__ != "0.0.0"

    def test_ships_a_typing_marker(self):
        assert (MODEL_DIR / "platevision" / "py.typed").exists()

    def test_carries_its_own_licence(self):
        # The wheel cannot reference ../LICENSE any more than it can reference ../shared.
        assert (MODEL_DIR / "LICENSE").exists()


def test_meta_and_food101_agree_on_where_the_contract_is():
    assert meta.META_PATH.parent == food101.LABELS_PATH.parent
