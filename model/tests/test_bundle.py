"""The published manifest, and the fixture the TypeScript client parses."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from platevision import bundle, export

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "shared" / "bundle.example.json"


def _load_writer():
    """Import scripts/write_bundle_example.py, which is not an installed module."""
    path = REPO_ROOT / "model" / "scripts" / "write_bundle_example.py"
    spec = importlib.util.spec_from_file_location("write_bundle_example", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(**overrides):
    base = dict(
        artifact={"name": "m.onnx", "bytes": 10, "sha256": "a" * 64},
        heads_trained={"logits": True, "nutrition_quantiles": True},
        provenance={"target_transform": {"mean": [1.0], "std": [2.0], "keys": ["energy"]}},
        quantization=None,
        generated_utc="2026-01-01T00:00:00+00:00",
    )
    base.update(overrides)
    return bundle.build_bundle(**base)


class TestBuildBundle:
    def test_declares_its_schema_version(self):
        assert _manifest()["schema_version"] == bundle.SCHEMA_VERSION

    def test_lifts_the_target_transform_to_the_top_level(self):
        # Without it the nutrition outputs are unitless numbers rather than kilocalories,
        # so a client needs it. Buried inside provenance it reads as a detail of how the
        # model was made rather than something required to interpret its output.
        manifest = _manifest()
        assert manifest["target_transform"]["keys"] == ["energy"]

    def test_target_transform_is_null_when_there_is_none(self):
        manifest = _manifest(provenance={"nutrition_source": "randomly initialised"})
        assert manifest["target_transform"] is None

    def test_records_untrained_heads_rather_than_omitting_them(self):
        # A missing key would read as "unknown" and a client defaulting either way is
        # guessing. False is a claim; absence is not.
        manifest = _manifest(heads_trained={"logits": True, "nutrition_quantiles": False})
        assert manifest["heads_trained"] == {"logits": True, "nutrition_quantiles": False}

    def test_coerces_numpy_style_booleans(self):
        # `bool(...)` in the builder is not decoration: a numpy bool serialises as an
        # object json cannot encode, and the failure would land at write time after a
        # multi-minute export.
        manifest = _manifest(heads_trained={"logits": 1, "nutrition_quantiles": 0})
        assert manifest["heads_trained"]["nutrition_quantiles"] is False

    def test_serialises_to_json(self):
        json.dumps(_manifest())


class TestDigestArtifact:
    def test_reports_size_and_sha256(self, tmp_path: Path):
        payload = b"onnx bytes" * 1000
        path = tmp_path / "model.onnx"
        path.write_bytes(payload)

        digest = export.digest_artifact(path)

        assert digest.name == "model.onnx"
        assert digest.bytes == len(payload)
        assert digest.sha256 == hashlib.sha256(payload).hexdigest()

    def test_chunking_does_not_change_the_hash(self, tmp_path: Path):
        # The chunk size exists so a large artifact is not read into memory. If it changed
        # the digest, every client would reject a correctly downloaded file.
        path = tmp_path / "model.onnx"
        path.write_bytes(bytes(range(256)) * 40)

        assert export.digest_artifact(path, chunk_bytes=7) == export.digest_artifact(path)

    def test_handles_an_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.onnx"
        path.write_bytes(b"")
        assert export.digest_artifact(path).bytes == 0


class TestCommittedExample:
    """The fixture the TypeScript tests read. Regenerate with scripts/write_bundle_example.py."""

    def test_exists(self):
        assert EXAMPLE_PATH.exists(), "run model/scripts/write_bundle_example.py"

    def test_matches_the_builder(self):
        # The point of this test. The client parses the committed file; if this repo
        # renames a key without regenerating it, the client keeps passing against a
        # manifest the exporter no longer produces.
        expected = _load_writer().example()
        assert json.loads(EXAMPLE_PATH.read_text(encoding="utf-8")) == expected

    def test_uses_snake_case_keys(self):
        # Recorded deliberately. The client was originally written expecting camelCase and
        # nothing ever fed it a real manifest, so `headsTrained` came back undefined and
        # every nutrition estimate would have been withheld regardless of the model.
        manifest = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        assert "heads_trained" in manifest
        assert "headsTrained" not in manifest


@pytest.mark.parametrize("field", ["schema_version", "artifact", "heads_trained", "contract"])
def test_example_carries_the_required_fields(field: str):
    manifest = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    assert field in manifest
