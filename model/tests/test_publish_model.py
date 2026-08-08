"""The checks that stand between a bad artifact and a published release."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from platevision import bundle, export

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load():
    path = REPO_ROOT / "model" / "scripts" / "publish_model.py"
    spec = importlib.util.spec_from_file_location("publish_model", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    """A directory that looks like a real export: artifact plus matching manifest."""
    artifact = tmp_path / "plate-vision-fp32.onnx"
    artifact.write_bytes(b"pretend onnx" * 500)

    digest = export.digest_artifact(artifact)
    manifest = bundle.build_bundle(
        artifact=digest.as_dict(),
        heads_trained={"logits": True, "nutrition_quantiles": False},
        provenance={"classifier_backbone": "efficientnet_b0", "classifier_metric": 86.12},
        quantization=None,
        generated_utc="2026-01-01T00:00:00+00:00",
    )
    (tmp_path / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return tmp_path


class TestVerify:
    def test_accepts_a_matching_pair(self, export_dir: Path):
        publish = _load()
        artifact_path, _, manifest = publish.verify(export_dir)

        assert artifact_path.name == "plate-vision-fp32.onnx"
        assert manifest["schema_version"] == bundle.SCHEMA_VERSION

    def test_rejects_an_artifact_that_changed_after_export(self, export_dir: Path):
        # The failure this exists to stop. Publishing a mismatched pair means every client
        # rejects the download and the error points at the network rather than the upload.
        (export_dir / "plate-vision-fp32.onnx").write_bytes(b"different bytes entirely")

        with pytest.raises(SystemExit, match="does not describe this artifact"):
            _load().verify(export_dir)

    def test_rejects_a_manifest_naming_a_missing_file(self, export_dir: Path):
        (export_dir / "plate-vision-fp32.onnx").unlink()

        with pytest.raises(SystemExit, match="not in"):
            _load().verify(export_dir)

    def test_rejects_a_stale_schema_version(self, export_dir: Path):
        manifest = json.loads((export_dir / "bundle.json").read_text(encoding="utf-8"))
        manifest["schema_version"] = 1
        (export_dir / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(SystemExit, match="schema version"):
            _load().verify(export_dir)

    def test_rejects_a_manifest_with_no_artifact_block(self, export_dir: Path):
        manifest = json.loads((export_dir / "bundle.json").read_text(encoding="utf-8"))
        manifest["artifact"] = "plate-vision-fp32.onnx"  # the schema 1 shape
        (export_dir / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(SystemExit, match="no artifact block"):
            _load().verify(export_dir)

    def test_reports_a_missing_manifest_by_name(self, export_dir: Path):
        (export_dir / "bundle.json").unlink()

        with pytest.raises(SystemExit, match="bundle.json"):
            _load().verify(export_dir)


class TestNotes:
    def test_leads_with_the_untrained_head(self, export_dir: Path):
        publish = _load()
        _, _, manifest = publish.verify(export_dir)

        body = publish.notes(manifest, "model-v0.1.0")

        assert "nutrition head is not" in body.lower()
        # Ahead of the metrics: it is the thing a reader most needs to know.
        assert body.lower().index("nutrition head is not") < body.index("86.12")

    def test_omits_the_warning_when_the_head_is_trained(self, export_dir: Path):
        publish = _load()
        manifest = json.loads((export_dir / "bundle.json").read_text(encoding="utf-8"))
        manifest["heads_trained"]["nutrition_quantiles"] = True

        body = publish.notes(manifest, "model-v0.2.0")

        assert "nutrition head is not" not in body.lower()

    def test_always_carries_the_licence_position(self, export_dir: Path):
        publish = _load()
        _, _, manifest = publish.verify(export_dir)

        body = publish.notes(manifest, "model-v0.1.0")

        # A reader who lands on a release page without reading the repository has to be
        # told here, because there is nowhere else they will look.
        assert "research use only" in body
        assert "Food-101" in body

    def test_records_the_hash_in_full(self, export_dir: Path):
        publish = _load()
        _, _, manifest = publish.verify(export_dir)

        body = publish.notes(manifest, "model-v0.1.0")

        assert manifest["artifact"]["sha256"] in body
