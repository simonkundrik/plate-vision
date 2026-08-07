"""Tests for ONNX export.

The parity test is the point of this file. Everything else in the project measures a
PyTorch model; the app runs an ONNX graph. If those two disagree, every reported number
describes something that was never shipped.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from platevision import export, meta

OUTPUT_NAMES = ["logits", "nutrition_quantiles"]


class TinyCombined(nn.Module):
    """Two heads over a trivial backbone, matching the contract's output shapes."""

    def __init__(self, num_targets=5, num_quantiles=3):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Linear(8, 101)
        self.nutrition_head = nn.Linear(8, num_targets * num_quantiles)
        self.num_targets = num_targets
        self.num_quantiles = num_quantiles

    def forward(self, x):
        features = self.pool(self.stem(x)).flatten(1)
        return (
            self.classifier_head(features),
            self.nutrition_head(features).view(-1, self.num_targets, self.num_quantiles),
        )


@pytest.fixture
def exported(tmp_path):
    model = TinyCombined().eval()
    path = tmp_path / "model.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export.export_onnx(model, path, output_names=OUTPUT_NAMES)
    return model, path


# --- the wrapper ------------------------------------------------------------------


def test_wrapper_reproduces_the_documented_preprocessing_order():
    """Float conversion, then resize, then normalise, exactly as the contract states."""
    model = nn.Identity()
    wrapper = export.ExportWrapper(model).eval()
    image = export.example_input(97, 143)

    height, width = meta.input_size()
    mean, std = meta.normalization()
    expected = image.permute(0, 3, 1, 2).float() / 255.0
    expected = F.interpolate(
        expected, size=(height, width), mode="bilinear", align_corners=False, antialias=True
    )
    expected = (expected - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

    with torch.no_grad():
        assert torch.allclose(wrapper(image), expected, atol=1e-6)


def test_wrapper_accepts_uint8_nhwc():
    wrapper = export.ExportWrapper(nn.Identity()).eval()
    with torch.no_grad():
        out = wrapper(export.example_input(64, 80))
    height, width = meta.input_size()
    assert out.shape == (1, 3, height, width)


def test_wrapper_rejects_an_unimplemented_preprocessing_order(monkeypatch):
    monkeypatch.setattr(meta, "preprocessing_order", lambda: ("resize", "normalize"))
    with pytest.raises(ValueError, match="unsupported preprocessing order"):
        export.ExportWrapper(nn.Identity())


def test_example_input_is_not_already_the_target_size():
    """Exporting at 224x224 would trace a graph whose resize is a no-op, and the export
    would look successful while doing nothing."""
    example = export.example_input()
    height, width = meta.input_size()
    assert example.shape[1] != height
    assert example.shape[2] != width
    assert example.dtype == torch.uint8


# --- the export -------------------------------------------------------------------


def test_export_produces_a_single_self_contained_file(exported):
    """The dynamo exporter writes a .onnx.data sidecar by default. The app bundles one
    asset, and shipping the .onnx alone fails at load time saying nothing about it."""
    _, path = exported
    assert path.is_file()
    assert not path.with_suffix(".onnx.data").exists()
    assert not list(path.parent.glob("*.data"))


def test_exported_graph_matches_pytorch(exported):
    """The test the whole in-graph preprocessing argument rests on."""
    model, path = exported
    results = export.check_parity(model, path, output_names=OUTPUT_NAMES)

    assert len(results) == 2
    for result in results:
        assert result.within_tolerance, (
            f"{result.output_name} differs by {result.max_absolute_difference}"
        )


def test_parity_reports_the_actual_difference(exported):
    model, path = exported
    results = export.check_parity(model, path, output_names=OUTPUT_NAMES)
    assert all(r.max_absolute_difference < 1e-4 for r in results)


def test_antialiased_resize_survived_the_export(exported):
    """Antialias is why this uses the dynamo exporter. If the Resize op vanished, the
    graph is downscaling differently from the eval transform and parity would drift on
    real images even while passing on the traced example."""
    _, path = exported
    assert "Resize" in export.graph_operators(path)


def test_export_uses_the_contract_opset(exported):
    import onnx

    _, path = exported
    model = onnx.load(str(path))
    declared = meta.load_meta()["runtime"]["onnx_opset"]
    versions = {opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")}
    assert declared in versions


# --- runtime behaviour ------------------------------------------------------------


@pytest.mark.parametrize(("height", "width"), [(224, 224), (480, 640), (97, 143), (720, 480)])
def test_arbitrary_input_resolutions_are_accepted(exported, height, width):
    """Height and width are dynamic so clients hand over camera frames untouched."""
    import onnxruntime as ort

    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits, nutrition = session.run(None, {"image": export.example_input(height, width).numpy()})

    assert logits.shape == (1, 101)
    assert nutrition.shape == (1, 5, 3)


def test_output_shapes_match_the_contract(exported):
    import onnxruntime as ort

    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits, nutrition = session.run(None, {"image": export.example_input().numpy()})

    declared = meta.load_meta()["outputs"]
    assert list(logits.shape[1:]) == declared["logits"]["shape"][1:]
    assert list(nutrition.shape[1:]) == declared["nutrition_quantiles"]["shape"][1:]


def test_output_names_match_the_contract(exported):
    import onnxruntime as ort

    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]

    assert names == list(meta.load_meta()["outputs"].keys())


def test_input_name_and_dtype_match_the_contract(exported):
    import onnxruntime as ort

    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    spec = session.get_inputs()[0]

    assert spec.name == meta.load_meta()["input"]["name"]
    assert spec.type == "tensor(uint8)"


def test_float_input_is_rejected(exported):
    """Clients pass raw camera bytes. Accepting floats would let a client silently
    pre-normalise and have the graph normalise again."""
    import onnxruntime as ort

    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    floats = export.example_input().numpy().astype("float32")

    with pytest.raises(Exception, match="(?i)uint8|type|invalid"):
        session.run(None, {"image": floats})
