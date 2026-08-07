"""Tests for int8 quantization.

The measurement discipline is the point here. "int8" on its own says nothing: a model that
is three times smaller and four points less accurate is a worse model, and one that is
smaller but slower on the target device is a different tradeoff again.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch import nn

from platevision import export
from platevision import quantization as q

OUTPUT_NAMES = ["logits", "nutrition_quantiles"]


class TinyCombined(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Linear(16, 101)
        self.nutrition_head = nn.Linear(16, 15)

    def forward(self, x):
        features = self.pool(self.mid(self.stem(x))).flatten(1)
        return self.classifier_head(features), self.nutrition_head(features).view(-1, 5, 3)


def images(count=8, height=240, width=320, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (1, height, width, 3), dtype=np.uint8) for _ in range(count)]


@pytest.fixture
def fp32_model(tmp_path):
    torch.manual_seed(0)
    model = TinyCombined().eval()
    path = tmp_path / "fp32.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export.export_onnx(model, path, output_names=OUTPUT_NAMES)
    return path


@pytest.fixture
def int8_model(tmp_path, fp32_model):
    path = tmp_path / "int8.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q.quantize_static(fp32_model, path, q.ArrayCalibrationReader(images(8)))
    return path


# --- calibration ------------------------------------------------------------------


def test_calibration_reader_yields_every_sample_then_stops():
    reader = q.ArrayCalibrationReader(images(3))
    assert sum(1 for _ in iter(reader.get_next, None)) == 3


def test_calibration_reader_rewinds():
    """The quantizer makes more than one pass over the calibration set."""
    reader = q.ArrayCalibrationReader(images(2))
    assert sum(1 for _ in iter(reader.get_next, None)) == 2
    reader.rewind()
    assert sum(1 for _ in iter(reader.get_next, None)) == 2


def test_calibration_reader_uses_the_graph_input_name():
    reader = q.ArrayCalibrationReader(images(1), input_name="image")
    assert "image" in reader.get_next()


def test_empty_calibration_is_rejected():
    """Calibration with no data produces arbitrary activation ranges."""
    with pytest.raises(ValueError, match="at least one sample"):
        q.ArrayCalibrationReader([])


# --- what gets quantized ----------------------------------------------------------


def test_only_weight_heavy_ops_are_quantized():
    """The graph carries the preprocessing chain too. Quantizing Resize, Sub, or Div would
    inject rounding into normalisation for no size benefit, since they hold no weights."""
    assert set(q.QUANTIZABLE_OPS) == {"Conv", "MatMul", "Gemm"}
    assert "Resize" not in q.QUANTIZABLE_OPS


def test_preprocessing_survives_quantization(fp32_model, int8_model):
    before = export.graph_operators(fp32_model)
    after = export.graph_operators(int8_model)
    assert after.get("Resize") == before.get("Resize")


def test_quantization_inserts_qdq_nodes(int8_model):
    ops = export.graph_operators(int8_model)
    assert ops.get("QuantizeLinear", 0) > 0
    assert ops.get("DequantizeLinear", 0) > 0


# --- the tradeoff -----------------------------------------------------------------


def test_int8_is_smaller(fp32_model, int8_model):
    assert int8_model.stat().st_size < fp32_model.stat().st_size


def test_top1_agreement_is_preserved(fp32_model, int8_model):
    """What the user sees is the argmax, not the logit. Quantization shifts every logit
    slightly; the question is whether the prediction moved."""
    _, agreement = q.compare_logits(fp32_model, int8_model, images(8, seed=5))
    assert agreement >= 0.9


def test_report_computes_the_ratios(fp32_model, int8_model):
    report = q.build_report(fp32_model, int8_model, images(4, seed=7), runs=3)

    assert report.size_ratio == pytest.approx(report.int8_bytes / report.fp32_bytes)
    assert report.speedup == pytest.approx(report.fp32_latency.p50_ms / report.int8_latency.p50_ms)
    assert 0.0 <= report.top1_agreement <= 1.0


def test_report_serialises_with_derived_fields(fp32_model, int8_model):
    payload = q.build_report(fp32_model, int8_model, images(2), runs=3).as_dict()
    assert "size_ratio" in payload
    assert "speedup" in payload


# --- latency measurement ----------------------------------------------------------


def test_latency_percentiles_are_ordered(fp32_model):
    stats = q.benchmark(fp32_model, images(1)[0], runs=20, warmup=2)
    assert stats.p50_ms <= stats.p95_ms
    assert stats.runs == 20


def test_warmup_runs_are_excluded(fp32_model):
    """The first inference pays for arena allocation and kernel selection and represents
    nothing. Including it would inflate the mean on short benchmarks."""
    stats = q.benchmark(fp32_model, images(1)[0], runs=5, warmup=3)
    assert stats.runs == 5
    assert stats.mean_ms > 0
