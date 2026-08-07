"""Post-training int8 quantization and the measurements that justify it.

Quantization is only worth doing if the tradeoff is measured. A smaller, faster model that
lost four points of accuracy is a worse model, and "int8" on its own says nothing about
which happened.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Only the weight-heavy ops. The exported graph also contains the preprocessing chain
# (Resize, Sub, Div) and quantizing that would inject rounding error into normalisation
# for no size benefit, since those nodes carry almost no parameters.
QUANTIZABLE_OPS = ["Conv", "MatMul", "Gemm"]


@dataclass(frozen=True, slots=True)
class LatencyStats:
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuantizationReport:
    fp32_bytes: int
    int8_bytes: int
    fp32_latency: LatencyStats
    int8_latency: LatencyStats
    max_absolute_difference: float
    top1_agreement: float

    @property
    def size_ratio(self) -> float:
        return self.int8_bytes / self.fp32_bytes

    @property
    def speedup(self) -> float:
        return self.fp32_latency.p50_ms / self.int8_latency.p50_ms

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "size_ratio": self.size_ratio,
            "speedup": self.speedup,
        }


class ArrayCalibrationReader:
    """Feeds calibration batches to ONNX Runtime's static quantizer.

    Calibration samples must come from the *training* distribution and must be real
    images. Random noise produces activation ranges nothing like the real ones, and the
    resulting quantization scales are wrong in a way that shows up only as degraded
    accuracy on real inputs.
    """

    def __init__(self, arrays: list[np.ndarray], input_name: str = "image") -> None:
        if not arrays:
            raise ValueError("calibration needs at least one sample")
        self.input_name = input_name
        self._arrays = arrays
        self._iterator: Iterator[np.ndarray] | None = None

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iterator is None:
            self._iterator = iter(self._arrays)
        batch = next(self._iterator, None)
        return None if batch is None else {self.input_name: batch}

    def rewind(self) -> None:
        self._iterator = None


def quantize_static(
    source: Path,
    destination: Path,
    reader: ArrayCalibrationReader,
    *,
    per_channel: bool = True,
) -> Path:
    """Static int8 quantization in QDQ format.

    ``per_channel`` defaults on and should stay on. EfficientNet is built from
    depthwise-separable convolutions, whose per-channel weight ranges differ by orders of
    magnitude. A single per-tensor scale is set by the widest channel and collapses the
    rest to near zero, which costs far more accuracy than quantization itself does.

    Static rather than dynamic because this is a convolutional network. Dynamic
    quantization computes activation ranges at runtime and mainly benefits matmul-heavy
    models; on a CNN it leaves the convolutions in float and delivers little.
    """
    from onnxruntime.quantization import QuantFormat, QuantType
    from onnxruntime.quantization import quantize_static as ort_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared = destination.with_suffix(".prepared.onnx")
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)

    reader.rewind()
    ort_static(
        str(prepared),
        str(destination),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        op_types_to_quantize=QUANTIZABLE_OPS,
    )
    prepared.unlink(missing_ok=True)
    return destination


def quantize_dynamic(source: Path, destination: Path) -> Path:
    """Dynamic int8 quantization. Included for comparison, not expected to win here."""
    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization import quantize_dynamic as ort_dynamic

    destination.parent.mkdir(parents=True, exist_ok=True)
    ort_dynamic(
        str(source),
        str(destination),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=QUANTIZABLE_OPS,
    )
    return destination


def benchmark(
    path: Path, example: np.ndarray, *, runs: int = 30, warmup: int = 5, threads: int | None = None
) -> LatencyStats:
    """Wall-clock latency for a single graph.

    p95 is reported alongside p50 because a phone's worst case is what a user notices, and
    a mean hides it. Warmup runs are discarded: the first inference pays for memory arena
    allocation and kernel selection and is not representative of anything.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: example}

    for _ in range(warmup):
        session.run(None, feed)

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, feed)
        timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    return LatencyStats(
        runs=runs,
        mean_ms=statistics.fmean(timings),
        p50_ms=timings[len(timings) // 2],
        p95_ms=timings[min(len(timings) - 1, int(len(timings) * 0.95))],
    )


def compare_logits(
    fp32_path: Path, int8_path: Path, examples: list[np.ndarray]
) -> tuple[float, float]:
    """Return (max absolute logit difference, top-1 agreement rate).

    Agreement matters more than the raw difference. Quantization shifts every logit
    slightly; what matters is whether the argmax moved, because that is what the user sees.
    """
    import onnxruntime as ort

    fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    name = fp32.get_inputs()[0].name

    worst = 0.0
    matches = 0
    total = 0
    for example in examples:
        reference = fp32.run(None, {name: example})[0]
        actual = int8.run(None, {name: example})[0]
        worst = max(worst, float(np.abs(reference - actual).max()))
        matches += int((reference.argmax(axis=1) == actual.argmax(axis=1)).sum())
        total += reference.shape[0]

    return worst, (matches / total if total else 0.0)


def build_report(
    fp32_path: Path,
    int8_path: Path,
    examples: list[np.ndarray],
    *,
    runs: int = 30,
) -> QuantizationReport:
    worst, agreement = compare_logits(fp32_path, int8_path, examples)
    return QuantizationReport(
        fp32_bytes=fp32_path.stat().st_size,
        int8_bytes=int8_path.stat().st_size,
        fp32_latency=benchmark(fp32_path, examples[0], runs=runs),
        int8_latency=benchmark(int8_path, examples[0], runs=runs),
        max_absolute_difference=worst,
        top1_agreement=agreement,
    )
