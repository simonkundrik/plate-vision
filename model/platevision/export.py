"""ONNX export with preprocessing compiled into the graph.

The exported model takes raw uint8 RGB pixels at any resolution and does the resize,
normalisation, and layout change itself. Clients perform none of it.

The alternative is each of Python, React Native, and the browser reimplementing the same
ImageNet constants and the same interpolation. When one of them drifts, nothing crashes:
the app keeps working and quietly returns wrong calories. Putting it in the graph deletes
that entire class of bug, and as a bonus it removes the need to extract raw pixel buffers
from a JPEG in React Native, which is the most annoying part of that stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from platevision import meta


class ExportWrapper(nn.Module):
    """Wraps a model so the exported graph owns preprocessing.

    Input is (batch, height, width, 3) uint8 in RGB order, with height and width dynamic.
    Steps follow ``preprocessing.order`` from the contract exactly: convert to float in the
    unit range, resize, normalise. Resizing uint8 first and converting afterwards rounds at
    a different point and produces measurably different pixels.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        order = meta.preprocessing_order()
        if order != meta.SUPPORTED_PREPROCESSING_ORDER:
            raise ValueError(f"unsupported preprocessing order {order}")

        resize = meta.load_meta()["preprocessing"]["resize"]
        mean, std = meta.normalization()

        self.model = model
        self.size = (resize["height"], resize["width"])
        self.mode = resize["mode"]
        self.antialias = bool(resize["antialias"])
        # Buffers rather than plain tensors so .to(device) moves them and they are
        # captured as graph initialisers on export.
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor):
        # NHWC uint8 to NCHW float in 0..1.
        x = image.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(
            x,
            size=self.size,
            mode=self.mode,
            align_corners=False,
            antialias=self.antialias,
        )
        x = (x - self.mean) / self.std
        return self.model(x)


@dataclass(frozen=True, slots=True)
class ParityResult:
    output_name: str
    max_absolute_difference: float
    within_tolerance: bool


def example_input(height: int = 480, width: int = 640, batch: int = 1) -> torch.Tensor:
    """A deliberately non-square, non-target-sized input.

    Exporting at 224x224 would trace a graph where the resize is a no-op, and the export
    would appear to work while doing nothing. The example has to exercise the resize.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (batch, height, width, 3), dtype=torch.uint8, generator=generator)


def export_onnx(
    model: nn.Module,
    path: Path,
    *,
    output_names: list[str],
    opset: int | None = None,
    example: torch.Tensor | None = None,
) -> Path:
    """Export a wrapped model to ONNX with dynamic height and width."""
    opset = opset or meta.load_meta()["runtime"]["onnx_opset"]
    example = example if example is not None else example_input()

    wrapper = ExportWrapper(model).eval()
    path.parent.mkdir(parents=True, exist_ok=True)

    # dynamo=True, not the TorchScript exporter. Antialiased bilinear resize
    # (aten::_upsample_bilinear2d_aa) has no TorchScript ONNX translation at any opset
    # this project can use, and dropping antialias would visibly degrade downscaling of
    # phone photos, which arrive far larger than 224px. The dynamo exporter handles it.
    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        input_names=["image"],
        output_names=output_names,
        opset_version=opset,
        dynamic_shapes={"image": {0: "batch", 1: "height", 2: "width"}},
        dynamo=True,
        verbose=False,
    )
    consolidate(path)
    return path


def consolidate(path: Path) -> Path:
    """Fold externally-stored weights back into a single ``.onnx`` file.

    The dynamo exporter writes initialisers to a sidecar ``.onnx.data`` by default. That
    is fine on a workstation and wrong for this project: the Android app bundles the model
    as one asset and the browser demo fetches one URL. Two files means both have to travel
    together, and shipping only the ``.onnx`` produces a load-time failure that says
    nothing about a missing sidecar.

    Safe here because the model is tens of megabytes, far below protobuf's 2 GB ceiling.
    """
    import onnx

    sidecar = path.with_suffix(path.suffix + ".data")
    if not sidecar.exists():
        return path

    model = onnx.load(str(path))  # pulls the external tensors into memory
    onnx.save(model, str(path), save_as_external_data=False)
    sidecar.unlink()
    return path


def graph_operators(path: Path) -> dict[str, int]:
    """Count operator types in an exported graph.

    Worth looking at before shipping: an op the mobile runtime lacks a kernel for falls
    back to a slow path or fails to load, and that is cheaper to find here than on a phone.
    """
    import onnx

    model = onnx.load(str(path))
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def check_parity(
    model: nn.Module,
    path: Path,
    *,
    output_names: list[str],
    example: torch.Tensor | None = None,
    tolerance: float = 1e-3,
) -> list[ParityResult]:
    """Compare the exported graph against the PyTorch model on identical input.

    This is the test that makes the whole in-graph preprocessing argument checkable. If
    the exported graph disagrees with the model that was evaluated, every reported metric
    describes something that was never shipped.
    """
    import onnxruntime as ort

    example = example if example is not None else example_input()
    wrapper = ExportWrapper(model).eval()

    with torch.no_grad():
        reference = wrapper(example)
    if isinstance(reference, torch.Tensor):
        reference = (reference,)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": example.numpy()})

    results = []
    for name, expected_tensor, actual_array in zip(output_names, reference, actual, strict=True):
        difference = (expected_tensor.numpy() - actual_array).__abs__().max().item()
        results.append(
            ParityResult(
                output_name=name,
                max_absolute_difference=difference,
                within_tolerance=difference <= tolerance,
            )
        )
    return results
