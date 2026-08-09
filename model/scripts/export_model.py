#!/usr/bin/env python
"""Turn trained checkpoints into the single artifact the app and the web demo load.

Assembles a combined two-head model, exports it to ONNX with preprocessing compiled into
the graph, quantises to int8 using real images for calibration, and checks that the
exported graph still agrees with PyTorch.

Writes a bundle manifest recording **which heads are actually trained**. The nutrition head
may be randomly initialised while the classifier is real, and a model that returns
confident numbers from an untrained head is exactly the failure this project keeps guarding
against. The app reads that flag and refuses to present untrained outputs as results.

Usage:
    python scripts/export_model.py \\
        --classifier runs/kaggle-baseline/runs/baseline/best.pt \\
        --calibration-root data/food101/food-101 \\
        --out runs/export
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from torch import nn

from platevision import bundle, checkpoint, datasets, export, meta, models, quantization

OUTPUT_NAMES = ["logits", "nutrition_quantiles"]

# Below this share of predictions surviving quantization, the int8 graph is a different
# model rather than a compressed one, and shipping it would trade accuracy nobody measured
# for megabytes nobody needed.
MIN_AGREEMENT = 0.90


def load_calibration_images(root: Path, count: int, seed: int = 0) -> list[np.ndarray]:
    """Real photographs, resized the way a phone would send them.

    Calibration decides the activation ranges every quantised layer uses. Random noise
    produces ranges nothing like real images, and the resulting scales are wrong in a way
    that only shows up as degraded accuracy on real input.
    """
    import random

    from PIL import Image

    samples = datasets.build_food101_index(root, "train")
    chosen = random.Random(seed).sample(samples, min(count, len(samples)))

    arrays: list[np.ndarray] = []
    for sample in chosen:
        with Image.open(sample.image_path) as image:
            rgb = image.convert("RGB")
            # The graph resizes internally, so the exact size does not matter for
            # correctness. Sending something phone-shaped keeps the calibration
            # distribution close to what the app will actually feed it.
            rgb.thumbnail((640, 640))
            arrays.append(np.asarray(rgb, dtype=np.uint8)[None, ...])
    return arrays


def build_combined(
    classifier_path: Path, nutrition_path: Path | None
) -> tuple[nn.Module, dict, bool]:
    """Assemble the two-head model. Returns (model, provenance, nutrition_trained)."""
    targets, quantiles = len(meta.target_keys()), len(meta.quantiles())

    classifier, payload = checkpoint.restore_classifier(classifier_path)

    if nutrition_path is not None:
        # Both heads sit on the nutrition run's backbone, which is the one that was
        # fine-tuned last. The classifier head must have been re-fitted against it by the
        # linear probe, or its logits describe features that no longer exist.
        nutrition_model, transform, _ = checkpoint.restore_nutrition_model(nutrition_path)
        backbone = nutrition_model.backbone
        nutrition_head = nutrition_model.head
        feature_dim = nutrition_model.feature_dim
        nutrition_trained = True
        source = str(nutrition_path)
    else:
        # No nutrition checkpoint: carry the classifier's backbone across and leave the
        # nutrition head at its random initialisation. The bundle records that so the
        # client refuses to present its output.
        holder = models.NutritionModel(
            checkpoint.backbone_of(payload),
            num_targets=targets,
            num_quantiles=quantiles,
            pretrained=False,
        )
        copied, _ = models.load_backbone_weights(holder, classifier.state_dict())
        if copied == 0:
            raise SystemExit("no backbone weights transferred; architectures disagree")

        backbone = holder.backbone
        nutrition_head = holder.head
        feature_dim = holder.feature_dim
        nutrition_trained = False
        source = "randomly initialised"
        transform = None

    classifier_head = _extract_head(classifier, feature_dim)

    model = models.CombinedModel(
        backbone,
        classifier_head,
        nutrition_head,
        num_targets=targets,
        num_quantiles=quantiles,
    ).eval()

    provenance = {
        "classifier_checkpoint": str(classifier_path),
        "classifier_backbone": checkpoint.backbone_of(payload),
        "classifier_metric": payload.get("best_metric"),
        "nutrition_source": source,
        "target_transform": transform.to_dict() if transform is not None else None,
    }
    return model, provenance, nutrition_trained


def _extract_head(classifier: nn.Module, feature_dim: int) -> nn.Linear:
    """Pull the final linear layer out of a timm classifier."""
    for module in reversed(list(classifier.modules())):
        if isinstance(module, nn.Linear) and module.in_features == feature_dim:
            return module
    raise SystemExit(
        f"could not find a classifier head with {feature_dim} input features; "
        "the backbone and head disagree"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--nutrition", type=Path, help="omit to export an untrained head")
    parser.add_argument(
        "--conformal",
        type=Path,
        help="conformal.json from the nutrition run; without it the intervals under-cover",
    )
    parser.add_argument("--calibration-root", type=Path, help="food-101 directory")
    parser.add_argument("--calibration-images", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("runs/export"))
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="ship the int8 artifact even when it changes most predictions",
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    model, provenance, nutrition_trained = build_combined(args.classifier, args.nutrition)
    print(
        f"classifier: {provenance['classifier_backbone']} "
        f"({provenance['classifier_metric']:.2f}% on the Food-101 test split)"
    )
    print(f"nutrition head: {provenance['nutrition_source']}")
    if not nutrition_trained:
        print("  the nutrition outputs of this artifact are meaningless and flagged as such")

    fp32 = args.out / "plate-vision-fp32.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export.export_onnx(model, fp32, output_names=OUTPUT_NAMES)
    print(f"\nfp32: {fp32.stat().st_size / 1024**2:.2f} MB")

    for result in export.check_parity(model, fp32, output_names=OUTPUT_NAMES):
        status = "ok" if result.within_tolerance else "FAILED"
        print(f"  parity {result.output_name:20} {result.max_absolute_difference:.2e} {status}")
        if not result.within_tolerance:
            raise SystemExit("export does not match PyTorch; refusing to ship it")

    artifact = fp32
    quant_report = None
    if not args.skip_quantization:
        if not args.calibration_root:
            raise SystemExit("--calibration-root is required unless --skip-quantization")
        print(f"\ncalibrating on {args.calibration_images} real photographs")
        calibration = load_calibration_images(args.calibration_root, args.calibration_images)

        int8 = args.out / "plate-vision-int8.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantization.quantize_static(
                fp32, int8, quantization.ArrayCalibrationReader(calibration)
            )

        report = quantization.build_report(fp32, int8, calibration[:16], runs=15)
        quant_report = report.as_dict()
        print(
            f"int8: {report.int8_bytes / 1024**2:.2f} MB "
            f"({report.size_ratio:.2f}x)  top-1 agreement {report.top1_agreement:.1%}"
        )
        print(
            f"  latency p50 {report.fp32_latency.p50_ms:.1f} -> "
            f"{report.int8_latency.p50_ms:.1f} ms on this machine"
        )

        # EfficientNet quantises badly. Depthwise separable convolutions with SiLU
        # activations have per-channel ranges int8 cannot represent, and more calibration
        # does not fix it: measured agreement was 25%, 15%, and 47% at 32, 128, and 384
        # calibration images, erratic rather than improving. A small artifact that predicts
        # something else is not a smaller model, so fp32 stays the shipped one by default.
        if report.top1_agreement < MIN_AGREEMENT:
            print(
                f"\n  int8 changes {1 - report.top1_agreement:.0%} of predictions. That is a "
                "different model, not a compressed one."
            )
            if args.allow_degraded:
                print("  shipping it anyway because --allow-degraded was passed")
                artifact = int8
            else:
                print("  keeping fp32 as the artifact. Pass --allow-degraded to override.")
        else:
            artifact = int8

    # Size and hash travel with the manifest so a client can tell a complete download from
    # a truncated one. A cut-off protobuf can still parse as a valid, shorter graph, and
    # without this the failure looks like a bad model rather than a bad download.
    digest = export.digest_artifact(artifact)

    conformal_payload = None
    if args.conformal:
        conformal_payload = json.loads(args.conformal.read_text(encoding="utf-8"))
        offsets = ", ".join(
            f"{k} +/-{v:.0f}"
            for k, v in zip(conformal_payload["keys"], conformal_payload["offsets"], strict=True)
        )
        print(f"conformal: {offsets}")
    elif nutrition_trained:
        # Loud, because the failure is silent otherwise: the client would present raw
        # quantiles as a 90% interval when they cover about 82%.
        print("
  WARNING: no --conformal offsets. The intervals will under-cover.")

    manifest = bundle.build_bundle(
        artifact=digest.as_dict(),
        conformal=conformal_payload,
        heads_trained={"logits": True, "nutrition_quantiles": nutrition_trained},
        provenance=provenance,
        quantization=quant_report,
        generated_utc=datetime.now(UTC).isoformat(),
    )
    (args.out / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n  wrote {artifact} ({digest.bytes / 1024**2:.2f} MB, sha256 {digest.sha256[:12]}…)")
    print(f"  wrote {args.out / 'bundle.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
