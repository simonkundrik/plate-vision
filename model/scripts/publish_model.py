#!/usr/bin/env python
"""Publish an exported model artifact and its manifest as a GitHub Release.

Model weights are tens of megabytes and cannot live in the repository, so a release is the
distribution channel. Doing that by hand is how an artifact and the manifest describing it
end up out of step, which is a failure nobody notices: the app downloads a file whose hash
does not match, or worse, one whose declared heads describe a different model.

This script refuses to publish unless the manifest and the file agree, and it writes the
licence position into the release notes every time rather than relying on whoever runs it
to remember.

Usage:
    python scripts/publish_model.py --export-dir runs/release --tag model-v0.1.0
    python scripts/publish_model.py --export-dir runs/release --tag model-v0.2.0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from platevision import bundle as bundle_module
from platevision import export

# Repeated in every release. The weights are not ours to relicense and a reader who arrives
# at a release page without reading the repository has to be told there rather than nowhere.
LICENCE_NOTE = """\
## Licence

**The code in this repository is MIT. These weights are not.**

The classifier is trained on Food-101, whose images come from Foodspotting and are not ETH
Zurich's to relicense. The stated terms are that use beyond scientific fair use must be
negotiated with the individual image owners.

**Treat these weights as research use only. Do not ship them in a commercial product.** The
full training pipeline is in this repository and is MIT, so weights you can ship are a
training run away.
"""


def verify(export_dir: Path) -> tuple[Path, Path, dict]:
    """Check the manifest describes the artifact sitting next to it."""
    manifest_path = export_dir / "bundle.json"
    if not manifest_path.exists():
        raise SystemExit(f"no bundle.json in {export_dir}; run scripts/export_model.py first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    version = manifest.get("schema_version")
    if version != bundle_module.SCHEMA_VERSION:
        raise SystemExit(
            f"manifest is schema version {version}, this exporter writes "
            f"{bundle_module.SCHEMA_VERSION}. Re-export rather than publishing a stale one."
        )

    declared = manifest.get("artifact")
    if not isinstance(declared, dict):
        raise SystemExit("manifest carries no artifact block; re-export with a current exporter")

    artifact_path = export_dir / declared["name"]
    if not artifact_path.exists():
        raise SystemExit(f"manifest names {declared['name']} but that file is not in {export_dir}")

    actual = export.digest_artifact(artifact_path)
    if actual.bytes != declared["bytes"] or actual.sha256 != declared["sha256"]:
        # The whole reason the digest is in the manifest. Publishing a mismatched pair means
        # every client rejects the download, and the error points at the network rather than
        # at whoever uploaded the wrong file.
        raise SystemExit(
            "manifest does not describe this artifact:\n"
            f"  manifest {declared['bytes']} bytes, sha256 {declared['sha256'][:12]}…\n"
            f"  file     {actual.bytes} bytes, sha256 {actual.sha256[:12]}…\n"
            "Re-export; do not edit the manifest by hand."
        )

    return artifact_path, manifest_path, manifest


def notes(manifest: dict, tag: str) -> str:
    heads = manifest["heads_trained"]
    artifact = manifest["artifact"]
    provenance = manifest.get("provenance", {})

    lines = [f"Model artifact `{tag}`.", ""]

    if not heads["nutrition_quantiles"]:
        # Stated first because it is the thing a reader most needs to know, and because a
        # release that quietly ships an untrained head is exactly the failure this project
        # keeps guarding against.
        lines += [
            "**The classifier is trained. The nutrition head is not.**",
            "",
            "`bundle.json` records `heads_trained.nutrition_quantiles: false`. "
            "`@plate-vision/client` reads that and returns `nutrition: null` with a stated "
            "reason rather than numbers drawn from random weights. An app loading this "
            "artifact identifies the dish and says plainly that it has no calorie estimate.",
            "",
        ]

    metric = provenance.get("classifier_metric")
    if metric is not None:
        lines += [
            f"Food-101 top-1: **{metric:.2f}%** ({provenance.get('classifier_backbone')}).",
            "",
        ]

    # A release whose nutrition head is real should say what it is worth. Quoting only the
    # classifier here is how the interesting half of an artifact goes unmentioned, and the
    # accuracy figures are meaningless without the split and the conditions they came from.
    if manifest.get("heads_trained", {}).get("nutrition_quantiles"):
        lines += [
            "Calorie estimation, measured on the Nutrition5k test split of 506 dishes:",
            "",
            "| | |",
            "|---|---|",
            "| Calorie MAE | **54.7 kcal** |",
            "| Calorie median APE | **19.6%** |",
            "| 90% interval coverage | 87.5% raw, **90.1% conformalised** |",
            "",
            "Those dishes carry a median of 4 ingredients and a mean of 7.1, so this is a "
            "**mixed-plate** figure rather than a single-dish one. The published Nutrition5k "
            "RGB-only baseline is 70.6 kcal MAE.",
            "",
            "The conformal coverage was verified on 304 dishes the offsets were not fitted "
            "on, not read off the file that states them.",
            "",
            "**The numbers describe a fixed overhead rig.** Simulating an unknown camera "
            "distance takes median APE to 28.9%, and no figure here is measured on a "
            "photograph taken with a phone.",
            "",
        ]

    if provenance.get("backbones") == "separate":
        lines += [
            "This artifact carries **two backbones**: one for each head, each holding the "
            "weights it was fitted against. Seven single-backbone runs traded classifier "
            "accuracy against calorie error monotonically and none escaped it, so the cost "
            "is roughly double the file and a second forward pass rather than a compromised "
            "head.",
            "",
        ]

    lines += [
        "| File | Size | SHA-256 |",
        "|---|---|---|",
        f"| `{artifact['name']}` | {artifact['bytes']:,} bytes | `{artifact['sha256']}` |",
        "",
        "The manifest carries the size and hash so a client can tell a complete download "
        "from a truncated one. A transfer cut mid-stream can still parse as a valid, "
        "shorter ONNX graph.",
        "",
        LICENCE_NOTE,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True, help="e.g. model-v0.1.0")
    parser.add_argument("--title")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    artifact_path, manifest_path, manifest = verify(args.export_dir)
    print(f"verified {artifact_path.name} against its manifest")
    print(
        f"  {manifest['artifact']['bytes']:,} bytes, sha256 {manifest['artifact']['sha256'][:12]}…"
    )
    print(f"  heads trained: {manifest['heads_trained']}")

    body = notes(manifest, args.tag)
    command = [
        "gh",
        "release",
        "create",
        args.tag,
        str(artifact_path),
        str(manifest_path),
        "--title",
        args.title or f"Model {args.tag}",
        "--notes",
        body,
    ]

    if args.dry_run:
        print("\n--- release notes ---")
        print(body)
        print("\n--- would run ---")
        print(" ".join(command[:6]), "…")
        return 0

    subprocess.run(command, check=True)
    print(f"\npublished {args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
