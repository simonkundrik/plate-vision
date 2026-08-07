#!/usr/bin/env python
"""Re-fit the Food-101 classification head on the final nutrition backbone.

Nutrition training fine-tunes the backbone, which invalidates the classification head from
stage one: that head was fitted against different features. Assembling both into one model
without this step produces logits that are confidently wrong while the nutrition outputs
are fine, and nothing about the export would reveal it.

The backbone is frozen, so this is a linear probe rather than a fine-tune. Features are
extracted once and cached, which is valid here precisely because the backbone is frozen and
no augmentation is applied. That is the same caching that does *not* work for distillation,
where the backbone is training and every epoch draws a different crop.

Usage:
    python scripts/fit_classifier_head.py \\
        --nutrition-checkpoint runs/nutrition/best.pt \\
        --data-root data/food101/food-101 --out runs/combined
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from platevision import checkpoint, datasets, engine, food101, models, transforms


@torch.no_grad()
def extract_features(backbone: nn.Module, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    features, labels = [], []
    for index, (images, batch_labels) in enumerate(loader):
        features.append(backbone(images.to(device, non_blocking=True)).float().cpu())
        labels.append(batch_labels)
        if index % 50 == 0:
            print(f"  batch {index}", flush=True)
    return torch.cat(features), torch.cat(labels)


def fit_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> tuple[nn.Linear, float]:
    head = nn.Linear(train_features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True
    )
    steps = max(1, len(loader))
    scheduler = engine.build_scheduler(optimizer, warmup_steps=steps, total_steps=steps * epochs)

    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    best = 0.0

    for epoch in range(epochs):
        head.train()
        total = 0.0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * labels.size(0)

        head.eval()
        with torch.no_grad():
            top1, top5 = engine.accuracy(head(val_features), val_labels, topk=(1, 5))
        best = max(best, top1)
        print(
            f"epoch {epoch:>3}  loss {total / len(train_labels):.4f}  "
            f"top1 {top1:6.2f}%  top5 {top5:6.2f}%",
            flush=True,
        )

    return head, best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nutrition-checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path, help="the food-101 directory")
    parser.add_argument("--out", type=Path, default=Path("runs/combined"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    args = parser.parse_args(argv)

    engine.seed_everything(args.seed)
    device = models.resolve_device(args.device)
    print(f"device: {device}")

    nutrition_model, target_transform, payload = checkpoint.restore_nutrition_model(
        args.nutrition_checkpoint
    )
    nutrition_model.to(device)
    print(f"backbone: {payload['backbone']} from {args.nutrition_checkpoint}")

    train_samples = datasets.build_food101_index(args.data_root, "train")
    val_samples = datasets.build_food101_index(args.data_root, "test")

    # Sampled, not sliced. Food-101's split files are grouped by class, so taking a prefix
    # yields the first two or three classes and an accuracy figure that means nothing. The
    # giveaway is top-5 pinned at 100 percent.
    rng = random.Random(args.seed)
    if args.limit_train and len(train_samples) > args.limit_train:
        train_samples = rng.sample(train_samples, args.limit_train)
    if args.limit_val and len(val_samples) > args.limit_val:
        val_samples = rng.sample(val_samples, args.limit_val)

    print(f"food-101: {len(train_samples):,} train, {len(val_samples):,} val")
    print(f"          {len({s.label for s in val_samples})} classes present in val")

    # The eval transform, not the training one. A linear probe on a frozen backbone gains
    # nothing from augmentation and it would make the features non-cacheable.
    def loader_for(samples):
        return DataLoader(
            datasets.Food101Dataset(samples, transform=transforms.eval_transform()),
            batch_size=args.extract_batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

    print("extracting train features")
    train_features, train_labels = extract_features(
        nutrition_model.backbone, loader_for(train_samples), device
    )
    print("extracting val features")
    val_features, val_labels = extract_features(
        nutrition_model.backbone, loader_for(val_samples), device
    )
    print(f"features: {tuple(train_features.shape)} train, {tuple(val_features.shape)} val")

    num_classes = len(food101.class_keys())
    head, best = fit_head(
        train_features,
        train_labels,
        val_features,
        val_labels,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"\nlinear probe top-1: {best:.2f}%")

    combined = models.CombinedModel(
        nutrition_model.backbone,
        head.cpu(),
        nutrition_model.head,
        num_targets=nutrition_model.num_targets,
        num_quantiles=nutrition_model.num_quantiles,
    ).cpu()

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint.save_checkpoint(
        args.out / "combined.pt",
        model=combined,
        epoch=0,
        backbone=payload["backbone"],
        config={
            "source_nutrition_checkpoint": str(args.nutrition_checkpoint),
            "target_transform": target_transform.to_dict(),
            "linear_probe_top1": best,
            "num_targets": nutrition_model.num_targets,
            "num_quantiles": nutrition_model.num_quantiles,
        },
        best_metric=best,
    )
    (args.out / "probe.json").write_text(
        json.dumps({"top1": best, "features": list(train_features.shape)}, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {args.out / 'combined.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
