#!/usr/bin/env python
"""Train the Food-101 classification baseline.

This establishes the number every later stage is measured against. It is deliberately
plain: cross-entropy, AdamW, cosine schedule with warmup. The training recipe (mixup,
cutmix, EMA, progressive resizing) lands in the next PR so its contribution can be
measured against this baseline rather than assumed.

Local smoke run, a few classes and a small backbone, proves the loop learns:

    python scripts/train_classifier.py --data-root data/food101/food-101 \\
        --backbone mobilenetv3_small_100 --subset-classes 5 --limit-train 400 \\
        --limit-val 100 --epochs 2 --batch-size 16

Full run (Kaggle GPU):

    python scripts/train_classifier.py --data-root /kaggle/input/food101/food-101 \\
        --epochs 30 --batch-size 128 --amp
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from platevision import checkpoint, datasets, engine, models, transforms


def build_loaders(args) -> tuple[DataLoader, DataLoader, int]:
    root = Path(args.data_root)
    train_samples = datasets.build_food101_index(root, "train")
    val_samples = datasets.build_food101_index(root, "test")

    num_classes = args.subset_classes or len(set(s.label for s in train_samples))
    if args.subset_classes:
        train_samples = [s for s in train_samples if s.label < args.subset_classes]
        val_samples = [s for s in val_samples if s.label < args.subset_classes]

    rng = random.Random(args.seed)
    if args.limit_train and len(train_samples) > args.limit_train:
        train_samples = rng.sample(train_samples, args.limit_train)
    if args.limit_val and len(val_samples) > args.limit_val:
        val_samples = rng.sample(val_samples, args.limit_val)

    print(f"train samples: {len(train_samples):,}")
    print(f"val samples:   {len(val_samples):,}")
    print(f"classes:       {num_classes}")

    train_ds = datasets.Food101Dataset(
        train_samples, transform=transforms.classification_train_transform()
    )
    val_ds = datasets.Food101Dataset(val_samples, transform=transforms.eval_transform())

    common = {"num_workers": args.workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=args.workers > 0,
        **common,
    )
    return train_loader, val_loader, num_classes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True, type=Path, help="the food-101 directory")
    parser.add_argument("--out", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--backbone", default=models.STUDENT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help="mixed precision, CUDA only")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--subset-classes", type=int, help="keep only the first N classes")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)

    engine.seed_everything(args.seed)
    device = models.resolve_device(args.device)
    print(f"device: {device}")

    train_loader, val_loader, num_classes = build_loaders(args)

    model = models.create_classifier(
        args.backbone,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)
    print(f"model:  {args.backbone}, {models.count_parameters(model):,} trainable parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(models.parameter_groups(model, args.weight_decay), lr=args.lr)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = engine.build_scheduler(
        optimizer,
        warmup_steps=int(steps_per_epoch * args.warmup_epochs),
        total_steps=total_steps,
    )

    amp_enabled = args.amp and device.type == "cuda"
    if args.amp and not amp_enabled:
        print("note: --amp ignored, mixed precision requires CUDA")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    history = engine.History()
    best = -1.0
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    for epoch in range(args.epochs):
        train_result = engine.train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            scheduler=scheduler,
            scaler=scaler if amp_enabled else None,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
        )
        history.add(train_result)
        print(train_result.format(), flush=True)

        val_result = engine.evaluate(model, val_loader, criterion, device, epoch=epoch)
        history.add(val_result)
        print(val_result.format(), flush=True)

        checkpoint.save_checkpoint(
            args.out / "last.pt",
            model=model,
            epoch=epoch,
            config=config,
            history=history.as_dicts(),
            best_metric=best,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if val_result.top1 > best:
            best = val_result.top1
            checkpoint.save_checkpoint(
                args.out / "best.pt",
                model=model,
                epoch=epoch,
                config=config,
                history=history.as_dicts(),
                best_metric=best,
            )
            print(f"  new best: {best:.2f}% top-1")

        (args.out / "history.json").write_text(
            json.dumps(history.as_dicts(), indent=2), encoding="utf-8"
        )

    top = history.best("val")
    print(f"\nbest val top-1: {top.top1:.2f}% at epoch {top.epoch}" if top else "\nno validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
