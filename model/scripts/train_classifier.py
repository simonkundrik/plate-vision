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
from platevision.ema import ModelEma
from platevision.mixing import MixingPolicy


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

    # Validation always runs at the contract resolution, even when training resolution
    # varies, or the reported metric would not describe the exported model.
    val_ds = datasets.Food101Dataset(val_samples, transform=transforms.eval_transform())
    common = {"num_workers": args.workers, "pin_memory": torch.cuda.is_available()}
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=args.workers > 0,
        **common,
    )

    def make_train_loader(size: int | None = None) -> DataLoader:
        dataset = datasets.Food101Dataset(
            train_samples, transform=transforms.classification_train_transform(size=size)
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            persistent_workers=args.workers > 0,
            **common,
        )

    return make_train_loader, val_loader, num_classes


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

    recipe = parser.add_argument_group("training recipe")
    recipe.add_argument("--mixup-alpha", type=float, default=0.0)
    recipe.add_argument("--cutmix-alpha", type=float, default=0.0)
    recipe.add_argument("--mix-prob", type=float, default=1.0)
    recipe.add_argument("--switch-prob", type=float, default=0.5)
    recipe.add_argument("--ema", action="store_true", help="track an EMA of the weights")
    recipe.add_argument("--ema-decay", type=float, default=0.9998)
    recipe.add_argument(
        "--progressive-resize",
        type=int,
        metavar="START",
        help="ramp training resolution from START up to the contract size",
    )
    args = parser.parse_args(argv)

    engine.seed_everything(args.seed)
    device = models.resolve_device(args.device)
    print(f"device: {device}")

    make_train_loader, val_loader, num_classes = build_loaders(args)
    resolutions = (
        transforms.resolution_schedule(args.epochs, start=args.progressive_resize)
        if args.progressive_resize
        else [0] * args.epochs
    )
    train_loader = make_train_loader(resolutions[0] or None)

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

    mixing_policy = MixingPolicy(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        prob=args.mix_prob,
        switch_prob=args.switch_prob,
        seed=args.seed,
    )
    if mixing_policy.enabled:
        print(f"mixing: mixup a={args.mixup_alpha} cutmix a={args.cutmix_alpha}")
    else:
        mixing_policy = None

    ema = ModelEma(model, decay=args.ema_decay) if args.ema else None
    if ema:
        print(f"ema:    decay {args.ema_decay} with warmup")
    if args.progressive_resize:
        print(f"resize: {resolutions[0]} -> {resolutions[-1]}")

    history = engine.History()
    best = -1.0
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    for epoch in range(args.epochs):
        # Rebuilding the loader is what actually changes the resolution, since the size
        # is baked into the transform. Only done when it changes, because respawning
        # workers is not free.
        if epoch > 0 and resolutions[epoch] != resolutions[epoch - 1]:
            train_loader = make_train_loader(resolutions[epoch] or None)

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
            mixing=mixing_policy,
            ema=ema,
            resolution=resolutions[epoch],
        )
        history.add(train_result)
        print(train_result.format(), flush=True)

        val_result = engine.evaluate(model, val_loader, criterion, device, epoch=epoch)
        history.add(val_result)
        print(val_result.format(), flush=True)

        # The EMA copy is evaluated separately rather than replacing the live number.
        # Early in training it is worse, and reporting only the better of the two would
        # hide that the average had not warmed up yet.
        selected, selected_name = val_result, "live"
        if ema is not None:
            ema_result = engine.evaluate(
                ema.module, val_loader, criterion, device, epoch=epoch, split="ema"
            )
            history.add(ema_result)
            print(ema_result.format(), flush=True)
            if ema_result.top1 > val_result.top1:
                selected, selected_name = ema_result, "ema"

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
        if selected.top1 > best:
            best = selected.top1
            # Save whichever set of weights produced the number, not always the live one.
            # Exporting live weights while reporting the EMA score is a mismatch that no
            # metric would catch, because both numbers are real.
            best_model = ema.module if selected_name == "ema" else model
            checkpoint.save_checkpoint(
                args.out / "best.pt",
                model=best_model,
                epoch=epoch,
                config={**config, "weights": selected_name},
                history=history.as_dicts(),
                best_metric=best,
            )
            print(f"  new best: {best:.2f}% top-1 ({selected_name} weights)")

        (args.out / "history.json").write_text(
            json.dumps(history.as_dicts(), indent=2), encoding="utf-8"
        )

    live_best = history.best("val")
    ema_best = history.best("ema")
    if live_best:
        print(f"\nbest live top-1: {live_best.top1:.2f}% at epoch {live_best.epoch}")
    if ema_best:
        print(f"best ema  top-1: {ema_best.top1:.2f}% at epoch {ema_best.epoch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
