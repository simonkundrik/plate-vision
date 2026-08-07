#!/usr/bin/env python
"""Train the nutrition regression head on Nutrition5k.

Stage two of the two-stage plan. The backbone arrives already knowing what food looks like
from Food-101's 101k images, and this fits a quantile head on 2,755 dishes. Training the
backbone from scratch on that little data is not viable, which is the whole reason stage
one exists.

Predictions are intervals, not point estimates, because portion size is genuinely
unknowable from a single photo. Interval coverage is reported alongside MAE so the
uncertainty claim is checkable rather than decorative.

Usage:
    python scripts/train_nutrition.py --data-root data/nutrition5k \\
        --backbone-from runs/distilled/best.pt --epochs 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from platevision import checkpoint, datasets, engine, meta, models, regression, transforms
from platevision.quantile import PinballLoss
from platevision.targets import TargetTransform


def build(args, device):
    root = Path(args.data_root)
    train_samples, train_stats = datasets.build_nutrition5k_index(root, "train")
    val_samples, val_stats = datasets.build_nutrition5k_index(root, "test")

    # Sampled rather than sliced, for the same reason as the classifier scripts: a prefix
    # of an ordered split is not a representative subset. Nutrition5k is ordered by dish id
    # rather than by class, so the bias is milder here, but a debugging flag that quietly
    # changes the distribution is not worth keeping around.
    rng = random.Random(args.seed)
    if args.limit_train and len(train_samples) > args.limit_train:
        train_samples = rng.sample(train_samples, args.limit_train)
    if args.limit_val and len(val_samples) > args.limit_val:
        val_samples = rng.sample(val_samples, args.limit_val)

    print(f"train: {train_stats.kept:,} kept of {train_stats.listed:,} listed")
    print(
        f"       dropped {train_stats.missing_image:,} without imagery, "
        f"{train_stats.nonpositive_calories:,} with non-positive calories"
    )
    print(f"val:   {val_stats.kept:,} kept of {val_stats.listed:,} listed")
    if args.limit_train or args.limit_val:
        print(f"using: {len(train_samples):,} train, {len(val_samples):,} val")

    # Fitted on the training split only. Fitting across train and test leaks the test
    # distribution into training in a way no loss curve reveals.
    target_transform = TargetTransform.fit(s.targets for s in train_samples)
    print(f"target log-space mean: {[round(v, 2) for v in target_transform.mean]}")

    train_ds = datasets.Nutrition5kDataset(
        train_samples,
        transform=transforms.nutrition_train_transform(),
        target_transform=target_transform,
    )
    val_ds = datasets.Nutrition5kDataset(
        val_samples,
        transform=transforms.eval_transform(),
        target_transform=target_transform,
    )

    common = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, val_loader, target_transform


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/nutrition"))
    parser.add_argument("--backbone", default=models.STUDENT_BACKBONE)
    parser.add_argument(
        "--backbone-from",
        type=Path,
        help="classifier checkpoint whose backbone initialises this model",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--drop-rate", type=float, default=0.3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args(argv)

    engine.seed_everything(args.seed)
    device = models.resolve_device(args.device)
    print(f"device: {device}")

    target_keys = meta.target_keys()
    quantiles = meta.quantiles()
    median_index = quantiles.index(0.5)

    train_loader, val_loader, target_transform = build(args, device)

    model = models.NutritionModel(
        args.backbone,
        num_targets=len(target_keys),
        num_quantiles=len(quantiles),
        pretrained=args.backbone_from is None,
        drop_rate=args.drop_rate,
    )
    if args.backbone_from:
        payload = checkpoint.load_checkpoint(args.backbone_from)
        copied, skipped = models.load_backbone_weights(model, payload["model"])
        print(f"backbone: {payload.get('backbone')}, copied {copied} tensors, skipped {skipped}")
        if copied == 0:
            raise SystemExit(
                "No backbone weights transferred. The checkpoint's architecture does not "
                f"match --backbone {args.backbone}, so this would silently train from scratch."
            )
    model.to(device)
    print(f"model:  {models.count_parameters(model):,} trainable parameters")

    criterion = PinballLoss(quantiles).to(device)
    optimizer = torch.optim.AdamW(models.parameter_groups(model, args.weight_decay), lr=args.lr)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = engine.build_scheduler(
        optimizer,
        warmup_steps=int(steps_per_epoch * args.warmup_epochs),
        total_steps=steps_per_epoch * args.epochs,
    )

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "target_transform.json").write_text(
        json.dumps(target_transform.to_dict(), indent=2), encoding="utf-8"
    )

    history: list[dict] = []
    best = float("inf")

    for epoch in range(args.epochs):
        train_result = regression.train_regression_epoch(
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
        history.append(train_result.as_dict())
        print(train_result.format(), flush=True)

        val_result = regression.evaluate_regression(
            model,
            val_loader,
            criterion,
            device,
            target_transform=target_transform,
            target_keys=target_keys,
            median_index=median_index,
            epoch=epoch,
        )
        history.append(val_result.as_dict())
        print(val_result.format(), flush=True)

        # Selected on calorie MAE in real units, not on the training loss. The loss lives
        # in standardised log space and improving it is not the same as getting closer in
        # kilocalories.
        energy_mae = val_result.mae["energy"]
        if energy_mae < best:
            best = energy_mae
            checkpoint.save_checkpoint(
                args.out / "best.pt",
                model=model,
                epoch=epoch,
                backbone=args.backbone,
                config={
                    **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "target_transform": target_transform.to_dict(),
                },
                history=history,
                best_metric=best,
            )
            print(f"  new best: {best:.1f} kcal MAE")

        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nbest calorie MAE: {best:.1f} kcal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
