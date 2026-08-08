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

from platevision import (
    checkpoint,
    conformal,
    datasets,
    ema,
    engine,
    meta,
    mixing,
    models,
    regression,
    transforms,
)
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

    # Held out of training entirely, so conformal calibration sees predictions the model was
    # never fitted to. Calibrating on data the model has trained on measures how well it
    # memorised, produces offsets that are far too small, and certifies nothing.
    calibration_samples: list = []
    if args.calibration_fraction > 0:
        shuffled = list(train_samples)
        random.Random(args.seed).shuffle(shuffled)
        cut = max(1, int(len(shuffled) * args.calibration_fraction))
        calibration_samples, train_samples = shuffled[:cut], shuffled[cut:]
        print(f"       {len(calibration_samples):,} held out for conformal calibration")

    if args.limit_train or args.limit_val:
        print(f"using: {len(train_samples):,} train, {len(val_samples):,} val")

    # Fitted on the training split only, and after the calibration split is removed. Fitting
    # across train and test leaks the test distribution into training in a way no loss curve
    # reveals.
    target_transform = TargetTransform.fit(s.targets for s in train_samples)
    print(f"target log-space mean: {[round(v, 2) for v in target_transform.mean]}")

    train_ds = datasets.Nutrition5kDataset(
        train_samples,
        transform=transforms.nutrition_train_transform(zoom_out=args.zoom_out),
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

    calibration_loader = None
    if calibration_samples:
        calibration_ds = datasets.Nutrition5kDataset(
            calibration_samples,
            transform=transforms.eval_transform(),
            target_transform=target_transform,
        )
        calibration_loader = DataLoader(
            calibration_ds, batch_size=args.batch_size, shuffle=False, **common
        )

    return train_loader, val_loader, calibration_loader, target_transform


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
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.12,
        help="share of train held out to fit conformal offsets; 0 disables",
    )
    parser.add_argument("--conformal-alpha", type=float, default=0.10)

    recipe = parser.add_argument_group("training recipe")
    recipe.add_argument("--mixup-alpha", type=float, default=0.0)
    recipe.add_argument("--cutmix-alpha", type=float, default=0.0)
    recipe.add_argument("--mix-prob", type=float, default=0.5)
    recipe.add_argument("--switch-prob", type=float, default=0.5)
    recipe.add_argument("--ema", action="store_true", help="track an EMA of the weights")
    recipe.add_argument("--ema-decay", type=float, default=0.999)
    recipe.add_argument(
        "--zoom-out",
        type=float,
        default=1.6,
        help="simulate a more distant camera; 1.0 disables. See measure_scale_dependence.py",
    )
    recipe.add_argument(
        "--select-on",
        default="pinball",
        choices=["pinball", "mae"],
        help="checkpoint criterion; mae picked an overconfident model on the first full run",
    )
    args = parser.parse_args(argv)

    engine.seed_everything(args.seed)
    device = models.resolve_device(args.device)
    print(f"device: {device}")

    target_keys = meta.target_keys()
    quantiles = meta.quantiles()
    median_index = quantiles.index(0.5)

    train_loader, val_loader, calibration_loader, target_transform = build(args, device)

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

    policy = mixing.MixingPolicy(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        prob=args.mix_prob,
        switch_prob=args.switch_prob,
        seed=args.seed,
    )
    if policy.enabled:
        print(f"mixing: mixup {args.mixup_alpha}, cutmix {args.cutmix_alpha}, p {args.mix_prob}")

    # 2,424 dishes is little data and the first full run memorised it, train loss 0.0325
    # against validation 0.0831. An average of the weights is the cheapest thing that helps.
    model_ema = ema.ModelEma(model, decay=args.ema_decay) if args.ema else None
    if model_ema is not None:
        print(f"ema: decay {args.ema_decay}")

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
            mixing=policy if policy.enabled else None,
            target_transform=target_transform,
            model_ema=model_ema,
        )
        history.append(train_result.as_dict())
        print(train_result.format(), flush=True)

        # The averaged weights are the ones that would ship, so they are the ones scored.
        # Selecting on the raw weights and exporting the EMA is a comparison between two
        # different models.
        evaluated = model_ema.module if model_ema is not None else model

        val_result = regression.evaluate_regression(
            evaluated,
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

        # Validation pinball loss by default, not calorie MAE.
        #
        # MAE ignores the interval entirely. On the first full run it selected epoch 33 at
        # 51.6 kcal MAE with its 90% interval covering 64.6% of the test set, over epoch 1
        # at 78.4 kcal and 91.7% coverage. For a project whose claim is calibrated
        # uncertainty, that is the wrong thing to optimise, and it failed silently because
        # the number it does optimise kept improving.
        criterion_value = (
            val_result.loss if args.select_on == "pinball" else val_result.mae["energy"]
        )
        if criterion_value < best:
            best = criterion_value
            checkpoint.save_checkpoint(
                args.out / "best.pt",
                model=evaluated,
                epoch=epoch,
                backbone=args.backbone,
                config={
                    **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "target_transform": target_transform.to_dict(),
                },
                history=history,
                best_metric=best,
            )
            unit = "pinball" if args.select_on == "pinball" else "kcal MAE"
            print(
                f"  new best: {best:.4f} {unit}"
                f"  (MAE {val_result.mae['energy']:.1f} kcal,"
                f" coverage {val_result.coverage.get('energy', 0) * 100:.1f}%)"
            )

        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    unit = "validation pinball loss" if args.select_on == "pinball" else "kcal MAE"
    print(f"\nbest {unit}: {best:.4f}")

    if calibration_loader is not None:
        fit_conformal(args, device, target_transform, calibration_loader)

    return 0


def fit_conformal(args, device, target_transform, loader) -> None:
    """Fit conformal offsets on the held-out split, using the best checkpoint.

    Pinball loss makes a quantile head's stated level an aspiration, not a guarantee. The
    first full run of this script reached 51.6 kcal MAE with its 90% interval covering
    64.6% of the test set: the median was well calibrated and both bounds were pulled
    inward. Conformal prediction repairs that after the fact, and unlike a tuned fudge
    factor it carries a finite-sample coverage guarantee.
    """
    model, _, _ = checkpoint.restore_nutrition_model(args.out / "best.pt")
    model.to(device).eval()

    predictions, targets = [], []
    with torch.no_grad():
        for images, batch_targets in loader:
            predictions.append(model(images.to(device)).float().cpu())
            targets.append(batch_targets.float().cpu())

    # Real units, and monotonic first: an inverted interval makes a conformity score
    # meaningless, since the miss would be measured against the wrong bound.
    stacked = regression.enforce_monotonic(torch.cat(predictions))
    predicted = target_transform.inverse(stacked)
    actual = target_transform.inverse(torch.cat(targets))

    calibration = conformal.ConformalCalibration.fit(
        predicted, actual, meta.target_keys(), alpha=args.conformal_alpha
    )

    path = args.out / "conformal.json"
    path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")

    print(f"\nconformal calibration on {calibration.calibration_size:,} held-out dishes")
    for key, offset in zip(calibration.keys, calibration.offsets, strict=True):
        print(f"  {key:<14} widen by +/- {offset:.1f}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
