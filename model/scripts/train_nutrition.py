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
    distillation,
    ema,
    engine,
    food101,
    meta,
    mixing,
    models,
    regression,
    transforms,
)
from platevision import (
    ingredients as ingredient_lib,
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

    # Carved out of the *validation* split, not the training one.
    #
    # The first version held out a slice of train, reasoning that the model had not been
    # fitted to it. That addresses the wrong concern. A conformal guarantee needs the
    # calibration data to be exchangeable with the data being predicted on, and a held-out
    # slice of train is not exchangeable with Nutrition5k's official test split: the model
    # generalises measurably better to it. Measured on the Phase B run, offsets fitted that
    # way came out near zero, energy +/- 1.0 and mass negative, and moved test coverage from
    # 82.2% to 83.4%. Refitting on a slice of the test split instead gave 90.5% +/- 2.3.
    #
    # The cost is evaluation data: the calibration slice is excluded from the reported
    # metrics, because scoring a model on the data used to calibrate its intervals is the
    # same circularity in a different place.
    calibration_samples: list = []
    if args.calibration_fraction > 0:
        shuffled = list(val_samples)
        random.Random(args.seed).shuffle(shuffled)
        cut = max(1, int(len(shuffled) * args.calibration_fraction))
        calibration_samples, val_samples = shuffled[:cut], shuffled[cut:]
        print(
            f"       {len(calibration_samples):,} of the validation split held out for "
            f"conformal calibration, {len(val_samples):,} left for reporting"
        )

    if args.limit_train or args.limit_val:
        print(f"using: {len(train_samples):,} train, {len(val_samples):,} val")

    # Fitted on the training split only, and after the calibration split is removed. Fitting
    # across train and test leaks the test distribution into training in a way no loss curve
    # reveals.
    target_transform = TargetTransform.fit(s.targets for s in train_samples)
    print(f"target log-space mean: {[round(v, 2) for v in target_transform.mean]}")

    # Built from the training split alone, for the same reason the transform is. A
    # vocabulary drawn from train and test together leaks which ingredients the test dishes
    # contain, and nothing in a loss curve would show it.
    vocabulary: list[str] = []
    if args.ingredient_weight > 0:
        vocabulary = ingredient_lib.build_vocabulary(
            (s.ingredients for s in train_samples), args.ingredient_min_count
        )
        print(f"ingredients: {len(vocabulary)} with >= {args.ingredient_min_count} dishes")

    train_ds = datasets.Nutrition5kDataset(
        train_samples,
        transform=transforms.nutrition_train_transform(zoom_out=args.zoom_out),
        target_transform=target_transform,
        ingredient_vocab=vocabulary or None,
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

    return train_loader, val_loader, calibration_loader, target_transform, vocabulary


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
    parser.add_argument("--food101-root", type=Path, help="the food-101 directory")
    parser.add_argument(
        "--classification-weight",
        type=float,
        default=0.0,
        # Real images and real labels, interleaved with the nutrition batches. Distillation
        # could only ever score the nutrition batch, so the backbone was pulled towards
        # 2,424 cafeteria trays by every gradient it received and defended by a teacher's
        # opinion about images it was not being shown.
        help="weight on a Food-101 cross-entropy term trained alongside; 0 disables",
    )
    parser.add_argument(
        "--limit-classification",
        type=int,
        help="use only the first N Food-101 images, for smoke tests",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        # The only configuration that guarantees the classifier rather than defending it.
        # Held at the stage-one weights, the Food-101 head stays valid by construction, so
        # no distillation and no linear probe are needed and there is nothing to re-verify.
        help="hold the backbone at its loaded weights; the nutrition head trains alone",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        # Unset by default, so this changes nothing until it is asked for. Measured: at
        # --kd-weight 0.010 the co-trained classifier head fell to 25.9% top-1 while calorie
        # error reached its best 54.7 kcal, and at 0.5 the head held 76.9% while calories
        # regressed to 68.8. The weight cannot hold both, because it asks a loss term to
        # repair damage the optimiser is doing at full rate. Slowing the backbone attacks
        # that directly.
        help="separate learning rate for the backbone; heads keep --lr",
    )
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
        default=0.4,
        help="share of the validation split held out to fit conformal offsets; 0 disables",
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
        "--ingredient-weight",
        type=float,
        default=0.0,
        # 0.057 puts the ingredient term at about 15% of the pinball loss. Raw BCE against
        # 133 weighted classes runs roughly 2.6x pinball, so a weight near 1 drowns the task
        # the model exists to do.
        help="weight on the auxiliary ingredient loss; 0 disables the head entirely",
    )
    recipe.add_argument("--ingredient-min-count", type=int, default=20)
    recipe.add_argument(
        "--kd-weight",
        type=float,
        default=0.0,
        help="weight on distilling the frozen Food-101 classifier; 0 disables",
    )
    recipe.add_argument("--kd-temperature", type=float, default=4.0)
    # A note on scale, because the first attempt got this wrong by an order of magnitude:
    # raw KL against a 101-class teacher runs about 15x the pinball loss, so --kd-weight 0.5
    # made the regression task 12% of the gradient and calorie error rose from 56.7 to 68.8
    # kcal. 0.010 puts it near 15%. The trainer prints the balance before the first epoch.
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

    train_loader, val_loader, calibration_loader, target_transform, vocabulary = build(args, device)

    model = models.NutritionModel(
        args.backbone,
        num_targets=len(target_keys),
        num_quantiles=len(quantiles),
        pretrained=args.backbone_from is None,
        drop_rate=args.drop_rate,
        num_ingredients=len(vocabulary),
        num_classes=(
            len(food101.class_keys()) if args.kd_weight > 0 or args.classification_weight > 0 else 0
        ),
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
        if model.classifier_head is not None:
            if models.load_classifier_head(model, payload["model"]):
                print("classifier head: started from the trained one, not from noise")
            else:
                print("classifier head: no match in the checkpoint, starting from noise")
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

    ingredient_criterion = None
    if vocabulary:
        # Positive weighting matters more than the loss choice here. Most ingredients are
        # absent from most dishes, so an unweighted head reaches a low loss by answering
        # "not present" to everything and learns nothing at all.
        weight = ingredient_lib.positive_weight(
            (s.ingredients for s in train_loader.dataset.samples), vocabulary
        ).to(device)
        ingredient_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        print(f"ingredient loss weight: {args.ingredient_weight}")

    # Distilling the classifier the backbone came from, on the same augmented images the
    # student sees. Fine-tuning on 2,424 cafeteria trays otherwise erodes the dish semantics
    # that make density predictable, and it invalidates the stage-one classification head,
    # which CombinedModel currently works around by re-fitting a linear probe before export.
    distiller = None
    if args.kd_weight > 0:
        if not args.backbone_from:
            raise SystemExit("--kd-weight needs --backbone-from to distil from")
        teacher, _ = checkpoint.restore_classifier(args.backbone_from)
        distiller = distillation.Distiller(
            teacher, distillation.DistillationConfig(alpha=1.0, temperature=args.kd_temperature)
        ).to(device)
        print(f"distilling the classifier: weight {args.kd_weight}, T {args.kd_temperature}")

    criterion = PinballLoss(quantiles).to(device)
    # Joint training on the classification set, which is what distillation was standing in
    # for. The teacher only ever scored cafeteria trays, so the backbone saw Food-101 data
    # exactly never and was defended by an opinion about the wrong distribution.
    classification_batches = classification_criterion = None
    if args.classification_weight > 0:
        if not args.food101_root:
            raise SystemExit("--classification-weight needs --food101-root to train against")
        if model.classifier_head is None:
            raise SystemExit(
                "--classification-weight needs a classifier head; the model was built "
                "without one because --kd-weight was 0. Pass --kd-weight 0 is fine, but "
                "the head has to exist, so this run must request it."
            )

        food_samples = datasets.build_food101_index(args.food101_root, "train")
        if args.limit_classification:
            food_samples = food_samples[: args.limit_classification]
        food_loader = DataLoader(
            datasets.Food101Dataset(food_samples, transform=transforms.nutrition_train_transform()),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        classification_batches = regression.endless(food_loader)
        classification_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        print(
            f"joint classification: {len(food_samples):,} Food-101 images, "
            f"weight {args.classification_weight}"
        )

    if args.freeze_backbone:
        frozen = models.freeze_backbone(model)
        print(
            f"backbone frozen: {frozen:,} parameters held at the weights the Food-101 "
            "head was fitted against, so that head stays exactly as accurate as measured"
        )

    optimizer = torch.optim.AdamW(
        models.parameter_groups(model, args.weight_decay, backbone_lr=args.backbone_lr),
        lr=args.lr,
    )
    if args.backbone_lr is not None:
        print(f"backbone learning rate {args.backbone_lr:.1e}, heads {args.lr:.1e}")

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

    # Any auxiliary term at all, not just the two that existed when this was written. A new
    # term that skips the balance report is exactly how the last one came to be wrong by a
    # factor of fifteen without anything saying so.
    if (
        ingredient_criterion is not None
        or distiller is not None
        or classification_criterion is not None
    ):
        report_loss_balance(
            model,
            train_loader,
            criterion,
            ingredient_criterion,
            distiller,
            args,
            device,
            classification_batches=classification_batches,
            classification_criterion=classification_criterion,
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
            ingredient_criterion=ingredient_criterion,
            ingredient_weight=args.ingredient_weight,
            distiller=distiller,
            kd_weight=args.kd_weight,
            classification_batches=classification_batches,
            classification_criterion=classification_criterion,
            classification_weight=args.classification_weight,
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


@torch.no_grad()
def report_loss_balance(
    model,
    loader,
    criterion,
    ingredient_criterion,
    distiller,
    args,
    device,
    classification_batches=None,
    classification_criterion=None,
) -> None:
    """Print what each loss term actually contributes, before training on them.

    Auxiliary weights were guessed on the first attempt and were wrong by an order of
    magnitude. Raw KL against a 101-class teacher runs about fifteen times the pinball loss,
    so a weight of 0.5 made the regression task a small minority of the gradient and calorie
    error rose from 56.7 to 68.8 kcal. Nothing in the loss curve said so, because the total
    was falling the whole time.

    One batch is enough to see the balance, and seeing it is the difference between choosing
    a weight and guessing one.
    """
    model.train()
    batch = next(iter(loader))
    images = batch[0].to(device)
    targets = batch[1].to(device)
    ingredient_targets = batch[2].to(device) if len(batch) > 2 else None

    predictions, ingredient_logits, class_logits = model.forward_with_aux(images)
    terms: list[tuple[str, float, float]] = [
        ("pinball", float(criterion(predictions, targets)), 1.0)
    ]

    if ingredient_criterion is not None and ingredient_targets is not None:
        raw = float(ingredient_criterion(ingredient_logits, ingredient_targets))
        terms.append(("ingredient", raw, args.ingredient_weight))
    if distiller is not None and class_logits is not None:
        raw = float(
            distillation.kd_loss(
                class_logits, distiller.teacher_logits(images), distiller.config.temperature
            )
        )
        terms.append(("distillation", raw, args.kd_weight))

    if classification_batches is not None and classification_criterion is not None:
        # Peeked rather than consumed would be nicer, but the generator is endless and one
        # batch out of Food-101's 75,750 costs nothing.
        cls_images, cls_labels = next(classification_batches)
        raw = float(
            classification_criterion(
                model.forward_with_aux(cls_images.to(device))[2], cls_labels.to(device)
            )
        )
        terms.append(("classification", raw, args.classification_weight))

    total = sum(raw * weight for _, raw, weight in terms)
    print()
    print("loss balance on the first batch")
    print(f"  {'term':<14}{'raw':>9}{'weight':>9}{'weighted':>10}{'share':>8}")
    for name, raw, weight in terms:
        weighted = raw * weight
        print(f"  {name:<14}{raw:>9.3f}{weight:>9.3f}{weighted:>10.3f}{weighted / total:>7.0%}")

    regression_share = terms[0][1] / total
    if regression_share < 0.5:
        print(
            f"  the regression task is {regression_share:.0%} of the gradient. "
            "Lower the auxiliary weights unless that is deliberate."
        )


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
