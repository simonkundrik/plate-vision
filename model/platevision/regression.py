"""Training and evaluation for the nutrition head.

Separate from ``engine`` because almost nothing transfers: there is no top-k accuracy, the
metrics have to be reported in real units rather than the space the model trains in, and
the headline number is interval coverage rather than a score.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from platevision.engine import AverageMeter
from platevision.quantile import enforce_monotonic

# Below this, a target is treated as too small for a percentage error to be meaningful.
MAPE_FLOOR = 1e-3


@dataclass(slots=True)
class RegressionResult:
    epoch: int
    split: str
    loss: float
    seconds: float
    lr: float = 0.0
    mae: dict[str, float] = field(default_factory=dict)
    mape: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    crossing_rate: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def format(self, key: str = "energy") -> str:
        line = f"epoch {self.epoch:>3} {self.split:<5} loss {self.loss:.4f}"
        if key in self.mae:
            line += f"  {key} MAE {self.mae[key]:7.1f}  MAPE {self.mape[key] * 100:5.1f}%"
        if key in self.coverage:
            line += f"  cover {self.coverage[key] * 100:5.1f}%"
        if self.split == "train":
            line += f"  lr {self.lr:.2e}"
        elif self.crossing_rate:
            line += f"  crossed {self.crossing_rate * 100:.2f}%"
        return line + f"  {self.seconds:.1f}s"


def train_regression_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int = 0,
    scheduler=None,
    scaler=None,
    max_grad_norm: float | None = None,
    log_every: int = 0,
) -> RegressionResult:
    model.train()
    loss_meter = AverageMeter()
    started = time.perf_counter()
    amp_enabled = scaler is not None and scaler.is_enabled()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predictions = model(images)
            loss = criterion(predictions, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), targets.size(0))
        if log_every and step % log_every == 0:
            print(f"  step {step:>5}  loss {loss_meter.mean:.4f}", flush=True)

    return RegressionResult(
        epoch=epoch,
        split="train",
        loss=loss_meter.mean,
        seconds=time.perf_counter() - started,
        lr=optimizer.param_groups[0]["lr"],
    )


@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    *,
    target_transform,
    target_keys: list[str],
    median_index: int,
    epoch: int = 0,
    split: str = "val",
) -> RegressionResult:
    """Evaluate and report every metric in real units.

    The model trains in standardised log space, so its loss there is not interpretable.
    Predictions are inverted back to kilocalories and grams before anything is measured,
    because "MAE 0.31" in log space tells nobody whether the model is usable.
    """
    model.eval()
    loss_meter = AverageMeter()
    started = time.perf_counter()

    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        predictions = model(images)
        loss_meter.update(criterion(predictions, targets).item(), targets.size(0))

        all_predictions.append(predictions.float().cpu())
        all_targets.append(targets.float().cpu())

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    # Measured before sorting: how often the model put its own bounds in the wrong order.
    # Sorting first would erase the diagnostic entirely.
    crossing_rate = crossings(predictions)
    predictions = enforce_monotonic(predictions)

    real_predictions = target_transform.inverse(predictions)
    real_targets = target_transform.inverse(targets)

    return RegressionResult(
        epoch=epoch,
        split=split,
        loss=loss_meter.mean,
        seconds=time.perf_counter() - started,
        mae=per_target_mae(real_predictions, real_targets, target_keys, median_index),
        mape=per_target_mape(real_predictions, real_targets, target_keys, median_index),
        coverage=per_target_coverage(real_predictions, real_targets, target_keys),
        crossing_rate=crossing_rate,
    )


def crossings(predictions: torch.Tensor) -> float:
    """Fraction of predictions whose quantile outputs are not already in ascending order."""
    ordered = predictions.sort(dim=-1).values
    disordered = (predictions != ordered).any(dim=-1)
    return disordered.float().mean().item()


def per_target_mae(
    predictions: torch.Tensor, targets: torch.Tensor, keys: list[str], median_index: int
) -> dict[str, float]:
    errors = (predictions[..., median_index] - targets).abs().mean(dim=0)
    return {key: errors[i].item() for i, key in enumerate(keys)}


def per_target_mape(
    predictions: torch.Tensor, targets: torch.Tensor, keys: list[str], median_index: int
) -> dict[str, float]:
    """Mean absolute percentage error, skipping targets too close to zero.

    A dish labelled near-zero calories would otherwise contribute an unbounded term and
    make the average meaningless.
    """
    medians = predictions[..., median_index]
    result: dict[str, float] = {}
    for i, key in enumerate(keys):
        usable = targets[:, i].abs() > MAPE_FLOOR
        if not usable.any():
            result[key] = float("nan")
            continue
        relative = (medians[usable, i] - targets[usable, i]).abs() / targets[usable, i].abs()
        result[key] = relative.mean().item()
    return result


def per_target_coverage(
    predictions: torch.Tensor, targets: torch.Tensor, keys: list[str]
) -> dict[str, float]:
    """How often the predicted interval actually contains the truth.

    The number that makes the uncertainty claim checkable. A stated 90 percent interval
    holding 60 percent of the time is not a 90 percent interval, and good MAE does not
    excuse it.
    """
    inside = (targets >= predictions[..., 0]) & (targets <= predictions[..., -1])
    fractions = inside.float().mean(dim=0)
    return {key: fractions[i].item() for i, key in enumerate(keys)}
