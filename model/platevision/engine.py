"""Training loop.

Written out rather than delegated to a framework Trainer. This is the part of the project
an interviewer will ask about, and it is also the part where the interesting bugs live:
schedulers stepped per epoch instead of per batch, evaluation quietly running in train
mode, gradients accumulating across steps.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from platevision.mixing import MixingPolicy, mixed_criterion


@dataclass(slots=True)
class AverageMeter:
    """Running mean weighted by batch size, so a short final batch cannot skew the epoch."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass(slots=True)
class EpochResult:
    epoch: int
    split: str
    loss: float
    top1: float
    top5: float
    seconds: float
    lr: float = 0.0
    resolution: int = 0
    # Recorded because top-1 on a mixed batch is not comparable to evaluation top-1: the
    # input is a blend of two images, so there is no single right answer. Without this
    # flag in the history, an ablation table silently compares different quantities.
    mixed: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def format(self) -> str:
        top1 = f"{self.top1:6.2f}%" + ("*" if self.mixed else " ")
        line = (
            f"epoch {self.epoch:>3} {self.split:<5} "
            f"loss {self.loss:.4f}  top1 {top1}  top5 {self.top5:6.2f}%  "
        )
        # Evaluation has no optimiser, so its lr is structurally zero. Printing it reads
        # like the schedule collapsed mid-epoch, which is exactly the bug people look for.
        if self.split == "train":
            line += f"lr {self.lr:.2e}  "
            if self.resolution:
                line += f"res {self.resolution}  "
        return line + f"{self.seconds:.1f}s"


@dataclass(slots=True)
class History:
    entries: list[EpochResult] = field(default_factory=list)

    def add(self, result: EpochResult) -> None:
        self.entries.append(result)

    def best(self, split: str = "val", key: str = "top1") -> EpochResult | None:
        candidates = [e for e in self.entries if e.split == split]
        return max(candidates, key=lambda e: getattr(e, key)) if candidates else None

    def as_dicts(self) -> list[dict]:
        return [e.as_dict() for e in self.entries]


def seed_everything(seed: int) -> None:
    """Seed Python, numpy, and torch.

    Note this makes runs comparable, not bitwise reproducible: cuDNN kernel selection and
    non-deterministic reductions still vary unless torch.use_deterministic_algorithms is
    forced, which costs real throughput. Comparability is what matters for ablations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> list[float]:
    """Top-k accuracy as percentages."""
    maxk = min(max(topk), output.shape[1])
    batch_size = target.size(0)
    if batch_size == 0:
        return [0.0 for _ in topk]

    _, predicted = output.topk(maxk, dim=1, largest=True, sorted=True)
    correct = predicted.eq(target.view(-1, 1).expand_as(predicted))

    results = []
    for k in topk:
        k_eff = min(k, maxk)
        hits = correct[:, :k_eff].any(dim=1).sum().item()
        results.append(100.0 * hits / batch_size)
    return results


def cosine_warmup_factor(
    step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.0
) -> float:
    """Learning-rate multiplier: linear warmup then cosine decay.

    Stepped per optimiser step, not per epoch. Stepping per epoch is the classic silent
    bug here: the schedule still runs, it just completes hundreds of times too slowly and
    the run finishes at nearly the peak learning rate.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    denominator = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_factor(step, warmup_steps, total_steps, min_ratio),
    )


def train_one_epoch(
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
    mixing: MixingPolicy | None = None,
    ema=None,
    resolution: int = 0,
    distiller=None,
) -> EpochResult:
    model.train()
    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    started = time.perf_counter()
    amp_enabled = scaler is not None and scaler.is_enabled()
    any_mixed = False

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        mix = mixing(images, targets) if mixing is not None else None
        if mix is not None:
            images = mix.images
            any_mixed = True

        # Zero before the forward pass, not after the step, so an exception mid-epoch
        # cannot leave stale gradients to be silently applied on the next iteration.
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = mixed_criterion(criterion, outputs, mix) if mix else criterion(outputs, targets)

            if distiller is not None:
                # The teacher scores the same augmented batch the student just saw, which
                # is why its opinion is usable at all. Inside autocast so both models run
                # at the same precision.
                teacher_logits = distiller.teacher_logits(images)
                loss = distiller.combine(outputs, teacher_logits, loss)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm:
                # Unscale first or the clip threshold is applied to scaled gradients and
                # effectively does nothing.
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # After the optimiser step, never before: averaging the pre-step weights lags the
        # live model by exactly one update and quietly weakens the average.
        if ema is not None:
            ema.update(model)

        if scheduler is not None:
            scheduler.step()

        batch = targets.size(0)
        scored_against = mix.dominant_target if mix else targets
        top1, top5 = accuracy(outputs.detach().float(), scored_against, topk=(1, 5))
        loss_meter.update(loss.item(), batch)
        top1_meter.update(top1, batch)
        top5_meter.update(top5, batch)

        if log_every and step % log_every == 0:
            print(
                f"  step {step:>5}  loss {loss_meter.mean:.4f}  top1 {top1_meter.mean:.2f}%",
                flush=True,
            )

    return EpochResult(
        epoch=epoch,
        split="train",
        loss=loss_meter.mean,
        top1=top1_meter.mean,
        top5=top5_meter.mean,
        seconds=time.perf_counter() - started,
        lr=optimizer.param_groups[0]["lr"],
        resolution=resolution,
        mixed=any_mixed,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    *,
    epoch: int = 0,
    split: str = "val",
) -> EpochResult:
    model.eval()
    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    started = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch = targets.size(0)
        top1, top5 = accuracy(outputs.float(), targets, topk=(1, 5))
        loss_meter.update(loss.item(), batch)
        top1_meter.update(top1, batch)
        top5_meter.update(top5, batch)

    return EpochResult(
        epoch=epoch,
        split=split,
        loss=loss_meter.mean,
        top1=top1_meter.mean,
        top5=top5_meter.mean,
        seconds=time.perf_counter() - started,
    )
