"""Exponential moving average of model weights.

The averaged weights usually evaluate a little better than the live ones and are far less
noisy near the end of training, so the exported model is normally the EMA copy rather than
the last step's weights.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class ModelEma:
    """Maintains a shadow copy of a model's state, updated as a moving average.

    Two details that are easy to get wrong and silent when wrong:

    The average covers the whole ``state_dict``, not just parameters. Batch-norm running
    statistics live in buffers, and an EMA that skips them ships averaged weights paired
    with whatever statistics happened to be current, which is a mismatch nothing reports.

    Integer buffers are copied rather than averaged. ``num_batches_tracked`` is a counter;
    multiplying it by a decay factor is meaningless, and on an integer tensor it silently
    truncates.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.9998,
        warmup: bool = True,
        device: torch.device | None = None,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")

        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        if device is not None:
            self.module.to(device)

        self.decay = decay
        self.warmup = warmup
        self.updates = 0
        self._shadow = self.module.state_dict()

    def effective_decay(self) -> float:
        """Decay ramped up over early updates.

        Without this the average is anchored to randomly initialised weights for
        thousands of steps, and the EMA copy evaluates far worse than the live model for
        most of a short run, which reads like the EMA itself being broken.
        """
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.effective_decay()

        for key, live in model.state_dict().items():
            shadow = self._shadow[key]
            if shadow.dtype.is_floating_point:
                shadow.mul_(decay).add_(live.detach().to(shadow.device), alpha=1.0 - decay)
            else:
                shadow.copy_(live.detach().to(shadow.device))

    @torch.no_grad()
    def set_to(self, model: nn.Module) -> None:
        """Reset the shadow to a model's current state, e.g. when resuming."""
        for key, live in model.state_dict().items():
            self._shadow[key].copy_(live.detach().to(self._shadow[key].device))

    def state_dict(self) -> dict[str, Any]:
        return {"updates": self.updates, "decay": self.decay, "module": self.module.state_dict()}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.updates = payload["updates"]
        self.decay = payload["decay"]
        self.module.load_state_dict(payload["module"])
        self._shadow = self.module.state_dict()
