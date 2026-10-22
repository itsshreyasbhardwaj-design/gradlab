"""Training history: the record of what actually happened during a run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class History:
    """Per-epoch metrics from a training run, plus the config that produced them.

    Everything here is measured during the run. Nothing in GradLab writes a
    plausible-looking curve: if a value is absent (no validation set, say) the
    list is simply empty, and the Microscope draws nothing rather than
    interpolating.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_metric: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def record(self, epoch: int, **metrics: float | None) -> None:
        self.epochs.append(int(epoch))
        for key, value in metrics.items():
            if value is None:
                continue
            getattr(self, key).append(float(value))

    @property
    def best_val_loss(self) -> float | None:
        return min(self.val_loss) if self.val_loss else None

    @property
    def best_epoch(self) -> int | None:
        if not self.val_loss:
            return None
        return self.epochs[int(np.argmin(self.val_loss))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_metric": self.train_metric,
            "val_metric": self.val_metric,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "config": self.config,
            "warnings": self.warnings,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def summary(self) -> str:
        if not self.epochs:
            return "History(empty)"
        lines = [
            f"{len(self.epochs)} epochs | "
            f"train loss {self.train_loss[0]:.6f} -> {self.train_loss[-1]:.6f}"
        ]
        if self.val_loss:
            lines.append(
                f"  val loss {self.val_loss[0]:.6f} -> {self.val_loss[-1]:.6f} "
                f"(best {self.best_val_loss:.6f} at epoch {self.best_epoch})"
            )
        if self.val_metric:
            lines.append(f"  val metric {self.val_metric[0]:.4f} -> {self.val_metric[-1]:.4f}")
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)
