"""``gradlab.training`` — a minimal, honest training loop.

    history = training.fit(model, optimizer, nn.mse_loss, loader, epochs=100)
    print(history.summary())

Every number in a :class:`~gradlab.training.history.History` was measured during
the run.
"""

from __future__ import annotations

from gradlab.training.history import History
from gradlab.training.trainer import clip_grad_norm, evaluate, fit, global_grad_norm

__all__ = ["History", "clip_grad_norm", "evaluate", "fit", "global_grad_norm"]
