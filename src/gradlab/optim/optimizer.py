"""The :class:`Optimizer` base class.

Common to every optimiser: holding the parameter list, clearing gradients,
validating hyperparameters, keeping per-parameter state, and performing the
actual write under :func:`~gradlab._config.no_grad` so that an update never
becomes part of the next forward pass's graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from gradlab._config import no_grad
from gradlab._errors import GradLabError
from gradlab.tensor import Tensor


class Optimizer:
    """Base class for gradient-based optimisers.

    Parameters
    ----------
    params
        The tensors to update — typically ``model.parameters()``.
    lr
        Learning rate. Must be finite and positive.
    error_on_nonfinite_grad
        When ``True`` (the default), a ``nan`` or ``inf`` gradient raises rather
        than being written into a parameter. GradLab does not silently clip or
        skip: a non-finite gradient destroys a parameter irrecoverably and
        almost always means the loss diverged several steps earlier, so the
        useful behaviour is to stop and say so. Set ``False`` if you are
        deliberately handling it yourself.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        error_on_nonfinite_grad: bool = True,
    ) -> None:
        self.params: list[Tensor] = list(params)
        if not self.params:
            raise GradLabError(
                "An optimiser was created with no parameters. If you passed "
                "model.parameters(), check the model actually registered any "
                "(assign Parameters to self in __init__, after super().__init__())."
            )
        for i, p in enumerate(self.params):
            if not isinstance(p, Tensor):
                raise TypeError(f"Parameter {i} is a {type(p).__name__}, not a Tensor.")
            if not p.is_leaf:
                raise GradLabError(
                    f"Parameter {i} is not a leaf tensor. An optimiser can only update "
                    "leaves; pass model.parameters(), not intermediate activations."
                )
        self.lr = self._check_lr(lr)
        self.error_on_nonfinite_grad = error_on_nonfinite_grad
        self.state: dict[int, dict[str, Any]] = {}
        self._step_count = 0

    @staticmethod
    def _check_lr(lr: float) -> float:
        lr = float(lr)
        if not np.isfinite(lr):
            raise ValueError(f"Learning rate must be finite; got {lr}.")
        if lr < 0:
            raise ValueError(f"Learning rate must be non-negative; got {lr}.")
        return lr

    def set_lr(self, lr: float) -> None:
        """Change the learning rate (for a schedule). Validated like the initial one."""
        self.lr = self._check_lr(lr)

    def zero_grad(self) -> None:
        """Clear ``.grad`` on every parameter.

        Needed because GradLab (like PyTorch) *accumulates* gradients, which is
        what makes gradient accumulation across micro-batches possible. The
        price is that forgetting this call silently sums gradients across steps.
        """
        for p in self.params:
            p.zero_grad()

    def _state_for(self, param: Tensor) -> dict[str, Any]:
        return self.state.setdefault(id(param), {})

    def _usable_grad(self, param: Tensor, index: int) -> np.ndarray | None:
        """Return the gradient to apply, or ``None`` to skip this parameter."""
        if not param.requires_grad:
            return None  # deliberately frozen
        if param._grad is None:
            return None  # took no part in this loss
        grad = param._grad.data
        if self.error_on_nonfinite_grad and not np.all(np.isfinite(grad)):
            bad = int((~np.isfinite(grad)).sum())
            raise GradLabError(
                f"Parameter {index} (shape {param.shape}) has {bad} non-finite "
                "gradient value(s). GradLab will not write nan/inf into a parameter, "
                "because that destroys the model and the real cause is upstream — "
                "usually a diverging loss, a learning rate that is too high, or a "
                "log/division domain problem. Inspect the loss history, or pass "
                "error_on_nonfinite_grad=False to take responsibility yourself."
            )
        return grad

    def step(self) -> None:
        """Apply one update. Implemented by subclasses inside ``no_grad``."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Serialise optimiser state, keyed by parameter position (not id)."""
        index = {id(p): i for i, p in enumerate(self.params)}
        return {
            "step_count": self._step_count,
            "lr": self.lr,
            "state": {
                index[pid]: {
                    k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()
                }
                for pid, s in self.state.items()
                if pid in index
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step_count = int(state.get("step_count", 0))
        self.lr = self._check_lr(state.get("lr", self.lr))
        self.state = {}
        for position, entries in state.get("state", {}).items():
            param = self.params[int(position)]
            self.state[id(param)] = {
                k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in entries.items()
            }

    def _apply(self, param: Tensor, delta: np.ndarray) -> None:
        """Write ``param -= delta`` under ``no_grad``.

        This is the one sanctioned in-place mutation of a tensor in GradLab. It
        bumps the version counter, so if a graph that saved this parameter is
        later differentiated again the engine raises
        ``InplaceModificationError`` rather than returning a stale gradient.
        """
        with no_grad():
            param.data = param.data - delta

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lr={self.lr}, params={len(self.params)})"
