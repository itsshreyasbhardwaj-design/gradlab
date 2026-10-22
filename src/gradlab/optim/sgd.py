"""Stochastic gradient descent, with optional momentum and weight decay."""

from __future__ import annotations

from collections.abc import Iterable

from gradlab.optim.optimizer import Optimizer
from gradlab.tensor import Tensor


class SGD(Optimizer):
    r"""Gradient descent: step downhill, proportionally to the slope.

    Plain update:

    .. math:: \theta \leftarrow \theta - \eta \, g

    With momentum, a velocity accumulates past gradients so that consistent
    directions build up speed and oscillating ones cancel:

    .. math::
        v \leftarrow \mu v + g, \qquad \theta \leftarrow \theta - \eta v

    This is the "heavy ball" form, in which ``momentum`` is a decay on the
    velocity and the learning rate multiplies the velocity. Note the effective
    step size for a constant gradient is :math:`\eta/(1-\mu)`, so turning
    momentum on at 0.9 makes steps roughly ten times larger — reduce ``lr``
    accordingly rather than being surprised by divergence.

    Nesterov momentum evaluates the gradient at the look-ahead point, which here
    reduces to :math:`\theta \leftarrow \theta - \eta(g + \mu v)`.

    ``weight_decay`` is **coupled** L2: it adds :math:`\lambda \theta` to the
    gradient, so it interacts with momentum and the learning rate. For the
    decoupled variant see :class:`~gradlab.optim.adamw.AdamW`.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        error_on_nonfinite_grad: bool = True,
    ) -> None:
        super().__init__(params, lr, error_on_nonfinite_grad)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1); got {momentum}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative; got {weight_decay}.")
        if nesterov and momentum == 0.0:
            raise ValueError("nesterov momentum requires momentum > 0.")
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)

    def step(self) -> None:
        self._step_count += 1
        for i, param in enumerate(self.params):
            grad = self._usable_grad(param, i)
            if grad is None:
                continue
            if self.weight_decay:
                grad = grad + self.weight_decay * param.data
            if self.momentum:
                state = self._state_for(param)
                velocity = state.get("velocity")
                velocity = grad.copy() if velocity is None else self.momentum * velocity + grad
                state["velocity"] = velocity
                direction = grad + self.momentum * velocity if self.nesterov else velocity
            else:
                direction = grad
            self._apply(param, self.lr * direction)

    def __repr__(self) -> str:
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}, nesterov={self.nesterov}, "
            f"params={len(self.params)})"
        )
