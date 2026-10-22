"""Adam — adaptive moment estimation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from gradlab.optim.optimizer import Optimizer
from gradlab.tensor import Tensor


class Adam(Optimizer):
    r"""Adam (Kingma & Ba, 2015): a per-parameter learning rate from gradient statistics.

    Adam keeps two exponential moving averages per parameter — the mean of the
    gradient and the mean of its square:

    .. math::
        m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad
        v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2

    and steps by the first divided by the root of the second:

    .. math:: \theta \leftarrow \theta - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}

    Dividing by :math:`\sqrt{v}` makes the step size roughly scale-invariant: a
    parameter whose gradients are consistently small gets a proportionally
    larger step, which is why Adam needs much less learning-rate tuning than
    SGD.

    **Bias correction is not cosmetic.** Both averages start at zero, so at
    :math:`t=1` with :math:`\beta_2 = 0.999` the raw :math:`v_1` is only
    :math:`0.001\,g^2` — a thousand times too small — and
    :math:`\sqrt{v_1}` in the denominator would make the very first step
    enormous. Dividing by :math:`1-\beta^t` rescales each average by exactly the
    total weight it has actually accumulated:

    .. math:: \hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}

    At :math:`t=1` this makes :math:`\hat m_1 = g_1` and
    :math:`\hat v_1 = g_1^2` exactly, so the first step has magnitude
    :math:`\eta` as intended. The correction decays geometrically and is
    negligible after a few hundred steps.

    ``weight_decay`` here is the **coupled** form (added to the gradient), which
    means it also gets divided by :math:`\sqrt{\hat v}` and is therefore not a
    true L2 penalty. :class:`~gradlab.optim.adamw.AdamW` fixes that.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        error_on_nonfinite_grad: bool = True,
    ) -> None:
        super().__init__(params, lr, error_on_nonfinite_grad)
        beta1, beta2 = betas
        for name, beta in (("beta1", beta1), ("beta2", beta2)):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"{name} must be in [0, 1); got {beta}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive; got {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative; got {weight_decay}.")
        self.beta1, self.beta2 = float(beta1), float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

    #: Subclass hook: AdamW overrides this to decay the parameter directly.
    _decoupled_decay = False

    def step(self) -> None:
        self._step_count += 1
        for i, param in enumerate(self.params):
            grad = self._usable_grad(param, i)
            if grad is None:
                continue

            state = self._state_for(param)
            if "m" not in state:
                state["m"] = np.zeros_like(param.data)
                state["v"] = np.zeros_like(param.data)
                state["t"] = 0
            state["t"] += 1
            t = state["t"]

            if self.weight_decay and not self._decoupled_decay:
                grad = grad + self.weight_decay * param.data

            state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * (grad * grad)

            m_hat = state["m"] / (1.0 - self.beta1**t)
            v_hat = state["v"] / (1.0 - self.beta2**t)

            delta = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay and self._decoupled_decay:
                delta = delta + self.lr * self.weight_decay * param.data
            self._apply(param, delta)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay}, params={len(self.params)})"
        )
