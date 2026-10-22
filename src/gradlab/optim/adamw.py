"""AdamW — Adam with decoupled weight decay."""

from __future__ import annotations

from collections.abc import Iterable

from gradlab.optim.adam import Adam
from gradlab.tensor import Tensor


class AdamW(Adam):
    r"""Adam with **decoupled** weight decay (Loshchilov & Hutter, 2019).

    In Adam, ``weight_decay`` is folded into the gradient, so it then passes
    through the :math:`1/\sqrt{\hat v}` rescaling. The amount of decay a weight
    receives therefore depends on its gradient history: a parameter with large,
    noisy gradients is barely regularised, and one with tiny gradients is
    regularised hard. That is not what an L2 penalty is supposed to do.

    AdamW applies the decay directly to the parameter, outside the adaptive
    rescaling:

    .. math::
        \theta \leftarrow \theta
        - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
        - \eta \lambda \theta

    Every weight now shrinks by the same relative amount per step, independent
    of its gradient scale. The practical consequence is that ``weight_decay``
    becomes a meaningful, transferable hyperparameter — and that a value tuned
    for ``Adam`` will be far too small here. The validation tests in
    ``tests/unit/test_optim.py`` assert exactly this divergence between the two.
    """

    _decoupled_decay = True

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        error_on_nonfinite_grad: bool = True,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            error_on_nonfinite_grad=error_on_nonfinite_grad,
        )
