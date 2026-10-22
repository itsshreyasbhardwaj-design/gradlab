"""Weight initialisation schemes, and the variance argument behind each.

A deep stack of layers multiplies signal variance at every step. If each layer
scales the variance by a factor :math:`k`, then after :math:`L` layers the
variance has been scaled by :math:`k^L` — so :math:`k` slightly above 1
explodes and slightly below 1 vanishes. Every scheme here is an answer to
"choose the weight scale so that :math:`k \\approx 1`".

For a layer :math:`y = Wx` with :math:`n_{in}` independent inputs of variance
:math:`\\sigma_x^2` and weights of variance :math:`\\sigma_w^2`:

.. math:: \\operatorname{Var}(y) = n_{in}\\,\\sigma_w^2\\,\\sigma_x^2

* **Xavier/Glorot** (Glorot & Bengio, 2010) targets unit gain in both
  directions at once — forward activations *and* backward gradients — so it
  compromises on :math:`\\sigma_w^2 = 2/(n_{in} + n_{out})`. Derived for
  activations that are roughly linear near the origin, which is why it suits
  ``tanh`` and ``sigmoid``.
* **He/Kaiming** (He et al., 2015) accounts for ``relu`` zeroing out about half
  of its inputs, which halves the variance. Compensating gives
  :math:`\\sigma_w^2 = 2/n_{in}`.

Neither is optimal for every architecture, and GradLab does not claim one is.
The default for :class:`~gradlab.nn.layers.Linear` is Xavier uniform because it
is the safer choice for a mixed or unknown activation; pass ``init="kaiming"``
for ReLU stacks. ``examples/init_comparison.py`` measures the difference.
"""

from __future__ import annotations

import numpy as np

from gradlab._config import DEFAULT_DTYPE, default_rng
from gradlab._errors import ShapeError


def _fan(shape: tuple[int, ...]) -> tuple[int, int]:
    """``(fan_in, fan_out)`` for a weight of shape ``(in_features, out_features)``."""
    if len(shape) != 2:
        raise ShapeError(
            f"Initialisation expects a 2-D weight of shape (in_features, out_features); "
            f"got {shape}."
        )
    return shape[0], shape[1]


def uniform(shape, low=-0.1, high=0.1, *, seed=None, generator=None, dtype=None) -> np.ndarray:
    rng = generator if generator is not None else default_rng(seed)
    return rng.uniform(low, high, shape).astype(dtype or DEFAULT_DTYPE)


def normal(shape, mean=0.0, std=0.05, *, seed=None, generator=None, dtype=None) -> np.ndarray:
    rng = generator if generator is not None else default_rng(seed)
    return (mean + std * rng.standard_normal(shape)).astype(dtype or DEFAULT_DTYPE)


def xavier_uniform(shape, gain=1.0, *, seed=None, generator=None, dtype=None) -> np.ndarray:
    """``U(-a, a)`` with ``a = gain*sqrt(6/(fan_in+fan_out))``.

    The bound is ``sqrt(3)*sigma`` because a uniform distribution on
    ``(-a, a)`` has variance ``a^2/3``; solving for the target
    ``sigma^2 = 2/(fan_in+fan_out)`` gives the familiar ``sqrt(6/...)``.
    """
    fan_in, fan_out = _fan(tuple(shape))
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, -bound, bound, seed=seed, generator=generator, dtype=dtype)


def xavier_normal(shape, gain=1.0, *, seed=None, generator=None, dtype=None) -> np.ndarray:
    """``N(0, sigma^2)`` with ``sigma = gain*sqrt(2/(fan_in+fan_out))``."""
    fan_in, fan_out = _fan(tuple(shape))
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return normal(shape, 0.0, std, seed=seed, generator=generator, dtype=dtype)


def kaiming_uniform(shape, *, mode="fan_in", seed=None, generator=None, dtype=None) -> np.ndarray:
    """ReLU-aware uniform init: ``U(-a, a)`` with ``a = sqrt(6/fan)``."""
    fan_in, fan_out = _fan(tuple(shape))
    fan = fan_in if mode == "fan_in" else fan_out
    bound = np.sqrt(6.0 / fan)
    return uniform(shape, -bound, bound, seed=seed, generator=generator, dtype=dtype)


def kaiming_normal(shape, *, mode="fan_in", seed=None, generator=None, dtype=None) -> np.ndarray:
    """ReLU-aware normal init: ``N(0, 2/fan)``."""
    fan_in, fan_out = _fan(tuple(shape))
    fan = fan_in if mode == "fan_in" else fan_out
    return normal(shape, 0.0, np.sqrt(2.0 / fan), seed=seed, generator=generator, dtype=dtype)


def zeros(shape, *, dtype=None, **_) -> np.ndarray:
    return np.zeros(shape, dtype=dtype or DEFAULT_DTYPE)


#: Schemes addressable by name from ``Linear(..., init="xavier")``.
INITIALIZERS = {
    "uniform": uniform,
    "normal": normal,
    "xavier": xavier_uniform,
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "kaiming": kaiming_uniform,
    "kaiming_uniform": kaiming_uniform,
    "kaiming_normal": kaiming_normal,
    "he": kaiming_uniform,
    "zeros": zeros,
}


def get_initializer(name):
    if callable(name):
        return name
    try:
        return INITIALIZERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown initialiser {name!r}. Available: {sorted(INITIALIZERS)}."
        ) from None
