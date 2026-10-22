"""Activation layers — thin ``Module`` wrappers over the tensor operations.

They exist so activations can sit inside a :class:`~gradlab.nn.module.Sequential`
and show up in a model's ``repr`` and in the Gradient Microscope's layer view.
The derivative lives with the operation in ``gradlab.ops.activation``.
"""

from __future__ import annotations

from gradlab.nn.module import Module
from gradlab.tensor import Tensor


class ReLU(Module):
    """``max(x, 0)``.

    The reason a network needs this (or any nonlinearity) is structural: a stack
    of affine maps is itself an affine map, so without a nonlinearity between
    them a deep network has exactly the expressive power of a single
    :class:`~gradlab.nn.layers.Linear`. ``examples/xor.py`` demonstrates the
    consequence — XOR is unlearnable without one.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    """``1/(1+e^-x)``, squashing to ``(0, 1)``."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    """``tanh(x)``, squashing to ``(-1, 1)`` and zero-centred."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Identity(Module):
    """Pass the input through unchanged. Useful as a placeholder in ablations."""

    def forward(self, x: Tensor) -> Tensor:
        return x
