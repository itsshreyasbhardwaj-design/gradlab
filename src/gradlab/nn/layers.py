"""Layers: Linear, Flatten, Dropout."""

from __future__ import annotations

import numpy as np

from gradlab._config import DEFAULT_DTYPE, default_rng
from gradlab._errors import ShapeError
from gradlab.nn.initialization import get_initializer
from gradlab.nn.module import Module, Parameter
from gradlab.tensor import Tensor


class Linear(Module):
    """A fully connected layer: ``y = x @ W + b``.

    The weight is stored with shape ``(in_features, out_features)`` — note the
    order. GradLab chose this over PyTorch's ``(out, in)`` so that the forward
    pass reads exactly like the mathematics: ``x @ W`` with no transpose. The
    cost is that copying weights from a PyTorch model needs a ``.T``; the
    conformance tests do this explicitly.

    Any number of leading batch axes is accepted; the last axis is features.

    Parameters
    ----------
    in_features, out_features
        Feature counts.
    bias
        Include a learnable bias (default ``True``).
    init
        Name from :data:`~gradlab.nn.initialization.INITIALIZERS` or a callable.
        Defaults to Xavier uniform. Use ``"kaiming"`` for ReLU stacks.
    seed
        Seed for this layer's initialisation, for reproducibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "xavier_uniform",
        seed: int | None = None,
        generator: np.random.Generator | None = None,
        dtype=None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"Linear needs positive feature counts; got in_features={in_features}, "
                f"out_features={out_features}."
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.init_name = init if isinstance(init, str) else getattr(init, "__name__", "callable")

        self.bias: Parameter | None
        initializer = get_initializer(init)
        rng = generator if generator is not None else default_rng(seed)
        weight = initializer(
            (self.in_features, self.out_features), generator=rng, dtype=dtype or DEFAULT_DTYPE
        )
        self.weight = Parameter(weight)
        if bias:
            self.bias = Parameter(np.zeros(self.out_features, dtype=dtype or DEFAULT_DTYPE))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0:
            raise ShapeError("Linear expects at least a 1-D input (features on the last axis).")
        if x.shape[-1] != self.in_features:
            raise ShapeError(
                f"Linear(in_features={self.in_features}) received an input whose last "
                f"axis is {x.shape[-1]} (full shape {x.shape}). The last axis must be "
                "the feature axis."
            )
        out = x @ self.weight
        if self.bias is not None:
            # Broadcasting the bias across every batch row is exactly the case
            # whose gradient must be summed back down to shape (out_features,).
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, init={self.init_name!r}"
        )


class Flatten(Module):
    """Collapse every axis from ``start_dim`` onward into one."""

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(start_dim=self.start_dim)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}"


class Dropout(Module):
    """Inverted dropout: zero each activation with probability ``p`` while training.

    The surviving activations are divided by ``1 - p`` during training ("inverted"
    dropout) so that the *expected* value of the output matches the input. That
    is what makes evaluation a plain identity — no rescaling is needed at test
    time, and ``model.eval()`` is all that has to change.

    Dropout owns its own generator. Pass ``seed`` for a reproducible mask
    sequence; note that the sequence advances on every forward pass, so
    reproducibility means "same seed, same number of forward passes".
    """

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout p must be in [0, 1); got {p}.")
        self.p = float(p)
        self._rng = default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = self._rng.random(x.shape) >= self.p
        mask = keep.astype(x.dtype) / (1.0 - self.p)
        return x * Tensor(mask, copy=False, _internal=True)

    def extra_repr(self) -> str:
        return f"p={self.p}"
