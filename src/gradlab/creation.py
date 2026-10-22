"""Tensor creation helpers.

Every randomised helper takes an explicit ``seed`` (or a ``numpy.random.Generator``)
and never touches NumPy's global random state, so a seeded GradLab program is
reproducible regardless of what else runs in the process.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gradlab._config import DEFAULT_DTYPE, default_rng
from gradlab.tensor import Tensor


def _shape(args: tuple) -> tuple[int, ...]:
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(int(s) for s in args[0])
    return tuple(int(s) for s in args)


def zeros(*shape: Any, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    return Tensor(np.zeros(_shape(shape), dtype=dtype or DEFAULT_DTYPE), requires_grad, copy=False)


def ones(*shape: Any, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    return Tensor(np.ones(_shape(shape), dtype=dtype or DEFAULT_DTYPE), requires_grad, copy=False)


def full(shape: Any, value: float, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    return Tensor(
        np.full(_shape((shape,)), value, dtype=dtype or DEFAULT_DTYPE), requires_grad, copy=False
    )


def zeros_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros_like(t.data), requires_grad, copy=False)


def ones_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones_like(t.data), requires_grad, copy=False)


def eye(n: int, m: int | None = None, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    return Tensor(np.eye(n, m, dtype=dtype or DEFAULT_DTYPE), requires_grad, copy=False)


def arange(*args: Any, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    return Tensor(np.arange(*args, dtype=dtype or DEFAULT_DTYPE), requires_grad, copy=False)


def linspace(
    start: float, stop: float, num: int = 50, requires_grad: bool = False, dtype: Any = None
) -> Tensor:
    return Tensor(
        np.linspace(start, stop, num, dtype=dtype or DEFAULT_DTYPE), requires_grad, copy=False
    )


def randn(
    *shape: Any,
    requires_grad: bool = False,
    seed: int | None = None,
    generator: np.random.Generator | None = None,
    dtype: Any = None,
) -> Tensor:
    """Standard-normal samples. Pass ``seed`` or ``generator`` for reproducibility."""
    rng = generator if generator is not None else default_rng(seed)
    data = rng.standard_normal(_shape(shape)).astype(dtype or DEFAULT_DTYPE)
    return Tensor(data, requires_grad, copy=False)


def rand(
    *shape: Any,
    requires_grad: bool = False,
    seed: int | None = None,
    generator: np.random.Generator | None = None,
    dtype: Any = None,
) -> Tensor:
    """Uniform ``[0, 1)`` samples. Pass ``seed`` or ``generator`` for reproducibility."""
    rng = generator if generator is not None else default_rng(seed)
    data = rng.random(_shape(shape)).astype(dtype or DEFAULT_DTYPE)
    return Tensor(data, requires_grad, copy=False)


def from_numpy(array: np.ndarray, requires_grad: bool = False, copy: bool = True) -> Tensor:
    """Wrap an existing array. Copies by default; pass ``copy=False`` to share."""
    return Tensor(array, requires_grad, copy=copy)


def concat(tensors: list[Tensor], axis: int = 0) -> Tensor:
    """Join tensors along an existing axis (differentiable)."""
    from gradlab.ops.shape import Concat

    return Concat.apply(*tensors, axis=axis)


def stack(tensors: list[Tensor], axis: int = 0) -> Tensor:
    """Join tensors along a new axis (differentiable, built from reshape + concat)."""
    expanded = [t.reshape((*t.shape[:axis], 1, *t.shape[axis:])) for t in tensors]
    return concat(expanded, axis=axis)
