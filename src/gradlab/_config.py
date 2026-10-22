"""Global configuration: gradient mode and default dtype.

The gradient-mode flag is a single process-global integer depth counter rather
than a boolean, so that :func:`no_grad` and :func:`enable_grad` nest correctly.
GradLab is not thread-safe; see ``docs/limitations.md``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

#: The dtype used for tensors created without an explicit one, and the dtype all
#: correctness tolerances in the test suite are stated for.
DEFAULT_DTYPE = np.float64

#: dtypes that may carry a gradient.
FLOAT_DTYPES = (np.float32, np.float64)

_grad_enabled: bool = True


def is_grad_enabled() -> bool:
    """Whether operations currently record graph history."""
    return _grad_enabled


def set_grad_enabled(mode: bool) -> None:
    """Set gradient recording on or off. Prefer the context managers."""
    global _grad_enabled
    _grad_enabled = bool(mode)


@contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph construction inside the block.

    Results computed inside are leaves with ``requires_grad=False``. Used for
    optimiser updates, evaluation, and any forward pass whose gradient is not
    needed.

    >>> import gradlab as gl
    >>> x = gl.Tensor(2.0, requires_grad=True)
    >>> with gl.no_grad():
    ...     y = x * 3
    >>> y.requires_grad, y.is_leaf
    (False, True)
    """
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


@contextmanager
def enable_grad() -> Iterator[None]:
    """Re-enable graph construction, even inside a :func:`no_grad` block."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = True
    try:
        yield
    finally:
        _grad_enabled = prev


def default_rng(seed: int | None = None) -> np.random.Generator:
    """A fresh, independent generator.

    GradLab never reads or mutates NumPy's global random state, so a seeded
    GradLab program is unaffected by anything else in the process.
    """
    return np.random.default_rng(seed)
