"""Reductions: sum, mean, max, min — and how a reduction's gradient expands.

A reduction collapses axes, so its gradient must **expand** back along exactly
those axes. The subtlety is ``keepdims``: when the forward pass dropped the
reduced axes, the incoming gradient has fewer dimensions than the input, and
those axes have to be put back before broadcasting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from gradlab.autograd.function import Function


def _normalise_axis(axis: Any, ndim: int) -> tuple[int, ...] | None:
    if axis is None:
        return None
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    return tuple(a % ndim for a in axes)


def _expand(
    g: np.ndarray, shape: tuple[int, ...], axes: tuple[int, ...] | None, keepdims: bool
) -> np.ndarray:
    """Broadcast a reduced gradient back to the input's shape."""
    if axes is None:
        g = g.reshape((1,) * len(shape))
    elif not keepdims:
        g = np.expand_dims(g, tuple(sorted(axes)))
    return np.broadcast_to(g, shape).copy()


class Sum(Function):
    op = "sum"
    backward_rule = "d(sum a)/da = g broadcast back over the reduced axes"
    auto_unbroadcast = False

    @staticmethod
    def forward(ctx, a, *, axis=None, keepdims=False):
        axes = _normalise_axis(axis, a.ndim)
        ctx.save(input_shape=a.shape, axes=axes, keepdims=keepdims)
        ctx.set_meta(axis=list(axes) if axes else None, keepdims=bool(keepdims))
        return a.sum(axis=axes, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        return _expand(g, ctx.input_shape, ctx.axes, ctx.keepdims)


class Mean(Function):
    op = "mean"
    backward_rule = "d(mean a)/da = g / N broadcast back over the reduced axes"
    auto_unbroadcast = False

    @staticmethod
    def forward(ctx, a, *, axis=None, keepdims=False):
        axes = _normalise_axis(axis, a.ndim)
        count = a.size if axes is None else int(np.prod([a.shape[i] for i in axes]))
        ctx.save(input_shape=a.shape, axes=axes, keepdims=keepdims, count=count)
        ctx.set_meta(axis=list(axes) if axes else None, keepdims=bool(keepdims), count=count)
        return a.mean(axis=axes, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        return _expand(g, ctx.input_shape, ctx.axes, ctx.keepdims) / ctx.count


class _Extremum(Function):
    """Shared implementation for :class:`Max` and :class:`Min`.

    ``max`` is not differentiable where several entries tie for the maximum. The
    choice GradLab makes — and tests, and documents — is to **split the gradient
    equally among all tied positions**. That is a valid subgradient, it is
    symmetric under permutation of the tied inputs, and it matches
    ``torch.amax``/``torch.amin`` so the conformance tests can compare directly.
    (``torch.max(dim=...)`` instead routes the whole gradient to one index; that
    is a different, equally valid choice.)
    """

    auto_unbroadcast = False
    #: ``np.max`` or ``np.min``; set by the concrete subclass.
    _numpy_fn: ClassVar[Callable[..., Any]]

    @classmethod
    def forward(cls, ctx, a, *, axis=None, keepdims=False):
        axes = _normalise_axis(axis, a.ndim)
        extreme_keep = cls._numpy_fn(a, axis=axes, keepdims=True)
        mask = (a == extreme_keep).astype(a.dtype)
        tie_count = mask.sum(axis=axes, keepdims=True)
        ctx.save(input_shape=a.shape, axes=axes, keepdims=keepdims, mask=mask, tie_count=tie_count)
        ctx.set_meta(
            axis=list(axes) if axes else None,
            keepdims=bool(keepdims),
            max_ties=int(tie_count.max()),
        )
        return cls._numpy_fn(a, axis=axes, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        g_full = _expand(g, ctx.input_shape, ctx.axes, ctx.keepdims)
        return g_full * ctx.mask / ctx.tie_count


class Max(_Extremum):
    op = "max"
    backward_rule = "gradient flows only to the maximal entries, split equally among ties"
    _numpy_fn = staticmethod(np.max)


class Min(_Extremum):
    op = "min"
    backward_rule = "gradient flows only to the minimal entries, split equally among ties"
    _numpy_fn = staticmethod(np.min)
