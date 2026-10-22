"""Shape manipulation: identity, reshape, transpose, indexing.

None of these change values, so each backward rule is purely the *inverse
rearrangement* of the forward one. They all set ``auto_unbroadcast = False``
because they already return a gradient in the input's shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gradlab._errors import ShapeError
from gradlab.autograd.function import Function
from gradlab.tensor import Tensor


def _unwrap_index(index: Any) -> Any:
    """Turn any ``Tensor`` inside an index expression into a NumPy array."""
    if isinstance(index, Tensor):
        arr = index.data
        return arr.astype(np.intp) if np.issubdtype(arr.dtype, np.floating) else arr
    if isinstance(index, tuple):
        return tuple(_unwrap_index(i) for i in index)
    return index


class Identity(Function):
    """A differentiable copy (``clone``). Useful as a graph marker."""

    op = "identity"
    backward_rule = "d(a)/da = g"
    auto_unbroadcast = False

    @staticmethod
    def forward(ctx, a):
        return a.copy()

    @staticmethod
    def backward(ctx, g):
        return g


class Reshape(Function):
    op = "reshape"
    backward_rule = "reshape the gradient back to the input's shape"
    auto_unbroadcast = False

    @staticmethod
    def forward(ctx, a, *, shape):
        ctx.save(input_shape=a.shape)
        ctx.set_meta(from_shape=list(a.shape), to_shape=list(shape))
        try:
            return a.reshape(shape)
        except ValueError as exc:
            raise ShapeError(
                f"Cannot reshape a tensor of shape {a.shape} ({a.size} elements) "
                f"into {tuple(shape)}: the element counts differ."
            ) from exc

    @staticmethod
    def backward(ctx, g):
        return g.reshape(ctx.input_shape)


class Transpose(Function):
    op = "transpose"
    backward_rule = "transpose the gradient by the inverse permutation"
    auto_unbroadcast = False

    @staticmethod
    def forward(ctx, a, *, axes=None):
        perm = tuple(reversed(range(a.ndim))) if axes is None else tuple(axes)
        if sorted(perm) != list(range(a.ndim)):
            raise ShapeError(
                f"axes={axes} is not a permutation of the {a.ndim} axes of a tensor "
                f"with shape {a.shape}."
            )
        ctx.save(perm=perm)
        ctx.set_meta(permutation=list(perm))
        return a.transpose(perm)

    @staticmethod
    def backward(ctx, g):
        perm = ctx.perm
        inverse = np.argsort(perm)
        return g.transpose(tuple(inverse))


class Index(Function):
    """Differentiable indexing and slicing.

    The backward rule is a **scatter-add**, not a scatter-assign. That matters
    whenever an index repeats — ``x[[0, 0, 1]]`` reads element 0 twice, so
    element 0 must receive the sum of both upstream gradients. ``np.add.at``
    does exactly this accumulation.
    """

    op = "index"
    backward_rule = "scatter-add the gradient into a zero tensor at the read positions"
    auto_unbroadcast = False

    @staticmethod
    def forward(ctx, a, *, index):
        idx = _unwrap_index(index)
        try:
            out = a[idx]
        except (IndexError, ValueError) as exc:
            raise ShapeError(
                f"Invalid index {index!r} for a tensor of shape {a.shape}: {exc}"
            ) from exc
        ctx.save(index=idx, input_shape=a.shape, dtype=a.dtype)
        ctx.set_meta(index=repr(index), from_shape=list(a.shape), to_shape=list(np.shape(out)))
        return np.asarray(out)

    @staticmethod
    def backward(ctx, g):
        out = np.zeros(ctx.input_shape, dtype=ctx.dtype)
        np.add.at(out, ctx.index, g)
        return out


class Concat(Function):
    """Join tensors along an existing axis; the gradient is split back apart."""

    op = "concat"
    backward_rule = "split the gradient along the concatenation axis"
    auto_unbroadcast = False

    @staticmethod
    def forward(ctx, *arrays, axis=0):
        sizes = [a.shape[axis] for a in arrays]
        ctx.save(sizes=sizes, axis=axis)
        ctx.set_meta(axis=int(axis), sizes=[int(s) for s in sizes])
        return np.concatenate(arrays, axis=axis)

    @staticmethod
    def backward(ctx, g):
        bounds = np.cumsum(ctx.sizes)[:-1]
        return tuple(np.split(g, bounds, axis=ctx.axis))
