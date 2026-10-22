"""Matrix multiplication and its derivative.

The rule itself is short — for ``C = A @ B``::

    dL/dA = G @ B^T
    dL/dB = A^T @ G

What takes care is everything around it: NumPy's ``matmul`` promotes 1-D
operands to matrices and broadcasts leading batch dimensions, and the gradient
must undo both of those. The strategy here is to normalise to at least 2-D,
apply the plain rule, reduce any broadcast batch axes with the same
:func:`~gradlab.autograd.function.unbroadcast` helper the elementwise ops use,
and finally restore the original 1-D shapes.
"""

from __future__ import annotations

import numpy as np

from gradlab._errors import ShapeError
from gradlab.autograd.function import Function, unbroadcast


class MatMul(Function):
    op = "matmul"
    backward_rule = "dL/dA = G @ B^T ;  dL/dB = A^T @ G"
    auto_unbroadcast = False  # handled explicitly below

    @staticmethod
    def forward(ctx, a, b):
        if a.ndim == 0 or b.ndim == 0:
            raise ShapeError("matmul needs at least 1-D operands; use * for scalar multiplication.")
        inner_a = a.shape[-1]
        inner_b = b.shape[0] if b.ndim == 1 else b.shape[-2]
        if inner_a != inner_b:
            raise ShapeError(
                f"Cannot matmul shapes {a.shape} and {b.shape}: the inner dimensions "
                f"{inner_a} and {inner_b} must agree."
            )
        try:
            out = a @ b
        except ValueError as exc:
            raise ShapeError(f"Cannot matmul shapes {a.shape} and {b.shape}: {exc}") from exc
        ctx.save(a=a, b=b)
        ctx.set_meta(a_shape=list(a.shape), b_shape=list(b.shape), out_shape=list(np.shape(out)))
        return out

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.a, ctx.b
        a_was_1d, b_was_1d = a.ndim == 1, b.ndim == 1

        # Promote to >= 2-D exactly the way matmul does.
        A = a[None, :] if a_was_1d else a
        B = b[:, None] if b_was_1d else b

        # Re-insert the axes matmul collapsed, so G lines up with A @ B.
        G = g
        if a_was_1d and b_was_1d:
            G = g.reshape(1, 1)
        elif a_was_1d:
            G = np.expand_dims(g, -2)
        elif b_was_1d:
            G = np.expand_dims(g, -1)

        grad_a = G @ np.swapaxes(B, -1, -2)
        grad_b = np.swapaxes(A, -1, -2) @ G

        # Batch dimensions may have been broadcast; sum them back down.
        grad_a = unbroadcast(grad_a, A.shape).reshape(a.shape)
        grad_b = unbroadcast(grad_b, B.shape).reshape(b.shape)
        return grad_a, grad_b
