"""The :class:`Function` base class — how a differentiable operation is defined.

Every differentiable operation in GradLab subclasses ``Function`` and provides a
static ``forward`` (numpy in, numpy out) and a static ``backward`` (the
derivative rule). ``Function.apply`` does all the graph bookkeeping, so a rule
author writes only the mathematics.

    class Mul(Function):
        op = "mul"
        backward_rule = "d(a*b)/da = b*g ;  d(a*b)/db = a*g"

        @staticmethod
        def forward(ctx, a, b):
            ctx.save(a=a, b=b)
            return a * b

        @staticmethod
        def backward(ctx, g):
            return ctx.a * g, ctx.b * g

Gradients returned by ``backward`` are in the *output* (broadcast) shape.
``apply`` reduces each one to its input's shape with :func:`unbroadcast`, so no
rule reimplements broadcast reduction. A rule whose gradients are already in
input shape (``matmul``, reductions, reshape) sets ``auto_unbroadcast = False``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gradlab._config import is_grad_enabled
from gradlab.autograd.graph import Context, Node


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` to ``shape`` by summing over the axes NumPy broadcast.

    This is the step that makes broadcasting differentiable, and it is where a
    hand-written autodiff engine most often goes wrong. Two distinct reductions
    are needed:

    1. Leading axes that the input did not have at all are summed away.
    2. Axes the input had as size 1, which broadcasting stretched, are summed
       with ``keepdims=True``.

    So adding a bias of shape ``(3,)`` to a batch of shape ``(32, 3)`` yields a
    bias gradient of shape ``(3,)`` that has been summed across all 32 rows —
    not a ``(32, 3)`` array, and not an unreduced row.

    >>> unbroadcast(np.ones((32, 3)), (3,)).tolist()
    [32.0, 32.0, 32.0]
    >>> unbroadcast(np.ones((4, 5)), ()).item()
    20.0
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    if shape:
        axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
        if axes:
            grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


class Function:
    """Base class for a differentiable operation."""

    #: Short name shown in the graph and the Microscope.
    op: str = ""
    #: Human-readable derivative, displayed verbatim in the Microscope.
    backward_rule: str = ""
    #: Whether ``apply`` should reduce each returned gradient to its input shape.
    auto_unbroadcast: bool = True

    # -- to be implemented by subclasses ----------------------------------

    @staticmethod
    def forward(ctx: Context, *args: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> Any:
        raise NotImplementedError

    # -- machinery ---------------------------------------------------------

    @classmethod
    def run_backward(cls, ctx: Context, grad: np.ndarray, inputs: tuple) -> tuple:
        """Call the rule and normalise its result to one gradient per input."""
        out = cls.backward(ctx, grad)
        grads = out if isinstance(out, tuple) else (out,)
        if len(grads) != len(inputs):
            raise RuntimeError(
                f"{cls.__name__}.backward returned {len(grads)} gradient(s) for "
                f"{len(inputs)} input(s). A backward rule must return exactly one "
                "gradient per input (use None for a non-differentiable input)."
            )
        result: list[np.ndarray | None] = []
        for g, tensor in zip(grads, inputs, strict=True):
            if g is None or not tensor.requires_grad:
                result.append(None)
                continue
            g = np.asarray(g)
            if cls.auto_unbroadcast:
                g = unbroadcast(g, tensor.shape)
            if g.shape != tensor.shape:
                raise RuntimeError(
                    f"{cls.__name__}.backward produced a gradient of shape {g.shape} "
                    f"for an input of shape {tensor.shape}. A gradient must always "
                    "have exactly the shape of the value it differentiates."
                )
            result.append(g.astype(tensor.dtype, copy=False))
        return tuple(result)

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any):
        """Run the op, wrap the result, and record a graph node if needed.

        All positional arguments are differentiable tensor inputs (coerced with
        ``as_tensor``). Non-differentiable configuration — axes, flags, constant
        exponents — goes in keyword arguments and is not differentiated.
        """
        from gradlab.tensor import Tensor, as_tensor

        inputs: tuple[Tensor, ...] = tuple(as_tensor(a) for a in args)
        ctx = Context()
        raw = cls.forward(ctx, *(t.data for t in inputs), **kwargs)
        data = np.asarray(raw)

        record = is_grad_enabled() and any(t.requires_grad for t in inputs)
        out = Tensor(data, requires_grad=record, copy=False, _internal=True)
        if record:
            out._grad_fn = Node(
                op=cls.op or cls.__name__.lower(),
                fn=cls,
                inputs=inputs,
                output=out,
                ctx=ctx,
                backward_rule=cls.backward_rule,
            )
        return out
