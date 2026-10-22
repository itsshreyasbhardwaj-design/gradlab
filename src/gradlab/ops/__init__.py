"""Operation library: every differentiable primitive GradLab provides.

Importing this package registers nothing globally — ``Tensor`` methods import
the operation they need lazily. This module exists so that
``from gradlab.ops import Add`` works and so the test suite can enumerate every
operation in one place.
"""

from __future__ import annotations

from gradlab.ops.activation import (
    Cos,
    Exp,
    Log,
    LogSoftmax,
    ReLU,
    Sigmoid,
    Sin,
    Softmax,
    Sqrt,
    Tanh,
)
from gradlab.ops.arithmetic import Abs, Add, Div, Mul, Neg, Pow, PowTensor, Sub
from gradlab.ops.matrix import MatMul
from gradlab.ops.reduction import Max, Mean, Min, Sum
from gradlab.ops.shape import Concat, Identity, Index, Reshape, Transpose

#: Every differentiable operation, keyed by its ``op`` name. The Gradient
#: Microscope's restricted executor uses this as its allowlist.
ALL_OPS = {
    cls.op: cls
    for cls in (
        Add,
        Sub,
        Mul,
        Div,
        Neg,
        Pow,
        PowTensor,
        Abs,
        Exp,
        Log,
        Sqrt,
        Tanh,
        Sigmoid,
        ReLU,
        Sin,
        Cos,
        LogSoftmax,
        Softmax,
        Sum,
        Mean,
        Max,
        Min,
        MatMul,
        Identity,
        Reshape,
        Transpose,
        Index,
        Concat,
    )
}

__all__ = ["ALL_OPS", *sorted(cls.__name__ for cls in ALL_OPS.values())]
