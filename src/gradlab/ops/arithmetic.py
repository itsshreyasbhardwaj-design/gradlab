"""Elementwise arithmetic and its derivatives.

Each rule is the textbook derivative, written out. The gradients returned are in
the *output* (broadcast) shape; ``Function.apply`` reduces them to each input's
shape, which is what makes ``matrix + bias_vector`` differentiate correctly.
"""

from __future__ import annotations

import numpy as np

from gradlab._errors import DomainError
from gradlab.autograd.function import Function


class Add(Function):
    op = "add"
    backward_rule = "d(a+b)/da = g ;  d(a+b)/db = g"

    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, g):
        return g, g


class Sub(Function):
    op = "sub"
    backward_rule = "d(a-b)/da = g ;  d(a-b)/db = -g"

    @staticmethod
    def forward(ctx, a, b):
        return a - b

    @staticmethod
    def backward(ctx, g):
        return g, -g


class Mul(Function):
    op = "mul"
    backward_rule = "d(a*b)/da = b*g ;  d(a*b)/db = a*g"

    @staticmethod
    def forward(ctx, a, b):
        ctx.save(a=a, b=b)
        return a * b

    @staticmethod
    def backward(ctx, g):
        return ctx.b * g, ctx.a * g


class Div(Function):
    op = "div"
    backward_rule = "d(a/b)/da = g/b ;  d(a/b)/db = -a*g/b^2"

    @staticmethod
    def forward(ctx, a, b):
        if np.any(b == 0):
            raise DomainError(
                "Division by zero. GradLab raises here instead of producing inf/nan, "
                "because the resulting gradient would be meaningless. Add an epsilon "
                "to the denominator if that is what you intend."
            )
        ctx.save(a=a, b=b)
        return a / b

    @staticmethod
    def backward(ctx, g):
        return g / ctx.b, -ctx.a * g / (ctx.b * ctx.b)


class Neg(Function):
    op = "neg"
    backward_rule = "d(-a)/da = -g"

    @staticmethod
    def forward(ctx, a):
        return -a

    @staticmethod
    def backward(ctx, g):
        return -g


class Pow(Function):
    """``a ** n`` for a fixed real exponent ``n``.

    The exponent is a keyword argument, not a tensor input, so it is a constant
    and is not differentiated. Use ``a ** b`` with a tensor ``b`` for
    :class:`PowTensor`.
    """

    op = "pow"
    backward_rule = "d(a^n)/da = n*a^(n-1)*g"

    @staticmethod
    def forward(ctx, a, *, exponent):
        if exponent != int(exponent) and np.any(a < 0):
            raise DomainError(
                f"a ** {exponent} is not real-valued for negative a "
                f"(min was {float(np.min(a)):.6g}). Supported domain for a "
                "non-integer exponent is a >= 0."
            )
        if exponent < 0 and np.any(a == 0):
            raise DomainError(f"0 ** {exponent} is undefined (negative exponent at zero).")
        ctx.save(a=a)
        ctx.set_meta(exponent=float(exponent))
        return a**exponent

    @staticmethod
    def backward(ctx, g):
        n = ctx.meta["exponent"]
        a = ctx.a
        if 0 < n < 1 and np.any(a == 0):
            raise DomainError(
                f"The derivative of a^{n} is unbounded at a = 0; GradLab will not "
                "return inf. Keep inputs strictly positive for fractional exponents."
            )
        return n * (a ** (n - 1)) * g


class PowTensor(Function):
    """``a ** b`` where both are tensors. Requires ``a > 0`` to differentiate in ``b``."""

    op = "pow_tensor"
    backward_rule = "d(a^b)/da = b*a^(b-1)*g ;  d(a^b)/db = a^b*ln(a)*g"

    @staticmethod
    def forward(ctx, a, b):
        if np.any(a < 0):
            raise DomainError(
                "a ** b with a tensor exponent requires a >= 0 to stay real-valued "
                f"(min a was {float(np.min(a)):.6g})."
            )
        out = a**b
        ctx.save(a=a, b=b, out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        a, b, out = ctx.a, ctx.b, ctx.out
        grad_a = b * (a ** (b - 1)) * g
        if np.any(a == 0):
            raise DomainError(
                "d(a^b)/db = a^b*ln(a) is undefined at a = 0. Keep the base strictly "
                "positive when differentiating with respect to the exponent."
            )
        grad_b = out * np.log(a) * g
        return grad_a, grad_b


class Abs(Function):
    """``|a|``. At ``a = 0`` the subgradient is taken as 0 (documented, not silent)."""

    op = "abs"
    backward_rule = "d|a|/da = sign(a)*g   (subgradient 0 at a = 0)"

    @staticmethod
    def forward(ctx, a):
        ctx.save(a=a)
        return np.abs(a)

    @staticmethod
    def backward(ctx, g):
        return np.sign(ctx.a) * g
