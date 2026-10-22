"""Transcendental functions and activations, with numerically careful forms.

``sigmoid`` is evaluated piecewise so that a large-magnitude input cannot
overflow ``exp``. ``log`` and ``sqrt`` validate their domain and raise rather
than emitting ``nan``.
"""

from __future__ import annotations

import numpy as np

from gradlab._errors import DomainError
from gradlab.autograd.function import Function


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """``1/(1+e^-x)`` without overflow.

    For ``x >= 0`` use ``1/(1+e^-x)``; for ``x < 0`` use ``e^x/(1+e^x)``. Both
    forms only ever exponentiate a non-positive number, so ``exp`` cannot
    overflow for any finite input.
    """
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class Exp(Function):
    op = "exp"
    backward_rule = "d(e^a)/da = e^a * g = out * g"

    @staticmethod
    def forward(ctx, a):
        out = np.exp(a)
        ctx.save(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        return ctx.out * g


class Log(Function):
    op = "log"
    backward_rule = "d(ln a)/da = g / a"

    @staticmethod
    def forward(ctx, a):
        if np.any(a <= 0):
            raise DomainError(
                "log() requires strictly positive input "
                f"(minimum was {float(np.min(a)):.6g}). GradLab raises instead of "
                "returning -inf/nan. For log-probabilities use a numerically stable "
                "loss such as nn.cross_entropy_with_logits, which never forms log(p)."
            )
        ctx.save(a=a)
        return np.log(a)

    @staticmethod
    def backward(ctx, g):
        return g / ctx.a


class Sqrt(Function):
    op = "sqrt"
    backward_rule = "d(sqrt a)/da = g / (2*sqrt a)"

    @staticmethod
    def forward(ctx, a):
        if np.any(a < 0):
            raise DomainError(f"sqrt() requires a >= 0 (minimum was {float(np.min(a)):.6g}).")
        out = np.sqrt(a)
        ctx.save(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        if np.any(ctx.out == 0):
            raise DomainError(
                "The derivative of sqrt is unbounded at 0; GradLab will not return inf. "
                "Keep inputs strictly positive, or add an epsilon."
            )
        return g / (2.0 * ctx.out)


class Tanh(Function):
    op = "tanh"
    backward_rule = "d(tanh a)/da = (1 - tanh(a)^2) * g = (1 - out^2) * g"

    @staticmethod
    def forward(ctx, a):
        out = np.tanh(a)
        ctx.save(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        return (1.0 - ctx.out * ctx.out) * g


class Sigmoid(Function):
    op = "sigmoid"
    backward_rule = "d(sigma a)/da = sigma(a)*(1-sigma(a))*g = out*(1-out)*g"

    @staticmethod
    def forward(ctx, a):
        out = _stable_sigmoid(a)
        ctx.save(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        return ctx.out * (1.0 - ctx.out) * g


class ReLU(Function):
    """``max(a, 0)``. The subgradient at exactly 0 is taken as 0."""

    op = "relu"
    backward_rule = "d(relu a)/da = g if a > 0 else 0   (subgradient 0 at a = 0)"

    @staticmethod
    def forward(ctx, a):
        ctx.save(mask=a > 0)
        return np.maximum(a, 0.0)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.mask


class Sin(Function):
    op = "sin"
    backward_rule = "d(sin a)/da = cos(a) * g"

    @staticmethod
    def forward(ctx, a):
        ctx.save(a=a)
        return np.sin(a)

    @staticmethod
    def backward(ctx, g):
        return np.cos(ctx.a) * g


class Cos(Function):
    op = "cos"
    backward_rule = "d(cos a)/da = -sin(a) * g"

    @staticmethod
    def forward(ctx, a):
        ctx.save(a=a)
        return np.cos(a)

    @staticmethod
    def backward(ctx, g):
        return -np.sin(ctx.a) * g


class LogSoftmax(Function):
    """``log(softmax(x))`` computed as ``x - logsumexp(x)``.

    The naive route — exponentiate, normalise, then take the log — overflows for
    large logits and loses all precision for very negative ones. Subtracting the
    row maximum first makes every exponential at most ``e^0 = 1``, and the
    identity ``log(softmax(x))_i = x_i - (m + log(sum_j e^(x_j - m)))`` means the
    log is never applied to a small probability at all.

    The backward rule is unusually clean because of that algebra:
    ``dL/dx = g - softmax(x) * sum(g)``, computed along the softmax axis.
    """

    op = "log_softmax"
    backward_rule = "dL/dx = g - softmax(x) * sum(g, axis)"
    auto_unbroadcast = False

    @staticmethod
    def forward(ctx, a, *, axis=-1):
        shifted = a - np.max(a, axis=axis, keepdims=True)
        log_denom = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        out = shifted - log_denom
        ctx.save(out=out, axis=axis)
        ctx.set_meta(axis=int(axis))
        return out

    @staticmethod
    def backward(ctx, g):
        softmax = np.exp(ctx.out)
        return g - softmax * np.sum(g, axis=ctx.axis, keepdims=True)


class Softmax(Function):
    """``softmax(x)`` with the max-subtraction shift; Jacobian applied implicitly."""

    op = "softmax"
    backward_rule = "dL/dx = p * (g - sum(p*g, axis))   where p = softmax(x)"
    auto_unbroadcast = False

    @staticmethod
    def forward(ctx, a, *, axis=-1):
        shifted = np.exp(a - np.max(a, axis=axis, keepdims=True))
        out = shifted / np.sum(shifted, axis=axis, keepdims=True)
        ctx.save(out=out, axis=axis)
        ctx.set_meta(axis=int(axis))
        return out

    @staticmethod
    def backward(ctx, g):
        p = ctx.out
        return p * (g - np.sum(p * g, axis=ctx.axis, keepdims=True))
