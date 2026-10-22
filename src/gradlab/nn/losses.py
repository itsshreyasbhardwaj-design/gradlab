"""Loss functions, in numerically stable form.

Two of the three losses here exist specifically *because* the naive formulation
is unstable:

* ``binary_cross_entropy_with_logits`` takes logits rather than probabilities so
  it never computes ``log(sigmoid(x))``, which underflows to ``-inf`` for
  moderately negative ``x``.
* ``cross_entropy_with_logits`` goes through log-sum-exp rather than forming
  ``log(softmax(x))``.

Each function documents its reduction and its label convention, and raises when
given shapes that do not match — a silent broadcast between predictions and
targets is one of the easiest ways to train a model against the wrong objective.
"""

from __future__ import annotations

import numpy as np

from gradlab._errors import ShapeError
from gradlab.autograd.function import Function
from gradlab.nn.module import Module
from gradlab.ops.activation import LogSoftmax, _stable_sigmoid
from gradlab.tensor import Tensor, as_tensor

_REDUCTIONS = ("mean", "sum", "none")


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction must be one of {_REDUCTIONS}; got {reduction!r}.")


def _check_same_shape(pred: Tensor, target: Tensor, name: str) -> None:
    if pred.shape != target.shape:
        raise ShapeError(
            f"{name}: predictions have shape {pred.shape} but targets have "
            f"{target.shape}. GradLab refuses to broadcast here, because a silent "
            "broadcast would compute a different objective than you intended "
            "(a common cause of a model that trains but never converges)."
        )


def mse_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean squared error, ``(pred - target)^2``.

    >>> import gradlab as gl
    >>> from gradlab import nn
    >>> nn.mse_loss(gl.Tensor([1.0, 2.0]), gl.Tensor([1.0, 4.0])).item()
    2.0
    """
    pred, target = as_tensor(pred), as_tensor(target)
    _check_same_shape(pred, target, "mse_loss")
    diff = pred - target
    return _reduce(diff * diff, reduction)


class _BCEWithLogits(Function):
    """Elementwise binary cross-entropy from logits, computed stably.

    The textbook form ``-[y·log σ(x) + (1-y)·log(1-σ(x))]`` evaluates a log of a
    quantity that underflows to exactly 0 once ``|x|`` is around 40, giving
    ``inf`` and then ``nan`` gradients. Folding ``σ`` into the log algebraically
    gives an equivalent expression whose only exponential has a non-positive
    argument:

    .. math:: L = \\max(x, 0) - x y + \\log(1 + e^{-|x|})

    The gradient collapses to something remarkably simple — the prediction error:

    .. math:: \\frac{\\partial L}{\\partial x} = \\sigma(x) - y
    """

    op = "bce_with_logits"
    backward_rule = "dL/dx = sigmoid(x) - y   (and dL/dy = -x)"

    @staticmethod
    def forward(ctx, x, y):
        loss = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
        ctx.save(x=x, y=y)
        return loss

    @staticmethod
    def backward(ctx, g):
        return (_stable_sigmoid(ctx.x) - ctx.y) * g, -ctx.x * g


def binary_cross_entropy_with_logits(
    logits: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    """Binary cross-entropy on **logits** (pre-sigmoid scores).

    ``target`` holds probabilities in ``[0, 1]`` — usually 0.0/1.0 labels — and
    must have the same shape as ``logits``. Do not pass probabilities as
    ``logits``; the whole point of this function is that it applies the sigmoid
    internally where it can do so stably.
    """
    logits, target = as_tensor(logits), as_tensor(target)
    _check_same_shape(logits, target, "binary_cross_entropy_with_logits")
    if np.any((target.data < 0.0) | (target.data > 1.0)):
        raise ValueError(
            "binary_cross_entropy_with_logits targets must lie in [0, 1]; got values "
            f"in [{float(target.data.min()):.4g}, {float(target.data.max()):.4g}]."
        )
    return _reduce(_BCEWithLogits.apply(logits, target), reduction)


def cross_entropy_with_logits(logits: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Multiclass cross-entropy on logits of shape ``(..., num_classes)``.

    ``target`` may be either:

    * **integer class indices** of shape ``logits.shape[:-1]`` — the usual case; or
    * **a probability distribution** with the same shape as ``logits`` (one-hot
      labels or soft labels).

    Which one you passed is inferred from the shape, and the inference is
    reported in the error message if neither fits. The computation is
    ``-log_softmax(logits)`` gathered (or dotted) with the target, so it is built
    from ordinary GradLab operations and is fully visible in the Gradient
    Microscope.
    """
    logits = as_tensor(logits)
    if logits.ndim < 1:
        raise ShapeError("cross_entropy_with_logits needs a class axis (the last axis).")
    target = target if isinstance(target, Tensor) else as_tensor(target)
    log_probs = LogSoftmax.apply(logits, axis=-1)

    if target.shape == logits.shape:
        # Soft / one-hot labels.
        picked = -(target * log_probs).sum(axis=-1)
    elif target.shape == logits.shape[:-1]:
        # Integer class indices.
        labels = target.data
        if not np.issubdtype(labels.dtype, np.integer):
            if not np.all(labels == np.round(labels)):
                raise ShapeError(
                    "cross_entropy_with_logits got float targets of shape "
                    f"{target.shape} that are not whole numbers, so they are neither "
                    "class indices nor a distribution over "
                    f"{logits.shape[-1]} classes."
                )
            labels = labels.astype(np.intp)
        num_classes = logits.shape[-1]
        if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
            raise ShapeError(
                f"Class index out of range: labels span [{int(labels.min())}, "
                f"{int(labels.max())}] but there are {num_classes} classes."
            )
        flat = log_probs.reshape(-1, num_classes)
        rows = np.arange(flat.shape[0], dtype=np.intp)
        picked = -flat[rows, labels.reshape(-1)]
        if target.ndim > 1:
            picked = picked.reshape(target.shape)
    else:
        raise ShapeError(
            f"cross_entropy_with_logits: logits have shape {logits.shape}, so targets "
            f"must be either class indices of shape {logits.shape[:-1]} or a "
            f"distribution of shape {logits.shape}. Got {target.shape}."
        )
    return _reduce(picked, reduction)


def accuracy(logits: Tensor, target: Tensor) -> float:
    """Fraction of correct predictions. Not differentiable — a reporting metric.

    Accepts binary logits of shape ``(N,)`` or ``(N, 1)`` (thresholded at 0), or
    multiclass logits of shape ``(N, C)`` (argmax).
    """
    logits, target = as_tensor(logits), as_tensor(target)
    scores = logits.data
    labels = target.data
    if scores.ndim >= 2 and scores.shape[-1] > 1:
        predicted = scores.argmax(axis=-1)
        truth = labels.argmax(axis=-1) if labels.shape == scores.shape else labels
    else:
        predicted = (scores.reshape(-1) > 0.0).astype(np.intp)
        truth = labels.reshape(-1)
    predicted, truth = predicted.reshape(-1), np.asarray(truth).reshape(-1)
    if predicted.shape != truth.shape:
        raise ShapeError(f"accuracy: {predicted.size} predictions vs {truth.size} targets.")
    return float((predicted == truth.astype(predicted.dtype)).mean())


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return mse_loss(pred, target, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(logits, target, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class CrossEntropyLoss(Module):
    """Multiclass cross-entropy on logits. Takes class indices or distributions."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return cross_entropy_with_logits(logits, target, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"
