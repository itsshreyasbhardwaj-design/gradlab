"""Numerical gradient checking by central finite differences.

This is GradLab's first line of defence against a wrong derivative rule, and it
is independent of PyTorch: it compares the analytic gradient the engine produced
against a purely numerical estimate built from forward evaluations only.

For each scalar input element :math:`x_i`:

.. math::
    \\frac{\\partial f}{\\partial x_i} \\approx
    \\frac{f(x_i + \\epsilon) - f(x_i - \\epsilon)}{2\\epsilon}

The *central* difference is used rather than the forward difference because its
truncation error is :math:`O(\\epsilon^2)` instead of :math:`O(\\epsilon)`, which
buys roughly three extra digits of agreement for the same step size.

Choosing ``eps`` is a real trade-off and the reason tolerances here are not
arbitrary. Truncation error falls as :math:`\\epsilon^2`; floating-point
cancellation error in ``f(x+eps) - f(x-eps)`` grows as :math:`\\epsilon^{-1}`.
For float64 the sum of the two is minimised near
:math:`\\epsilon \\approx \\sqrt[3]{\\text{machine }\\epsilon} \\approx 6\\times10^{-6}`,
which is why the default is ``1e-6`` and the default tolerances are around
``1e-7`` absolute. A check that needs looser tolerances than these is telling
you something, and ``docs/validation.md`` records the measured errors rather
than only asserting that they passed.

**Finite differences are not valid everywhere.** At a kink (``relu`` at 0,
``abs`` at 0, a tie in ``max``) the two one-sided limits differ and the central
difference returns their average, which no correct subgradient need match. Keep
test inputs away from kinks; :func:`gradcheck` warns when it detects one.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from gradlab._config import no_grad
from gradlab.tensor import Tensor


@dataclass
class GradCheckResult:
    """The measured outcome of a gradient check — numbers, not just a verdict."""

    passed: bool
    max_abs_error: float
    max_rel_error: float
    per_input: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"gradcheck {'PASSED' if self.passed else 'FAILED'}: "
            f"max|analytic - numeric| = {self.max_abs_error:.3e}, "
            f"max relative = {self.max_rel_error:.3e}"
        ]
        for entry in self.per_input:
            lines.append(
                f"  input[{entry['index']}] shape={entry['shape']} "
                f"abs={entry['max_abs_error']:.3e} rel={entry['max_rel_error']:.3e}"
            )
            if entry.get("worst") is not None:
                i, analytic, numeric = entry["worst"]
                lines.append(f"    worst at {i}: analytic={analytic:.12g} numeric={numeric:.12g}")
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def numerical_gradient(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
) -> list[np.ndarray]:
    """Central-difference estimate of ``d fn / d input`` for every input.

    ``fn`` must map the given tensors to a **scalar** tensor. Perturbation
    happens under ``no_grad`` and the original values are always restored, so
    this function has no side effects on the inputs beyond their version
    counters.
    """
    grads: list[np.ndarray] = []
    with no_grad():
        for tensor in inputs:
            grad = np.zeros(tensor.shape, dtype=np.float64)
            flat = tensor.data.reshape(-1)
            for k in range(flat.size):
                original = float(flat[k])

                flat[k] = original + eps
                tensor._bump_version()
                plus = float(fn(*inputs).item())

                flat[k] = original - eps
                tensor._bump_version()
                minus = float(fn(*inputs).item())

                flat[k] = original
                tensor._bump_version()

                grad.reshape(-1)[k] = (plus - minus) / (2.0 * eps)
            grads.append(grad)
    return grads


def _kink_warning(fn: Callable[..., Tensor], inputs: Sequence[Tensor], eps: float) -> list[str]:
    """Detect inputs sitting on or very near a non-differentiable point."""
    notes = []
    for i, tensor in enumerate(inputs):
        near_zero = np.abs(tensor.data) < 10 * eps
        if np.any(near_zero):
            notes.append(
                f"input[{i}] has {int(near_zero.sum())} element(s) within 10*eps of 0; "
                "if the function has a kink there (relu, abs, max ties) the finite "
                "difference is an average of two different one-sided slopes and need "
                "not match any valid subgradient."
            )
    return notes


def gradcheck(
    fn: Callable[..., Tensor],
    *inputs: Tensor,
    eps: float = 1e-6,
    atol: float = 1e-7,
    rtol: float = 1e-5,
    raise_exception: bool = True,
    check_kinks: bool = True,
) -> GradCheckResult:
    """Compare GradLab's analytic gradient against a finite-difference estimate.

    Parameters
    ----------
    fn
        Maps the inputs to a scalar ``Tensor``.
    inputs
        Leaf tensors with ``requires_grad=True``.
    eps, atol, rtol
        Step size and tolerances. See the module docstring for why the defaults
        are what they are.
    raise_exception
        Raise ``AssertionError`` with the full report on failure (default), or
        return the result for the caller to inspect.

    Returns
    -------
    GradCheckResult
        Carries the measured errors, not just a boolean.

    >>> import gradlab as gl
    >>> x = gl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> res = gl.gradcheck(lambda t: (t * t).sum(), x)
    >>> res.passed
    True
    """
    diff_inputs = [t for t in inputs if t.requires_grad]
    if not diff_inputs:
        raise ValueError("gradcheck needs at least one input with requires_grad=True.")
    for t in diff_inputs:
        if not t.is_leaf:
            raise ValueError("gradcheck inputs must be leaf tensors.")
        if t.dtype != np.float64:
            warnings.warn(
                f"gradcheck on a {t.dtype} tensor: finite differences need float64 to "
                "reach the stated tolerances. Expect to loosen atol/rtol substantially.",
                UserWarning,
                stacklevel=2,
            )

    for t in diff_inputs:
        t.zero_grad()
    out = fn(*inputs)
    if out.size != 1:
        raise ValueError(
            f"gradcheck needs a scalar output; fn returned shape {out.shape}. "
            "Reduce it (e.g. .sum()) inside fn."
        )
    out.backward()
    analytic = [
        (t.grad.data.copy() if t.grad is not None else np.zeros(t.shape)) for t in diff_inputs
    ]

    numeric = numerical_gradient(fn, diff_inputs, eps=eps)

    notes = _kink_warning(fn, diff_inputs, eps) if check_kinks else []
    per_input = []
    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for i, (a, n, t) in enumerate(zip(analytic, numeric, diff_inputs, strict=True)):
        abs_err = np.abs(a - n)
        scale = np.maximum(np.abs(a), np.abs(n))
        rel_err = np.where(scale > 0, abs_err / np.maximum(scale, 1e-300), 0.0)
        ok = np.allclose(a, n, atol=atol, rtol=rtol)
        passed = passed and bool(ok)
        worst_idx = (
            np.unravel_index(int(np.argmax(abs_err)), abs_err.shape) if abs_err.size else None
        )
        worst_abs = float(abs_err.max()) if abs_err.size else 0.0
        worst_rel = float(rel_err.max()) if rel_err.size else 0.0
        per_input.append(
            {
                "index": i,
                "shape": tuple(t.shape),
                "max_abs_error": worst_abs,
                "max_rel_error": worst_rel,
                "worst": (
                    (worst_idx, float(a[worst_idx]), float(n[worst_idx]))
                    if worst_idx is not None
                    else None
                ),
            }
        )
        max_abs = max(max_abs, worst_abs)
        max_rel = max(max_rel, worst_rel)

    result = GradCheckResult(
        passed=passed,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        per_input=per_input,
        warnings=notes,
    )
    if not passed and raise_exception:
        raise AssertionError(result.report())
    return result
