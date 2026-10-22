"""The trace format: a stable, JSON-serialisable view of a real execution.

This schema is the contract between the engine and any user interface. The
Gradient Microscope reads nothing else — no private attributes, no re-derived
values — so a change here is a deliberate, versioned API change.

Every number in a trace was produced by the engine during the run that created
the trace. Nothing is synthesised, interpolated, or illustrative.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Bumped on any incompatible change to the trace structure. The web app checks it.
SCHEMA_VERSION = "1.0"

#: Values larger than this are truncated in traces, to keep a trace file loadable
#: in a browser. Truncation is always reported in the payload, never silent.
MAX_ELEMENTS = 128

#: Readable algebraic templates, used to render a forward step as mathematics
#: rather than as a function call. Falls back to ``op(a, b)`` for anything absent.
EXPR_TEMPLATES = {
    "add": "{0} + {1}",
    "sub": "{0} - {1}",
    "mul": "{0} · {1}",
    "div": "{0} / {1}",
    "neg": "-{0}",
    "pow": "{0}^{exponent}",
    "pow_tensor": "{0}^{1}",
    "abs": "|{0}|",
    "matmul": "{0} @ {1}",
    "exp": "e^{0}",
    "log": "ln({0})",
    "sqrt": "sqrt({0})",
    "tanh": "tanh({0})",
    "sigmoid": "sigma({0})",
    "relu": "max({0}, 0)",
    "sin": "sin({0})",
    "cos": "cos({0})",
    "sum": "sum({0})",
    "mean": "mean({0})",
    "max": "max({0})",
    "min": "min({0})",
    "identity": "{0}",
    "reshape": "reshape({0})",
    "transpose": "{0}^T",
    "index": "{0}[...]",
    "log_softmax": "log softmax({0})",
    "softmax": "softmax({0})",
    "bce_with_logits": "BCE({0}, {1})",
}


def describe_array(array: Any, max_elements: int = MAX_ELEMENTS) -> dict[str, Any]:
    """Summarise an array for display: shape, dtype, values (possibly truncated), stats.

    Truncation is explicit: when ``truncated`` is true the ``values`` field holds
    only the first ``max_elements`` entries in flat order, and the statistics are
    still computed over the *whole* array so they remain honest.
    """
    if array is None:
        return {"present": False}
    arr = np.asarray(array)
    flat = arr.reshape(-1)
    truncated = flat.size > max_elements
    payload: dict[str, Any] = {
        "present": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "truncated": bool(truncated),
        "values": [_finite(v) for v in flat[:max_elements].tolist()],
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        with np.errstate(all="ignore"):
            payload["stats"] = {
                "min": _finite(float(np.min(arr))),
                "max": _finite(float(np.max(arr))),
                "mean": _finite(float(np.mean(arr))),
                "std": _finite(float(np.std(arr))),
                "l2_norm": _finite(float(np.sqrt(np.sum(arr.astype(np.float64) ** 2)))),
                "num_nonfinite": int(np.sum(~np.isfinite(arr)))
                if np.issubdtype(arr.dtype, np.floating)
                else 0,
            }
    return payload


def _finite(value: float) -> Any:
    """JSON has no NaN or Infinity; represent them as strings so nothing is lost."""
    if isinstance(value, (int, bool)):
        return value
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return float(value)


def render_expression(op: str, operand_names: list[str], meta: dict[str, Any]) -> str:
    """Render one operation as readable mathematics."""
    template = EXPR_TEMPLATES.get(op)
    if template is None:
        return f"{op}({', '.join(operand_names)})"
    try:
        return template.format(*operand_names, **meta)
    except (IndexError, KeyError):
        return f"{op}({', '.join(operand_names)})"
