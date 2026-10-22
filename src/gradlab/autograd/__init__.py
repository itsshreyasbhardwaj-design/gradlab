"""GradLab's automatic differentiation engine.

Public surface:

* :class:`Function` — base class for a differentiable operation.
* :class:`Node`, :class:`Context` — the computation graph.
* :func:`topological_order` — the ordering the engine walks.
* :func:`gradcheck`, :func:`numerical_gradient` — independent validation.
* :func:`unbroadcast` — broadcast-gradient reduction, used by every rule.
"""

from __future__ import annotations

from gradlab.autograd.engine import backward, topological_order
from gradlab.autograd.function import Function, unbroadcast
from gradlab.autograd.gradcheck import GradCheckResult, gradcheck, numerical_gradient
from gradlab.autograd.graph import Context, Node

__all__ = [
    "Context",
    "Function",
    "GradCheckResult",
    "Node",
    "backward",
    "gradcheck",
    "numerical_gradient",
    "topological_order",
    "unbroadcast",
]
