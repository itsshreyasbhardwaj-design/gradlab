"""``gradlab.inspection`` — the stable interface between the engine and a UI.

This package is the **only** sanctioned way for a user interface to read the
computation graph. It exposes what the engine actually did, serialised in the
documented schema of :mod:`gradlab.inspection.schema`, so the Gradient
Microscope never reaches into private attributes and never recomputes a value
the engine already produced.

    from gradlab.inspection import trace_expression, graph_summary

    trace = trace_expression(lambda x: x * x + 2 * x, 3.0, names=["x"])
    print(trace["output"]["value"]["values"])      # [15.0]
    print(trace["leaf_gradients"][0]["gradient"])  # 8.0
"""

from __future__ import annotations

from gradlab.inspection.graph_export import (
    TensorRegistry,
    describe_node,
    describe_tensor,
    export_graph,
    graph_summary,
)
from gradlab.inspection.schema import SCHEMA_VERSION, describe_array
from gradlab.inspection.tracer import (
    ExecutionRecorder,
    decision_surface,
    trace_expression,
    trace_training_run,
    trace_training_step,
)

__all__ = [
    "SCHEMA_VERSION",
    "ExecutionRecorder",
    "TensorRegistry",
    "decision_surface",
    "describe_array",
    "describe_node",
    "describe_tensor",
    "export_graph",
    "graph_summary",
    "trace_expression",
    "trace_training_run",
    "trace_training_step",
]
