"""Export a real computation graph as the documented trace schema.

The exporter walks the graph through ``Node``'s public attributes and the
engine's own :func:`~gradlab.autograd.engine.topological_order`. It never
recomputes a forward value or a derivative — it reports what the engine stored.
"""

from __future__ import annotations

from typing import Any

from gradlab.autograd.engine import topological_order
from gradlab.autograd.graph import Node
from gradlab.inspection.schema import (
    MAX_ELEMENTS,
    SCHEMA_VERSION,
    describe_array,
    render_expression,
)
from gradlab.tensor import Tensor


class TensorRegistry:
    """Assigns a stable display id and name to every tensor in one trace.

    Leaves are named from a caller-supplied mapping when available (``x``, ``W1``,
    ``bias``), which is what makes the Microscope's forward view readable rather
    than a wall of ``t17``.
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        self._names = dict(names or {})
        self._ids: dict[int, str] = {}
        self._leaf_count = 0

    def id_for(self, tensor: Tensor) -> str:
        key = id(tensor)
        if key in self._ids:
            return self._ids[key]
        producer = tensor.grad_fn
        if producer is None:
            name = self._names.get(key)
            if name:
                ident = name
            elif not tensor.requires_grad and tensor.size == 1:
                # An unnamed, non-differentiable scalar is a literal constant from
                # the expression. Showing its value is far more use than "leaf3".
                value = tensor.data.reshape(()).item()
                ident = f"{value:g}"
            else:
                ident = f"leaf{self._leaf_count}"
            self._leaf_count += 1
        else:
            ident = f"t{producer.seq}"
        self._ids[key] = ident
        return ident

    def name_for(self, tensor: Tensor) -> str:
        return self._names.get(id(tensor), self.id_for(tensor))


def describe_tensor(
    tensor: Tensor,
    registry: TensorRegistry,
    include_values: bool = True,
    max_elements: int = MAX_ELEMENTS,
) -> dict[str, Any]:
    """One tensor, as the trace schema describes it."""
    producer = tensor.grad_fn
    payload: dict[str, Any] = {
        "id": registry.id_for(tensor),
        "name": registry.name_for(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "requires_grad": tensor.requires_grad,
        "is_leaf": tensor.is_leaf,
        "is_parameter": type(tensor).__name__ == "Parameter",
        "producer": None if producer is None else f"node{producer.seq}",
    }
    if include_values:
        payload["value"] = describe_array(tensor.data, max_elements)
    return payload


def describe_node(
    node: Node,
    registry: TensorRegistry,
    include_values: bool = True,
    max_elements: int = MAX_ELEMENTS,
) -> dict[str, Any]:
    """One graph node, with its operands, output, saved metadata and rule."""
    operand_names = [registry.name_for(t) for t in node.inputs]
    output = node.output()
    return {
        "id": f"node{node.seq}",
        "op": node.op,
        "seq": node.seq,
        "backward_rule": node.backward_rule,
        "expression": render_expression(node.op, operand_names, node.ctx.meta),
        "inputs": [describe_tensor(t, registry, include_values, max_elements) for t in node.inputs],
        "output": {
            "id": f"t{node.seq}",
            "shape": list(node.output_shape),
            "dtype": str(node.output_dtype),
            "value": describe_array(output.data, max_elements)
            if (include_values and output is not None)
            else {"present": False, "reason": "intermediate tensor already freed"},
        },
        "meta": dict(node.ctx.meta),
        "released": node.released,
        "num_broadcast_inputs": sum(
            1 for t in node.inputs if tuple(t.shape) != tuple(node.output_shape)
        ),
    }


def export_graph(
    root: Tensor,
    names: dict[int, str] | None = None,
    include_values: bool = True,
    max_elements: int = MAX_ELEMENTS,
) -> dict[str, Any]:
    """Export the graph behind ``root`` in the trace schema.

    Call this **before** ``backward()`` if you want intermediate forward values:
    a backward pass without ``retain_graph=True`` releases each node's saved
    state, and the exporter reports that honestly as
    ``"intermediate tensor already freed"`` rather than inventing a number.
    """
    registry = TensorRegistry(names)
    if root.grad_fn is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": [],
            "edges": [],
            "leaves": [describe_tensor(root, registry, include_values, max_elements)],
            "root": registry.id_for(root),
            "forward_steps": [],
            "note": "root is a leaf; no operations were recorded",
        }

    order = topological_order(root.grad_fn)
    forward_order = sorted(order, key=lambda n: n.seq)

    nodes = [describe_node(n, registry, include_values, max_elements) for n in forward_order]

    edges: list[dict[str, Any]] = []
    leaves: dict[str, dict[str, Any]] = {}
    for node in forward_order:
        for position, tensor in enumerate(node.inputs):
            source = (
                f"node{tensor.grad_fn.seq}"
                if tensor.grad_fn is not None
                else registry.id_for(tensor)
            )
            edges.append(
                {
                    "from": source,
                    "to": f"node{node.seq}",
                    "tensor": registry.id_for(tensor),
                    "operand": position,
                    "shape": list(tensor.shape),
                }
            )
            if tensor.grad_fn is None:
                leaves[registry.id_for(tensor)] = describe_tensor(
                    tensor, registry, include_values, max_elements
                )

    forward_steps = [
        {
            "step": i + 1,
            "node": f"node{node.seq}",
            "op": node.op,
            "expression": nodes[i]["expression"],
            "inputs": [
                {"id": registry.id_for(t), "name": registry.name_for(t), "shape": list(t.shape)}
                for t in node.inputs
            ],
            "output": {"id": f"t{node.seq}", "shape": list(node.output_shape)},
            "backward_rule": node.backward_rule,
        }
        for i, node in enumerate(forward_order)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
        "leaves": list(leaves.values()),
        "root": registry.id_for(root),
        "root_node": f"node{root.grad_fn.seq}",
        "num_nodes": len(nodes),
        "depth": _graph_depth(root.grad_fn),
        "forward_steps": forward_steps,
    }


def _graph_depth(root: Node) -> int:
    """Longest path from ``root`` down to a leaf, computed iteratively."""
    depth: dict[int, int] = {}
    for node in reversed(topological_order(root)):
        children = [n for n in node.next_nodes if n is not None]
        depth[id(node)] = 1 + max((depth.get(id(c), 0) for c in children), default=0)
    return depth.get(id(root), 0)


def graph_summary(root: Tensor) -> str:
    """A short text summary of the graph behind ``root``, for terminal use.

    >>> import gradlab as gl
    >>> from gradlab.inspection import graph_summary
    >>> x = gl.Tensor(2.0, requires_grad=True)
    >>> print(graph_summary(x * x + x))
    graph: 2 nodes, depth 2
      ops: add x1, mul x1
      leaves: 1
    """
    if root.grad_fn is None:
        return "graph: 0 nodes (root is a leaf)"
    order = topological_order(root.grad_fn)
    counts: dict[str, int] = {}
    leaves = set()
    for node in order:
        counts[node.op] = counts.get(node.op, 0) + 1
        for t in node.inputs:
            if t.grad_fn is None:
                leaves.add(id(t))
    ops = ", ".join(f"{op} x{n}" for op, n in sorted(counts.items()))
    return (
        f"graph: {len(order)} nodes, depth {_graph_depth(root.grad_fn)}\n"
        f"  ops: {ops}\n"
        f"  leaves: {len(leaves)}"
    )
