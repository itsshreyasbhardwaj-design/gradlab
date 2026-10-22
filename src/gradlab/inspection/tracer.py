"""Record a real forward and backward pass as an inspectable trace.

The recorder is passed into :func:`gradlab.autograd.engine.backward` as an
observer, so the Microscope's backward view is a transcript of the engine's
actual traversal — same order, same intermediate gradients — rather than a
second implementation that could drift.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

import gradlab
from gradlab._config import no_grad
from gradlab.autograd.engine import backward as engine_backward
from gradlab.inspection.graph_export import TensorRegistry, export_graph
from gradlab.inspection.schema import MAX_ELEMENTS, SCHEMA_VERSION, describe_array
from gradlab.tensor import Tensor

#: Ops whose derivative is elementwise in a single input, so a per-element local
#: derivative can be recovered from the recorded gradients.
_ELEMENTWISE_UNARY = {
    "exp",
    "log",
    "sqrt",
    "tanh",
    "sigmoid",
    "relu",
    "sin",
    "cos",
    "neg",
    "abs",
    "pow",
}


class ExecutionRecorder:
    """Observer that turns an engine backward pass into trace steps."""

    def __init__(
        self,
        registry: TensorRegistry,
        include_values: bool = True,
        max_elements: int = MAX_ELEMENTS,
    ) -> None:
        self.registry = registry
        self.include_values = include_values
        self.max_elements = max_elements
        self.steps: list[dict[str, Any]] = []
        self.seed: dict[str, Any] | None = None
        self.traversal_order: list[str] = []

    # -- engine callbacks --------------------------------------------------

    def on_seed(self, root: Tensor, seed: np.ndarray, order: list) -> None:
        self.seed = {
            "tensor": self.registry.id_for(root),
            "shape": list(np.shape(seed)),
            "value": describe_array(seed, self.max_elements),
            "explanation": (
                "The backward pass starts from dL/dL = 1. For a scalar loss that is "
                "the single number 1.0; for a non-scalar output you supplied it."
            ),
        }
        self.traversal_order = [f"node{n.seq}" for n in order]

    def on_backward_step(self, index: int, node, upstream: np.ndarray, input_grads: tuple) -> None:
        contributions = []
        for position, (tensor, grad) in enumerate(zip(node.inputs, input_grads, strict=True)):
            entry: dict[str, Any] = {
                "operand": position,
                "tensor": self.registry.id_for(tensor),
                "name": self.registry.name_for(tensor),
                "shape": list(tensor.shape),
                "is_leaf": tensor.is_leaf,
                "requires_grad": tensor.requires_grad,
                "received_gradient": grad is not None,
            }
            if grad is None:
                entry["why_none"] = (
                    "this input does not require grad, so no gradient is computed for it"
                )
            else:
                entry["gradient"] = describe_array(grad, self.max_elements)

                # Two different shape changes can happen between the upstream
                # gradient and an input's gradient, and conflating them would
                # teach the wrong lesson:
                #   * an elementwise op whose input was *broadcast* in the
                #     forward pass has its gradient SUMMED back down;
                #   * a reduction (sum/mean/max) has its gradient EXPANDED back
                #     out over the axes it collapsed.
                # The op's own auto_unbroadcast flag tells them apart.
                shape_changed = tuple(np.shape(grad)) != tuple(np.shape(upstream))
                broadcast_reduced = bool(
                    shape_changed
                    and node.fn.auto_unbroadcast
                    and tuple(tensor.shape) != tuple(node.output_shape)
                )
                entry["broadcast_reduced"] = broadcast_reduced
                entry["shape_changed"] = bool(shape_changed)
                if broadcast_reduced:
                    entry["broadcast_note"] = (
                        f"the forward pass broadcast this input from {list(tensor.shape)} to "
                        f"{list(node.output_shape)}, so its gradient was summed back down "
                        "over the stretched axes"
                    )
                elif shape_changed:
                    entry["shape_note"] = (
                        f"{node.op} rearranged shapes in the forward pass "
                        f"({list(tensor.shape)} -> {list(node.output_shape)}), so its rule "
                        f"maps the gradient back to {list(np.shape(grad))}"
                    )
                if (
                    node.op in _ELEMENTWISE_UNARY
                    and len(node.inputs) == 1
                    and np.shape(grad) == np.shape(upstream)
                ):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        local = np.where(
                            upstream != 0, grad / np.where(upstream == 0, 1, upstream), np.nan
                        )
                    entry["local_derivative"] = describe_array(local, self.max_elements)
                    entry["local_derivative_note"] = (
                        "recovered from the recorded gradients as "
                        "(downstream gradient) / (upstream gradient); the chain rule "
                        "multiplies this local slope by the upstream gradient"
                    )
            contributions.append(entry)

        self.steps.append(
            {
                "step": index + 1,
                "node": f"node{node.seq}",
                "op": node.op,
                "backward_rule": node.backward_rule,
                "upstream_gradient": describe_array(upstream, self.max_elements),
                "output_shape": list(node.output_shape),
                "contributions": contributions,
                "meta": dict(node.ctx.meta),
            }
        )


def _trace_envelope(kind: str, label: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "gradlab_version": gradlab.__version__,
        "kind": kind,
        "label": label,
    }


def trace_expression(
    fn: Callable[..., Tensor],
    *values: Any,
    names: list[str] | None = None,
    label: str | None = None,
    upstream: Any = None,
    include_values: bool = True,
    max_elements: int = MAX_ELEMENTS,
) -> dict[str, Any]:
    """Run ``fn`` on ``values`` and return a complete trace of the result.

    ``fn`` receives one ``Tensor`` per value and returns one tensor. If that
    output is not scalar, pass ``upstream`` explicitly.

    >>> from gradlab.inspection import trace_expression
    >>> trace = trace_expression(lambda x: x * x + 2 * x, 3.0, names=["x"])
    >>> trace["output"]["value"]["values"]
    [15.0]
    >>> trace["leaf_gradients"][0]["gradient"]["values"]
    [8.0]
    """
    labels = names or [chr(ord("x") + i) for i in range(len(values))]
    if len(labels) != len(values):
        raise ValueError(f"got {len(values)} values but {len(labels)} names")

    inputs = [v if isinstance(v, Tensor) else Tensor(v, requires_grad=True) for v in values]
    for t in inputs:
        if t.is_leaf and not t.requires_grad:
            t.requires_grad = True
        t.zero_grad()

    name_map = {id(t): n for t, n in zip(inputs, labels, strict=True)}
    registry = TensorRegistry(name_map)

    output = fn(*inputs)
    if not isinstance(output, Tensor):
        raise TypeError(f"fn must return a Tensor; got {type(output).__name__}")

    # Export the graph BEFORE backward so intermediate forward values are intact.
    graph = export_graph(
        output, names=name_map, include_values=include_values, max_elements=max_elements
    )

    trace = _trace_envelope("expression", label or _default_label(graph, labels))
    trace.update(graph)
    trace["inputs"] = [
        {
            "id": registry.id_for(t),
            "name": n,
            "shape": list(t.shape),
            "value": describe_array(t.data, max_elements),
        }
        for t, n in zip(inputs, labels, strict=True)
    ]
    trace["output"] = {
        "id": registry.id_for(output),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "value": describe_array(output.data, max_elements),
    }

    recorder = ExecutionRecorder(registry, include_values, max_elements)
    seed = None if upstream is None else Tensor(upstream)
    engine_backward(output, grad=seed, retain_graph=True, recorder=recorder)

    trace["seed"] = recorder.seed
    trace["traversal_order"] = recorder.traversal_order
    trace["backward_steps"] = recorder.steps
    trace["leaf_gradients"] = [
        {
            "id": registry.id_for(t),
            "name": n,
            "shape": list(t.shape),
            "gradient": describe_array(t.grad.data if t.grad is not None else None, max_elements),
        }
        for t, n in zip(inputs, labels, strict=True)
    ]
    return trace


def _default_label(graph: dict[str, Any], names: list[str]) -> str:
    if graph.get("nodes"):
        return graph["nodes"][-1]["expression"]
    return ", ".join(names)


def trace_training_step(
    model,
    optimizer,
    criterion: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    label: str = "training step",
    include_values: bool = True,
    max_elements: int = MAX_ELEMENTS,
) -> dict[str, Any]:
    """Trace one complete optimisation step: forward, backward, and the update.

    Records each parameter **before** and **after** ``optimizer.step()`` along
    with the gradient and the learning rate, so the Microscope's parameter-update
    view can show the arithmetic of the update rather than describe it. For Adam
    and AdamW the moment estimates and bias-correction denominators are captured
    too.

    The model is left in its updated state — this performs a real training step.
    """
    optimizer.zero_grad()
    name_map = {id(p): n for n, p in model.named_parameters()}
    name_map[id(x)] = "input"
    name_map[id(y)] = "target"
    registry = TensorRegistry(name_map)

    predictions = model(x)
    loss = criterion(predictions, y)

    graph = export_graph(
        loss, names=name_map, include_values=include_values, max_elements=max_elements
    )

    recorder = ExecutionRecorder(registry, include_values, max_elements)
    engine_backward(loss, retain_graph=True, recorder=recorder)

    before = {name: p.data.copy() for name, p in model.named_parameters()}
    grads = {
        name: (p.grad.data.copy() if p.grad is not None else None)
        for name, p in model.named_parameters()
    }
    lr = optimizer.lr
    optimizer_state_before = _optimizer_state_snapshot(optimizer, model)

    optimizer.step()

    after = {name: p.data.copy() for name, p in model.named_parameters()}
    optimizer_state_after = _optimizer_state_snapshot(optimizer, model)

    trace = _trace_envelope("training_step", label)
    trace.update(graph)
    trace["model"] = {
        "repr": repr(model),
        "num_parameters": model.num_parameters(),
        "layers": [
            {"name": n or "(root)", "type": type(m).__name__, "config": m.extra_repr()}
            for n, m in model.named_modules()
        ],
    }
    trace["loss"] = {
        "value": describe_array(loss.data, max_elements),
        "criterion": getattr(criterion, "__name__", type(criterion).__name__),
    }
    trace["predictions"] = describe_array(predictions.data, max_elements)
    trace["targets"] = describe_array(y.data, max_elements)
    trace["seed"] = recorder.seed
    trace["traversal_order"] = recorder.traversal_order
    trace["backward_steps"] = recorder.steps
    trace["optimizer"] = {
        "repr": repr(optimizer),
        "type": type(optimizer).__name__,
        "learning_rate": lr,
        "step_count": optimizer._step_count,
        "update_rule": _update_rule_text(optimizer),
    }
    trace["parameter_updates"] = [
        {
            "name": name,
            "shape": list(before[name].shape),
            "before": describe_array(before[name], max_elements),
            "gradient": describe_array(grads[name], max_elements),
            "after": describe_array(after[name], max_elements),
            "delta": describe_array(
                after[name] - before[name] if grads[name] is not None else None, max_elements
            ),
            "learning_rate": lr,
            "state_before": optimizer_state_before.get(name, {}),
            "state_after": optimizer_state_after.get(name, {}),
        }
        for name, _ in model.named_parameters()
    ]
    return trace


def _optimizer_state_snapshot(optimizer, model) -> dict[str, dict[str, Any]]:
    """Per-parameter optimiser state (momentum buffers, Adam moments, step count)."""
    snapshot: dict[str, dict[str, Any]] = {}
    for name, param in model.named_parameters():
        state = optimizer.state.get(id(param))
        if not state:
            continue
        entry: dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                entry[key] = describe_array(value)
            else:
                entry[key] = value
        if "t" in state and hasattr(optimizer, "beta1"):
            t = state["t"]
            entry["bias_correction"] = {
                "t": t,
                "one_minus_beta1_t": float(1.0 - optimizer.beta1**t),
                "one_minus_beta2_t": float(1.0 - optimizer.beta2**t),
                "note": (
                    "Both moving averages start at zero, so at small t they understate "
                    "the true moments. Dividing by these factors rescales each average "
                    f"by the weight it has actually accumulated; at t={t} the first "
                    f"moment is scaled up by {1.0 / (1.0 - optimizer.beta1**t):.4f}x."
                ),
            }
        snapshot[name] = entry
    return snapshot


def _update_rule_text(optimizer) -> str:
    kind = type(optimizer).__name__
    if kind == "SGD":
        if optimizer.momentum:
            return "v <- mu*v + g ;  theta <- theta - lr*v"
        return "theta <- theta - lr*g"
    if kind == "Adam":
        return (
            "m <- b1*m + (1-b1)*g ;  v <- b2*v + (1-b2)*g^2 ;  "
            "theta <- theta - lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)"
        )
    if kind == "AdamW":
        return (
            "m <- b1*m + (1-b1)*g ;  v <- b2*v + (1-b2)*g^2 ;  "
            "theta <- theta - lr*(m/(1-b1^t))/(sqrt(v/(1-b2^t))+eps) - lr*wd*theta"
        )
    return "see the optimiser's documentation"


def trace_training_run(history, label: str = "training run") -> dict[str, Any]:
    """Wrap a measured :class:`~gradlab.training.history.History` as a trace.

    Pure repackaging — every value comes from the run that produced the history.
    """
    trace = _trace_envelope("training_run", label)
    trace.update(history.to_dict())
    return trace


def decision_surface(model, x_range, y_range, resolution: int = 60) -> dict[str, Any]:
    """Evaluate a 2-D classifier on a grid, for the Microscope's boundary view.

    Runs under ``no_grad`` in eval mode. Returns the raw scores so the UI can
    draw a contour rather than only a hard class assignment.
    """
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    was_training = model.training
    model.eval()
    try:
        with no_grad():
            scores = model(Tensor(grid)).data
    finally:
        model.train(was_training)
    scores = scores.reshape(resolution, resolution, -1)
    return {
        "x": xs.tolist(),
        "y": ys.tolist(),
        "resolution": resolution,
        "channels": int(scores.shape[-1]),
        "scores": scores.squeeze(-1).tolist() if scores.shape[-1] == 1 else scores.tolist(),
    }
