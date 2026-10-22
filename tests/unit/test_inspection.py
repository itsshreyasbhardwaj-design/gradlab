"""The inspection API, and the claim that the Microscope shows the real engine.

The central test here is :meth:`TestConsistency.test_trace_matches_a_plain_backward_pass`
— the trace's numbers must equal what an ordinary ``backward()`` produces. If
that ever fails, the Microscope is showing something the engine did not do.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import gradlab as gl
from gradlab import nn, optim
from gradlab.inspection import (
    SCHEMA_VERSION,
    export_graph,
    graph_summary,
    trace_expression,
    trace_training_step,
)
from gradlab.inspection.expressions import ExpressionError, evaluate, parse_expression


class TestGraphExport:
    def test_exports_every_node_and_edge(self):
        x = gl.Tensor(3.0, requires_grad=True)
        graph = export_graph(x * x + 2 * x, names={id(x): "x"})
        assert graph["num_nodes"] == 3
        assert {n["op"] for n in graph["nodes"]} == {"mul", "add"}
        assert graph["depth"] == 2
        assert all({"from", "to"} <= set(e) for e in graph["edges"])

    def test_leaf_root_exports_cleanly(self):
        graph = export_graph(gl.Tensor(1.0, requires_grad=True))
        assert graph["nodes"] == [] and "note" in graph

    def test_values_are_reported_after_release_as_absent_not_invented(self):
        x = gl.Tensor(3.0, requires_grad=True)
        y = x * x
        y.backward()  # releases saved state
        graph = export_graph(y, names={id(x): "x"})
        value = graph["nodes"][0]["output"]["value"]
        assert value["present"] is True  # the output tensor itself is still alive
        assert graph["nodes"][0]["released"] is True

    def test_graph_summary_is_readable(self):
        x = gl.Tensor(2.0, requires_grad=True)
        text = graph_summary(x * x + x)
        assert "2 nodes" in text and "mul x1" in text and "add x1" in text

    def test_export_is_json_serialisable(self):
        x = gl.randn(3, 4, seed=0, requires_grad=True)
        graph = export_graph((x @ x.T).sum(), names={id(x): "x"})
        json.dumps(graph)  # must not raise

    def test_broadcast_inputs_are_flagged(self):
        W = gl.ones(4, 3, requires_grad=True)
        b = gl.zeros(3, requires_grad=True)
        graph = export_graph(W + b, names={id(W): "W", id(b): "b"})
        assert graph["nodes"][0]["num_broadcast_inputs"] == 1


class TestTraceExpression:
    def test_the_headline_example(self):
        trace = trace_expression(lambda x: x * x + 2 * x, 3.0, names=["x"])
        assert trace["output"]["value"]["values"] == [15.0]
        assert trace["leaf_gradients"][0]["gradient"]["values"] == [8.0]
        assert trace["schema_version"] == SCHEMA_VERSION

    def test_forward_steps_are_in_execution_order(self):
        trace = trace_expression(lambda x: (x * 2).tanh().exp(), 0.5, names=["x"])
        assert [s["op"] for s in trace["forward_steps"]] == ["mul", "tanh", "exp"]

    def test_backward_steps_are_in_reverse_order(self):
        trace = trace_expression(lambda x: (x * 2).tanh().exp(), 0.5, names=["x"])
        assert [s["op"] for s in trace["backward_steps"]] == ["exp", "tanh", "mul"]

    def test_seed_is_recorded_and_explained(self):
        trace = trace_expression(lambda x: x * x, 2.0, names=["x"])
        assert trace["seed"]["value"]["values"] == [1.0]
        assert "dL/dL = 1" in trace["seed"]["explanation"]

    def test_broadcast_reduction_is_labelled(self):
        trace = trace_expression(
            lambda W, b: (W + b).sum(), np.ones((4, 3)), np.zeros(3), names=["W", "b"]
        )
        add_step = next(s for s in trace["backward_steps"] if s["op"] == "add")
        bias = next(c for c in add_step["contributions"] if c["name"] == "b")
        assert bias["broadcast_reduced"] is True
        assert bias["gradient"]["values"] == [4.0, 4.0, 4.0]

    def test_reduction_is_not_mislabelled_as_broadcast(self):
        """sum() expands its gradient; that is a different thing from unbroadcasting."""
        trace = trace_expression(lambda x: x.sum(), np.ones(5), names=["x"])
        step = trace["backward_steps"][0]
        assert step["contributions"][0]["broadcast_reduced"] is False
        assert step["contributions"][0]["shape_changed"] is True

    def test_non_differentiable_input_is_explained(self):
        trace = trace_expression(lambda x: x * 2, 3.0, names=["x"])
        mul = trace["backward_steps"][0]
        constant = next(c for c in mul["contributions"] if not c["received_gradient"])
        assert "does not require grad" in constant["why_none"]

    def test_local_derivative_is_recovered_for_unary_elementwise_ops(self):
        trace = trace_expression(lambda x: x.tanh(), 0.5, names=["x"])
        step = trace["backward_steps"][0]
        local = step["contributions"][0]["local_derivative"]["values"][0]
        assert local == pytest.approx(1 - np.tanh(0.5) ** 2)

    def test_explicit_upstream_gradient_is_used(self):
        trace = trace_expression(
            lambda x: x * x, [1.0, 2.0, 3.0], names=["x"], upstream=[1.0, 0.0, 2.0]
        )
        assert trace["leaf_gradients"][0]["gradient"]["values"] == [2.0, 0.0, 12.0]

    def test_non_scalar_without_upstream_is_refused(self):
        with pytest.raises(gl.GradientError):
            trace_expression(lambda x: x * x, [1.0, 2.0], names=["x"])

    def test_large_values_are_truncated_and_say_so(self):
        trace = trace_expression(lambda x: x.sum(), np.ones(500), names=["x"], max_elements=16)
        value = trace["inputs"][0]["value"]
        assert value["truncated"] is True
        assert len(value["values"]) == 16
        assert value["size"] == 500
        # stats are still computed over the whole array
        assert value["stats"]["mean"] == 1.0

    def test_non_finite_values_survive_json(self):
        from gradlab.inspection.schema import describe_array

        payload = describe_array(np.array([np.nan, np.inf, -np.inf, 1.0]))
        assert payload["values"][:3] == ["NaN", "Infinity", "-Infinity"]
        json.dumps(payload)


class TestConsistency:
    def test_trace_matches_a_plain_backward_pass(self):
        """The Microscope must show the engine's numbers, not its own."""

        def build():
            x = gl.Tensor(np.array([0.3, -1.2, 2.0]), requires_grad=True)
            y = gl.Tensor(np.array([1.5, 0.4, -0.7]), requires_grad=True)
            return x, y

        x1, y1 = build()
        ((x1 * y1).tanh() + x1.exp()).sum().backward()
        plain = (x1.grad.data.copy(), y1.grad.data.copy())

        x2, y2 = build()
        trace = trace_expression(
            lambda a, b: ((a * b).tanh() + a.exp()).sum(), x2, y2, names=["x", "y"]
        )
        traced = tuple(np.array(g["gradient"]["values"]) for g in trace["leaf_gradients"])

        np.testing.assert_allclose(traced[0], plain[0], rtol=1e-14)
        np.testing.assert_allclose(traced[1], plain[1], rtol=1e-14)

    def test_traversal_order_is_a_valid_reverse_topological_order(self):
        """Check the ordering property itself, against the trace's own edges.

        Comparing node ids against a separately built graph would compare
        allocation counters, not ordering — the seq counter is global and
        monotonic, so two builds of the same expression never share ids.
        """
        trace = trace_expression(lambda t: (t * t + t).tanh(), 2.0, names=["x"])
        order = trace["traversal_order"]
        node_ids = {n["id"] for n in trace["nodes"]}

        assert set(order) == node_ids, "every node must be visited exactly once"
        assert len(order) == len(set(order))

        position = {node_id: i for i, node_id in enumerate(order)}
        for edge in trace["edges"]:
            if edge["from"] in position and edge["to"] in position:
                assert position[edge["to"]] < position[edge["from"]], (
                    f"{edge['to']} consumes {edge['from']}, so it must be processed first"
                )

    def test_forward_values_in_the_trace_match_direct_evaluation(self):
        trace = trace_expression(lambda x: (x * 3).tanh(), 0.7, names=["x"])
        tanh_node = next(n for n in trace["nodes"] if n["op"] == "tanh")
        assert tanh_node["output"]["value"]["values"][0] == pytest.approx(np.tanh(2.1))


class TestTrainingStepTrace:
    def test_captures_before_gradient_and_after(self):
        model = nn.Linear(2, 1, seed=0)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        x, y = gl.randn(4, 2, seed=1), gl.zeros(4, 1)
        trace = trace_training_step(model, optimizer, nn.mse_loss, x, y)

        weight = next(u for u in trace["parameter_updates"] if u["name"] == "weight")
        before = np.array(weight["before"]["values"])
        after = np.array(weight["after"]["values"])
        grad = np.array(weight["gradient"]["values"])
        delta = np.array(weight["delta"]["values"])

        np.testing.assert_allclose(delta, after - before, rtol=1e-12)
        np.testing.assert_allclose(after, before - 0.1 * grad, rtol=1e-10)

    def test_adam_state_includes_bias_correction(self):
        model = nn.Linear(2, 1, seed=0)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        trace = trace_training_step(
            model, optimizer, nn.mse_loss, gl.randn(4, 2, seed=1), gl.zeros(4, 1)
        )
        state = trace["parameter_updates"][0]["state_after"]
        assert state["bias_correction"]["t"] == 1
        assert state["bias_correction"]["one_minus_beta1_t"] == pytest.approx(0.1)
        assert "m" in state and "v" in state

    def test_trace_is_json_serialisable(self):
        model = nn.Sequential(nn.Linear(2, 3, seed=0), nn.Tanh(), nn.Linear(3, 1, seed=1))
        optimizer = optim.AdamW(model.parameters(), lr=0.01)
        trace = trace_training_step(
            model, optimizer, nn.mse_loss, gl.randn(4, 2, seed=1), gl.zeros(4, 1)
        )
        json.dumps(trace)
        assert trace["optimizer"]["type"] == "AdamW"
        assert trace["model"]["num_parameters"] == model.num_parameters()


class TestRestrictedExpressions:
    def test_parses_and_evaluates(self):
        out, variables = evaluate("x^2 + 2*x", {"x": 3.0})
        out.backward()
        assert out.item() == 15.0 and variables["x"].grad.item() == 8.0

    def test_constant_exponent_uses_the_scalar_pow_op(self):
        out, _ = evaluate("x^2", {"x": 3.0})
        assert out.grad_fn.op == "pow"

    def test_tensor_exponent_uses_pow_tensor(self):
        out, _ = evaluate("x^y", {"x": 2.0, "y": 3.0})
        assert out.grad_fn.op == "pow_tensor"

    @pytest.mark.parametrize(
        "hostile",
        [
            "__import__('os')",
            "x.__class__",
            "open('/etc/passwd')",
            "eval('1')",
            "x; import os",
            "os.system('ls')",
            "lambda: 1",
            "x if x else x",
            "[1,2,3]",
            "{'a':1}",
            "x`y",
            "x\\ny",
            "globals()",
        ],
    )
    def test_hostile_input_is_rejected(self, hostile):
        with pytest.raises(ExpressionError):
            evaluate(hostile, {})

    @pytest.mark.parametrize(
        "bad,match",
        [
            ("x" * 500, "characters"),
            ("(" * 40 + "x" + ")" * 40, "nests deeper"),
            ("foo(x)", "Unknown function"),
            (r"x +", "ended unexpectedly"),
            ("exp", "needs parentheses"),
            ("", "empty"),
        ],
    )
    def test_limits_and_syntax_errors_are_explained(self, bad, match):
        with pytest.raises(ExpressionError, match=match):
            evaluate(bad, {})

    def test_oversized_tensor_is_rejected(self):
        with pytest.raises(ExpressionError, match="elements"):
            evaluate("sum(x)", {"x": np.ones(10000).tolist()})

    def test_non_finite_variable_is_rejected(self):
        with pytest.raises(ExpressionError, match="non-finite"):
            evaluate("x*2", {"x": float("nan")})

    def test_unused_variable_is_reported(self):
        with pytest.raises(ExpressionError, match="never uses"):
            evaluate("x*2", {"x": 1.0, "zzz": 5.0})

    def test_too_many_operations_is_rejected(self):
        with pytest.raises(ExpressionError, match=r"too many tokens|more than"):
            evaluate("+".join(["x"] * 200), {"x": 1.0})

    def test_every_allowed_function_round_trips(self):
        from gradlab.inspection.expressions import ALLOWED_OPS

        for name in ALLOWED_OPS:
            out, variables = evaluate(f"{name}(x)", {"x": 1.5})
            out.sum().backward()
            assert variables["x"].grad is not None, name

    def test_parse_produces_plain_data_only(self):
        tree = parse_expression("tanh(x*2) + y")
        assert json.loads(json.dumps(tree)) == tree
