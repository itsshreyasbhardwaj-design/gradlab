"""Graph construction, traversal order, and gradient-flow semantics."""

from __future__ import annotations

import gc

import numpy as np
import pytest

import gradlab as gl
from gradlab.autograd import topological_order


class TestGraphStructure:
    def test_node_records_op_inputs_and_rule(self):
        a = gl.Tensor(2.0, requires_grad=True)
        b = gl.Tensor(3.0, requires_grad=True)
        c = a * b
        node = c.grad_fn
        assert node.op == "mul"
        assert node.inputs == (a, b)
        assert "d(a*b)/da" in node.backward_rule
        assert node.output() is c

    def test_seq_increases_in_forward_order(self):
        x = gl.Tensor(1.0, requires_grad=True)
        first = (x + 1).grad_fn
        second = (x + 2).grad_fn
        assert second.seq > first.seq

    def test_output_reference_is_weak_so_there_is_no_cycle(self):
        """Tensor -> Node -> Tensor would be an uncollectable cycle if it were strong."""
        x = gl.Tensor(1.0, requires_grad=True)
        y = x * 2
        node = y.grad_fn
        del y
        gc.collect()
        assert node.output() is None, "the node must not keep its output alive"

    def test_no_node_is_created_when_nothing_requires_grad(self):
        a, b = gl.Tensor(1.0), gl.Tensor(2.0)
        assert (a * b).grad_fn is None

    def test_a_node_is_created_if_any_input_requires_grad(self):
        a = gl.Tensor(1.0, requires_grad=True)
        assert (a * gl.Tensor(2.0)).grad_fn is not None

    def test_fan_out_is_shared_not_duplicated(self):
        """Using one tensor twice must reference the same tensor, not copy it."""
        x = gl.Tensor(2.0, requires_grad=True)
        y = x * x
        assert y.grad_fn.inputs[0] is y.grad_fn.inputs[1] is x


class TestTopologicalOrder:
    def test_order_puts_every_consumer_before_its_producer(self):
        x = gl.Tensor(2.0, requires_grad=True)
        a = x * x
        b = a + x
        c = b * a
        order = topological_order(c.grad_fn)
        position = {id(n): i for i, n in enumerate(order)}
        for node in order:
            for child in node.next_nodes:
                if child is not None:
                    assert position[id(node)] < position[id(child)], (
                        "a node must be processed before the node that produced its input"
                    )

    def test_each_node_appears_exactly_once(self):
        x = gl.Tensor(2.0, requires_grad=True)
        a = x.tanh()
        out = (a + a) * (a * a)
        order = topological_order(out.grad_fn)
        assert len({id(n) for n in order}) == len(order)

    def test_deep_graph_does_not_hit_the_recursion_limit(self):
        """The traversal is iterative on purpose; 5000 chained ops must work."""
        x = gl.Tensor(0.5, requires_grad=True)
        y = x
        for _ in range(5000):
            y = y * 1.0001
        y.backward()
        assert np.isfinite(x.grad.item())
        assert x.grad.item() == pytest.approx(1.0001**5000, rel=1e-9)


class TestGradientFlow:
    def test_chain_rule_through_three_ops(self):
        x = gl.Tensor(2.0, requires_grad=True)
        (x * x * x).backward()
        assert x.grad.item() == pytest.approx(12.0)  # 3x^2

    def test_shared_subgraph_accumulates_from_both_branches(self):
        x = gl.Tensor(2.0, requires_grad=True)
        y = x * x  # dy/dx = 4
        (y + y).backward()  # dz/dy = 2 from each branch
        assert x.grad.item() == 8.0

    def test_diamond_dependency(self):
        x = gl.Tensor(3.0, requires_grad=True)
        a = x * 2
        b = x * 5
        (a * b).backward()  # d(10x^2)/dx = 20x = 60
        assert x.grad.item() == pytest.approx(60.0)

    def test_gradient_stops_at_detach(self):
        x = gl.Tensor(2.0, requires_grad=True)
        y = (x * 3).detach()
        z = x * 5
        (z + y).backward()
        assert x.grad.item() == 5.0, "the detached branch must contribute nothing"

    def test_unused_branch_gets_no_gradient(self):
        x = gl.Tensor(2.0, requires_grad=True)
        y = gl.Tensor(5.0, requires_grad=True)
        (x * 2).backward()
        assert x.grad.item() == 2.0
        assert y.grad is None

    def test_retain_graph_allows_a_second_backward(self):
        x = gl.Tensor(2.0, requires_grad=True)
        y = x * x * x
        y.backward(retain_graph=True)
        y.backward()
        assert x.grad.item() == pytest.approx(24.0), "two passes accumulate 12 + 12"

    def test_non_scalar_backward_requires_a_matching_seed_shape(self):
        x = gl.Tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(gl.GradientError, match="shape"):
            (x * x).backward(gl.Tensor([1.0, 1.0, 1.0]))

    def test_explicit_upstream_gradient_is_respected(self):
        x = gl.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        (x * x).backward(gl.Tensor([1.0, 0.0, 2.0]))
        assert x.grad.tolist() == [2.0, 0.0, 12.0]

    def test_leaf_root_backward_deposits_the_seed(self):
        x = gl.Tensor(5.0, requires_grad=True)
        x.backward()
        assert x.grad.item() == 1.0


class TestGradModes:
    def test_no_grad_blocks_history(self):
        x = gl.Tensor(2.0, requires_grad=True)
        with gl.no_grad():
            y = x * x
        assert y.grad_fn is None and not y.requires_grad

    def test_no_grad_nests_and_restores(self):
        assert gl.is_grad_enabled()
        with gl.no_grad():
            assert not gl.is_grad_enabled()
            with gl.no_grad():
                assert not gl.is_grad_enabled()
            assert not gl.is_grad_enabled()
        assert gl.is_grad_enabled()

    def test_enable_grad_overrides_inside_no_grad(self):
        x = gl.Tensor(2.0, requires_grad=True)
        with gl.no_grad(), gl.enable_grad():
            y = x * x
        assert y.grad_fn is not None

    def test_grad_mode_is_restored_after_an_exception(self):
        with pytest.raises(ValueError), gl.no_grad():
            raise ValueError
        assert gl.is_grad_enabled()


class TestInplaceSafety:
    def test_mutating_a_saved_input_is_detected(self):
        x = gl.Tensor([2.0, 3.0], requires_grad=True)
        y = x * x
        x.data = np.array([9.0, 9.0])
        with pytest.raises(gl.InplaceModificationError, match="modified in place"):
            y.sum().backward()

    def test_mutating_through_a_detached_view_is_also_detected(self):
        """detach() shares the buffer, so it must share the version counter too."""
        x = gl.Tensor([2.0, 3.0], requires_grad=True)
        y = x * x
        x.data = x.data * 1.0  # a fresh assignment still bumps the version
        with pytest.raises(gl.InplaceModificationError):
            y.sum().backward()

    def test_an_optimiser_style_update_after_backward_is_fine(self):
        x = gl.Tensor([2.0, 3.0], requires_grad=True)
        y = (x * x).sum()
        y.backward()
        with gl.no_grad():
            x.data = x.data - 0.1 * x.grad.data
        assert x.data.tolist() == pytest.approx([1.6, 2.4])
