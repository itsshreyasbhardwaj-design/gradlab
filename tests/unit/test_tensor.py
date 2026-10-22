"""Tensor semantics: construction, dtypes, shapes, views, errors."""

from __future__ import annotations

import numpy as np
import pytest

import gradlab as gl


class TestConstruction:
    def test_scalar_is_zero_dimensional(self):
        t = gl.Tensor(3.0)
        assert t.shape == () and t.ndim == 0 and t.size == 1
        assert isinstance(t.data, np.ndarray)

    def test_nested_sequence_infers_shape(self):
        t = gl.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert t.shape == (2, 3) and t.dtype == np.float64

    def test_default_dtype_is_float64(self):
        assert gl.Tensor([1.0]).dtype == np.float64

    def test_integers_are_promoted_when_grad_is_requested(self):
        t = gl.Tensor([1, 2, 3], requires_grad=True)
        assert t.dtype == np.float64

    def test_integers_stay_integers_without_grad(self):
        assert gl.Tensor([1, 2, 3]).dtype == np.int64

    def test_explicit_int_dtype_with_grad_is_an_error(self):
        with pytest.raises(gl.DTypeError, match="is not possible for dtype int64"):
            gl.Tensor([1, 2], dtype="int64", requires_grad=True)

    def test_float32_is_supported(self):
        t = gl.Tensor([1.0, 2.0], dtype="float32", requires_grad=True)
        assert t.dtype == np.float32
        (t * t).sum().backward()
        assert t.grad.dtype == np.float32

    def test_construction_copies_by_default(self):
        arr = np.ones(3)
        t = gl.Tensor(arr)
        arr[0] = 99.0
        assert t.data[0] == 1.0

    def test_copy_false_shares_the_buffer(self):
        arr = np.ones(3)
        t = gl.from_numpy(arr, copy=False)
        arr[0] = 99.0
        assert t.data[0] == 99.0


class TestProperties:
    def test_user_created_tensor_is_a_leaf(self):
        assert gl.Tensor(1.0, requires_grad=True).is_leaf

    def test_operation_result_is_not_a_leaf(self):
        x = gl.Tensor(1.0, requires_grad=True)
        y = x * 2
        assert not y.is_leaf and y.grad_fn is not None and y.grad_fn.op == "mul"

    def test_requires_grad_cannot_be_set_on_a_non_leaf(self):
        x = gl.Tensor(1.0, requires_grad=True)
        y = x * 2
        with pytest.raises(RuntimeError, match="only be changed on a leaf"):
            y.requires_grad = True

    def test_detach_gives_a_leaf_sharing_storage(self):
        x = gl.Tensor([1.0, 2.0], requires_grad=True)
        y = (x * 2).detach()
        assert y.is_leaf and not y.requires_grad

    def test_clone_is_differentiable(self):
        x = gl.Tensor(3.0, requires_grad=True)
        x.clone().backward()
        assert x.grad.item() == 1.0

    def test_grad_on_a_non_leaf_warns(self):
        x = gl.Tensor(1.0, requires_grad=True)
        y = x * 2
        y.backward()
        with pytest.warns(UserWarning, match="non-leaf"):
            assert y.grad is None

    def test_retain_grad_keeps_an_intermediate_gradient(self):
        x = gl.Tensor(3.0, requires_grad=True)
        y = x * x
        y.retain_grad()
        (y * 2).backward()
        assert y.grad.item() == 2.0
        assert x.grad.item() == 12.0


class TestShapeOps:
    def test_reshape_round_trips(self):
        t = gl.Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        assert t.reshape(3, 2).shape == (3, 2)
        assert t.reshape((6,)).shape == (6,)

    def test_reshape_with_wrong_element_count_is_an_error(self):
        t = gl.Tensor(np.arange(6.0))
        with pytest.raises(gl.ShapeError, match="element counts differ"):
            t.reshape(4, 2)

    def test_transpose_default_reverses_axes(self):
        t = gl.Tensor(np.zeros((2, 3, 4)))
        assert t.T.shape == (4, 3, 2)
        assert t.transpose(1, 0, 2).shape == (3, 2, 4)

    def test_transpose_rejects_a_non_permutation(self):
        with pytest.raises(gl.ShapeError, match="not a permutation"):
            gl.Tensor(np.zeros((2, 3))).transpose(0, 0)

    def test_flatten(self):
        t = gl.Tensor(np.zeros((2, 3, 4)))
        assert t.flatten().shape == (24,)
        assert t.flatten(start_dim=1).shape == (2, 12)

    def test_indexing_is_differentiable(self):
        t = gl.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        t[1].backward()
        assert t.grad.tolist() == [0.0, 1.0, 0.0]

    def test_repeated_index_accumulates(self):
        """x[[0, 0, 1]] reads element 0 twice, so it must receive 2.0."""
        t = gl.Tensor([1.0, 2.0], requires_grad=True)
        t[[0, 0, 1]].sum().backward()
        assert t.grad.tolist() == [2.0, 1.0]

    def test_slice_backward(self):
        t = gl.Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        t[:, 1].sum().backward()
        assert t.grad.tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]

    def test_stack_and_concat(self):
        a = gl.Tensor([1.0, 2.0], requires_grad=True)
        b = gl.Tensor([3.0, 4.0], requires_grad=True)
        assert gl.concat([a, b]).shape == (4,)
        s = gl.stack([a, b])
        assert s.shape == (2, 2)
        s.sum().backward()
        assert a.grad.tolist() == [1.0, 1.0] and b.grad.tolist() == [1.0, 1.0]


class TestErrors:
    def test_matmul_shape_mismatch_names_both_shapes(self):
        a, b = gl.Tensor(np.zeros((2, 3))), gl.Tensor(np.zeros((4, 5)))
        with pytest.raises(gl.ShapeError, match=r"inner dimensions 3 and 4"):
            a @ b

    def test_division_by_zero_raises(self):
        with pytest.raises(gl.DomainError, match="Division by zero"):
            gl.Tensor(1.0) / gl.Tensor(0.0)

    def test_log_of_non_positive_raises(self):
        with pytest.raises(gl.DomainError, match="strictly positive"):
            gl.Tensor([-1.0]).log()

    def test_sqrt_of_negative_raises(self):
        with pytest.raises(gl.DomainError, match="a >= 0"):
            gl.Tensor([-1.0]).sqrt()

    def test_fractional_power_of_negative_raises(self):
        with pytest.raises(gl.DomainError, match="not real-valued"):
            gl.Tensor([-2.0]) ** 0.5

    def test_data_assignment_cannot_change_shape(self):
        t = gl.Tensor([1.0, 2.0])
        with pytest.raises(gl.ShapeError, match="shape is fixed"):
            t.data = np.zeros(5)

    def test_item_on_a_multi_element_tensor_raises(self):
        with pytest.raises(ValueError, match="single-element"):
            gl.Tensor([1.0, 2.0]).item()

    def test_bool_of_a_multi_element_tensor_raises(self):
        with pytest.raises(ValueError, match="ambiguous"):
            bool(gl.Tensor([1.0, 2.0]))

    def test_backward_without_requires_grad_raises(self):
        with pytest.raises(gl.GradientError, match="does not require grad"):
            gl.Tensor(1.0).backward()
