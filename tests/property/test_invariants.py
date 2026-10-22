"""Property-based tests: invariants that must hold for *any* valid input.

Example-based tests check the cases an author thought of. These check properties
the mathematics guarantees, over inputs Hypothesis chooses — including the
awkward shapes and magnitudes a human would not write down.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

import gradlab as gl

SETTINGS = settings(max_examples=60, deadline=None)

# Bounded magnitudes: these are tests of algebraic structure, not of float limits.
finite = st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=0.2, max_value=8.0, allow_nan=False, allow_infinity=False)
shapes = npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4)


def arrays(shape_strategy=shapes, elements=finite):
    return npst.arrays(dtype=np.float64, shape=shape_strategy, elements=elements)


@SETTINGS
@given(arrays())
def test_gradient_always_has_the_shape_of_its_tensor(a):
    x = gl.Tensor(a, requires_grad=True)
    (x * x).sum().backward()
    assert x.grad.shape == x.shape


@SETTINGS
@given(arrays(), arrays())
def test_broadcast_gradients_keep_their_own_shapes(a, b):
    try:
        np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        assume(False)
    x, y = gl.Tensor(a, requires_grad=True), gl.Tensor(b, requires_grad=True)
    (x * y).sum().backward()
    assert x.grad.shape == x.shape
    assert y.grad.shape == y.shape


@SETTINGS
@given(arrays(), st.floats(min_value=-5, max_value=5), st.floats(min_value=-5, max_value=5))
def test_gradient_is_linear_in_the_output(a, alpha, beta):
    """d(alpha*f + beta*g)/dx == alpha*df/dx + beta*dg/dx, by linearity of d/dx."""
    x = gl.Tensor(a, requires_grad=True)
    (alpha * (x * x).sum() + beta * x.sum()).backward()
    combined = x.grad.data.copy()

    x.zero_grad()
    (x * x).sum().backward()
    d_square = x.grad.data.copy()
    x.zero_grad()
    x.sum().backward()
    d_linear = x.grad.data.copy()

    np.testing.assert_allclose(combined, alpha * d_square + beta * d_linear, atol=1e-9)


@SETTINGS
@given(arrays())
def test_sum_gradient_is_all_ones(a):
    x = gl.Tensor(a, requires_grad=True)
    x.sum().backward()
    np.testing.assert_allclose(x.grad.data, np.ones_like(a))


@SETTINGS
@given(arrays())
def test_mean_gradient_is_one_over_n(a):
    x = gl.Tensor(a, requires_grad=True)
    x.mean().backward()
    np.testing.assert_allclose(x.grad.data, np.full_like(a, 1.0 / a.size))


@SETTINGS
@given(arrays(npst.array_shapes(min_dims=2, max_dims=3, min_side=1, max_side=4)))
def test_reduction_over_all_axes_matches_chained_reductions(a):
    x = gl.Tensor(a, requires_grad=True)
    total = x.sum()
    axis_by_axis = x
    for _ in range(a.ndim):
        axis_by_axis = axis_by_axis.sum(axis=0)
    np.testing.assert_allclose(total.data, axis_by_axis.data, rtol=1e-12)


@SETTINGS
@given(arrays())
def test_backward_is_deterministic(a):
    def grad_of():
        x = gl.Tensor(a, requires_grad=True)
        ((x * x).tanh() + x.sigmoid()).sum().backward()
        return x.grad.data.copy()

    np.testing.assert_array_equal(grad_of(), grad_of())


@SETTINGS
@given(arrays())
def test_accumulation_over_n_passes_is_n_times_one_pass(a):
    x = gl.Tensor(a, requires_grad=True)
    (x * x).sum().backward()
    single = x.grad.data.copy()
    for _ in range(2):
        (x * x).sum().backward()
    np.testing.assert_allclose(x.grad.data, 3.0 * single, rtol=1e-12)


@SETTINGS
@given(arrays())
def test_reshape_round_trip_is_the_identity_for_gradients(a):
    x = gl.Tensor(a, requires_grad=True)
    (x.reshape(-1 if False else (a.size,)).reshape(a.shape) * 3).sum().backward()
    np.testing.assert_allclose(x.grad.data, np.full_like(a, 3.0))


@SETTINGS
@given(arrays())
def test_transpose_twice_is_the_identity_for_gradients(a):
    x = gl.Tensor(a, requires_grad=True)
    (x.T.T * 2).sum().backward()
    np.testing.assert_allclose(x.grad.data, np.full_like(a, 2.0))


@SETTINGS
@given(arrays(elements=positive))
def test_log_exp_round_trip_has_unit_gradient(a):
    x = gl.Tensor(a, requires_grad=True)
    x.log().exp().sum().backward()
    np.testing.assert_allclose(x.grad.data, np.ones_like(a), rtol=1e-8)


@SETTINGS
@given(arrays())
def test_detached_branches_contribute_nothing(a):
    x = gl.Tensor(a, requires_grad=True)
    ((x * 3).detach() + x * 5).sum().backward()
    np.testing.assert_allclose(x.grad.data, np.full_like(a, 5.0))


@SETTINGS
@given(arrays(npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5)))
def test_matmul_gradient_shapes_are_always_the_input_shapes(a):
    b = np.ones((a.shape[1], 3))
    x, y = gl.Tensor(a, requires_grad=True), gl.Tensor(b, requires_grad=True)
    (x @ y).sum().backward()
    assert x.grad.shape == x.shape and y.grad.shape == y.shape


@SETTINGS
@given(arrays(elements=st.floats(min_value=-3, max_value=3)))
def test_gradcheck_passes_for_a_smooth_composition(a):
    assume(np.all(np.abs(a) > 1e-3))  # stay clear of relu/abs kinks
    x = gl.Tensor(a, requires_grad=True)
    result = gl.gradcheck(
        lambda t: (t.tanh() * t.sigmoid() + t * t).sum(),
        x,
        atol=1e-6,
        rtol=1e-5,
        check_kinks=False,
        raise_exception=False,
    )
    assert result.passed, result.report()


@SETTINGS
@given(st.integers(min_value=0, max_value=2**31 - 1), st.integers(min_value=1, max_value=6))
def test_seeded_initialisation_is_reproducible(seed, n):
    from gradlab import nn

    a = nn.Linear(n, n, seed=seed)
    b = nn.Linear(n, n, seed=seed)
    np.testing.assert_array_equal(a.weight.data, b.weight.data)


@SETTINGS
@given(arrays(npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=4)))
def test_softmax_rows_sum_to_one(a):
    from gradlab.ops.activation import Softmax

    out = Softmax.apply(gl.Tensor(a), axis=-1)
    np.testing.assert_allclose(out.data.sum(axis=-1), np.ones(a.shape[0]), rtol=1e-12)


@SETTINGS
@given(arrays(npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=4)))
def test_log_softmax_equals_log_of_softmax_in_the_safe_range(a):
    from gradlab.ops.activation import LogSoftmax, Softmax

    stable = LogSoftmax.apply(gl.Tensor(a), axis=-1).data
    naive = np.log(Softmax.apply(gl.Tensor(a), axis=-1).data)
    np.testing.assert_allclose(stable, naive, atol=1e-10)


@SETTINGS
@given(arrays(), st.sampled_from([0.0, 0.3, 0.9]))
def test_an_sgd_step_always_moves_against_the_gradient(a, momentum):
    from gradlab import nn, optim

    p = nn.Parameter(gl.Tensor(a))
    opt = optim.SGD([p], lr=0.01, momentum=momentum)
    grad = np.sign(a) + 0.5
    p._grad = gl.Tensor(grad)
    before = p.data.copy()
    opt.step()
    moved = p.data - before
    assert np.all(moved * grad <= 1e-12), "every component must move downhill"
