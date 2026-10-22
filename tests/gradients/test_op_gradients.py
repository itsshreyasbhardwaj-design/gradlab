"""Finite-difference validation of every differentiable operation.

Each entry in ``CASES`` is checked with central differences at ``float64``
precision. Inputs are chosen to stay inside each operation's domain and away
from kinks: strictly positive for ``log``/``sqrt``, bounded for ``exp``, and
offset from zero for ``relu``/``abs``/``max`` so the finite difference is
actually measuring a one-sided slope that the subgradient agrees with.
"""

from __future__ import annotations

import numpy as np
import pytest

import gradlab as gl

RNG = np.random.default_rng(7)


def _pos(*shape):
    """Strictly positive input, for log / sqrt / fractional powers."""
    return RNG.uniform(0.5, 2.5, shape)


def _gen(*shape):
    """General input, bounded so exp() stays well-conditioned."""
    return RNG.uniform(-1.5, 1.5, shape)


def _off_zero(*shape):
    """Input kept clear of 0 so relu/abs have a well-defined local slope."""
    x = RNG.uniform(0.3, 1.5, shape)
    return x * RNG.choice([-1.0, 1.0], shape)


# (id, function, [input arrays])
CASES = [
    # --- arithmetic -------------------------------------------------------
    ("add", lambda a, b: (a + b).sum(), [_gen(3, 4), _gen(3, 4)]),
    ("sub", lambda a, b: (a - b).sum(), [_gen(3, 4), _gen(3, 4)]),
    ("mul", lambda a, b: (a * b).sum(), [_gen(3, 4), _gen(3, 4)]),
    ("div", lambda a, b: (a / b).sum(), [_gen(3, 4), _pos(3, 4)]),
    (
        "neg",
        lambda a: (-a).sum(),
        [
            _gen(
                4,
            )
        ],
    ),
    (
        "abs",
        lambda a: a.__abs__().sum(),
        [
            _off_zero(
                5,
            )
        ],
    ),
    (
        "pow_const_2",
        lambda a: (a**2).sum(),
        [
            _gen(
                4,
            )
        ],
    ),
    (
        "pow_const_3",
        lambda a: (a**3).sum(),
        [
            _gen(
                4,
            )
        ],
    ),
    (
        "pow_const_neg1",
        lambda a: (a**-1).sum(),
        [
            _pos(
                4,
            )
        ],
    ),
    (
        "pow_const_half",
        lambda a: (a**0.5).sum(),
        [
            _pos(
                4,
            )
        ],
    ),
    (
        "pow_tensor",
        lambda a, b: (a**b).sum(),
        [
            _pos(
                3,
            ),
            _gen(
                3,
            ),
        ],
    ),
    (
        "rsub_scalar",
        lambda a: (3.0 - a).sum(),
        [
            _gen(
                4,
            )
        ],
    ),
    (
        "rdiv_scalar",
        lambda a: (2.0 / a).sum(),
        [
            _pos(
                4,
            )
        ],
    ),
    # --- broadcasting -----------------------------------------------------
    (
        "broadcast_row_bias",
        lambda a, b: (a + b).sum(),
        [
            _gen(5, 3),
            _gen(
                3,
            ),
        ],
    ),
    ("broadcast_col", lambda a, b: (a * b).sum(), [_gen(5, 3), _gen(5, 1)]),
    ("broadcast_scalar_tensor", lambda a, b: (a * b).sum(), [_gen(4, 2), _gen()]),
    ("broadcast_both_ways", lambda a, b: (a + b).sum(), [_gen(4, 1), _gen(1, 3)]),
    ("broadcast_3d", lambda a, b: (a * b).sum(), [_gen(2, 3, 4), _gen(3, 1)]),
    # --- matrix -----------------------------------------------------------
    ("matmul_2d", lambda a, b: (a @ b).sum(), [_gen(3, 4), _gen(4, 2)]),
    (
        "matmul_vec_vec",
        lambda a, b: (a @ b).sum(),
        [
            _gen(
                4,
            ),
            _gen(
                4,
            ),
        ],
    ),
    (
        "matmul_mat_vec",
        lambda a, b: (a @ b).sum(),
        [
            _gen(3, 4),
            _gen(
                4,
            ),
        ],
    ),
    (
        "matmul_vec_mat",
        lambda a, b: (a @ b).sum(),
        [
            _gen(
                3,
            ),
            _gen(3, 5),
        ],
    ),
    ("matmul_batched", lambda a, b: (a @ b).sum(), [_gen(2, 3, 4), _gen(2, 4, 3)]),
    ("matmul_batch_broadcast", lambda a, b: (a @ b).sum(), [_gen(2, 3, 4), _gen(4, 3)]),
    ("matmul_chain", lambda a, b, c: (a @ b @ c).sum(), [_gen(2, 3), _gen(3, 4), _gen(4, 2)]),
    # --- shape ------------------------------------------------------------
    ("reshape", lambda a: (a.reshape(6, 2) * 2).sum(), [_gen(3, 4)]),
    ("transpose", lambda a: (a.T * 3).sum(), [_gen(3, 4)]),
    ("transpose_3d", lambda a: (a.transpose(2, 0, 1) ** 2).sum(), [_gen(2, 3, 4)]),
    ("flatten", lambda a: (a.flatten() ** 2).sum(), [_gen(2, 3)]),
    ("index_int", lambda a: (a[1] ** 2).sum(), [_gen(4, 3)]),
    ("index_slice", lambda a: (a[1:3, :2] ** 2).sum(), [_gen(4, 3)]),
    ("index_fancy_repeat", lambda a: (a[[0, 0, 2]] ** 2).sum(), [_gen(4, 3)]),
    (
        "clone",
        lambda a: (a.clone() * a).sum(),
        [
            _gen(
                4,
            )
        ],
    ),
    # --- activations ------------------------------------------------------
    (
        "exp",
        lambda a: a.exp().sum(),
        [
            _gen(
                5,
            )
        ],
    ),
    (
        "log",
        lambda a: a.log().sum(),
        [
            _pos(
                5,
            )
        ],
    ),
    (
        "sqrt",
        lambda a: a.sqrt().sum(),
        [
            _pos(
                5,
            )
        ],
    ),
    (
        "tanh",
        lambda a: a.tanh().sum(),
        [
            _gen(
                5,
            )
        ],
    ),
    (
        "sigmoid",
        lambda a: a.sigmoid().sum(),
        [
            _gen(
                5,
            )
        ],
    ),
    (
        "relu",
        lambda a: a.relu().sum(),
        [
            _off_zero(
                6,
            )
        ],
    ),
    (
        "sin",
        lambda a: a.sin().sum(),
        [
            _gen(
                5,
            )
        ],
    ),
    (
        "cos",
        lambda a: a.cos().sum(),
        [
            _gen(
                5,
            )
        ],
    ),
    # --- reductions -------------------------------------------------------
    ("sum_all", lambda a: a.sum(), [_gen(3, 4)]),
    ("sum_axis0", lambda a: (a.sum(axis=0) ** 2).sum(), [_gen(3, 4)]),
    ("sum_axis1_keepdims", lambda a: (a.sum(axis=1, keepdims=True) ** 2).sum(), [_gen(3, 4)]),
    ("sum_multi_axis", lambda a: (a.sum(axis=(0, 2)) ** 2).sum(), [_gen(2, 3, 4)]),
    ("mean_all", lambda a: a.mean(), [_gen(3, 4)]),
    ("mean_axis1", lambda a: (a.mean(axis=1) ** 2).sum(), [_gen(3, 4)]),
    ("mean_axis_neg", lambda a: (a.mean(axis=-1) ** 2).sum(), [_gen(3, 4)]),
    ("max_all", lambda a: a.max(), [_gen(3, 4)]),
    ("max_axis1", lambda a: (a.max(axis=1) ** 2).sum(), [_gen(3, 4)]),
    ("min_axis0", lambda a: (a.min(axis=0) ** 2).sum(), [_gen(3, 4)]),
    # --- compositions that exercise fan-out and shared subgraphs ----------
    (
        "shared_x_twice",
        lambda a: (a * a + 2 * a).sum(),
        [
            _gen(
                4,
            )
        ],
    ),
    (
        "diamond",
        lambda a: ((a.exp() + a.tanh()) * a.sigmoid()).sum(),
        [
            _gen(
                4,
            )
        ],
    ),
    (
        "deep_chain",
        lambda a: a.tanh().tanh().tanh().tanh().sum(),
        [
            _gen(
                3,
            )
        ],
    ),
    (
        "mlp_like",
        lambda x, w1, b1, w2: (((x @ w1 + b1).tanh() @ w2) ** 2).sum(),
        [
            _gen(4, 3),
            _gen(3, 5),
            _gen(
                5,
            ),
            _gen(5, 2),
        ],
    ),
    ("log_of_sum_exp", lambda a: a.exp().sum(axis=-1).log().sum(), [_gen(3, 4)]),
    ("normalise", lambda a: (a / a.sum(axis=-1, keepdims=True)).sum(), [_pos(3, 4)]),
]


@pytest.mark.parametrize("fn,arrays", [(c[1], c[2]) for c in CASES], ids=[c[0] for c in CASES])
def test_gradient_matches_central_differences(fn, arrays):
    inputs = [gl.Tensor(a.copy(), requires_grad=True) for a in arrays]
    result = gl.gradcheck(fn, *inputs, eps=1e-6, atol=1e-7, rtol=1e-5, check_kinks=False)
    assert result.passed, result.report()
    # Record the measured error so docs/validation.md can quote real numbers.
    assert result.max_abs_error < 1e-6, result.report()


def test_softmax_and_log_softmax_gradients():
    from gradlab.ops.activation import LogSoftmax, Softmax

    x = gl.Tensor(_gen(4, 5), requires_grad=True)
    assert gl.gradcheck(lambda t: (LogSoftmax.apply(t, axis=-1) ** 2).sum(), x).passed
    x.zero_grad()
    assert gl.gradcheck(lambda t: (Softmax.apply(t, axis=-1) ** 2).sum(), x).passed


def test_log_softmax_is_stable_for_large_logits():
    """The naive log(softmax(x)) route overflows here; the log-sum-exp form must not."""
    from gradlab.ops.activation import LogSoftmax

    x = gl.Tensor([[1000.0, 1001.0, 999.0]], requires_grad=True)
    out = LogSoftmax.apply(x, axis=-1)
    assert np.all(np.isfinite(out.data)), out.data
    # exp(log_softmax) must still be a probability distribution
    np.testing.assert_allclose(np.exp(out.data).sum(), 1.0, atol=1e-12)
    out.sum().backward()
    assert np.all(np.isfinite(x.grad.data))


def test_max_splits_gradient_equally_among_ties():
    """The documented subgradient choice for a tie, asserted rather than assumed."""
    x = gl.Tensor([2.0, 2.0, 1.0], requires_grad=True)
    x.max().backward()
    assert x.grad.tolist() == [0.5, 0.5, 0.0]


def test_relu_subgradient_at_zero_is_zero():
    x = gl.Tensor([0.0], requires_grad=True)
    x.relu().backward()
    assert x.grad.tolist() == [0.0]


def test_gradcheck_detects_a_deliberately_wrong_rule(monkeypatch):
    """A gradient checker that never fails is worthless. Prove this one fails."""
    from gradlab.ops.arithmetic import Mul

    monkeypatch.setattr(Mul, "backward", staticmethod(lambda ctx, g: (ctx.b * g * 1.5, ctx.a * g)))
    x = gl.Tensor([1.0, 2.0], requires_grad=True)
    y = gl.Tensor([3.0, 4.0], requires_grad=True)
    with pytest.raises(AssertionError, match="gradcheck FAILED"):
        gl.gradcheck(lambda a, b: (a * b).sum(), x, y)
