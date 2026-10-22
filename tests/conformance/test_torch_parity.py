"""Independent cross-checks against PyTorch.

PyTorch is used strictly as an **oracle**: these tests build the same
computation twice, once in GradLab and once in PyTorch, from identical input
arrays, and compare forward values and gradients. Nothing in ``src/gradlab``
imports torch (``tests/unit/test_architecture.py`` enforces that), so a pass
here is genuine agreement between two independent implementations.

Everything is done in float64 on CPU so the comparison is about the derivative
rules, not about floating-point width.
"""

from __future__ import annotations

import numpy as np
import pytest

import gradlab as gl

try:
    import torch

    torch.set_default_dtype(torch.float64)
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(
        not HAS_TORCH,
        reason="PyTorch not installed; install the 'oracle' extra to run conformance tests",
    ),
]

RNG = np.random.default_rng(1234)
TOL = {"rtol": 1e-10, "atol": 1e-12}


def _pair(arrays):
    gls = [gl.Tensor(a.copy(), requires_grad=True) for a in arrays]
    tts = [torch.tensor(a.copy(), requires_grad=True) for a in arrays]
    return gls, tts


def _compare(gl_fn, torch_fn, arrays, tol=None):
    """Run both implementations and compare forward value and every gradient."""
    tol = tol or TOL
    gls, tts = _pair(arrays)
    gl_out = gl_fn(*gls)
    tt_out = torch_fn(*tts)
    np.testing.assert_allclose(gl_out.data, tt_out.detach().numpy(), **tol)
    gl_out.backward()
    tt_out.backward()
    for i, (g, t) in enumerate(zip(gls, tts, strict=True)):
        assert g.grad is not None, f"GradLab produced no gradient for input {i}"
        np.testing.assert_allclose(
            g.grad.data, t.grad.numpy(), err_msg=f"gradient mismatch for input {i}", **tol
        )
    return gl_out, tt_out


def g(*shape):
    return RNG.uniform(-1.2, 1.2, shape)


def p(*shape):
    return RNG.uniform(0.4, 2.0, shape)


# --- elementwise and broadcasting ---------------------------------------


@pytest.mark.parametrize(
    "name,fn,arrays",
    [
        ("add", lambda a, b: (a + b).sum(), [g(3, 4), g(3, 4)]),
        ("mul", lambda a, b: (a * b).sum(), [g(3, 4), g(3, 4)]),
        ("div", lambda a, b: (a / b).sum(), [g(3, 4), p(3, 4)]),
        ("sub", lambda a, b: (a - b).sum(), [g(3, 4), g(3, 4)]),
        (
            "pow2",
            lambda a: (a**2).sum(),
            [
                g(
                    4,
                )
            ],
        ),
        (
            "exp",
            lambda a: a.exp().sum(),
            [
                g(
                    4,
                )
            ],
        ),
        (
            "log",
            lambda a: a.log().sum(),
            [
                p(
                    4,
                )
            ],
        ),
        (
            "sqrt",
            lambda a: a.sqrt().sum(),
            [
                p(
                    4,
                )
            ],
        ),
        (
            "tanh",
            lambda a: a.tanh().sum(),
            [
                g(
                    4,
                )
            ],
        ),
        (
            "sigmoid",
            lambda a: a.sigmoid().sum(),
            [
                g(
                    4,
                )
            ],
        ),
        (
            "relu",
            lambda a: a.relu().sum(),
            [
                g(
                    8,
                )
            ],
        ),
        (
            "sin",
            lambda a: a.sin().sum(),
            [
                g(
                    4,
                )
            ],
        ),
        (
            "cos",
            lambda a: a.cos().sum(),
            [
                g(
                    4,
                )
            ],
        ),
        (
            "bias_broadcast",
            lambda a, b: (a + b).sum(),
            [
                g(6, 3),
                g(
                    3,
                ),
            ],
        ),
        ("col_broadcast", lambda a, b: (a * b).sum(), [g(6, 3), g(6, 1)]),
        ("mutual_broadcast", lambda a, b: (a + b).sum(), [g(4, 1), g(1, 5)]),
        ("broadcast_3d", lambda a, b: (a * b).sum(), [g(2, 3, 4), g(3, 1)]),
        ("matmul_2d", lambda a, b: (a @ b).sum(), [g(3, 4), g(4, 2)]),
        (
            "matmul_mat_vec",
            lambda a, b: (a @ b).sum(),
            [
                g(3, 4),
                g(
                    4,
                ),
            ],
        ),
        (
            "matmul_vec_mat",
            lambda a, b: (a @ b).sum(),
            [
                g(
                    3,
                ),
                g(3, 5),
            ],
        ),
        (
            "matmul_vec_vec",
            lambda a, b: (a @ b).sum(),
            [
                g(
                    4,
                ),
                g(
                    4,
                ),
            ],
        ),
        ("matmul_batched", lambda a, b: (a @ b).sum(), [g(2, 3, 4), g(2, 4, 3)]),
        ("matmul_batch_bcast", lambda a, b: (a @ b).sum(), [g(5, 3, 4), g(4, 2)]),
        ("sum_axis", lambda a: (a.sum(axis=0) ** 2).sum(), [g(3, 4)]),
        ("mean_axis", lambda a: (a.mean(axis=1) ** 2).sum(), [g(3, 4)]),
        ("sum_keepdims", lambda a: (a.sum(axis=1, keepdims=True) ** 2).sum(), [g(3, 4)]),
        ("reshape", lambda a: (a.reshape(4, 3) ** 2).sum(), [g(3, 4)]),
        ("transpose", lambda a: (a.T**2).sum(), [g(3, 4)]),
        (
            "chain",
            lambda a: ((a * 2 + 1).tanh().exp()).sum(),
            [
                g(
                    5,
                )
            ],
        ),
        (
            "shared",
            lambda a: (a * a + 2 * a).sum(),
            [
                g(
                    5,
                )
            ],
        ),
        (
            "diamond",
            lambda a: ((a.exp() + a.tanh()) * a.sigmoid()).sum(),
            [
                g(
                    5,
                )
            ],
        ),
        (
            "mlp_forward",
            lambda x, w1, b1, w2, b2: (((x @ w1 + b1).tanh() @ w2 + b2) ** 2).sum(),
            [
                g(8, 3),
                g(3, 6),
                g(
                    6,
                ),
                g(6, 2),
                g(
                    2,
                ),
            ],
        ),
    ],
)
def test_matches_pytorch(name, fn, arrays):
    """The same expression, written once per framework, must agree to 1e-10."""
    _compare(fn, fn, arrays)


def test_amax_tie_convention_matches_torch_amax():
    """GradLab's max splits ties equally — the same convention as torch.amax."""
    arr = np.array([2.0, 2.0, 1.0, 2.0])
    x = gl.Tensor(arr.copy(), requires_grad=True)
    t = torch.tensor(arr.copy(), requires_grad=True)
    x.max().backward()
    torch.amax(t, dim=0).backward()
    np.testing.assert_allclose(x.grad.data, t.grad.numpy(), **TOL)


def test_log_softmax_matches_torch():
    from gradlab.ops.activation import LogSoftmax

    arr = g(4, 5)
    x = gl.Tensor(arr.copy(), requires_grad=True)
    t = torch.tensor(arr.copy(), requires_grad=True)
    out_g = LogSoftmax.apply(x, axis=-1)
    out_t = torch.log_softmax(t, dim=-1)
    np.testing.assert_allclose(out_g.data, out_t.detach().numpy(), **TOL)
    (out_g**2).sum().backward()
    (out_t**2).sum().backward()
    np.testing.assert_allclose(x.grad.data, t.grad.numpy(), **TOL)


def test_log_softmax_matches_torch_on_extreme_logits():
    """Stability claim, checked against an implementation known to be stable."""
    from gradlab.ops.activation import LogSoftmax

    arr = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]])
    x = gl.Tensor(arr.copy(), requires_grad=True)
    t = torch.tensor(arr.copy(), requires_grad=True)
    out_g = LogSoftmax.apply(x, axis=-1)
    out_t = torch.log_softmax(t, dim=-1)
    np.testing.assert_allclose(out_g.data, out_t.detach().numpy(), **TOL)
    out_g.sum().backward()
    out_t.sum().backward()
    np.testing.assert_allclose(x.grad.data, t.grad.numpy(), atol=1e-12)


def test_non_scalar_upstream_gradient_matches_torch():
    arr, seed = g(3, 4), g(3, 4)
    x = gl.Tensor(arr.copy(), requires_grad=True)
    t = torch.tensor(arr.copy(), requires_grad=True)
    (x * x).backward(gl.Tensor(seed.copy()))
    (t * t).backward(torch.tensor(seed.copy()))
    np.testing.assert_allclose(x.grad.data, t.grad.numpy(), **TOL)


def test_indexing_gradient_matches_torch_including_repeats():
    arr = g(5, 3)
    x = gl.Tensor(arr.copy(), requires_grad=True)
    t = torch.tensor(arr.copy(), requires_grad=True)
    (x[[0, 0, 3]] ** 2).sum().backward()
    (t[[0, 0, 3]] ** 2).sum().backward()
    np.testing.assert_allclose(x.grad.data, t.grad.numpy(), **TOL)
