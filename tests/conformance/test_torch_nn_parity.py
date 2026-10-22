"""Layers, losses and optimisers checked against PyTorch.

Each test constructs the same model twice — once in GradLab, once in PyTorch —
copies the *identical* initial parameters into both, feeds identical data, and
compares forward values, parameter gradients, and post-step parameters.

One layout difference matters and is handled explicitly everywhere below:
GradLab stores a Linear weight as ``(in_features, out_features)`` and computes
``x @ W``; PyTorch stores ``(out_features, in_features)`` and computes
``x @ Wᵀ``. So a weight copied between them is transposed, and so is its
gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

import gradlab as gl
from gradlab import nn, optim

try:
    import torch

    torch.set_default_dtype(torch.float64)
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed"),
]

RNG = np.random.default_rng(99)
TOL = {"rtol": 1e-10, "atol": 1e-12}


def paired_linear(in_f, out_f, bias=True):
    """A GradLab Linear and a torch Linear holding exactly the same parameters."""
    gl_layer = nn.Linear(in_f, out_f, bias=bias, seed=0)
    tt_layer = torch.nn.Linear(in_f, out_f, bias=bias)
    with torch.no_grad():
        tt_layer.weight.copy_(torch.tensor(gl_layer.weight.data.T.copy()))
        if bias:
            tt_layer.bias.copy_(torch.tensor(gl_layer.bias.data.copy()))
    return gl_layer, tt_layer


class TestLinearParity:
    @pytest.mark.parametrize("shape", [(5, 3), (2, 5, 3), (3,)])
    def test_forward_matches(self, shape):
        gl_layer, tt_layer = paired_linear(3, 4)
        x = RNG.standard_normal(shape)
        np.testing.assert_allclose(
            gl_layer(gl.Tensor(x.copy())).data,
            tt_layer(torch.tensor(x.copy())).detach().numpy(),
            **TOL,
        )

    def test_parameter_gradients_match(self):
        gl_layer, tt_layer = paired_linear(3, 4)
        x = RNG.standard_normal((6, 3))
        gl_layer(gl.Tensor(x.copy(), requires_grad=False)).sum().backward()
        tt_layer(torch.tensor(x.copy())).sum().backward()
        np.testing.assert_allclose(gl_layer.weight.grad.data, tt_layer.weight.grad.numpy().T, **TOL)
        np.testing.assert_allclose(gl_layer.bias.grad.data, tt_layer.bias.grad.numpy(), **TOL)

    def test_input_gradient_matches(self):
        gl_layer, tt_layer = paired_linear(3, 4)
        x = RNG.standard_normal((6, 3))
        gx = gl.Tensor(x.copy(), requires_grad=True)
        tx = torch.tensor(x.copy(), requires_grad=True)
        gl_layer(gx).sum().backward()
        tt_layer(tx).sum().backward()
        np.testing.assert_allclose(gx.grad.data, tx.grad.numpy(), **TOL)

    def test_without_bias(self):
        gl_layer, tt_layer = paired_linear(4, 2, bias=False)
        x = RNG.standard_normal((3, 4))
        np.testing.assert_allclose(
            gl_layer(gl.Tensor(x.copy())).data,
            tt_layer(torch.tensor(x.copy())).detach().numpy(),
            **TOL,
        )


class TestLossParity:
    def test_mse_matches(self):
        pred, target = RNG.standard_normal((8, 3)), RNG.standard_normal((8, 3))
        gp = gl.Tensor(pred.copy(), requires_grad=True)
        tp = torch.tensor(pred.copy(), requires_grad=True)
        gl_loss = nn.mse_loss(gp, gl.Tensor(target.copy()))
        tt_loss = torch.nn.functional.mse_loss(tp, torch.tensor(target.copy()))
        np.testing.assert_allclose(gl_loss.item(), tt_loss.item(), **TOL)
        gl_loss.backward()
        tt_loss.backward()
        np.testing.assert_allclose(gp.grad.data, tp.grad.numpy(), **TOL)

    def test_bce_with_logits_matches(self):
        logits = RNG.standard_normal(12) * 3
        targets = (RNG.random(12) > 0.5).astype(np.float64)
        gx = gl.Tensor(logits.copy(), requires_grad=True)
        tx = torch.tensor(logits.copy(), requires_grad=True)
        gl_loss = nn.binary_cross_entropy_with_logits(gx, gl.Tensor(targets.copy()))
        tt_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            tx, torch.tensor(targets.copy())
        )
        np.testing.assert_allclose(gl_loss.item(), tt_loss.item(), **TOL)
        gl_loss.backward()
        tt_loss.backward()
        np.testing.assert_allclose(gx.grad.data, tx.grad.numpy(), **TOL)

    def test_bce_with_logits_matches_at_extreme_logits(self):
        """Both implementations claim stability; check they agree where it matters."""
        logits = np.array([-800.0, 800.0, -50.0, 50.0, 0.0])
        targets = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        gx = gl.Tensor(logits.copy(), requires_grad=True)
        tx = torch.tensor(logits.copy(), requires_grad=True)
        gl_loss = nn.binary_cross_entropy_with_logits(gx, gl.Tensor(targets.copy()))
        tt_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            tx, torch.tensor(targets.copy())
        )
        np.testing.assert_allclose(gl_loss.item(), tt_loss.item(), rtol=1e-12)
        gl_loss.backward()
        tt_loss.backward()
        np.testing.assert_allclose(gx.grad.data, tx.grad.numpy(), atol=1e-14)

    def test_cross_entropy_with_class_indices_matches(self):
        logits = RNG.standard_normal((10, 5))
        labels = RNG.integers(0, 5, 10)
        gx = gl.Tensor(logits.copy(), requires_grad=True)
        tx = torch.tensor(logits.copy(), requires_grad=True)
        gl_loss = nn.cross_entropy_with_logits(gx, gl.Tensor(labels.copy()))
        tt_loss = torch.nn.functional.cross_entropy(tx, torch.tensor(labels.copy()))
        np.testing.assert_allclose(gl_loss.item(), tt_loss.item(), **TOL)
        gl_loss.backward()
        tt_loss.backward()
        np.testing.assert_allclose(gx.grad.data, tx.grad.numpy(), **TOL)

    def test_cross_entropy_matches_at_extreme_logits(self):
        logits = np.array([[1000.0, 1001.0, 999.0], [-900.0, -901.0, -899.0]])
        labels = np.array([1, 0])
        gx = gl.Tensor(logits.copy(), requires_grad=True)
        tx = torch.tensor(logits.copy(), requires_grad=True)
        gl_loss = nn.cross_entropy_with_logits(gx, gl.Tensor(labels.copy()))
        tt_loss = torch.nn.functional.cross_entropy(tx, torch.tensor(labels.copy()))
        np.testing.assert_allclose(gl_loss.item(), tt_loss.item(), rtol=1e-12)
        gl_loss.backward()
        tt_loss.backward()
        np.testing.assert_allclose(gx.grad.data, tx.grad.numpy(), atol=1e-14)


def _paired_mlp(sizes=(3, 6, 2)):
    gl_model = nn.Sequential(
        nn.Linear(sizes[0], sizes[1], seed=1),
        nn.Tanh(),
        nn.Linear(sizes[1], sizes[2], seed=2),
    )
    tt_model = torch.nn.Sequential(
        torch.nn.Linear(sizes[0], sizes[1]),
        torch.nn.Tanh(),
        torch.nn.Linear(sizes[1], sizes[2]),
    )
    with torch.no_grad():
        tt_model[0].weight.copy_(torch.tensor(gl_model[0].weight.data.T.copy()))
        tt_model[0].bias.copy_(torch.tensor(gl_model[0].bias.data.copy()))
        tt_model[2].weight.copy_(torch.tensor(gl_model[2].weight.data.T.copy()))
        tt_model[2].bias.copy_(torch.tensor(gl_model[2].bias.data.copy()))
    return gl_model, tt_model


def _assert_params_match(gl_model, tt_model, **tol):
    np.testing.assert_allclose(
        gl_model[0].weight.data, tt_model[0].weight.detach().numpy().T, **tol
    )
    np.testing.assert_allclose(gl_model[0].bias.data, tt_model[0].bias.detach().numpy(), **tol)
    np.testing.assert_allclose(
        gl_model[2].weight.data, tt_model[2].weight.detach().numpy().T, **tol
    )
    np.testing.assert_allclose(gl_model[2].bias.data, tt_model[2].bias.detach().numpy(), **tol)


class TestOptimizerParity:
    @pytest.mark.parametrize(
        "make_pair,tol",
        [
            (lambda g, t: (optim.SGD(g, lr=0.1), torch.optim.SGD(t, lr=0.1)), TOL),
            (
                lambda g, t: (
                    optim.SGD(g, lr=0.1, momentum=0.9),
                    torch.optim.SGD(t, lr=0.1, momentum=0.9),
                ),
                TOL,
            ),
            (
                lambda g, t: (
                    optim.SGD(g, lr=0.1, momentum=0.9, nesterov=True),
                    torch.optim.SGD(t, lr=0.1, momentum=0.9, nesterov=True),
                ),
                TOL,
            ),
            (lambda g, t: (optim.Adam(g, lr=0.01), torch.optim.Adam(t, lr=0.01)), TOL),
            (
                lambda g, t: (
                    optim.AdamW(g, lr=0.01, weight_decay=0.05),
                    torch.optim.AdamW(t, lr=0.01, weight_decay=0.05),
                ),
                TOL,
            ),
        ],
        ids=["sgd", "sgd_momentum", "sgd_nesterov", "adam", "adamw"],
    )
    def test_twenty_steps_stay_identical(self, make_pair, tol):
        """Twenty steps is long enough for any disagreement in the rule to show."""
        gl_model, tt_model = _paired_mlp()
        gl_opt, tt_opt = make_pair(gl_model.parameters(), tt_model.parameters())
        x = RNG.standard_normal((16, 3))
        y = RNG.standard_normal((16, 2))

        for _ in range(20):
            gl_opt.zero_grad()
            gl_loss = nn.mse_loss(gl_model(gl.Tensor(x.copy())), gl.Tensor(y.copy()))
            gl_loss.backward()
            gl_opt.step()

            tt_opt.zero_grad()
            tt_loss = torch.nn.functional.mse_loss(
                tt_model(torch.tensor(x.copy())), torch.tensor(y.copy())
            )
            tt_loss.backward()
            tt_opt.step()

            np.testing.assert_allclose(gl_loss.item(), tt_loss.item(), **tol)

        _assert_params_match(gl_model, tt_model, **tol)

    def test_sgd_weight_decay_matches(self):
        gl_model, tt_model = _paired_mlp()
        gl_opt = optim.SGD(gl_model.parameters(), lr=0.1, weight_decay=0.03)
        tt_opt = torch.optim.SGD(tt_model.parameters(), lr=0.1, weight_decay=0.03)
        x, y = RNG.standard_normal((8, 3)), RNG.standard_normal((8, 2))
        for _ in range(10):
            gl_opt.zero_grad()
            nn.mse_loss(gl_model(gl.Tensor(x.copy())), gl.Tensor(y.copy())).backward()
            gl_opt.step()
            tt_opt.zero_grad()
            torch.nn.functional.mse_loss(
                tt_model(torch.tensor(x.copy())), torch.tensor(y.copy())
            ).backward()
            tt_opt.step()
        _assert_params_match(gl_model, tt_model, **TOL)


class TestLearningCurveParity:
    def test_full_training_runs_produce_the_same_curve(self):
        """The headline comparison: identical setup, identical learning curve."""
        gl_model, tt_model = _paired_mlp((2, 8, 1))
        gl_opt = optim.Adam(gl_model.parameters(), lr=0.02)
        tt_opt = torch.optim.Adam(tt_model.parameters(), lr=0.02)

        from gradlab import data

        X, y = data.make_moons(n_samples=128, noise=0.2, seed=42)
        xb, yb = X.data, y.data

        gl_curve, tt_curve = [], []
        for _ in range(60):
            gl_opt.zero_grad()
            loss = nn.binary_cross_entropy_with_logits(
                gl_model(gl.Tensor(xb.copy())), gl.Tensor(yb.copy())
            )
            loss.backward()
            gl_opt.step()
            gl_curve.append(loss.item())

            tt_opt.zero_grad()
            t_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                tt_model(torch.tensor(xb.copy())), torch.tensor(yb.copy())
            )
            t_loss.backward()
            tt_opt.step()
            tt_curve.append(t_loss.item())

        np.testing.assert_allclose(gl_curve, tt_curve, rtol=1e-9, atol=1e-11)
        assert gl_curve[-1] < gl_curve[0] / 2, "and the run must actually have learned"
