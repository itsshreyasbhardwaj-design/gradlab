"""Verify the concrete numeric claims the documentation makes.

The docs quote a lot of specific numbers. Numbers in prose drift silently as
code changes, and a reader has no way to tell a stale figure from a current one.
These tests recompute the load-bearing claims and fail if the documentation and
the engine disagree.

Each test names the file and section it is protecting.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import gradlab as gl
from gradlab import data, nn, optim, training

DOCS = Path(__file__).resolve().parents[2] / "docs"


def doc(name: str) -> str:
    return (DOCS / name).read_text()


class TestLessonsClaims:
    """docs/lessons.md"""

    def test_lesson_3_chain_rule_values(self):
        x = gl.Tensor(0.5, requires_grad=True)
        u = x * 3
        u.retain_grad()
        u.tanh().backward()
        text = doc("lessons.md")
        assert f"{u.grad.item():.10f}" in text, "lesson 3's dy/du is stale"
        assert f"{x.grad.item():.10f}" in text, "lesson 3's dy/dx is stale"

    def test_lesson_6_reverse_mode_gradient(self):
        x = gl.Tensor(3.0, requires_grad=True)
        y = x * x + 2 * x
        y.backward()
        assert y.item() == 15.0 and x.grad.item() == 8.0
        assert "f(3) = 15.0" in doc("lessons.md")
        assert "accumulated on x: [8.0]" in doc("lessons.md")

    def test_lesson_7_accumulation_across_calls(self):
        x = gl.Tensor(3.0, requires_grad=True)
        seen = []
        for _ in range(3):
            (x * x).backward()
            seen.append(x.grad.item())
        assert seen == [6.0, 12.0, 18.0]
        text = doc("lessons.md")
        for i, value in enumerate(seen, start=1):
            assert f"backward() call {i}: x.grad = {value}" in text

    def test_lesson_8_broadcast_bias_gradients(self):
        W = gl.Tensor(np.ones((4, 3)), requires_grad=True)
        b = gl.Tensor(np.zeros(3), requires_grad=True)
        (W + b).sum().backward()
        assert b.grad.tolist() == [4.0, 4.0, 4.0]

        layer = nn.Linear(2, 3, seed=0)
        layer(gl.ones(16, 2)).sum().backward()
        assert layer.bias.grad.data.tolist() == [16.0, 16.0, 16.0]

        text = doc("lessons.md")
        assert "[4.0, 4.0, 4.0]" in text
        assert "[16.0,\n16.0, 16.0]" in text or "[16.0, 16.0, 16.0]" in text

    def test_lesson_11_two_linear_layers_compose_to_one(self):
        a = nn.Linear(3, 5, seed=0)
        b = nn.Linear(5, 2, seed=1)
        x = gl.randn(4, 3, seed=2)
        stacked = b(a(x))
        equivalent = x @ (a.weight @ b.weight) + (a.bias @ b.weight + b.bias)
        difference = float(np.abs(stacked.data - equivalent.data).max())
        assert difference < 1e-15, difference
        assert "1.665e-16" in doc("lessons.md"), "lesson 11's composition figure is stale"

    def test_lesson_12_stability_threshold_is_exact(self):
        """The |1-2*lr| analysis: converges below 1, oscillates at 1, diverges above."""
        outcomes = {}
        for lr in (0.5, 1.0, 1.1):
            w = gl.Tensor(0.0, requires_grad=True)
            for _ in range(20):
                w.zero_grad()
                ((w - 3.0) ** 2).backward()
                with gl.no_grad():
                    w.data = w.data - lr * w.grad.data
            outcomes[lr] = w.item()
        assert outcomes[0.5] == pytest.approx(3.0, abs=1e-12), "lr=0.5 lands exactly"
        assert outcomes[1.0] == pytest.approx(0.0, abs=1e-12), "lr=1.0 oscillates"
        assert abs(outcomes[1.1]) > 100, "lr=1.1 diverges"
        text = doc("lessons.md")
        assert "-112.012800" in text or "−112.012800" in text

    def test_lesson_14_adam_first_step_table(self):
        for gradient, expected in [
            (0.001, "0.09999900"),
            (1.0, "0.10000000"),
            (1000.0, "0.10000000"),
        ]:
            p = nn.Parameter(gl.Tensor(np.array([0.0])))
            opt = optim.Adam([p], lr=0.1)
            p._grad = gl.Tensor(np.array([gradient]))
            opt.step()
            assert f"{abs(p.item()):.8f}" == expected, gradient
            assert expected in doc("lessons.md")


class TestValidationClaims:
    """docs/validation.md"""

    def test_quoted_test_total_matches_reality(self):
        """The inventory table's total must equal the sum of its rows."""
        text = doc("validation.md")
        rows = re.findall(r"^\| `?tests/\w+`? \| (\d+) \|", text, re.M)
        doctests = re.search(r"^\| doctests \| (\d+) \|", text, re.M)
        total = re.search(r"^\| \*\*total\*\* \| \*\*(\d+)\*\* \|", text, re.M)
        assert rows and doctests and total, "the inventory table has changed shape"
        assert sum(int(r) for r in rows) + int(doctests.group(1)) == int(total.group(1))

    def test_worst_gradcheck_error_claim(self):
        """The worst-case figure quoted must actually be the worst case."""
        from gradlab.ops.activation import LogSoftmax

        rng = np.random.default_rng(7)
        x = gl.Tensor(rng.uniform(-1.5, 1.5, (4, 5)), requires_grad=True)
        result = gl.gradcheck(
            lambda t: (LogSoftmax.apply(t, axis=-1) ** 2).sum(), x, check_kinks=False
        )
        assert result.passed
        assert result.max_abs_error < 1e-7, (
            f"log_softmax error {result.max_abs_error:.2e} exceeds the 1.04e-08 "
            "figure quoted in docs/validation.md"
        )


class TestExperimentClaims:
    """README and experiment write-ups"""

    def test_xor_is_unlearnable_without_a_nonlinearity(self):
        X, y = data.make_xor(n_samples=400, noise=0.0, seed=7)
        loader = data.DataLoader(data.TensorDataset(X, y), batch_size=32, shuffle=True, seed=7)
        accuracies = {}
        for nonlinear in (False, True):
            layers = [nn.Linear(2, 8, seed=7)]
            if nonlinear:
                layers.append(nn.Tanh())
            layers.append(nn.Linear(8, 1, seed=8))
            model = nn.Sequential(*layers)
            loader.reset_epochs()
            training.fit(
                model,
                optim.Adam(model.parameters(), lr=0.05),
                nn.binary_cross_entropy_with_logits,
                loader,
                epochs=400,
            )
            with gl.no_grad():
                accuracies[nonlinear] = nn.accuracy(model(X), y)
        assert accuracies[False] < 0.7, "an affine model must not solve XOR"
        assert accuracies[True] > 0.95, "one tanh must make it solvable"

    def test_linear_regression_reaches_the_noise_floor(self):
        """The README claims convergence to the irreducible noise variance."""
        X, y, truth = data.make_regression(
            n_samples=300, n_features=3, bias=1.5, noise=0.1, seed=42
        )
        Xtr, _, ytr, _ = data.train_test_split(X, y, test_size=0.25, seed=42)
        model = nn.Linear(3, 1, seed=42)
        history = training.fit(
            model,
            optim.SGD(model.parameters(), lr=0.05),
            nn.mse_loss,
            data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=Xtr.shape[0]),
            epochs=200,
        )
        noise_floor = truth["noise"] ** 2
        assert history.train_loss[-1] == pytest.approx(noise_floor, rel=0.25), (
            f"final loss {history.train_loss[-1]:.6f} vs noise floor {noise_floor:.6f}"
        )
