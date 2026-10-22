"""Optimiser update rules, state, and refusal behaviour.

Where possible the expected value is computed by hand in the test from the
published update rule, so a passing test means the implementation matches the
mathematics — not merely that it matches itself.
"""

from __future__ import annotations

import numpy as np
import pytest

import gradlab as gl
from gradlab import nn, optim


def param(value, requires_grad=True):
    return nn.Parameter(gl.Tensor(np.asarray(value, dtype=float)), requires_grad=requires_grad)


def set_grad(p, value):
    p._grad = gl.Tensor(np.asarray(value, dtype=float))


class TestSGD:
    def test_plain_update(self):
        p = param([1.0, 2.0])
        opt = optim.SGD([p], lr=0.1)
        set_grad(p, [1.0, -1.0])
        opt.step()
        np.testing.assert_allclose(p.data, [0.9, 2.1])

    def test_momentum_accumulates_a_velocity(self):
        """v <- mu*v + g, theta <- theta - lr*v, computed by hand for 3 steps."""
        p = param([1.0])
        opt = optim.SGD([p], lr=0.1, momentum=0.9)
        velocity = 0.0
        expected = 1.0
        for _ in range(3):
            set_grad(p, [1.0])
            opt.step()
            velocity = 0.9 * velocity + 1.0
            expected -= 0.1 * velocity
        np.testing.assert_allclose(p.data, [expected])

    def test_momentum_increases_the_effective_step(self):
        a, b = param([0.0]), param([0.0])
        oa = optim.SGD([a], lr=0.1)
        ob = optim.SGD([b], lr=0.1, momentum=0.9)
        for _ in range(10):
            set_grad(a, [1.0])
            set_grad(b, [1.0])
            oa.step()
            ob.step()
        assert abs(b.item()) > abs(a.item()) * 3

    def test_nesterov_differs_from_plain_momentum(self):
        a, b = param([1.0]), param([1.0])
        oa = optim.SGD([a], lr=0.1, momentum=0.9)
        ob = optim.SGD([b], lr=0.1, momentum=0.9, nesterov=True)
        for _ in range(3):
            set_grad(a, [1.0])
            set_grad(b, [1.0])
            oa.step()
            ob.step()
        assert a.item() != b.item()

    def test_weight_decay_adds_to_the_gradient(self):
        p = param([2.0])
        opt = optim.SGD([p], lr=0.1, weight_decay=0.5)
        set_grad(p, [1.0])
        opt.step()
        np.testing.assert_allclose(p.data, [2.0 - 0.1 * (1.0 + 0.5 * 2.0)])

    def test_zero_gradient_leaves_the_parameter_alone(self):
        p = param([3.0])
        opt = optim.SGD([p], lr=0.5)
        set_grad(p, [0.0])
        opt.step()
        assert p.item() == 3.0

    def test_nesterov_without_momentum_is_rejected(self):
        with pytest.raises(ValueError, match="nesterov momentum requires"):
            optim.SGD([param([1.0])], lr=0.1, nesterov=True)


class TestAdam:
    def test_first_step_is_lr_up_to_the_epsilon_term(self):
        """Bias correction makes the t=1 step essentially lr, whatever the gradient.

        At t=1, m_hat = g and v_hat = g^2 exactly, so the update collapses to
        ``lr * g / (|g| + eps)``. That is ``lr`` to within ``eps/|g|`` — exact in
        the limit, and measurably short of it for a gradient near eps. The test
        asserts the exact expression and then checks the approximation error is
        where the algebra says it should be, rather than papering over it with a
        loose tolerance.
        """
        lr, eps = 0.1, 1e-8
        for gradient in (0.001, 1.0, 1000.0):
            p = param([0.0])
            opt = optim.Adam([p], lr=lr, eps=eps)
            set_grad(p, [gradient])
            opt.step()
            exact = lr * gradient / (abs(gradient) + eps)
            assert abs(p.item()) == pytest.approx(exact, rel=1e-12), f"gradient={gradient}"
            relative_shortfall = (lr - abs(p.item())) / lr
            assert relative_shortfall == pytest.approx(eps / (abs(gradient) + eps), rel=1e-6)

    def test_matches_a_hand_computed_three_step_trajectory(self):
        beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.1
        p = param([1.0])
        opt = optim.Adam([p], lr=lr, betas=(beta1, beta2), eps=eps)
        m = v = 0.0
        theta = 1.0
        for t in range(1, 4):
            g = 0.5 * t
            set_grad(p, [g])
            opt.step()
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            theta -= lr * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
        np.testing.assert_allclose(p.data, [theta], rtol=1e-12)

    def test_moments_are_stored_per_parameter(self):
        a, b = param([1.0]), param([1.0, 2.0])
        opt = optim.Adam([a, b], lr=0.1)
        set_grad(a, [1.0])
        set_grad(b, [1.0, 1.0])
        opt.step()
        assert opt.state[id(a)]["m"].shape == (1,)
        assert opt.state[id(b)]["m"].shape == (2,)

    def test_bias_correction_factors_are_as_published(self):
        p = param([1.0])
        opt = optim.Adam([p], lr=0.1)
        for _ in range(5):
            set_grad(p, [1.0])
            opt.step()
        t = opt.state[id(p)]["t"]
        assert t == 5
        assert (1 - 0.9**t) == pytest.approx(0.40951)

    def test_adam_is_scale_invariant_in_the_gradient(self):
        """Dividing by sqrt(v) means a uniformly rescaled gradient gives the same path."""
        a, b = param([1.0]), param([1.0])
        oa, ob = optim.Adam([a], lr=0.01), optim.Adam([b], lr=0.01)
        for i in range(8):
            g = 0.3 + 0.1 * i
            set_grad(a, [g])
            set_grad(b, [g * 1000.0])
            oa.step()
            ob.step()
        np.testing.assert_allclose(a.data, b.data, rtol=1e-6)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"betas": (1.0, 0.999)}, "beta1"),
            ({"betas": (0.9, -0.1)}, "beta2"),
            ({"eps": 0.0}, "eps must be positive"),
            ({"weight_decay": -1.0}, "weight_decay"),
        ],
    )
    def test_invalid_hyperparameters_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            optim.Adam([param([1.0])], lr=0.1, **kwargs)


class TestAdamW:
    def test_decoupled_decay_differs_from_coupled(self):
        a, w = param([1.0]), param([1.0])
        oa = optim.Adam([a], lr=0.1, weight_decay=0.1)
        ow = optim.AdamW([w], lr=0.1, weight_decay=0.1)
        set_grad(a, [2.0])
        set_grad(w, [2.0])
        oa.step()
        ow.step()
        assert a.item() != w.item()

    def test_decay_term_is_exactly_lr_times_wd_times_theta(self):
        """AdamW subtracts lr*wd*theta on top of the adaptive step, unscaled."""
        plain, decayed = param([3.0]), param([3.0])
        op = optim.AdamW([plain], lr=0.1, weight_decay=0.0)
        od = optim.AdamW([decayed], lr=0.1, weight_decay=0.2)
        set_grad(plain, [1.0])
        set_grad(decayed, [1.0])
        op.step()
        od.step()
        assert (plain.item() - decayed.item()) == pytest.approx(0.1 * 0.2 * 3.0)

    def test_shrinks_an_unused_parameter_toward_zero(self):
        p = param([5.0])
        opt = optim.AdamW([p], lr=0.1, weight_decay=0.5)
        for _ in range(20):
            set_grad(p, [0.0])
            opt.step()
        assert 0.0 < p.item() < 5.0


class TestOptimizerBase:
    def test_frozen_parameters_are_never_updated(self):
        frozen = param([5.0], requires_grad=False)
        opt = optim.SGD([frozen], lr=1.0)
        set_grad(frozen, [100.0])
        opt.step()
        assert frozen.item() == 5.0

    def test_a_parameter_with_no_gradient_is_skipped(self):
        used, unused = param([1.0]), param([7.0])
        opt = optim.SGD([used, unused], lr=0.1)
        set_grad(used, [1.0])
        opt.step()
        assert unused.item() == 7.0 and used.item() == pytest.approx(0.9)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_gradients_raise_and_preserve_the_parameter(self, bad):
        p = param([1.0])
        opt = optim.SGD([p], lr=0.1)
        set_grad(p, [bad])
        with pytest.raises(gl.GradLabError, match="non-finite"):
            opt.step()
        assert p.item() == 1.0

    def test_non_finite_can_be_opted_out_of(self):
        p = param([1.0])
        opt = optim.SGD([p], lr=0.1, error_on_nonfinite_grad=False)
        set_grad(p, [float("nan")])
        opt.step()
        assert np.isnan(p.item())

    def test_empty_parameter_list_is_an_error(self):
        with pytest.raises(gl.GradLabError, match="no parameters"):
            optim.SGD([], lr=0.1)

    def test_non_leaf_parameter_is_rejected(self):
        x = gl.Tensor([1.0], requires_grad=True)
        with pytest.raises(gl.GradLabError, match="not a leaf"):
            optim.SGD([x * 2], lr=0.1)

    @pytest.mark.parametrize("lr", [-0.1, float("nan"), float("inf")])
    def test_invalid_learning_rate_is_rejected(self, lr):
        with pytest.raises(ValueError, match="Learning rate"):
            optim.SGD([param([1.0])], lr=lr)

    def test_set_lr_validates_too(self):
        opt = optim.SGD([param([1.0])], lr=0.1)
        opt.set_lr(0.5)
        assert opt.lr == 0.5
        with pytest.raises(ValueError):
            opt.set_lr(-1.0)

    def test_zero_grad_clears_every_parameter(self):
        a, b = param([1.0]), param([1.0])
        opt = optim.SGD([a, b], lr=0.1)
        set_grad(a, [1.0])
        set_grad(b, [1.0])
        opt.zero_grad()
        assert a.grad is None and b.grad is None

    def test_state_dict_round_trip_reproduces_the_trajectory(self):
        def run(resume_at=None):
            p = param([1.0])
            opt = optim.Adam([p], lr=0.1)
            saved = None
            for i in range(6):
                if resume_at is not None and i == resume_at:
                    saved = (p.data.copy(), opt.state_dict())
                    p2 = param(saved[0])
                    opt2 = optim.Adam([p2], lr=0.1)
                    opt2.load_state_dict(saved[1])
                    p, opt = p2, opt2
                set_grad(p, [0.5 + 0.1 * i])
                opt.step()
            return p.item()

        assert run() == pytest.approx(run(resume_at=3), rel=1e-12)

    def test_an_update_bumps_the_version_counter(self):
        """This is what makes a stale-gradient second backward detectable."""
        p = param([1.0])
        before = p._version
        opt = optim.SGD([p], lr=0.1)
        set_grad(p, [1.0])
        opt.step()
        assert p._version > before

    def test_update_does_not_create_graph_history(self):
        p = param([1.0])
        opt = optim.SGD([p], lr=0.1)
        set_grad(p, [1.0])
        opt.step()
        assert p.is_leaf and p.grad_fn is None
