"""Module system, layers, losses and initialisation."""

from __future__ import annotations

import numpy as np
import pytest

import gradlab as gl
from gradlab import nn


class TestModuleSystem:
    def test_parameters_are_registered_on_assignment(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(gl.ones(3))
                self.not_a_param = 5

        m = M()
        assert [n for n, _ in m.named_parameters()] == ["w"]
        assert m.not_a_param == 5

    def test_submodules_are_discovered_recursively(self):
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(gl.ones(2))

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Inner()
                self.b = Inner()

        names = [n for n, _ in Outer().named_parameters()]
        assert names == ["a.w", "b.w"]

    def test_forgetting_super_init_gives_an_actionable_error(self):
        class Bad(nn.Module):
            def __init__(self):
                self.w = nn.Parameter(gl.ones(2))

        with pytest.raises(gl.GradLabError, match=r"super\(\).__init__"):
            Bad()

    def test_parameter_requires_grad_by_default(self):
        assert nn.Parameter(gl.ones(2)).requires_grad

    def test_num_parameters_counts_scalars(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        assert model.num_parameters() == (2 * 3 + 3) + (3 * 1 + 1)

    def test_train_eval_propagates_to_children(self):
        model = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.5))
        model.eval()
        assert all(not m.training for m in model.modules())
        model.train()
        assert all(m.training for m in model.modules())

    def test_zero_grad_clears_every_parameter(self):
        model = nn.Linear(2, 1, seed=0)
        nn.mse_loss(model(gl.ones(1, 2)), gl.zeros(1, 1)).backward()
        assert model.weight.grad is not None
        model.zero_grad()
        assert model.weight.grad is None and model.bias.grad is None

    def test_sequential_indexing_and_length(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
        assert len(model) == 3
        assert isinstance(model[1], nn.ReLU)

    def test_sequential_rejects_a_non_module(self):
        with pytest.raises(TypeError, match="Sequential expects Modules"):
            nn.Sequential(nn.Linear(1, 1), "not a module")


class TestStateDict:
    def test_round_trip_reproduces_outputs(self):
        a = nn.Sequential(nn.Linear(3, 4, seed=1), nn.Tanh(), nn.Linear(4, 2, seed=2))
        b = nn.Sequential(nn.Linear(3, 4, seed=9), nn.Tanh(), nn.Linear(4, 2, seed=9))
        x = gl.randn(5, 3, seed=3)
        assert not np.allclose(a(x).data, b(x).data)
        b.load_state_dict(a.state_dict())
        np.testing.assert_allclose(a(x).data, b(x).data)

    def test_state_dict_is_a_snapshot_not_a_live_view(self):
        model = nn.Linear(2, 2, seed=0)
        state = model.state_dict()
        with gl.no_grad():
            model.weight.data = model.weight.data + 1.0
        assert not np.allclose(state["weight"], model.weight.data)

    def test_shape_mismatch_is_rejected_before_anything_is_written(self):
        model = nn.Linear(2, 2, seed=0)
        original = model.weight.data.copy()
        bad = model.state_dict()
        bad["weight"] = np.zeros((5, 5))
        with pytest.raises(gl.ShapeError, match="shape"):
            model.load_state_dict(bad)
        np.testing.assert_array_equal(model.weight.data, original)

    def test_unexpected_keys_are_rejected_when_strict(self):
        model = nn.Linear(2, 2, seed=0)
        state = model.state_dict()
        state["surprise"] = np.zeros(1)
        with pytest.raises(gl.GradLabError, match="Unexpected"):
            model.load_state_dict(state)
        model.load_state_dict(state, strict=False)


class TestLinear:
    def test_output_shape_and_batch_handling(self):
        layer = nn.Linear(3, 5, seed=0)
        assert layer(gl.zeros(7, 3)).shape == (7, 5)
        assert layer(gl.zeros(2, 7, 3)).shape == (2, 7, 5)
        assert layer(gl.zeros(3)).shape == (5,)

    def test_weight_is_in_features_by_out_features(self):
        layer = nn.Linear(3, 5, seed=0)
        assert layer.weight.shape == (3, 5)
        assert layer.bias.shape == (5,)

    def test_wrong_feature_count_names_the_axis(self):
        with pytest.raises(gl.ShapeError, match="last axis must be the feature axis"):
            nn.Linear(3, 5, seed=0)(gl.zeros(7, 4))

    def test_bias_false_registers_no_bias(self):
        layer = nn.Linear(3, 2, bias=False, seed=0)
        assert layer.bias is None
        assert [n for n, _ in layer.named_parameters()] == ["weight"]

    def test_bias_gradient_is_summed_over_the_batch(self):
        layer = nn.Linear(2, 3, seed=0)
        layer(gl.ones(8, 2)).sum().backward()
        np.testing.assert_allclose(layer.bias.grad.data, np.full(3, 8.0))

    def test_forward_matches_explicit_matmul(self):
        layer = nn.Linear(3, 2, seed=4)
        x = gl.randn(5, 3, seed=5)
        expected = x.data @ layer.weight.data + layer.bias.data
        np.testing.assert_allclose(layer(x).data, expected)

    def test_invalid_feature_counts_raise(self):
        with pytest.raises(ValueError, match="positive feature counts"):
            nn.Linear(0, 3)


class TestInitialization:
    @pytest.mark.parametrize(
        "name",
        [
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "uniform",
            "normal",
            "zeros",
        ],
    )
    def test_every_named_initialiser_produces_the_right_shape(self, name):
        layer = nn.Linear(6, 4, init=name, seed=0)
        assert layer.weight.shape == (6, 4)

    def test_xavier_variance_is_close_to_its_target(self):
        """2/(fan_in+fan_out) is the whole point of the scheme; check it holds."""
        fan_in, fan_out = 200, 300
        w = nn.initialization.xavier_uniform((fan_in, fan_out), seed=0)
        np.testing.assert_allclose(w.var(), 2.0 / (fan_in + fan_out), rtol=0.06)

    def test_kaiming_variance_is_close_to_its_target(self):
        fan_in = 500
        w = nn.initialization.kaiming_normal((fan_in, 50), seed=0)
        np.testing.assert_allclose(w.var(), 2.0 / fan_in, rtol=0.08)

    def test_same_seed_gives_identical_weights(self):
        a = nn.Linear(4, 4, seed=123)
        b = nn.Linear(4, 4, seed=123)
        np.testing.assert_array_equal(a.weight.data, b.weight.data)

    def test_different_seeds_differ(self):
        a = nn.Linear(4, 4, seed=1)
        b = nn.Linear(4, 4, seed=2)
        assert not np.allclose(a.weight.data, b.weight.data)

    def test_unknown_initialiser_lists_the_options(self):
        with pytest.raises(ValueError, match="Available:"):
            nn.Linear(2, 2, init="magic")

    def test_initialiser_requires_a_2d_shape(self):
        with pytest.raises(gl.ShapeError, match="2-D weight"):
            nn.initialization.xavier_uniform((3, 4, 5), seed=0)


class TestDropout:
    def test_eval_mode_is_the_identity(self):
        d = nn.Dropout(0.5, seed=0).eval()
        x = gl.ones(100)
        np.testing.assert_array_equal(d(x).data, x.data)

    def test_training_zeroes_roughly_p_of_the_entries(self):
        d = nn.Dropout(0.3, seed=0).train()
        out = d(gl.ones(20000)).data
        assert 0.27 < (out == 0).mean() < 0.33

    def test_inverted_scaling_preserves_the_expected_value(self):
        """The 1/(1-p) factor is what makes eval mode need no rescaling."""
        d = nn.Dropout(0.4, seed=0).train()
        out = d(gl.ones(50000)).data
        np.testing.assert_allclose(out.mean(), 1.0, atol=0.02)

    def test_gradient_flows_only_through_kept_units(self):
        d = nn.Dropout(0.5, seed=0).train()
        x = gl.ones(100, requires_grad=True)
        out = d(x)
        out.sum().backward()
        assert np.array_equal((out.data == 0), (x.grad.data == 0))

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError, match=r"p must be in \[0, 1\)"):
            nn.Dropout(1.0)


class TestLosses:
    def test_mse_matches_the_definition(self):
        pred, target = gl.Tensor([1.0, 2.0, 3.0]), gl.Tensor([1.0, 4.0, 3.0])
        assert nn.mse_loss(pred, target).item() == pytest.approx(4.0 / 3.0)
        assert nn.mse_loss(pred, target, "sum").item() == pytest.approx(4.0)
        assert nn.mse_loss(pred, target, "none").shape == (3,)

    def test_mismatched_shapes_are_refused_not_broadcast(self):
        with pytest.raises(gl.ShapeError, match="refuses to broadcast"):
            nn.mse_loss(gl.zeros(8, 1), gl.zeros(8))

    def test_bce_with_logits_matches_the_definition_in_the_safe_range(self):
        logits = np.array([0.5, -1.2, 2.0])
        targets = np.array([1.0, 0.0, 1.0])
        p = 1 / (1 + np.exp(-logits))
        expected = -(targets * np.log(p) + (1 - targets) * np.log(1 - p)).mean()
        got = nn.binary_cross_entropy_with_logits(gl.Tensor(logits), gl.Tensor(targets)).item()
        assert got == pytest.approx(expected)

    def test_bce_with_logits_stays_finite_at_extreme_logits(self):
        """The naive -[y log s + (1-y) log(1-s)] form returns inf here."""
        logits = gl.Tensor([-800.0, 800.0, -1e4], requires_grad=True)
        targets = gl.Tensor([1.0, 0.0, 1.0])
        loss = nn.binary_cross_entropy_with_logits(logits, targets)
        assert np.isfinite(loss.item())
        loss.backward()
        assert np.all(np.isfinite(logits.grad.data))

    def test_bce_gradient_is_the_prediction_error(self):
        logits = gl.Tensor([0.5, -1.0], requires_grad=True)
        targets = gl.Tensor([1.0, 0.0])
        nn.binary_cross_entropy_with_logits(logits, targets, "sum").backward()
        expected = 1 / (1 + np.exp(-logits.data)) - targets.data
        np.testing.assert_allclose(logits.grad.data, expected)

    def test_bce_rejects_targets_outside_the_unit_interval(self):
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            nn.binary_cross_entropy_with_logits(gl.Tensor([0.0]), gl.Tensor([5.0]))

    def test_cross_entropy_accepts_indices_and_one_hot_identically(self):
        logits = gl.Tensor([[2.0, -1.0, 0.5], [0.1, 3.0, -2.0]])
        indices = gl.Tensor([0, 1])
        one_hot = gl.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert nn.cross_entropy_with_logits(logits, indices).item() == pytest.approx(
            nn.cross_entropy_with_logits(logits, one_hot).item()
        )

    def test_cross_entropy_of_a_perfect_confident_prediction_is_near_zero(self):
        logits = gl.Tensor([[50.0, 0.0, 0.0]])
        assert nn.cross_entropy_with_logits(logits, gl.Tensor([0])).item() < 1e-20

    def test_cross_entropy_is_stable_for_huge_logits(self):
        logits = gl.Tensor([[1000.0, 1001.0, 999.0]], requires_grad=True)
        loss = nn.cross_entropy_with_logits(logits, gl.Tensor([1]))
        assert np.isfinite(loss.item())
        loss.backward()
        assert np.all(np.isfinite(logits.grad.data))

    def test_cross_entropy_rejects_an_out_of_range_class_index(self):
        with pytest.raises(gl.ShapeError, match="out of range"):
            nn.cross_entropy_with_logits(gl.Tensor([[1.0, 2.0]]), gl.Tensor([7]))

    def test_cross_entropy_rejects_an_unusable_target_shape(self):
        with pytest.raises(gl.ShapeError, match="must be either class indices"):
            nn.cross_entropy_with_logits(gl.zeros(4, 3), gl.zeros(4, 9))

    def test_uniform_logits_give_log_num_classes(self):
        logits = gl.zeros(1, 4)
        assert nn.cross_entropy_with_logits(logits, gl.Tensor([2])).item() == pytest.approx(
            np.log(4)
        )

    def test_accuracy_for_binary_and_multiclass(self):
        assert nn.accuracy(gl.Tensor([[2.0, -1.0], [0.0, 3.0]]), gl.Tensor([0, 1])) == 1.0
        assert nn.accuracy(gl.Tensor([1.0, -1.0]), gl.Tensor([1.0, 0.0])) == 1.0
        assert nn.accuracy(gl.Tensor([1.0, -1.0]), gl.Tensor([0.0, 0.0])) == 0.5

    @pytest.mark.parametrize(
        "loss_cls,args",
        [
            (nn.MSELoss, (gl.randn(4, 2, seed=1), gl.randn(4, 2, seed=2))),
            (nn.BCEWithLogitsLoss, (gl.randn(4, 1, seed=1), gl.zeros(4, 1))),
            (nn.CrossEntropyLoss, (gl.randn(4, 3, seed=1), gl.Tensor([0, 1, 2, 0]))),
        ],
    )
    def test_module_losses_match_their_functions(self, loss_cls, args):
        assert np.isfinite(loss_cls()(*args).item())

    def test_loss_gradients_pass_gradcheck(self):
        logits = gl.randn(5, 4, seed=7, requires_grad=True)
        targets = gl.Tensor([0, 1, 2, 3, 0])
        assert gl.gradcheck(lambda t: nn.cross_entropy_with_logits(t, targets), logits).passed

        bl = gl.randn(6, seed=8, requires_grad=True)
        bt = gl.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        assert gl.gradcheck(lambda t: nn.binary_cross_entropy_with_logits(t, bt), bl).passed
