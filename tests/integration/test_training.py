"""End-to-end training: does a model actually learn, and reproducibly?"""

from __future__ import annotations

import numpy as np
import pytest

import gradlab as gl
from gradlab import data, nn, optim, training


def loader_for(X, y, batch_size=32, seed=0, shuffle=True):
    return data.DataLoader(
        data.TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, seed=seed
    )


class TestLinearRegression:
    def test_recovers_the_generating_parameters(self):
        X, y, truth = data.make_regression(
            n_samples=400, n_features=3, bias=1.5, noise=0.01, seed=1
        )
        model = nn.Linear(3, 1, seed=1)
        history = training.fit(
            model,
            optim.Adam(model.parameters(), lr=0.05),
            nn.mse_loss,
            loader_for(X, y, seed=1),
            epochs=150,
        )
        np.testing.assert_allclose(model.weight.data.reshape(-1), truth["weights"], atol=0.02)
        assert abs(model.bias.item() - truth["bias"]) < 0.02
        assert history.train_loss[-1] < history.train_loss[0] / 100

    def test_loss_decreases_monotonically_in_full_batch_gradient_descent(self):
        """With a small enough step, every full-batch step must not increase the loss."""
        X, y, _ = data.make_regression(n_samples=120, n_features=2, noise=0.05, seed=2)
        model = nn.Linear(2, 1, seed=2)
        history = training.fit(
            model,
            optim.SGD(model.parameters(), lr=0.01),
            nn.mse_loss,
            loader_for(X, y, batch_size=120, shuffle=False),
            epochs=60,
        )
        diffs = np.diff(history.train_loss)
        assert np.all(diffs <= 1e-9), f"loss increased at steps {np.nonzero(diffs > 1e-9)[0]}"


class TestXOR:
    def test_an_mlp_with_a_nonlinearity_learns_xor(self):
        X, y = data.make_xor(n_samples=400, seed=3)
        model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.Tanh(), nn.Linear(8, 1, seed=4))
        training.fit(
            model,
            optim.Adam(model.parameters(), lr=0.05),
            nn.binary_cross_entropy_with_logits,
            loader_for(X, y, seed=3),
            epochs=250,
        )
        with gl.no_grad():
            assert nn.accuracy(model(X), y) > 0.95

    def test_without_a_nonlinearity_xor_is_unlearnable(self):
        """Two stacked Linear layers compose to one; no hyperplane separates XOR."""
        X, y = data.make_xor(n_samples=400, seed=3)
        model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.Linear(8, 1, seed=4))
        training.fit(
            model,
            optim.Adam(model.parameters(), lr=0.05),
            nn.binary_cross_entropy_with_logits,
            loader_for(X, y, seed=3),
            epochs=250,
        )
        with gl.no_grad():
            accuracy = nn.accuracy(model(X), y)
        assert accuracy < 0.70, f"a purely affine model should not solve XOR (got {accuracy})"


class TestTwoMoons:
    @pytest.mark.slow
    def test_mlp_beats_a_linear_baseline_and_is_reproducible(self):
        X, y = data.make_moons(n_samples=400, noise=0.2, seed=4)
        Xtr, Xte, ytr, yte = data.train_test_split(X, y, test_size=0.25, seed=4)

        def run():
            model = nn.Sequential(nn.Linear(2, 16, seed=4), nn.Tanh(), nn.Linear(16, 1, seed=5))
            training.fit(
                model,
                optim.Adam(model.parameters(), lr=0.01),
                nn.binary_cross_entropy_with_logits,
                loader_for(Xtr, ytr, seed=4),
                epochs=150,
            )
            with gl.no_grad():
                return nn.accuracy(model(Xte), yte), model

        acc_a, _ = run()
        acc_b, _ = run()
        assert acc_a == acc_b, "same seeds must give the same result"

        baseline = nn.Linear(2, 1, seed=4)
        training.fit(
            baseline,
            optim.Adam(baseline.parameters(), lr=0.01),
            nn.binary_cross_entropy_with_logits,
            loader_for(Xtr, ytr, seed=4),
            epochs=150,
        )
        with gl.no_grad():
            base_acc = nn.accuracy(baseline(Xte), yte)
        # The claim worth testing is that curvature helps and the model is good —
        # not an arbitrary margin, which depends on the draw. Measured here:
        # MLP 0.98, linear 0.93.
        assert acc_a > base_acc, f"MLP {acc_a} did not beat linear {base_acc}"
        assert acc_a >= 0.95, f"MLP only reached {acc_a}"


class TestMulticlass:
    def test_blobs_are_classified_as_well_as_a_nearest_centroid_oracle(self):
        """Judge the model against what the data actually allows.

        These clusters overlap, so a fixed accuracy target would be a guess. The
        nearest-centroid classifier is the natural reference for isotropic
        Gaussian blobs, and it reaches 0.9267 here — the network should match it,
        not some number chosen in advance.
        """
        X, y = data.make_blobs(n_samples=300, centers=3, seed=6)
        points, labels = X.data, y.data
        centroids = np.stack([points[labels == c].mean(axis=0) for c in range(3)])
        oracle = np.argmin(((points[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)
        ceiling = float((oracle == labels).mean())

        model = nn.Sequential(nn.Linear(2, 16, seed=6), nn.ReLU(), nn.Linear(16, 3, seed=7))
        training.fit(
            model,
            optim.Adam(model.parameters(), lr=0.05),
            nn.cross_entropy_with_logits,
            loader_for(X, y, seed=6),
            epochs=80,
        )
        with gl.no_grad():
            accuracy = nn.accuracy(model(X), y)
        assert accuracy >= ceiling - 0.01, f"{accuracy} vs nearest-centroid {ceiling}"
        assert accuracy > 0.85, "and comfortably above the 1/3 chance rate"


class TestReproducibility:
    def test_two_identical_runs_agree_bit_for_bit(self):
        def run():
            X, y = data.make_moons(n_samples=200, noise=0.2, seed=8)
            model = nn.Sequential(nn.Linear(2, 8, seed=8), nn.Tanh(), nn.Linear(8, 1, seed=9))
            history = training.fit(
                model,
                optim.Adam(model.parameters(), lr=0.02),
                nn.binary_cross_entropy_with_logits,
                loader_for(X, y, seed=8),
                epochs=30,
            )
            return history.train_loss, model.state_dict()

        loss_a, state_a = run()
        loss_b, state_b = run()
        assert loss_a == loss_b
        for key in state_a:
            np.testing.assert_array_equal(state_a[key], state_b[key])

    def test_dataloader_shuffling_is_seed_reproducible(self):
        ds = data.TensorDataset(gl.arange(20).reshape(20, 1), gl.arange(20).reshape(20, 1))
        a = [b[0].data.reshape(-1).tolist() for b in data.DataLoader(ds, 4, shuffle=True, seed=1)]
        b = [b[0].data.reshape(-1).tolist() for b in data.DataLoader(ds, 4, shuffle=True, seed=1)]
        c = [b[0].data.reshape(-1).tolist() for b in data.DataLoader(ds, 4, shuffle=True, seed=2)]
        assert a == b and a != c

    def test_reset_epochs_restarts_the_shuffle_sequence(self):
        ds = data.TensorDataset(gl.arange(12).reshape(12, 1))
        loader = data.DataLoader(ds, 4, shuffle=True, seed=3)
        first = [b[0].data.reshape(-1).tolist() for b in loader]
        _ = list(loader)
        loader.reset_epochs()
        again = [b[0].data.reshape(-1).tolist() for b in loader]
        assert first == again


class TestCheckpointing:
    def test_training_can_be_stopped_and_resumed_exactly(self):
        X, y, _ = data.make_regression(n_samples=120, n_features=2, noise=0.05, seed=10)
        loader = loader_for(X, y, batch_size=120, shuffle=False)

        straight = nn.Linear(2, 1, seed=10)
        opt_s = optim.Adam(straight.parameters(), lr=0.05)
        training.fit(straight, opt_s, nn.mse_loss, loader, epochs=40)

        part = nn.Linear(2, 1, seed=10)
        opt_p = optim.Adam(part.parameters(), lr=0.05)
        training.fit(part, opt_p, nn.mse_loss, loader, epochs=20)
        model_state, opt_state = part.state_dict(), opt_p.state_dict()

        resumed = nn.Linear(2, 1, seed=99)
        resumed.load_state_dict(model_state)
        opt_r = optim.Adam(resumed.parameters(), lr=0.05)
        opt_r.load_state_dict(opt_state)
        training.fit(resumed, opt_r, nn.mse_loss, loader, epochs=20)

        np.testing.assert_allclose(straight.weight.data, resumed.weight.data, rtol=1e-10)


class TestTrainingSafety:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_a_diverging_run_is_stopped_with_an_explanation(self):
        """Overflow is expected here, so the suite-wide RuntimeWarning-as-error
        filter is lifted for this test only; that filter would otherwise fire on
        the overflow before GradLab's own non-finite check could run, and this
        test exists to prove that check works."""
        X, y, _ = data.make_regression(n_samples=100, n_features=4, noise=0.1, seed=11)
        model = nn.Sequential(nn.Linear(4, 32, seed=11), nn.ReLU(), nn.Linear(32, 1, seed=12))
        with pytest.raises(gl.GradLabError, match=r"non-finite objective|non-finite"):
            training.fit(
                model,
                optim.SGD(model.parameters(), lr=5.0),
                nn.mse_loss,
                loader_for(X, y, seed=11),
                epochs=60,
            )

    def test_evaluate_runs_in_eval_mode_and_restores_the_previous_mode(self):
        X, y = data.make_moons(n_samples=64, seed=13)
        model = nn.Sequential(
            nn.Linear(2, 8, seed=13), nn.Dropout(0.5, seed=13), nn.Linear(8, 1, seed=14)
        )
        model.train()
        loss, _ = training.evaluate(
            model, loader_for(X, y, shuffle=False), nn.binary_cross_entropy_with_logits
        )
        assert np.isfinite(loss)
        assert model.training, "evaluate must restore the mode it found"

    def test_evaluate_is_deterministic_despite_dropout(self):
        X, y = data.make_moons(n_samples=64, seed=15)
        model = nn.Sequential(
            nn.Linear(2, 8, seed=15), nn.Dropout(0.5, seed=15), nn.Linear(8, 1, seed=16)
        )
        loader = loader_for(X, y, shuffle=False)
        a, _ = training.evaluate(model, loader, nn.binary_cross_entropy_with_logits)
        b, _ = training.evaluate(model, loader, nn.binary_cross_entropy_with_logits)
        assert a == b

    def test_grad_clipping_caps_the_norm_and_reports_the_original(self):
        p = nn.Parameter(gl.Tensor([3.0, 4.0]))
        p._grad = gl.Tensor([3.0, 4.0])
        original = training.clip_grad_norm([p], 1.0)
        assert original == pytest.approx(5.0)
        assert training.global_grad_norm([p]) == pytest.approx(1.0)

    def test_clipping_below_the_threshold_changes_nothing(self):
        p = nn.Parameter(gl.Tensor([0.3, 0.4]))
        p._grad = gl.Tensor([0.3, 0.4])
        training.clip_grad_norm([p], 10.0)
        np.testing.assert_allclose(p._grad.data, [0.3, 0.4])

    def test_history_records_one_entry_per_epoch(self):
        X, y, _ = data.make_regression(n_samples=40, n_features=2, seed=17)
        model = nn.Linear(2, 1, seed=17)
        history = training.fit(
            model,
            optim.SGD(model.parameters(), lr=0.01),
            nn.mse_loss,
            loader_for(X, y, batch_size=40),
            epochs=7,
        )
        assert len(history.epochs) == len(history.train_loss) == len(history.grad_norm) == 7
        assert history.config["num_parameters"] == 3
