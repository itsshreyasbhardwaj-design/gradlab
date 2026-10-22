"""The optdigits loader, tested entirely offline.

The parsing is a pure function precisely so it can be verified without ever
reaching the network, and so the download path has nothing in it but I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradlab.data.optdigits import FILES, load_split, local_path, parse_optdigits


def make_row(pixels, label):
    return ",".join(str(v) for v in pixels) + f",{label}"


def valid_text(n=3):
    rng = np.random.default_rng(0)
    rows = [make_row(rng.integers(0, 17, 64).tolist(), i % 10) for i in range(n)]
    return "\n".join(rows)


class TestParsing:
    def test_shapes_and_dtypes(self):
        pixels, labels = parse_optdigits(valid_text(5))
        assert pixels.shape == (5, 64) and pixels.dtype == np.float64
        assert labels.shape == (5,) and np.issubdtype(labels.dtype, np.integer)

    def test_pixels_are_scaled_to_the_unit_interval(self):
        pixels, _ = parse_optdigits(make_row([16] * 64, 3))
        np.testing.assert_allclose(pixels, np.ones((1, 64)))
        pixels, _ = parse_optdigits(make_row([0] * 64, 3))
        np.testing.assert_allclose(pixels, np.zeros((1, 64)))

    def test_scaling_divides_by_sixteen(self):
        pixels, _ = parse_optdigits(make_row([8] + [0] * 63, 1))
        assert pixels[0, 0] == 0.5

    def test_labels_are_preserved(self):
        _, labels = parse_optdigits("\n".join(make_row([0] * 64, d) for d in range(10)))
        assert labels.tolist() == list(range(10))

    def test_blank_lines_are_ignored(self):
        pixels, _ = parse_optdigits("\n\n" + valid_text(2) + "\n\n")
        assert pixels.shape[0] == 2

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            parse_optdigits("   \n  ")

    def test_wrong_column_count_is_rejected(self):
        with pytest.raises(ValueError, match="65 columns"):
            parse_optdigits(make_row([0] * 10, 1))

    def test_non_integer_values_are_rejected(self):
        with pytest.raises(ValueError, match="comma-separated integers"):
            parse_optdigits(make_row(["x"] * 64, 1))

    def test_out_of_range_label_is_rejected(self):
        with pytest.raises(ValueError, match="Labels must be"):
            parse_optdigits(make_row([0] * 64, 42))

    def test_out_of_range_pixel_is_rejected(self):
        with pytest.raises(ValueError, match="0-16"):
            parse_optdigits(make_row([99] * 64, 1))


class TestLoading:
    def test_round_trips_through_a_file(self, tmp_path):
        (tmp_path / FILES["train"]).write_text(valid_text(4))
        pixels, labels = load_split(tmp_path, "train")
        assert pixels.shape == (4, 64) and labels.shape == (4,)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_split(tmp_path, "test")

    def test_unknown_split_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="split must be one of"):
            local_path(tmp_path, "validation")

    def test_loading_never_touches_the_network(self, tmp_path, monkeypatch):
        """Guard the policy: a plain load must not be able to open a connection."""
        import urllib.request

        def forbidden(*args, **kwargs):
            raise AssertionError("load_split attempted a network connection")

        monkeypatch.setattr(urllib.request, "urlopen", forbidden)
        (tmp_path / FILES["train"]).write_text(valid_text(2))
        load_split(tmp_path, "train")

    def test_parsed_data_trains_a_model(self, tmp_path):
        """The loader's output must be directly usable by the nn stack."""
        import gradlab as gl
        from gradlab import nn, optim

        rng = np.random.default_rng(1)
        rows = []
        for i in range(40):
            label = i % 10
            base = np.full(64, label, dtype=int)
            rows.append(make_row(np.clip(base + rng.integers(0, 3, 64), 0, 16).tolist(), label))
        (tmp_path / FILES["train"]).write_text("\n".join(rows))

        pixels, labels = load_split(tmp_path, "train")
        X, y = gl.Tensor(pixels, copy=False), gl.Tensor(labels, copy=False)
        model = nn.Sequential(nn.Linear(64, 16, seed=0), nn.ReLU(), nn.Linear(16, 10, seed=1))
        opt = optim.Adam(model.parameters(), lr=0.05)
        first = last = None
        for step in range(60):
            opt.zero_grad()
            loss = nn.cross_entropy_with_logits(model(X), y)
            loss.backward()
            opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        assert last < first / 2, f"loss went {first} -> {last}"
