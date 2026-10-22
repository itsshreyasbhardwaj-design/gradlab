"""Reproducible synthetic datasets for the documented experiments.

Every generator takes a ``seed`` and uses its own generator, so an experiment is
reproducible from its seed alone. These are GradLab's own implementations of
well-known toy datasets — no scikit-learn dependency — and each docstring states
the exact construction so a reader can reproduce the data by hand.
"""

from __future__ import annotations

import numpy as np

from gradlab._config import DEFAULT_DTYPE, default_rng
from gradlab.tensor import Tensor


def make_regression(
    n_samples: int = 100,
    n_features: int = 1,
    weights: np.ndarray | None = None,
    bias: float = 0.0,
    noise: float = 0.1,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, dict]:
    """A linear target with Gaussian noise: ``y = X w + b + eps``.

    Returns ``(X, y, truth)`` where ``truth`` holds the exact ``weights`` and
    ``bias`` used, so an experiment can report how close the fitted parameters
    came to the generating ones.
    """
    rng = default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    if weights is None:
        weights = rng.uniform(-3.0, 3.0, n_features)
    weights = np.asarray(weights, dtype=DEFAULT_DTYPE).reshape(n_features)
    y = X @ weights + bias + noise * rng.standard_normal(n_samples)
    truth = {"weights": weights.copy(), "bias": float(bias), "noise": float(noise)}
    return (
        Tensor(X.astype(DEFAULT_DTYPE), copy=False),
        Tensor(y.astype(DEFAULT_DTYPE).reshape(-1, 1), copy=False),
        truth,
    )


def make_xor(
    n_samples: int = 200, noise: float = 0.0, seed: int | None = None
) -> tuple[Tensor, Tensor]:
    """The XOR problem: label is 1 when exactly one coordinate is positive.

    With ``noise=0`` and ``n_samples=4`` this is the classic four-point truth
    table ``(0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0``. Larger ``n_samples`` draws
    points uniformly from ``[-1, 1]^2`` and labels by the sign pattern.

    XOR is the standard demonstration that a single linear layer is not enough:
    no straight line separates the two classes, so a network with no hidden
    nonlinearity cannot do better than chance. ``examples/xor.py`` runs both.
    """
    rng = default_rng(seed)
    if n_samples == 4 and noise == 0.0:
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0.0, 1.0, 1.0, 0.0])
    else:
        X = rng.uniform(-1.0, 1.0, (n_samples, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(DEFAULT_DTYPE)
        if noise:
            X = X + noise * rng.standard_normal(X.shape)
    return (
        Tensor(X.astype(DEFAULT_DTYPE), copy=False),
        Tensor(y.astype(DEFAULT_DTYPE).reshape(-1, 1), copy=False),
    )


def make_moons(
    n_samples: int = 200, noise: float = 0.1, seed: int | None = None
) -> tuple[Tensor, Tensor]:
    """Two interleaving half-circles — a nonlinearly separable binary problem.

    Construction (the standard one, implemented here directly): class 0 is the
    upper half-circle ``(cos t, sin t)`` for ``t`` in ``[0, pi]``; class 1 is
    that shape rotated and shifted to ``(1 - cos t, 0.5 - sin t)``. Isotropic
    Gaussian noise of standard deviation ``noise`` is then added.

    The two crescents interlock, so the decision boundary has to curve. A
    logistic regression tops out near 85% on it; a small MLP reaches the high
    90s. ``examples/two_moons.py`` reports both.
    """
    rng = default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    t_out = np.linspace(0, np.pi, n_out)
    t_in = np.linspace(0, np.pi, n_in)
    outer = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)
    inner = np.stack([1.0 - np.cos(t_in), 0.5 - np.sin(t_in)], axis=1)

    X = np.vstack([outer, inner])
    y = np.hstack([np.zeros(n_out), np.ones(n_in)])
    if noise:
        X = X + noise * rng.standard_normal(X.shape)

    order = rng.permutation(n_samples)
    return (
        Tensor(X[order].astype(DEFAULT_DTYPE), copy=False),
        Tensor(y[order].astype(DEFAULT_DTYPE).reshape(-1, 1), copy=False),
    )


def make_blobs(
    n_samples: int = 300,
    centers: int = 3,
    n_features: int = 2,
    cluster_std: float = 0.8,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Isotropic Gaussian clusters — a simple multiclass problem.

    Returns ``(X, y)`` with ``y`` as **integer class indices**, the convention
    :func:`~gradlab.nn.losses.cross_entropy_with_logits` expects.
    """
    rng = default_rng(seed)
    centroids = rng.uniform(-5.0, 5.0, (centers, n_features))
    labels = rng.integers(0, centers, n_samples)
    X = centroids[labels] + cluster_std * rng.standard_normal((n_samples, n_features))
    return (
        Tensor(X.astype(DEFAULT_DTYPE), copy=False),
        Tensor(labels.astype(np.int64), copy=False),
    )


def make_spiral(
    n_samples: int = 300, classes: int = 3, noise: float = 0.2, seed: int | None = None
):
    """Interleaved spiral arms — harder than two-moons, still 2-D and plottable."""
    rng = default_rng(seed)
    per_class = n_samples // classes
    arms: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for c in range(classes):
        r = np.linspace(0.1, 1.0, per_class)
        theta = np.linspace(c * 2 * np.pi / classes, c * 2 * np.pi / classes + 3.0, per_class)
        theta = theta + noise * rng.standard_normal(per_class)
        arms.append(np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1))
        labels.append(np.full(per_class, c))
    X = np.vstack(arms)
    y = np.hstack(labels)
    order = rng.permutation(len(y))
    return (
        Tensor(X[order].astype(DEFAULT_DTYPE), copy=False),
        Tensor(y[order].astype(np.int64), copy=False),
    )
