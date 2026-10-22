"""Dataset abstraction: an indexable, length-aware collection of samples."""

from __future__ import annotations

from typing import Any

import numpy as np

from gradlab._errors import ShapeError
from gradlab.tensor import Tensor, as_tensor


class Dataset:
    """Minimal dataset protocol: ``__len__`` and ``__getitem__``.

    Subclass it to wrap your own data. ``__getitem__`` receives either a single
    integer or an array of indices and must return a tuple of tensors; handling
    arrays is what lets :class:`~gradlab.data.dataloader.DataLoader` fetch a
    whole batch in one slice instead of per-sample.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: Any) -> tuple[Tensor, ...]:
        raise NotImplementedError


class TensorDataset(Dataset):
    """Wrap tensors (or arrays) that share a first dimension.

    >>> import gradlab as gl
    >>> from gradlab.data import TensorDataset
    >>> ds = TensorDataset(gl.zeros(10, 3), gl.ones(10))
    >>> len(ds), [t.shape for t in ds[0:4]]
    (10, [(4, 3), (4,)])
    """

    def __init__(self, *tensors: Any) -> None:
        if not tensors:
            raise ValueError("TensorDataset needs at least one tensor.")
        self.tensors = [as_tensor(t) for t in tensors]
        lengths = {t.shape[0] if t.ndim else 1 for t in self.tensors}
        if len(lengths) != 1:
            raise ShapeError(
                "Every tensor in a TensorDataset must share its first dimension; got "
                f"lengths {sorted(lengths)} for shapes {[t.shape for t in self.tensors]}."
            )
        self._length = lengths.pop()

    def __len__(self) -> int:
        return int(self._length)

    def __getitem__(self, index: Any) -> tuple[Tensor, ...]:
        idx = np.asarray(index) if isinstance(index, (list, np.ndarray)) else index
        return tuple(Tensor(t.data[idx], copy=False, _internal=True) for t in self.tensors)

    def __repr__(self) -> str:
        return f"TensorDataset(n={len(self)}, shapes={[t.shape for t in self.tensors]})"


def train_test_split(
    *tensors: Any,
    test_size: float = 0.2,
    seed: int | None = None,
    shuffle: bool = True,
) -> tuple[Tensor, ...]:
    """Split tensors into train and test parts along the first axis.

    Returns ``(a_train, a_test, b_train, b_test, ...)`` in input order. The
    split is a permutation of indices drawn from a seeded generator, so a given
    ``seed`` always produces the same partition — and the same partition for
    every tensor passed, which is what keeps features aligned with labels.
    """
    parts = [as_tensor(t) for t in tensors]
    n = parts[0].shape[0]
    for t in parts:
        if t.shape[0] != n:
            raise ShapeError(
                f"All tensors must share their first dimension; got {t.shape[0]} vs {n}."
            )
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size}.")

    indices = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    cut = int(round(n * (1.0 - test_size)))
    if cut == 0 or cut == n:
        raise ValueError(
            f"test_size={test_size} on {n} samples leaves one side empty. Use more "
            "samples or a less extreme split."
        )
    train_idx, test_idx = indices[:cut], indices[cut:]

    out: list[Tensor] = []
    for t in parts:
        out.append(Tensor(t.data[train_idx], copy=True))
        out.append(Tensor(t.data[test_idx], copy=True))
    return tuple(out)
