"""Mini-batch iteration with reproducible shuffling."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from gradlab._config import default_rng
from gradlab.data.datasets import Dataset
from gradlab.tensor import Tensor


class DataLoader:
    """Iterate a :class:`~gradlab.data.datasets.Dataset` in mini-batches.

    Why mini-batches at all: the gradient of the full-dataset loss is the mean
    of the per-sample gradients, so a batch gives an unbiased estimate of it at
    a fraction of the cost. Smaller batches mean noisier estimates — which is
    not purely a drawback, since the noise helps escape sharp minima — and more
    parameter updates per epoch.

    Parameters
    ----------
    dataset
        Anything implementing ``__len__`` and ``__getitem__``.
    batch_size
        Samples per batch.
    shuffle
        Reshuffle the order every epoch (recommended for training; pointless and
        wasteful for evaluation).
    drop_last
        Discard a final short batch. Useful when a layer needs a fixed batch
        size; off by default so no data is silently dropped.
    seed
        Seed for the shuffling generator. The generator is owned by this loader,
        so two loaders with the same seed shuffle identically and neither is
        affected by anything else in the process.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self._rng = default_rng(seed)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        n = len(self.dataset)
        order = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(order)
        for start in range(0, n, self.batch_size):
            batch = order[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                return
            yield self.dataset[batch]

    def reset_epochs(self) -> None:
        """Restart the shuffling sequence from the original seed.

        Needed for exact reproducibility when re-running a training loop in the
        same process: the generator advances every epoch, so a second ``fit()``
        would otherwise see a different batch order.
        """
        self._rng = default_rng(self.seed)

    def __repr__(self) -> str:
        return (
            f"DataLoader(n={len(self.dataset)}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, batches={len(self)})"
        )
