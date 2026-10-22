"""``gradlab.data`` — datasets, mini-batch loading, and reproducible synthetic data."""

from __future__ import annotations

from gradlab.data.dataloader import DataLoader
from gradlab.data.datasets import Dataset, TensorDataset, train_test_split
from gradlab.data.synthetic import (
    make_blobs,
    make_moons,
    make_regression,
    make_spiral,
    make_xor,
)

__all__ = [
    "DataLoader",
    "Dataset",
    "TensorDataset",
    "make_blobs",
    "make_moons",
    "make_regression",
    "make_spiral",
    "make_xor",
    "train_test_split",
]
