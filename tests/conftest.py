"""Shared fixtures. PyTorch is optional and used only as an independent oracle."""

from __future__ import annotations

import numpy as np
import pytest

try:  # pragma: no cover - availability depends on the environment
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH,
    reason="PyTorch is not installed; install the 'oracle' extra to run conformance tests",
)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so every test is reproducible."""
    return np.random.default_rng(20261023)
