"""Loader for the UCI *Optical Recognition of Handwritten Digits* dataset.

Provenance and licence
----------------------
Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits*.
UCI Machine Learning Repository. Licensed **CC BY 4.0**.
https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

Each sample is an 8x8 image of a handwritten digit: 64 integers in 0-16,
followed by a class label 0-9. The official split is 3823 training and 1797
test samples.

Network policy
--------------
GradLab has **no network dependency**. :func:`load_split` reads a local file and
raises :class:`FileNotFoundError` if it is absent. :func:`download_split` is the
only function here that touches the network, it is never called implicitly, and
it writes only to the directory it is given. The parsing is a separate pure
function so it can be — and is — tested with no network access at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits"
FILES = {"train": "optdigits.tra", "test": "optdigits.tes"}
APPROX_SIZES = {"train": "~530 KB", "test": "~250 KB"}
LICENCE = "CC BY 4.0 (Alpaydin & Kaynak, 1998)"
NUM_PIXELS = 64
NUM_CLASSES = 10


def parse_optdigits(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse the comma-separated optdigits format into ``(pixels, labels)``.

    Pixel values are integers in 0-16; they are scaled to ``[0, 1]`` so inputs
    sit on the scale the initialisation schemes assume. Labels are returned as
    ``int64`` class indices, which is what
    :func:`~gradlab.nn.losses.cross_entropy_with_logits` expects.

    >>> pixels, labels = parse_optdigits("0,1,2," + "3," * 61 + "7")
    >>> pixels.shape, labels.tolist()
    ((1, 64), [7])
    >>> float(pixels[0, 1])
    0.0625
    """
    rows = [line for line in text.strip().split("\n") if line.strip()]
    if not rows:
        raise ValueError("optdigits data is empty.")
    try:
        values = np.array([[int(v) for v in row.split(",")] for row in rows], dtype=np.int64)
    except ValueError as exc:
        raise ValueError(f"optdigits rows must be comma-separated integers: {exc}") from None
    if values.ndim != 2 or values.shape[1] != NUM_PIXELS + 1:
        raise ValueError(
            f"Expected {NUM_PIXELS + 1} columns ({NUM_PIXELS} pixels + 1 label); "
            f"got {values.shape[1] if values.ndim == 2 else 'ragged rows'}. "
            "This does not look like the optdigits format."
        )
    pixels = values[:, :NUM_PIXELS].astype(np.float64) / 16.0
    labels = values[:, NUM_PIXELS]
    label_lo, label_hi = int(np.min(labels)), int(np.max(labels))
    if label_lo < 0 or label_hi >= NUM_CLASSES:
        raise ValueError(f"Labels must be 0-{NUM_CLASSES - 1}; got [{label_lo}, {label_hi}].")
    if float(np.min(pixels)) < 0.0 or float(np.max(pixels)) > 1.0:
        raise ValueError("Pixel values must be 0-16 before scaling.")
    return pixels, labels


def local_path(directory: Path | str, split: str) -> Path:
    if split not in FILES:
        raise ValueError(f"split must be one of {sorted(FILES)}; got {split!r}.")
    return Path(directory) / FILES[split]


def load_split(directory: Path | str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Read and parse one split from a local directory. Never downloads."""
    path = local_path(directory, split)
    if not path.exists():
        raise FileNotFoundError(path)
    return parse_optdigits(path.read_text())


def download_split(directory: Path | str, split: str, timeout: float = 60.0) -> Path:
    """Fetch one split from UCI into ``directory``.

    This is the only function in GradLab that opens a network connection, and
    nothing calls it implicitly — ``examples/digits.py`` requires an explicit
    ``--download`` flag from the user.
    """
    import urllib.request

    target = local_path(directory, split)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{FILES[split]}"
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        payload = response.read()
    parse_optdigits(payload.decode("utf-8"))  # validate before writing anything
    target.write_bytes(payload)
    return target
