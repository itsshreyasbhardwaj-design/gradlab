"""Shared helpers for the experiment scripts: ASCII plots and JSON export.

GradLab has no plotting dependency, so the examples draw to the terminal and
write a JSON artifact that the Gradient Microscope renders properly. Both are
built from measured values only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "_out"


def save_artifact(name: str, payload: dict) -> Path:
    """Write an experiment result where the Microscope can find it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_safe))
    return path


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"{type(obj)} is not JSON serialisable")


def loss_curve(values, val_values=None, height: int = 12, width: int = 64, log: bool = True) -> str:
    """A log-scale ASCII line chart of one or two loss curves."""
    series = [np.asarray(values, dtype=float)]
    if val_values:
        series.append(np.asarray(val_values, dtype=float))
    marks = ["*", "o"]

    def transform(a):
        return np.log10(np.maximum(a, 1e-12)) if log else a

    tseries = [transform(s) for s in series]
    lo = min(float(s.min()) for s in tseries)
    hi = max(float(s.max()) for s in tseries)
    if hi - lo < 1e-12:
        hi = lo + 1.0

    grid = [[" "] * width for _ in range(height)]
    for si, s in enumerate(tseries):
        n = len(s)
        for col in range(width):
            idx = round(col * (n - 1) / max(width - 1, 1))
            row = round((hi - s[idx]) / (hi - lo) * (height - 1))
            grid[row][col] = marks[si]

    label = "log10(loss)" if log else "loss"
    lines = [f"  {label}"]
    for r, row in enumerate(grid):
        value = hi - r * (hi - lo) / (height - 1)
        lines.append(f"  {value:8.3f} |{''.join(row)}")
    lines.append("  " + " " * 9 + "+" + "-" * width)
    lines.append(f"  {'':9} epoch 1{' ' * (width - 16)}epoch {len(series[0])}")
    legend = "  * train" + ("   o validation" if val_values else "")
    lines.append(legend)
    return "\n".join(lines)


def decision_boundary(predict, X, y, width: int = 62, height: int = 24) -> str:
    """ASCII map of a 2-D classifier's decision regions with the data overlaid.

    ``predict`` takes an ``(n, 2)`` array and returns class indices. Background
    cells show the predicted class as a shade; data points are drawn as digits,
    uppercased when the model gets that point wrong.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1).astype(int)
    pad = 0.35
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad

    gx = np.linspace(x0, x1, width)
    gy = np.linspace(y1, y0, height)
    mesh = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    regions = np.asarray(predict(mesh)).reshape(height, width).astype(int)

    shades = ["·", ":", "+", "=", "#"]
    grid = [[shades[regions[r, c] % len(shades)] for c in range(width)] for r in range(height)]

    predictions = np.asarray(predict(X)).reshape(-1).astype(int)
    for (px, py), truth, pred in zip(X, y, predictions, strict=True):
        c = round((px - x0) / (x1 - x0) * (width - 1))
        r = round((y1 - py) / (y1 - y0) * (height - 1))
        if 0 <= r < height and 0 <= c < width:
            glyph = str(truth % 10)
            grid[r][c] = (
                glyph
                if truth == pred
                else glyph.translate(str.maketrans("0123456789", "ABCDEFGHIJ"))
            )

    lines = ["  +" + "-" * width + "+"]
    lines += ["  |" + "".join(row) + "|" for row in grid]
    lines.append("  +" + "-" * width + "+")
    lines.append("  background shade = predicted region;  digits = true class;")
    lines.append("  a LETTER (A=class 0, B=class 1) marks a point the model gets wrong")
    return "\n".join(lines)
