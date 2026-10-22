"""Experiment 4 — handwritten digit classification (optional download).

Run: ``python examples/digits.py``            (uses a local copy if present)
     ``python examples/digits.py --download``  (fetches the dataset first)

**Dataset provenance.** This uses *Optical Recognition of Handwritten Digits*
from the UCI Machine Learning Repository (Alpaydin & Kaynak, 1998), licensed
CC BY 4.0. Each sample is an 8×8 image of a handwritten digit, stored as 64
integers in 0–16 plus a class label 0–9. The training split has 3823 samples and
the test split 1797.

    https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

**The download is opt-in and never automatic.** GradLab has no network
dependency; this script will not fetch anything unless you pass ``--download``,
and it tells you exactly what it would fetch and from where. Without it, the
script explains how to obtain the files and exits cleanly — it does not
substitute synthetic data and call it digits.

No state-of-the-art claim is made here. A small MLP on 8×8 digits is a
well-understood, easy problem; the point is that GradLab trains it correctly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import gradlab as gl
from gradlab import data, nn, optim, training
from gradlab.data.optdigits import (
    APPROX_SIZES,
    BASE_URL,
    FILES,
    LICENCE,
    download_split,
    load_split,
)

from _report import loss_curve, save_artifact

DATA_DIR = Path(__file__).resolve().parent / "_data" / "optdigits"

SEED = 7
EPOCHS = 40
LR = 0.002
BATCH = 64


def load(split: str, allow_download: bool) -> tuple[np.ndarray, np.ndarray]:
    """Read a split, downloading it first only if the user allowed that."""
    try:
        return load_split(DATA_DIR, split)
    except FileNotFoundError:
        if not allow_download:
            raise
        print(f"  downloading {FILES[split]} ({APPROX_SIZES[split]}) from {BASE_URL}")
        saved = download_split(DATA_DIR, split)
        print(f"    saved to {saved}")
        return load_split(DATA_DIR, split)


def explain_missing() -> None:
    print(f"""
The optdigits files are not present at {DATA_DIR}.

GradLab will not download anything unless you ask it to. Either:

  1. Re-run with the download flag:
         python examples/digits.py --download

     That fetches two files from the UCI Machine Learning Repository:
         {BASE_URL}/{FILES["train"]}   ({APPROX_SIZES["train"]})
         {BASE_URL}/{FILES["test"]}   ({APPROX_SIZES["test"]})
     Licence: {LICENCE}.

  2. Or download them yourself and place them in that directory.

Every other experiment in examples/ runs offline with no external data:
    scalar_autograd.py  linear_regression.py  xor.py
    two_moons.py        classification.py     pytorch_comparison.py
""")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--download",
        action="store_true",
        help="allow fetching the dataset from UCI (off by default)",
    )
    args = parser.parse_args()

    print(__doc__.strip().split("**Dataset")[0].strip())
    try:
        Xtr_raw, ytr_raw = load("train", args.download)
        Xte_raw, yte_raw = load("test", args.download)
    except FileNotFoundError:
        explain_missing()
        return 0
    except OSError as exc:
        print(f"\nDownload failed: {exc}")
        print("No data was written. Fetch the files manually and re-run.")
        return 1

    Xtr = gl.Tensor(Xtr_raw, copy=False)
    ytr = gl.Tensor(ytr_raw, copy=False)
    Xte = gl.Tensor(Xte_raw, copy=False)
    yte = gl.Tensor(yte_raw, copy=False)

    print(f"""
dataset: UCI optdigits (CC BY 4.0)
  train {Xtr.shape[0]} samples, test {Xte.shape[0]} samples
  features {Xtr.shape[1]} (8x8 pixels, scaled 0-16 -> 0-1)
  classes  {int(ytr.data.max()) + 1}
  class balance (train): {np.bincount(ytr_raw).tolist()}""")

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 64, init="kaiming", seed=SEED),
        nn.ReLU(),
        nn.Dropout(0.2, seed=SEED),
        nn.Linear(64, 32, init="kaiming", seed=SEED + 1),
        nn.ReLU(),
        nn.Linear(32, 10, seed=SEED + 2),
    )
    print(
        f"\nmodel: 64 -> 64 -> 32 -> 10, ReLU + Dropout(0.2), {model.num_parameters()} parameters"
    )
    print(f"optimiser: Adam(lr={LR}), batch {BATCH}, {EPOCHS} epochs, seed {SEED}\n")

    train_loader = data.DataLoader(
        data.TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, seed=SEED
    )
    test_loader = data.DataLoader(data.TensorDataset(Xte, yte), batch_size=256)

    history = training.fit(
        model,
        optim.Adam(model.parameters(), lr=LR),
        nn.cross_entropy_with_logits,
        train_loader,
        epochs=EPOCHS,
        val_loader=test_loader,
        metric=nn.accuracy,
        verbose=5,
    )

    model.eval()
    with gl.no_grad():
        train_acc = nn.accuracy(model(Xtr), ytr)
        test_acc = nn.accuracy(model(Xte), yte)
        predictions = model(Xte).data.argmax(axis=-1)

    confusion = np.zeros((10, 10), dtype=int)
    for truth, predicted in zip(yte_raw, predictions, strict=True):
        confusion[truth, predicted] += 1

    print(f"""
{"=" * 72}
Results (measured on the held-out test split)
{"=" * 72}
  train accuracy {train_acc:.4f}
  test  accuracy {test_acc:.4f}
  test  error    {(1 - test_acc) * 100:.2f}%  ({round((1 - test_acc) * len(yte_raw))} of {len(yte_raw)} wrong)

  This is an easy, well-studied dataset; published results reach roughly 97-99%
  with small models. No state-of-the-art claim is made here. What the number
  demonstrates is that GradLab's engine trains a real multiclass model to the
  accuracy the architecture should reach.""")
    print("\n" + loss_curve(history.train_loss, history.val_loss))

    print("\n  confusion matrix (rows = true, columns = predicted)")
    print("      " + "".join(f"{d:>5}" for d in range(10)))
    for d in range(10):
        row = "".join(f"{confusion[d, p]:>5}" for p in range(10))
        print(f"    {d} {row}")
    worst = int(np.argmin(np.diag(confusion) / confusion.sum(axis=1)))
    per_class = np.diag(confusion) / confusion.sum(axis=1)
    print(f"\n  weakest class: {worst} at {per_class[worst]:.3f} recall")

    path = save_artifact(
        "digits",
        {
            "experiment": "digits",
            "dataset": "UCI optdigits (CC BY 4.0, Alpaydin & Kaynak 1998)",
            "seed": SEED,
            "hyperparameters": {
                "epochs": EPOCHS,
                "lr": LR,
                "batch_size": BATCH,
                "dropout": 0.2,
                "optimizer": "Adam",
            },
            "architecture": "64 -> 64 -> 32 -> 10",
            "num_parameters": model.num_parameters(),
            "results": {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "per_class_recall": per_class.tolist(),
            },
            "confusion_matrix": confusion.tolist(),
            "history": history.to_dict(),
        },
    )
    print(f"\nartifact written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
