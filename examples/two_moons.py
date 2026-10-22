"""Experiment 3 — two-moons classification, reported in full.

Run: ``python examples/two_moons.py``

Everything needed to reproduce this run is printed: the seed, the split, the
architecture, the optimiser configuration, and the measured train/validation
curves and accuracies. A logistic regression is trained alongside the MLP as a
baseline, because "the MLP got 97%" means little without knowing what a linear
model gets on the same data.
"""

from __future__ import annotations

import gradlab as gl
from gradlab import data, nn, optim, training

from _report import decision_boundary, loss_curve, save_artifact

SEED = 20261023
N_SAMPLES = 600
NOISE = 0.20
TEST_SIZE = 0.25
EPOCHS = 300
LR = 0.01
BATCH_SIZE = 32
HIDDEN = 16

print(__doc__.strip())

X, y = data.make_moons(n_samples=N_SAMPLES, noise=NOISE, seed=SEED)
Xtr, Xte, ytr, yte = data.train_test_split(X, y, test_size=TEST_SIZE, seed=SEED)

print(f"""
reproducibility block
---------------------
  seed                 {SEED}
  samples              {N_SAMPLES}  (noise sd {NOISE})
  split                {Xtr.shape[0]} train / {Xte.shape[0]} validation  (test_size={TEST_SIZE})
  class balance train  {float(ytr.data.mean()):.3f}
  class balance val    {float(yte.data.mean()):.3f}
  architecture         2 -> {HIDDEN} -> {HIDDEN} -> 1, Tanh activations
  loss                 binary_cross_entropy_with_logits
  optimiser            Adam(lr={LR})
  batch size           {BATCH_SIZE}
  epochs               {EPOCHS}
  dtype                float64""")

train_loader = data.DataLoader(
    data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, seed=SEED
)
val_loader = data.DataLoader(data.TensorDataset(Xte, yte), batch_size=128)


def predictor(model):
    def predict(points):
        with gl.no_grad():
            return (model(gl.Tensor(points)).data.reshape(-1) > 0).astype(int)

    return predict


# ------------------------------------------------------------- linear baseline
print("\n" + "=" * 72)
print("Baseline: logistic regression (a single Linear layer)")
print("=" * 72)
baseline = nn.Linear(2, 1, seed=SEED)
base_history = training.fit(
    baseline,
    optim.Adam(baseline.parameters(), lr=LR),
    nn.binary_cross_entropy_with_logits,
    train_loader,
    epochs=EPOCHS,
    val_loader=val_loader,
    metric=nn.accuracy,
    verbose=100,
)
with gl.no_grad():
    base_acc = nn.accuracy(baseline(Xte), yte)
print(f"\n  validation accuracy: {base_acc:.4f}")

# ------------------------------------------------------------------------- MLP
print("\n" + "=" * 72)
print(f"Model: 2 -> {HIDDEN} -> {HIDDEN} -> 1 with Tanh")
print("=" * 72)
train_loader.reset_epochs()
model = nn.Sequential(
    nn.Linear(2, HIDDEN, seed=SEED),
    nn.Tanh(),
    nn.Linear(HIDDEN, HIDDEN, seed=SEED + 1),
    nn.Tanh(),
    nn.Linear(HIDDEN, 1, seed=SEED + 2),
)
print(model)
print(f"trainable parameters: {model.num_parameters()}")

history = training.fit(
    model,
    optim.Adam(model.parameters(), lr=LR),
    nn.binary_cross_entropy_with_logits,
    train_loader,
    epochs=EPOCHS,
    val_loader=val_loader,
    metric=nn.accuracy,
    verbose=50,
)

with gl.no_grad():
    train_acc = nn.accuracy(model(Xtr), ytr)
    val_acc = nn.accuracy(model(Xte), yte)

print("\n" + "=" * 72)
print("Measured results")
print("=" * 72)
print(f"""
  train loss       {history.train_loss[0]:.6f} -> {history.train_loss[-1]:.6f}
  validation loss  {history.val_loss[0]:.6f} -> {history.val_loss[-1]:.6f}
  best val loss    {history.best_val_loss:.6f}  (epoch {history.best_epoch})
  train accuracy   {train_acc:.4f}
  val accuracy     {val_acc:.4f}
  gradient norm    {history.grad_norm[0]:.3e} -> {history.grad_norm[-1]:.3e}

  linear baseline  {base_acc:.4f} val accuracy  ({base_history.val_loss[-1]:.6f} val loss)
  improvement      {val_acc - base_acc:+.4f}""")
if history.warnings:
    for w in history.warnings:
        print(f"  WARNING: {w}")
print("\n" + loss_curve(history.train_loss, history.val_loss))
print("\nDecision boundary learned by the MLP (validation points):")
print(decision_boundary(predictor(model), Xte.data, yte.data))
print("\nDecision boundary of the linear baseline (same points) — note it is a straight line:")
print(decision_boundary(predictor(baseline), Xte.data, yte.data))

gap = history.train_loss[-1] - history.val_loss[-1]
print(f"""
  Overfitting check: final train loss {history.train_loss[-1]:.6f} vs
  validation {history.val_loss[-1]:.6f} (difference {gap:+.6f}). The validation
  curve reaching its minimum at epoch {history.best_epoch} of {EPOCHS} is the
  signal to watch — training past it buys train loss at validation's expense.""")

path = save_artifact(
    "two_moons",
    {
        "experiment": "two_moons",
        "seed": SEED,
        "hyperparameters": {
            "n_samples": N_SAMPLES,
            "noise": NOISE,
            "test_size": TEST_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "hidden": HIDDEN,
            "optimizer": "Adam",
            "loss": "binary_cross_entropy_with_logits",
            "activation": "Tanh",
            "dtype": "float64",
        },
        "architecture": f"2 -> {HIDDEN} -> {HIDDEN} -> 1",
        "num_parameters": model.num_parameters(),
        "results": {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "best_val_loss": history.best_val_loss,
            "best_epoch": history.best_epoch,
            "baseline_val_accuracy": base_acc,
        },
        "history": history.to_dict(),
        "baseline_history": base_history.to_dict(),
        "data": {"X_val": Xte.data, "y_val": yte.data},
    },
)
print(f"\nartifact written: {path}")
