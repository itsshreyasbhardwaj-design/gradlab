"""Experiment 1 — linear regression, written both by hand and with ``fit``.

Run: ``python examples/linear_regression.py``

The point of this experiment is twofold: show that gradient descent recovers the
parameters that generated the data, and show that ``gradlab.training.fit`` is
nothing more than the hand-written loop with bookkeeping attached. Both loops
are run from the same seed and the results are compared.
"""

from __future__ import annotations

import numpy as np

from gradlab import data, nn, optim, training

from _report import loss_curve, save_artifact

SEED = 42
EPOCHS = 200
LR = 0.05

print(__doc__.strip())

X, y, truth = data.make_regression(n_samples=300, n_features=3, bias=1.5, noise=0.1, seed=SEED)
Xtr, Xte, ytr, yte = data.train_test_split(X, y, test_size=0.25, seed=SEED)
print(
    f"\ndata: {X.shape[0]} samples, {X.shape[1]} features "
    f"({Xtr.shape[0]} train / {Xte.shape[0]} val), seed={SEED}"
)
print(f"generating weights: {truth['weights'].round(4).tolist()}  bias: {truth['bias']}")

# ---------------------------------------------------------------- hand-written
print("\n" + "=" * 72)
print("A. The loop written out by hand")
print("=" * 72)

model_a = nn.Linear(3, 1, seed=SEED)
opt_a = optim.SGD(model_a.parameters(), lr=LR)
hand_losses = []

for epoch in range(1, EPOCHS + 1):
    opt_a.zero_grad()  # gradients accumulate; clear them
    predictions = model_a(Xtr)  # forward pass builds the graph
    loss = nn.mse_loss(predictions, ytr)  # scalar objective
    loss.backward()  # reverse-mode pass
    opt_a.step()  # update under no_grad
    hand_losses.append(loss.item())
    if epoch % 50 == 0 or epoch == 1:
        print(f"  epoch {epoch:>4}  loss {loss.item():.8f}")

# ------------------------------------------------------------------- with fit
print("\n" + "=" * 72)
print("B. The same thing through training.fit (full-batch, same seed)")
print("=" * 72)

model_b = nn.Linear(3, 1, seed=SEED)
loader = data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=Xtr.shape[0])
val_loader = data.DataLoader(data.TensorDataset(Xte, yte), batch_size=Xte.shape[0])
history = training.fit(
    model_b,
    optim.SGD(model_b.parameters(), lr=LR),
    nn.mse_loss,
    loader,
    epochs=EPOCHS,
    val_loader=val_loader,
    verbose=50,
)

same = np.allclose(model_a.weight.data, model_b.weight.data, atol=1e-12)
print(f"\nhand-written and fit() reached identical parameters: {same}")
print(f"  max |difference| = {np.abs(model_a.weight.data - model_b.weight.data).max():.2e}")

# ------------------------------------------------------------------- results
print("\n" + "=" * 72)
print("Results")
print("=" * 72)
learned_w = model_b.weight.data.reshape(-1)
learned_b = model_b.bias.item()
print(f"  true weights    {truth['weights'].round(5).tolist()}   true bias    {truth['bias']}")
print(f"  learned weights {learned_w.round(5).tolist()}   learned bias {learned_b:.5f}")
print(f"  max weight error {np.abs(learned_w - truth['weights']).max():.6f}")
print(f"  bias error       {abs(learned_b - truth['bias']):.6f}")
print(f"\n  train loss {history.train_loss[0]:.6f} -> {history.train_loss[-1]:.6f}")
print(f"  val   loss {history.val_loss[0]:.6f} -> {history.val_loss[-1]:.6f}")
print(f"  irreducible noise floor (variance of the added noise) = {truth['noise'] ** 2:.6f}")
print("\n" + loss_curve(history.train_loss, history.val_loss))

path = save_artifact(
    "linear_regression",
    {
        "experiment": "linear_regression",
        "seed": SEED,
        "hyperparameters": {"epochs": EPOCHS, "lr": LR, "optimizer": "SGD", "batch": "full"},
        "truth": {"weights": truth["weights"], "bias": truth["bias"], "noise": truth["noise"]},
        "learned": {"weights": learned_w, "bias": learned_b},
        "history": history.to_dict(),
    },
)
print(f"\nartifact written: {path}")
