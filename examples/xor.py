"""Experiment 2 — XOR, with and without a hidden nonlinearity.

Run: ``python examples/xor.py``

XOR is the canonical demonstration that depth alone buys nothing. The two
models below have the same shape — 2 -> 8 -> 1 — and differ only in whether a
``tanh`` sits between the layers. Without it, the composition of two affine maps
is itself affine, so the model can only draw a straight line and XOR is
unlearnable. The experiment trains both from the same seed and reports both.
"""

from __future__ import annotations

import gradlab as gl
from gradlab import data, nn, optim, training

from _report import decision_boundary, loss_curve, save_artifact

SEED = 7
EPOCHS = 400
LR = 0.05

print(__doc__.strip())

X, y = data.make_xor(n_samples=400, noise=0.0, seed=SEED)
print(f"\ndata: {X.shape[0]} points in [-1,1]^2, label = sign(x0) XOR sign(x1), seed={SEED}")
print("the four-point truth table:")
Xt, yt = data.make_xor(n_samples=4)
for row, label in zip(Xt.tolist(), yt.tolist(), strict=True):
    print(f"    {row} -> {int(label[0])}")


def build(nonlinear: bool) -> nn.Sequential:
    """Same architecture both ways; the only difference is the activation."""
    layers = [nn.Linear(2, 8, seed=SEED)]
    if nonlinear:
        layers.append(nn.Tanh())
    layers.append(nn.Linear(8, 1, seed=SEED + 1))
    return nn.Sequential(*layers)


def run(nonlinear: bool) -> tuple[nn.Sequential, training.History]:
    model = build(nonlinear)
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=32, shuffle=True, seed=SEED)
    history = training.fit(
        model,
        optim.Adam(model.parameters(), lr=LR),
        nn.binary_cross_entropy_with_logits,
        loader,
        epochs=EPOCHS,
        metric=nn.accuracy,
    )
    return model, history


results = {}
for nonlinear in (False, True):
    label = "WITH tanh (nonlinear)" if nonlinear else "WITHOUT any activation (purely affine)"
    print("\n" + "=" * 72)
    print(f"{label}   —   2 -> 8 -> 1, Adam(lr={LR}), {EPOCHS} epochs")
    print("=" * 72)
    model, history = run(nonlinear)
    with gl.no_grad():
        logits = model(X)
    acc = nn.accuracy(logits, y)
    print(f"  final train loss : {history.train_loss[-1]:.6f}")
    print(f"  final accuracy   : {acc:.4f}")
    print("\n" + loss_curve(history.train_loss))

    def predict(points, _model=model):
        with gl.no_grad():
            return (_model(gl.Tensor(points)).data.reshape(-1) > 0).astype(int)

    print("\n" + decision_boundary(predict, X.data, y.data))
    results["nonlinear" if nonlinear else "linear"] = {
        "accuracy": acc,
        "final_loss": history.train_loss[-1],
        "history": history.to_dict(),
    }

print("\n" + "=" * 72)
print("Conclusion")
print("=" * 72)
lin = results["linear"]["accuracy"]
non = results["nonlinear"]["accuracy"]
print(f"""
  purely affine 2->8->1 : accuracy {lin:.4f}
  with tanh between     : accuracy {non:.4f}

  Two stacked Linear layers with nothing between them satisfy
  (x W1 + b1) W2 + b2 = x (W1 W2) + (b1 W2 + b2), which is a single Linear
  layer with weight W1W2. Its decision boundary is a hyperplane, and no
  hyperplane separates XOR — so roughly {lin:.0%} is the ceiling, exactly what
  guessing one class gets you. Inserting one tanh breaks the affine
  composition and the same parameter count then solves the problem.
""")

path = save_artifact(
    "xor",
    {
        "experiment": "xor",
        "seed": SEED,
        "hyperparameters": {
            "epochs": EPOCHS,
            "lr": LR,
            "optimizer": "Adam",
            "loss": "binary_cross_entropy_with_logits",
            "batch_size": 32,
        },
        "architecture": "2 -> 8 -> 1",
        "results": results,
    },
)
print(f"artifact written: {path}")
