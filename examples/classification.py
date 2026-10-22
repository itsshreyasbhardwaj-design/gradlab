"""Experiment 3b — multiclass classification with softmax cross-entropy.

Run: ``python examples/classification.py``

Three-class problems on two 2-D datasets: well-separated blobs, and interleaved
spirals that no linear model can touch. Both report accuracy against a stated
reference rather than against a number chosen in advance.
"""

from __future__ import annotations

import numpy as np

import gradlab as gl
from gradlab import data, nn, optim, training

from _report import decision_boundary, loss_curve, save_artifact

SEED = 2026
print(__doc__.strip())


def nearest_centroid_accuracy(X, y, k):
    """The natural reference for isotropic clusters: assign to the nearest mean."""
    points, labels = X.data, y.data.reshape(-1)
    centroids = np.stack([points[labels == c].mean(axis=0) for c in range(k)])
    predicted = np.argmin(((points[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)
    return float((predicted == labels).mean())


def run(name, X, y, k, hidden, epochs, lr):
    Xtr, Xte, ytr, yte = data.train_test_split(X, y, test_size=0.25, seed=SEED)
    loader = data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=32, shuffle=True, seed=SEED)
    val_loader = data.DataLoader(data.TensorDataset(Xte, yte), batch_size=128)

    model = nn.Sequential(
        nn.Linear(2, hidden, init="kaiming", seed=SEED),
        nn.ReLU(),
        nn.Linear(hidden, hidden, init="kaiming", seed=SEED + 1),
        nn.ReLU(),
        nn.Linear(hidden, k, seed=SEED + 2),
    )
    print(
        f"\n{'=' * 72}\n{name}  —  2 → {hidden} → {hidden} → {k}, ReLU + Kaiming init\n{'=' * 72}"
    )
    print(f"  {Xtr.shape[0]} train / {Xte.shape[0]} val, {k} classes, seed {SEED}")
    print(f"  {model.num_parameters()} parameters, Adam(lr={lr}), {epochs} epochs\n")

    history = training.fit(
        model,
        optim.Adam(model.parameters(), lr=lr),
        nn.cross_entropy_with_logits,
        loader,
        epochs=epochs,
        val_loader=val_loader,
        metric=nn.accuracy,
        verbose=max(1, epochs // 5),
    )
    with gl.no_grad():
        train_acc = nn.accuracy(model(Xtr), ytr)
        val_acc = nn.accuracy(model(Xte), yte)

    chance = 1.0 / k
    print(f"\n  train accuracy {train_acc:.4f} | val accuracy {val_acc:.4f}")
    print(f"  chance rate for {k} classes: {chance:.4f}")
    print(
        f"  cross-entropy of a uniform guess: {np.log(k):.4f} "
        f"(final train loss {history.train_loss[-1]:.4f})"
    )
    print("\n" + loss_curve(history.train_loss, history.val_loss))

    def predict(points):
        with gl.no_grad():
            return model(gl.Tensor(points)).data.argmax(axis=-1)

    print("\n" + decision_boundary(predict, Xte.data, yte.data))
    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "history": history.to_dict(),
        "num_parameters": model.num_parameters(),
    }


results = {}

X, y = data.make_blobs(n_samples=600, centers=3, cluster_std=1.1, seed=SEED)
ceiling = nearest_centroid_accuracy(X, y, 3)
results["blobs"] = run("Gaussian blobs", X, y, 3, hidden=16, epochs=60, lr=0.05)
print(f"\n  nearest-centroid reference on this data: {ceiling:.4f}")
print(f"  the network reached {results['blobs']['val_accuracy']:.4f} on held-out points.")
results["blobs"]["nearest_centroid_reference"] = ceiling

X, y = data.make_spiral(n_samples=900, classes=3, noise=0.12, seed=SEED)
spiral_ceiling = nearest_centroid_accuracy(X, y, 3)
results["spiral"] = run("Interleaved spirals", X, y, 3, hidden=32, epochs=300, lr=0.02)
print(f"\n  nearest-centroid reference on the spirals: {spiral_ceiling:.4f}")
print("  Spiral arms wrap around a common centre, so every class has roughly the same")
print("  mean — a centroid classifier is near chance, and the gap to the network is")
print("  entirely the curved decision boundary a nonlinearity makes possible.")
results["spiral"]["nearest_centroid_reference"] = spiral_ceiling

path = save_artifact(
    "classification",
    {
        "experiment": "classification",
        "seed": SEED,
        "loss": "cross_entropy_with_logits",
        "results": results,
    },
)
print(f"\nartifact written: {path}")
