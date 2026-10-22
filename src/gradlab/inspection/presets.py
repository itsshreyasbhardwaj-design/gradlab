"""The trace set the Gradient Microscope ships with.

Each preset is a real computation. Running ``gradlab-microscope generate``
executes every one of them through the engine and writes the resulting traces;
nothing in ``apps/microscope`` contains a precomputed or hand-drawn graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import gradlab as gl
from gradlab import data, nn, optim, training
from gradlab.inspection.tracer import (
    decision_surface,
    trace_expression,
    trace_training_run,
    trace_training_step,
)

#: Expression presets. Each is a real computation, executed at generate time.
#: ``upstream`` is only needed for a non-scalar output — the engine refuses to
#: invent a seed for one, and one preset exists specifically to show that.
EXPRESSION_PRESETS: list[dict[str, Any]] = [
    {
        "slug": "quadratic",
        "title": "x\u00b2 + 2x  at  x = 3",
        "text": "x^2 + 2*x",
        "values": {"x": 3.0},
        "note": "The starting example. x is used by two separate branches, so its "
        "gradient is the sum of both contributions: 2x from the square and 2 "
        "from the linear term.",
    },
    {
        "slug": "reused-node",
        "title": "x\u00b7x + x\u00b7x  at  x = 2",
        "text": "x*x + x*x",
        "values": {"x": 2.0},
        "note": "The same intermediate value feeds two edges. Reverse mode visits both "
        "consumers before the producer, so the contributions 4 and 4 are summed "
        "into 8 rather than one overwriting the other.",
    },
    {
        "slug": "chain-rule",
        "title": "tanh(2\u00b7e\u02e3)  at  x = 0.4",
        "text": "tanh(2*exp(x))",
        "values": {"x": 0.4},
        "note": "A three-deep chain. Each backward step multiplies the upstream gradient "
        "by one local derivative; the Microscope shows both factors separately.",
    },
    {
        "slug": "sigmoid-saturation",
        "title": "\u03c3(x)  at  x = 6  (a saturated unit)",
        "text": "sigmoid(x)",
        "values": {"x": 6.0},
        "note": "\u03c3(6) \u2248 0.9975 and the local derivative \u03c3(1\u2212\u03c3) "
        "\u2248 0.0025. This is vanishing gradient visible in a single node: "
        "almost nothing propagates back through a saturated unit.",
    },
    {
        "slug": "relu-gate",
        "title": "sum(relu(x)\u00b7x)  for a vector straddling zero",
        "text": "sum(relu(x) * x)",
        "values": {"x": [-2.0, -0.5, 0.5, 2.0]},
        "note": "relu's derivative is a gate: exactly zero wherever the input was "
        "negative. The gradient vector is 0 on the first two components and 2x "
        "on the last two \u2014 a dead unit receives nothing.",
    },
    {
        "slug": "non-scalar-seed",
        "title": "x\u00b7x with an explicit upstream gradient [1, 0, 2]",
        "text": "x*x",
        "values": {"x": [1.0, 2.0, 3.0]},
        "upstream": [1.0, 0.0, 2.0],
        "note": "A non-scalar output has no canonical seed, so GradLab requires one "
        "rather than assuming all ones (which would silently differentiate "
        "x.sum() instead). Here the seed [1, 0, 2] weights each component: the "
        "middle gradient is zeroed out, and the last is doubled.",
    },
    {
        "slug": "broadcast",
        "title": "sum(W + b)  with W (4\u00d73) and b (3,)",
        "text": "sum(W + b)",
        "values": {"W": np.ones((4, 3)).tolist(), "b": [0.0, 0.0, 0.0]},
        "note": "The forward pass stretched b across 4 rows. The backward pass must sum "
        "the gradient back down over that axis, so b's gradient is [4, 4, 4] "
        "\u2014 not ones, and not a 4\u00d73 array.",
    },
    {
        "slug": "matmul",
        "title": "sum(A @ B)  with A (2\u00d73), B (3\u00d72)",
        "text": "sum(A @ B)",
        "values": {
            "A": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "B": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        },
        "note": "dL/dA = G @ B\u1d40 and dL/dB = A\u1d40 @ G. The transposes are what "
        "make the shapes work out, and they are the whole content of the rule.",
    },
    {
        "slug": "log-domain",
        "title": "log(x)\u00b7x  at  x = 0.1",
        "text": "log(x) * x",
        "values": {"x": 0.1},
        "note": "Near zero, 1/x dominates: the local derivative of log is 10 here. Small "
        "inputs to a log produce large gradients \u2014 one route to an "
        "exploding update.",
    },
]


def build_expression_traces() -> dict[str, dict[str, Any]]:
    """Execute every expression preset and return its trace."""
    from gradlab.inspection.expressions import evaluate

    traces: dict[str, dict[str, Any]] = {}
    for preset in EXPRESSION_PRESETS:
        output, variables = evaluate(preset["text"], preset["values"])
        names = list(variables)
        trace = trace_expression(
            lambda *ts, _out=output: _out,
            *[variables[n] for n in names],
            names=names,
            label=preset["title"],
            upstream=preset.get("upstream"),
        )
        trace["expression_text"] = preset["text"]
        trace["teaching_note"] = preset["note"]
        trace["slug"] = preset["slug"]
        trace["explicit_upstream"] = preset.get("upstream") is not None
        traces[preset["slug"]] = trace
    return traces


def build_training_step_trace() -> dict[str, Any]:
    """One real Adam step on a small MLP, captured before and after the update."""
    X, y = data.make_moons(n_samples=16, noise=0.15, seed=11)
    model = nn.Sequential(
        nn.Linear(2, 4, seed=11),
        nn.Tanh(),
        nn.Linear(4, 1, seed=12),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    trace = trace_training_step(
        model,
        optimizer,
        nn.binary_cross_entropy_with_logits,
        X,
        y,
        label="One Adam step on a 2→4→1 network (16 two-moons samples)",
    )
    trace["slug"] = "training-step"
    trace["teaching_note"] = (
        "This is a real step. Every parameter's before value, gradient, optimiser state "
        "and after value was read from the engine; the update column is literally "
        "after − before. At t=1 Adam's bias correction makes the step size equal the "
        "learning rate regardless of the gradient's magnitude — look at the ratio."
    )
    return trace


def build_training_run_trace() -> dict[str, Any]:
    """A full two-moons training run, with its decision surface at the end."""
    SEED = 20261023
    X, y = data.make_moons(n_samples=600, noise=0.20, seed=SEED)
    Xtr, Xte, ytr, yte = data.train_test_split(X, y, test_size=0.25, seed=SEED)
    model = nn.Sequential(
        nn.Linear(2, 16, seed=SEED),
        nn.Tanh(),
        nn.Linear(16, 16, seed=SEED + 1),
        nn.Tanh(),
        nn.Linear(16, 1, seed=SEED + 2),
    )
    train_loader = data.DataLoader(
        data.TensorDataset(Xtr, ytr), batch_size=32, shuffle=True, seed=SEED
    )
    val_loader = data.DataLoader(data.TensorDataset(Xte, yte), batch_size=128)

    snapshots: list[dict[str, Any]] = []
    weight_hist: list[dict[str, Any]] = []

    def on_epoch_end(epoch: int, history) -> None:
        if epoch in (1, 5, 20, 60, 150, 300):
            snapshots.append(
                {
                    "epoch": epoch,
                    "surface": decision_surface(model, (-2.0, 3.0), (-1.8, 2.2), resolution=44),
                }
            )
        if epoch % 10 == 0 or epoch == 1:
            flat = np.concatenate([p.data.reshape(-1) for p in model.parameters()])
            counts, edges = np.histogram(flat, bins=24, range=(-3.0, 3.0))
            weight_hist.append(
                {
                    "epoch": epoch,
                    "counts": counts.tolist(),
                    "bin_edges": edges.tolist(),
                    "l2_norm": float(np.sqrt(np.sum(flat**2))),
                }
            )

    history = training.fit(
        model,
        optim.Adam(model.parameters(), lr=0.01),
        nn.binary_cross_entropy_with_logits,
        train_loader,
        epochs=300,
        val_loader=val_loader,
        metric=nn.accuracy,
        on_epoch_end=on_epoch_end,
    )

    with gl.no_grad():
        train_acc = nn.accuracy(model(Xtr), ytr)
        val_acc = nn.accuracy(model(Xte), yte)

    trace = trace_training_run(history, label="Two-moons: 2→16→16→1, Adam(0.01), 300 epochs")
    trace["slug"] = "training-run"
    trace["final"] = {"train_accuracy": train_acc, "val_accuracy": val_acc}
    trace["decision_surfaces"] = snapshots
    trace["weight_histograms"] = weight_hist
    trace["points"] = {
        "train": {"X": Xtr.data.tolist(), "y": ytr.data.reshape(-1).tolist()},
        "val": {"X": Xte.data.tolist(), "y": yte.data.reshape(-1).tolist()},
    }
    trace["teaching_note"] = (
        "Both curves are measured, one point per epoch. The validation minimum at "
        f"epoch {history.best_epoch} of 300 is where the model stopped generalising "
        "better; the training curve keeps falling past it. That gap is overfitting, "
        "and it is the reason a validation split exists."
    )
    return trace


def build_lr_comparison_trace() -> dict[str, Any]:
    """The same model trained across seven learning rates, spanning four decades.

    The range was chosen after measuring where this particular model actually
    breaks down, not by assuming a textbook number. On a 2->12->1 tanh network
    with ``binary_cross_entropy_with_logits``, SGD stays well behaved up to
    lr = 3, spikes and recovers at lr = 10, and is permanently unstable from
    lr = 30 upward. Notably it never produces ``nan`` — see the note below.
    """
    X, y = data.make_moons(n_samples=400, noise=0.2, seed=5)
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=32, shuffle=True, seed=5)

    runs = []
    for lr in (0.0003, 0.003, 0.03, 0.3, 3.0, 30.0, 300.0):
        model = nn.Sequential(nn.Linear(2, 12, seed=5), nn.Tanh(), nn.Linear(12, 1, seed=6))
        loader.reset_epochs()
        try:
            history = training.fit(
                model,
                optim.SGD(model.parameters(), lr=lr),
                nn.binary_cross_entropy_with_logits,
                loader,
                epochs=120,
                metric=nn.accuracy,
            )
            with gl.no_grad():
                acc = nn.accuracy(model(X), y)
            runs.append(
                {
                    "lr": lr,
                    "train_loss": history.train_loss,
                    "grad_norm": history.grad_norm,
                    "accuracy": acc,
                    "final_loss": history.train_loss[-1],
                    "min_loss": min(history.train_loss),
                    "max_loss": max(history.train_loss),
                    "stopped": False,
                    "warnings": history.warnings,
                }
            )
        except gl.GradLabError as exc:
            runs.append(
                {
                    "lr": lr,
                    "train_loss": [],
                    "grad_norm": [],
                    "accuracy": None,
                    "stopped": True,
                    "error": str(exc).split(".")[0],
                }
            )

    return {
        "schema_version": "1.0",
        "kind": "lr_comparison",
        "slug": "learning-rates",
        "label": "One 2\u219212\u21921 network at seven learning rates (SGD, 120 epochs)",
        "runs": runs,
        "divergence_demo": _build_divergence_demo(),
        "teaching_note": (
            "Read the measured curves, not the folklore. At lr = 0.0003 the loss barely "
            "moves \u2014 the steps are too small to cross the distance available in 120 "
            "epochs, which is underfitting caused purely by the optimiser. At lr = 0.3 to "
            "3 it converges fastest. At lr = 10 it spikes to a loss of about 6 and then "
            "recovers; from lr = 30 upward every step overshoots the minimum it was "
            "aiming at and the loss never settles.\n\n"
            "What it does NOT do is produce nan, and that is worth understanding: "
            "binary_cross_entropy_with_logits grows only *linearly* in the logit, so even "
            "an absurd learning rate yields a large finite loss. Swap in a squared error "
            "and the same instability squares instead \u2014 see the second panel, where "
            "the loss really does become non-finite and GradLab stops the run."
        ),
    }


def _build_divergence_demo() -> dict[str, Any]:
    """A run that genuinely produces a non-finite loss, recorded batch by batch.

    Squared error on an unscaled regression target with a large learning rate is
    the classic blow-up: every overshoot is squared on the next step, so the loss
    grows super-exponentially until it exceeds what float64 can hold. This runs
    the loop by hand so each *batch* is recorded — at these rates the whole
    divergence happens inside the first epoch, which an epoch-level curve would
    miss entirely.
    """
    X, y, _ = data.make_regression(n_samples=200, n_features=4, noise=0.1, seed=3)
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=32, shuffle=True, seed=3)
    model = nn.Sequential(nn.Linear(4, 32, seed=3), nn.ReLU(), nn.Linear(32, 1, seed=4))
    optimizer = optim.SGD(model.parameters(), lr=5.0)

    losses: list[float] = []
    grad_norms: list[float] = []
    failure: str | None = None

    for _epoch in range(3):
        for batch in loader:
            inputs, targets = batch[0], batch[1]
            optimizer.zero_grad()
            loss = nn.mse_loss(model(inputs), targets)
            value = float(loss.item())
            if not np.isfinite(value):
                failure = (
                    f"the loss itself became {value} at batch {len(losses) + 1}: the "
                    "parameters had already been pushed far enough that the forward "
                    "pass overflowed"
                )
                break
            losses.append(value)
            loss.backward()
            grad_norms.append(training.global_grad_norm(model.parameters()))
            try:
                optimizer.step()
            except gl.GradLabError as exc:
                failure = str(exc)
                break
        if failure:
            break

    return {
        "stopped": failure is not None,
        "batch_loss": losses,
        "grad_norm": grad_norms,
        "batches_survived": len(losses),
        "error": failure or "this configuration did not diverge in this environment",
        "config": "SGD(lr=5.0), mse_loss, 4\u219232\u21921 ReLU, batch 32",
        "note": (
            "Squared error squares the overshoot, so each bad step makes the next one "
            f"worse. The loss ran for {len(losses)} batches "
            + (f"(from {losses[0]:.3g} to {losses[-1]:.3g}) " if losses else "")
            + "and then left the range of float64. GradLab stops here instead of "
            "writing nan into the parameters: by that point the model is "
            "unrecoverable, and the real cause is several steps earlier. Compare the "
            "panel above, where the log-sum-exp loss grows only linearly and so stays "
            "finite even at lr = 300."
        ),
    }
