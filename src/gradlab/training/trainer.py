"""A small training loop — and nothing more.

This is deliberately not a framework-within-a-framework. :func:`fit` does the
same six things a hand-written loop does, in the same order, and
``examples/linear_regression.py`` writes that loop out by hand to show they are
identical. What :func:`fit` adds is bookkeeping: per-epoch metrics, gradient
norms, evaluation under ``no_grad``, and detection of the two failure modes that
otherwise waste a training run — a non-finite loss and an exploding gradient.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from gradlab._config import no_grad
from gradlab._errors import GradLabError
from gradlab.data.dataloader import DataLoader
from gradlab.nn.module import Module
from gradlab.optim.optimizer import Optimizer
from gradlab.tensor import Tensor
from gradlab.training.history import History


def global_grad_norm(params: Iterable[Tensor]) -> float:
    """The L2 norm of all gradients concatenated into one vector.

    This single number is the most useful diagnostic in training. If it grows by
    orders of magnitude across epochs the run is diverging; if it collapses
    toward zero early, the signal is not reaching the first layers.
    """
    total = 0.0
    for p in params:
        if p._grad is not None:
            total += float(np.sum(p._grad.data.astype(np.float64) ** 2))
    return float(np.sqrt(total))


def clip_grad_norm(params: Iterable[Tensor], max_norm: float) -> float:
    """Scale all gradients down so their global norm is at most ``max_norm``.

    Returns the norm *before* clipping. This must be called explicitly — GradLab
    never clips on your behalf, because silently shrinking a gradient changes
    the optimisation problem without telling you.
    """
    params = list(params)
    norm = global_grad_norm(params)
    if norm > max_norm > 0:
        scale = max_norm / (norm + 1e-12)
        for p in params:
            if p._grad is not None:
                p._grad._data = p._grad._data * scale
    return norm


def evaluate(
    model: Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    metric: Callable[[Tensor, Tensor], float] | None = None,
) -> tuple[float, float | None]:
    """Mean loss (and optional metric) over a loader, in eval mode under ``no_grad``.

    Both matter: ``model.eval()`` turns Dropout into the identity, and
    ``no_grad`` stops the evaluation pass from building a graph it would
    immediately discard.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    seen = 0
    try:
        with no_grad():
            for batch in loader:
                inputs, targets = batch[0], batch[1]
                n = inputs.shape[0]
                predictions = model(inputs)
                total_loss += float(criterion(predictions, targets).item()) * n
                if metric is not None:
                    total_metric += float(metric(predictions, targets)) * n
                seen += n
    finally:
        model.train(was_training)
    if seen == 0:
        raise GradLabError("evaluate() received an empty loader.")
    return total_loss / seen, (total_metric / seen if metric is not None else None)


def fit(
    model: Module,
    optimizer: Optimizer,
    criterion: Callable[[Tensor, Tensor], Tensor],
    train_loader: DataLoader,
    epochs: int = 10,
    val_loader: DataLoader | None = None,
    metric: Callable[[Tensor, Tensor], float] | None = None,
    verbose: bool | int = False,
    grad_clip: float | None = None,
    lr_schedule: Callable[[int], float] | None = None,
    explosion_factor: float = 1e4,
    on_epoch_end: Callable[[int, History], None] | None = None,
) -> History:
    """Train ``model`` for ``epochs`` epochs and return the measured history.

    The loop body is exactly::

        optimizer.zero_grad()      # gradients accumulate, so clear them
        predictions = model(x)     # forward pass, building the graph
        loss = criterion(...)      # scalar objective
        loss.backward()            # reverse-mode pass over that graph
        optimizer.step()           # update parameters under no_grad

    Parameters
    ----------
    verbose
        ``True`` prints every epoch; an integer *n* prints every *n*th epoch.
    grad_clip
        If set, clip the global gradient norm to this value before stepping.
        Off by default — clipping is a decision, not a default.
    lr_schedule
        ``epoch -> learning_rate``. Applied at the start of each epoch.
    explosion_factor
        Record a warning if the gradient norm exceeds this multiple of its
        first-epoch value. Detection only; nothing is altered.
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive; got {epochs}.")

    params = list(optimizer.params)
    history = History(
        config={
            "epochs": int(epochs),
            "optimizer": repr(optimizer),
            "criterion": getattr(criterion, "__name__", type(criterion).__name__),
            "model": repr(model),
            "num_parameters": model.num_parameters(),
            "batch_size": train_loader.batch_size,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset) if val_loader is not None else 0,
            "shuffle_seed": train_loader.seed,
            "grad_clip": grad_clip,
        }
    )
    first_norm: float | None = None

    for epoch in range(1, epochs + 1):
        if lr_schedule is not None:
            optimizer.set_lr(float(lr_schedule(epoch)))

        model.train()
        epoch_loss = 0.0
        epoch_metric = 0.0
        epoch_norm = 0.0
        seen = 0
        batches = -1

        for batches, batch in enumerate(train_loader):
            inputs, targets = batch[0], batch[1]
            n = inputs.shape[0]

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)

            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                raise GradLabError(
                    f"Loss became {loss_value} at epoch {epoch}, batch {batches + 1}. "
                    "GradLab stops rather than continuing to train on a non-finite "
                    "objective. The usual causes are a learning rate that is too high, "
                    "an unstable loss formulation (use the *_with_logits losses), or a "
                    "log/division domain problem. Inspect history.train_loss for the "
                    "epoch where it started to grow."
                )

            loss.backward()
            if grad_clip is not None:
                epoch_norm += clip_grad_norm(params, grad_clip) * n
            else:
                epoch_norm += global_grad_norm(params) * n
            optimizer.step()

            epoch_loss += loss_value * n
            if metric is not None:
                epoch_metric += float(metric(predictions, targets)) * n
            seen += n

        if seen == 0:
            raise GradLabError("The training loader produced no batches.")

        batches += 1
        mean_norm = epoch_norm / seen
        if first_norm is None:
            first_norm = max(mean_norm, 1e-12)
        elif mean_norm > first_norm * explosion_factor:
            history.warnings.append(
                f"epoch {epoch}: gradient norm {mean_norm:.3e} is more than "
                f"{explosion_factor:g}x the first epoch's {first_norm:.3e} — the run is "
                "probably diverging. Nothing was changed; lower the learning rate or "
                "pass grad_clip."
            )

        val_loss = val_metric = None
        if val_loader is not None:
            val_loss, val_metric = evaluate(model, val_loader, criterion, metric)

        history.record(
            epoch,
            train_loss=epoch_loss / seen,
            train_metric=(epoch_metric / seen) if metric is not None else None,
            val_loss=val_loss,
            val_metric=val_metric,
            grad_norm=mean_norm,
            learning_rate=optimizer.lr,
        )

        if verbose and (verbose is True or epoch % int(verbose) == 0 or epoch == epochs):
            parts = [f"epoch {epoch:>4}/{epochs}", f"train_loss {epoch_loss / seen:.6f}"]
            if metric is not None:
                parts.append(f"train_metric {epoch_metric / seen:.4f}")
            if val_loss is not None:
                parts.append(f"val_loss {val_loss:.6f}")
            if val_metric is not None:
                parts.append(f"val_metric {val_metric:.4f}")
            parts.append(f"|g| {mean_norm:.3e}")
            print("  ".join(parts))

        if on_epoch_end is not None:
            on_epoch_end(epoch, history)

    return history
