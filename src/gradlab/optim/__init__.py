"""``gradlab.optim`` — gradient-based optimisation.

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

Every update is performed by GradLab under ``no_grad``, writing to each
parameter's buffer in place. Because that bumps the tensor's version counter, a
graph that saved the parameter cannot be differentiated again afterwards without
the engine noticing.
"""

from __future__ import annotations

from gradlab.optim.adam import Adam
from gradlab.optim.adamw import AdamW
from gradlab.optim.optimizer import Optimizer
from gradlab.optim.sgd import SGD

__all__ = ["SGD", "Adam", "AdamW", "Optimizer"]
