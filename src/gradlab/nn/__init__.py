"""``gradlab.nn`` — the neural network library.

Layers, activations, losses and initialisation, all built on GradLab tensors so
every forward pass records a graph the engine (and the Microscope) can walk.

    from gradlab import nn

    model = nn.Sequential(
        nn.Linear(2, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )
"""

from __future__ import annotations

from gradlab.nn import initialization
from gradlab.nn.activations import Identity, ReLU, Sigmoid, Tanh
from gradlab.nn.initialization import (
    kaiming_normal,
    kaiming_uniform,
    xavier_normal,
    xavier_uniform,
)
from gradlab.nn.layers import Dropout, Flatten, Linear
from gradlab.nn.losses import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    MSELoss,
    accuracy,
    binary_cross_entropy_with_logits,
    cross_entropy_with_logits,
    mse_loss,
)
from gradlab.nn.module import Module, Parameter, Sequential

__all__ = [
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "Dropout",
    "Flatten",
    "Identity",
    "Linear",
    "MSELoss",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "accuracy",
    "binary_cross_entropy_with_logits",
    "cross_entropy_with_logits",
    "initialization",
    "kaiming_normal",
    "kaiming_uniform",
    "mse_loss",
    "xavier_normal",
    "xavier_uniform",
]
