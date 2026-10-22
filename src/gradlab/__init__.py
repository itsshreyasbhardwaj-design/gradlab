"""GradLab — see how neural networks actually learn.

An educational machine learning framework built from first principles: a custom
tensor type, a genuine reverse-mode automatic differentiation engine, a neural
network library, optimisers, and a visual interface that shows the real graph
the engine built.

    >>> import gradlab as gl
    >>> x = gl.Tensor(3.0, requires_grad=True)
    >>> y = x * x + 2 * x
    >>> y.backward()
    >>> y.item(), x.grad.item()
    (15.0, 8.0)

NumPy supplies array storage and elementwise kernels. Every derivative, the
computation graph, the traversal order, and the gradient accumulation are
GradLab's own. PyTorch is used only as an independent test oracle and is never
imported by this package.
"""

from __future__ import annotations

from gradlab import data, nn, optim, training
from gradlab._config import (
    DEFAULT_DTYPE,
    default_rng,
    enable_grad,
    is_grad_enabled,
    no_grad,
    set_grad_enabled,
)
from gradlab._errors import (
    DomainError,
    DTypeError,
    GradientError,
    GradLabError,
    GraphReleasedError,
    InplaceModificationError,
    ShapeError,
)
from gradlab.autograd import Function, Node, gradcheck, numerical_gradient, topological_order
from gradlab.creation import (
    arange,
    concat,
    eye,
    from_numpy,
    full,
    linspace,
    ones,
    ones_like,
    rand,
    randn,
    stack,
    zeros,
    zeros_like,
)
from gradlab.tensor import Tensor, as_tensor, tensor

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DTYPE",
    "DTypeError",
    "DomainError",
    "Function",
    "GradLabError",
    "GradientError",
    "GraphReleasedError",
    "InplaceModificationError",
    "Node",
    "ShapeError",
    "Tensor",
    "__version__",
    "arange",
    "as_tensor",
    "concat",
    "data",
    "default_rng",
    "enable_grad",
    "eye",
    "from_numpy",
    "full",
    "gradcheck",
    "is_grad_enabled",
    "linspace",
    "nn",
    "no_grad",
    "numerical_gradient",
    "ones",
    "ones_like",
    "optim",
    "rand",
    "randn",
    "set_grad_enabled",
    "stack",
    "tensor",
    "topological_order",
    "training",
    "zeros",
    "zeros_like",
]
