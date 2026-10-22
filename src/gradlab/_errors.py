"""GradLab's exception hierarchy.

Every error GradLab raises deliberately is a subclass of :class:`GradLabError`,
so user code can catch the whole family. Messages are written to tell the caller
what to *do*, not merely what went wrong.
"""

from __future__ import annotations


class GradLabError(Exception):
    """Base class for every error GradLab raises on purpose."""


class ShapeError(GradLabError, ValueError):
    """An operation received shapes it cannot combine."""


class DTypeError(GradLabError, TypeError):
    """A dtype was requested that the operation cannot support."""


class DomainError(GradLabError, ValueError):
    """A value fell outside an operation's mathematical domain.

    Raised instead of silently producing ``nan`` or ``inf`` — e.g. ``log(-1)``,
    ``sqrt(-1)``, or division by zero.
    """


class GradientError(GradLabError, RuntimeError):
    """Differentiation was requested in a way that is not well defined.

    The most common cause is calling ``backward()`` on a non-scalar tensor
    without supplying an upstream gradient.
    """


class GraphReleasedError(GradientError):
    """A second backward pass was attempted through an already-freed graph."""


class InplaceModificationError(GradientError):
    """A tensor needed for backward was modified in place after the forward pass."""
