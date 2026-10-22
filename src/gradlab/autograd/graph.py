"""The computation graph: :class:`Context` and :class:`Node`.

The graph is built eagerly during the forward pass. Its **nodes are operations**
and its **edges are tensors**: a :class:`Node` holds strong references to its
input tensors and a weak reference to the tensor it produced.

Acyclicity is structural. A ``Node`` can only reference tensors that already
existed when it was constructed, so ``seq`` (a monotonically increasing creation
counter) strictly decreases as you walk backwards along any edge. There is no
cycle to detect.
"""

from __future__ import annotations

import itertools
import weakref
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from gradlab.autograd.function import Function
    from gradlab.tensor import Tensor

_seq_counter = itertools.count()


def _next_seq() -> int:
    return next(_seq_counter)


class Context:
    """Scratch space a :class:`~gradlab.autograd.function.Function` uses to carry
    values from its forward pass to its backward pass.

    Saved values are reachable as attributes::

        ctx.save(a=a, b=b)   # in forward
        ctx.a, ctx.b         # in backward

    ``meta`` holds non-array information (axes, shapes, flags) that the
    Microscope can display. Everything in ``meta`` must be JSON-serialisable.
    """

    __slots__ = ("_saved", "meta", "released")

    def __init__(self) -> None:
        self._saved: dict[str, Any] = {}
        self.meta: dict[str, Any] = {}
        self.released = False

    def save(self, **values: Any) -> None:
        """Save values needed by ``backward``."""
        self._saved.update(values)

    def set_meta(self, **values: Any) -> None:
        """Record JSON-serialisable metadata for inspection."""
        self.meta.update(values)

    def release(self) -> None:
        """Drop saved values. Called by the engine unless ``retain_graph=True``."""
        self._saved.clear()
        self.released = True

    @property
    def saved(self) -> dict[str, Any]:
        return self._saved

    def __getattr__(self, name: str) -> Any:
        try:
            return self._saved[name]
        except KeyError:
            if self.released:
                from gradlab._errors import GraphReleasedError

                raise GraphReleasedError(
                    f"Saved value {name!r} was freed after a backward pass. "
                    "Call backward(retain_graph=True) if you need to differentiate "
                    "through this graph more than once."
                ) from None
            raise AttributeError(name) from None


class Node:
    """One operation in the computation graph.

    Attributes are the public, stable surface that ``gradlab.inspection`` reads;
    treat them as API.

    ``op``
        Short operation name, e.g. ``"mul"``, ``"matmul"``, ``"sum"``.
    ``inputs``
        The input :class:`~gradlab.tensor.Tensor` objects, in argument order.
    ``input_versions``
        Each input's ``_version`` at forward time. The engine compares these
        before calling ``backward`` and raises
        :class:`~gradlab._errors.InplaceModificationError` on a mismatch, so an
        in-place edit between forward and backward can never silently corrupt a
        gradient.
    ``output``
        A weak reference to the produced tensor (weak, to avoid a reference
        cycle ``Tensor -> Node -> Tensor``). May return ``None`` once the
        intermediate result has been garbage collected; the engine does not need
        it to propagate gradients.
    ``backward_rule``
        A human-readable derivative, shown verbatim in the Gradient Microscope.
    ``seq``
        Creation index. Strictly increasing in forward order.
    """

    __slots__ = (
        "backward_rule",
        "ctx",
        "fn",
        "input_versions",
        "inputs",
        "op",
        "output",
        "output_dtype",
        "output_shape",
        "released",
        "seq",
    )

    def __init__(
        self,
        op: str,
        fn: type[Function],
        inputs: tuple[Tensor, ...],
        output: Tensor,
        ctx: Context,
        backward_rule: str,
    ) -> None:
        self.op = op
        self.fn = fn
        self.inputs = inputs
        self.input_versions = tuple(t._version for t in inputs)
        self.output = weakref.ref(output)
        self.output_shape = output.shape
        self.output_dtype = output.dtype
        self.ctx = ctx
        self.backward_rule = backward_rule
        self.seq = _next_seq()
        self.released = False

    # -- graph walking ----------------------------------------------------

    @property
    def next_nodes(self) -> tuple[Node | None, ...]:
        """The ``grad_fn`` of each input: where gradient flows next.

        ``None`` marks a leaf input (gradient stops there and is deposited).
        """
        return tuple(t.grad_fn for t in self.inputs)

    def check_versions(self) -> None:
        """Verify no input was modified in place since the forward pass."""
        from gradlab._errors import InplaceModificationError

        for tensor, saved_version in zip(self.inputs, self.input_versions, strict=True):
            if tensor._version != saved_version:
                raise InplaceModificationError(
                    f"Input to {self.op!r} (shape {tensor.shape}) was modified in place "
                    f"after the forward pass: version {saved_version} -> "
                    f"{tensor._version}. Its saved value is stale, so the gradient "
                    "would be wrong. Only mutate a tensor's .data (e.g. in an "
                    "optimiser step) after backward() has finished."
                )

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        """Run this operation's derivative rule. Returns one gradient per input."""
        if self.released:
            from gradlab._errors import GraphReleasedError

            raise GraphReleasedError(
                f"The graph through {self.op!r} was freed by a previous backward pass. "
                "Pass retain_graph=True to backward() to differentiate through the "
                "same graph twice."
            )
        self.check_versions()
        return self.fn.run_backward(self.ctx, grad, self.inputs)

    def release(self) -> None:
        self.ctx.release()
        self.released = True

    def __repr__(self) -> str:
        shapes = ", ".join(str(t.shape) for t in self.inputs)
        return f"<Node {self.op} seq={self.seq} inputs=({shapes}) -> {self.output_shape}>"
