"""The reverse-mode engine: topological ordering and gradient propagation.

This is the component the whole project exists to demonstrate, so it is written
to be read. The algorithm is:

1. Collect every node reachable from the root's ``grad_fn``.
2. Order them so that a node is always processed *before* any node it depends
   on — a reverse topological order.
3. Walk that order once, carrying a gradient for each node's output, calling
   each operation's derivative rule, and **adding** the result into the running
   gradient of each input.
4. Deposit finished gradients on leaf tensors (and on non-leaves that asked for
   it with ``retain_grad()``).

Step 2 is what makes a single pass correct. Because every consumer of a tensor
is processed before the node that produced it, a node's incoming gradient is
already complete (fully summed over all fan-out paths) by the time its rule
runs. No node is ever visited twice, so reverse-mode costs one graph traversal
regardless of how many inputs the function has.
"""

from __future__ import annotations

import numpy as np

from gradlab._errors import GradientError
from gradlab.autograd.graph import Node


def topological_order(root: Node) -> list[Node]:
    """Nodes reachable from ``root``, in reverse topological order (root first).

    Implemented as an iterative depth-first post-order traversal, then reversed.
    Iterative rather than recursive on purpose: a deep graph (thousands of
    chained operations) would otherwise hit Python's recursion limit, and graph
    depth is one of the things ``benchmarks/`` measures.

    Ties are broken by descending ``seq`` so the order is deterministic for a
    given forward pass — the Microscope relies on that to number its steps.
    """
    post: list[Node] = []
    # stack entries: (node, children_expanded?)
    stack: list[tuple[Node, bool]] = [(root, False)]
    seen: set[int] = set()
    while stack:
        node, expanded = stack.pop()
        if expanded:
            post.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        children = [n for n in node.next_nodes if n is not None and id(n) not in seen]
        # push in ascending seq so the highest seq is popped first
        for child in sorted(children, key=lambda n: n.seq):
            stack.append((child, False))
    post.reverse()
    return post


def backward(root, grad=None, retain_graph: bool = False, recorder=None) -> None:
    """Accumulate gradients of ``root`` with respect to every leaf behind it.

    ``grad`` is the upstream gradient (``dL/d root``). It may be omitted only
    when ``root`` holds a single element, in which case the seed is 1.0. For a
    non-scalar root an explicit ``grad`` is **required**: GradLab will not guess
    that every element of a non-scalar output has an upstream gradient of one,
    because that silently computes the gradient of ``root.sum()`` instead.

    ``recorder`` is an optional observer used by ``gradlab.inspection`` to build
    the Gradient Microscope's backward trace. It receives every step as it
    happens:

        recorder.on_backward_step(index, node, upstream_grad, input_grads)

    It exists so there is exactly **one** backward implementation in GradLab.
    The Microscope shows what this function actually did, not a second
    reimplementation of it that could silently drift out of agreement.
    """
    from gradlab.tensor import Tensor

    if not root.requires_grad:
        raise GradientError(
            "backward() called on a tensor that does not require grad. Create the "
            "inputs with requires_grad=True, and check you are not inside no_grad()."
        )

    if grad is None:
        if root.size != 1:
            raise GradientError(
                f"backward() on a non-scalar output of shape {root.shape} needs an "
                "explicit upstream gradient: call backward(gradient) with a tensor "
                f"of shape {root.shape}, or reduce the output to a scalar first "
                "(e.g. .sum() or .mean()). GradLab will not assume a seed of all "
                "ones, because that would silently compute d(output.sum())/dx."
            )
        seed = np.ones_like(root.data)
    else:
        seed = grad.data if isinstance(grad, Tensor) else np.asarray(grad, dtype=root.dtype)
        if seed.shape != root.shape:
            raise GradientError(
                f"Upstream gradient has shape {seed.shape} but the output has shape "
                f"{root.shape}. They must match exactly."
            )

    # A leaf root: nothing to traverse, deposit and return.
    if root.grad_fn is None:
        root._accumulate_grad(seed)
        return

    node_grads: dict[int, np.ndarray] = {id(root.grad_fn): seed}
    order = topological_order(root.grad_fn)
    if recorder is not None:
        recorder.on_seed(root, seed, order)

    # The root itself may want to keep its gradient.
    if root._retain_grad:
        root._accumulate_grad(seed)

    step_index = 0
    for node in order:
        g = node_grads.pop(id(node), None)
        if g is None:
            # Reachable from the root, but on no path that carries gradient
            # (every input of this node had requires_grad=False).
            continue

        # Deposit here, not afterwards: in reverse topological order every
        # contribution to this node's output has already arrived, and the
        # node's saved values are about to be released.
        out = node.output()
        if out is not None and out._retain_grad and out.grad_fn is node:
            out._accumulate_grad(g)

        input_grads = node.backward(g)
        if recorder is not None:
            recorder.on_backward_step(step_index, node, g, input_grads)
            step_index += 1

        for tensor, ig in zip(node.inputs, input_grads, strict=True):
            if ig is None:
                continue
            if tensor.grad_fn is None:
                # A leaf: this is where the gradient comes to rest.
                tensor._accumulate_grad(ig)
                continue
            key = id(tensor.grad_fn)
            prev = node_grads.get(key)
            node_grads[key] = ig if prev is None else prev + ig

        if not retain_graph:
            node.release()


def run_backward(root, grad=None, retain_graph: bool = False, recorder=None) -> None:
    """Public entry point used by :meth:`gradlab.tensor.Tensor.backward`."""
    backward(root, grad=grad, retain_graph=retain_graph, recorder=recorder)
