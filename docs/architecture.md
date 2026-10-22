# GradLab architecture

This document is the Phase-1 contract. Everything in `src/gradlab` is written to
satisfy it, and the tests in `tests/` are written to check it. Where the
implementation had to deviate, this document was updated — it is meant to
describe the code that exists, not an aspiration.

## 1. Layering and dependency direction

```
            ┌──────────────────────────────┐
            │ apps/microscope (static web) │   no Python import of gradlab
            └──────────────┬───────────────┘
                           │ reads JSON trace files
            ┌──────────────▼───────────────┐
            │ gradlab.inspection           │   stable, public graph/trace API
            └──────────────┬───────────────┘
      ┌────────────┬───────┴───────┬──────────────┐
      │ gradlab.nn │ gradlab.optim │ gradlab.data │ gradlab.training
      └────────────┴───────┬───────┴──────────────┘
                           │
            ┌──────────────▼───────────────┐
            │ gradlab.ops (forward+backward rules)
            └──────────────┬───────────────┘
            ┌──────────────▼───────────────┐
            │ gradlab.autograd (Function, Node, engine)
            └──────────────┬───────────────┘
            ┌──────────────▼───────────────┐
            │ gradlab.tensor (Tensor)      │
            └──────────────┬───────────────┘
            ┌──────────────▼───────────────┐
            │ numpy  (array storage + kernels only)
            └──────────────────────────────┘
```

Dependency rules, enforced by `tests/unit/test_architecture.py`:

* No module in `src/gradlab` imports `torch`, `tensorflow`, `jax`, or `autograd`.
  PyTorch appears only under `tests/conformance/` and `benchmarks/`.
* `gradlab.autograd` does not import `gradlab.nn`, `gradlab.optim`, or
  `gradlab.ops` at module scope. `tensor.py` imports op modules lazily inside
  methods to attach operators without creating an import cycle.
* `gradlab.inspection` reads only the documented public attributes of `Node` and
  `Tensor`. It is the *only* sanctioned way for a UI to read the graph.

## 2. What is actually ours vs. NumPy's

GradLab implements: the `Tensor` wrapper and its semantics, the computation
graph, the reverse-mode engine (topological ordering, gradient accumulation,
graph lifetime), every derivative rule, broadcasting-gradient reduction, the
neural-network layer/loss library, the optimisers, the training loop, and the
graph-inspection/trace format.

NumPy provides: contiguous array storage, dtype handling, elementwise kernels,
and `matmul` (which dispatches to a BLAS). **GradLab therefore makes no claim to
a custom high-performance matrix kernel.** See `docs/benchmarks.md`.

## 3. Tensor contract

See `docs/tensor-design.md` for the full semantics. Summary of the invariants
the engine relies on:

* `Tensor.data` is always a `numpy.ndarray` (0-d for scalars, never a Python float).
* `requires_grad` may only be `True` for inexact (floating) dtypes.
* `is_leaf == (grad_fn is None)`. A leaf is user-created or `detach()`ed.
* `grad` is `None` or a `Tensor` with `grad.shape == self.shape` and
  `grad.requires_grad is False`.
* `_version` is a monotonically increasing integer bumped on every in-place
  mutation of the underlying buffer.

## 4. Graph contract

The graph is built eagerly during the forward pass ("define-by-run"). It is a
directed acyclic graph whose **nodes are operations** and whose **edges are
tensors**.

* Each non-leaf `Tensor` holds exactly one `grad_fn: Node`.
* A `Node` records: `op` (string name), `inputs` (the input `Tensor`s),
  `output` (a weak reference to the produced `Tensor`), `ctx` (saved values and
  metadata needed for backward), `backward_rule` (a human-readable string), and
  `seq` (a monotonically increasing creation index, used for deterministic
  ordering and for the Microscope's forward-step order).
* Acyclicity is structural, not checked: a `Node` can only reference tensors
  that already existed when it was constructed, so no edge can point forward in
  time. `seq` is strictly increasing along every edge.
* Fan-out is represented by several `Node`s listing the same `Tensor` in their
  `inputs`. Nothing is duplicated.

## 5. Differentiation semantics

`out.backward(grad=None, retain_graph=False)`:

1. **Seed.** If `out` is scalar (`out.size == 1`) and `grad is None`, the seed
   is `1.0`. If `out` is non-scalar and `grad is None`, raise `GradientError`.
   GradLab never silently assumes a seed of all-ones for a non-scalar output.
2. **Order.** Build the set of nodes reachable from `out.grad_fn`, then process
   them in a reverse topological order (an iterative DFS post-order, reversed;
   ties broken by descending `seq` for determinism).
3. **Accumulate.** Gradients are accumulated in an engine-local buffer keyed by
   tensor identity. A node's backward is only called once all of its output's
   consumers have contributed, which reverse-topological order guarantees.
4. **Deposit.** After traversal, for every leaf with `requires_grad=True`, the
   accumulated gradient is *added* to `leaf.grad` (`leaf.grad` is created if
   `None`). Non-leaf tensors get `.grad` only if `retain_grad()` was called on
   them. Accumulation across repeated `backward()` calls is intentional and
   mirrors the convention optimisers rely on; clear it with `zero_grad()`.
5. **Release.** Unless `retain_graph=True`, every visited node's saved values
   are freed and the node is marked `released`. A second `backward()` through a
   released node raises `GraphReleasedError` with a message naming
   `retain_graph=True` — it never returns a silently wrong number.
6. **Version check.** When a node saved a tensor for backward it also saved that
   tensor's `_version`. If the version differs at backward time, the engine
   raises `InplaceModificationError`. This is what makes the "mutate a parameter
   only after backward" rule enforceable rather than merely documented.

### Gradient mode

`gradlab.no_grad()` is a reentrant context manager backed by a module-level
flag. Inside it, `Function.apply` skips node creation entirely, so results are
leaves with `requires_grad=False`. Optimiser updates run inside it.
`gradlab.enable_grad()` re-enables inside a `no_grad` region.
The flag is process-global, not thread-local; GradLab is not thread-safe and
`docs/limitations.md` says so.

## 6. Operation contract

Every differentiable operation is a subclass of `gradlab.autograd.Function`:

```python
class Mul(Function):
    op = "mul"
    backward_rule = "d(a*b)/da = b·g ; d(a*b)/db = a·g"

    @staticmethod
    def forward(ctx, a, b):            # a, b are numpy arrays
        ctx.save(a=a, b=b)
        return a * b

    @staticmethod
    def backward(ctx, g):              # g is a numpy array
        return ctx.a * g, ctx.b * g    # one grad per input, pre-unbroadcast
```

`Function.apply(*tensors, **kwargs)` does the bookkeeping: unwraps `.data`,
calls `forward`, wraps the result, and — if grad is enabled and any input
requires grad — creates the `Node`. Backward rules return gradients in the
*broadcast output shape*; `Function.apply` installs an automatic
`unbroadcast(grad, input.shape)` step so no rule has to reimplement reduction
over broadcast axes. A rule that needs non-standard reduction (e.g. `matmul`)
returns gradients already in input shape and sets `auto_unbroadcast = False`.

## 7. Numerical policy

* Default dtype is `float64`; it is the dtype all correctness tolerances are
  stated for. `float32` is supported and tested with looser tolerances.
* `log` of a non-positive value, `sqrt` of a negative, and `0**negative` raise
  `DomainError` at forward time rather than producing `nan`/`inf` silently.
* Division by zero raises `DomainError`.
* Cross-entropy and `sigmoid`/`softmax` use log-sum-exp / `logaddexp`
  formulations; the naive `log(softmax(x))` form is never used.
* NumPy `RuntimeWarning`s are promoted to errors in the test suite
  (`filterwarnings = ["error::RuntimeWarning"]`) so overflow cannot hide.
* Nothing is silently clipped. `gradlab.training` *detects* non-finite loss and
  exploding gradient norms and reports them; it does not alter them unless the
  caller explicitly asks for clipping.

## 8. Determinism

Every stochastic component takes an explicit seed and uses its own
`numpy.random.Generator` — GradLab never touches the global NumPy random state.
Seeded runs are bit-for-bit reproducible on one machine, same Python, same
NumPy, same dtype. Across BLAS implementations or thread counts, `matmul`
reduction order may differ; see `docs/limitations.md`.
