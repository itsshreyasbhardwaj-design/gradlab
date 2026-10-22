# The autograd engine

This is the component the project exists for. It is about 150 lines of Python
across `autograd/graph.py`, `autograd/function.py` and `autograd/engine.py`, and
this document explains all of it.

## The problem

You have a function built out of many small operations, and you want the
derivative of its single scalar output with respect to every one of its inputs.
A neural network has thousands to billions of inputs (its parameters) and one
output (the loss).

Three ways to get those derivatives:

**Symbolic differentiation** — manipulate the expression algebraically. Exact,
but the expression for a derivative grows explosively with depth, and a network
is not written as a closed-form expression anyway.

**Numerical differentiation** — evaluate `(f(x+ε) − f(x−ε)) / 2ε` per input.
Needs two full forward passes *per parameter*. For a million parameters that is
two million forward passes to get one gradient. GradLab uses this only to
*check* its answers (see `autograd/gradcheck.py`), never to train.

**Automatic differentiation** — apply the chain rule mechanically to the
operations that were actually executed. Exact to floating-point, and in reverse
mode it costs one extra traversal of the graph regardless of how many inputs
there are. That last property is the whole reason deep learning is feasible.

## Why *reverse* mode

The chain rule for a composition `f: Rⁿ → Rᵐ` is a product of Jacobians. You can
associate that product from either end:

* **Forward mode** propagates derivatives input-to-output. One pass gives you the
  derivative of *every output* with respect to *one input*. Cost scales with the
  number of inputs.
* **Reverse mode** propagates output-to-input. One pass gives you the derivative
  of *one output* with respect to *every input*. Cost scales with the number of
  outputs.

Training has n ≈ 10⁶ parameters and m = 1 loss, so reverse mode wins by a factor
of n. The price is memory: you cannot walk the graph backwards until you have
built it, so every intermediate value needed by a derivative rule must be kept
alive through the forward pass. **Reverse mode trades memory for time**, and
that trade is visible in `docs/benchmarks.md` section 4.

## The graph

GradLab builds the graph *eagerly, while the forward pass runs* — "define by
run". There is no compilation step and no static graph; the graph is simply a
record of what happened.

Nodes are **operations**; edges are **tensors**.

```
    x ────┐
          ├──▶ [mul] ──▶ t0 ──┐
    x ────┘                   ├──▶ [add] ──▶ t2
                              │
    2 ────┐                   │
          ├──▶ [mul] ──▶ t1 ──┘
    x ────┘
```

That is `y = x*x + 2*x`. Note `x` appears on three edges: fan-out is represented
by several nodes referencing the *same* tensor object, never by copying.

Each `Node` records:

| field | purpose |
|---|---|
| `op` | the operation's name, e.g. `"mul"` |
| `inputs` | the input tensors, in argument order (strong references) |
| `input_versions` | each input's version counter at forward time |
| `output` | a **weak** reference to the tensor produced |
| `ctx` | values saved for backward, plus JSON-safe metadata |
| `backward_rule` | the derivative, as readable text |
| `seq` | a monotonic creation index |

Two of those deserve explanation.

**`output` is weak** because `Tensor` holds a strong reference to its `grad_fn`.
A strong back-reference would make every graph an uncollectable reference cycle,
and a training loop would leak one graph per step.

**`input_versions` exists** so an in-place modification cannot silently corrupt a
gradient. See "Why a version counter" below.

### Acyclicity

The graph is acyclic by construction, not by checking. A `Node` can only
reference tensors that already existed when it was built, and `seq` increases
monotonically, so `seq` strictly decreases as you follow any edge backwards.
There is no cycle to find.

## Defining an operation

Every differentiable operation subclasses `Function` and supplies the
mathematics only:

```python
class Mul(Function):
    op = "mul"
    backward_rule = "d(a*b)/da = b*g ;  d(a*b)/db = a*g"

    @staticmethod
    def forward(ctx, a, b):        # numpy arrays in
        ctx.save(a=a, b=b)         # keep what backward needs
        return a * b               # numpy array out

    @staticmethod
    def backward(ctx, g):          # g is the upstream gradient
        return ctx.b * g, ctx.a * g
```

`Function.apply` does everything else: unwrap `.data`, call `forward`, wrap the
result, and — only if gradient mode is on and at least one input requires grad —
build the `Node`.

Rules return gradients in the **output** shape. `apply` then reduces each one to
its input's shape with `unbroadcast`, so no rule reimplements broadcast
reduction. Rules whose gradients are already in input shape (`matmul`,
reductions, `reshape`) opt out with `auto_unbroadcast = False`.

## The backward pass

```python
def backward(root, grad=None, retain_graph=False, recorder=None):
```

### 1. Seed

The pass starts from `dL/dL = 1`. If `root` holds one element and no `grad` was
given, the seed is `ones_like(root)`.

If `root` is **not** scalar and no `grad` was given, GradLab raises. It will not
assume a seed of all ones, because that silently computes `d(root.sum())/dx` —
a different function — and the user would never find out.

### 2. Ordering

```python
def topological_order(root):  # reverse topological order, root first
```

An iterative depth-first post-order traversal, reversed. Iterative rather than
recursive because a 5000-deep chain would exceed Python's recursion limit, and
graph depth is something the benchmarks deliberately push.

This ordering is what makes a single pass correct. Because every consumer of a
tensor is processed before the node that produced it, a node's incoming gradient
is **already complete** — fully summed over every fan-out path — by the time its
rule runs. No node is visited twice.

### 3. Accumulation

Gradients are accumulated in a dictionary keyed by node identity. When a node's
rule returns a gradient for an input:

* if the input is a **leaf**, the gradient is added into `leaf.grad`;
* otherwise it is added into the running total for the input's producer.

**Added**, not assigned. That is the entire reason `x` in `x*x + 2*x` gets `8`
and not `2` or `6`: both branches contribute and the contributions sum. If you
remember one thing about reverse mode, make it this.

### 4. Release

Unless `retain_graph=True`, each node's saved values are freed as it is
processed and the node is marked released. A second `backward()` through a
released node raises `GraphReleasedError` naming the fix. It never returns a
quietly wrong number.

### 5. Why a version counter

Consider:

```python
y = x * x          # Mul saved a reference to x's buffer
x.data = ...       # the buffer is overwritten
y.backward()       # the saved value is now the NEW x
```

The gradient would be computed from the wrong value, silently. So every `Tensor`
carries `_version`, incremented on any in-place write, and every `Node` records
its inputs' versions at forward time. At backward time the engine compares, and
raises `InplaceModificationError` on a mismatch.

This is what turns "call `optimizer.step()` only after `backward()`" from
documentation into an enforced invariant.

## Gradient mode

`no_grad()` is a reentrant context manager over a module-level depth flag.
Inside it `Function.apply` skips node creation entirely, so results are leaves
with `requires_grad=False`. Optimiser updates and evaluation run inside it.

The flag is process-global, not thread-local: **GradLab is not thread-safe**.
See `docs/limitations.md`.

## The recorder hook

`backward()` takes an optional `recorder`, an observer that receives every step
as it happens. `gradlab.inspection` passes one in to build the Gradient
Microscope's backward trace.

This exists so there is exactly **one** backward implementation in GradLab. The
Microscope shows what this function actually did, rather than a second
reimplementation that could drift out of agreement without anyone noticing.

## Complexity

For a graph with `E` operations:

* **Forward**: `O(E)` node allocations on top of the arithmetic.
* **Backward**: `O(E)` — each node's rule runs exactly once.
* **Memory**: `O(E)` live `Node` objects plus whatever the rules saved.

Measured per-node costs are in `docs/benchmarks.md` section 3; they are flat in
depth, as the analysis predicts.

## What is deliberately absent

* **Higher-order derivatives.** Backward runs on NumPy arrays, not tensors, so
  the backward pass builds no graph of its own and cannot be differentiated
  again. Supporting it would mean rewriting every rule in terms of GradLab
  operations.
* **In-place differentiable ops.** No `add_`, no mutating `__iadd__`.
* **Thread safety, GPU, sparse tensors, complex dtypes.**

Each is listed in `docs/limitations.md` and raises rather than approximating.
