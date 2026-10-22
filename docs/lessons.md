# Guided lessons

Fourteen lessons that build from "what is a tensor" to "why Adam needs less
tuning than SGD". Each one is backed by runnable code:

```bash
python examples/lessons.py          # all fourteen
python examples/lessons.py 7        # just lesson 7
python examples/lessons.py --list   # titles
```

**Every number quoted below was produced by that script.** Where a lesson claims
a value, you can reproduce it in one command. Nothing here is illustrative.

The lessons assume no machine-learning background, but they do assume you can
read Python and are comfortable with the idea of a derivative.

---

## 1. What is a tensor?

A tensor is an n-dimensional array that **remembers where it came from**. The
array part is NumPy's; the remembering is what makes it differentiable.

```python
x = gl.Tensor(3.0, requires_grad=True)
x.is_leaf          # True   — nothing produced it
x.grad_fn          # None

y = x * 2
y.is_leaf          # False
y.grad_fn          # <Node mul ...>
```

That `grad_fn` is the entire difference between a tensor and an array. It is the
edge back into the computation graph, and it is what `backward()` follows.

Rank carries no special meaning: a scalar is `shape=()`, a vector `(3,)`, a
matrix `(2, 2)`. Batch dimensions are a convention of the *layers*, not of the
tensor — `nn.Linear` treats the last axis as features and everything before it
as batch.

**Try it:** construct `gl.Tensor([1, 2], dtype="int64", requires_grad=True)` and
read the error. It explains which dtypes can carry a gradient and why.

---

## 2. Why neural networks need derivatives

Training means choosing parameters that make a loss small. The loss depends on
millions of numbers and we cannot try them all. The derivative answers the only
question we can afford: *if I nudge this parameter, does the loss go up or down,
and how fast?*

Minimising `(w − 3)²` from `w = 0`, stepping against the gradient with `lr=0.3`:

| step | w | loss | dL/dw |
|---:|---:|---:|---:|
| 0 | 0.000000 | 9.00000000 | −6.000000 |
| 1 | 1.800000 | 1.44000000 | −2.400000 |
| 2 | 2.520000 | 0.23040000 | −0.960000 |
| 3 | 2.808000 | 0.03686400 | −0.384000 |
| 4 | 2.923200 | 0.00589824 | −0.153600 |
| 5 | 2.969280 | 0.00094372 | −0.061440 |

The gradient is `2(w − 3)`: negative while `w < 3`, so stepping *against* it
increases `w`; and shrinking as `w` approaches 3, so the steps get smaller on
their own. That is gradient descent, complete, in one parameter.

---

## 3. The chain rule

If `y` depends on `u` and `u` depends on `x`:

> **dy/dx = dy/du · du/dx**

A deep network is this rule applied a few hundred times. Nothing else.

For `y = tanh(3x)` at `x = 0.5`:

```
u  = 3x             = 1.5000000000
y  = tanh(u)        = 0.9051482536
dy/du = 1 − tanh²u  = 0.1807066389     <- GradLab agrees exactly
du/dx               = 3.0000000000
dy/dx = product     = 0.5421199168     <- GradLab agrees exactly
```

Each backward step multiplies by **exactly one** local derivative.

Notice `dy/du = 0.1807` — smaller than 1. Chain several of those and the product
goes to zero. **The vanishing gradient problem is nothing more than repeated
multiplication by numbers below 1.**

---

## 4. Computational graphs

GradLab records operations as they run — "define by run", not compile-ahead.
Nodes are operations; edges are tensors.

```
graph: 4 nodes, depth 3
  ops: add x1, mul x2, tanh x1
  leaves: 2
```

**Why acyclic:** a node can only reference tensors that already existed when it
was built. Time runs one way, so there is no cycle to detect.

**Why that matters:** an acyclic graph has a topological order, and a topological
order is exactly what lets the backward pass visit every node once.

**Try it:** `gradlab-microscope serve`, then use the Graph tab on any preset.

---

## 5. Forward propagation

A layer is a weighted sum followed by a squashing function. The forward pass
computes the output *and* records what each operation will need to be
differentiated later.

```
x      [[1.0, 2.0]]
W      shape (2, 3)
b      [0.0, 0.0, 0.0]
x @ W  [[-1.818402, 0.868309, 0.802928]]
+ b    [[-1.818402, 0.868309, 0.802928]]
tanh   [[-0.948679, 0.700514, 0.66567]]
```

**Try it:** stack three `Linear` layers with no activation, multiply the three
weight matrices by hand, and confirm the result equals a single layer. That is
lesson 11's punchline, discovered early.

---

## 6. Reverse-mode automatic differentiation

* **Forward mode:** one pass gives d(every output)/d(one input).
* **Reverse mode:** one pass gives d(one output)/d(every input).

Training has ~10⁶ inputs and 1 output, so reverse mode is a million times
cheaper. **That asymmetry is why deep learning is computationally possible.**

For `f(x) = x² + 2x` at `x = 3`, the real engine trace:

```
f(3) = 15.0
seed: dL/dL = 1.0

step 1: add   upstream [1.0]
         -> t3 gets [1.0]
         -> t4 gets [1.0]
step 2: mul   upstream [1.0]
         -> x  gets [3.0]
         -> x  gets [3.0]
step 3: mul   upstream [1.0]
         -> x  gets [2.0]

accumulated on x: [8.0]      (f'(x) = 2x + 2 = 8)
```

Three contributions — 3, 3, 2 — summed into 8.

---

## 7. Gradient accumulation

When a value is used more than once, its gradient is the **sum** of the
contributions from every use. Not the last. Not the largest.

With `y = x·x` and `z = y + y` at `x = 2`: `z = 2x²`, so `dz/dx = 4x = 8`.
GradLab gives exactly 8.

Accumulation also happens **across** `backward()` calls:

```
backward() call 1: x.grad = 6.0
backward() call 2: x.grad = 12.0
backward() call 3: x.grad = 18.0
after zero_grad(): None
```

This is deliberate — it is what makes gradient accumulation over micro-batches
possible. The price is that forgetting `zero_grad()` silently sums gradients
across training steps.

**Try it:** remove `optimizer.zero_grad()` from a training loop and watch the
loss behave strangely. This is the most common beginner bug in deep learning.

---

## 8. Broadcasting and gradient reduction

Broadcasting lets a small tensor act as if it were large. Differentiating it
requires undoing that — **and the undo is a sum.**

`W` is `(4, 3)`, `b` is `(3,)`, so `W + b` broadcasts `b` across all 4 rows:

```
W.grad  shape (4, 3), all ones
b.grad  = [4.0, 4.0, 4.0]       <- NOT ones
```

`b[0]` was used by all four rows, so its gradient is the sum of four
contributions. In a real layer with a batch of 16, the bias gradient is `[16.0,
16.0, 16.0]`.

> **A gradient always has the shape of the thing it differentiates.**

This is the single most common bug in a hand-written autodiff engine: two arrays
having the same *output* shape does not mean their gradients have that shape.

---

## 9. Gradient descent

```
θ ← θ − lr · dL/dθ
```

That is the whole algorithm. Fitting `y = wx + b` where the true `w = -1.82936`
and `b = 2.0`:

| step | loss | w | b |
|---:|---:|---:|---:|
| 0 | 8.752029 | 0.47445 | 0.00000 |
| 8 | 0.322183 | -1.37124 | 1.64448 |
| 16 | 0.021799 | -1.74029 | 1.93060 |
| 24 | 0.010908 | -1.81385 | 1.98065 |
| 32 | 0.010508 | -1.82847 | 1.98945 |
| 40 | 0.010493 | -1.83137 | 1.99100 |

Recovered `w = -1.83150` against a true `-1.82936`, and `b = 1.99107` against 2.0.

---

## 10. Backpropagation through a neural network

Everything so far, applied to a real model. The loss flows backward through
every layer, and each parameter receives a gradient shaped like itself:

| parameter | shape | grad shape | ‖grad‖ |
|---|---|---|---:|
| `0.weight` | (2, 4) | (2, 4) | 0.284958 |
| `0.bias` | (4,) | (4,) | 0.055408 |
| `2.weight` | (4, 1) | (4, 1) | 0.431117 |
| `2.bias` | (1,) | (1,) | 0.005105 |

Every gradient has exactly its parameter's shape. Internalise that invariant: if
a gradient's shape is wrong, the derivation is wrong, and no amount of training
will fix it.

---

## 11. Why activation functions matter

Two `Linear` layers with nothing between them compose into **one**:

> (xW₁ + b₁)W₂ + b₂ = x(W₁W₂) + (b₁W₂ + b₂)

Measured directly: stacking two layers versus the single equivalent layer gives
a maximum difference of **1.665e-16** — machine epsilon. They are the same
function.

So depth without nonlinearity buys exactly nothing. On XOR, same architecture,
same seed, same data, 200 epochs:

| model | accuracy |
|---|---:|
| `Linear(2,8) → Linear(8,1)` | **0.6267** |
| `Linear(2,8) → Tanh → Linear(8,1)` | **0.9933** |

One `tanh`, same parameter count, and the problem becomes solvable. No
hyperplane separates XOR; a curved boundary does.

---

## 12. Why learning rates matter

For `L(w) = (w − t)²`, one step is `w ← w − lr·2(w − t)`, so the distance to the
target is multiplied by `|1 − 2·lr|` each step. **That gives an exact stability
threshold: convergence needs lr < 1.**

Measured, starting at `w = 0` with target 3, after 20 steps:

| lr | \|1−2·lr\| | after 20 steps | verdict |
|---:|---:|---:|---|
| 0.05 | 0.90 | 2.635270 | converged (slowly) |
| 0.3 | 0.40 | 3.000000 | converged |
| 0.5 | 0.00 | 3.000000 | **exact in one step** |
| 0.9 | 0.80 | 2.965412 | converged |
| 1.0 | 1.00 | 0.000000 | oscillates forever |
| 1.1 | 1.20 | −112.012800 | diverges |

`lr = 0.5` makes the factor exactly zero and lands on the target in a single
step. `lr = 1.0` bounces between two points forever. `lr = 1.1` diverges
geometrically.

Real losses are not quadratic, but the mechanism is identical: too small and you
never arrive, too large and each step overshoots what it aimed at. The
Microscope's "Learning rates" trace shows the same effect on a real network
across four decades.

---

## 13. Overfitting and validation

Training loss measures memorisation; validation loss measures generalisation.
They agree early and diverge later.

A deliberately oversized model (4417 parameters) on 72 noisy samples:

| epoch | train loss | val loss | gap |
|---:|---:|---:|---:|
| 1 | 0.490081 | 0.417895 | −0.072186 |
| 25 | 0.314977 | 0.358667 | +0.043689 |
| 100 | 0.306188 | 0.363344 | +0.057156 |
| 200 | 0.023499 | 0.415218 | +0.391720 |
| 400 | 0.002324 | 0.694942 | +0.692618 |

Best validation loss **0.267137 at epoch 157 of 400**. After that the training
loss keeps falling — all the way to 0.0023, near-perfect memorisation — while
validation loss **triples**.

Everything learned after epoch 157 was about these particular 72 samples, not
about the problem. Without a validation split you would see only the training
curve, which looks like excellent progress throughout.

---

## 14. How Adam differs from SGD

SGD steps by the gradient. Adam keeps running averages of `g` and `g²` and steps
by the first divided by the root of the second — which makes the step roughly
**scale-invariant**.

Identical gradient sequences, 1000× apart, 10 steps from `θ = 1.0`:

| | SGD | Adam |
|---|---:|---:|
| gradient x1 | 0.94750000 | 0.89838666 |
| gradient x1000 | -51.50000000 | 0.89838666 |

SGD's result scales with the gradient and overshoots catastrophically. Adam's
is identical to all shown digits. **That is why Adam needs so much less
learning-rate tuning.**

Bias correction, which matters most at step 1 with `lr = 0.1`:

| gradient | first step |
|---:|---:|
| 0.001 | 0.09999900 |
| 1.0 | 0.10000000 |
| 1000.0 | 0.10000000 |

Whatever the gradient's magnitude, the first step has size `lr`. Without the
`1/(1−βᵗ)` correction, the moving averages start at zero and the first step
would be roughly 30× too large. (The 0.001 row falls short by exactly
`ε/(|g|+ε)` — the one place `ε` is visible.)

---

## Where to go next

* **`docs/backpropagation.md`** — every derivative rule, derived.
* **`docs/autograd-engine.md`** — how the engine is actually built.
* **The Gradient Microscope** — `gradlab-microscope serve` — step through any of
  these graphs forward and backward, one operation at a time.
* **`docs/mastery-guide.md`** — the questions you should be able to answer, and
  exercises to implement yourself.
