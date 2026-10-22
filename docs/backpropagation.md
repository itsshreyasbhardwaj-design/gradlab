# Backpropagation: the derivations

Every rule GradLab implements, derived. Notation: `L` is the scalar output being
differentiated, `g = ∂L/∂out` is the upstream gradient arriving at an operation,
and the goal at each node is `∂L/∂in` for each input.

The chain rule in the form used throughout:

> **∂L/∂in = ∂L/∂out · ∂out/∂in**
>
> upstream gradient × local derivative = downstream gradient

## Elementwise arithmetic

For an elementwise `out = f(a, b)`, each output element depends on exactly the
corresponding input elements, so the Jacobian is diagonal and the chain rule
reduces to an elementwise product — no matrix is ever formed.

| operation | ∂L/∂a | ∂L/∂b |
|---|---|---|
| `a + b` | `g` | `g` |
| `a − b` | `g` | `−g` |
| `a · b` | `b ⊙ g` | `a ⊙ g` |
| `a / b` | `g / b` | `−a ⊙ g / b²` |
| `−a` | `−g` | — |
| `\|a\|` | `sign(a) ⊙ g` | — |

**Product rule.** With `out = a·b`, perturb `a` by `δ`: the output changes by
`b·δ`. So `∂out/∂a = b`, and `∂L/∂a = b ⊙ g`.

**Quotient rule.** `out = a·b⁻¹`. In `a` it is linear, giving `1/b`. In `b`,
`∂(b⁻¹)/∂b = −b⁻²`, so `∂L/∂b = −a ⊙ g / b²`.

## Powers

For a **constant** exponent `n`, `out = aⁿ`:

> **∂L/∂a = n · a^(n−1) ⊙ g**

GradLab validates the domain at forward time: non-integer `n` requires `a ≥ 0`
(else the result is not real), and `n < 0` requires `a ≠ 0`. For `0 < n < 1` the
derivative is unbounded at `a = 0`, so the backward pass raises there rather
than returning `inf`.

For a **tensor** exponent, `out = a^b`, write `a^b = exp(b·ln a)`:

> **∂L/∂a = b · a^(b−1) ⊙ g**  and  **∂L/∂b = a^b · ln(a) ⊙ g**

The second needs `a > 0` for `ln a` to exist, which GradLab checks.

## Transcendental functions

| operation | local derivative | implementation note |
|---|---|---|
| `exp(a)` | `exp(a) = out` | the output is saved, not the input |
| `ln(a)` | `1/a` | requires `a > 0`; raises otherwise |
| `√a` | `1/(2√a) = 1/(2·out)` | unbounded at 0; raises there |
| `sin(a)` | `cos(a)` | |
| `cos(a)` | `−sin(a)` | |

**`tanh`.** From `tanh = (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)`, the quotient rule gives
`sech²(x) = 1 − tanh²(x)`:

> **∂L/∂a = (1 − out²) ⊙ g**

Expressed in the *output*, so the forward value is reused and no exponential is
recomputed.

**`σ` (sigmoid).** With `σ(x) = 1/(1+e⁻ˣ)`:

```
σ'(x) = e⁻ˣ/(1+e⁻ˣ)²  =  σ(x)·(1 − σ(x))
```

> **∂L/∂a = out · (1 − out) ⊙ g**

Note `σ'` peaks at `0.25` and decays to zero in both tails. A unit driven to
`σ(6) ≈ 0.9975` has local derivative `≈ 0.0025` — a 400× attenuation of whatever
arrives. Stack several and the gradient vanishes. This is the **vanishing
gradient problem**, and it is one multiplication, visible in the Microscope's
`sigmoid-saturation` trace.

GradLab evaluates `σ` piecewise so `exp` never sees a positive argument:

```python
out[x >= 0] = 1 / (1 + exp(-x))        # exp of a non-positive number
out[x <  0] = exp(x) / (1 + exp(x))    # exp of a negative number
```

**`relu`.** `max(a, 0)` is differentiable everywhere except `a = 0`:

> **∂L/∂a = g where a > 0, else 0**

A gate, not a scaling. The subgradient at exactly 0 is **defined as 0** — a
choice, documented and tested, not an accident. Any value in `[0, 1]` would be a
valid subgradient. The consequence of the gate: a unit whose pre-activation is
negative for every input receives *exactly zero* gradient forever — a "dead
ReLU".

## Reductions

A reduction collapses axes, so its gradient must **expand** back along exactly
those axes.

**Sum.** `out = Σᵢ aᵢ`. Each `aᵢ` contributes with coefficient 1:

> **∂L/∂a = broadcast(g) over the reduced axes**

**Mean.** `out = (1/N)·Σᵢ aᵢ`:

> **∂L/∂a = broadcast(g)/N**

**Max / min.** Not differentiable at ties. GradLab's documented choice is to
**split the gradient equally among all tied extrema**:

> **∂L/∂aᵢ = g/k if aᵢ is extremal, else 0** (k = number of tied extrema)

This is a valid subgradient, symmetric under permutation of the tied inputs, and
it matches `torch.amax`/`torch.amin` so the conformance tests can compare
directly. (`torch.max(dim=…)` instead routes the whole gradient to one index —
an equally valid, different choice.)

The `keepdims` subtlety: when the forward pass dropped the reduced axes, the
incoming gradient has fewer dimensions than the input, so those axes must be
re-inserted with `expand_dims` before broadcasting.

## Broadcasting — the one most hand-rolled engines get wrong

Broadcasting makes an input *act as if* it had a larger shape. Differentiation
must undo that, and the undo is a **sum**.

Adding a bias `b` of shape `(3,)` to a batch `W` of shape `(32, 3)`: `b₀` was
used by all 32 rows, so `∂L/∂b₀` is the sum of 32 separate contributions.

> **The gradient of a broadcast input is the sum over the axes that were
> stretched.**

Two distinct reductions are needed, and `unbroadcast` does both:

1. **Leading axes the input did not have at all** are summed away entirely.
   `(32, 3)` gradient → `(3,)` input: sum over axis 0.
2. **Axes the input had as size 1** are summed with `keepdims=True`.
   `(4, 3)` gradient → `(4, 1)` input: sum over axis 1, keeping the dimension.

```python
unbroadcast(np.ones((32, 3)), (3,))   # -> [32., 32., 32.]   not ones
unbroadcast(np.ones((4, 5)), ())      # -> 20.0
```

The trap: *two arrays having the same output shape does not mean their gradients
have that shape*. Every elementwise rule in GradLab returns output-shaped
gradients and lets `Function.apply` reduce them, so this is handled in exactly
one place.

## Matrix multiplication

For `C = A·B` with `A: (m,k)`, `B: (k,n)`, `C: (m,n)`:

```
C[i,j] = Σₚ A[i,p]·B[p,j]
```

`A[i,p]` appears in `C[i,j]` for every `j`, with coefficient `B[p,j]`:

```
∂L/∂A[i,p] = Σⱼ g[i,j]·B[p,j] = (G·Bᵀ)[i,p]
```

and symmetrically for `B`:

> **∂L/∂A = G · Bᵀ**
> **∂L/∂B = Aᵀ · G**

The transposes are not a trick — they are what makes the shapes work:
`(m,n)·(n,k) = (m,k) = shape(A)`, and `(k,m)·(m,n) = (k,n) = shape(B)`.

A useful sanity check: **a gradient always has the shape of the thing it
differentiates.** If your matmul backward produces the wrong shape, you have
transposed the wrong operand.

GradLab's implementation also handles what NumPy's `matmul` does around the
core rule: 1-D operands are promoted to matrices (and the result squeezed back),
and leading batch axes broadcast. Promotion is undone by reshaping the gradient
to the original shape; batch broadcasting is undone by the same `unbroadcast`
helper the elementwise ops use.

## Softmax and log-softmax

`softmax(x)ᵢ = e^xᵢ / Σⱼ e^xⱼ`. Its Jacobian is dense:

```
∂pᵢ/∂xⱼ = pᵢ(δᵢⱼ − pⱼ)
```

Contracting with `g` avoids ever forming that matrix:

> **∂L/∂x = p ⊙ (g − Σⱼ pⱼ·gⱼ)**

For **log-softmax** the algebra is cleaner still. Using
`log softmax(x)ᵢ = xᵢ − log Σⱼ e^xⱼ`:

```
∂/∂xⱼ [xᵢ − log Σ e^x] = δᵢⱼ − pⱼ
```

> **∂L/∂x = g − p · Σⱼ gⱼ**

**Why log-sum-exp.** Computing `log(softmax(x))` naively exponentiates, divides,
then takes a log — and `e^1000` overflows while `e^-1000` underflows to exactly
0, making the log `-inf`. Subtracting the row maximum first:

```
log softmax(x)ᵢ = (xᵢ − m) − log Σⱼ e^(xⱼ − m)      where m = max x
```

Every exponent is now `≤ 0`, so `exp` cannot overflow, and the log is applied to
a sum that is at least 1. GradLab's `log_softmax` is finite for logits of ±1000;
the conformance tests check it against PyTorch at exactly those values.

## Binary cross-entropy from logits

The textbook form `−[y·log σ(x) + (1−y)·log(1−σ(x))]` evaluates a log of a
quantity that underflows to 0 around `|x| ≈ 40`, giving `inf` then `nan`.
Folding `σ` into the log algebraically:

> **L = max(x, 0) − x·y + log(1 + e^−|x|)**

Equivalent for all `x`, and its only exponential has a non-positive argument.
The gradient collapses to something remarkable:

> **∂L/∂x = σ(x) − y**

The prediction error, exactly. (This simplicity is why logit-space losses are
standard: the sigmoid's derivative cancels against the cross-entropy's.)

## Indexing

`out = a[idx]` reads selected elements. Backward is a **scatter-add**:

> **∂L/∂a = zeros(a.shape) with g added at the read positions**

Add, not assign. `x[[0, 0, 1]]` reads element 0 twice, so element 0 must receive
the sum of both upstream gradients. `np.add.at` performs exactly this
accumulation; plain fancy-index assignment would keep only the last write and
silently halve the gradient.

## Shape operations

Reshape, transpose, flatten change no values, so each backward rule is purely
the **inverse rearrangement**: reshape the gradient back to the input's shape,
transpose it by the inverse permutation. All return gradients already in input
shape, so they set `auto_unbroadcast = False`.

## Putting it together: one linear layer

For `y = x·W + b` with `x: (N, dᵢₙ)`, `W: (dᵢₙ, dₒᵤₜ)`, `b: (dₒᵤₜ,)`:

```
∂L/∂x = G · Wᵀ          (N, dout)·(dout, din) = (N, din)   ✓
∂L/∂W = xᵀ · G          (din, N)·(N, dout)   = (din, dout) ✓
∂L/∂b = Σ_batch G       (N, dout) summed over N = (dout,)   ✓
```

That last line is the broadcasting rule doing its job: `b` was stretched across
all `N` rows in the forward pass, so its gradient sums over them. Three
derivations, three shape checks, and a layer trains.
