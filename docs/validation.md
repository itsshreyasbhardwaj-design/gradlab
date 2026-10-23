# Validation

How GradLab's gradients are known to be correct, and the measured numbers behind
that claim.

**A green test suite is not a proof.** 506 passing tests mean the behaviours
that were tested are correct on the inputs that were tried. The sections below
describe four independent strategies, each able to catch failures the others
would miss, and report the actual error magnitudes rather than only a verdict.

## Why four strategies

| strategy | catches | blind to |
|---|---|---|
| analytic tests | a wrong rule on a known case | cases nobody wrote down |
| finite differences | any wrong rule, numerically | kinks; non-scalar seeds |
| PyTorch comparison | a rule that is wrong in a way both of my tests shared | anything torch also gets wrong |
| property tests | shape/invariant bugs on inputs nobody thought of | magnitudes and constants |

The combination matters. A hand-written test and a hand-written rule can be
wrong together, because the same person derived both. Finite differences know
nothing about the rule. PyTorch knows nothing about this codebase.

## 1. Analytic tests

Functions with derivatives that can be checked by inspection:

```python
x = Tensor(3.0, requires_grad=True)
y = x * x + 2 * x       # f'(x) = 2x + 2
y.backward()
assert x.grad.item() == 8.0
```

These cover the specification's required edge cases — `x*x` at 3 giving 6,
`y + y` with `y = x*x` at 2 giving 8 — plus fan-out, diamond dependencies,
detach, no-grad, retain_graph, and gradient accumulation across passes. They are
in `tests/unit/test_autograd_engine.py`.

## 2. Finite-difference gradient checking

`gradlab.gradcheck` compares the engine's analytic gradient against a central
difference built purely from forward evaluations:

```
∂f/∂xᵢ ≈ [f(xᵢ+ε) − f(xᵢ−ε)] / 2ε
```

### Why central, and why ε = 1e-6

Central differences have truncation error `O(ε²)` rather than the forward
difference's `O(ε)` — about three extra digits for the same step.

Choosing `ε` is a real trade-off and the reason the tolerances are not
arbitrary. Truncation error falls as `ε²`; floating-point cancellation in
`f(x+ε) − f(x−ε)` grows as `ε⁻¹`. Their sum is minimised near

```
ε ≈ ∛(machine epsilon) ≈ 6×10⁻⁶   for float64
```

Hence the default `eps=1e-6` and tolerances around `1e-7` absolute. A check that
needs looser tolerances than these is telling you something, so the measured
errors are recorded below rather than hidden behind a pass/fail.

### Measured errors

All 36 operations below were checked at float64 with `eps=1e-6`, `atol=1e-7`,
`rtol=1e-5`. **All passed.** Reproduce with `pytest tests/gradients -q`.

| operation | max abs error | max rel error |
|---|---:|---:|
| add | 3.04e-10 | 3.04e-10 |
| sub | 3.04e-10 | 3.04e-10 |
| mul | 3.33e-10 | 4.59e-09 |
| div | 4.25e-10 | 2.03e-09 |
| neg | 3.04e-10 | 3.04e-10 |
| abs | 1.40e-10 | 1.40e-10 |
| pow (n=2) | 2.43e-10 | 1.33e-10 |
| pow (n=−1) | 4.26e-10 | 2.39e-09 |
| pow (n=0.5) | 4.25e-10 | 9.83e-10 |
| pow_tensor | 2.10e-10 | 1.31e-09 |
| exp | 6.32e-10 | 6.56e-10 |
| log | 2.01e-10 | 2.37e-10 |
| sqrt | 8.37e-10 | 2.62e-09 |
| tanh | 1.22e-10 | 4.48e-10 |
| sigmoid | 2.14e-10 | 1.31e-09 |
| relu | 1.40e-10 | 1.40e-10 |
| sin | 1.80e-10 | 5.66e-10 |
| cos | 2.22e-10 | 6.16e-10 |
| sum(axis) | 4.76e-10 | 2.39e-10 |
| mean(axis) | 2.55e-10 | 5.01e-10 |
| max(axis) | 2.46e-10 | 9.35e-11 |
| min(axis) | 3.52e-10 | 1.51e-10 |
| reshape | 1.65e-10 | 8.23e-11 |
| transpose | 4.69e-10 | 1.56e-10 |
| index (repeated) | 3.18e-10 | 6.10e-10 |
| matmul 2-D | 4.45e-10 | 6.72e-10 |
| matmul batched | 7.52e-10 | 3.90e-09 |
| matmul batch-broadcast | 5.99e-10 | 4.20e-09 |
| broadcast bias | 7.48e-10 | 7.48e-10 |
| broadcast mutual | 4.19e-10 | 1.40e-10 |
| log_softmax | **1.04e-08** | **2.31e-08** |
| softmax | 2.03e-10 | 2.14e-08 |
| mse_loss | 5.36e-11 | 3.84e-09 |
| bce_with_logits | 7.80e-11 | 5.75e-10 |
| cross_entropy | 3.36e-10 | 8.07e-09 |
| MLP (4 layers) | 1.51e-09 | 7.17e-09 |

**Worst absolute error across everything: 1.04e-08. Worst relative: 2.31e-08.**

Both worst cases are `log_softmax`, and that is expected rather than concerning:
it involves a difference of logs, so the finite difference suffers more
cancellation than elsewhere. The *analytic* side is independently confirmed
against PyTorch to 1e-10 below.

### Where finite differences are not valid

At a kink the two one-sided limits differ and the central difference returns
their average, which no correct subgradient need match. `relu` at 0, `abs` at 0,
and ties in `max` are all such points. Test inputs are deliberately kept away
from them, and `gradcheck` warns when it detects an input within `10ε` of zero.

### The checker is itself checked

A gradient checker that never fails is worthless.
`test_gradcheck_detects_a_deliberately_wrong_rule` monkey-patches `Mul.backward`
to return `1.5×` the correct gradient and asserts that `gradcheck` raises. It
does.

## 3. Independent comparison with PyTorch

PyTorch is used **only** as an oracle. `tests/unit/test_architecture.py` parses
every shipped module's AST and fails if any of them imports `torch`,
`tensorflow`, `jax` or `autograd` — so agreement here is agreement between two
implementations sharing no code.

Everything runs in float64 on CPU so the comparison is about derivative rules,
not float width.

**Covered** (`tests/conformance/`, 55 tests):

* 33 elementwise, broadcasting, matmul, reduction and shape cases
* `log_softmax` at ordinary and extreme logits (±1000)
* non-scalar upstream gradients
* repeated-index gradients
* `Linear` forward, parameter gradients, input gradients, with and without bias
* `mse_loss`, `binary_cross_entropy_with_logits`, `cross_entropy` — including at
  logits of ±800 where the naive formulations return `inf`
* SGD, SGD+momentum, Nesterov, Adam, AdamW over 20-step trajectories
* a 60-epoch training run, loss curve compared epoch by epoch

**Tolerance: `rtol=1e-10, atol=1e-12`.** All pass.

### The end-to-end result

`examples/pytorch_comparison.py` builds the same 2→16→16→1 network in both
frameworks from identical weights and trains both for 120 epochs:

```
forward pass           max |Δ|  = 0.00e+00
loss (single pass)     |Δ|      = 1.11e-16
parameter gradients    max |Δ|  = 5.55e-17
loss across 120 epochs max |Δ|  = 1.39e-16
final loss             |Δ|      = 0.00e+00
parameters after 120   max |Δ|  = 2.17e-15
```

That is machine epsilon. The residual comes from summation **order**, not from
different mathematics: NumPy's BLAS and PyTorch's ATen block and accumulate
differently, and floating-point addition is not associative. A disagreement at
1e-6 or larger would indicate a genuinely different derivative; 1e-16 is
rounding.

## 4. Property-based testing

Hypothesis generates shapes and values and checks invariants that must hold for
*any* valid input (`tests/property/`, 18 properties, 60 examples each):

* a gradient always has the shape of the tensor it differentiates
* broadcast gradients keep their own input shapes
* `d(αf + βg)/dx = α·df/dx + β·dg/dx` — linearity
* `sum`'s gradient is all ones; `mean`'s is `1/N`
* backward is deterministic — the same graph twice gives bitwise-identical results
* `N` backward passes accumulate exactly `N×` one pass
* reshape and double-transpose round-trip to identity gradients
* `log∘exp` has unit gradient
* detached branches contribute nothing
* softmax rows sum to 1; stable and naive log-softmax agree in the safe range
* an SGD step always moves every component against its gradient

## Structural guarantees

Beyond numerics, `tests/unit/test_architecture.py` enforces:

* **No autodiff framework is imported by any shipped module** — AST-checked per
  file, so it cannot be bypassed by an import inside a function.
* **No `eval`, `exec` or `compile` anywhere in `src/gradlab`** — the Microscope's
  expression engine is a hand-written parser, and this is what guarantees it.
* **The runtime dependency list is exactly `["numpy"]`**, read from
  `pyproject.toml`.

## Test inventory

| suite | tests | what it covers |
|---|---:|---|
| `tests/unit` | 332 | tensor, engine, nn, optim, inspection, CLI, server, optdigits |
| `tests/gradients` | 62 | finite-difference checks on every operation |
| `tests/conformance` | 55 | PyTorch oracle comparisons |
| `tests/property` | 18 | Hypothesis invariants (60 generated examples each) |
| `tests/integration` | 27 | end-to-end training, reproducibility, checkpointing, documented claims |
| doctests | 12 | every documented example actually runs |
| **total** | **506** | **84% line coverage** |

`pytest -q` takes roughly a minute — median 71 s over four runs on the
reference machine, ranging 58–88 s with background load. The cost is
concentrated in a handful of tests that genuinely train models: the SGD
optimiser-parity trajectory (~11 s), and the XOR claims in
`test_documented_claims.py` (~6 s), which train two 400-epoch networks to show
that an affine model cannot solve it.

## What is *not* established

* **Untested inputs.** Coverage is 84%; the uncovered 16% is mostly error paths
  and CLI plumbing.
* **float32.** Supported and smoke-tested, but every tolerance above is for
  float64. float32 gradients are correct to float32 precision, which is roughly
  1e-6, not 1e-10.
* **Numerical behaviour at extremes** beyond the specific cases tested (±800
  logits, 5000-deep graphs).
* **Thread safety.** Not tested, because it is not supported.
* **Other platforms.** All numbers above are from one machine; CI runs the suite
  on Linux with Python 3.11 and 3.12.
