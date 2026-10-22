# Optimisers

Every optimiser GradLab implements, what its state is for, and how each was
validated. All three are checked against PyTorch over 20-step trajectories in
`tests/conformance/test_torch_nn_parity.py`, agreeing to 1e-10.

## The shared contract

```python
optimizer.zero_grad()    # gradients ACCUMULATE; clear them first
loss.backward()          # fill .grad on every parameter
optimizer.step()         # update, under no_grad
```

`step()` writes to each parameter's buffer in place, inside `no_grad()`. This is
the one sanctioned mutation in GradLab, and it bumps the tensor's version
counter — so a graph that saved the parameter cannot be differentiated again
afterwards without the engine raising. The rule "step only after backward" is
therefore enforced, not merely documented.

**Parameters that are skipped**, silently and by design:

* `requires_grad=False` — deliberately frozen.
* `.grad is None` — took no part in this loss.

**Non-finite gradients raise** rather than being clipped or skipped. Writing
`nan` into a parameter destroys the model irrecoverably, and the real cause is
always several steps upstream. Pass `error_on_nonfinite_grad=False` to take
responsibility yourself.

## SGD

```
θ ← θ − η·g
```

The whole algorithm. `g` points uphill, so stepping against it reduces the loss
for a small enough `η`.

### Momentum

```
v ← μ·v + g
θ ← θ − η·v
```

A velocity accumulates past gradients. Directions that stay consistent build up;
directions that oscillate cancel. The classic picture is a narrow ravine — plain
SGD zig-zags across it, momentum averages the zig-zags away and accelerates
along the floor.

**The scale consequence is easy to miss.** For a constant gradient the velocity
converges to `g/(1−μ)`, so the effective step is `η/(1−μ)`. Turning momentum on
at `μ = 0.9` makes steps roughly **ten times larger**. Reduce `η` accordingly
rather than being surprised by divergence. `tests/unit/test_optim.py` asserts
this amplification directly.

### Nesterov

Evaluate the gradient at the look-ahead point rather than the current one, which
here reduces to `θ ← θ − η(g + μ·v)`. A small, usually helpful correction.

### Weight decay (coupled)

Adds `λ·θ` to the gradient before the momentum step. Because it passes through
momentum and the learning rate, it is not a clean L2 penalty — see AdamW.

## Adam

```
m ← β₁·m + (1−β₁)·g            first moment  (mean of gradients)
v ← β₂·v + (1−β₂)·g²           second moment (mean of squared gradients)
m̂ ← m/(1−β₁ᵗ)                  bias correction
v̂ ← v/(1−β₂ᵗ)
θ ← θ − η·m̂/(√v̂ + ε)
```

Two exponential moving averages per parameter. The step is the first divided by
the root of the second, which makes it roughly **scale-invariant**: multiply
every gradient by 1000 and `m̂` grows by 1000 while `√v̂` also grows by 1000, so
the step is unchanged. A parameter with consistently tiny gradients gets a
proportionally larger step than SGD would give it.

`tests/unit/test_optim.py::test_adam_is_scale_invariant_in_the_gradient` asserts
exactly this, by running two parameters whose gradients differ by 1000× and
checking their trajectories match.

### Bias correction is not cosmetic

Both averages start at **zero**, which biases them toward zero early on. At
`t = 1` with `β₂ = 0.999`:

```
v₁ = 0.999·0 + 0.001·g² = 0.001·g²
```

A thousand times too small. `√v₁ ≈ 0.032·|g|` in the denominator would make the
very first step about **30× larger** than intended. Dividing by `1 − βᵗ`
rescales each average by the total weight it has actually accumulated:

```
m̂₁ = m₁/(1−β₁) = g        v̂₁ = v₁/(1−β₂) = g²
```

so the first step is `η·g/(|g| + ε)` — magnitude `η`, whatever the gradient.
`test_first_step_is_lr_up_to_the_epsilon_term` asserts this for gradients
spanning six orders of magnitude, and also asserts the residual is exactly
`ε/(|g|+ε)`, which is where the approximation's error actually lives.

The correction decays geometrically and is negligible after a few hundred steps.

### ε

Prevents division by zero and bounds the step when `v̂` is tiny. The default
`1e-8` is inside the square root's denominator, so it is compared against
`√v̂ ≈ |g|`, not against `g²`.

## AdamW

Identical to Adam except that weight decay is applied **outside** the adaptive
rescaling:

```
θ ← θ − η·m̂/(√v̂ + ε) − η·λ·θ
```

### Why decoupling matters

In Adam, `λ·θ` is folded into `g` and then divided by `√v̂`. The amount of decay
a weight receives therefore depends on its own gradient history: a parameter
with large, noisy gradients is barely regularised, and one with tiny gradients
is regularised hard. That is not what an L2 penalty is supposed to do.

AdamW decays every weight by the same relative amount per step, independent of
gradient scale. The practical consequence: `weight_decay` becomes a meaningful,
transferable hyperparameter — and a value tuned for `Adam` will be far too small
for `AdamW`. `test_decay_term_is_exactly_lr_times_wd_times_theta` pins the term
down to exactly `η·λ·θ`.

## Choosing one

| | SGD | SGD + momentum | Adam | AdamW |
|---|---|---|---|---|
| state per parameter | none | 1 buffer | 2 buffers | 2 buffers |
| lr sensitivity | high | high | low | low |
| typical lr | 0.01–0.1 | 0.001–0.01 | 0.001 | 0.001 |
| regularisation | coupled L2 | coupled L2 | coupled L2 | **decoupled** |

Adam needs far less learning-rate tuning, which is why it is the sensible
default for the experiments here. SGD with momentum and a schedule often
generalises slightly better on large vision models — a finding from the
literature, not something GradLab's experiments measured, and so not claimed as
one.

`examples/_out/learning-rates.json` (and the Microscope's "Learning rates"
trace) shows what the learning rate actually does to one of these models across
four decades, measured rather than asserted.

## Validation

Every claim above is tested:

* **Hand-computed trajectories.** `test_matches_a_hand_computed_three_step_trajectory`
  recomputes Adam's update in the test body from the published rule and compares
  to 1e-12. A test that only compares the implementation to itself proves
  nothing; this one compares it to the paper.
* **PyTorch parity.** 20 optimiser steps on a 3-layer MLP, for SGD, SGD with
  momentum, Nesterov, Adam and AdamW, agreeing with `torch.optim` to 1e-10.
* **Refusals.** Frozen parameters, missing gradients, `nan`/`inf` gradients,
  invalid hyperparameters, non-leaf parameters, empty parameter lists.
* **State round-trip.** Training is stopped at step 3, `state_dict()` is saved
  and reloaded into fresh objects, and the resumed run reaches the same
  parameters as an uninterrupted one to 1e-12.
