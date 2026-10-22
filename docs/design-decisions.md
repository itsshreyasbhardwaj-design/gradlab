# Design decisions

The choices that could reasonably have gone another way, and why they went this
way. Where a decision has a cost, the cost is stated.

## 1. Nodes are operations, edges are tensors

**Alternative:** nodes are tensors, edges are operations.

Both are workable. Operation-as-node was chosen because it makes the backward
pass a plain loop over nodes — each node has one incoming gradient and produces
one gradient per input — and because it is what the Microscope needs to display
(an operation has a *rule* to show; a tensor does not).

**Cost:** a tensor used in three places appears in three nodes' `inputs` lists,
so "find every consumer of this tensor" is a scan rather than a lookup. Nothing
in the engine needs that query, so the cost is never paid.

## 2. `Function` subclasses rather than closures

**Alternative:** have each operation return a closure capturing what backward
needs.

Closures are shorter. Subclasses were chosen because the class *is* the
documentation: `op`, `backward_rule` and `auto_unbroadcast` are declared
attributes that the Microscope reads directly, and a reader can see the forward
and backward of one operation side by side in ten lines.

**Cost:** more ceremony per operation.

## 3. Automatic unbroadcasting in `Function.apply`

Broadcast-gradient reduction is the single most common place a hand-rolled
engine goes wrong. Rather than asking every rule to handle it, rules return
gradients in the *output* shape and `apply` reduces them.

Rules that already produce input-shaped gradients (`matmul`, reductions,
`reshape`, `transpose`, `index`) opt out with `auto_unbroadcast = False`.

**Cost:** two conventions instead of one, and a rule author has to know which
applies. Mitigated by `apply` raising immediately if a returned gradient does
not match its input's shape — so getting it wrong fails loudly on the first
test, not subtly in production.

## 4. A non-scalar `backward()` requires an explicit seed

**Alternative:** default to all-ones, as some tutorials do.

Defaulting to ones silently computes `d(output.sum())/dx`, which is a different
function. A user who wanted a Jacobian-vector product and got that would have no
way to notice.

**Cost:** slightly more typing in the rare non-scalar case. Worth it: this is
precisely the kind of silent wrongness the project exists to avoid, and one of
the Microscope's presets demonstrates the difference.

## 5. A version counter instead of a documented rule

**Alternative:** document "don't mutate a tensor that's in a live graph" and
trust the user.

Every `Tensor` carries `_version`, bumped on any in-place write; every `Node`
records its inputs' versions at forward time and the engine compares before
running a rule.

This converts a documentation convention into an enforced invariant. The failure
it prevents — an optimiser step between forward and backward producing a
gradient computed from the *new* parameters — is silent, plausible, and
genuinely hard to debug.

**Cost:** one integer per tensor, one comparison per node per backward pass.
Negligible.

## 6. Gradients accumulate; `zero_grad` is the caller's job

**Alternative:** clear `.grad` automatically at the start of each `backward()`.

Accumulation is what makes gradient accumulation across micro-batches possible,
and it matches the convention every optimiser in the ecosystem expects.

**Cost:** forgetting `zero_grad()` silently sums gradients across steps, which is
a classic beginner bug. Mitigated by `fit()` always calling it, and by the
training loop in `examples/linear_regression.py` showing the call explicitly with
a comment saying why.

## 7. Linear weights are `(in_features, out_features)`

**Alternative:** PyTorch's `(out, in)`, with `x @ W.T`.

GradLab's forward pass reads exactly like the mathematics: `x @ W + b`, no
transpose. For a framework whose purpose is making the mathematics visible, that
matters.

**Cost:** copying weights to or from a PyTorch model needs a `.T`. The
conformance tests do this explicitly and comment on it, so the difference is
documented rather than a trap.

## 8. The default initialiser is Xavier, not Kaiming

Neither is optimal for every architecture, and GradLab does not claim one is.
Xavier is the default because it is the safer choice when the activation is
unknown or mixed; `init="kaiming"` is one keyword away and is used in the ReLU
examples.

`docs/optimizers.md` and `nn/initialization.py` both derive the variance
argument behind each rather than presenting them as folklore.

## 9. Losses take logits, not probabilities

`binary_cross_entropy_with_logits` and `cross_entropy_with_logits` both apply
the sigmoid/softmax internally. There is **no** `binary_cross_entropy` taking
probabilities.

Offering one would mean offering a numerically unstable path, and users would
reach for it because the name is shorter. The stable formulations are the only
formulations.

**Cost:** a user with probabilities already in hand has to convert back to
logits, or compute the loss by hand. Rare, and the error message points at the
logit-space function.

## 10. Non-finite gradients raise rather than being skipped

**Alternatives:** skip the parameter, clip the gradient, or write the `nan`.

Writing `nan` destroys the model irrecoverably. Skipping or clipping hides the
cause, which is always several steps upstream. Raising stops the run at the
first sign and says what to look at.

**Cost:** a run that would have recovered on its own is stopped.
`error_on_nonfinite_grad=False` is available for callers who want that.

## 11. Domain errors raise instead of returning `nan`

`log(0)`, `sqrt(-1)`, `x/0`, `0**-1` all raise `DomainError`.

NumPy's default is a warning and a `nan`, which then propagates silently through
a whole training run. Raising at the point of the error means the traceback
points at the actual mistake.

**Cost:** code that relied on `nan` propagation has to add an epsilon
explicitly, which is better practice anyway.

## 12. The engine takes a recorder hook rather than being reimplemented

The Microscope needs a step-by-step transcript of the backward pass. It could
have re-walked the graph itself.

Instead `backward()` accepts an optional observer. There is exactly **one**
backward implementation, and the Microscope shows what it actually did.

**Cost:** a parameter on a hot function that is `None` in all production use.
`test_trace_matches_a_plain_backward_pass` asserts the trace's numbers equal an
ordinary `backward()`'s to 1e-14.

## 13. The Microscope is static JSON, not a live server

**Alternative:** a Flask/FastAPI backend executing user code on demand.

The default is: `gradlab-microscope generate` runs the real presets and writes
JSON; a zero-dependency static page reads them. There is **no execution surface
at all** in that mode. The optional `--allow-expressions` endpoint is gated
behind a flag and restricted to an allowlist.

This also means the library is fully usable without the website, and the website
needs no npm, no build step, and no network.

**Cost:** traces must be regenerated to reflect engine changes. The schema
version is checked by the page, which warns on a mismatch.

## 14. A hand-written parser, not `ast.literal_eval` or `eval`

User-typed expressions are tokenised by a fixed character set and parsed by
recursive descent into a plain-data tree holding only numbers, variable names,
and operation names drawn from a 14-entry allowlist.

Using Python's own parser — even via `ast` with a node allowlist — means
tracking what each new Python version adds to the grammar. A 200-line parser
with an explicit character set has no such exposure.
`test_no_eval_or_exec_in_shipped_code` enforces that no `eval`/`exec`/`compile`
call exists anywhere in the package.

**Cost:** GradLab's expression syntax is its own small language, not Python.

## 15. NumPy is the only runtime dependency

No SciPy (not needed), no scikit-learn (the dataset generators are 40 lines
each), no matplotlib (the examples draw to the terminal, the Microscope draws
SVG), no pandas.

`test_dependencies_are_only_numpy` reads `pyproject.toml` and asserts the list.

**Cost:** the terminal plots are ASCII. In practice they turned out to be an
asset — `examples/xor.py`'s decision boundary is legible in a CI log.

## 16. float64 by default

**Alternative:** float32, as every production framework defaults to.

float32 halves memory and is faster. But a framework for *learning* should have
numerical noise far below the effects being taught, and finite-difference
gradient checking needs float64 to reach useful tolerances at all.

float32 is supported and tested; it is just not the default.
