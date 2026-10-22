# Development journal

A record of the actual engineering, including the things that went wrong. This
is not a reconstructed narrative: every entry below corresponds to a real defect
found, a real decision made, or a real measurement that contradicted an
assumption.

**Authorship note.** GradLab was built in a single session by Claude (Anthropic)
working from a detailed specification written by the project owner. The code is
AI-generated; the specification, the direction, and the acceptance criteria are
the owner's. See `README.md` § "AI-assisted development" for the full statement.
Nothing in this journal is presented as human-authored work.

---

## Phase 1 — Environment and contracts

**The system Python was too old.** The target was 3.11+, the machine had 3.9.6,
and this project has a no-Homebrew constraint. Resolved with a `uv`-managed
3.12 virtual environment. Worth recording because it is the first thing an
independent developer will hit on a similar machine, and the README now says so.

**Contracts were written before code.** `docs/architecture.md` and
`docs/tensor-design.md` were written first and fixed the semantics that later
turned out to matter most: that a non-scalar `backward()` would *require* a
seed, that graphs would be released after one pass unless told otherwise, and
that in-place mutation would be *detected* rather than merely discouraged.

Writing the version-counter design down in Phase 1 is the reason it exists. It
would not have been retrofitted later — by the time the symptom appears, the
cost of adding it looks higher than the cost of documenting a rule instead.

---

## Phase 2 — Tensor and engine

**Defect: `__slots__` without `__weakref__`.** `Tensor` used `__slots__` for
memory, and `Node` holds a weak reference to its output to avoid a
`Tensor → Node → Tensor` cycle. Those are incompatible by default: the first
multiplication raised `TypeError: cannot create weak reference`.

Fixed by declaring `__weakref__` explicitly in the slots tuple, with a comment
explaining why it is there — because the obvious "cleanup" of deleting an
apparently unused slot would reintroduce the bug.

**Defect in my own design: the engine deposited retained gradients after
releasing the graph.** The first draft of `backward()` walked the nodes, then
looped again afterwards to deposit `.grad` on non-leaves that had called
`retain_grad()`. By then the saved state was gone, and the second loop was a
no-op with a `pass` in it.

Fixed by depositing during the walk: at the moment a node is popped in reverse
topological order, every contribution to its output's gradient has arrived, and
the node has not yet been released. Caught by writing the test before trusting
the code.

**All 12 specification edge cases passed on first run** once those two were
fixed, including `x*x` → 6, `y+y` → 8, broadcast bias → `[4,4,4]`, the
non-scalar refusal, and `InplaceModificationError`.

---

## Phase 3 — Operations

**My implementation contradicted my own specification.**
`docs/tensor-design.md` said integer tensors stay integers unless
`requires_grad=True` forces a float promotion. The implementation promoted
*every* integer array to float64 regardless.

The test suite caught it immediately, and the resolution went the right way
round: the spec was better (class labels and masks need to stay integers), so
the *code* changed, not the document. Noted because the tempting fix is always
to edit the test.

**A test's own regex was wrong**, expecting "cannot carry a gradient" where the
message said "can carry a gradient" in a longer sentence. The message was good;
the test was corrected.

After those, 62 finite-difference checks across every operation passed, with the
worst absolute error at 1.04e-8 (`log_softmax`, where the numerical check
suffers cancellation — the analytic rule matches PyTorch to 1e-10).

**The gradient checker was itself checked.** A checker that never fails proves
nothing, so `Mul.backward` was monkey-patched to return 1.5× the correct
gradient, and `gradcheck` was asserted to raise. It did.

---

## Phase 4–5 — Layers, losses, optimisers, training

**A measurement, not a bug: Adam's "first step equals the learning rate".**
The test asserted `|Δθ| == lr` at `t=1` with `rel=1e-6`, and failed for a
gradient of 0.001: it got 0.0999990 instead of 0.1.

Investigating showed the claim is only approximately true. The exact `t=1`
update is `lr·g/(|g| + ε)`, which equals `lr` only as `ε/|g| → 0`. The fix was
to assert the *exact* expression and then separately assert that the shortfall
is precisely `ε/(|g|+ε)` — a stronger test than the loose one would have been,
and it now documents where the approximation's error lives.

**Linear weight layout was chosen deliberately**, `(in, out)` rather than
PyTorch's `(out, in)`, so the forward pass reads as `x @ W + b` with no
transpose. The cost is a `.T` on every weight copied to or from PyTorch; the
conformance tests do it explicitly and say why.

---

## Phase 6 — Experiments

**The linear regression converged to the noise floor**, which is the right
answer rather than a suspiciously good one: train loss 0.010373 against an
irreducible noise variance of 0.0100. The hand-written loop and `fit()` reached
*bitwise identical* parameters, which is the claim `fit()` makes about itself.

**XOR behaved exactly as the theory says.** 53.25% without a hidden
nonlinearity (the majority-class rate), 100% with one `tanh`. Same parameter
count, same seed, same data.

**Two measurements contradicted the narrative I had written, and the narrative
changed — twice.**

*First:* the learning-rate sweep preset was written assuming high learning rates
would diverge to `nan`. They did not. Measuring the real behaviour showed SGD on
that model is stable to `lr = 3`, spikes and recovers at `lr = 10`, and is
permanently unstable from `lr = 30`, **without ever producing a non-finite
loss**. The reason is itself the lesson: `binary_cross_entropy_with_logits`
grows only *linearly* in the logit, so even an absurd rate gives a large finite
number.

Rather than delete the claim, a second panel was added that genuinely diverges —
squared error on an unscaled regression at `lr = 5.0`, where each overshoot is
squared. Recorded per *batch*, because the whole blow-up happens inside the
first epoch:

```
13 → 3.24e4 → 9.22e15 → 3.88e50 → 1.01e155 → inf
```

GradLab stops at batch 6 rather than writing `nan` into the parameters. The
preset now teaches the real distinction between a loss that is linear in its
input and one that is quadratic.

*Second:* an integration test asserted the blob classifier would exceed 93%
accuracy, and it hit exactly 0.93. Rather than nudge the threshold, a
nearest-centroid classifier was run on the same data as a reference: it reaches
**0.9267**. The clusters genuinely overlap, and 93% was asking for better than
the data supports. The test now compares against that data-derived ceiling
instead of a number chosen in advance — a better test, arrived at only by
checking why the original failed.

The same approach fixed the two-moons test, which demanded a 5-point margin over
a linear baseline and measured exactly 5 points. It now asserts the two claims
actually worth making: that the MLP beats the baseline, and that it reaches 95%.

---

## Phase 7 — The Gradient Microscope

**The engine gained an observer hook rather than a second implementation.**
`backward()` takes an optional `recorder`. The alternative — having the
inspection layer re-walk the graph — would have created a second backward
implementation free to drift out of agreement with the first. A test now asserts
the trace's gradients equal a plain `backward()`'s to 1e-14.

**Defect: the trace mislabelled reductions as broadcasts.** The first version
flagged any shape change between upstream and downstream gradient as "summed
over broadcast axes". That is wrong for reductions, whose gradients are
*expanded*, not summed — and it is exactly backwards as a teaching point.

Fixed by consulting the operation's own `auto_unbroadcast` flag to tell the two
cases apart, and reporting them with different language.

**Defect: two views rendered on top of each other.** `#view-graph` had
`display: grid` in CSS, and an ID selector outranks the browser's `[hidden]`
rule. Switching tabs hid nothing. Caught by actually opening the page in a
browser and noticing the inspector panel visible under the backward stepper —
not something any Python test would have found.

Fixed with an explicit `[hidden] { display: none !important; }`, and the stepper
views were then given their own graph panels deliberately, which turned out
better than the original design.

**Defect: a rename left the CLI reading a field that no longer existed.**
The learning-rate preset's `diverged` key became `stopped` during the rewrite
above, but `cmd_generate` still read `diverged`. Generation crashed *after*
writing two trace files but *before* writing `index.json`, so the app silently
showed 10 traces instead of 12 — a failure mode that looked like a browser cache
problem.

Two lessons recorded: the index should be written in the same operation as the
files it describes, and a crash between them is worse than a crash before them.
Tests now cover `generate` end to end and assert that every file named in the
index exists.

**Constants rendered as `leaf1`, `leaf2`.** Cosmetic, but it defeated the point:
in `x^2 + 2*x` the literal `2` should read as `2`. Unnamed non-differentiable
scalars are now labelled by their value. The same pass routed a constant
exponent to the scalar `Pow` op instead of `PowTensor`, which removed a spurious
leaf from the graph and gave a cleaner backward rule to display.

---

## Phase 8 — Hardening

**19 mypy errors, all genuine.** Missing slot annotations, a `type` where
`type[Function]` was meant, `self.bias = None` on a `Parameter`-typed attribute,
`AdamW.__init__` forwarding `*args` in a way that could collide with its own
keyword. All fixed properly rather than silenced; the suite is now clean.

**79 ruff findings**, mostly formatting. Two were worth keeping as exceptions:
Greek letters and typographic minus signs in docstrings are deliberate
mathematical notation, so `RUF001-003` are configured off with a comment saying
why.

**Doctests were enabled after one was found wrong.** `graph_summary`'s docstring
claimed `x*x + x` produces 3 nodes; it produces 2. Since the documentation makes
many concrete numerical claims, `--doctest-modules` was added to the pytest
configuration so a stale example now fails CI. 12 doctests run.

---

## Phase 9 — Documentation and benchmarks

**The benchmark narrative had to be rewritten after reading its own output.**
The report was drafted expecting GradLab to be comfortably slower than PyTorch.
The measurement came back at **0.84×–1.12×** — occasionally faster.

That is a result worth being careful with. Three explanations were identified,
only one of which is about GradLab: at these sizes both frameworks are dominated
by per-operation dispatch rather than arithmetic; the thread counts are *not*
matched (PyTorch pinned to one thread, NumPy's BLAS left at default); and the
models are tiny CPU models that PyTorch's engineering is not aimed at.

The report now states all three and explicitly says the ratio below 1.0 is not
evidence of a faster framework. Writing "GradLab beats PyTorch" would have been
true of the number and false about reality.

**A packaging defect that only a clean install could find.** `.gitignore`
contained a bare `data/` rule, intended for downloaded datasets. Hatchling
honours `.gitignore` when building, so that rule also matched
`src/gradlab/data/` and **silently excluded it from the wheel**. Everything
passed locally — the editable install reads the source tree — and the built
package raised `ModuleNotFoundError: No module named 'gradlab.data'` on first
import.

Found by installing the wheel into a fresh virtual environment and importing it,
which is now a CI job. The rule is scoped to `examples/_data/` with a comment
explaining the trap.

The general lesson: an editable install is not a test of packaging. The only
test of packaging is installing the artefact you would actually ship.

---

## Defect summary

| # | Defect | Found by | Phase |
|---|---|---|---|
| 1 | `__slots__` blocked weak references | first execution | 2 |
| 2 | Retained gradients deposited after graph release | writing the test first | 2 |
| 3 | Integer promotion contradicted the spec | unit test | 3 |
| 4 | Test regex did not match the real message | unit test | 3 |
| 5 | Adam `t=1` claim was approximate, not exact | unit test | 5 |
| 6 | Reductions mislabelled as broadcasts in traces | reading the CLI output | 7 |
| 7 | Two views rendered simultaneously (CSS specificity) | opening the browser | 7 |
| 8 | Renamed field left the CLI crashing mid-generation | running `generate` | 7 |
| 9 | Constants displayed as `leaf1`/`leaf2` | opening the browser | 7 |
| 10 | 19 type errors | mypy | 8 |
| 11 | Wrong node count in a docstring | enabling doctests | 8 |
| 12 | `.gitignore` excluded `gradlab/data/` from the wheel | clean-env install | 9 |

Four of these (6, 7, 9, 12) were only findable by *using* the thing — reading
output, opening a browser, installing the built artefact — not by running the
test suite against the source tree. That is the argument for building the
visualiser and for the clean-install CI job: both exercise the project in ways
the unit tests structurally cannot.

Two (3, 5) were cases where the test and the implementation disagreed and the
*test* turned out to encode the better claim. Two more (the blobs and two-moons
thresholds) were cases where the test's number was arbitrary and measuring a
real reference produced a better test.
