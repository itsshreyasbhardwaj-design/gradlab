# GradLab: Design and Validation of an Educational Automatic Differentiation and Neural Network Framework

**An engineering report.** This is not peer-reviewed research and makes no claim
to novelty. It documents the design, implementation and validation of a
framework built from well-established algorithms, and reports measurements taken
from the implementation.

---

## 1. Abstract

GradLab is an educational machine learning framework implementing reverse-mode
automatic differentiation from first principles. It provides a custom `Tensor`
type over NumPy storage, a dynamically constructed computation graph, an
independent reverse-mode engine, a neural network library, three optimisers, a
reproducible training loop, and an interactive visualiser that renders the
engine's actual execution.

The implementation comprises approximately 2,750 statements across 42 modules
with a single runtime dependency (NumPy). Correctness is established by four
independent strategies: analytic tests on known derivatives; central
finite-difference checking across 36 operations (worst absolute error
1.04 × 10⁻⁸); property-based testing over generated inputs; and comparison
against PyTorch as an external oracle (agreement to 1 × 10⁻¹⁰ on operations, and
to 1.4 × 10⁻¹⁶ on a 120-epoch training run). 506 tests pass with 84% line
coverage.

Four experiments are reported: linear regression converging to the irreducible
noise floor (final loss 0.010373 against a noise variance of 0.0100); XOR, which
a purely affine network of the same capacity fails at 53.3% while one `tanh`
raises it to 100%; two-moons classification at 96.7% validation accuracy against
an 82.7% linear baseline; and handwritten-digit classification on UCI optdigits
at 96.83% test accuracy with a 6,570-parameter network.

The contribution is pedagogical, not algorithmic. Every algorithm implemented is
standard; the engineering decisions, the validation design, and the inspection
interface are the project's own.

---

## 2. Motivation

Automatic differentiation is usually encountered as `.backward()` — a method
that works. The mechanism underneath is rarely examined, with the result that
practitioners can train models without being able to explain why a bias gradient
is summed across a batch, why `max` needs a subgradient convention, or why
forgetting `zero_grad()` produces the symptom it does.

Existing teaching materials occupy two extremes. Minimal implementations
(Karpathy's `micrograd` and its descendants) are short enough to read in full
but operate on scalars, so the problems that dominate real autodiff —
broadcasting, matrix calculus, numerical stability — never arise. Production
frameworks handle all of those, but the handling is distributed across hundreds
of thousands of lines of C++ and is not readable as an explanation.

GradLab targets the gap: a tensor-valued engine with broadcasting and matrix
operations, small enough that the whole engine is about 150 lines, validated
rigorously enough to be trusted, and instrumented so the execution can be
inspected rather than inferred.

A secondary motivation shaped the engineering: **the project should be
defensible under review.** That produced the architecture test that forbids
importing another autodiff framework, the documentation-claim tests that
recompute quoted figures, and the development journal recording defects found.

---

## 3. Related work

**micrograd** (Karpathy, 2020) — ~150 lines, scalar-valued, with a topological
sort and backward closures. GradLab shares the define-by-run structure and
extends it to n-dimensional tensors, which introduces broadcasting-gradient
reduction and matrix calculus as the substantial new problems.

**Autograd** (Maclaurin, Duvenaud & Adams, 2015) — reverse-mode AD over NumPy by
tracing wrapped functions. Architecturally the closest ancestor. GradLab uses an
explicit `Function` class per operation rather than tracing, which makes each
derivative rule a readable, independently testable unit and supplies the
metadata the visualiser needs.

**PyTorch** (Paszke et al., 2019) — define-by-run with a C++ engine. GradLab
mirrors its user-facing API deliberately so that what is learned transfers, and
uses it here as an independent test oracle.

**JAX** (Bradbury et al., 2018) — composable transformations over a functional
tracing model, supporting arbitrary-order differentiation. GradLab's mutable
`.grad` and single-order design are a deliberate simplification.

**Visualisation.** TensorFlow Playground (Smilkov & Carter, 2016) visualises
training dynamics but not the differentiation mechanism; TensorBoard graphs show
structure without execution values. The Gradient Microscope differs in showing
the actual backward traversal — upstream gradient, local derivative and
accumulated result at each step — read from the engine via an observer hook
rather than reconstructed.

GradLab introduces no new algorithm. Every rule is textbook; the design of the
validation and the inspection interface is where the original engineering lies.

---

## 4. Reverse-mode automatic differentiation

### 4.1 Why reverse mode

The chain rule for a composition is a product of Jacobians, and the product may
be associated from either end. Forward mode costs one pass per *input*; reverse
mode costs one pass per *output*. For a loss function with n ≈ 10⁶ parameters
and one scalar output, reverse mode is cheaper by a factor of n.

The cost is memory: the graph cannot be walked backwards until it has been
built, so values needed by backward rules must survive the forward pass.
Reverse mode trades memory for time, and section 11 measures the trade.

### 4.2 Graph representation

Nodes are operations; edges are tensors. A `Node` records the operation name,
its input tensors (strong references), a weak reference to its output, the
values saved for backward, a human-readable rule, and a monotonic creation
index.

The output reference is weak because `Tensor` holds a strong reference to its
`grad_fn`. A strong back-reference would make every graph an uncollectable cycle
and leak one graph per training step.

Acyclicity is structural rather than checked: a node may only reference tensors
that existed when it was built, so the creation index strictly decreases along
any backward edge.

### 4.3 Traversal

```
1. seed       dL/dL = 1, or an explicitly supplied upstream gradient
2. order      reverse topological order (iterative DFS post-order, reversed)
3. propagate  for each node: run its rule, ADD the result into each input's total
4. deposit    leaves receive .grad; non-leaves only if retain_grad() was called
5. release    free saved state unless retain_graph=True
```

Step 2 is what makes a single pass correct: every consumer of a tensor is
processed before its producer, so a node's incoming gradient is complete when its
rule runs. Step 3's *addition* is what makes shared subgraphs correct.

The traversal is iterative rather than recursive so that deep graphs do not
exhaust Python's stack; section 11 measures it to 5,000 nodes.

### 4.4 Non-scalar outputs

A non-scalar output has no canonical seed. GradLab requires one explicitly
rather than defaulting to ones, because defaulting silently computes
`d(output.sum())/dx` — a different function the caller would have no way to
detect.

---

## 5. Tensor engine architecture

`Tensor` wraps a `numpy.ndarray` and adds: `requires_grad`, an accumulated
`.grad`, a `grad_fn` edge, a version counter, and a retain flag. `__slots__` is
used for memory, with `__weakref__` declared explicitly so nodes can hold weak
output references.

**Supported dtypes** are float64 (default, and the dtype all correctness
tolerances are stated for) and float32. Integer and boolean tensors exist for
labels and masks but cannot carry gradients. Complex dtypes are rejected.

**Copy and aliasing** are explicit: construction copies unless told otherwise;
`detach()` shares the buffer but not the history; `clone()` is a differentiable
copy.

**In-place modification** is detected rather than forbidden. Every write bumps
the tensor's version; every node records its inputs' versions at forward time;
the engine compares before running a rule and raises on a mismatch. This
converts "call `step()` only after `backward()`" from a documented convention
into an enforced invariant.

---

## 6. Computational graph design

The `Function` base class separates mathematics from bookkeeping. An operation
supplies `forward(ctx, *arrays)` and `backward(ctx, grad)`; `Function.apply`
unwraps tensors, calls forward, wraps the result, and creates a node only when
gradient mode is active and some input requires grad.

Backward rules return gradients in the **output** shape, and `apply` reduces each
to its input's shape via `unbroadcast`. Centralising this is deliberate:
broadcast-gradient reduction is the most common defect in hand-rolled engines,
and here it exists in exactly one place. Operations whose gradients are already
input-shaped (`matmul`, reductions, shape ops) opt out via a class attribute,
and `apply` raises immediately if any returned gradient has the wrong shape.

`gradlab.inspection` provides a versioned, JSON-serialisable view of the graph
and of a backward pass. The backward trace is produced by an **observer passed
into the engine**, not by a second traversal, so there is exactly one backward
implementation and the visualiser cannot drift from it. A test asserts the
trace's gradients equal a plain `backward()`'s to 10⁻¹⁴.

---

## 7. Mathematical derivations

Full derivations are in `docs/backpropagation.md`. Summary of the rules
implemented:

| class | operations | representative rule |
|---|---|---|
| arithmetic | `+ − × ÷ neg abs pow` | `∂(ab)/∂a = b ⊙ g` |
| transcendental | `exp log sqrt sin cos` | `∂ln a/∂a = g/a` |
| activations | `tanh sigmoid relu` | `∂tanh/∂a = (1 − out²) ⊙ g` |
| reductions | `sum mean max min` | `∂Σa/∂a = broadcast(g)` |
| matrix | `matmul` | `∂L/∂A = G·Bᵀ`, `∂L/∂B = Aᵀ·G` |
| shape | `reshape transpose index concat` | inverse rearrangement |
| stable | `log_softmax softmax bce_logits` | `∂L/∂x = g − p·Σg` |

Three deserve note.

**Broadcasting.** Two distinct reductions are required: leading axes the input
never possessed are summed away; axes present as size 1 are summed with
`keepdims`. A bias of shape `(d,)` added to a `(N, d)` batch receives a gradient
summed across all N rows.

**Extrema.** `max` is non-differentiable at ties. GradLab's documented choice is
to split the gradient equally among tied extrema — a valid subgradient,
symmetric under permutation, matching `torch.amax` so conformance tests can
compare directly.

**Indexing.** Backward is a scatter-*add*, not an assignment, so that a repeated
index receives the sum of its contributions.

---

## 8. Neural network implementation

`Module` intercepts attribute assignment to register `Parameter`s and child
modules, giving recursive discovery without manual bookkeeping. `state_dict()`
returns copies; `load_state_dict()` validates every name and shape *before*
writing anything, so a mismatched checkpoint leaves the model untouched.

**Layers:** `Linear`, `Flatten`, `Dropout` (inverted, so evaluation needs no
rescaling), and `Sequential`. `Linear` stores its weight as
`(in_features, out_features)` so the forward pass reads as `x @ W + b` with no
transpose — a readability decision whose cost is a transpose when exchanging
weights with PyTorch.

**Initialisation:** uniform, normal, Xavier and Kaiming, each documented with the
variance argument behind it. The default is Xavier uniform; no claim is made
that one scheme is optimal for every architecture.

**Losses:** mean squared error, and binary and multiclass cross-entropy that take
**logits only**. No probability-space variant is offered, because offering one
would offer a numerically unstable path with a shorter name.

---

## 9. Optimisation algorithms

SGD (with optional momentum, Nesterov and coupled weight decay), Adam, and AdamW
with decoupled weight decay. All updates occur under `no_grad`, writing in place
and bumping the version counter.

Adam's bias correction receives particular attention in both code and
documentation because it is the component most often treated as incidental. At
`t = 1` with `β₂ = 0.999`, the raw second moment is `0.001·g²` — a factor of
1000 too small — and `√v` in the denominator would make the first step roughly
30× too large. The correction makes the first step's magnitude exactly
`η·|g|/(|g| + ε)`, which the test suite asserts for gradients spanning six
orders of magnitude, together with the residual being exactly `ε/(|g| + ε)`.

Non-finite gradients raise rather than being clipped or skipped: writing `nan`
into a parameter is irrecoverable, and the cause is always upstream.

---

## 10. Independent correctness evaluation

Four strategies, each able to catch failures the others miss. Full detail and
the complete error table are in `docs/validation.md`.

**Analytic.** Known derivatives, including every edge case the specification
required: `x·x` at 3 → 6; `y + y` with `y = x·x` at 2 → 8; broadcast bias
→ `[4,4,4]`; non-scalar backward refused; double backward refused.

**Finite differences.** Central differences at float64 with `ε = 10⁻⁶`, chosen
because `∛(machine epsilon)` balances truncation against cancellation. All 36
operations pass; the worst absolute error is 1.04 × 10⁻⁸ (`log_softmax`, where
the *numerical* estimate suffers cancellation). The checker is itself validated
by monkey-patching a rule to be wrong by 50% and asserting it fails.

**PyTorch oracle.** 55 conformance tests at `rtol = 10⁻¹⁰`. An AST-level test
forbids any shipped module from importing torch, jax, tensorflow or autograd, so
agreement is between implementations sharing no code. End to end, the same
network trained 120 epochs in both frameworks agrees to **1.4 × 10⁻¹⁶** on every
epoch's loss, with a final-loss difference of exactly zero. The residual is
summation order in BLAS versus ATen, not different mathematics.

**Property-based.** 18 Hypothesis properties over generated shapes and values:
gradient shapes, linearity, determinism, accumulation multiplicity,
round-tripping, and that an SGD step always moves every component downhill.

**Documentation claims.** 11 further tests recompute the figures quoted in the
docs and fail if the engine and the prose disagree.

**Totals:** 506 tests, 84% line coverage, ~21 seconds.

### 10.1 A multiclass result on real data

The experiments above are synthetic by design, so that the ground truth is known
and the result is checkable. One experiment uses real data: UCI optdigits
(Alpaydin & Kaynak, 1998; CC BY 4.0), 8x8 handwritten digit images, 3823
training and 1797 test samples.

A 64 -> 64 -> 32 -> 10 network with ReLU, Dropout(0.2) and Adam(0.002), trained
for 40 epochs with seed 7 (6,570 parameters):

| metric | value |
|---|---:|
| train accuracy | 0.9997 |
| **test accuracy** | **0.9683** |
| test errors | 57 of 1797 (3.17%) |
| best validation loss | 0.112062 (epoch 29 of 40) |
| final train / val loss | 0.029646 / 0.123483 |

Per-class recall ranges from 1.000 (class 0) to 0.925 (class 8). The confusion
matrix shows the expected structure for this dataset: the residual errors are
plausible digit confusions (8 mistaken for 1 six times; 7 for 5 six times),
not scattered noise.

The train/test gap — 0.9997 against 0.9683 — is mild overfitting, visible in the
loss curve as the training loss continuing to fall after validation flattens at
around epoch 29. Dropout(0.2) reduces it but does not remove it on 3823 samples.

**No state-of-the-art claim is made.** Published results on optdigits reach
roughly 97-99% with small models; this result is in that range and the point is
only that GradLab's engine trains a real multiclass model to the accuracy its
architecture should reach. The dataset download is opt-in
(`python examples/digits.py --download`) and GradLab never fetches it
implicitly.

---

## 11. Performance benchmarks

Full methodology, environment and raw data in `docs/benchmarks.md` and
`benchmarks/results.json`. Measurements were taken on an Apple Silicon machine
with Python 3.12, NumPy 2.5.4 and PyTorch 2.14.1, reporting medians with
interquartile ranges after warm-up.

**Framework overhead is per-operation and roughly constant** — about 9.6 µs on a
100-element elementwise expression and 113.8 µs on a 1,000,000-element one. The
*ratio* to raw NumPy therefore collapses from 9.8× to 1.1×.

**Matrix multiplication is BLAS.** At n = 512, GradLab's forward overhead is
1.9 µs against a 1.23 ms kernel call — under 0.2%. The GFLOP/s figures in the
report are NumPy's, labelled as such; **GradLab implements no matrix kernel and
claims none.**

**Engine scaling is linear in graph depth** with a flat per-node cost, as the
`O(E)` analysis predicts, verified to 5,000 nodes.

**On full training steps GradLab measured 0.84×–1.12× PyTorch's time.** This is
reported but explicitly **not** claimed as a performance win. Three factors
explain it, only one concerning GradLab: at these sizes both frameworks are
dominated by per-operation dispatch rather than arithmetic; the thread counts
are not matched (PyTorch pinned to one thread, NumPy's BLAS left at default);
and the models are small CPU models that PyTorch's engineering does not target.
The defensible reading is that GradLab's graph overhead is small enough to be
irrelevant at realistic sizes.

---

## 12. Educational visualisation system

The Gradient Microscope is a zero-dependency static web application reading JSON
traces produced by `gradlab.inspection`. Five views: the computation graph with
per-node inspection; a forward stepper; a backward stepper showing upstream
gradient, local derivative, chain-rule product and per-input accumulation; a
parameter-update view including Adam's moments and bias-correction factors; and
training dynamics with loss curves, gradient norms, decision-boundary evolution
and weight histograms.

**Every displayed value originates from a real execution.** Traces are generated
by running the presets through the engine. The backward view is a transcript of
the engine's own traversal, captured by an observer hook. Where a value is
unavailable — an intermediate freed by a backward pass — the trace says so
rather than substituting one. Large tensors are truncated with the truncation
reported, and statistics are still computed over the whole array.

**Security.** The default mode has **no execution surface**: static JSON, no
server-side evaluation. An optional endpoint, enabled only by an explicit flag,
evaluates user expressions through a hand-written tokeniser and recursive-descent
parser producing a plain-data tree restricted to an allowlist of 14 functions
plus arithmetic. There is no `eval`, `exec`, `compile` or `ast` use anywhere in
the package, and a test enforces that by AST inspection. Input length, token
count, tree depth, variable count, tensor size, request body size and request
rate are all bounded. The server binds to localhost only.

---

## 13. Limitations and threats to validity

Full list in `docs/limitations.md`. The ones that bear on the validity of the
claims above:

**Threats to the correctness claim.** Coverage is 84%, so 16% of lines are
untested, mostly error paths. Property tests run 60 generated examples per
property, not exhaustive enumeration. Conformance with PyTorch cannot catch an
error both implementations share. All tolerances are stated for float64;
float32 is supported but validated only to float32 precision.

**Threats to the reproducibility claim.** Bit-for-bit reproducibility holds on
one machine with fixed library versions. It is not guaranteed across BLAS
implementations, thread counts, NumPy versions or architectures, because matmul
reduction order changes the final bits.

**Threats to the performance claim.** One machine, one run, unmatched thread
counts, and `tracemalloc` seeing only Python allocations. The benchmark section
states each of these inline rather than in a footnote.

**Functional limits.** No higher-order derivatives, no in-place differentiable
operations, no GPU, no convolution, no thread safety. Each raises or is absent;
none is silently approximated.

---

## 14. Future work

Ordered by value relative to the project's purpose.

1. **A convolution layer**, derived and validated to the same standard. It would
   extend the visualiser to vision models, where decision boundaries and feature
   maps are far more legible.
2. **Higher-order differentiation**, by expressing backward rules in GradLab
   operations rather than NumPy. This makes `grad(grad(f))` possible and would
   let the Microscope visualise a Hessian-vector product.
3. **Checkpointing** — recompute subgraphs during backward instead of storing
   them, making the memory/time trade-off of reverse mode directly measurable
   and teachable.
4. **A guided-lesson mode in the Microscope**, stepping a learner through the
   fourteen lessons with the live graph beside each.
5. **A broader conformance matrix** across NumPy and Python versions in CI, to
   put bounds on the reproducibility claim.

Deliberately **not** planned: a GPU backend, distributed training, or any work
aimed at competing with production frameworks on speed. Those would add surface
area without serving the project's purpose.

---

## 15. References

Algorithms implemented here are standard; these are the sources for the
formulations used.

1. Linnainmaa, S. (1976). *Taylor expansion of the accumulated rounding error.*
   BIT 16(2), 146–160. — the earliest description of reverse-mode accumulation.
2. Rumelhart, D., Hinton, G. & Williams, R. (1986). *Learning representations by
   back-propagating errors.* Nature 323, 533–536.
3. Baydin, A., Pearlmutter, B., Radul, A. & Siskind, J. (2018). *Automatic
   differentiation in machine learning: a survey.* JMLR 18(153), 1–43.
4. Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep
   feedforward neural networks.* AISTATS. — Xavier initialisation.
5. He, K., Zhang, X., Ren, S. & Sun, J. (2015). *Delving deep into rectifiers.*
   ICCV. — Kaiming initialisation.
6. Kingma, D. & Ba, J. (2015). *Adam: A method for stochastic optimization.* ICLR.
7. Loshchilov, I. & Hutter, F. (2019). *Decoupled weight decay regularization.*
   ICLR. — AdamW.
8. Srivastava, N. et al. (2014). *Dropout: a simple way to prevent neural
   networks from overfitting.* JMLR 15, 1929–1958.
9. Paszke, A. et al. (2019). *PyTorch: An imperative style, high-performance deep
   learning library.* NeurIPS. — used here as a test oracle.
10. Maclaurin, D., Duvenaud, D. & Adams, R. (2015). *Autograd: Effortless
    gradients in NumPy.* ICML AutoML Workshop.
11. Harris, C. et al. (2020). *Array programming with NumPy.* Nature 585,
    357–362. — the array layer GradLab is built on.
12. Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten
    Digits.* UCI Machine Learning Repository. CC BY 4.0. — the optional dataset.

---

## Authorship

GradLab's source code was generated by Claude (Anthropic) in a single session,
working from a detailed written specification. The specification, the
requirements, the acceptance criteria and the direction are the project owner's;
the implementation is AI-generated.

This is stated in the README, in the development journal, and here, because a
reader evaluating the engineering is entitled to know how it was produced.
`docs/mastery-guide.md` exists to make ownership real rather than nominal: it
sets out what the owner should be able to answer for, and reserves a complete,
non-trivial extension (adding GELU end to end) to be implemented independently.
