# Limitations

The authoritative list of what GradLab does not do. Everything here either
raises a clear error or is simply absent — **nothing on this list is silently
approximated**.

If you find GradLab doing something this file does not describe, that is a bug.

## Not implemented, and raises

| feature | behaviour |
|---|---|
| higher-order derivatives (`backward` of a gradient) | the backward pass runs on NumPy arrays, builds no graph, and cannot be differentiated again |
| in-place differentiable ops (`add_`, mutating `+=`) | not provided; `x += 1` rebinds to a new tensor |
| complex dtypes | `requires_grad=True` raises `DTypeError` |
| integer gradients | `requires_grad=True` on an explicit integer dtype raises `DTypeError` |
| GPU / device placement | no backend exists and none is claimed |
| sparse tensors | absent |
| distributed training | absent |
| JIT / graph compilation | absent; GradLab is define-by-run only |
| automatic mixed precision | absent |
| ONNX or any export format | absent |

## Not implemented (P2 scope, deliberately)

Convolutional layers, pooling, batch/layer normalisation, recurrent layers,
attention, embeddings, learning-rate scheduler objects (pass a callable to
`fit(lr_schedule=…)` instead), and optimisers beyond SGD/Adam/AdamW.

These are omitted because the project's stated priority was a small, correct,
fully validated core over a large surface of partially correct features. A
convolution with a subtly wrong gradient would be worse than no convolution.

## Thread safety

**GradLab is not thread-safe.** The gradient-mode flag (`no_grad`) is a single
process-global integer, not thread-local. Two threads doing forward passes where
one is inside `no_grad()` will interfere. The graph itself has no locking.

Use one process per training run. This is a deliberate simplification, not an
oversight waiting to be fixed.

## Numerical limits

* **float64 is the correctness dtype.** Every tolerance in `docs/validation.md`
  is stated for it. float32 works and is tested, but its gradients are correct to
  roughly 1e-6, not 1e-10.
* **float16 / bfloat16** are not supported.
* **`log_softmax` has the largest finite-difference disagreement** of any
  operation (1.04e-8 absolute). That is cancellation in the *numerical* check,
  not an error in the rule — the analytic result matches PyTorch to 1e-10.
* **Overflow is detected, not prevented.** A loss that exceeds float64 range
  raises; GradLab does not rescale to avoid it.
* **No gradient clipping by default.** `clip_grad_norm` exists and must be called
  explicitly. Silently shrinking a gradient would change the optimisation problem
  without telling you.

## Reproducibility

Seeded runs are bit-for-bit reproducible **on one machine, with one Python, one
NumPy and one dtype**. They are *not* guaranteed:

* across different BLAS implementations or thread counts — `matmul` reduction
  order changes the last bits;
* across NumPy versions, if a kernel's accumulation order changed;
* across platforms (x86 vs ARM may differ in FMA contraction).

`integration/test_training.py::test_two_identical_runs_agree_bit_for_bit`
verifies the within-machine guarantee, which is the one actually made.

## Performance

GradLab is a **single-threaded, Python-level framework**. Its per-operation
overhead is roughly 5–10 µs (see `docs/benchmarks.md`), which is irrelevant for
large tensors and dominant for small ones.

`docs/benchmarks.md` section 5 shows GradLab at 0.84×–1.12× PyTorch's time per
training step on small CPU models. **That is not a claim of being faster.** The
thread counts in that comparison are not matched — PyTorch is pinned to one
thread while NumPy's BLAS is left at its default — and the models are small
enough that both frameworks are dominated by per-operation dispatch rather than
arithmetic. Read it as "the graph overhead is small enough not to matter", not
as a competitive result.

No claim is made about: GPU performance (there is no GPU backend), large-model
performance, memory efficiency relative to any other framework, or performance
on any hardware other than the machine named in the benchmark report.

## The Gradient Microscope

* **Traces are static.** `gradlab-microscope generate` runs the presets and
  writes JSON; the web app reads those files. It does not re-run anything.
* **Large tensors are truncated in traces** to 128 elements by default, which is
  always reported in the payload (`truncated: true`) and shown in the UI.
  Statistics are still computed over the whole array.
* **Intermediate values may be unavailable.** If a graph was exported after a
  `backward()` without `retain_graph=True`, saved values have been freed; the
  trace says `"intermediate tensor already freed"` rather than inventing a value.
* **The expression endpoint is opt-in** (`--allow-expressions`) and restricted to
  an allowlist of 14 functions plus arithmetic. It parses with a hand-written
  recursive-descent parser — no `eval`, no `exec`, no `ast` — and bounds input
  length, token count, tree depth, variable count and tensor size.
* **The server binds to 127.0.0.1 only** and has no authentication. It is a local
  teaching tool. Do not expose it; see `SECURITY.md`.
* **No live training in the browser.** Training curves come from completed runs.

## Datasets

* The synthetic generators (`make_moons`, `make_xor`, `make_blobs`,
  `make_spiral`, `make_regression`) are GradLab's own implementations of
  well-known toy datasets, not wrappers around scikit-learn.
* The only external dataset is UCI optdigits, and **it is never downloaded
  automatically** — `examples/digits.py` requires an explicit `--download` flag
  and prints exactly what it would fetch and from where.

## Scope of the validation

Covered in `docs/validation.md`, but worth repeating here: 84% line coverage
means 16% is not covered, mostly error paths and CLI plumbing. Property tests
run 60 generated examples per property, not exhaustive enumeration. Conformance
testing proves agreement with PyTorch, which would not catch an error both
implementations share.

## Honest summary

GradLab is a correct, well-validated, small CPU framework for learning how
automatic differentiation and neural network training actually work. It is not
a production training framework, it is not fast at scale, and it does not try to
be either.
