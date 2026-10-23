# Changelog

All notable changes to GradLab are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0, the public API may change in a minor release. Anything that would break
`import gradlab as gl; gl.Tensor(...)` will be called out explicitly.

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-10-23

First release. Everything below is new.

### Core

* `Tensor` over NumPy storage: `data`, `shape`, `ndim`, `size`, `dtype`,
  `requires_grad`, `grad`, `is_leaf`, `grad_fn`, `backward()`, `zero_grad()`,
  `detach()`, `clone()`, `retain_grad()`.
* Reverse-mode autodiff engine: define-by-run graph, iterative reverse
  topological traversal, gradient accumulation, leaf/non-leaf semantics,
  explicit graph release.
* **Version counters** on every tensor, so a tensor mutated between forward and
  backward raises `InplaceModificationError` instead of yielding a stale
  gradient.
* **Non-scalar `backward()` requires an explicit upstream gradient** — GradLab
  never assumes a seed of ones.
* `no_grad()` / `enable_grad()` context managers.
* `gradcheck` with central finite differences, reporting measured error
  statistics rather than only a verdict.

### Operations

Arithmetic (`+ − × ÷`, negation, absolute value, constant and tensor powers),
transcendentals (`exp`, `log`, `sqrt`, `sin`, `cos`), activations (`tanh`,
`sigmoid`, `relu`, `softmax`, `log_softmax`), reductions (`sum`, `mean`, `max`,
`min` with `axis`/`keepdims`), matrix multiplication (1-D through batched with
broadcasting), and shape operations (`reshape`, `transpose`, `flatten`,
indexing with scatter-add backward, `concat`, `stack`).

NumPy-compatible broadcasting with correct gradient reduction throughout.

### Neural networks

`Module`, `Parameter`, `Sequential`, recursive parameter and module discovery,
train/eval modes, `state_dict()`/`load_state_dict()` with validation before
mutation. Layers: `Linear`, `Flatten`, `Dropout` (inverted). Activations:
`ReLU`, `Sigmoid`, `Tanh`, `Identity`. Losses: `mse_loss`,
`binary_cross_entropy_with_logits`, `cross_entropy_with_logits` — all
numerically stable, logits only. Initialisation: uniform, normal, Xavier,
Kaiming.

### Optimisation and training

`SGD` (momentum, Nesterov, weight decay), `Adam` (bias-corrected), `AdamW`
(decoupled decay), all with state serialisation. `Dataset`, `TensorDataset`,
`DataLoader` with seeded shuffling, `train_test_split`. `fit()`, `evaluate()`,
`global_grad_norm`, `clip_grad_norm`, and a `History` of measured metrics.
Synthetic generators: regression, XOR, moons, blobs, spirals. Optional UCI
optdigits loader (download is opt-in); measured at 96.83% test accuracy.

### Gradient Microscope

Stable, versioned JSON trace schema. Graph export, expression tracing, training
step tracing, decision surfaces. Zero-dependency static web app with five views.
`gradlab-microscope` CLI: `generate`, `trace`, `serve`. Optional restricted
expression endpoint with a hand-written parser, an operation allowlist, and
bounded input.

### Validation

506 tests, 84% line coverage: 332 unit, 62 finite-difference,
55 PyTorch conformance, 18 property-based, 27 integration, 12 doctests. Architecture tests
forbidding other autodiff frameworks and `eval`/`exec` in shipped code.
Documentation-claim tests recomputing quoted figures.

### Documentation

Engine internals, full derivations, tensor design, architecture, optimisers,
validation with a measured error table, benchmarks, 16 design decisions,
limitations, 14 guided lessons, a mastery guide with exercises, a development
journal, and the engineering report.

### Known limitations at 0.1.0

No higher-order derivatives, no in-place differentiable operations, no GPU, no
convolution or recurrent layers, not thread-safe, float64/float32 only. See
[docs/limitations.md](docs/limitations.md).

[Unreleased]: https://github.com/itsshreyasbhardwaj-design/gradlab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/itsshreyasbhardwaj-design/gradlab/releases/tag/v0.1.0
