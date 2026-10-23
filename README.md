# GradLab

**See how neural networks actually learn.**

An educational machine learning framework built from first principles: a custom
tensor type, a genuine reverse-mode automatic differentiation engine, a neural
network library, optimisers, and an interactive visualiser that shows the real
graph the engine built.

```python
import gradlab as gl

x = gl.Tensor(3.0, requires_grad=True)
y = x * x + 2 * x
y.backward()

print(y.item())        # 15.0
print(x.grad.item())   # 8.0
```

That `8.0` is `2x + 2` at `x = 3`, assembled by GradLab's own engine from two
separate branches of a computation graph it built while the forward pass ran.
NumPy supplies array storage and arithmetic. **Every derivative, the graph, the
traversal order and the gradient accumulation are GradLab's.** PyTorch is used
only as an independent test oracle and is never imported by the package — a test
enforces that by parsing every module.

---

## Why this exists

Most people meet automatic differentiation as `.backward()` — a method that
works. GradLab is for the next question: *how?*

Minimal teaching implementations are readable but scalar-valued, so the problems
that actually dominate autodiff never appear: broadcasting, matrix calculus,
numerical stability. Production frameworks handle all of those across hundreds
of thousands of lines of C++, which is not an explanation.

GradLab sits in between. The engine is about 150 lines. It handles tensors,
broadcasting and matmul properly. And it is instrumented, so you can watch a
backward pass happen instead of inferring it.

---

## Install

Requires Python 3.11+. The only runtime dependency is NumPy.

```bash
pip install gradlab-ml             # then: import gradlab
```

> **The PyPI name is `gradlab-ml`, the import name is `gradlab`.** They differ
> because `gradlab` was already taken on PyPI by an unrelated
> reinforcement-learning project. If you have that package installed too, the
> two will clash on `import gradlab`; they are unrelated and not intended to be
> used together.

To work on it instead:

```bash
git clone https://github.com/itsshreyasbhardwaj-design/gradlab
cd gradlab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",oracle" for the PyTorch comparison tests
```

> **If your system Python is older than 3.11** — common on macOS — use
> [uv](https://github.com/astral-sh/uv):
> `uv venv --python 3.12 .venv && uv pip install --python ./.venv/bin/python -e ".[dev]"`

Verify:

```bash
pytest -q                          # 506 tests, ~1 min
python examples/scalar_autograd.py
```

---

## The Gradient Microscope

```bash
gradlab-microscope generate        # run the presets through the engine
gradlab-microscope serve           # open the visualiser
```

Five views over a real execution:

| view | shows |
|---|---|
| **A · Graph** | the actual graph, per-node inputs, outputs, saved values and rule |
| **B · Forward** | step through the forward pass, one operation at a time |
| **C · Backward** | upstream gradient × local derivative = what each input receives |
| **D · Updates** | every parameter before, its gradient, the update, and after — plus Adam's moments and bias-correction factors |
| **E · Dynamics** | measured loss curves, gradient norms, decision boundaries over training, weight histograms |

Every number comes from a trace produced by running real code. There are no
hardcoded diagrams: with no traces generated, the page says so rather than
showing an illustration.

`--allow-expressions` additionally enables typing your own expression and
watching it differentiate. That path is a hand-written parser over an allowlist
of 14 functions — no `eval`, no `exec`, no `ast` — bounded in length, depth,
variable count and tensor size, bound to localhost. See
[SECURITY.md](SECURITY.md).

---

## A real example

```python
import gradlab as gl
from gradlab import data, nn, optim, training

X, y = data.make_moons(n_samples=600, noise=0.2, seed=0)
Xtr, Xte, ytr, yte = data.train_test_split(X, y, test_size=0.25, seed=0)

model = nn.Sequential(
    nn.Linear(2, 16), nn.Tanh(),
    nn.Linear(16, 16), nn.Tanh(),
    nn.Linear(16, 1),
)

history = training.fit(
    model,
    optim.Adam(model.parameters(), lr=0.01),
    nn.binary_cross_entropy_with_logits,
    data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=32, shuffle=True, seed=0),
    epochs=300,
    val_loader=data.DataLoader(data.TensorDataset(Xte, yte), batch_size=128),
    metric=nn.accuracy,
)
print(history.summary())
```

Or write the loop by hand — `fit` does exactly this and nothing more:

```python
optimizer.zero_grad()                       # gradients accumulate; clear them
predictions = model(x)                      # forward pass builds the graph
loss = nn.mse_loss(predictions, targets)    # scalar objective
loss.backward()                             # reverse-mode pass
optimizer.step()                            # update, under no_grad
```

`examples/linear_regression.py` runs both from the same seed and asserts they
reach bitwise-identical parameters.

---

## Experiments

All reproducible from a seed. Measured results, not targets.

| experiment | result |
|---|---|
| **Linear regression** | converges to the irreducible noise floor: final loss **0.010373** against a noise variance of 0.0100; recovers the generating weights to ±0.011 |
| **XOR** | **53.3%** with a purely affine 2→8→1 network, **100%** with one `tanh` — same capacity, same seed, same data |
| **Two-moons** | **96.7%** validation accuracy vs **82.7%** for a logistic-regression baseline |
| **Spirals (3-class)** | **99.6%** validation vs **67.3%** for a nearest-centroid reference |
| **PyTorch comparison** | identical weights, 120 epochs: losses agree to **1.4e-16**, final loss difference exactly **0** |
| **Handwritten digits** | **96.83%** test accuracy on UCI optdigits (1797 held-out samples, 57 wrong) with a 6,570-parameter MLP; the dataset download is **opt-in** |

```bash
python examples/scalar_autograd.py      # the engine, from first principles
python examples/linear_regression.py    # hand-written loop vs fit()
python examples/xor.py                  # why nonlinearity is not optional
python examples/two_moons.py            # full reproducibility report
python examples/classification.py       # multiclass: blobs and spirals
python examples/pytorch_comparison.py   # independent verification
python examples/digits.py --download    # UCI optdigits (opt-in download)
python examples/lessons.py              # the fourteen lessons
```

---

## Architecture

```
gradlab/
├── tensor.py              Tensor: data, grad, grad_fn, version counter
├── autograd/
│   ├── graph.py           Node and Context — the graph itself
│   ├── function.py        Function base class + unbroadcast
│   ├── engine.py          topological order + the backward pass
│   └── gradcheck.py       central finite differences
├── ops/                   every derivative rule, one class per operation
├── nn/                    Module, Parameter, layers, losses, initialisation
├── optim/                 SGD, Adam, AdamW
├── data/                  datasets, batching, synthetic generators
├── training/              the training loop and measured history
└── inspection/            the stable trace API the Microscope reads
```

Dependency direction is one-way and enforced by tests: `inspection` reads the
engine's public surface, nothing in `src/` imports another autodiff framework,
and the runtime dependency list is exactly `["numpy"]`.

---

## Mathematical foundations

Fully derived in [docs/backpropagation.md](docs/backpropagation.md). The three
that matter most:

**The chain rule, as implemented.** `∂L/∂in = ∂L/∂out · ∂out/∂in` — upstream
gradient times local derivative. Each rule implements only the local part.

**Broadcasting.** A tensor stretched in the forward pass has its gradient
**summed** over the stretched axes. A bias `(3,)` added to a `(32, 3)` batch
gets a gradient of `[32, 32, 32]`, not ones. This is the single most common bug
in a hand-rolled engine, so GradLab handles it in exactly one place.

**Matrix multiplication.** For `C = A·B`: `∂L/∂A = G·Bᵀ` and `∂L/∂B = Aᵀ·G`. The
transposes are what make the shapes work — and a gradient always has the shape
of the thing it differentiates.

---

## How correctness is established

Four independent strategies, because a hand-written test and a hand-written rule
can be wrong together.

* **Analytic** — known derivatives, including every specification edge case.
* **Finite differences** — central differences across 36 operations, worst
  absolute error **1.04e-08**. The checker is itself validated by corrupting a
  rule and asserting it fails.
* **PyTorch oracle** — 55 conformance tests at `rtol=1e-10`, including
  optimisers over 20-step trajectories and a 60-epoch learning curve.
* **Property-based** — 18 Hypothesis invariants over generated shapes and values.

Plus 11 tests that recompute the numbers quoted in the documentation, so a stale
figure fails CI.

```
506 tests · 84% line coverage · ruff clean · mypy clean · ~1 min
```

Full measured error table: [docs/validation.md](docs/validation.md).

---

## Performance, honestly

GradLab's arithmetic is NumPy's, and matmul is a BLAS call. **GradLab implements
no matrix kernel and claims none.** What it adds is a Python-level graph, and
that overhead is per-*operation* — roughly 5–10 µs — so it dominates on tiny
tensors and vanishes on large ones.

On small CPU training steps it measured **0.84×–1.12× PyTorch's time**. That is
*not* a claim of being faster: the thread counts are unmatched (PyTorch pinned to
one thread, NumPy's BLAS left at default), and at these sizes both frameworks
are dominated by dispatch rather than arithmetic.

Methodology, environment and raw data: [docs/benchmarks.md](docs/benchmarks.md).

---

## Known limitations

No higher-order derivatives, no in-place differentiable operations, no GPU, no
convolution or recurrent layers, not thread-safe, float64/float32 only.

Each of these raises a clear error or is simply absent. **Nothing is silently
approximated.** Full list: [docs/limitations.md](docs/limitations.md).

---

## Documentation

| | |
|---|---|
| [autograd-engine.md](docs/autograd-engine.md) | how reverse mode is implemented, in full |
| [backpropagation.md](docs/backpropagation.md) | every derivative rule, derived |
| [tensor-design.md](docs/tensor-design.md) | tensor semantics, dtypes, aliasing |
| [architecture.md](docs/architecture.md) | module layering and contracts |
| [optimizers.md](docs/optimizers.md) | SGD, momentum, Adam, AdamW |
| [validation.md](docs/validation.md) | how correctness is established, with numbers |
| [benchmarks.md](docs/benchmarks.md) | measured performance and its caveats |
| [design-decisions.md](docs/design-decisions.md) | 16 choices and their costs |
| [limitations.md](docs/limitations.md) | what GradLab does not do |
| [lessons.md](docs/lessons.md) | fourteen guided lessons |
| [mastery-guide.md](docs/mastery-guide.md) | questions to answer, exercises to implement |
| [development-journal.md](docs/development-journal.md) | the real build record, including defects |
| [report.md](docs/report.md) | the full engineering report |

---

## AI-assisted development

**GradLab's source code was generated by Claude (Anthropic) in a single session,
working from a detailed written specification.**

The specification, requirements, acceptance criteria and direction are the
project owner's. The implementation is AI-generated. This is stated here, in
[docs/development-journal.md](docs/development-journal.md) and in
[docs/report.md](docs/report.md) because anyone evaluating the engineering is
entitled to know how it was produced.

[docs/mastery-guide.md](docs/mastery-guide.md) exists to make ownership real
rather than nominal. It sets out what you should be able to answer for, an
oral-examination bank, exercises — and reserves one complete, non-trivial
extension (adding GELU end to end: derivation, forward, backward,
finite-difference test, PyTorch comparison, `nn` module, Microscope support) to
be implemented independently.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: every new operation needs a
derivation, a forward, a backward, a finite-difference test and — where an
equivalent exists — a PyTorch comparison.

## Licence

MIT. See [LICENSE](LICENSE).

The optional handwritten-digits dataset is UCI optdigits (Alpaydin & Kaynak,
1998), CC BY 4.0, and is never downloaded without an explicit flag.
