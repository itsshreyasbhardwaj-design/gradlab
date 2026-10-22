# Contributing to GradLab

GradLab is a teaching framework. The bar for a change is therefore not only "it
works" but "a reader can see why it is correct".

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,oracle]"
pytest -q          # 506 tests, ~21s
```

Python 3.11+ is required. If your system Python is older, use
`uv venv --python 3.12 .venv`.

## The quality gates

All four must pass. CI runs exactly these.

```bash
ruff check src tests examples benchmarks
ruff format --check src tests examples
mypy
pytest -q
```

## Adding a differentiable operation

This is the most common contribution, and it has seven required parts. A PR with
fewer will be asked for the rest.

1. **The derivation**, added to `docs/backpropagation.md` in the existing style.
   Show the working — that file is the explanation, not a reference card.
2. **A `Function` subclass** in the right `src/gradlab/ops/` module, with `op`,
   `backward_rule`, and a docstring covering domain and numerical behaviour.
3. **The backward rule**, returning one gradient per input. Return gradients in
   the *output* shape and let `Function.apply` unbroadcast, unless your
   gradients are already input-shaped — then set `auto_unbroadcast = False`.
4. **A finite-difference test**: add a row to `CASES` in
   `tests/gradients/test_op_gradients.py`. Choose inputs inside the operation's
   domain and **away from any kink** — at a kink the central difference averages
   two different one-sided slopes and need match no valid subgradient.
5. **A PyTorch comparison** in `tests/conformance/`, if an equivalent exists.
6. **Registration**: add it to `ALL_OPS` in `src/gradlab/ops/__init__.py`, and a
   `Tensor` method or `nn` module if it should be user-facing.
7. **Microscope support**: an `EXPR_TEMPLATES` entry in
   `src/gradlab/inspection/schema.py`, and `ALLOWED_OPS` in
   `inspection/expressions.py` if it should be typeable by users.

A worked example of all seven — adding GELU — is specified in
[docs/mastery-guide.md](docs/mastery-guide.md).

## Principles

These are not style preferences; they are why the project is trustworthy.

**Correctness over convenience.** If an operation cannot be implemented
correctly, it should raise rather than approximate. `docs/limitations.md` is a
feature.

**Fail loudly.** GradLab raises on domain errors, non-finite gradients, shape
mismatches and stale saved values. It does not clip, skip, or return `nan`. A
silent wrong answer is worse than an exception.

**No number without a measurement.** Documentation quotes specific figures, and
`tests/integration/test_documented_claims.py` recomputes them. If you change
behaviour that a documented figure depends on, update both — CI will tell you.

**No performance claim without a benchmark.** `benchmarks/bench.py` produces
`docs/benchmarks.md`; do not hand-edit that file.

**One backward implementation.** The visualiser reads the engine through an
observer hook. Do not add a second traversal.

**Dependencies stay at one.** NumPy. A test asserts it. Dev and oracle extras are
separate and never imported by `src/`.

## Tests

| directory | for |
|---|---|
| `tests/unit` | behaviour of one component |
| `tests/gradients` | finite-difference checks |
| `tests/conformance` | PyTorch oracle comparisons (mark `oracle`) |
| `tests/property` | Hypothesis invariants |
| `tests/integration` | end-to-end training, reproducibility, doc claims |

Prefer a test that computes the expectation *independently* — from the published
formula, or from a reference model — over one that asserts the implementation
equals itself. `test_matches_a_hand_computed_three_step_trajectory` is the
pattern to copy.

**Never report a skipped test as passing.** Conformance tests skip cleanly
without PyTorch; that is a skip, not a pass.

## Commit and PR

* Conventional-ish subject lines (`fix:`, `feat:`, `docs:`, `test:`).
* Say what you measured, not only what you changed.
* Update `CHANGELOG.md` under `[Unreleased]`.
* If you found a defect, consider adding it to `docs/development-journal.md` —
  that file's value is that it is honest.

## Reporting a bug

A gradient bug report should include: the expression, the expected gradient and
where it comes from, what GradLab produced, and ideally a `gl.gradcheck(...)`
report. Security issues go to [SECURITY.md](SECURITY.md) instead.
