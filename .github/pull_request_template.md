## What this changes

## Why

## Quality gates

- [ ] `ruff check src tests examples benchmarks`
- [ ] `ruff format --check src tests examples`
- [ ] `mypy`
- [ ] `pytest -q`

## If this adds or changes a differentiable operation

- [ ] Derivation added to `docs/backpropagation.md`
- [ ] Forward and backward implemented as a `Function`
- [ ] Finite-difference case added to `tests/gradients/test_op_gradients.py`
- [ ] PyTorch comparison added (or noted why no equivalent exists)
- [ ] Registered in `ALL_OPS`, and exposed on `Tensor`/`nn` if user-facing
- [ ] Microscope support (`EXPR_TEMPLATES`, and `ALLOWED_OPS` if typeable)

## If this changes behaviour a documented number depends on

- [ ] Documentation updated
- [ ] `tests/integration/test_documented_claims.py` still passes
- [ ] Benchmarks regenerated with `python benchmarks/bench.py` (never hand-edited)

## What I measured

(Not just what changed — what did you observe?)
