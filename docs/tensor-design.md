# Tensor design

## Construction

```python
Tensor(3.0)                           # 0-d, float64, requires_grad=False
Tensor([1, 2, 3], requires_grad=True) # 1-d, float64 (ints promoted for grad)
Tensor(np.zeros((2, 3)), dtype="float32")
```

`Tensor(obj)` accepts a Python scalar, nested sequence, `numpy.ndarray`, or
another `Tensor`. The data is **copied** unless the input is already a NumPy
array of the right dtype and `copy=False` is passed. `Tensor` never takes
ownership of a caller's array silently.

If `requires_grad=True` and the inferred dtype is not floating, the dtype is
promoted to the default float dtype. This is a deliberate convenience:
`Tensor([1, 2], requires_grad=True)` is almost always a user wanting floats.
`Tensor([1, 2], dtype="int64", requires_grad=True)` raises `DTypeError`, because
there the user was explicit and the request is not differentiable.

## Properties

| property        | meaning |
|-----------------|---------|
| `data`          | the backing `numpy.ndarray`. Assigning to it bumps `_version`. |
| `shape`         | tuple, `()` for scalars |
| `ndim`, `size`  | ints |
| `dtype`         | `numpy.dtype` |
| `requires_grad` | whether this tensor participates in differentiation |
| `grad`          | `Tensor` or `None`; accumulated gradient |
| `is_leaf`       | `grad_fn is None` |
| `grad_fn`       | the `Node` that produced this tensor, or `None` |

## Semantics by rank

GradLab places no special meaning on rank. A "scalar" is a tensor with
`size == 1` (usually `shape == ()`); only such a tensor may call `backward()`
without an explicit seed. Vectors, matrices and higher-rank tensors are all just
`ndarray`s; batch dimensions are a *convention of the layers*, not of the
Tensor: `nn.Linear` treats the last axis as features and all leading axes as
batch.

## Copy and aliasing

* `detach()` returns a **new leaf** sharing the same buffer, with
  `requires_grad=False`. Mutating through either view is visible in both, and
  the version counter is shared — this is what makes mutation detectable.
* `clone()` returns a differentiable copy with its own buffer (identity op, so
  gradient flows through).
* `numpy()` returns the underlying array (no copy); `tolist()`/`item()` copy out.
* Reshape/transpose return views where NumPy would; the gradient rule reverses
  the view regardless.

## In-place modification

GradLab exposes **no in-place differentiable operators** — there is no `add_`,
no `__iadd__` that mutates. `x += 1` rebinds `x` to a new tensor.

The one sanctioned mutation is writing to `.data` (used by optimisers under
`no_grad`). It is legal, and it bumps `_version`. If any live node had saved
that tensor for backward, the next `backward()` raises
`InplaceModificationError` rather than returning a wrong gradient. So the rule
"call `optimizer.step()` only after `backward()` has finished" is enforced, not
just documented.

## Gradient ownership

A tensor owns its own `.grad`. Gradients *accumulate* (`+=`) across
`backward()` calls; `zero_grad()` sets `.grad = None`. `Module.zero_grad()` and
`Optimizer.zero_grad()` walk their parameters and do the same. A non-leaf gets
no `.grad` unless `retain_grad()` was called on it, and GradLab warns loudly
(`UserWarning`) if you read `.grad` on a non-leaf that never retained it — a
common and otherwise silent source of confusion.

## dtypes

Supported: `float64` (default, all tolerances stated for it), `float32`.
Integer and boolean tensors exist but cannot require grad; they are useful as
class labels and masks. Mixed-dtype binary ops follow NumPy promotion, except
that a `float32` tensor combined with a Python float stays `float32`.

## Unsupported — and raising, not approximating

Complex dtypes, sparse tensors, GPU/device placement, higher-order derivatives
(`backward()` of a gradient), and in-place differentiable ops all raise
`NotImplementedError` or are simply absent. `docs/limitations.md` is the
authoritative list. GradLab never substitutes a wrong approximation for a
missing feature.
