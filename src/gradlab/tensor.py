"""The :class:`Tensor` — GradLab's multidimensional array with gradient history.

A ``Tensor`` is a thin, deliberate wrapper around a ``numpy.ndarray``. NumPy
owns the bytes and the elementwise kernels; GradLab owns the differentiation
semantics, the graph edge this tensor represents, and the gradient that
accumulates on it.

    >>> import gradlab as gl
    >>> x = gl.Tensor(3.0, requires_grad=True)
    >>> y = x * x + 2 * x
    >>> y.backward()
    >>> y.item(), x.grad.item()
    (15.0, 8.0)

Those numbers come from GradLab's own engine: ``y`` is built from two
multiplications and an addition recorded as graph nodes, and ``8.0`` is
``2x + 2`` at ``x = 3`` assembled by the chain rule from both branches.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from gradlab._config import DEFAULT_DTYPE, FLOAT_DTYPES
from gradlab._errors import DTypeError, ShapeError

if TYPE_CHECKING:  # pragma: no cover
    from gradlab.autograd.graph import Node

ArrayLike = Any


class Tensor:
    """A multidimensional array that can carry gradient history.

    Parameters
    ----------
    data
        A Python scalar, nested sequence, ``numpy.ndarray``, or ``Tensor``.
    requires_grad
        Whether to record operations on this tensor for differentiation.
        Only floating-point tensors may require grad.
    dtype
        Explicit dtype. Defaults to ``float64`` for new data, or the input
        array's dtype if it is already floating point.
    copy
        Copy the incoming buffer (default ``True``). GradLab does not silently
        take ownership of a caller's array.
    """

    # __weakref__ is declared explicitly: graph nodes hold a *weak* reference to
    # the tensor they produced, which is what keeps Tensor -> Node -> Tensor from
    # forming an uncollectable reference cycle.
    _data: np.ndarray
    _requires_grad: bool
    _grad: Tensor | None
    _grad_fn: Node | None
    _version: int
    _retain_grad: bool

    __slots__ = (
        "__weakref__",
        "_data",
        "_grad",
        "_grad_fn",
        "_requires_grad",
        "_retain_grad",
        "_version",
    )

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        dtype: Any = None,
        copy: bool = True,
        *,
        _internal: bool = False,
    ) -> None:
        if isinstance(data, Tensor):
            data = data._data

        if _internal:
            arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        else:
            explicit_dtype = dtype is not None
            arr = (
                np.array(data, dtype=dtype, copy=copy)
                if copy or explicit_dtype
                else np.asarray(data, dtype=dtype)
            )
            if not np.issubdtype(arr.dtype, np.floating) and requires_grad:
                if explicit_dtype:
                    raise DTypeError(
                        f"requires_grad=True is not possible for dtype {arr.dtype}: only "
                        f"{', '.join(np.dtype(d).name for d in FLOAT_DTYPES)} can carry a "
                        "gradient. Drop the explicit dtype to get float64, or set "
                        "requires_grad=False."
                    )
                # Convenience: Tensor([1, 2], requires_grad=True) means floats. An
                # integer tensor without requires_grad is left alone, because that is
                # exactly what class labels and masks need.
                arr = arr.astype(DEFAULT_DTYPE)
            if requires_grad and arr.dtype not in [np.dtype(d) for d in FLOAT_DTYPES]:
                raise DTypeError(
                    f"dtype {arr.dtype} cannot carry a gradient; use float32 or float64."
                )

        self._data = arr
        self._requires_grad = bool(requires_grad)
        self._grad: Tensor | None = None
        self._grad_fn: Node | None = None
        self._version = 0
        self._retain_grad = False

    # -- core properties ---------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """The backing NumPy array.

        Writing to this array (or assigning a new one) bumps the version
        counter. If a live graph node had saved this tensor for backward, the
        next ``backward()`` raises :class:`InplaceModificationError` instead of
        returning a quietly wrong gradient.
        """
        return self._data

    @data.setter
    def data(self, value: ArrayLike) -> None:
        arr = value._data if isinstance(value, Tensor) else np.asarray(value)
        if arr.shape != self._data.shape:
            raise ShapeError(
                f"Cannot assign data of shape {arr.shape} to a tensor of shape "
                f"{self._data.shape}; a tensor's shape is fixed after construction."
            )
        self._data = arr.astype(self._data.dtype, copy=False)
        self._version += 1

    def _bump_version(self) -> None:
        """Record that this tensor's buffer was written to in place."""
        self._version += 1

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return int(self._data.size)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if value and not self.is_leaf:
            raise RuntimeError(
                "requires_grad can only be changed on a leaf tensor. This tensor was "
                "produced by an operation, so its requires_grad follows its inputs. "
                "Use .detach() to get a leaf copy."
            )
        if value and self.dtype not in [np.dtype(d) for d in FLOAT_DTYPES]:
            raise DTypeError(f"dtype {self.dtype} cannot carry a gradient.")
        self._requires_grad = bool(value)

    @property
    def grad(self) -> Tensor | None:
        """The accumulated gradient, or ``None``.

        Reading ``.grad`` on a non-leaf that did not call :meth:`retain_grad`
        warns, because it will always be ``None`` and that is a frequent source
        of silent confusion.
        """
        if self._grad is None and not self.is_leaf and not self._retain_grad:
            warnings.warn(
                "Reading .grad on a non-leaf tensor, which is always None. GradLab "
                "only stores gradients on leaves; call .retain_grad() before "
                "backward() if you want this intermediate gradient.",
                UserWarning,
                stacklevel=2,
            )
        return self._grad

    @grad.setter
    def grad(self, value: Tensor | None) -> None:
        if value is None:
            self._grad = None
            return
        value = as_tensor(value)
        if value.shape != self.shape:
            raise ShapeError(
                f"A gradient must have the shape of the value it differentiates: "
                f"got {value.shape}, expected {self.shape}."
            )
        self._grad = value.detach()

    @property
    def grad_fn(self) -> Node | None:
        """The graph node that produced this tensor, or ``None`` for a leaf."""
        return self._grad_fn

    @property
    def is_leaf(self) -> bool:
        """``True`` when no operation produced this tensor."""
        return self._grad_fn is None

    # -- gradient machinery ------------------------------------------------

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        """Add ``grad`` into ``.grad`` (creating it if needed). Engine-internal."""
        if not self._requires_grad:
            return
        g = np.asarray(grad).astype(self.dtype, copy=False)
        if self._grad is None:
            self._grad = Tensor(g.copy(), copy=False, _internal=True)
        else:
            self._grad._data = self._grad._data + g

    def backward(self, grad: ArrayLike | None = None, retain_graph: bool = False) -> None:
        """Propagate gradients from this tensor back to every leaf behind it.

        ``grad`` may be omitted only for a single-element tensor. Gradients
        **accumulate** into ``.grad``; call :meth:`zero_grad` (or the
        optimiser's) between steps.
        """
        from gradlab.autograd.engine import run_backward

        run_backward(self, grad=grad, retain_graph=retain_graph)

    def zero_grad(self) -> None:
        """Discard any accumulated gradient on this tensor."""
        self._grad = None

    def retain_grad(self) -> Tensor:
        """Ask the engine to store this non-leaf tensor's gradient too."""
        if not self._requires_grad:
            raise RuntimeError(
                "retain_grad() is meaningless on a tensor that does not require grad."
            )
        self._retain_grad = True
        return self

    def detach(self) -> Tensor:
        """A new leaf sharing this tensor's buffer, with no gradient history."""
        out = Tensor(self._data, copy=False, _internal=True)
        out._requires_grad = False
        return out

    def clone(self) -> Tensor:
        """A differentiable copy with its own buffer."""
        from gradlab.ops.shape import Identity

        return Identity.apply(self)

    # -- conversions -------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """The underlying array (no copy). Mutating it bumps nothing — use .data."""
        return self._data

    def item(self) -> float:
        if self.size != 1:
            raise ValueError(f"item() needs a single-element tensor; this one has {self.size}.")
        return self._data.reshape(()).item()

    def tolist(self) -> Any:
        return self._data.tolist()

    def astype(self, dtype: Any) -> Tensor:
        """A detached copy in a new dtype (not differentiable across the cast)."""
        return Tensor(self._data.astype(dtype), copy=False, _internal=True)

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other: Any) -> Tensor:
        from gradlab.ops.arithmetic import Add

        return Add.apply(self, other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Tensor:
        from gradlab.ops.arithmetic import Sub

        return Sub.apply(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        from gradlab.ops.arithmetic import Sub

        return Sub.apply(other, self)

    def __mul__(self, other: Any) -> Tensor:
        from gradlab.ops.arithmetic import Mul

        return Mul.apply(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Tensor:
        from gradlab.ops.arithmetic import Div

        return Div.apply(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        from gradlab.ops.arithmetic import Div

        return Div.apply(other, self)

    def __neg__(self) -> Tensor:
        from gradlab.ops.arithmetic import Neg

        return Neg.apply(self)

    def __pow__(self, exponent: Any) -> Tensor:
        from gradlab.ops.arithmetic import Pow, PowTensor

        if isinstance(exponent, Tensor):
            return PowTensor.apply(self, exponent)
        return Pow.apply(self, exponent=float(exponent))

    def __rpow__(self, base: Any) -> Tensor:
        from gradlab.ops.arithmetic import PowTensor

        return PowTensor.apply(as_tensor(base), self)

    def __matmul__(self, other: Any) -> Tensor:
        from gradlab.ops.matrix import MatMul

        return MatMul.apply(self, other)

    def __rmatmul__(self, other: Any) -> Tensor:
        from gradlab.ops.matrix import MatMul

        return MatMul.apply(other, self)

    def __abs__(self) -> Tensor:
        from gradlab.ops.arithmetic import Abs

        return Abs.apply(self)

    # -- shape -------------------------------------------------------------

    def reshape(self, *shape: Any) -> Tensor:
        from gradlab.ops.shape import Reshape

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, shape=tuple(int(s) for s in shape))

    def flatten(self, start_dim: int = 0) -> Tensor:
        from gradlab.ops.shape import Reshape

        lead = self.shape[:start_dim]
        rest = int(np.prod(self.shape[start_dim:])) if self.shape[start_dim:] else 1
        return Reshape.apply(self, shape=(*lead, rest))

    def transpose(self, *axes: int) -> Tensor:
        from gradlab.ops.shape import Transpose

        return Transpose.apply(self, axes=tuple(axes) if axes else None)

    @property
    def T(self) -> Tensor:
        """Reverse the axes (the usual matrix transpose for a 2-D tensor)."""
        return self.transpose()

    def __getitem__(self, index: Any) -> Tensor:
        from gradlab.ops.shape import Index

        return Index.apply(self, index=index)

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    # -- maths -------------------------------------------------------------

    def exp(self) -> Tensor:
        from gradlab.ops.activation import Exp

        return Exp.apply(self)

    def log(self) -> Tensor:
        from gradlab.ops.activation import Log

        return Log.apply(self)

    def sqrt(self) -> Tensor:
        from gradlab.ops.activation import Sqrt

        return Sqrt.apply(self)

    def tanh(self) -> Tensor:
        from gradlab.ops.activation import Tanh

        return Tanh.apply(self)

    def sigmoid(self) -> Tensor:
        from gradlab.ops.activation import Sigmoid

        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        from gradlab.ops.activation import ReLU

        return ReLU.apply(self)

    def sin(self) -> Tensor:
        from gradlab.ops.activation import Sin

        return Sin.apply(self)

    def cos(self) -> Tensor:
        from gradlab.ops.activation import Cos

        return Cos.apply(self)

    # -- reductions --------------------------------------------------------

    def sum(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from gradlab.ops.reduction import Sum

        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from gradlab.ops.reduction import Mean

        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def max(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from gradlab.ops.reduction import Max

        return Max.apply(self, axis=axis, keepdims=keepdims)

    def min(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from gradlab.ops.reduction import Min

        return Min.apply(self, axis=axis, keepdims=keepdims)

    # -- comparisons (not differentiable; results are boolean data) --------

    def _compare(self, other: Any, fn) -> Tensor:
        other_data = other._data if isinstance(other, Tensor) else np.asarray(other)
        return Tensor(fn(self._data, other_data), copy=False, _internal=True)

    def __lt__(self, other: Any) -> Tensor:
        return self._compare(other, np.less)

    def __le__(self, other: Any) -> Tensor:
        return self._compare(other, np.less_equal)

    def __gt__(self, other: Any) -> Tensor:
        return self._compare(other, np.greater)

    def __ge__(self, other: Any) -> Tensor:
        return self._compare(other, np.greater_equal)

    def eq(self, other: Any) -> Tensor:
        """Elementwise equality as a boolean tensor (``==`` stays identity)."""
        return self._compare(other, np.equal)

    def ne(self, other: Any) -> Tensor:
        return self._compare(other, np.not_equal)

    __hash__ = object.__hash__

    # -- display -----------------------------------------------------------

    def __repr__(self) -> str:
        body = np.array2string(self._data, precision=4, suppress_small=True, threshold=24)
        bits = [f"Tensor({body}"]
        if self.dtype != np.dtype(DEFAULT_DTYPE):
            bits.append(f"dtype={self.dtype}")
        if self._requires_grad:
            bits.append("requires_grad=True")
        if self._grad_fn is not None:
            bits.append(f"grad_fn=<{self._grad_fn.op}>")
        return ", ".join(bits) + ")"

    def __bool__(self) -> bool:
        if self.size != 1:
            raise ValueError(
                f"The truth value of a tensor with {self.size} elements is ambiguous; "
                "use .numpy().any() or .numpy().all()."
            )
        return bool(self._data.reshape(()).item())

    def __float__(self) -> float:
        return float(self.item())


def as_tensor(value: ArrayLike, dtype: Any = None) -> Tensor:
    """Return ``value`` as a ``Tensor``, without copying when it already is one."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value, dtype=dtype)


def tensor(data: ArrayLike, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Construct a tensor. Alias of ``Tensor(...)`` reading like a factory."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)
