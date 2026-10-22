"""`Module` and `Parameter` — the container abstraction for learnable models.

A ``Module`` is an object that owns parameters and child modules and knows how
to run a forward pass. The only real machinery here is attribute interception:
assigning a ``Parameter`` or a ``Module`` to ``self`` registers it, which is
what makes ``model.parameters()`` work recursively without the user keeping a
list by hand.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from gradlab._config import no_grad
from gradlab._errors import GradLabError, ShapeError
from gradlab.tensor import Tensor


class Parameter(Tensor):
    """A ``Tensor`` that a module owns and an optimiser is expected to update.

    Functionally it is a leaf tensor with ``requires_grad=True``; the distinct
    type is what lets :meth:`Module.parameters` find it.
    """

    def __init__(self, data: Any, requires_grad: bool = True) -> None:
        if isinstance(data, Tensor):
            super().__init__(data.data, requires_grad=requires_grad, copy=False, _internal=True)
            self._requires_grad = requires_grad
        else:
            super().__init__(data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()


class Module:
    """Base class for every layer, loss, and composite model.

    Subclasses assign parameters and child modules in ``__init__`` and implement
    ``forward``. Calling the module runs ``forward``.

        class Affine(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.w = nn.Parameter(gl.ones(n))
                self.b = nn.Parameter(gl.zeros(n))

            def forward(self, x):
                return x * self.w + self.b
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    # -- registration ------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._require_init()
            self._parameters[name] = value
            self.__dict__.pop(name, None)
        elif isinstance(value, Module):
            self._require_init()
            self._modules[name] = value
            self.__dict__.pop(name, None)
        else:
            if name in getattr(self, "_parameters", {}):
                del self._parameters[name]
            if name in getattr(self, "_modules", {}):
                del self._modules[name]
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        for store in ("_parameters", "_modules", "_buffers"):
            container = self.__dict__.get(store)
            if container is not None and name in container:
                return container[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _require_init(self) -> None:
        if "_parameters" not in self.__dict__:
            raise GradLabError(
                f"{type(self).__name__} assigned a Parameter or Module before calling "
                "super().__init__(). Add super().__init__() as the first line of __init__."
            )

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """Register non-learnable state that still belongs in ``state_dict``."""
        self._require_init()
        self._buffers[name] = tensor

    # -- discovery ---------------------------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Every parameter in this module and its descendants, depth-first."""
        for name, param in self._parameters.items():
            yield (f"{prefix}{name}", param)
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix=f"{prefix}{name}.")

    def parameters(self) -> list[Parameter]:
        """Every parameter, in a stable order suitable for an optimiser."""
        return [p for _, p in self.named_parameters()]

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        yield (prefix.rstrip("."), self)
        for name, module in self._modules.items():
            yield from module.named_modules(prefix=f"{prefix}{name}.")

    def modules(self) -> list[Module]:
        return [m for _, m in self.named_modules()]

    def children(self) -> list[Module]:
        return list(self._modules.values())

    def num_parameters(self) -> int:
        """Total scalar count across all parameters — the model's size."""
        return sum(p.size for p in self.parameters())

    # -- modes -------------------------------------------------------------

    def train(self, mode: bool = True) -> Module:
        """Set training mode on this module and every descendant."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        """Set evaluation mode (e.g. Dropout becomes the identity)."""
        return self.train(False)

    def zero_grad(self) -> None:
        """Clear ``.grad`` on every parameter."""
        for param in self.parameters():
            param.zero_grad()

    # -- serialisation -----------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        """A flat ``name -> array`` snapshot of all parameters and buffers.

        Arrays are copies, so a state dict is a genuine checkpoint and not a
        live view that keeps changing as training continues.
        """
        state = {name: param.data.copy() for name, param in self.named_parameters()}
        for prefix, module in self.named_modules():
            for name, buf in module._buffers.items():
                key = f"{prefix}.{name}" if prefix else name
                state[key] = buf.data.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        """Restore parameters from a :meth:`state_dict`.

        Validates names and shapes before writing anything, so a mismatched
        checkpoint leaves the model untouched rather than half-loaded.
        """
        own = dict(self.named_parameters())
        for prefix, module in self.named_modules():
            for name, buf in module._buffers.items():
                own[f"{prefix}.{name}" if prefix else name] = buf

        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if strict and (missing or unexpected):
            raise GradLabError(
                f"state_dict does not match this model. Missing: {missing}. "
                f"Unexpected: {unexpected}."
            )
        for key, target in own.items():
            if key not in state:
                continue
            incoming = np.asarray(state[key])
            if incoming.shape != target.shape:
                raise ShapeError(
                    f"state_dict[{key!r}] has shape {incoming.shape} but the model's "
                    f"parameter has shape {target.shape}."
                )
        with no_grad():
            for key, target in own.items():
                if key in state:
                    target.data = np.asarray(state[key])

    # -- calling -----------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement forward().")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    # -- display -----------------------------------------------------------

    def extra_repr(self) -> str:
        """Subclass hook: a short description of the layer's configuration."""
        return ""

    def __repr__(self) -> str:
        head = f"{type(self).__name__}({self.extra_repr()})"
        if not self._modules:
            return head
        lines = [f"{type(self).__name__}("]
        for name, module in self._modules.items():
            block = repr(module).split("\n")
            lines.append(f"  ({name}): {block[0]}")
            lines.extend(f"  {line}" for line in block[1:])
        lines.append(")")
        return "\n".join(lines)


class Sequential(Module):
    """Run a list of modules in order.

    >>> import gradlab as gl
    >>> from gradlab import nn
    >>> model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    >>> model(gl.zeros(3, 2)).shape
    (3, 1)
    """

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for i, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError(
                    f"Sequential expects Modules; item {i} is a {type(module).__name__}."
                )
            setattr(self, str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, i: int) -> Module:
        return list(self._modules.values())[i]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
