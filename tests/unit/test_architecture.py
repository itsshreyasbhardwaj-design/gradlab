"""Structural guarantees that are easy to break by accident.

The headline claim of this project is that the gradients are GradLab's own. That
claim is only credible if it is mechanically checked, so this file greps the
shipped source for any import of another autodiff framework.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "gradlab"
FORBIDDEN = {"torch", "tensorflow", "jax", "autograd", "theano", "mxnet", "chainer"}
PY_FILES = sorted(SRC.rglob("*.py"))


def _imported_roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_source_tree_is_not_empty():
    assert len(PY_FILES) > 15, "sanity check: the scan should see the whole package"


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_autodiff_framework_is_imported(path: pathlib.Path):
    """No shipped module may import another autodiff framework, at any depth.

    PyTorch is allowed to exist in the environment and is used in
    ``tests/conformance`` and ``benchmarks`` — but never inside ``src/gradlab``.
    """
    offenders = _imported_roots(path) & FORBIDDEN
    assert not offenders, (
        f"{path.relative_to(SRC)} imports {sorted(offenders)}. GradLab's gradients "
        "must be its own; other frameworks are permitted only as test oracles."
    )


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_eval_or_exec_in_shipped_code(path: pathlib.Path):
    """The visualiser's backend must never execute user-supplied Python."""
    tree = ast.parse(path.read_text(), filename=str(path))
    bad = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"eval", "exec", "compile"}
    ]
    assert not bad, f"{path.relative_to(SRC)} calls {bad}; see docs/limitations.md on sandboxing"


def test_dependencies_are_only_numpy():
    """The runtime dependency surface stays tiny, by policy."""
    import tomllib

    pyproject = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    meta = tomllib.loads(pyproject.read_text())
    deps = meta["project"]["dependencies"]
    assert [d.split(">")[0].split("=")[0].strip() for d in deps] == ["numpy"]
