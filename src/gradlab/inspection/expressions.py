"""A restricted expression engine for the Gradient Microscope.

The Microscope lets a user type an expression and watch GradLab differentiate
it. That is a code-execution surface, so this module is written as one.

**Nothing here evaluates Python.** There is no ``eval``, no ``exec``, no
``compile``, and no ``ast`` round-trip — ``tests/unit/test_architecture.py``
checks the whole package for those calls. Instead:

1. :func:`tokenize` splits the input into a fixed set of tokens. Any character
   outside that set is rejected by position.
2. :func:`parse_expression` is a hand-written recursive-descent parser that
   produces a plain-data tree. It is the only thing that reads the text, and the
   tree it produces can hold nothing but numbers, variable names, and operation
   names drawn from :data:`ALLOWED_OPS`.
3. :func:`build_expression` walks that tree and calls the corresponding GradLab
   operation. A name not in the allowlist cannot be reached, because the parser
   would have refused it.

Every stage is bounded: input length, token count, tree depth, node count,
variable count, and tensor size. The limits exist so a hostile or careless
submission cannot exhaust memory or CPU on a host serving the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gradlab._errors import GradLabError
from gradlab.tensor import Tensor

#: Maximum characters in a submitted expression.
MAX_EXPRESSION_LENGTH = 400
#: Maximum nodes in the parsed tree.
MAX_NODES = 120
#: Maximum nesting depth.
MAX_DEPTH = 24
#: Maximum distinct variables.
MAX_VARIABLES = 8
#: Maximum elements in any single input tensor.
MAX_TENSOR_ELEMENTS = 4096

#: The functions a submitted expression may name. Each maps to a GradLab tensor
#: method. This allowlist is the security boundary: a name absent here is a parse
#: error, so no other attribute of a Tensor is reachable from user input.
ALLOWED_OPS: dict[str, str] = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "tanh": "tanh",
    "sigmoid": "sigmoid",
    "relu": "relu",
    "sin": "sin",
    "cos": "cos",
    "abs": "__abs__",
    "sum": "sum",
    "mean": "mean",
    "max": "max",
    "min": "min",
}

_BINARY = {"+", "-", "*", "/", "^", "@"}


class ExpressionError(GradLabError, ValueError):
    """A submitted expression was rejected. The message says exactly why."""


@dataclass(frozen=True)
class Token:
    kind: str  # "number" | "name" | "op" | "lparen" | "rparen" | "comma"
    text: str
    position: int


def tokenize(text: str) -> list[Token]:
    """Split ``text`` into tokens, rejecting anything outside the grammar."""
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"Expression is {len(text)} characters; the limit is {MAX_EXPRESSION_LENGTH}."
        )
    tokens: list[Token] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < len(text) and text[i + 1].isdigit()):
            j = i
            seen_dot = False
            while j < len(text) and (text[j].isdigit() or (text[j] == "." and not seen_dot)):
                seen_dot = seen_dot or text[j] == "."
                j += 1
            if j < len(text) and text[j] in "eE":
                k = j + 1
                if k < len(text) and text[k] in "+-":
                    k += 1
                if k < len(text) and text[k].isdigit():
                    while k < len(text) and text[k].isdigit():
                        k += 1
                    j = k
            tokens.append(Token("number", text[i:j], i))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(Token("name", text[i:j], i))
            i = j
            continue
        if ch in _BINARY:
            tokens.append(Token("op", ch, i))
            i += 1
            continue
        if ch == "(":
            tokens.append(Token("lparen", ch, i))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token("rparen", ch, i))
            i += 1
            continue
        if ch == ",":
            tokens.append(Token("comma", ch, i))
            i += 1
            continue
        raise ExpressionError(
            f"Character {ch!r} at position {i} is not allowed in an expression. "
            "Permitted: numbers, names, + - * / ^ @ ( ) and commas."
        )
    if len(tokens) > MAX_NODES * 3:
        raise ExpressionError(f"Expression has too many tokens ({len(tokens)}).")
    return tokens


class _Parser:
    """Recursive descent over the token list. Produces plain data only.

    Grammar (standard precedence, ``^`` right-associative)::

        expr   := term (('+' | '-') term)*
        term   := power (('*' | '/' | '@') power)*
        power  := unary ('^' power)?
        unary  := '-' unary | atom
        atom   := number | name | name '(' expr ')' | '(' expr ')'
    """

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.depth = 0
        self.nodes = 0

    def peek(self) -> Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> Token:
        token = self.peek()
        if token is None:
            raise ExpressionError("Expression ended unexpectedly.")
        self.pos += 1
        return token

    def _node(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ExpressionError(f"Expression has more than {MAX_NODES} operations.")
        return payload

    def parse(self) -> dict[str, Any]:
        tree = self.expr()
        remaining = self.peek()
        if remaining is not None:
            raise ExpressionError(
                f"Unexpected {remaining.text!r} at position {remaining.position}."
            )
        return tree

    def expr(self) -> dict[str, Any]:
        node = self.term()
        while (token := self.peek()) and token.kind == "op" and token.text in "+-":
            self.take()
            node = self._node(
                {"op": "add" if token.text == "+" else "sub", "args": [node, self.term()]}
            )
        return node

    def term(self) -> dict[str, Any]:
        node = self.power()
        while (token := self.peek()) and token.kind == "op" and token.text in "*/@":
            self.take()
            op = {"*": "mul", "/": "div", "@": "matmul"}[token.text]
            node = self._node({"op": op, "args": [node, self.power()]})
        return node

    def power(self) -> dict[str, Any]:
        node = self.unary()
        token = self.peek()
        if token and token.kind == "op" and token.text == "^":
            self.take()
            return self._node({"op": "pow", "args": [node, self.power()]})
        return node

    def unary(self) -> dict[str, Any]:
        token = self.peek()
        if token and token.kind == "op" and token.text == "-":
            self.take()
            return self._node({"op": "neg", "args": [self.unary()]})
        return self.atom()

    def atom(self) -> dict[str, Any]:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExpressionError(f"Expression nests deeper than {MAX_DEPTH} levels.")
        try:
            token = self.take()
            if token.kind == "number":
                return self._node({"const": float(token.text)})
            if token.kind == "lparen":
                inner = self.expr()
                closing = self.take()
                if closing.kind != "rparen":
                    raise ExpressionError(f"Expected ')' at position {closing.position}.")
                return inner
            if token.kind == "name":
                following = self.peek()
                if following and following.kind == "lparen":
                    if token.text not in ALLOWED_OPS:
                        raise ExpressionError(
                            f"Unknown function {token.text!r} at position {token.position}. "
                            f"Allowed: {', '.join(sorted(ALLOWED_OPS))}."
                        )
                    self.take()
                    args = [self.expr()]
                    while (nxt := self.peek()) and nxt.kind == "comma":
                        self.take()
                        args.append(self.expr())
                    closing = self.take()
                    if closing.kind != "rparen":
                        raise ExpressionError(f"Expected ')' at position {closing.position}.")
                    if len(args) != 1:
                        raise ExpressionError(
                            f"{token.text}() takes exactly one argument; got {len(args)}."
                        )
                    return self._node({"op": token.text, "args": args})
                if token.text in ALLOWED_OPS:
                    raise ExpressionError(
                        f"{token.text!r} is a function and needs parentheses, e.g. {token.text}(x)."
                    )
                return self._node({"var": token.text})
            raise ExpressionError(f"Unexpected {token.text!r} at position {token.position}.")
        finally:
            self.depth -= 1


def parse_expression(text: str) -> dict[str, Any]:
    """Parse ``text`` into a validated plain-data tree.

    >>> tree = parse_expression("x^2 + 2*x")
    >>> tree["op"]
    'add'
    >>> tree["args"][0]
    {'op': 'pow', 'args': [{'var': 'x'}, {'const': 2.0}]}
    """
    if not text or not text.strip():
        raise ExpressionError("Expression is empty.")
    return _Parser(tokenize(text)).parse()


def free_variables(tree: dict[str, Any]) -> list[str]:
    """Variable names the tree references, in first-appearance order."""
    found: list[str] = []

    def walk(node):
        if "var" in node and node["var"] not in found:
            found.append(node["var"])
        for child in node.get("args", ()):
            walk(child)

    walk(tree)
    return found


def validate_variables(values: dict[str, Any]) -> dict[str, Tensor]:
    """Coerce submitted variable values to tensors, enforcing the size limits."""
    if len(values) > MAX_VARIABLES:
        raise ExpressionError(f"At most {MAX_VARIABLES} variables are allowed.")
    out: dict[str, Tensor] = {}
    for name, value in values.items():
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ExpressionError(f"Variable {name!r} is not numeric: {exc}") from None
        if arr.size > MAX_TENSOR_ELEMENTS:
            raise ExpressionError(
                f"Variable {name!r} has {arr.size} elements; the limit is {MAX_TENSOR_ELEMENTS}."
            )
        if arr.ndim > 3:
            raise ExpressionError(f"Variable {name!r} has {arr.ndim} dimensions; at most 3.")
        if not np.all(np.isfinite(arr)):
            raise ExpressionError(f"Variable {name!r} contains non-finite values.")
        out[name] = Tensor(arr, requires_grad=True)
    return out


def build_expression(tree: dict[str, Any], variables: dict[str, Tensor]) -> Tensor:
    """Evaluate a parsed tree with GradLab operations, recording a real graph.

    The returned tensor is an ordinary GradLab tensor with ordinary gradient
    history, so it can be differentiated and traced like anything else.
    """
    if "const" in tree:
        return Tensor(float(tree["const"]))
    if "var" in tree:
        name = tree["var"]
        if name not in variables:
            raise ExpressionError(f"Variable {name!r} has no value. Provide one, or use a number.")
        return variables[name]

    op = tree.get("op")

    # A constant exponent routes to the scalar Pow op rather than PowTensor: the
    # exponent is then genuinely a constant, the graph gains no spurious leaf,
    # and the displayed rule is the familiar n*a^(n-1) rather than the general
    # two-argument form.
    if op == "pow":
        children = tree.get("args", ())
        if len(children) == 2 and "const" in children[1]:
            return build_expression(children[0], variables) ** float(children[1]["const"])

    args = [build_expression(child, variables) for child in tree.get("args", ())]

    if op in ("add", "sub", "mul", "div", "matmul", "pow"):
        if len(args) != 2:
            raise ExpressionError(f"{op} needs two operands; got {len(args)}.")
        a, b = args
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "div":
            return a / b
        if op == "matmul":
            return a @ b
        return a**b
    if op == "neg":
        return -args[0]
    if op in ALLOWED_OPS:
        return getattr(args[0], ALLOWED_OPS[op])()
    raise ExpressionError(f"Operation {op!r} is not allowed.")


def evaluate(text: str, values: dict[str, Any] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    """Parse, validate and evaluate ``text``. Returns ``(output, variables)``.

    Any variable the expression names but ``values`` omits defaults to 1.0, so a
    bare ``x^2 + 2*x`` is usable without ceremony.
    """
    tree = parse_expression(text)
    names = free_variables(tree)
    supplied = dict(values or {})
    for name in names:
        supplied.setdefault(name, 1.0)
    unused = sorted(set(supplied) - set(names))
    if unused:
        raise ExpressionError(f"Values given for variables the expression never uses: {unused}.")
    variables = validate_variables({name: supplied[name] for name in names})
    return build_expression(tree, variables), variables
