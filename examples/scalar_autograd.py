"""Experiment 0 — what the autograd engine actually does, one scalar at a time.

Run: ``python examples/scalar_autograd.py``

This is the smallest complete demonstration: a function of one variable, the
graph GradLab builds for it, the reverse walk, and the chain-rule arithmetic at
each step — all read back out of the real engine rather than narrated.
"""

from __future__ import annotations

import math

import gradlab as gl
from gradlab.autograd import topological_order
from gradlab.inspection import trace_expression

print(__doc__.strip())
print("\n" + "=" * 72)
print("f(x) = x^2 + 2x   at   x = 3")
print("=" * 72)

x = gl.Tensor(3.0, requires_grad=True)
y = x * x + 2 * x
y.backward()

print(f"\nforward value   f(3)  = {y.item()}          (expected 3^2 + 2*3 = 15)")
print(f"analytic deriv  f'(x) = 2x + 2  ->  f'(3) = {2 * 3 + 2}")
print(f"GradLab gives   x.grad = {x.grad.item()}")

print("\n--- the graph GradLab built (reverse topological order) ---")
for i, node in enumerate(topological_order(y.grad_fn), 1):
    shapes = ", ".join(str(t.shape) for t in node.inputs)
    print(f"  {i}. {node.op:<6} inputs=({shapes})  ->  {node.output_shape}")
    print(f"     rule: {node.backward_rule}")

print("\n--- the same expression, stepped through by the inspection API ---")
trace = trace_expression(lambda t: t * t + 2 * t, 3.0, names=["x"], label="x^2 + 2x")
print(f"  seed: dL/dL = {trace['seed']['value']['values']}")
for step in trace["backward_steps"]:
    upstream = step["upstream_gradient"]["values"]
    print(
        print(
            f"  step {step['step']}: {step['op']:<4} "
            f"upstream={upstream}   rule: {step['backward_rule']}"
        )
    )
    for c in step["contributions"]:
        if c["received_gradient"]:
            print(f"        -> {c['name']:<6} receives {c['gradient']['values']}")
        else:
            print(f"        -> {c['name']:<6} {c['why_none']}")

leaf = trace["leaf_gradients"][0]
print(f"\n  accumulated gradient on the leaf {leaf['name']}: {leaf['gradient']['values']}")
print(f"  graph: {trace['num_nodes']} nodes, depth {trace['depth']}")

print("\n" + "=" * 72)
print("Why the two branches both matter")
print("=" * 72)
print("""
  x appears THREE times in the graph: twice inside x*x and once inside 2*x.
  Reverse-mode visits every consumer of x before x itself, and *adds* each
  contribution:

      from x*x :  d(x^2)/dx = 2x = 6
      from 2*x :  d(2x)/dx  = 2
      total    :              8

  That addition is the whole reason a shared value needs gradient accumulation
  rather than assignment. Setting instead of adding would give 2 or 6.
""")

print("=" * 72)
print("A chain three deep:  g(x) = tanh(exp(x) * 2)")
print("=" * 72)
x2 = gl.Tensor(0.4, requires_grad=True)
inner = x2.exp() * 2
out = inner.tanh()
inner.retain_grad()
out.backward()
e = math.exp(0.4)
expected = (1 - math.tanh(2 * e) ** 2) * 2 * e
print(f"  g(0.4)            = {out.item():.12f}")
print(f"  d/d(inner)        = {inner.grad.item():.12f}   (1 - tanh^2 at inner)")
print(f"  d/dx  GradLab     = {x2.grad.item():.12f}")
print(f"  d/dx  by hand     = {expected:.12f}")
print(f"  agreement         = {abs(x2.grad.item() - expected):.2e}")

print("\n" + "=" * 72)
print("And a check against finite differences, which knows nothing about the graph")
print("=" * 72)
z = gl.Tensor(0.4, requires_grad=True)
res = gl.gradcheck(lambda t: (t.exp() * 2).tanh(), z)
print(res.report())
