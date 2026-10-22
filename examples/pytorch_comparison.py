"""Experiment 5 — GradLab against PyTorch, same model, same data, same seed.

Run: ``python examples/pytorch_comparison.py``   (needs the ``oracle`` extra)

PyTorch is used here only as an independent reference implementation. Nothing in
``src/gradlab`` imports it — ``tests/unit/test_architecture.py`` checks that
mechanically — so agreement below is agreement between two implementations that
share no code.

The objective is **correctness and transparency, not speed**. GradLab is a
single-threaded NumPy framework with a Python-level graph; PyTorch is a compiled
tensor library with a C++ autograd engine. Where timings appear, they are
reported with that difference stated, not as a competitive claim.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import gradlab as gl
from gradlab import data, nn, optim

from _report import save_artifact

try:
    import torch

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
except ImportError:
    print("PyTorch is not installed. Install the oracle extra to run this comparison:")
    print("    uv pip install 'gradlab[oracle]'    (or: pip install torch)")
    print("\nThis script is the only part of GradLab that needs PyTorch, and the")
    print("framework itself never imports it.")
    sys.exit(0)

print(__doc__.strip())

SEED = 1234
EPOCHS = 120
LR = 0.02
TOL = 1e-10


def build_pair(in_f=2, hidden=16, out_f=1):
    """One architecture, built twice, with PyTorch given GradLab's exact weights."""
    gl_model = nn.Sequential(
        nn.Linear(in_f, hidden, seed=SEED),
        nn.Tanh(),
        nn.Linear(hidden, hidden, seed=SEED + 1),
        nn.Tanh(),
        nn.Linear(hidden, out_f, seed=SEED + 2),
    )
    tt_model = torch.nn.Sequential(
        torch.nn.Linear(in_f, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, out_f),
    )
    # GradLab stores (in, out) and computes x @ W; torch stores (out, in) and
    # computes x @ W.T. Hence the transpose on every weight copy.
    with torch.no_grad():
        for gl_idx, tt_idx in ((0, 0), (2, 2), (4, 4)):
            tt_model[tt_idx].weight.copy_(torch.tensor(gl_model[gl_idx].weight.data.T.copy()))
            tt_model[tt_idx].bias.copy_(torch.tensor(gl_model[gl_idx].bias.data.copy()))
    return gl_model, tt_model


X, y = data.make_moons(n_samples=512, noise=0.2, seed=SEED)
xb, yb = X.data, y.data
gl_model, tt_model = build_pair()

print(f"\n{'=' * 72}\n1. Identical initial parameters\n{'=' * 72}")
for i, (gl_idx, tt_idx) in enumerate(((0, 0), (2, 2), (4, 4))):
    dw = np.abs(gl_model[gl_idx].weight.data - tt_model[tt_idx].weight.detach().numpy().T).max()
    db = np.abs(gl_model[gl_idx].bias.data - tt_model[tt_idx].bias.detach().numpy()).max()
    print(f"  layer {i}: max |Δweight| = {dw:.3e}   max |Δbias| = {db:.3e}")

print(f"\n{'=' * 72}\n2. Forward pass\n{'=' * 72}")
gl_out = gl_model(gl.Tensor(xb.copy()))
tt_out = tt_model(torch.tensor(xb.copy()))
forward_err = float(np.abs(gl_out.data - tt_out.detach().numpy()).max())
print(f"  max |Δ output| over {xb.shape[0]} samples = {forward_err:.3e}")

print(f"\n{'=' * 72}\n3. Loss and parameter gradients (one backward pass)\n{'=' * 72}")
gl_loss = nn.binary_cross_entropy_with_logits(gl_out, gl.Tensor(yb.copy()))
tt_loss = torch.nn.functional.binary_cross_entropy_with_logits(tt_out, torch.tensor(yb.copy()))
gl_loss.backward()
tt_loss.backward()
print(f"  loss: GradLab {gl_loss.item():.15f}")
print(f"        PyTorch {tt_loss.item():.15f}")
print(f"        |Δ| = {abs(gl_loss.item() - tt_loss.item()):.3e}")
grad_errors = []
for i, (gl_idx, tt_idx) in enumerate(((0, 0), (2, 2), (4, 4))):
    dw = np.abs(gl_model[gl_idx].weight.grad.data - tt_model[tt_idx].weight.grad.numpy().T).max()
    db = np.abs(gl_model[gl_idx].bias.grad.data - tt_model[tt_idx].bias.grad.numpy()).max()
    grad_errors += [float(dw), float(db)]
    print(f"  layer {i}: max |Δ dW| = {dw:.3e}   max |Δ db| = {db:.3e}")

print(f"\n{'=' * 72}\n4. {EPOCHS} full training epochs, Adam(lr={LR})\n{'=' * 72}")
gl_model, tt_model = build_pair()
gl_opt = optim.Adam(gl_model.parameters(), lr=LR)
tt_opt = torch.optim.Adam(tt_model.parameters(), lr=LR)

gl_curve, tt_curve = [], []
gl_time = tt_time = 0.0
for _epoch in range(EPOCHS):
    t0 = time.perf_counter()
    gl_opt.zero_grad()
    loss = nn.binary_cross_entropy_with_logits(gl_model(gl.Tensor(xb.copy())), gl.Tensor(yb.copy()))
    loss.backward()
    gl_opt.step()
    gl_time += time.perf_counter() - t0
    gl_curve.append(loss.item())

    t0 = time.perf_counter()
    tt_opt.zero_grad()
    t_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        tt_model(torch.tensor(xb.copy())), torch.tensor(yb.copy())
    )
    t_loss.backward()
    tt_opt.step()
    tt_time += time.perf_counter() - t0
    tt_curve.append(t_loss.item())

curve_err = float(np.abs(np.array(gl_curve) - np.array(tt_curve)).max())
final_err = abs(gl_curve[-1] - tt_curve[-1])
print(f"  epoch   1: GradLab {gl_curve[0]:.12f} | PyTorch {tt_curve[0]:.12f}")
print(
    f"  epoch  {EPOCHS // 2:>3}: GradLab {gl_curve[EPOCHS // 2]:.12f} | "
    f"PyTorch {tt_curve[EPOCHS // 2]:.12f}"
)
print(f"  epoch {EPOCHS:>3}: GradLab {gl_curve[-1]:.12f} | PyTorch {tt_curve[-1]:.12f}")
print(f"\n  max |Δ loss| across all {EPOCHS} epochs = {curve_err:.3e}")
print(f"  final |Δ loss|                      = {final_err:.3e}")

param_err = max(
    float(np.abs(gl_model[g].weight.data - tt_model[t].weight.detach().numpy().T).max())
    for g, t in ((0, 0), (2, 2), (4, 4))
)
print(f"  max |Δ parameter| after {EPOCHS} steps  = {param_err:.3e}")

print(f"\n{'=' * 72}\n5. Where the remaining difference comes from\n{'=' * 72}")
print(f"""
  Agreement is at the {curve_err:.0e} level, not exactly zero, and that is expected.
  Both frameworks compute the same quantities in float64, but not in the same
  ORDER: NumPy's matmul dispatches to a BLAS whose blocking and accumulation
  order differ from ATen's. Floating-point addition is not associative, so a
  different summation order gives a different last bit. Those last bits then
  compound slightly over {EPOCHS} optimiser steps.

  The size of the disagreement is the test. Anything at or below roughly 1e-12
  after a full training run is accumulated rounding and nothing more. A
  disagreement at 1e-6 or larger would indicate a genuinely different
  derivative, not arithmetic.

  Not compared here: speed as a claim. See docs/benchmarks.md, which measures
  timings properly and explains why a Python-level graph cannot compete with a
  C++ engine — and why that was never the objective.""")

print(
    f"\n  (wall clock for the loop above — context, not a benchmark: "
    f"GradLab {gl_time:.2f}s, PyTorch {tt_time:.2f}s)"
)

verdict = "PASS" if curve_err < 1e-6 and forward_err < TOL else "INVESTIGATE"
print(f"\n{'=' * 72}\nVerdict: {verdict}\n{'=' * 72}")

path = save_artifact(
    "pytorch_comparison",
    {
        "experiment": "pytorch_comparison",
        "seed": SEED,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "gradlab_version": gl.__version__,
        "dtype": "float64",
        "epochs": EPOCHS,
        "lr": LR,
        "forward_max_abs_error": forward_err,
        "loss_abs_error": abs(gl_loss.item() - tt_loss.item()),
        "gradient_max_abs_errors": grad_errors,
        "curve_max_abs_error": curve_err,
        "final_loss_abs_error": final_err,
        "parameter_max_abs_error": param_err,
        "gradlab_curve": gl_curve,
        "pytorch_curve": tt_curve,
        "wall_clock_seconds": {"gradlab": gl_time, "pytorch": tt_time},
        "verdict": verdict,
    },
)
print(f"\nartifact written: {path}")
