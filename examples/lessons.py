"""The fourteen guided lessons, as runnable code.

Run: ``python examples/lessons.py``           (all lessons)
     ``python examples/lessons.py 7``          (one lesson)
     ``python examples/lessons.py --list``     (titles only)

Every number printed is computed by GradLab as the lesson runs. ``docs/lessons.md``
is the written companion and quotes this script's actual output.
"""

from __future__ import annotations

import sys

import numpy as np

import gradlab as gl
from gradlab import data, nn, optim, training
from gradlab.inspection import graph_summary, trace_expression

LESSONS = []


def lesson(number, title):
    def register(fn):
        LESSONS.append((number, title, fn))
        return fn

    return register


def rule(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def experiment(text):
    print(f"\n  >> Try it yourself: {text}")


@lesson(1, "What is a tensor?")
def lesson_1():
    print("""
  A tensor is an n-dimensional array that remembers where it came from.
  The array part is NumPy's. The remembering is what makes it differentiable:
  a tensor created by an operation holds a reference to that operation.""")

    scalar = gl.Tensor(3.0, requires_grad=True)
    vector = gl.Tensor([1.0, 2.0, 3.0])
    matrix = gl.Tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = gl.zeros(8, 3, 2)

    for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix), ("3-D", batch)]:
        print(f"  {name:<8} shape={t.shape!s:<12} ndim={t.ndim}  size={t.size}  dtype={t.dtype}")

    print(f"\n  a leaf:      is_leaf={scalar.is_leaf}  grad_fn={scalar.grad_fn}")
    produced = scalar * 2
    print(f"  a result:    is_leaf={produced.is_leaf}  grad_fn=<{produced.grad_fn.op}>")
    print("""
  That grad_fn is the entire difference between a tensor and an array. It is the
  edge back into the graph, and it is what backward() follows.""")
    experiment(
        "make a Tensor with dtype='int64' and requires_grad=True. "
        "Read the error message — it explains which dtypes can carry a gradient."
    )


@lesson(2, "Why neural networks need derivatives")
def lesson_2():
    print("""
  Training means choosing parameters that make a loss small. The loss is a
  function of millions of numbers, and we cannot try them all. The derivative
  answers the only question we can afford to ask: 'if I nudge this parameter,
  does the loss go up or down, and how fast?'""")

    w = gl.Tensor(0.0, requires_grad=True)
    target = 3.0
    print(f"\n  minimising (w - {target})^2 by following the derivative:\n")
    print(f"  {'step':>4}  {'w':>10}  {'loss':>12}  {'dL/dw':>10}")
    for step in range(6):
        w.zero_grad()
        loss = (w - target) ** 2
        loss.backward()
        print(f"  {step:>4}  {w.item():>10.6f}  {loss.item():>12.8f}  {w.grad.item():>10.6f}")
        with gl.no_grad():
            w.data = w.data - 0.3 * w.grad.data

    print("""
  The gradient is 2(w - 3). It is negative while w < 3, so stepping *against* it
  increases w. It shrinks as w approaches 3, so the steps get smaller on their
  own. That is gradient descent, complete, in one parameter.""")
    experiment(
        "change the step size from 0.3 to 1.1 and watch it oscillate away. "
        "The threshold for this loss is exactly 1.0 — see lesson 12."
    )


@lesson(3, "The chain rule")
def lesson_3():
    print("""
  If y depends on u and u depends on x, then

      dy/dx = dy/du · du/dx

  A deep network is this rule applied a few hundred times. Nothing else.""")

    x = gl.Tensor(0.5, requires_grad=True)
    u = x * 3
    u.retain_grad()
    y = u.tanh()
    y.backward()

    du_dx = 3.0
    dy_du = 1 - np.tanh(1.5) ** 2
    print(f"""
  y = tanh(3x)  at  x = 0.5

    u = 3x            = {u.item():.10f}
    y = tanh(u)       = {y.item():.10f}

    dy/du = 1-tanh²(u) = {dy_du:.10f}    <- GradLab: {u.grad.item():.10f}
    du/dx              = {du_dx:.10f}
    dy/dx = product    = {dy_du * du_dx:.10f}    <- GradLab: {x.grad.item():.10f}""")
    print(f"""
  Each backward step multiplies by exactly one local derivative. Notice that
  dy/du is {dy_du:.4f} — smaller than 1. Chain enough of those together and the
  product goes to zero. That is the vanishing gradient problem, and it is
  nothing more than repeated multiplication by numbers below 1.""")
    experiment(
        "stack four tanh layers (y = tanh(tanh(tanh(tanh(3x))))) and compare "
        "dy/dx to the single-layer case."
    )


@lesson(4, "Computational graphs")
def lesson_4():
    print("""
  GradLab records every operation as it runs. Nodes are operations; edges are
  tensors. The graph is built during the forward pass, not compiled in advance.""")

    x = gl.Tensor(2.0, requires_grad=True)
    y = (x * x + 3 * x).tanh()
    print(f"\n{graph_summary(y)}")

    trace = trace_expression(lambda t: (t * t + 3 * t).tanh(), 2.0, names=["x"])
    print("\n  forward order (the order operations actually ran):")
    for step in trace["forward_steps"]:
        print(
            f"    {step['step']}. {step['expression']:<22} -> {step['output']['shape'] or 'scalar'}"
        )

    print("""
  Why acyclic: a node can only reference tensors that already existed when it
  was built. Time only runs one way, so there is no cycle to detect.

  Why that matters: an acyclic graph has a topological order, and a topological
  order is what lets the backward pass visit every node exactly once.""")
    experiment(
        "open this graph in the Gradient Microscope: "
        "gradlab-microscope serve, then pick a preset and use the Graph tab."
    )


@lesson(5, "Forward propagation")
def lesson_5():
    print("""
  The forward pass computes the output and, at the same time, records what each
  operation will need in order to be differentiated later.""")

    x = gl.Tensor([[1.0, 2.0]], requires_grad=True)
    layer = nn.Linear(2, 3, seed=0)
    out = layer(x)
    hidden = out.tanh()

    print(f"\n  x        {x.data.tolist()}")
    print(f"  W        shape {layer.weight.shape}")
    print(f"  b        {layer.bias.data.round(4).tolist()}")
    print(f"  x @ W    {(x.data @ layer.weight.data).round(6).tolist()}")
    print(f"  + b      {out.data.round(6).tolist()}")
    print(f"  tanh     {hidden.data.round(6).tolist()}")
    print(f"\n  graph recorded: {graph_summary(hidden).splitlines()[0]}")
    print("""
  A 'weighted sum then a squashing function' is the whole of a layer. The
  weighted sum is a matmul; the squashing is elementwise. Stack them and you
  have a network.""")
    experiment(
        "set the activation to Identity and stack three Linear layers. "
        "Multiply the three weight matrices by hand and confirm the result "
        "equals a single layer — that is why XOR needs a nonlinearity."
    )


@lesson(6, "Reverse-mode automatic differentiation")
def lesson_6():
    print("""
  Forward mode: one pass gives d(every output)/d(one input).
  Reverse mode: one pass gives d(one output)/d(every input).

  Training has a million inputs and one output, so reverse mode is a million
  times cheaper. That asymmetry is why deep learning is computationally
  possible at all.""")

    trace = trace_expression(lambda x: x * x + 2 * x, 3.0, names=["x"])
    print(f"\n  f(x) = x² + 2x at x = 3  ->  f(3) = {trace['output']['value']['values'][0]}")
    print(f"  seed: dL/dL = {trace['seed']['value']['values'][0]}\n")
    for step in trace["backward_steps"]:
        up = step["upstream_gradient"]["values"]
        print(f"  step {step['step']}: {step['op']:<5} upstream {up}")
        for c in step["contributions"]:
            if c["received_gradient"]:
                print(f"           -> {c['name']:<5} gets {c['gradient']['values']}")
    print(
        f"\n  accumulated on x: {trace['leaf_gradients'][0]['gradient']['values']}  "
        f"(f'(x) = 2x+2 = 8)"
    )
    experiment("add a third branch (x*x*x) and predict the gradient before running it.")


@lesson(7, "Gradient accumulation")
def lesson_7():
    print("""
  When a value is used more than once, its gradient is the SUM of the
  contributions from every use. Not the last one. Not the largest.""")

    x = gl.Tensor(2.0, requires_grad=True)
    y = x * x
    z = y + y
    z.backward()
    print(f"""
  y = x·x,  z = y + y,  at x = 2

    z = 2x², so dz/dx = 4x = 8
    GradLab: {x.grad.item()}

  The two branches each contribute dz/dy = 1, giving dz/dy = 2 total,
  then dy/dx = 2x = 4, so 2 × 4 = 8.""")

    x2 = gl.Tensor(3.0, requires_grad=True)
    for i in range(1, 4):
        (x2 * x2).backward()
        print(f"  backward() call {i}: x.grad = {x2.grad.item()}  (each pass adds 6)")
    x2.zero_grad()
    print(f"  after zero_grad(): {x2.grad}")
    print("""
  Accumulation across separate backward() calls is deliberate: it is what makes
  gradient accumulation over micro-batches possible. The price is that
  forgetting zero_grad() silently sums gradients across training steps.""")
    experiment(
        "remove optimizer.zero_grad() from a training loop and watch the "
        "loss behave strangely. This is the most common beginner bug in all "
        "of deep learning."
    )


@lesson(8, "Broadcasting and gradient reduction")
def lesson_8():
    print("""
  Broadcasting lets a small tensor act as if it were large. Differentiating it
  requires undoing that — and the undo is a SUM.""")

    W = gl.Tensor(np.ones((4, 3)), requires_grad=True)
    b = gl.Tensor(np.zeros(3), requires_grad=True)
    (W + b).sum().backward()
    print(f"""
  W is (4, 3), b is (3,).  W + b broadcasts b across all 4 rows.

    W.grad  shape {W.grad.shape}, all ones
    b.grad  = {b.grad.tolist()}   <- NOT ones

  b[0] was used by all 4 rows, so its gradient is the sum of 4 contributions.
  A gradient always has the shape of the thing it differentiates.""")

    print("\n  the same thing in a real layer:")
    layer = nn.Linear(2, 3, seed=0)
    layer(gl.ones(16, 2)).sum().backward()
    print(f"    batch of 16 -> bias gradient = {layer.bias.grad.data.tolist()}")
    print("""
  This is the single most common bug in a hand-written autodiff engine. Two
  arrays having the same OUTPUT shape does not mean their gradients have that
  shape.""")
    experiment(
        "construct a case where both operands broadcast: (4,1) + (1,3). "
        "Predict both gradient shapes before running it."
    )


@lesson(9, "Gradient descent")
def lesson_9():
    print("""
  theta <- theta - lr * dL/dtheta

  That is the whole algorithm. Everything else is a refinement of it.""")

    X, y, truth = data.make_regression(n_samples=200, n_features=1, bias=2.0, noise=0.1, seed=0)
    model = nn.Linear(1, 1, seed=0)
    opt = optim.SGD(model.parameters(), lr=0.1)
    print(f"\n  fitting y = wx + b;  true w={truth['weights'][0]:.4f}, b={truth['bias']}\n")
    print(f"  {'step':>5}  {'loss':>10}  {'w':>9}  {'b':>9}")
    for step in range(41):
        opt.zero_grad()
        loss = nn.mse_loss(model(X), y)
        loss.backward()
        if step % 8 == 0:
            print(
                f"  {step:>5}  {loss.item():>10.6f}  {model.weight.item():>9.5f}  "
                f"{model.bias.item():>9.5f}"
            )
        opt.step()
    print(
        f"\n  recovered w={model.weight.item():.5f} (true {truth['weights'][0]:.5f}), "
        f"b={model.bias.item():.5f} (true {truth['bias']})"
    )
    experiment("set lr=1.5 and watch it diverge; set lr=0.001 and watch it crawl.")


@lesson(10, "Backpropagation through a neural network")
def lesson_10():
    print("""
  Everything so far, applied to a real model. The loss flows backward through
  every layer, and each layer's parameters receive a gradient shaped like
  themselves.""")

    model = nn.Sequential(nn.Linear(2, 4, seed=0), nn.Tanh(), nn.Linear(4, 1, seed=1))
    X, y = data.make_moons(n_samples=8, noise=0.1, seed=0)
    loss = nn.binary_cross_entropy_with_logits(model(X), y)
    loss.backward()

    print(f"\n  loss = {loss.item():.6f}\n")
    print(f"  {'parameter':<12} {'shape':<10} {'grad shape':<12} {'||grad||':>10}")
    for name, p in model.named_parameters():
        norm = float(np.sqrt((p.grad.data**2).sum()))
        print(f"  {name:<12} {p.shape!s:<10} {p.grad.shape!s:<12} {norm:>10.6f}")
    print(f"\n  global gradient norm: {training.global_grad_norm(model.parameters()):.6f}")
    print(f"  {graph_summary(loss).splitlines()[0]}")
    print("""
  Note every gradient has exactly its parameter's shape. That invariant is worth
  internalising: if a gradient's shape is wrong, the derivation is wrong, and no
  amount of training will fix it.""")
    experiment(
        "call model.zero_grad(), run backward twice without it, and confirm "
        "every gradient has exactly doubled."
    )


@lesson(11, "Why activation functions matter")
def lesson_11():
    print("""
  Two Linear layers with nothing between them compose into ONE Linear layer:

      (xW1 + b1)W2 + b2 = x(W1W2) + (b1W2 + b2)

  So depth without nonlinearity buys exactly nothing.""")

    a = nn.Linear(3, 5, seed=0)
    b = nn.Linear(5, 2, seed=1)
    x = gl.randn(4, 3, seed=2)
    stacked = b(a(x))
    equivalent = x @ (a.weight @ b.weight) + (a.bias @ b.weight + b.bias)
    print("\n  two stacked Linear layers vs one equivalent layer:")
    print(f"    max |difference| = {np.abs(stacked.data - equivalent.data).max():.3e}")

    print("\n  so XOR is unlearnable without a nonlinearity:")
    X, y = data.make_xor(n_samples=300, seed=1)
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=32, shuffle=True, seed=1)
    for use_tanh in (False, True):
        layers = [nn.Linear(2, 8, seed=1)]
        if use_tanh:
            layers.append(nn.Tanh())
        layers.append(nn.Linear(8, 1, seed=2))
        model = nn.Sequential(*layers)
        loader.reset_epochs()
        training.fit(
            model,
            optim.Adam(model.parameters(), lr=0.05),
            nn.binary_cross_entropy_with_logits,
            loader,
            epochs=200,
        )
        with gl.no_grad():
            acc = nn.accuracy(model(X), y)
        label = "with tanh   " if use_tanh else "without tanh"
        print(f"    {label}: accuracy {acc:.4f}")
    experiment(
        "replace Tanh with ReLU and with Sigmoid. All three work here; "
        "compare how many epochs each needs."
    )


@lesson(12, "Why learning rates matter")
def lesson_12():
    print("""
  For the loss L(w) = (w - t)², one gradient-descent step is

      w <- w - lr·2(w - t)

  so the distance to the target is multiplied by |1 - 2·lr| each step. That
  gives an exact stability threshold: convergence needs lr < 1.""")

    print(f"\n  {'lr':>6}  {'|1-2·lr|':>9}  {'after 20 steps':>16}  verdict")
    for lr in (0.05, 0.3, 0.5, 0.9, 1.0, 1.1):
        w = gl.Tensor(0.0, requires_grad=True)
        for _ in range(20):
            w.zero_grad()
            ((w - 3.0) ** 2).backward()
            with gl.no_grad():
                w.data = w.data - lr * w.grad.data
        factor = abs(1 - 2 * lr)
        verdict = "converged" if factor < 1 else "oscillates forever" if factor == 1 else "diverges"
        value = w.item()
        shown = f"{value:16.6f}" if abs(value) < 1e6 else f"{value:16.3e}"
        print(f"  {lr:>6}  {factor:>9.2f}  {shown}  {verdict}")
    print("""
  lr = 0.5 lands exactly on the target in one step (the factor is 0).
  lr = 1.0 bounces between two points forever.
  lr > 1.0 diverges geometrically.

  Real losses are not quadratic, but the mechanism is identical: too small and
  you never arrive, too large and each step overshoots what it aimed at.""")
    experiment(
        "run examples/two_moons.py with lr=0.5 and compare the loss curve to the default 0.01."
    )


@lesson(13, "Overfitting and validation")
def lesson_13():
    print("""
  Training loss measures memorisation. Validation loss measures generalisation.
  They agree early and diverge later, and the gap is what a validation split
  exists to reveal.""")

    X, y = data.make_moons(n_samples=120, noise=0.35, seed=3)
    Xtr, Xte, ytr, yte = data.train_test_split(X, y, test_size=0.4, seed=3)
    model = nn.Sequential(
        nn.Linear(2, 64, seed=3),
        nn.Tanh(),
        nn.Linear(64, 64, seed=4),
        nn.Tanh(),
        nn.Linear(64, 1, seed=5),
    )
    history = training.fit(
        model,
        optim.Adam(model.parameters(), lr=0.02),
        nn.binary_cross_entropy_with_logits,
        data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=16, shuffle=True, seed=3),
        epochs=400,
        val_loader=data.DataLoader(data.TensorDataset(Xte, yte), batch_size=64),
        metric=nn.accuracy,
    )
    print(
        f"\n  a deliberately oversized model ({model.num_parameters()} parameters) "
        f"on {Xtr.shape[0]} noisy samples\n"
    )
    print(f"  {'epoch':>6}  {'train loss':>11}  {'val loss':>10}  {'gap':>9}")
    for i in (0, 24, 99, 199, 399):
        print(
            f"  {history.epochs[i]:>6}  {history.train_loss[i]:>11.6f}  "
            f"{history.val_loss[i]:>10.6f}  {history.val_loss[i] - history.train_loss[i]:>9.6f}"
        )
    print(f"""
  best validation loss {history.best_val_loss:.6f} at epoch {history.best_epoch} of 400
  final train {history.train_loss[-1]:.6f} vs final validation {history.val_loss[-1]:.6f}

  Past epoch {history.best_epoch} the training loss keeps falling while validation does
  not. Everything learned after that point was about these particular samples,
  not about the problem.""")
    experiment(
        "shrink the model to 2->4->1 and re-run. A smaller model overfits "
        "less — capacity is one of the levers."
    )


@lesson(14, "How Adam differs from SGD")
def lesson_14():
    print("""
  SGD:  theta <- theta - lr·g
  Adam: keeps running averages of g and g², and steps by the first divided by
        the root of the second — which makes the step roughly scale-invariant.""")

    print("\n  scale invariance, demonstrated: identical gradients, 1000x apart\n")
    results = {}
    for name, scale in [("gradient x1", 1.0), ("gradient x1000", 1000.0)]:
        for opt_name, make in [
            ("SGD", lambda p: optim.SGD(p, lr=0.01)),
            ("Adam", lambda p: optim.Adam(p, lr=0.01)),
        ]:
            p = nn.Parameter(gl.Tensor(np.array([1.0])))
            opt = make([p])
            for i in range(10):
                p._grad = gl.Tensor(np.array([(0.3 + 0.05 * i) * scale]))
                opt.step()
            results[(opt_name, name)] = p.item()
    print(f"  {'':>16}  {'SGD':>14}  {'Adam':>14}")
    for name in ("gradient x1", "gradient x1000"):
        print(f"  {name:>16}  {results[('SGD', name)]:>14.8f}  {results[('Adam', name)]:>14.8f}")
    print("""
  SGD's result scales with the gradient; Adam's barely moves. That is why Adam
  needs so much less learning-rate tuning.

  Bias correction, which matters most at step 1:""")
    for grad in (0.001, 1.0, 1000.0):
        p = nn.Parameter(gl.Tensor(np.array([0.0])))
        opt = optim.Adam([p], lr=0.1)
        p._grad = gl.Tensor(np.array([grad]))
        opt.step()
        print(f"    gradient {grad:>8}  ->  first step = {abs(p.item()):.8f}  (lr = 0.1)")
    print("""
  Whatever the gradient's magnitude, the first step has size lr. Without the
  1/(1-beta^t) correction the moving averages start at zero and the first step
  would be about 30x too large.""")
    experiment(
        "train two_moons with SGD(lr=0.01) and Adam(lr=0.01) and compare. "
        "Then find the SGD learning rate that matches Adam's curve."
    )


def main():
    args = sys.argv[1:]
    if "--list" in args:
        for number, title, _ in LESSONS:
            print(f"  {number:>2}. {title}")
        return 0
    wanted = [int(a) for a in args if a.isdigit()]
    selected = [item for item in LESSONS if not wanted or item[0] in wanted]
    if not selected:
        print(f"No lesson matched {wanted}. Use --list to see them.")
        return 1
    for number, title, fn in selected:
        rule(f"LESSON {number} — {title}")
        fn()
    print(
        f"\n{'=' * 74}\n{len(selected)} lesson(s) complete. "
        f"docs/lessons.md is the written companion.\n{'=' * 74}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
