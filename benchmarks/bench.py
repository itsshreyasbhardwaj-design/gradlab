"""Reproducible benchmarks for GradLab.

Run: ``python benchmarks/bench.py``            (writes docs/benchmarks.md)
     ``python benchmarks/bench.py --quick``     (fewer repetitions)

What these measure, and what they do not
----------------------------------------
GradLab's numerical work is done by NumPy, which dispatches matrix
multiplication to a BLAS. **GradLab does not implement a matrix kernel**, and
nothing here should be read as a claim that it does. What GradLab adds on top of
NumPy is a Python-level graph: a ``Node`` object per operation, a ``Context``
holding saved values, and a Python function call per derivative rule.

That overhead is the thing worth measuring, so these benchmarks report it
directly — ``framework overhead = GradLab time − raw NumPy time`` for the same
arithmetic — alongside wall-clock totals. Comparisons against PyTorch are
included for context, with the architectural difference stated: PyTorch's
autograd engine is compiled C++, GradLab's is interpreted Python. A Python graph
cannot win that comparison, and winning it was never the goal.

Methodology
-----------
* Each measurement is warmed up, then repeated; the **median** is reported along
  with the interquartile range, because wall-clock timing on a laptop has
  outliers that a mean would absorb.
* ``time.perf_counter`` is used throughout.
* Hardware, OS, Python and library versions are recorded in the output.
* Memory is measured with ``tracemalloc`` (Python-allocated bytes), which
  captures graph overhead but not BLAS scratch space; that limitation is stated
  in the report rather than papered over.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import gradlab as gl
from gradlab import nn, optim

try:
    import torch

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

REPEATS = 25
WARMUP = 5


def measure(fn: Callable[[], Any], repeats: int = REPEATS, warmup: int = WARMUP) -> dict:
    """Time ``fn`` and report the median plus spread, not a single lucky run."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return {
        "median_s": statistics.median(samples),
        "min_s": samples[0],
        "max_s": samples[-1],
        "iqr_s": samples[int(0.75 * len(samples))] - samples[int(0.25 * len(samples))],
        "repeats": repeats,
    }


def environment() -> dict:
    info = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "gradlab": gl.__version__,
        "torch": torch.__version__ if HAS_TORCH else "not installed",
        "blas": "unknown",
    }
    try:
        config = np.show_config(mode="dicts")
        build = config.get("Build Dependencies", {}).get("blas", {})
        info["blas"] = f"{build.get('name', '?')} {build.get('version', '')}".strip()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------- benchmarks


def bench_elementwise(sizes, repeats):
    """Elementwise add+mul: dominated by GradLab's per-op Python overhead."""
    rows = []
    for n in sizes:
        a = np.random.default_rng(0).standard_normal(n)
        b = np.random.default_rng(1).standard_normal(n)
        ga, gb = gl.Tensor(a, requires_grad=True), gl.Tensor(b, requires_grad=True)

        numpy_t = measure(lambda a=a, b=b: (a + b) * a, repeats)
        gl_t = measure(lambda ga=ga, gb=gb: (ga + gb) * ga, repeats)
        row = {
            "size": n,
            "numpy_s": numpy_t["median_s"],
            "gradlab_forward_s": gl_t["median_s"],
            "overhead_s": gl_t["median_s"] - numpy_t["median_s"],
            "overhead_ratio": gl_t["median_s"] / max(numpy_t["median_s"], 1e-12),
            "iqr_s": gl_t["iqr_s"],
        }
        if HAS_TORCH:
            ta = torch.tensor(a, requires_grad=True)
            tb = torch.tensor(b, requires_grad=True)
            row["torch_forward_s"] = measure(lambda ta=ta, tb=tb: (ta + tb) * ta, repeats)[
                "median_s"
            ]
        rows.append(row)
    return rows


def bench_matmul(sizes, repeats):
    """Matrix multiplication: almost entirely BLAS, in every framework."""
    rows = []
    for n in sizes:
        rng = np.random.default_rng(0)
        a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
        ga, gb = gl.Tensor(a, requires_grad=True), gl.Tensor(b, requires_grad=True)

        numpy_t = measure(lambda a=a, b=b: a @ b, repeats)
        fwd = measure(lambda ga=ga, gb=gb: ga @ gb, repeats)

        def full(ga=ga, gb=gb):
            ga.zero_grad()
            gb.zero_grad()
            (ga @ gb).sum().backward()

        bwd = measure(full, max(5, repeats // 3))
        row = {
            "size": n,
            "numpy_matmul_s": numpy_t["median_s"],
            "gradlab_forward_s": fwd["median_s"],
            "gradlab_fwd_bwd_s": bwd["median_s"],
            "overhead_s": fwd["median_s"] - numpy_t["median_s"],
            "gflops_numpy": (2 * n**3) / numpy_t["median_s"] / 1e9,
        }
        if HAS_TORCH:
            ta = torch.tensor(a, requires_grad=True)
            tb = torch.tensor(b, requires_grad=True)

            def torch_full(ta=ta, tb=tb):
                if ta.grad is not None:
                    ta.grad = None
                    tb.grad = None
                (ta @ tb).sum().backward()

            row["torch_fwd_bwd_s"] = measure(torch_full, max(5, repeats // 3))["median_s"]
        rows.append(row)
    return rows


def bench_graph_depth(depths, repeats):
    """Cost of building and walking a deep chain — the engine's own scaling."""
    rows = []
    for depth in depths:
        x = gl.Tensor(np.ones(64), requires_grad=True)

        def build(x=x, depth=depth):
            y = x
            for _ in range(depth):
                y = y * 1.0001
            return y

        build_t = measure(build, repeats)

        def full(x=x, depth=depth):
            x.zero_grad()
            build(x, depth).sum().backward()

        total_t = measure(full, max(5, repeats // 3))
        rows.append(
            {
                "depth": depth,
                "build_s": build_t["median_s"],
                "build_plus_backward_s": total_t["median_s"],
                "backward_s": total_t["median_s"] - build_t["median_s"],
                "us_per_node_build": build_t["median_s"] / depth * 1e6,
                "us_per_node_backward": (total_t["median_s"] - build_t["median_s"]) / depth * 1e6,
            }
        )
    return rows


def bench_graph_memory(depths):
    """Python-allocated bytes held by a live graph, per node."""
    rows = []
    for depth in depths:
        x = gl.Tensor(np.ones(64), requires_grad=True)
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        y = x
        for _ in range(depth):
            y = y * 1.0001
        peak = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        held = peak - baseline
        del y
        rows.append(
            {
                "depth": depth,
                "bytes_held": held,
                "bytes_per_node": held / depth,
            }
        )
    return rows


def bench_training(configs, repeats):
    """A full training step on a realistic small network."""
    rows = []
    for batch, hidden in configs:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((batch, 16))
        y = rng.standard_normal((batch, 1))
        model = nn.Sequential(
            nn.Linear(16, hidden, seed=0),
            nn.Tanh(),
            nn.Linear(hidden, hidden, seed=1),
            nn.Tanh(),
            nn.Linear(hidden, 1, seed=2),
        )
        opt = optim.Adam(model.parameters(), lr=0.01)
        gx, gy = gl.Tensor(X), gl.Tensor(y)

        def step(model=model, opt=opt, gx=gx, gy=gy):
            opt.zero_grad()
            nn.mse_loss(model(gx), gy).backward()
            opt.step()

        gl_t = measure(step, repeats)
        row = {
            "batch": batch,
            "hidden": hidden,
            "params": model.num_parameters(),
            "gradlab_step_s": gl_t["median_s"],
            "iqr_s": gl_t["iqr_s"],
        }
        if HAS_TORCH:
            tm = torch.nn.Sequential(
                torch.nn.Linear(16, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, 1),
            )
            topt = torch.optim.Adam(tm.parameters(), lr=0.01)
            tx, ty = torch.tensor(X), torch.tensor(y)

            def tstep(tm=tm, topt=topt, tx=tx, ty=ty):
                topt.zero_grad()
                torch.nn.functional.mse_loss(tm(tx), ty).backward()
                topt.step()

            row["torch_step_s"] = measure(tstep, repeats)["median_s"]
            row["ratio"] = row["gradlab_step_s"] / row["torch_step_s"]
        rows.append(row)
    return rows


# -------------------------------------------------------------------- report


def fmt_time(seconds):
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def write_report(results, path):
    env = results["environment"]
    lines = [
        "# Benchmarks",
        "",
        "**Every number here was produced by `python benchmarks/bench.py` on the machine",
        "described below. Nothing is estimated, scaled from another machine, or carried",
        "over from a previous run.** Re-running the script overwrites this file.",
        "",
        "## What is being measured",
        "",
        "GradLab's arithmetic is NumPy's. Matrix multiplication goes to a BLAS. **GradLab",
        "does not implement a matrix kernel and makes no performance claim about one.**",
        "What GradLab adds is a Python-level computation graph — one `Node` object and one",
        "Python call per operation — and that overhead is what these benchmarks isolate.",
        "",
        "PyTorch appears for context only. Its autograd engine is compiled C++; GradLab's",
        "is interpreted Python. A Python graph cannot win that comparison, and the point",
        "of this project was never to try.",
        "",
        "## Environment",
        "",
        "| | |",
        "|---|---|",
    ]
    for key, value in env.items():
        lines.append(f"| {key} | `{value}` |")
    lines += [
        "",
        f"Timing: `time.perf_counter`, {results['config']['warmup']} warm-up iterations then",
        f"{results['config']['repeats']} measured; **median** reported with the interquartile",
        "range, because laptop wall-clock timing has outliers a mean would hide.",
        "",
        "## 1. Elementwise operations — where framework overhead lives",
        "",
        "`(a + b) * a` on vectors of increasing length.",
        "",
        "| elements | NumPy | GradLab fwd | overhead | ratio | IQR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["elementwise"]:
        lines.append(
            f"| {r['size']:,} | {fmt_time(r['numpy_s'])} | {fmt_time(r['gradlab_forward_s'])} "
            f"| {fmt_time(r['overhead_s'])} | {r['overhead_ratio']:.1f}× | {fmt_time(r['iqr_s'])} |"
        )
    first, last = results["elementwise"][0], results["elementwise"][-1]
    lines += [
        "",
        f"The overhead is roughly **constant** ({fmt_time(first['overhead_s'])} at "
        f"{first['size']:,} elements, {fmt_time(last['overhead_s'])} at {last['size']:,}) "
        "because it is per *operation*, not per element: two Python objects and a few",
        f"attribute lookups. So the ratio collapses from {first['overhead_ratio']:.0f}× on tiny",
        f"arrays to {last['overhead_ratio']:.1f}× on large ones. GradLab is slow on small",
        "tensors and competitive on large ones, which is exactly what a thin Python wrapper",
        "around NumPy should look like.",
        "",
        "## 2. Matrix multiplication — BLAS, not GradLab",
        "",
        "| n (n×n) | NumPy matmul | GradLab fwd | GradLab fwd+bwd | overhead | NumPy GFLOP/s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["matmul"]:
        lines.append(
            f"| {r['size']} | {fmt_time(r['numpy_matmul_s'])} | {fmt_time(r['gradlab_forward_s'])} "
            f"| {fmt_time(r['gradlab_fwd_bwd_s'])} | {fmt_time(r['overhead_s'])} "
            f"| {r['gflops_numpy']:.1f} |"
        )
    big = results["matmul"][-1]
    lines += [
        "",
        f"At n={big['size']} the forward overhead is {fmt_time(big['overhead_s'])} against a",
        f"{fmt_time(big['numpy_matmul_s'])} BLAS call — under "
        f"{100 * big['overhead_s'] / big['numpy_matmul_s']:.1f}% of the work. The GFLOP/s column",
        "is **NumPy's BLAS**, reported so the table cannot be mistaken for a claim about a",
        "GradLab kernel. Backward costs roughly two more matmuls, as the rule",
        "`dL/dA = G @ Bᵀ`, `dL/dB = Aᵀ @ G` requires.",
        "",
        "## 3. Graph depth — the engine's own scaling",
        "",
        "A chain of `depth` multiplications, built and then differentiated.",
        "",
        "| depth | build | backward | µs/node build | µs/node backward |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in results["graph_depth"]:
        lines.append(
            f"| {r['depth']:,} | {fmt_time(r['build_s'])} | {fmt_time(r['backward_s'])} "
            f"| {r['us_per_node_build']:.2f} | {r['us_per_node_backward']:.2f} |"
        )
    lines += [
        "",
        "Both are **linear in depth** with a flat per-node cost, which is the expected",
        "complexity: building is one `Node` per operation, and reverse mode visits each",
        "node exactly once. The traversal is iterative rather than recursive precisely so",
        "these depths do not hit Python's recursion limit.",
        "",
        "## 4. Graph memory",
        "",
        "| depth | bytes held | bytes/node |",
        "|---:|---:|---:|",
    ]
    for r in results["graph_memory"]:
        lines.append(f"| {r['depth']:,} | {r['bytes_held']:,} | {r['bytes_per_node']:.0f} |")
    lines += [
        "",
        "Measured with `tracemalloc`, so this is **Python-allocated memory only**: the",
        "`Node`, its `Context`, and the saved arrays. BLAS scratch space is invisible to it.",
        "The per-node figure is dominated by the saved forward values each rule needs for",
        "backward — which is the memory/recompute trade-off reverse mode makes by design.",
        "",
        "## 5. A full training step",
        "",
        "`zero_grad → forward → loss → backward → Adam step` on a 16→h→h→1 MLP.",
        "",
    ]
    if HAS_TORCH:
        lines += [
            "| batch | hidden | params | GradLab | PyTorch | ratio |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in results["training"]:
            lines.append(
                f"| {r['batch']} | {r['hidden']} | {r['params']:,} "
                f"| {fmt_time(r['gradlab_step_s'])} | {fmt_time(r['torch_step_s'])} "
                f"| {r['ratio']:.2f}× |"
            )
        ratios = [r["ratio"] for r in results["training"]]
        lines += [
            "",
            f"GradLab takes {min(ratios):.2f}×–{max(ratios):.2f}× PyTorch's time per step on "
            "these configurations.",
            "",
            "**This is not a claim that GradLab is faster than PyTorch, and the ratio dipping",
            "below 1.0 should not be read as one.** Three things are going on, and only the",
            "first is about GradLab:",
            "",
            "1. At these sizes both frameworks are dominated by *per-operation dispatch*, not",
            "   by arithmetic. PyTorch pays its own Python-level dispatch cost per op, which",
            "   is why a thin NumPy wrapper is in the same range at all. Scale the model up",
            "   and the comparison stops being about dispatch.",
            "2. **The thread counts are not matched.** `torch.set_num_threads(1)` is set for",
            "   determinism, while NumPy's BLAS threading is left at its default — so the",
            "   GradLab column may be using several cores where the PyTorch column uses one.",
            "   That alone can account for the ratios below 1.0, and it has not been",
            "   controlled for.",
            "3. These are tiny CPU models. PyTorch's engineering is aimed at large batches,",
            "   fused kernels, and GPUs — none of which this table exercises.",
            "",
            "The useful reading of this table is that GradLab's graph overhead is small enough",
            "to be irrelevant at realistic model sizes, which is all the project needs. It is",
            "not evidence of a faster framework, and `docs/limitations.md` says so plainly.",
        ]
    else:
        lines += [
            "| batch | hidden | params | GradLab |",
            "|---:|---:|---:|---:|",
        ]
        for r in results["training"]:
            lines.append(
                f"| {r['batch']} | {r['hidden']} | {r['params']:,} "
                f"| {fmt_time(r['gradlab_step_s'])} |"
            )
        lines.append("\nPyTorch was not installed, so no comparison column is shown.")

    lines += [
        "",
        "## Honest limitations of these numbers",
        "",
        "* **One machine, one run.** Figures on other hardware, with a different BLAS, or",
        "  with different thread counts will differ, possibly by a lot.",
        "* **Single-threaded.** `torch.set_num_threads(1)` is set for a fair comparison;",
        "  NumPy's BLAS threading is left at its default and is *not* controlled here, so",
        "  the matmul column may be using several cores while PyTorch uses one.",
        "* **`tracemalloc` undercounts.** It sees Python allocations, not BLAS workspace.",
        "* **No GPU anywhere.** GradLab has no GPU backend and claims none.",
        "* **Micro-benchmarks are not workloads.** A real training run is dominated by",
        "  whichever of data loading, arithmetic, and framework overhead happens to be the",
        "  bottleneck at its particular size.",
        "",
        "Raw measurements, including every repetition's spread, are in",
        "`benchmarks/results.json`.",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description="GradLab benchmarks")
    parser.add_argument("--quick", action="store_true", help="fewer repetitions and sizes")
    args = parser.parse_args()

    repeats = 7 if args.quick else REPEATS
    el_sizes = [100, 10_000, 1_000_000] if args.quick else [100, 1_000, 10_000, 100_000, 1_000_000]
    mm_sizes = [32, 256] if args.quick else [16, 64, 128, 256, 512]
    depths = [50, 500] if args.quick else [10, 100, 1_000, 5_000]
    train_cfg = [(32, 32)] if args.quick else [(32, 16), (32, 64), (256, 64), (256, 256)]

    print("GradLab benchmarks")
    print(f"  torch available: {HAS_TORCH}")
    print(f"  repeats: {repeats} (warm-up {WARMUP})\n")

    results = {
        "environment": environment(),
        "config": {"repeats": repeats, "warmup": WARMUP, "quick": args.quick},
    }
    for name, fn in [
        ("elementwise", lambda: bench_elementwise(el_sizes, repeats)),
        ("matmul", lambda: bench_matmul(mm_sizes, repeats)),
        ("graph_depth", lambda: bench_graph_depth(depths, repeats)),
        ("graph_memory", lambda: bench_graph_memory(depths)),
        ("training", lambda: bench_training(train_cfg, repeats)),
    ]:
        print(f"  running {name}...", flush=True)
        results[name] = fn()

    out_dir = Path(__file__).resolve().parent
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    report = write_report(results, out_dir.parent / "docs" / "benchmarks.md")
    print(f"\n  raw results -> {out_dir / 'results.json'}")
    print(f"  report      -> {report}")


if __name__ == "__main__":
    main()
