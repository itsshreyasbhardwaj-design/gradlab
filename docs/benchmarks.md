# Benchmarks

**Every number here was produced by `python benchmarks/bench.py` on the machine
described below. Nothing is estimated, scaled from another machine, or carried
over from a previous run.** Re-running the script overwrites this file.

## What is being measured

GradLab's arithmetic is NumPy's. Matrix multiplication goes to a BLAS. **GradLab
does not implement a matrix kernel and makes no performance claim about one.**
What GradLab adds is a Python-level computation graph — one `Node` object and one
Python call per operation — and that overhead is what these benchmarks isolate.

PyTorch appears for context only. Its autograd engine is compiled C++; GradLab's
is interpreted Python. A Python graph cannot win that comparison, and the point
of this project was never to try.

## Environment

| | |
|---|---|
| timestamp_utc | `2026-10-22T20:49:30+00:00` |
| platform | `macOS-26.6.2-arm64-arm-64bit` |
| processor | `arm` |
| machine | `arm64` |
| python | `3.12.13` |
| numpy | `2.5.4` |
| gradlab | `0.1.0` |
| torch | `2.14.1` |
| blas | `accelerate unknown` |

Timing: `time.perf_counter`, 5 warm-up iterations then
25 measured; **median** reported with the interquartile
range, because laptop wall-clock timing has outliers a mean would hide.

## 1. Elementwise operations — where framework overhead lives

`(a + b) * a` on vectors of increasing length.

| elements | NumPy | GradLab fwd | overhead | ratio | IQR |
|---:|---:|---:|---:|---:|---:|
| 100 | 1.1 µs | 9.3 µs | 8.2 µs | 8.6× | 459 ns |
| 1,000 | 1.6 µs | 10.2 µs | 8.5 µs | 6.3× | 291 ns |
| 10,000 | 10.0 µs | 19.5 µs | 9.5 µs | 2.0× | 292 ns |
| 100,000 | 78.5 µs | 92.6 µs | 14.1 µs | 1.2× | 709 ns |
| 1,000,000 | 842.4 µs | 845.4 µs | 3.0 µs | 1.0× | 10.5 µs |

The overhead is roughly **constant** (8.2 µs at 100 elements, 3.0 µs at 1,000,000) because it is per *operation*, not per element: two Python objects and a few
attribute lookups. So the ratio collapses from 9× on tiny
arrays to 1.0× on large ones. GradLab is slow on small
tensors and competitive on large ones, which is exactly what a thin Python wrapper
around NumPy should look like.

## 2. Matrix multiplication — BLAS, not GradLab

| n (n×n) | NumPy matmul | GradLab fwd | GradLab fwd+bwd | overhead | NumPy GFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 16 | 2.0 µs | 9.0 µs | 47.3 µs | 7.1 µs | 4.2 |
| 64 | 5.5 µs | 12.7 µs | 62.1 µs | 7.1 µs | 94.6 |
| 128 | 26.5 µs | 30.6 µs | 126.2 µs | 4.1 µs | 158.3 |
| 256 | 153.1 µs | 157.5 µs | 557.9 µs | 4.4 µs | 219.1 |
| 512 | 1.23 ms | 1.24 ms | 4.04 ms | 11.2 µs | 218.8 |

At n=512 the forward overhead is 11.2 µs against a
1.23 ms BLAS call — under 0.9% of the work. The GFLOP/s column
is **NumPy's BLAS**, reported so the table cannot be mistaken for a claim about a
GradLab kernel. Backward costs roughly two more matmuls, as the rule
`dL/dA = G @ Bᵀ`, `dL/dB = Aᵀ @ G` requires.

## 3. Graph depth — the engine's own scaling

A chain of `depth` multiplications, built and then differentiated.

| depth | build | backward | µs/node build | µs/node backward |
|---:|---:|---:|---:|---:|
| 10 | 60.0 µs | 70.5 µs | 6.00 | 7.05 |
| 100 | 647.0 µs | 507.5 µs | 6.47 | 5.08 |
| 1,000 | 6.75 ms | 5.05 ms | 6.75 | 5.05 |
| 5,000 | 58.10 ms | 15.25 ms | 11.62 | 3.05 |

Both are **linear in depth** with a flat per-node cost, which is the expected
complexity: building is one `Node` per operation, and reverse mode visits each
node exactly once. The traversal is iterative rather than recursive precisely so
these depths do not hit Python's recursion limit.

## 4. Graph memory

| depth | bytes held | bytes/node |
|---:|---:|---:|
| 10 | 14,264 | 1426 |
| 100 | 151,360 | 1514 |
| 1,000 | 1,558,992 | 1559 |
| 5,000 | 8,002,792 | 1601 |

Measured with `tracemalloc`, so this is **Python-allocated memory only**: the
`Node`, its `Context`, and the saved arrays. BLAS scratch space is invisible to it.
The per-node figure is dominated by the saved forward values each rule needs for
backward — which is the memory/recompute trade-off reverse mode makes by design.

## 5. A full training step

`zero_grad → forward → loss → backward → Adam step` on a 16→h→h→1 MLP.

| batch | hidden | params | GradLab | PyTorch | ratio |
|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 561 | 315.0 µs | 280.4 µs | 1.12× |
| 32 | 64 | 5,313 | 384.5 µs | 351.0 µs | 1.10× |
| 256 | 64 | 5,313 | 665.7 µs | 758.5 µs | 0.88× |
| 256 | 256 | 70,401 | 2.20 ms | 2.61 ms | 0.84× |

GradLab takes 0.84×–1.12× PyTorch's time per step on these configurations.

**This is not a claim that GradLab is faster than PyTorch, and the ratio dipping
below 1.0 should not be read as one.** Three things are going on, and only the
first is about GradLab:

1. At these sizes both frameworks are dominated by *per-operation dispatch*, not
   by arithmetic. PyTorch pays its own Python-level dispatch cost per op, which
   is why a thin NumPy wrapper is in the same range at all. Scale the model up
   and the comparison stops being about dispatch.
2. **The thread counts are not matched.** `torch.set_num_threads(1)` is set for
   determinism, while NumPy's BLAS threading is left at its default — so the
   GradLab column may be using several cores where the PyTorch column uses one.
   That alone can account for the ratios below 1.0, and it has not been
   controlled for.
3. These are tiny CPU models. PyTorch's engineering is aimed at large batches,
   fused kernels, and GPUs — none of which this table exercises.

The useful reading of this table is that GradLab's graph overhead is small enough
to be irrelevant at realistic model sizes, which is all the project needs. It is
not evidence of a faster framework, and `docs/limitations.md` says so plainly.

## Honest limitations of these numbers

* **One machine, one run.** Figures on other hardware, with a different BLAS, or
  with different thread counts will differ, possibly by a lot.
* **Single-threaded.** `torch.set_num_threads(1)` is set for a fair comparison;
  NumPy's BLAS threading is left at its default and is *not* controlled here, so
  the matmul column may be using several cores while PyTorch uses one.
* **`tracemalloc` undercounts.** It sees Python allocations, not BLAS workspace.
* **No GPU anywhere.** GradLab has no GPU backend and claims none.
* **Micro-benchmarks are not workloads.** A real training run is dominated by
  whichever of data loading, arithmetic, and framework overhead happens to be the
  bottleneck at its particular size.

Raw measurements, including every repetition's spread, are in
`benchmarks/results.json`.
