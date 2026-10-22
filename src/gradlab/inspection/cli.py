"""``gradlab-microscope`` — generate traces and serve the Gradient Microscope.

    gradlab-microscope generate            # run every preset, write the traces
    gradlab-microscope serve               # open the app in a browser
    gradlab-microscope trace "x^2 + 2*x" --at x=3

``generate`` executes real computations through the engine. The web app contains
no precomputed graph of its own: with no traces present it says so rather than
displaying an illustration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parents[3] / "apps" / "microscope"
TRACE_DIR = APP_DIR / "traces"


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return path


def cmd_generate(args: argparse.Namespace) -> int:
    from gradlab.inspection import presets

    out_dir = Path(args.out) if args.out else TRACE_DIR
    index: list[dict[str, Any]] = []

    print("Running the expression presets through the engine...")
    for slug, trace in presets.build_expression_traces().items():
        path = _write(out_dir / f"{slug}.json", trace)
        index.append(
            {
                "slug": slug,
                "kind": "expression",
                "label": trace["label"],
                "file": path.name,
                "nodes": trace.get("num_nodes", 0),
            }
        )
        print(f"  {slug:<22} {trace.get('num_nodes', 0):>3} nodes  -> {path.name}")

    print("Running one real Adam training step...")
    trace = presets.build_training_step_trace()
    path = _write(out_dir / "training-step.json", trace)
    index.append(
        {
            "slug": "training-step",
            "kind": "training_step",
            "label": trace["label"],
            "file": path.name,
            "nodes": trace.get("num_nodes", 0),
        }
    )
    print(f"  training-step          {trace.get('num_nodes', 0):>3} nodes  -> {path.name}")

    if not args.fast:
        print("Training the two-moons model (300 epochs)...")
        trace = presets.build_training_run_trace()
        path = _write(out_dir / "training-run.json", trace)
        index.append(
            {
                "slug": "training-run",
                "kind": "training_run",
                "label": trace["label"],
                "file": path.name,
                "val_accuracy": trace["final"]["val_accuracy"],
            }
        )
        print(
            f"  training-run           val acc {trace['final']['val_accuracy']:.4f} -> {path.name}"
        )

        print("Sweeping seven learning rates + a divergence demo...")
        trace = presets.build_lr_comparison_trace()
        path = _write(out_dir / "learning-rates.json", trace)
        index.append(
            {
                "slug": "learning-rates",
                "kind": "lr_comparison",
                "label": trace["label"],
                "file": path.name,
            }
        )
        worst = max(r.get("max_loss", 0.0) for r in trace["runs"])
        demo = trace.get("divergence_demo", {})
        print(
            f"  learning-rates         worst loss {worst:.3g}; divergence demo "
            f"{'stopped' if demo.get('stopped') else 'completed'} after "
            f"{demo.get('batches_survived', 0)} batches -> {path.name}"
        )

    import gradlab
    from gradlab.inspection.schema import SCHEMA_VERSION

    _write(
        out_dir / "index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "gradlab_version": gradlab.__version__,
            "generated_by": "gradlab-microscope generate",
            "traces": index,
        },
    )
    print(f"\n{len(index)} traces written to {out_dir}")
    print("View them with: gradlab-microscope serve")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from gradlab.inspection.expressions import ExpressionError, evaluate
    from gradlab.inspection.tracer import trace_expression

    values: dict[str, Any] = {}
    for assignment in args.at or []:
        if "=" not in assignment:
            print(f"error: --at expects name=value, got {assignment!r}", file=sys.stderr)
            return 2
        name, raw = assignment.split("=", 1)
        try:
            values[name.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            print(f"error: could not read a number from {raw!r}", file=sys.stderr)
            return 2

    try:
        output, variables = evaluate(args.expression, values)
    except ExpressionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    names = list(variables)
    trace = trace_expression(
        lambda *ts, _out=output: _out,
        *[variables[n] for n in names],
        names=names,
        label=args.expression,
    )
    trace["expression_text"] = args.expression

    if args.json:
        path = _write(Path(args.json), trace)
        print(f"trace written to {path}")
        return 0

    print(f"\n{args.expression}")
    print("=" * max(len(args.expression), 40))
    for name, tensor in variables.items():
        print(f"  {name} = {tensor.data.tolist()}")
    print(f"\n  value  = {trace['output']['value']['values']}")
    print(f"  graph  = {trace['num_nodes']} nodes, depth {trace['depth']}\n")
    print("  forward pass")
    for step in trace["forward_steps"]:
        print(f"    {step['step']:>2}. {step['expression']:<28} -> shape {step['output']['shape']}")
    print("\n  backward pass")
    for step in trace["backward_steps"]:
        upstream = step["upstream_gradient"]["values"]
        print(f"    {step['step']:>2}. {step['op']:<12} upstream {upstream}")
        print(f"        {step['backward_rule']}")
        for contribution in step["contributions"]:
            if contribution["received_gradient"]:
                if contribution.get("broadcast_reduced"):
                    note = "  [summed back down over broadcast axes]"
                elif contribution.get("shape_changed"):
                    note = "  [reshaped by the rule]"
                else:
                    note = ""
                values = contribution["gradient"]["values"]
                print(f"        -> {contribution['name']:<8} {values}{note}")
    print("\n  gradients")
    for leaf in trace["leaf_gradients"]:
        print(f"    d/d{leaf['name']:<8} = {leaf['gradient']['values']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from gradlab.inspection.server import serve

    if not (TRACE_DIR / "index.json").exists():
        print("No traces found. Generating them first...\n")
        cmd_generate(argparse.Namespace(out=None, fast=args.fast))
        print()
    return serve(
        directory=APP_DIR,
        port=args.port,
        open_browser=not args.no_browser,
        allow_expressions=args.allow_expressions,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gradlab-microscope",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="run every preset and write its trace")
    gen.add_argument("--out", help="output directory (default: apps/microscope/traces)")
    gen.add_argument("--fast", action="store_true", help="skip the multi-epoch training presets")
    gen.set_defaults(func=cmd_generate)

    tr = sub.add_parser("trace", help="trace one expression")
    tr.add_argument("expression", help='e.g. "x^2 + 2*x"')
    tr.add_argument(
        "--at",
        action="append",
        metavar="NAME=VALUE",
        help="variable value, repeatable (e.g. --at x=3)",
    )
    tr.add_argument("--json", help="write the full trace to this path instead of printing")
    tr.set_defaults(func=cmd_trace)

    sv = sub.add_parser("serve", help="serve the Gradient Microscope locally")
    sv.add_argument("--port", type=int, default=8371)
    sv.add_argument("--no-browser", action="store_true")
    sv.add_argument("--fast", action="store_true")
    sv.add_argument(
        "--allow-expressions",
        action="store_true",
        help="enable the restricted expression endpoint (allowlisted ops only, no "
        "Python evaluation; see docs/limitations.md before exposing this)",
    )
    sv.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
