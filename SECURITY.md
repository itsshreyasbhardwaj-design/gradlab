# Security

## Scope

GradLab is a local, offline library. The library itself has no network access,
no deserialisation of untrusted formats, and one dependency.

**The security-relevant component is the Gradient Microscope's optional
expression endpoint**, because it is the only place GradLab accepts input from
outside the process.

## Reporting a vulnerability

Open a [GitHub security advisory](https://github.com/itsshreyasbhardwaj-design/gradlab/security/advisories/new),
or email the maintainer. Please do not open a public issue for a vulnerability.

Include: what you can make it do, the smallest input that does it, and the
version or commit.

## The expression endpoint

### Off by default

`gradlab-microscope serve` has **no execution surface at all**: it serves static
JSON traces generated beforehand. The endpoint exists only with
`--allow-expressions`, and a request to it otherwise returns 403.

### How user expressions are handled

**No Python is evaluated.** There is no `eval`, no `exec`, no `compile`, and no
use of the `ast` module anywhere in the package —
`tests/unit/test_architecture.py` enforces this by parsing every shipped module.

Instead:

1. **Tokenisation** against a fixed character set. Anything else is rejected by
   position: `__import__('os')` fails on the quote at position 11.
2. **A hand-written recursive-descent parser** produces a plain-data tree that
   can contain only numbers, variable names, and operation names drawn from a
   14-entry allowlist. A name not in that allowlist is a parse error, so no
   other attribute of a tensor is reachable.
3. **Evaluation** maps allowlisted names to GradLab tensor methods. Nothing else
   is callable.

Using Python's own parser — even `ast` with a node allowlist — would mean
tracking what each Python release adds to the grammar. A 200-line parser with an
explicit character set has no such exposure.

### Limits

| bound | value |
|---|---|
| expression length | 400 characters |
| tokens | 360 |
| parsed nodes | 120 |
| nesting depth | 24 |
| variables | 8 |
| elements per tensor | 4,096 |
| tensor dimensions | 3 |
| request body | 16 KiB |
| requests per client | 120/minute |

Non-finite variable values are rejected.

### Server

* **Binds to 127.0.0.1 only.** Never `0.0.0.0`.
* **Static files** are resolved then checked to be inside the served directory,
  so `..` cannot escape, and only a fixed set of extensions is served.
* **No authentication**, because it is a local tool.

### Do not expose it

The Microscope is a local teaching tool. Do not run it on a public interface or
behind a reverse proxy. If you need a shared deployment, serve the generated
traces as static files **without** `--allow-expressions` — that mode runs no
user code at all.

## What GradLab does not do

* No network access anywhere, except `gradlab.data.optdigits.download_split`,
  which is never called implicitly and requires an explicit `--download` flag.
* No pickle, no YAML, no untrusted deserialisation. `state_dict()` is a plain
  dict of NumPy arrays; saving it is the caller's choice and `pickle` of a
  state dict carries the usual pickle risks.
* No code execution from configuration or data files.
* No telemetry.

## Supported versions

Pre-1.0: only the latest release receives fixes.

| version | supported |
|---|---|
| 0.1.x | yes |
