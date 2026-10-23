#!/usr/bin/env bash
# Build, verify and upload a GradLab release.
#
# The PyPI distribution name is "gradlab-ml"; the import name is "gradlab".
#
# This script never stores or echoes a credential. twine reads the token from
# your environment or ~/.pypirc, and prompts you for it if neither is set.
# Create one at https://pypi.org/manage/account/token/ and scope it to this
# project after the first upload.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-./.venv/bin/python}

echo "==> working tree must be clean and pushed"
git diff --quiet || { echo "uncommitted changes"; exit 1; }
git fetch -q origin
git diff --quiet origin/main || { echo "not identical to origin/main"; exit 1; }

echo "==> gates"
$PY -m ruff check src tests examples benchmarks
$PY -m ruff format --check src tests examples benchmarks
$PY -m mypy
$PY -m pytest -q

echo "==> build"
rm -rf dist
uv build

echo "==> metadata check"
$PY -m twine check dist/*

echo "==> clean-environment install test"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
uv venv --python 3.12 "$TMP/venv" >/dev/null
uv pip install --python "$TMP/venv/bin/python" -q dist/*.whl
"$TMP/venv/bin/python" - <<'PY'
import gradlab as gl
x = gl.Tensor(3.0, requires_grad=True); y = x * x + 2 * x; y.backward()
assert (y.item(), x.grad.item()) == (15.0, 8.0), "smoke test failed"
print(f"    gradlab {gl.__version__} imports and differentiates correctly")
PY

VERSION=$($PY -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
cat <<EOF

==> ready to upload gradlab-ml $VERSION

    A PyPI version number can never be reused, even after deletion.
    Dry run first if you want:  $PY -m twine upload --repository testpypi dist/*

EOF

# Require a real terminal. Without this, a piped or redirected stdin makes
# `read` return instantly and the upload would proceed unattended — on an
# operation whose version number can never be reused.
if [ ! -t 0 ]; then
  echo "Refusing to upload: stdin is not a terminal."
  echo "Run this directly from a shell, not piped or in CI."
  exit 1
fi

printf 'Press Enter to upload to PyPI, or Ctrl-C to stop. '
read -r
$PY -m twine upload dist/*
echo "==> done: https://pypi.org/project/gradlab-ml/$VERSION/"
