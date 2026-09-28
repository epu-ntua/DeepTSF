#!/usr/bin/env bash
# Creates the Python environment the test suite runs in (see README.md).
#
# Everything the tests write - the virtualenv, pip/uv caches, the MLflow and S3
# stores, temp files - lives under $DEEPTSF_TEST_HOME, so a large disk can be
# chosen for it. Default: deeptsf_backend/tests/.work (gitignored).
#
# The environment mirrors the docker images: Python 3.12 and the pip pins of
# ../conda.yaml, except that torch is the CPU build (no CUDA wheels).
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$TESTS_DIR")"
export DEEPTSF_TEST_HOME="${DEEPTSF_TEST_HOME:-$TESTS_DIR/.work}"
mkdir -p "$DEEPTSF_TEST_HOME/tmp"

export TMPDIR="$DEEPTSF_TEST_HOME/tmp"
export PIP_CACHE_DIR="$DEEPTSF_TEST_HOME/cache/pip"
export UV_CACHE_DIR="$DEEPTSF_TEST_HOME/cache/uv"
export UV_PYTHON_INSTALL_DIR="$DEEPTSF_TEST_HOME/python"
export UV_PYTHON_BIN_DIR="$DEEPTSF_TEST_HOME/python/bin"   # keep uv from linking into ~/.local/bin

# uv gives us the pinned Python version without conda.
if [ ! -x "$DEEPTSF_TEST_HOME/bootstrap/bin/uv" ]; then
    python3 -m venv "$DEEPTSF_TEST_HOME/bootstrap"
    "$DEEPTSF_TEST_HOME/bootstrap/bin/pip" install -q uv
fi
UV="$DEEPTSF_TEST_HOME/bootstrap/bin/uv"
"$UV" python install 3.12
VENV="$DEEPTSF_TEST_HOME/venv"
[ -x "$VENV/bin/python" ] || "$UV" venv --python 3.12 "$VENV"

# Pins from conda.yaml's pip section, minus CUDA / GPU torch wheels (replaced
# by the CPU builds below) and the editable darts_mlp (installed separately).
REQ="$DEEPTSF_TEST_HOME/requirements-from-conda.txt"
awk '/- pip:/{f=1;next} f && /^ *- /{print $2}' "$BACKEND_DIR/conda.yaml" \
    | grep -viE '^(nvidia-|triton|torch==|torchvision==|torchaudio==|-e|darts-mlp|darts_mlp)' > "$REQ"

"$UV" pip install --python "$VENV/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.2.2+cpu" "torchvision==0.17.2+cpu" "torchaudio==2.2.2+cpu" \
    -r "$REQ" -r "$TESTS_DIR/requirements-test.txt"
"$UV" pip install --python "$VENV/bin/python" --no-deps -e "$BACKEND_DIR/darts_mlp"

echo
echo "Test environment ready: $VENV"
echo "Run the tests with:  DEEPTSF_TEST_HOME=$DEEPTSF_TEST_HOME $TESTS_DIR/run_tests.sh"
