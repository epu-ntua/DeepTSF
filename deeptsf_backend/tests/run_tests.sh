#!/usr/bin/env bash
# Runs the DeepTSF test suite (see README.md). Extra arguments go to pytest, e.g.
#   ./run_tests.sh -k "LightGBM and 1h"        # a subset
#   DEEPTSF_TEST_WORKERS=2 ./run_tests.sh      # fewer parallel workers
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEEPTSF_TEST_HOME="${DEEPTSF_TEST_HOME:-$TESTS_DIR/.work}"
PYTHON="$DEEPTSF_TEST_HOME/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "No test environment in $DEEPTSF_TEST_HOME, run setup_test_env.sh first." >&2
    exit 1
fi

export DEEPTSF_TEST_RUN_ID="${DEEPTSF_TEST_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
export TMPDIR="$DEEPTSF_TEST_HOME/tmp"
mkdir -p "$TMPDIR" "$DEEPTSF_TEST_HOME/runs"

# An interrupted run (Ctrl-C, killed terminal) can leave its local MLflow servers
# behind; stop any process of this test environment whose parent has exited.
ps -eo pid,ppid,pgid,args | awk -v venv="$DEEPTSF_TEST_HOME/venv/" \
    '$2 == 1 && index($0, venv) { print $3 }' | sort -u | while read -r pgid; do
    kill -TERM -- "-$pgid" 2>/dev/null || true
done

# keep the files of the last 3 runs only
ls -1dt "$DEEPTSF_TEST_HOME"/runs/*/ 2>/dev/null | tail -n +4 | xargs -r rm -rf

cd "$TESTS_DIR"
# --dist loadgroup: the pipeline and inference tests of a case share a worker (and its run)
exec "$PYTHON" -m pytest -n "${DEEPTSF_TEST_WORKERS:-4}" --dist loadgroup \
    --junitxml "$DEEPTSF_TEST_HOME/runs/$DEEPTSF_TEST_RUN_ID/junit.xml" "$@"
