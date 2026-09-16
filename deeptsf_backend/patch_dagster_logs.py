"""Backport of the upstream Dagster fix for captured-log decoding (dagster >= 1.9).

Dagster 1.8.x reads compute logs in 100 KB chunks and decodes each chunk with
strict UTF-8. When a chunk boundary splits a multi-byte character (e.g. the
progress bar blocks printed by PyTorch Lightning), the CapturedLogsQuery fails with
"'utf-8' codec can't decode bytes ...: unexpected end of data".
This replaces the strict decode with errors="replace", as upstream Dagster does.
"""
import importlib.util
import os
import sys

TARGETS = [
    ("dagster_graphql", "schema/logs/compute_logs.py"),
    ("dagster", "_core/storage/compute_log_manager.py"),
]
STRICT = '.decode("utf-8")'
LENIENT = '.decode("utf-8", errors="replace")'


def main():
    for package, rel_path in TARGETS:
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            print(f"[patch_dagster_logs] {package} not installed, skipping")
            continue
        path = os.path.join(spec.submodule_search_locations[0], rel_path)
        with open(path) as f:
            source = f.read()
        count = source.count(STRICT)
        if count == 0:
            print(f"[patch_dagster_logs] {path}: nothing to patch")
            continue
        with open(path, "w") as f:
            f.write(source.replace(STRICT, LENIENT))
        print(f"[patch_dagster_logs] {path}: patched {count} decode call(s)")


if __name__ == "__main__":
    sys.exit(main())
