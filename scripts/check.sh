#!/usr/bin/env bash
set -euo pipefail

echo "=================================================================="
echo " Vehicle Risk Agent — Local Preflight Verification"
echo "=================================================================="

echo "==> [1/5] Running Ruff linter..."
uv run ruff check .

echo "==> [2/5] Checking code formatting..."
uv run ruff format --check .

echo "==> [3/5] Running mypy strict type checker..."
uv run mypy src tests

echo "==> [4/5] Running pytest test suite..."
uv run pytest

echo "==> [5/5] Verifying package build with uv build..."
uv build

echo "=================================================================="
echo " All local preflight checks PASSED successfully!"
echo "=================================================================="
