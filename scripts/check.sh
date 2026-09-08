#!/usr/bin/env bash
set -euo pipefail

echo "=================================================================="
echo " Vehicle Risk Agent — Local Preflight Verification"
echo "=================================================================="

echo "==> [1/7] Preflight check — RED baseline"
echo "FAIL: container contract not integrated into preflight" >&2
exit 1

echo "==> [2/6] Checking code formatting..."
uv run ruff format --check .

echo "==> [3/6] Running mypy strict type checker..."
uv run mypy src tests

echo "==> [4/6] Verifying Alembic database migrations..."
uv run pytest tests/persistence/test_migrations.py -q

echo "==> [5/6] Running full pytest test suite..."
uv run pytest

echo "==> [6/6] Verifying package build with uv build..."
uv build

echo "=================================================================="
echo " All local preflight checks PASSED successfully!"
echo "=================================================================="
