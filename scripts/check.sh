#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "=================================================================="
echo " Vehicle Risk Agent — Local Preflight Verification"
echo "=================================================================="

echo "==> [1/7] Checking code formatting..."
uv run ruff format --check .

echo "==> [2/7] Running Ruff linter..."
uv run ruff check .

echo "==> [3/7] Running mypy strict type checker..."
uv run mypy src tests

echo "==> [4/7] Verifying Alembic database migrations..."
uv run pytest tests/persistence/test_migrations.py -q

echo "==> [5/7] Verifying container and compose contract..."
uv run pytest tests/acceptance/test_compose_contract.py -q

echo "==> [6/7] Running full pytest test suite..."
uv run pytest

echo "==> [7/7] Verifying package build with uv build..."
uv build

echo "=================================================================="
echo " All local preflight checks PASSED successfully!"
echo "=================================================================="
