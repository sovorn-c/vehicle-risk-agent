#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Isolate local .env during preflight to avoid environmental pollution of tests
if [ -f .env ]; then
  ENV_TEMP="$(mktemp -d)"
  mv .env "${ENV_TEMP}/.env"
  cleanup_env() {
    if [ -f "${ENV_TEMP}/.env" ]; then
      mv "${ENV_TEMP}/.env" .env
      rm -rf "${ENV_TEMP}"
    fi
  }
  trap cleanup_env EXIT INT TERM
fi

echo "=================================================================="
echo " Vehicle Risk Agent — Local Preflight Verification"
echo "=================================================================="

echo "==> [1/8] Checking code formatting..."
uv run ruff format --check .

echo "==> [2/8] Running Ruff linter..."
uv run ruff check .

echo "==> [3/8] Running mypy strict type checker..."
uv run mypy src tests

echo "==> [4/8] Verifying Alembic database migrations..."
uv run pytest tests/persistence/test_migrations.py -q

echo "==> [5/8] Verifying container and compose contract..."
uv run pytest tests/acceptance/test_compose_contract.py -q

echo "==> [6/8] Running full pytest test suite with coverage..."
uv run pytest \
  --cov=src/vehicle_risk_agent \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=80

echo "==> [7/8] Checking business-logic coverage..."
uv run coverage report \
  --include='src/vehicle_risk_agent/evidence/*,src/vehicle_risk_agent/investigation/*,src/vehicle_risk_agent/retrieval/*,src/vehicle_risk_agent/risk/*,src/vehicle_risk_agent/workflow/*' \
  --fail-under=80

echo "==> [8/8] Verifying package build with uv build..."
uv build

echo "=================================================================="
echo " All local preflight checks PASSED successfully!"
echo "=================================================================="
