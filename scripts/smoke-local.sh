#!/usr/bin/env bash
# story: e07s03
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=================================================================="
echo " Vehicle Risk Agent — Local Smoke Verification"
echo "=================================================================="
echo "Workspace: ${ROOT_DIR}"

cd "${ROOT_DIR}"

# 1. Run migrations and deterministic seed
echo "==> [1/2] Seeding database with deterministic policies and corpora..."
uv run python -m vehicle_risk_agent.cli.seed

# 2. Run full domain smoke sequence
echo "==> [2/2] Running end-to-end smoke verification..."
uv run python -m vehicle_risk_agent.cli.smoke

echo "=================================================================="
echo " All local smoke scenarios PASSED successfully!"
echo "=================================================================="
