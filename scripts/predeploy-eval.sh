#!/usr/bin/env bash
# Offline predeploy gate over the frozen e12 config. Usage: predeploy-eval.sh [--force-fail] [--output-file PATH]
# Exit codes: 0 pass, 1 check failed, 2 usage/runner error, 3 --force-fail.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec uv run --frozen python -m vehicle_risk_agent.cli.predeploy_eval "$@"
