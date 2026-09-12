#!/usr/bin/env bash
# Run the real, credential-gated e12 walkthrough. No offline fallback is allowed.
set -euo pipefail

: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY for the real model run.}"
: "${MCP_SERVER_URL:?Set MCP_SERVER_URL for the real MCP run.}"

output_file="${E12_REPORT_FILE:-artifacts/e12-comparative-report.json}"
mkdir -p "$(dirname "$output_file")"

uv run python -m vehicle_risk_agent.evaluation.live \
  --enable-live-eval \
  --suite e12-comparative \
  --model claude-sonnet-4-6 \
  --max-budget-usd 15.00 \
  --max-scenarios 4 \
  --output-file "$output_file"

echo "Measured e12 report: $output_file"
echo "The workflow stops at AWAITING_REVIEW; it never performs an approval action."
echo "Vehicle evidence is synthetic demonstration data."
echo "The report must be reviewed by an operator before any PASS is treated as evidence."
