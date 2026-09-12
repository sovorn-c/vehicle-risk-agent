#!/usr/bin/env bash
# Run the real, credential-gated e12 walkthrough. No offline fallback is allowed.
set -euo pipefail

if [[ -z "${ANTHROPIC_API_KEY:-}" || -z "${MCP_SERVER_URL:-}" ]]; then
  echo "BLOCKED: set ANTHROPIC_API_KEY and MCP_SERVER_URL for the real run." >&2
  exit 1
fi

output_file="${E12_REPORT_FILE:-artifacts/e12-comparative-report.json}"
publication_file="${E12_PUBLICATION_FILE:-${output_file%.json}.publication.json}"
mkdir -p "$(dirname "$output_file")" "$(dirname "$publication_file")"

uv run python -m vehicle_risk_agent.evaluation.live \
  --enable-live-eval \
  --suite e12-comparative \
  --model claude-sonnet-4-6 \
  --max-budget-usd 15.00 \
  --max-scenarios 4 \
  --output-file "$output_file"
uv run python -m vehicle_risk_agent.evaluation.publication \
  --report "$output_file" \
  --publication-file "$publication_file" >/dev/null

echo "Measured e12 report: $output_file"
echo "Artifact-bound publication: $publication_file"
echo "The workflow stops at AWAITING_REVIEW; it never performs an approval action."
echo "Vehicle evidence is synthetic demonstration data."
echo "The report must be reviewed by an operator before any PASS is treated as evidence."
