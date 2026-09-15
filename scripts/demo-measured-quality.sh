#!/usr/bin/env bash
# Run the real, credential-gated e12 walkthrough. No offline fallback is allowed.
set -euo pipefail

provider="${E12_PROVIDER:-gemini}"
case "$provider" in
  gemini)
    credential_name="GEMINI_API_KEY"
    model="${E12_MODEL:-gemini-3.1-flash-lite}"
    ;;
  anthropic)
    credential_name="ANTHROPIC_API_KEY"
    model="${E12_MODEL:-claude-sonnet-4-6}"
    ;;
  *)
    echo "BLOCKED: E12_PROVIDER must be gemini or anthropic." >&2
    exit 1
    ;;
esac

if [[ -z "${!credential_name:-}" || -z "${MCP_SERVER_URL:-}" ]]; then
  echo "BLOCKED: set $credential_name and MCP_SERVER_URL for the real run." >&2
  exit 1
fi

output_file="${E12_REPORT_FILE:-artifacts/e12-comparative-report.json}"
publication_file="${E12_PUBLICATION_FILE:-${output_file%.json}.publication.json}"
mkdir -p "$(dirname "$output_file")" "$(dirname "$publication_file")"

evaluation_status=0
uv run python -m vehicle_risk_agent.evaluation.live \
  --enable-live-eval \
  --suite e12-comparative \
  --provider "$provider" \
  --model "$model" \
  --max-budget-usd 15.00 \
  --max-scenarios 4 \
  --output-file "$output_file" || evaluation_status=$?
uv run python -m vehicle_risk_agent.evaluation.publication \
  --report "$output_file" \
  --publication-file "$publication_file" >/dev/null

echo "Measured e12 report: $output_file"
echo "Artifact-bound publication: $publication_file"
echo "The workflow stops at AWAITING_REVIEW; it never performs an approval action."
echo "Vehicle evidence is synthetic demonstration data."
echo "The report must be reviewed by an operator before any PASS is treated as evidence."
exit "$evaluation_status"
