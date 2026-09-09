#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="compose.quickstart.yaml"
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-vehicle-risk-agent-quickstart}"
MCP_URL="${QUICKSTART_MCP_SERVER_URL:-https://vehicle-mcp.chhlatbot.com/mcp}"

print_local_fallback() {
    cat <<'EOF'

The quickstart does not silently switch to local or fake vehicle evidence.
To use the full local stack instead, clone the sibling repositories beside this repository:

  git clone git@github.com:sovorn-c/vehicle-mcp-server.git ../vehicle-mcp-server
  git clone git@github.com:sovorn-c/nz-vehicle-data-pipeline.git ../nz-vehicle-data-pipeline
  docker compose up -d --build --wait
  bash scripts/smoke-local.sh
EOF
}

echo "Checking hosted Vehicle Intelligence MCP endpoint: ${MCP_URL}"
if ! status_code=$(curl -sS --connect-timeout 5 --max-time 10 \
    -o /dev/null -w "%{http_code}" \
    -X POST "${MCP_URL}" \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","method":"tools/list","id":1}' 2>/dev/null); then
    echo "ERROR: Hosted MCP endpoint is unreachable or timed out."
    print_local_fallback
    exit 1
fi

if [[ ! "${status_code}" =~ ^2[0-9][0-9]$ ]]; then
    echo "ERROR: Hosted MCP endpoint returned an unusable HTTP response."
    print_local_fallback
    exit 1
fi

echo "Hosted MCP endpoint reachable (HTTP ${status_code})."
docker compose -p "${COMPOSE_PROJECT}" -f "${COMPOSE_FILE}" up -d --build --wait
uv run python -m vehicle_risk_agent.cli.smoke --base-url "http://localhost:${VEHICLE_RISK_AGENT_QUICKSTART_PORT:-8001}"
