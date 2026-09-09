#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="compose.quickstart.yaml"
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-vehicle-risk-agent-quickstart}"

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

compose_args=(-p "${COMPOSE_PROJECT}" -f "${COMPOSE_FILE}")
if ! compose_config=$(docker compose "${compose_args[@]}" config --format json 2>/dev/null); then
    echo "ERROR: Could not resolve the quickstart Compose configuration."
    print_local_fallback
    exit 1
fi

if ! MCP_URL=$(printf '%s' "${compose_config}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["services"]["agent-api"]["environment"]["MCP_SERVER_URL"])'); then
    echo "ERROR: Quickstart MCP URL configuration is invalid."
    print_local_fallback
    exit 1
fi

if ! MCP_URL=$(MCP_URL="${MCP_URL}" python3 -c '
import os
from urllib.parse import urlsplit, urlunsplit

parsed = urlsplit(os.environ["MCP_URL"].strip())
if parsed.scheme not in {"http", "https"} or not parsed.netloc:
    raise SystemExit(1)
if parsed.username or parsed.password:
    raise SystemExit(1)
path = parsed.path.rstrip("/") or "/mcp"
if path != "/mcp":
    raise SystemExit(1)
print(urlunsplit((parsed.scheme, parsed.netloc, "/mcp", parsed.query, "")))
' 2>/dev/null); then
    echo "ERROR: Quickstart MCP URL must be an HTTP(S) /mcp endpoint without credentials."
    print_local_fallback
    exit 1
fi

if ! API_PORT=$(printf '%s' "${compose_config}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["services"]["agent-api"]["ports"][0]["published"])'); then
    echo "ERROR: Could not resolve the quickstart Agent API port."
    print_local_fallback
    exit 1
fi

echo "Checking hosted Vehicle Intelligence MCP endpoint"
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
docker compose "${compose_args[@]}" up -d --build --wait
uv run python -m vehicle_risk_agent.cli.smoke --base-url "http://localhost:${API_PORT}"
