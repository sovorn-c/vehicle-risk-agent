#!/usr/bin/env bash
set -Eeuo pipefail

DB_PORT="${VEHICLE_RISK_AGENT_DB_PORT:-54329}"
if [[ -z "${VEHICLE_RISK_AGENT_DB_PORT+x}" ]] && nc -z localhost "${DB_PORT}" 2>/dev/null; then
  DB_PORT=54330
fi
export VEHICLE_RISK_AGENT_DB_PORT="${DB_PORT}"
BASE_URL="${VEHICLE_RISK_AGENT_BASE_URL:-http://localhost:8001}"

cleanup() {
  docker compose -f compose.yaml down -v --remove-orphans
}
trap cleanup EXIT

printf '%s\n' '==> Building and starting the seeded local stack'
docker compose -f compose.yaml up -d --build --wait

printf '%s\n' '==> Restarting Agent API before boundary smoke'
docker compose -f compose.yaml restart agent-api

printf '%s\n' '==> Exercising pipeline -> MCP -> Agent API -> review'
uv run python -m vehicle_risk_agent.cli.smoke --base-url "${BASE_URL}"

printf '%s\n' '==> Local stack smoke passed; teardown is handled by EXIT trap'
