#!/usr/bin/env bash
# story: e38s04
# check-blind-spots.sh — TEA-inspired heuristic blind-spot detector.
# Reads execution-status.yaml + traceability-matrix.json, runs 7 structural
# quality checks, and emits specs/blind-spots.json.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLIND_SPOTS_JSON="$REPO_ROOT/specs/blind-spots.json"
EXEC_STATUS="$REPO_ROOT/specs/execution-status.yaml"
MATRIX_JSON="$REPO_ROOT/specs/traceability-matrix.json"
VERIFICATIONS_DIR="$REPO_ROOT/specs/verifications"
EPICS_DIR="$REPO_ROOT/specs/epics"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat <<'USAGE'
Usage: check-blind-spots.sh [flags]

Run 7 heuristic blind-spot checks on the codebase.

Flags:
  --help   Print this message and exit.

Checks: verify-gap, test-gap, epic-orphan, stale-tag, double-tag, bootstrap-testless, sc-gap.
Output: specs/blind-spots.json
Exit: 0 = no HIGH findings, 1 = HIGH findings or input files missing.
USAGE
      exit 0
      ;;
    *)
      echo "check-blind-spots.sh: unknown flag: $1" >&2
      echo "Try --help for usage." >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$EXEC_STATUS" ]]; then
  echo "check-blind-spots.sh: execution-status.yaml: not found at $EXEC_STATUS" >&2
  exit 1
fi

PYTHON="python3"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
fi

exec "$PYTHON" "$REPO_ROOT/scripts/lib/blind-spots.py" \
  "$REPO_ROOT" "$BLIND_SPOTS_JSON" "$EXEC_STATUS" "$MATRIX_JSON" "$VERIFICATIONS_DIR" "$EPICS_DIR"
