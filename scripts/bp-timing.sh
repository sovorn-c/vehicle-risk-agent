#!/usr/bin/env bash
# bp-timing.sh — Record skill invocation timings for stocktake effectiveness analysis.
# Usage: bash scripts/bp-timing.sh start <skill-name>
#        bash scripts/bp-timing.sh end <skill-name>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_YAML="$REPO_ROOT/specs/state.yaml"

PYTHON="python3"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
fi

print_usage() {
  echo "Usage: $0 start|end <skill-name>" >&2
  exit 1
}

[ $# -ne 2 ] && print_usage

ACTION="$1"
SKILL="$2"

case "$ACTION" in
  start)
    STAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    $PYTHON -c "
import yaml, sys
try:
    with open('$STATE_YAML') as f:
        data = yaml.safe_load(f) or {}
except Exception:
    data = {}

metrics = data.setdefault('metrics', {})
timings = metrics.setdefault('skill_timings', {})

if '$SKILL' not in timings:
    timings['$SKILL'] = {'calls': 0, 'total_seconds': 0, 'avg_seconds': 0}

timings['$SKILL']['_start'] = '$STAMP'
with open('$STATE_YAML', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
" 2>/dev/null || echo "WARN: timing start failed" >&2
    ;;

  end)
    STAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    $PYTHON -c "
import yaml
from datetime import datetime

try:
    with open('$STATE_YAML') as f:
        data = yaml.safe_load(f) or {}
except Exception:
    data = {}

metrics = data.setdefault('metrics', {})
timings = metrics.setdefault('skill_timings', {})

if '$SKILL' in timings and '_start' in timings['$SKILL']:
    start_t = datetime.fromisoformat(timings['$SKILL']['_start'].replace('Z','+00:00'))
    end_t = datetime.fromisoformat('$STAMP'.replace('Z','+00:00'))
    elapsed = (end_t - start_t).total_seconds()
    del timings['$SKILL']['_start']
    timings['$SKILL']['calls'] = timings['$SKILL'].get('calls', 0) + 1
    timings['$SKILL']['total_seconds'] = round(timings['$SKILL'].get('total_seconds', 0) + elapsed, 2)
    calls = timings['$SKILL']['calls']
    timings['$SKILL']['avg_seconds'] = round(timings['$SKILL']['total_seconds'] / calls, 2) if calls else 0
    with open('$STATE_YAML', 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
" 2>/dev/null || echo "WARN: timing end failed" >&2
    ;;

  *)
    print_usage
    ;;
esac
