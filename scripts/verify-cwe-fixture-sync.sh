#!/usr/bin/env bash
# story: e80s05
# Assert security-review's CWE rule table and its fixtures/ directory agree.
# Usage:
#   bash scripts/verify-cwe-fixture-sync.sh [--self-test]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_MD="$REPO_ROOT/skills/security-review/SKILL.md"
FIXTURES="$REPO_ROOT/skills/security-review/fixtures"

if [ ! -f "$SKILL_MD" ]; then
  for candidate in \
    "${HOME}/.gemini/config/skills/security-review/SKILL.md" \
    "${HOME}/.gemini/bigpowers/skills/security-review/SKILL.md" \
    "${HOME}/.pi/agent/npm/node_modules/bigpowers/skills/security-review/SKILL.md"; do
    if [ -f "$candidate" ]; then
      SKILL_MD="$candidate"
      FIXTURES="$(dirname "$candidate")/fixtures"
      break
    fi
  done
fi

fail=0
note() { echo "  $1"; }

check_sync() {
  local skill_md="$1" fixtures="$2"
  local rc=0

  [ -f "$skill_md" ] || { echo "FAIL: missing $skill_md"; return 1; }
  [ -d "$fixtures" ] || { echo "FAIL: missing $fixtures"; return 1; }

  local disk_classes
  disk_classes=$(find "$fixtures" -name '*-positive.*' -exec basename {} \; 2>/dev/null \
    | sed 's/-positive\..*$//' | sort -u)

  local c
  for c in $disk_classes; do
    if ! grep -q "$c" "$skill_md"; then
      note "FAIL: fixture class '$c' has no row in the CWE rule table"
      rc=1
    fi
  done

  local ref
  for ref in $(grep -oE 'fixtures/[A-Za-z0-9._-]+' "$skill_md" | sed 's|fixtures/||' | sort -u); do
    if [ ! -f "$fixtures/$ref" ]; then
      note "FAIL: table references fixtures/$ref which does not exist"
      rc=1
    fi
  done

  for c in $disk_classes; do
    local pos neg
    pos=$(find "$fixtures" -name "$c-positive.*" | head -1)
    neg=$(find "$fixtures" -name "$c-negative.*" | head -1)
    if [ -z "$pos" ] || [ -z "$neg" ]; then
      note "FAIL: class '$c' needs BOTH a -positive and a -negative fixture"
      rc=1
    fi
  done

  return $rc
}

if [ "${1:-}" = "--self-test" ]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  mkdir -p "$tmp/fixtures"
  printf '| Rule | CWE | Positive fixture | Negative fixture |\n' > "$tmp/SKILL.md"
  : > "$tmp/fixtures/CWE-999-unlisted-positive.py"
  : > "$tmp/fixtures/CWE-999-unlisted-negative.py"

  if check_sync "$tmp/SKILL.md" "$tmp/fixtures" >/dev/null 2>&1; then
    echo "FAIL: self-test — gate passed on an unlisted fixture class (it cannot fail)"
    exit 1
  fi
  echo "PASS: self-test — gate rejects a fixture class missing from the table"
  exit 0
fi

if check_sync "$SKILL_MD" "$FIXTURES"; then
  echo "PASS: CWE rule table and fixtures/ are in sync"
  exit 0
fi
echo "FAIL: security-review CWE table and fixtures/ have drifted"
exit 1
