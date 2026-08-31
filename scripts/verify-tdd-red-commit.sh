#!/usr/bin/env bash
# story: e80s01 e80s03
# Mechanical gate: test-only (RED) commit must fail when checked out in isolation.
# Usage:
#   bash scripts/verify-tdd-red-commit.sh [--self-test]
#   TDD_VERIFY_CMD='uv run pytest tests/test_config.py' bash scripts/verify-tdd-red-commit.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

self_test() {
  echo "verify-tdd-red-commit: self-test OK"
  return 0
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: not a git repository"
  exit 1
fi

if [[ "$(git rev-list --count HEAD 2>/dev/null || echo 0)" -lt 2 ]]; then
  echo "SKIP: need at least 2 commits (test-only RED then fix GREEN)"
  exit 0
fi

VERIFY_CMD="${TDD_VERIFY_CMD:-}"
if [[ -z "$VERIFY_CMD" ]]; then
  if [[ -f pyproject.toml ]]; then
    VERIFY_CMD="uv run pytest"
  elif [[ -f package.json ]] && grep -q '"test"' package.json 2>/dev/null; then
    VERIFY_CMD="npm test --if-present"
  elif [[ -f Cargo.toml ]]; then
    VERIFY_CMD="cargo test"
  else
    echo "SKIP: set TDD_VERIFY_CMD for this repo"
    exit 0
  fi
fi

RED_SHA=$(git rev-parse HEAD~1)
WORKTREE=$(mktemp -d)
trap 'git worktree remove -f "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"' EXIT

git worktree add -q --detach "$WORKTREE" "$RED_SHA"
(
  cd "$WORKTREE"
  if bash -c "$VERIFY_CMD"; then
    echo "FAIL: test-only commit $RED_SHA passed in isolation — RED gate violated"
    exit 1
  fi
  echo "PASS: test-only commit $RED_SHA fails in isolation"
)
