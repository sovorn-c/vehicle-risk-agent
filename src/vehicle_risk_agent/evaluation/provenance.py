"""Reproducible provenance helpers for evaluation artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
UNKNOWN_SOURCE_COMMIT = "unknown"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            check=True,
            cwd=_REPOSITORY_ROOT,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip().lower()
    return commit if _COMMIT_PATTERN.fullmatch(commit) else None


def resolve_source_commit() -> str:
    """Resolve the exact source revision used for an evaluation run."""
    head = _git_head()
    for variable in ("SOURCE_COMMIT", "GIT_COMMIT", "COMMIT_SHA"):
        value = os.environ.get(variable, "").strip().lower()
        if value:
            if _COMMIT_PATTERN.fullmatch(value) and value == head:
                return value
            return UNKNOWN_SOURCE_COMMIT
    return head or UNKNOWN_SOURCE_COMMIT


def is_real_source_commit(value: str) -> bool:
    """Return whether a provenance value identifies a concrete Git commit."""
    return bool(_COMMIT_PATTERN.fullmatch(value))


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one exact file."""
    return sha256_bytes(Path(path).read_bytes())
