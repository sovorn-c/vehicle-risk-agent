#!/usr/bin/env python3
"""TEA-inspired heuristic blind-spot detector (Python engine).

Reads execution-status.yaml + traceability-matrix.json, runs 7 structural
quality checks, and emits specs/blind-spots.json.

Usage: called by scripts/check-blind-spots.sh with positional args:
  python3 scripts/lib/blind-spots.py <repo_root> <blind_spots_json>
      <exec_status_yaml> <matrix_json> <verifications_dir> <epics_dir>
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(sys.argv[1])
BLIND_SPOTS_JSON = Path(sys.argv[2])
EXEC_STATUS = Path(sys.argv[3])
MATRIX_JSON = Path(sys.argv[4])
VERIFICATIONS_DIR = Path(sys.argv[5])
EPICS_DIR = Path(sys.argv[6])

findings: list[dict] = []


def parse_exec_status(path: Path) -> dict[str, str]:
    status: dict[str, str] = {}
    if not path.exists():
        return status
    with contextlib.suppress(Exception):
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        dev_stat = data.get("development_status", {})
        if isinstance(dev_stat, dict):
            return {str(k): str(v) for k, v in dev_stat.items()}

    # Fallback to line parser
    text = path.read_text(encoding="utf-8")
    in_dev = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("development_status:"):
            in_dev = True
            continue
        if in_dev:
            if s and not line.startswith(" ") and not line.startswith("\t"):
                break
            if ":" in s and not s.startswith("#"):
                k, _, v = s.partition(":")
                status[k.strip()] = v.strip().strip('"').strip("'")
    return status


dev_status = parse_exec_status(EXEC_STATUS)

matrix_data: dict | None = None
if MATRIX_JSON.exists():
    with contextlib.suppress(Exception):
        matrix_data = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))

tagged_files: dict[str, list[str]] = {}
file_story_counts: dict[str, set[str]] = {}
if matrix_data:
    for s in matrix_data.get("stories", []):
        sid = s.get("id")
        if not sid:
            continue
        for link in s.get("links", []):
            f = link.get("file")
            if f:
                tagged_files.setdefault(sid, []).append(f)
                file_story_counts.setdefault(f, set()).add(sid)

CODE_EXTENSIONS = {
    ".sh",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".md",
    ".yaml",
    ".yml",
    ".cjs",
    ".mjs",
    ".json",
}
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".cursor",
    ".gemini",
    ".pi",
    "specs",
    "dist",
    "build",
    "__pycache__",
    ".bigpowers",
    ".venv",
}


def is_code_file(p: Path) -> bool:
    return p.suffix in CODE_EXTENSIONS and not any(s in p.parts for s in SKIP_DIRS)


all_code_files: list[str] = []
for root, dirs, files in os.walk(str(ROOT)):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for fname in files:
        fpath = Path(root) / fname
        if is_code_file(fpath):
            all_code_files.append(str(fpath.relative_to(ROOT)))

TEST_PATTERNS = [
    re.compile(r"(^|/)test(s)?/", re.IGNORECASE),
    re.compile(r"(^|/)__tests__/", re.IGNORECASE),
    re.compile(r"\.test\.", re.IGNORECASE),
    re.compile(r"\.spec\.", re.IGNORECASE),
    re.compile(r"_test\.", re.IGNORECASE),
    re.compile(r"test_", re.IGNORECASE),
]


def has_test_files(code_files: list[str]) -> bool:
    for cf in code_files:
        cf_stem = Path(cf).stem
        for pat in TEST_PATTERNS:
            if pat.search(cf):
                return True
        for tf in all_code_files:
            for pat in TEST_PATTERNS:
                if pat.search(tf) and cf_stem in Path(tf).name:
                    return True
    return False


# Check A: verify-gap (stories explicitly marked done)
for sid, sstatus in dev_status.items():
    if sstatus == "done":
        verify_file = VERIFICATIONS_DIR / f"{sid}-verify.yaml"
        if not verify_file.exists():
            findings.append(
                {
                    "check": "verify-gap",
                    "story_id": sid,
                    "severity": "HIGH",
                    "description": (
                        f"Story {sid} is marked done but no verification evidence exists at "
                        f"specs/verifications/{sid}-verify.yaml"
                    ),
                    "remediation": f"Run verify-work for {sid} or create the evidence file",
                }
            )

# Check B: test-gap
for sid, files in tagged_files.items():
    if dev_status.get(sid) in ("done", "active", "passing") and not has_test_files(files):
        code_only = [
            f
            for f in files
            if Path(f).suffix in {".sh", ".py", ".js", ".ts", ".jsx", ".tsx"}
            and "SKILL.md" not in f
        ]
        if code_only:
            findings.append(
                {
                    "check": "test-gap",
                    "story_id": sid,
                    "severity": "MEDIUM",
                    "description": (
                        f"Story {sid} has {len(code_only)} tagged code file(s) with no tests"
                    ),
                    "remediation": f"Add tests for: {', '.join(code_only[:3])}",
                }
            )

# Check C: epic-orphan
if EPICS_DIR.exists():
    for epic_dir in sorted(EPICS_DIR.iterdir()):
        if not epic_dir.is_dir() or epic_dir.name == "archive":
            continue
        for task_file in sorted(epic_dir.glob("e*[0-9]-tasks.yaml")):
            sid_match = re.match(r"(e\d+s\d+)", task_file.name)
            if not sid_match:
                continue
            sid = sid_match.group(1)
            sstatus = dev_status.get(sid, "backlog")
            if sstatus in ("done", "active") and sid not in tagged_files:
                findings.append(
                    {
                        "check": "epic-orphan",
                        "story_id": sid,
                        "severity": "LOW",
                        "description": (f"Story {sid} has capsule tasks but no story tags in code"),
                        "remediation": f"Add `// story: {sid}` tags to implementing files",
                    }
                )

# Check D: stale-tag
stale_from_matrix: set[str] = set()
if matrix_data:
    stale_from_matrix = set(matrix_data.get("summary", {}).get("stale_tags", []))

for sid in stale_from_matrix:
    findings.append(
        {
            "check": "stale-tag",
            "story_id": sid,
            "severity": "LOW",
            "description": f"Story {sid} is marked done but still has story tags in code",
            "remediation": f"Remove `// story: {sid}` tags or confirm the story status",
        }
    )

# Check E: double-tag
for fpath, sids in file_story_counts.items():
    if len(sids) > 1:
        findings.append(
            {
                "check": "double-tag",
                "file": fpath,
                "severity": "MEDIUM",
                "description": f"File '{fpath}' is tagged with multiple: {', '.join(sorted(sids))}",
                "remediation": "Review whether this file genuinely implements multiple stories.",
            }
        )

# Check F: bootstrap-testless
for sid, sstat in sorted(dev_status.items()):
    if sstat in ("active",) and sid in tagged_files:
        code_files = tagged_files[sid]
        code_only = [
            f
            for f in code_files
            if Path(f).suffix in {".sh", ".py", ".js", ".ts", ".jsx", ".tsx"}
            and "SKILL.md" not in f
        ]
        if code_only and not has_test_files(code_files):
            findings.append(
                {
                    "check": "bootstrap-testless",
                    "story_id": sid,
                    "severity": "HIGH",
                    "description": (
                        f"Active story {sid} has {len(code_only)} code files but no tests"
                    ),
                    "remediation": f"Add test files for: {', '.join(code_only[:3])}",
                }
            )

# Check G: sc-gap
TECH_ARCH_DIR = ROOT / "specs" / "tech-architecture"
if EPICS_DIR.exists():
    for epic_dir in sorted(EPICS_DIR.iterdir()):
        if not epic_dir.is_dir() or epic_dir.name == "archive":
            continue
        epic_match = re.match(r"(e\d+)", epic_dir.name)
        if not epic_match:
            continue
        epic_id = epic_match.group(1)

        test_plan = TECH_ARCH_DIR / f"{epic_id}-TEST_PLAN_LATEST.md"
        if not test_plan.exists():
            continue

        for task_file in sorted(epic_dir.glob("e*s[0-9]*-tasks.yaml")):
            sid_match = re.match(r"(e\d+s\d+)", task_file.name)
            if not sid_match:
                continue
            sid = sid_match.group(1)

            content = task_file.read_text(encoding="utf-8")
            if "risk: P0" in content:
                found_sc = False
                for f in tagged_files.get(sid, []):
                    fpath = ROOT / f
                    if fpath.exists():
                        fcontent = fpath.read_text(encoding="utf-8", errors="ignore")
                        if re.search(rf"SC-{sid}-P0", fcontent):
                            found_sc = True
                            break
                if not found_sc:
                    findings.append(
                        {
                            "check": "sc-gap",
                            "story_id": sid,
                            "severity": "MEDIUM",
                            "description": (
                                f"P0 story {sid} has a TEST_PLAN but no scenario tags in code"
                            ),
                            "remediation": f"Add `// scenario: SC-{sid}-P0-XX` tags to tests",
                        }
                    )

# Deduplicate
seen: set[tuple] = set()
deduped: list[dict] = []
for f in findings:
    key = (f["check"], f.get("story_id", f.get("file", "")))
    if key not in seen:
        seen.add(key)
        deduped.append(f)
findings = deduped

# Emit blind-spots.json
high_count = sum(1 for f in findings if f["severity"] == "HIGH")
medium_count = sum(1 for f in findings if f["severity"] == "MEDIUM")
low_count = sum(1 for f in findings if f["severity"] == "LOW")

blind_spots = {
    "generated_at": datetime.now(UTC).isoformat(),
    "version": "1.0",
    "summary": {
        "total_findings": len(findings),
        "high": high_count,
        "medium": medium_count,
        "low": low_count,
    },
    "findings": findings,
}

BLIND_SPOTS_JSON.parent.mkdir(parents=True, exist_ok=True)
BLIND_SPOTS_JSON.write_text(json.dumps(blind_spots, indent=2), encoding="utf-8")

print(
    f"check-blind-spots.sh: {len(findings)} findings "
    f"({high_count} H, {medium_count} M, {low_count} L)"
)
for f in findings:
    print(f"  [{f['severity']}] {f['check']}: {f['description'][:100]}")

if high_count > 0:
    print(f"\n⚠️  {high_count} HIGH-severity finding(s) — review specs/blind-spots.json")
    sys.exit(1)

sys.exit(0)
