#!/usr/bin/env python3
"""Deterministic spec-to-code coverage matrix builder.

Parses release-plan.yaml + execution-status.yaml, greps codebase for story tags,
builds oracle-tiered coverage matrix, emits JSON + markdown.

Usage: called by scripts/trace-stories.sh with positional args:
  python3 scripts/lib/trace-stories.py <repo_root> <matrix_json> <trace_md>
      <okf_dir> <strict> <mode>
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(sys.argv[1])
MATRIX_JSON = Path(sys.argv[2])
TRACE_MD = Path(sys.argv[3])
OKF_DIR = Path(sys.argv[4])
STRICT = int(sys.argv[5])
MODE = sys.argv[6]

_MIN_STORY_BASELINE = 1
_STRICT_UNIMPLEMENTED_STATUSES = frozenset({"backlog", "todo", "planned", "failing"})


def _load_yaml(path: Path) -> dict:
    """Load a YAML file safely."""
    try:
        import yaml

        with open(str(path), encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"trace-stories.py: ERROR parsing {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _extract_wsjf(val: any) -> float:
    if isinstance(val, dict):
        return float(val.get("score", 0) or 0)
    try:
        return float(val or 0)
    except (ValueError, TypeError):
        return 0.0


release = _load_yaml(ROOT / "specs" / "release-plan.yaml")
exec_status = _load_yaml(ROOT / "specs" / "execution-status.yaml")
dev_status = exec_status.get("development_status", {})

stories: dict[str, dict] = {}
epics = release.get("epics", [])
if not isinstance(epics, list):
    epics = []

for epic in epics:
    if not isinstance(epic, dict):
        continue
    eid = epic.get("id", "")
    ebcp = epic.get("bcps", 0)
    etitle = epic.get("title", "")
    ewsjf = _extract_wsjf(epic.get("wsjf", 0))
    capsule_dir = epic.get("capsule_dir", "")
    if capsule_dir:
        capsule_path = ROOT / "specs" / capsule_dir / "epic.yaml"
        if capsule_path.exists():
            cap = _load_yaml(capsule_path)
            for s in cap.get("stories", []) or []:
                if isinstance(s, dict):
                    sid = s.get("id", "")
                    stories[sid] = {
                        "id": sid,
                        "title": s.get("title", ""),
                        "epic_id": eid,
                        "epic_title": etitle,
                        "bcp": s.get("bcp", 0),
                        "wsjf": ewsjf,
                        "description": s.get("description", ""),
                    }
    file_key = epic.get("file", "")
    if file_key and not capsule_dir:
        legacy_path = ROOT / "specs" / file_key
        if legacy_path.exists():
            leg = _load_yaml(legacy_path)
            for s in leg.get("stories", []) or []:
                if isinstance(s, dict):
                    sid = s.get("id", "")
                    stories[sid] = {
                        "id": sid,
                        "title": s.get("title", ""),
                        "epic_id": eid,
                        "epic_title": etitle,
                        "bcp": s.get("bcp", 0),
                        "wsjf": ewsjf,
                        "description": s.get("description", ""),
                    }

# Fallback: discover stories directly from capsule dirs under specs/epics
if not stories:
    epics_dir = ROOT / "specs" / "epics"
    if epics_dir.exists():
        for edir in sorted(epics_dir.iterdir()):
            if not edir.is_dir() or edir.name == "archive":
                continue
            eyaml = edir / "epic.yaml"
            if eyaml.exists():
                cap = _load_yaml(eyaml)
                eid = cap.get("id", edir.name)
                etitle = cap.get("title", "")
                ewsjf = _extract_wsjf(cap.get("wsjf", 0))
                for s in cap.get("stories", []) or []:
                    if isinstance(s, dict):
                        sid = s.get("id", "")
                        stories[sid] = {
                            "id": sid,
                            "title": s.get("title", ""),
                            "epic_id": eid,
                            "epic_title": etitle,
                            "bcp": s.get("bcp", 0),
                            "wsjf": ewsjf,
                            "description": s.get("description", ""),
                        }

result = subprocess.run(
    [
        "grep",
        "-rn",
        "--include=*.md",
        "--include=*.sh",
        "--include=*.py",
        "--include=*.js",
        "--include=*.ts",
        "--include=*.yaml",
        "--include=*.yml",
        "-E",
        r"story:\s*e[0-9]{2}s[0-9]{2}",
        str(ROOT),
    ],
    capture_output=True,
    text=True,
    cwd=str(ROOT),
)

tag_inventory: list[dict] = []
tagged_sids: set[str] = set()
tag_index: dict[str, list] = {}

for line in result.stdout.splitlines():
    m = re.match(r"^(.+?):(\d+):(.*story:\s*(e\d{2}s\d{2}).*)$", line)
    if m:
        fpath = m.group(1)
        fline = int(m.group(2))
        sid = m.group(4)
        with contextlib.suppress(ValueError):
            fpath = str(Path(fpath).relative_to(ROOT))
        if not fpath.startswith("specs/archive"):
            tag_inventory.append({"file": fpath, "line": fline, "story_id": sid})
            tagged_sids.add(sid)
            tag_index.setdefault(sid, []).append({"file": fpath, "line": fline})

EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    ".cursor",
    ".gemini",
    ".pi",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".bigpowers",
}
EXCLUDE_PREFIXES = ("specs/archive/",)


def _is_excluded(rel_path: str) -> bool:
    parts = rel_path.split(os.sep)
    if any(d in EXCLUDE_DIRS for d in parts):
        return True
    return any(rel_path.startswith(p) for p in EXCLUDE_PREFIXES)


all_files = []
for root_dir, dirs, files in os.walk(str(ROOT)):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for f in files:
        rel = str(Path(root_dir, f).relative_to(ROOT))
        if not _is_excluded(rel):
            all_files.append(rel)


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text.lower())
    return re.sub(r"\s+", "-", text.strip())


def heuristic_match(story_title: str, file_path: str) -> bool:
    slug = slugify(story_title)
    words = slug.split("-")
    fname = Path(file_path).stem.lower()
    sig_words = [w for w in words if len(w) > 2]
    if len(sig_words) < 2:
        sig_words = words
    matches = sum(1 for w in sig_words if w in fname)
    return matches >= min(2, len(sig_words))


def find_task_references(story_id: str) -> list[dict]:
    refs = []
    epics_dir = ROOT / "specs" / "epics"
    if not epics_dir.exists():
        return refs
    for task_yaml in epics_dir.rglob("*tasks.yaml"):
        with contextlib.suppress(Exception):
            content = task_yaml.read_text(encoding="utf-8")
            if f"story_id: {story_id}" in content or story_id in content:
                refs.append({"file": str(task_yaml.relative_to(ROOT)), "type": "task_yaml"})
    return refs


matrix_stories = []
dark_stories = []
for sid, sinfo in sorted(stories.items()):
    links = []
    sid_status = dev_status.get(sid, "backlog")
    if sid in tag_index:
        for t in tag_index[sid]:
            links.append(
                {
                    "file": t["file"],
                    "line": t["line"],
                    "confidence": "high",
                    "method": "explicit_tag",
                }
            )
    for fpath in all_files:
        if heuristic_match(sinfo["title"], fpath) and not any(
            lnk["file"] == fpath for lnk in links
        ):
            links.append(
                {
                    "file": fpath,
                    "line": 0,
                    "confidence": "medium",
                    "method": "file_heuristic",
                }
            )
    for tr in find_task_references(sid):
        existing = {lnk["file"] for lnk in links}
        if tr["file"] not in existing:
            links.append(
                {"file": tr["file"], "line": 0, "confidence": "low", "method": "task_reference"}
            )
    matrix_stories.append(
        {
            "id": sid,
            "title": sinfo["title"],
            "epic_id": sinfo["epic_id"],
            "epic_title": sinfo["epic_title"],
            "bcp": sinfo["bcp"],
            "wsjf": sinfo["wsjf"],
            "status": sid_status,
            "links": links,
            "link_count": len(links),
        }
    )
    if len(links) == 0 and sid_status != "backlog":
        dark_stories.append(sid)

orphan_tags = [sid for sid in sorted(tagged_sids) if sid not in stories]
stale_tags = [
    sid for sid in sorted(tagged_sids) if sid in stories and dev_status.get(sid) == "done"
]

tagged_count = len(tagged_sids & set(stories.keys()))
total_count = len(stories)
cov_pct = round(tagged_count / total_count * 100, 1) if total_count else 100.0

matrix = {
    "generated_at": datetime.now(UTC).isoformat(),
    "matrix_version": "1.0",
    "stories": matrix_stories,
    "summary": {
        "total_stories": total_count,
        "tagged_stories": tagged_count,
        "coverage_percent": cov_pct,
        "dark_stories": dark_stories,
        "dark_count": len(dark_stories),
        "orphan_tags": orphan_tags,
        "orphan_count": len(orphan_tags),
        "stale_tags": stale_tags,
        "stale_count": len(stale_tags),
        "oracle_stats": {
            "high": sum(
                1 for s in matrix_stories for lnk in s["links"] if lnk["confidence"] == "high"
            ),
            "medium": sum(
                1 for s in matrix_stories for lnk in s["links"] if lnk["confidence"] == "medium"
            ),
            "low": sum(
                1 for s in matrix_stories for lnk in s["links"] if lnk["confidence"] == "low"
            ),
        },
    },
}
MATRIX_JSON.parent.mkdir(parents=True, exist_ok=True)
MATRIX_JSON.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

lines = [
    "# Traceability Matrix",
    "",
    f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    f"**Total stories:** {total_count}",
    f"**Tagged stories:** {tagged_count}",
    f"**Coverage:** {cov_pct}%",
    f"**Dark stories:** {len(dark_stories)}",
    f"**Orphan tags:** {len(orphan_tags)}",
    f"**Stale tags:** {len(stale_tags)}",
    "",
    "| Story | Title | Status | Links | Oracle Tier |",
    "|---|---|---|---|---|",
]
for s in matrix_stories:
    confs = {lnk["confidence"] for lnk in s["links"]}
    mc = (
        "high"
        if "high" in confs
        else ("medium" if "medium" in confs else ("low" if confs else "none"))
    )
    lines.append(f"| {s['id']} | {s['title'][:50]} | {s['status']} | {s['link_count']} | {mc} |")

TRACE_MD.parent.mkdir(parents=True, exist_ok=True)
TRACE_MD.write_text("\n".join(lines), encoding="utf-8")

if OKF_DIR:
    with contextlib.suppress(Exception):
        OKF_DIR.mkdir(parents=True, exist_ok=True)
        idx_lines = [
            "# Story Index",
            "",
            "| Story | Title | Oracle Tier | Links |",
            "|---|---|---|---|",
        ]
        for s in matrix_stories:
            confs = {lnk["confidence"] for lnk in s["links"]}
            mc = (
                "high"
                if "high" in confs
                else ("medium" if "medium" in confs else ("low" if confs else "none"))
            )
            idx_lines.append(
                f"| [{s['id']}](./{s['id']}.md) | {s['title'][:60]} | {mc} | {s['link_count']} |"
            )
        idx_lines.append("")
        (OKF_DIR / "index.md").write_text("\n".join(idx_lines), encoding="utf-8")

if MODE != "json":
    print("\n".join(lines))
else:
    print(f"Emitted {MATRIX_JSON} ({cov_pct}% coverage)")

if STRICT and len(stories) < _MIN_STORY_BASELINE:
    print(
        f"trace-stories.py: STRICT FAIL — story count {len(stories)} below {_MIN_STORY_BASELINE}",
        file=sys.stderr,
    )
    sys.exit(2)

sys.exit(0)
