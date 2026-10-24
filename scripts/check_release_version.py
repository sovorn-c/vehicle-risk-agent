"""Fail unless a release tag, pyproject version, and package __version__ agree."""

import re
import sys
import tomllib
from pathlib import Path

from vehicle_risk_agent import __version__

TAG_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("ERROR: expected exactly one argument: the release tag", file=sys.stderr)
        return 1
    tag = argv[0]
    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        print(f"ERROR: tag {tag!r} must match v<MAJOR>.<MINOR>.<PATCH>", file=sys.stderr)
        return 1
    tag_version = ".".join(match.groups())
    with PYPROJECT.open("rb") as handle:
        project_version = tomllib.load(handle)["project"]["version"]
    if not tag_version == project_version == __version__:
        print(
            f"ERROR: version mismatch: tag {tag_version}, "
            f"pyproject.toml {project_version}, __version__ {__version__}",
            file=sys.stderr,
        )
        return 1
    print(f"release version OK: {tag_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
