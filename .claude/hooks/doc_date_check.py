#!/usr/bin/env python3
"""Soft-warn when a Markdown doc lacks a current `Last updated: YYYY-MM-DD`.

Doc convention: every doc/README starts with `Last updated: YYYY-MM-DD`
(global CLAUDE.md "Documentation" section, also the auto-loaded doc-
conventions skill). Missing or stale stamps drift over time without
this check.

Hook contract:
- stdin: Claude Code PostToolUse payload (JSON) with tool_input.file_path.
- exit 0 always (this is a soft warn, not a blocker).
- stderr: notice when the stamp is missing or stale.
- Skips: non-`.md` files, internal/log/ (auto-dated by filename),
  internal/scratch/ (gitignored throwaway), archive/ (frozen history).
- Fails open on parse errors / missing files.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys


_DATE_LINE = re.compile(r"^Last updated:\s+(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)
_SKIP_PREFIXES = (
    "internal/log/",       # log filenames already carry the date
    "internal/scratch/",   # gitignored throwaway
    "archive/",            # frozen history
    "internal/archive/",
    ".claude/",            # Claude Code config (skills, agents, hooks)
    "coderef/",            # external repo refs, read-only
)


def _find_project_root() -> str:
    """Walk up from this file until a .git or pyproject.toml is found."""
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    while cur != os.path.dirname(cur):
        if os.path.exists(os.path.join(cur, ".git")) or os.path.exists(
            os.path.join(cur, "pyproject.toml")
        ):
            return cur
        cur = os.path.dirname(cur)
    return here


def _rel(file_path: str) -> str | None:
    root = os.environ.get("CLAUDE_PROJECT_DIR") or _find_project_root()
    try:
        rel = os.path.relpath(file_path, root)
    except ValueError:
        return None
    return None if rel.startswith("..") else rel


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or ""
    if not file_path or not file_path.endswith(".md"):
        return 0

    rel = _rel(file_path)
    if rel is None:
        return 0
    if any(rel.startswith(p) for p in _SKIP_PREFIXES):
        return 0

    if not os.path.exists(file_path):
        return 0

    try:
        # Read just the head; the stamp lives at the top.
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            head = "".join(fh.readline() for _ in range(8))
    except Exception:
        return 0

    today = datetime.date.today().isoformat()
    match = _DATE_LINE.search(head)
    if match is None:
        print(
            f"NOTICE: {rel} missing 'Last updated: YYYY-MM-DD' at top "
            f"(doc convention).",
            file=sys.stderr,
        )
    elif match.group(1) != today:
        print(
            f"NOTICE: {rel} 'Last updated: {match.group(1)}' is stale "
            f"(today is {today}).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
