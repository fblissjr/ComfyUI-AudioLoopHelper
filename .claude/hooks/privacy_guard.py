#!/usr/bin/env python3
# path-privacy: skip-file — this hook's regex patterns + docstring examples necessarily mention the path shapes it detects
"""Block Write|Edit on public files that would leak private paths or username.

Enforces the "Privacy" rule from global CLAUDE.md: never write private absolute
paths or usernames into any publishable file (README, CHANGELOG, docs/,
example_workflows/, nodes*.py, scripts/, tests/, pyproject.toml, __init__.py,
conftest.py).

The actual leak patterns are loaded from `.claude/privacy_patterns.local.json`
(gitignored) so this hook file itself stays public-clean. The config holds the
literal patterns; this file holds only the matching machinery + a tiny safe
default for repos that haven't installed a config yet.

Files under internal/, coderef/, .claude/, .venv/, .pytest_cache/,
__pycache__/, and profile_output/ are exempt — they're gitignored or
not intended for publication.

Hook contract:
- stdin: Claude Code hook payload (JSON) with tool_input.file_path plus
  either content (Write) or new_string (Edit).
- stdout: diagnostic on match.
- exit 0: allow. exit 2: block (stderr shown to user + Claude).
- Fails open on unexpected errors (missing keys, malformed JSON, missing
  config) so the guard never wedges the session.
"""
from __future__ import annotations

# stdlib `json` (not orjson) is intentional — hooks must run on a fresh clone
# before `uv sync`, so they can't depend on installed packages. Same rationale
# applies to .claude/hooks/doc_date_check.py.
import json
import os
import re
import sys


# Files where a leak would be public. Relative to project root.
_PUBLIC_FILE_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"^README(\.md)?$",
        r"^CHANGELOG\.md$",
        r"^LICENSE$",
        r"^docs/",
        r"^example_workflows/",
        r"^nodes\.py$",
        r"^nodes_analysis\.py$",
        r"^scripts/",
        r"^tests/",
        r"^pyproject\.toml$",
        r"^__init__\.py$",
        r"^conftest\.py$",
        r"^update-coderef\.sh$",
    )
)

# Fallback when no config is installed. Generic absolute-home-path catch-all
# only — won't catch tilde forms or specific usernames. Real protection comes
# from .claude/privacy_patterns.local.json.
_FALLBACK_PATTERNS = (re.compile(r"/home/[a-z][a-z0-9_-]*/"),)


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


def _rel(file_path: str, root: str) -> str | None:
    """Return project-relative path or None if outside project."""
    try:
        rel = os.path.relpath(file_path, root)
    except ValueError:
        return None
    return None if rel.startswith("..") else rel


def _is_public(rel_path: str) -> bool:
    return any(p.match(rel_path) for p in _PUBLIC_FILE_PATTERNS)


def _load_patterns(root: str) -> tuple[tuple[re.Pattern, ...], dict[str, str]]:
    """Load leak patterns from .claude/privacy_patterns.local.json.

    Returns (compiled_patterns, replacements_by_pattern). Falls back to the
    minimal generic-home regex if the config is missing or malformed.
    """
    cfg_path = os.path.join(root, ".claude", "privacy_patterns.local.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        compiled: list[re.Pattern] = []
        replacements: dict[str, str] = {}
        for entry in cfg.get("leak_patterns", []):
            pattern = entry.get("pattern")
            if not pattern:
                continue
            try:
                compiled.append(re.compile(pattern))
                replacements[pattern] = entry.get("replacement", "(remove)")
            except re.error:
                continue
        if compiled:
            return tuple(compiled), replacements
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return _FALLBACK_PATTERNS, {}


def _scan(blob: str, patterns: tuple[re.Pattern, ...]) -> list[tuple[str, str]]:
    """Return list of (matched_text, source_pattern_string).

    Uses finditer so files with multiple distinct matches against the same
    regex (e.g. "/home/<user>" alone AND "/home/<user>/ComfyUI/foo") report
    every leak — single-match search would understate scope and let a second
    leak slip past a copy-paste fix.
    """
    hits: list[tuple[str, str]] = []
    for p in patterns:
        for m in p.finditer(blob):
            hits.append((m.group(0), p.pattern))
    return hits


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # fail open on parse errors

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or ""
    if not file_path:
        return 0

    root = os.environ.get("CLAUDE_PROJECT_DIR") or _find_project_root()
    rel = _rel(file_path, root)
    if rel is None or not _is_public(rel):
        return 0

    texts: list[str] = []
    for key in ("content", "new_string"):
        v = tool_input.get(key)
        if isinstance(v, str):
            texts.append(v)
    if not texts:
        return 0

    patterns, replacements = _load_patterns(root)
    hits = _scan("\n".join(texts), patterns)
    if not hits:
        return 0

    # Deduplicate by matched text, preserve replacement hints.
    by_match: dict[str, str] = {}
    for matched, pattern_str in hits:
        if matched not in by_match:
            by_match[matched] = replacements.get(pattern_str, "(remove)")

    lines = [
        "Blocked: privacy leak in public file.",
        f"  file:    {rel}",
        "  matched:",
    ]
    for matched, suggestion in sorted(by_match.items()):
        lines.append(f"    {matched!r} → {suggestion}")
    lines.extend(
        [
            "  rule:    global CLAUDE.md 'Privacy' section forbids private",
            "           absolute paths and usernames in public files.",
            "  config:  .claude/privacy_patterns.local.json (gitignored)",
            "  bypass:  write to internal/ (gitignored) or update the config.",
        ]
    )
    print("\n".join(lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
