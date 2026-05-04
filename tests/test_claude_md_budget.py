"""Guard CLAUDE.md hygiene: size budget, pointer-target integrity, orphans.

Three assertions:

1. Root CLAUDE.md is bounded (200 lines / 30 KB hard cap). Subtree CLAUDE.md
   files soft-warn at 500 lines.
2. Every repo-relative path mentioned in any CLAUDE.md (docs/X.md,
   scripts/X.py, internal/X.md, tests/X.py) resolves to an existing file.
   Catches stale pointers — the main failure mode of pointer discipline.
3. Every atomic note in docs/reference/ is referenced from at least one of
   {root CLAUDE.md, subtree CLAUDE.md files, docs/README.md, another doc}.
   Catches orphan reference notes — Karpathy-wiki "lint mode".

Policy lives in .claude/CLAUDE.md "CLAUDE.md governance" section.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

ROOT_CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
ROOT_LINE_BUDGET = 200
ROOT_BYTE_BUDGET = 30 * 1024  # 30 KB
SUBTREE_LINE_SOFT_WARN = 500


def _all_claude_md_files() -> list[Path]:
    """All committed CLAUDE.md files, excluding gitignored / vendor trees."""
    excluded = {"coderef", ".git", "node_modules", "data", "models"}
    out: list[Path] = []
    for path in REPO_ROOT.rglob("CLAUDE.md"):
        rel = path.relative_to(REPO_ROOT)
        if any(part in excluded for part in rel.parts):
            continue
        out.append(path)
    return sorted(out)


def _extract_repo_relative_paths(text: str) -> set[str]:
    """Pull out anything that looks like a repo-relative path.

    Matches: foo/bar.md, foo/bar.py, foo/bar/baz.json (one or more / segments
    with a known extension). Excludes URLs and absolute paths (other tests
    catch path-privacy leaks).

    Filters out documentation-placeholder shapes that can't be resolved:
    `apply_X.py`, `docs/reference/X.md`, `foo/.../bar.py` truncations.
    """
    pattern = re.compile(
        r"(?<![/:.\w])"
        r"((?:tests|scripts|docs|internal|nodes|example_workflows|coderef|\.claude|\.github)/[\w./-]+\.(?:md|py|sh|json|jsonl|toml|yaml|yml|txt))"
    )
    out: set[str] = set()
    for match in pattern.finditer(text):
        path = match.group(1)
        # Filter placeholders: any segment is the literal "X" or contains "..."
        if any(seg == "X" or "..." in seg for seg in path.split("/")):
            continue
        # Filter trailing placeholder shapes like apply_X.py, foo_X.md
        leaf = path.rsplit("/", 1)[-1]
        stem = leaf.rsplit(".", 1)[0]
        if stem.endswith("_X") or stem == "X":
            continue
        out.add(path)
    return out


def test_root_claude_md_within_budget() -> None:
    """Hard cap: 200 lines AND 30 KB on root CLAUDE.md."""
    text = ROOT_CLAUDE_MD.read_text()
    line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
    byte_count = len(text.encode("utf-8"))

    failures: list[str] = []
    if line_count > ROOT_LINE_BUDGET:
        failures.append(
            f"root CLAUDE.md = {line_count} lines (budget {ROOT_LINE_BUDGET}). "
            f"Compress or move rules to subtree CLAUDE.md / docs/."
        )
    if byte_count > ROOT_BYTE_BUDGET:
        failures.append(
            f"root CLAUDE.md = {byte_count} bytes (budget {ROOT_BYTE_BUDGET}). "
            f"Compress or move rules to subtree CLAUDE.md / docs/."
        )
    if failures:
        pytest.fail("\n".join(failures))


def test_subtree_claude_md_soft_budget() -> None:
    """Soft warn: subtree CLAUDE.md > 500 lines. Fails the test, but the fix
    can be either compression OR splitting further."""
    failures: list[str] = []
    for path in _all_claude_md_files():
        if path == ROOT_CLAUDE_MD:
            continue
        text = path.read_text()
        line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
        if line_count > SUBTREE_LINE_SOFT_WARN:
            rel = path.relative_to(REPO_ROOT)
            failures.append(
                f"{rel} = {line_count} lines (soft cap {SUBTREE_LINE_SOFT_WARN}). "
                f"Compress or split."
            )
    if failures:
        pytest.fail("\n".join(failures))


def test_pointer_targets_exist() -> None:
    """Every repo-relative path in any CLAUDE.md must resolve to an existing
    file. Catches stale pointers when files move/rename.

    Gitignored prefixes (internal/, coderef/, data/) are exempted: their
    targets legitimately don't exist on a public clone.
    """
    gitignored_prefixes = ("internal/", "coderef/", "data/")
    misses: list[str] = []
    for claude_md in _all_claude_md_files():
        text = claude_md.read_text()
        for relpath in _extract_repo_relative_paths(text):
            if relpath.startswith(gitignored_prefixes):
                continue
            target = REPO_ROOT / relpath
            if not target.exists():
                rel_md = claude_md.relative_to(REPO_ROOT)
                misses.append(f"{rel_md} -> {relpath} (does not exist)")
    if misses:
        pytest.fail(
            "Stale pointers in CLAUDE.md files:\n  "
            + "\n  ".join(sorted(misses))
        )


def test_no_orphan_reference_notes() -> None:
    """Every docs/reference/*.md must be cited from at least one of:
    CLAUDE.md files, docs/README.md, or another docs/ file. Karpathy-wiki
    lint pass — orphan notes mean broken navigation."""
    ref_dir = REPO_ROOT / "docs" / "reference"
    if not ref_dir.is_dir():
        return

    citation_corpus_paths: list[Path] = list(_all_claude_md_files())
    docs_readme = REPO_ROOT / "docs" / "README.md"
    if docs_readme.exists():
        citation_corpus_paths.append(docs_readme)
    for doc in (REPO_ROOT / "docs").rglob("*.md"):
        if doc == docs_readme:
            continue
        if doc.is_relative_to(ref_dir):
            citation_corpus_paths.append(doc)
        else:
            citation_corpus_paths.append(doc)

    corpus = "\n".join(p.read_text() for p in citation_corpus_paths if p.exists())

    orphans: list[str] = []
    for note in ref_dir.glob("*.md"):
        rel = note.relative_to(REPO_ROOT).as_posix()
        bare = note.name
        if rel not in corpus and bare not in corpus:
            orphans.append(rel)
    if orphans:
        pytest.fail(
            "Orphan reference notes (not cited from CLAUDE.md / docs/README.md "
            "/ other docs):\n  " + "\n  ".join(sorted(orphans))
        )


def test_extract_repo_relative_paths_picks_up_common_shapes() -> None:
    text = """
    See docs/reference/frame_planner_reference.md for details.
    Run scripts/audit_workflows.py to validate.
    Internal: internal/postmortem_concat_av_latent_investigation.md
    Not a path: foo.md (no leading subdir)
    Not a path: https://example.com/foo.md (URL)
    Placeholder: scripts/apply_X.py (X-as-template-marker)
    Placeholder: docs/reference/X.md
    Truncated: coderef/LTX-2/.../distilled.py
    """
    paths = _extract_repo_relative_paths(text)
    assert "docs/reference/frame_planner_reference.md" in paths
    assert "scripts/audit_workflows.py" in paths
    assert "internal/postmortem_concat_av_latent_investigation.md" in paths
    assert "foo.md" not in paths
    assert "example.com/foo.md" not in paths
    assert "scripts/apply_X.py" not in paths
    assert "docs/reference/X.md" not in paths
    assert "coderef/LTX-2/.../distilled.py" not in paths
