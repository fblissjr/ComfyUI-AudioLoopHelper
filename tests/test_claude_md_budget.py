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


def _fail_if_any(failures: list[str], header: str) -> None:
    if failures:
        pytest.fail(f"{header}\n  " + "\n  ".join(failures))


def test_root_claude_md_within_budget() -> None:
    """Hard cap: 200 lines AND 30 KB on root CLAUDE.md."""
    text = ROOT_CLAUDE_MD.read_text()
    line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
    byte_count = len(text.encode("utf-8"))

    failures: list[str] = []
    if line_count > ROOT_LINE_BUDGET:
        failures.append(
            f"{line_count} lines (budget {ROOT_LINE_BUDGET})"
        )
    if byte_count > ROOT_BYTE_BUDGET:
        failures.append(
            f"{byte_count} bytes (budget {ROOT_BYTE_BUDGET})"
        )
    _fail_if_any(
        failures,
        "root CLAUDE.md over budget — compress or move rules to subtree CLAUDE.md / docs/:",
    )


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
            failures.append(f"{rel} = {line_count} lines")
    _fail_if_any(
        failures,
        f"Subtree CLAUDE.md over soft cap {SUBTREE_LINE_SOFT_WARN} — compress or split:",
    )


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
                misses.append(f"{rel_md} -> {relpath}")
    _fail_if_any(sorted(misses), "Stale pointers in CLAUDE.md files:")


def test_no_orphan_reference_notes() -> None:
    """Every docs/reference/*.md must be cited from at least one of:
    CLAUDE.md files, docs/README.md, or another docs/ file. Karpathy-wiki
    lint pass — orphan notes mean broken navigation."""
    ref_dir = REPO_ROOT / "docs" / "reference"
    if not ref_dir.is_dir():
        return

    citation_corpus_paths = list(_all_claude_md_files())
    citation_corpus_paths.extend((REPO_ROOT / "docs").rglob("*.md"))
    corpus = "\n".join(p.read_text() for p in citation_corpus_paths if p.exists())

    orphans: list[str] = []
    for note in ref_dir.glob("*.md"):
        rel = note.relative_to(REPO_ROOT).as_posix()
        # A note can match itself (its own filename appears in its body) — to
        # count as cited, the citation must come from another file. Strip the
        # note's own contents from the corpus before checking.
        cited_corpus = corpus.replace(note.read_text(), "", 1)
        if rel not in cited_corpus and note.name not in cited_corpus:
            orphans.append(rel)
    _fail_if_any(
        sorted(orphans),
        "Orphan reference notes (not cited from CLAUDE.md / docs/README.md / other docs):",
    )


_INTERNAL_PATH_PATTERN = re.compile(r"`(internal/[\w/.-]+\.md)`")
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def test_internal_citations_marked() -> None:
    """Specific-file citations to internal/X.md from public docs must be
    marked with '(private clone only)' or 'gitignored' on the same line,
    OR live inside an HTML comment block. Catches the dead-link UX issue
    for public-clone readers who follow a citation and find nothing.

    Skipped: meta-references with glob/brace shapes (`internal/postmortem_*.md`,
    `internal/{design,analysis}/X.md`) — only specific .md filenames trigger.
    """
    public_paths: list[Path] = []
    public_paths.extend(_all_claude_md_files())
    # docs/reference/ holds the wiki notes the marker convention targets;
    # docs/guides/ and docs/analysis/ are pre-existing prose with their own
    # citation conventions — skip until they're touched organically.
    ref_dir = REPO_ROOT / "docs" / "reference"
    if ref_dir.is_dir():
        public_paths.extend(ref_dir.glob("*.md"))

    misses: list[str] = []
    for path in public_paths:
        rel = path.relative_to(REPO_ROOT)
        if rel.parts and rel.parts[0] == "internal":
            continue
        text = _HTML_COMMENT_PATTERN.sub("", path.read_text())
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in _INTERNAL_PATH_PATTERN.finditer(line):
                cited = m.group(1)
                if "*" in cited or "{" in cited:
                    continue
                lower = line.lower()
                if "private clone" in lower or "gitignored" in lower:
                    continue
                misses.append(f"{rel}:{lineno} cites `{cited}`")
    _fail_if_any(
        sorted(misses),
        "Unmarked internal/ citations in public docs (add '(private clone only)' marker):",
    )


def _audit_workflows_check_ids() -> set[str]:
    """Extract audit-check IDs from scripts/audit_workflows.py record() calls."""
    audit_path = REPO_ROOT / "scripts" / "audit_workflows.py"
    if not audit_path.exists():
        return set()
    text = audit_path.read_text()
    pattern = re.compile(r'record\(\s*[^,]+,\s*["\']([a-z_][a-z0-9_]+)["\']')
    return set(pattern.findall(text))


def _cited_audit_ids(text: str) -> set[str]:
    """Extract audit IDs from markdown via four citation forms used in docs.

    Forms (using ID as the backticked snake_case identifier):
      - "Audit: ID"          immediate citation
      - "Audit: F<N> (ID)"   numbered with id in parens
      - "(F<N> -- ID)"       compound paren citation with em-dash or hyphen
      - "ID (F<N>)"          id-first form
    """
    cited: set[str] = set()
    cited.update(re.findall(r"[Aa]udit[s]?:\s*`([a-z_][a-z0-9_]+)`", text))
    cited.update(re.findall(r"[Aa]udit[s]?:\s*F\d+\s*\(\s*`([a-z_][a-z0-9_]+)`", text))
    cited.update(re.findall(r"\(F\d+\s*[—\-]\s*`([a-z_][a-z0-9_]+)`", text))
    cited.update(re.findall(r"`([a-z_][a-z0-9_]+)`\s*\(F\d+\)", text))
    return cited


def test_cited_audit_ids_exist() -> None:
    """Every audit ID cited in CLAUDE.md / docs/reference/ must exist as a
    record(...) call in scripts/audit_workflows.py. Drift class: rename
    in audit_workflows.py without propagating to docs."""
    truth_set = _audit_workflows_check_ids()
    if not truth_set:
        pytest.skip("audit_workflows.py not found or no record() calls extracted")

    paths = list(_all_claude_md_files())
    ref_dir = REPO_ROOT / "docs" / "reference"
    if ref_dir.is_dir():
        paths.extend(ref_dir.glob("*.md"))

    misses: list[str] = []
    for path in paths:
        text = path.read_text()
        for cited in _cited_audit_ids(text):
            if cited not in truth_set:
                rel = path.relative_to(REPO_ROOT)
                misses.append(f"{rel} cites '{cited}'")
    _fail_if_any(
        sorted(misses),
        "Cited audit IDs not found in scripts/audit_workflows.py:",
    )


def test_audit_id_extractor_picks_up_citation_forms() -> None:
    text = """
    Audit: `frame_planner_present` (F8)
    Audit: F7 (`planner_no_stride_input`)
    The `alc_widget_drift` (F6) check fires when...
    Plain prose with `not_an_audit_call` should not match.
    """
    cited = _cited_audit_ids(text)
    assert "frame_planner_present" in cited
    assert "planner_no_stride_input" in cited
    assert "alc_widget_drift" in cited
    assert "not_an_audit_call" not in cited


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
