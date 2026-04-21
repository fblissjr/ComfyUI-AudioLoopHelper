"""Guard against known-stale phrases re-entering the public docs.

Each pattern corresponds to a breaking change where docs must be
updated alongside code. Historical markers let legitimate
"this is the OLD value we fixed" callouts through.

Extending: to add a new stale phrase after the next breaking change,
edit STALE_PATTERNS / HISTORICAL_MARKERS in
`scripts/validate_docs_consistency.py`.
"""

from pathlib import Path

import pytest

from validate_docs_consistency import (
    HISTORICAL_MARKERS,
    STALE_PATTERNS,
    scan_docs,
    scan_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_docs_have_no_stale_stride_formula():
    """docs/ must not carry the pre-2026-04-20 stride formula or values."""
    results = scan_docs(REPO_ROOT / "docs")
    if results:
        lines = ["Stale phrases found in docs/:"]
        for path, issues in results.items():
            rel = path.relative_to(REPO_ROOT)
            for lineno, line, reason in issues:
                lines.append(f"  {rel}:{lineno}  {reason}")
                lines.append(f"    > {line[:100]}")
        pytest.fail("\n".join(lines))


def test_synthetic_stale_formula_is_caught():
    text = "The stride = window - overlap gives 17.88s at default."
    issues = scan_text(text)
    assert issues, "Expected stale-formula hit"


def test_synthetic_unicode_minus_stale_formula_is_caught():
    text = "Stride = window_seconds − overlap_seconds gives drift."
    issues = scan_text(text)
    assert issues, "Unicode-minus variant should also be caught"


def test_synthetic_stale_value_is_caught():
    text = "Default stride is 17.88 seconds at overlap=2."
    issues = scan_text(text)
    assert any("17.92" in reason for _, _, reason in issues), (
        "Expected 17.88 → 17.92 reason to surface"
    )


def test_16_88_at_overlap_3_is_caught():
    text = "At overlap=3 stride is 16.88s."
    issues = scan_text(text)
    assert any("16.96" in reason for _, _, reason in issues)


def test_15_88_at_overlap_4_is_caught():
    text = "At overlap=4 stride is 15.88s."
    issues = scan_text(text)
    assert any("16.00" in reason for _, _, reason in issues)


def test_historical_marker_suppresses_hit():
    text = "Expect stride_seconds = 17.92 (not 17.88) at overlap=2.0."
    issues = scan_text(text)
    assert issues == [], f"Historical marker should exempt: {issues}"


def test_pre_fix_marker_suppresses_hit():
    text = "Prior to 2026-04-20, stride = window - overlap produced 17.88s."
    issues = scan_text(text)
    assert issues == [], f"'Prior to 2026-04-20' should exempt: {issues}"


def test_continuous_seconds_marker_suppresses_hit():
    text = "computed stride as `window_seconds - overlap_seconds` (continuous seconds)."
    issues = scan_text(text)
    assert issues == [], f"'(continuous seconds)' should exempt: {issues}"


def test_scan_docs_skips_non_markdown():
    tmpdir = REPO_ROOT / "docs"
    results = scan_docs(tmpdir)
    for path in results:
        assert path.suffix == ".md"


def test_patterns_and_markers_are_non_empty():
    assert STALE_PATTERNS
    assert HISTORICAL_MARKERS
