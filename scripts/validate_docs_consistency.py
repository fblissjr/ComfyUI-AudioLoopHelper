#!/usr/bin/env python3
"""Scan docs/ for known-stale phrases.

Each entry in STALE_PATTERNS corresponds to a breaking change where
a claim in the docs must change. The HISTORICAL_MARKERS list lets
legitimate "the OLD value was X" callouts pass through — without
them we'd have to choose between "can never reference history" and
"validator is useless."

Run: `uv run python scripts/validate_docs_consistency.py`
CI:   wired in `tests/test_docs_consistency.py`.

To add a new rule after the next breaking change:
1. Add a regex to STALE_PATTERNS with a one-sentence reason that
   points to the correct current value.
2. Run the script; fix any offending docs.
3. If a legitimate historical callout trips the new rule, add a
   specific substring to HISTORICAL_MARKERS. Prefer narrow markers
   that only appear in historical context ("(continuous seconds)",
   "pre-2026-04-20").
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# (old_value, overlap_seconds, new_value) for the 2026-04-20
# integer-latent stride fix at window=19.88, fps=25.
_STRIDE_FIXES: tuple[tuple[float, float, float], ...] = (
    (17.88, 2.0, 17.92),
    (16.88, 3.0, 16.96),
    (15.88, 4.0, 16.00),
)


def _stride_value_pattern(old: float) -> str:
    # Lookaround only forbids ADJACENT digits — so "17.88s" and
    # "17.88 seconds" both trip, but "117.88" and "17.881" do not.
    return rf"(?<![\d.]){old}(?!\d)"


# Regex → reason. Each regex is applied per-line.
STALE_PATTERNS: dict[str, str] = {
    # Pre-2026-04-20 stride formula (continuous seconds rather than
    # integer-latent quantized). Current source of truth:
    # `AudioLoopController.execute` — see CLAUDE.md "Key patterns".
    r"stride\s*=\s*window(_seconds)?\s*[-−]\s*overlap(_seconds)?": (
        "pre-2026-04-20 stride formula; current stride is integer-"
        "latent quantized — see AudioLoopController"
    ),
    r"window_seconds\s*[-−]\s*overlap_seconds": (
        "pre-2026-04-20 stride formula; integer-latent quantized now"
    ),
    **{
        _stride_value_pattern(old): (
            f"pre-fix stride at overlap={overlap}; current "
            f"effective stride is {new:.2f}"
        )
        for old, overlap, new in _STRIDE_FIXES
    },
    # 2026-06-06 decoder swap: the loop family's final decode (node 1604)
    # is LTXVSpatioTemporalTiledVAEDecode with stride-aligned temporal
    # chunks. Claims that 1604 is the core VAEDecodeTiled are stale.
    r"1604\s+VAEDecodeTiled|VAEDecodeTiled\s*\(typically node 1604\)": (
        "node 1604 is LTXVSpatioTemporalTiledVAEDecode since 2026-06-06 "
        "(apply_ltx_decoder.py --spatiotemporal)"
    ),
    # 2026-06-06 checkpoint rotation: latent banking lives on
    # PreDecodeCleanup.checkpoint_keep; the standalone SaveLatent toggle
    # UX is retired (apply_latent_checkpoint.py).
    r"SaveLatent toggle|toggled SaveLatent|Save assembled latent \(toggle\)": (
        "latent banking moved to PreDecodeCleanup.checkpoint_keep "
        "(rotated; scripts/apply_latent_checkpoint.py)"
    ),
}

# Substrings that flag a line as a historical callout — all stale
# patterns on that line are suppressed. Keep these narrow.
HISTORICAL_MARKERS: tuple[str, ...] = (
    "(not 17.88)",
    "not 17.88",
    "pre-2026-04-20",
    "Pre-2026-04-20",
    "prior to 2026-04-20",
    "Prior to 2026-04-20",
    "(continuous seconds)",
)


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """Return [(line_no, line, reason), ...] for stale matches in `text`."""
    issues: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(marker in line for marker in HISTORICAL_MARKERS):
            continue
        for pattern, reason in STALE_PATTERNS.items():
            if re.search(pattern, line):
                issues.append((lineno, line.strip(), reason))
                break  # one flag per line is enough
    return issues


def scan_docs(root: Path) -> dict[Path, list[tuple[int, str, str]]]:
    """Scan all .md files under `root`. Returns {path: issues}."""
    results: dict[Path, list[tuple[int, str, str]]] = {}
    for md in sorted(root.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        issues = scan_text(text)
        if issues:
            results[md] = issues
    return results


def main() -> int:
    results = scan_docs(DOCS_DIR)
    if not results:
        print(f"OK: no stale phrases in {DOCS_DIR.relative_to(REPO_ROOT)}/")
        return 0
    print(f"FAIL: stale phrases found in {DOCS_DIR.relative_to(REPO_ROOT)}/")
    for path, issues in results.items():
        rel = path.relative_to(REPO_ROOT)
        for lineno, line, reason in issues:
            print(f"  {rel}:{lineno}  {reason}")
            print(f"    > {line[:100]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
