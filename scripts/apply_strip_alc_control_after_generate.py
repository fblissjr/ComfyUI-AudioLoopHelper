"""apply_strip_alc_control_after_generate.

Last updated: 2026-04-27

Symptom it fixes: at execute time, AudioLoopController fails validation with:
    "Failed to convert an input value to a INT value: fps, randomize,
     invalid literal for int() with base 10: 'randomize'"
ComfyUI's frontend, when the input was named `seed` (pre-rename), attached a
`control_after_generate` dropdown widget that serialized as a 6th entry in
`widgets_values` (literal string `'randomize'` / `'fixed'` / `'increment'`).

Root cause: `apply_alc_seed_rename.py` renamed `inputs[].name` from `seed` to
`base_seed` in 1f6b830 — which prevented ComfyUI from re-attaching the
dropdown on future loads — but did NOT prune the leftover 6th element from
`widgets_values`. The schema's widget order is
    [current_iteration, window_seconds, overlap_seconds, base_seed, fps]
(5 widgets; AUDIO has no widget). The backend pops widgets positionally; with
6 saved values, `'randomize'` lands in the `fps` slot and explodes the INT
parse. AudioLoopController is the only known node carrying the leftover.

Fix: detect 6-element `widgets_values` on AudioLoopController, drop the
non-numeric entry at index 4 (the `control_after_generate` value), and save.

Compatibility with other apply scripts:
  - Strict superset of `apply_alc_seed_rename.py` (which only renames the
    input). Run `apply_alc_seed_rename.py` first if not already applied,
    then this. Both are idempotent and orthogonal in scope.
  - Independent of F2 (preprocess), F3 (cropguides), iterations_autowire,
    sage mode, melband default — touches only this single node's widget list.

Usage:
    uv run --group dev python scripts/apply_strip_alc_control_after_generate.py
    uv run --group dev python scripts/apply_strip_alc_control_after_generate.py --revert
    uv run --group dev python scripts/apply_strip_alc_control_after_generate.py --dry-run

Idempotent. `--revert` re-inserts `'randomize'` at index 4 to restore the
exact pre-fix shape (mainly for round-trip testing of this migration; not
useful in production).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

EXPECTED_WIDGET_LEN = 5  # [current_iteration, window_seconds, overlap_seconds, base_seed, fps]
DRIFT_INDEX = 4          # position of the leaked control_after_generate value
LEGACY_DEFAULT = "randomize"


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    alcs = ed.find_nodes_by_type("AudioLoopController")
    if not alcs:
        return "skip (no AudioLoopController)"

    changed: list[str] = []
    for node in alcs:
        wv = node.get("widgets_values") or []
        nid = node.get("id")
        if revert:
            if len(wv) == EXPECTED_WIDGET_LEN + 1:
                changed.append(f"#{nid} already reverted")
                continue
            if len(wv) != EXPECTED_WIDGET_LEN:
                changed.append(f"#{nid} skip (unexpected len={len(wv)})")
                continue
            wv = list(wv)
            wv.insert(DRIFT_INDEX, LEGACY_DEFAULT)
            if not dry_run:
                node["widgets_values"] = wv
            changed.append(f"#{nid} re-inserted '{LEGACY_DEFAULT}'")
        else:
            if len(wv) == EXPECTED_WIDGET_LEN:
                changed.append(f"#{nid} no change")
                continue
            if len(wv) != EXPECTED_WIDGET_LEN + 1:
                changed.append(f"#{nid} skip (unexpected len={len(wv)})")
                continue
            stale = wv[DRIFT_INDEX]
            if isinstance(stale, (int, float)):
                changed.append(f"#{nid} skip (index {DRIFT_INDEX} = {stale!r}, not a stray string)")
                continue
            new_wv = wv[:DRIFT_INDEX] + wv[DRIFT_INDEX + 1:]
            if not dry_run:
                node["widgets_values"] = new_wv
            changed.append(f"#{nid} stripped {stale!r}")

    if any("stripped" in c or "re-inserted" in c for c in changed) and not dry_run:
        ed.save(wf_path)

    verb_prefix = "would " if dry_run else ""
    return f"{verb_prefix}{', '.join(changed)}"


def _iter_workflow_paths():
    yield from sorted(WORKFLOWS_DIR.glob("*.json"))
    yield from sorted((WORKFLOWS_DIR / "experimental").glob("*.json"))


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} apply_strip_alc_control_after_generate across example_workflows/...")
    fail = 0
    for wf_path in _iter_workflow_paths():
        rel = wf_path.relative_to(REPO_ROOT)
        status = _apply_one(wf_path, revert, dry_run)
        print(f"  {rel}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--revert", action="store_true",
                    help="Re-insert the legacy 'randomize' string at index 4.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
