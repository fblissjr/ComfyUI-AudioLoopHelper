"""apply_alc_seed_rename.

Last updated: 2026-04-26

Symptom it fixes: saved workflow JSONs show `AudioLoopController.widgets_values`
drifting between renders (the seed widget value is a different 64-bit integer
each time) even when the seed input is wired to a stable upstream constant.
This made the eight-render ID-LoRA ablation look like cross-run seed
randomization was the source of variance, when in fact every render received
seed=42 via the Get_start_seed wire — the widget drift was cosmetic.

Root cause: ComfyUI's frontend auto-attaches a `control_after_generate`
dropdown to any INT widget literally named `"seed"` or `"noise_seed"`. After
each successful run the dropdown mutates the saved widget value (default
mode is `randomize`). At execute time the wired link supersedes the widget,
so the runtime seed is unchanged — but the widget value still gets serialized
and the saved JSON drifts. Diagnosed 2026-04-26 in
`internal/analysis/id_lora_ablation_and_seed_widget_audit.md`.

Fix: rename the input from `"seed"` → `"base_seed"`. ComfyUI only
auto-attaches the control to those two specific widget names, so renaming
defuses the trap. The schema-level rename ships in `nodes.py:482`; this script
migrates the persisted workflow JSONs so they stay in sync. Both
`inputs[].name` and `inputs[].widget.name` need to be rewritten — ComfyUI
reads input names from the JSON when reattaching wires after node creation,
so a stale `"seed"` would dangle the wire.

Compatibility with other apply scripts: orthogonal — touches only the
`AudioLoopController` node's `seed` input. Does not interact with
apply_sage_mode, apply_iclora_initial_render, F2 (preprocess symmetry), or
F3 (cropguides symmetry).

Usage:
    uv run --group dev python scripts/apply_alc_seed_rename.py
    uv run --group dev python scripts/apply_alc_seed_rename.py --revert
    uv run --group dev python scripts/apply_alc_seed_rename.py --dry-run

Idempotent. Run repeatedly; already-renamed workflows report "no change".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"
EXTRA_PATHS = (
    REPO_ROOT / "internal" / "scratch" / "audio-loop-music-video_man_girl_guitar.json",
)

OLD_NAME = "seed"
NEW_NAME = "base_seed"


def _collect_rename_targets(node: dict, target_old: str) -> list[tuple[dict, str]]:
    """Return list of (container, field_name) pairs that currently hold
    target_old and need to be set to the new name. Inspects both the input
    dict's `name` and its nested `widget.name`.
    """
    targets = []
    for inp in node.get("inputs") or []:
        if inp.get("name") == target_old:
            targets.append((inp, "name"))
        widget = inp.get("widget")
        if isinstance(widget, dict) and widget.get("name") == target_old:
            targets.append((widget, "name"))
    return targets


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    alc_nodes = ed.find_nodes_by_type("AudioLoopController")
    if not alc_nodes:
        return "skip (no AudioLoopController)"

    target_old, target_new = (NEW_NAME, OLD_NAME) if revert else (OLD_NAME, NEW_NAME)

    all_targets: list[tuple[dict, str]] = []
    for node in alc_nodes:
        all_targets.extend(_collect_rename_targets(node, target_old))

    if not all_targets:
        # Nothing to rename in the source direction. Either already migrated
        # to target_new or in an unexpected state.
        # Check if it's already in target_new state — that's the no-op case.
        already = []
        for node in alc_nodes:
            for inp in node.get("inputs") or []:
                if inp.get("name") == target_new:
                    already.append(inp)
        if already:
            return "already reverted" if revert else "no change (already renamed)"
        return f"skip (no '{target_old}' input found on any AudioLoopController)"

    verb = "would revert" if dry_run and revert else \
           "would rename" if dry_run else \
           "reverted" if revert else "renamed"

    if not dry_run:
        for obj, field in all_targets:
            obj[field] = target_new
        ed.save(wf_path)

    n = len(all_targets)
    return f"{verb} ({n} field{'s' if n != 1 else ''}: '{target_old}' -> '{target_new}')"


def _iter_workflow_paths():
    yield from sorted(WORKFLOWS_DIR.glob("*.json"))
    yield from sorted((WORKFLOWS_DIR / "experimental").glob("*.json"))
    for p in EXTRA_PATHS:
        if p.exists():
            yield p


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} apply_alc_seed_rename across example_workflows/ + scratch...")
    fail = 0
    for wf_path in _iter_workflow_paths():
        rel = wf_path.relative_to(REPO_ROOT)
        status = _apply_one(wf_path, revert, dry_run)
        print(f"  {rel}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--revert", action="store_true",
                    help="Revert the rename ('base_seed' -> 'seed').")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without touching files.")
    args = ap.parse_args()
    return apply(args.revert, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
