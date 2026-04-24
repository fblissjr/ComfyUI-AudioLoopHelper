"""<SCRIPT_NAME>.

Last updated: <YYYY-MM-DD>

Stages an experimental variant of the canonical latent workflow to
`internal/scratch/` — does NOT mutate `example_workflows/` in place.

Promotion to `example_workflows/` follows the "ships AND stabilizes"
criterion in `internal/PLAN.md`.

Symptom / motivation: <SYMPTOM>

Root cause / what this variant tests: <ROOT_CAUSE>

Fix / change applied: <FIX>

Compatibility with other apply scripts:
  - <COMPATIBILITY_NOTES>

Usage:
    uv run --group dev python scripts/apply_<NAME>.py
    uv run --group dev python scripts/apply_<NAME>.py --revert
    uv run --group dev python scripts/apply_<NAME>.py --dry-run

Idempotent on the OUTPUT path. `--revert` deletes the staging file.
`--dry-run` reports the planned ops without writing or copying anything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

# TODO: node ID constants
NODE_ID_A = 0   # <NodeTypeA> -- <role>

REQUIRED_SOURCE_NODES = (
    NODE_ID_A,
    # ...
)

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/scratch/audio-loop-music-video_latent_<FEATURE>_<PHASE>.json"


def _already_migrated(ed: WorkflowEditor) -> bool:
    # TODO: return True iff the staging file already has this migration applied.
    # Typical check: `bool(ed.find_nodes_by_type("<NewNodeType>"))`.
    raise NotImplementedError


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = ed.require_nodes(REQUIRED_SOURCE_NODES)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source node(s) missing: {missing}. "
            "This script assumes the canonical latent workflow layout."
        )


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if output_path.exists() and input_path != output_path and _already_migrated(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
        return

    if dry_run:
        ed = WorkflowEditor(input_path)
        _assert_required_nodes_present(ed)
        print(f"would copy {input_path} -> {output_path}")
        print(f"would apply <SCRIPT_NAME> ops to {output_path}")
        # TODO: print the specific ops (node adds, rewires) that would run here.
        return

    if input_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)
    if _already_migrated(ed):
        print(f"{output_path.name}: already migrated, skipping.")
        return

    _assert_required_nodes_present(ed)

    # TODO: apply edits here. Use rewire_input / add_link / remove_link /
    # add_top_level_node from WorkflowEditor.

    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Load in ComfyUI: open {output_path}")
    print( "  3. A/B render against the canonical baseline (same seed, same prompts).")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied/changed without writing.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return

    _migrate(Path(args.input), output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
