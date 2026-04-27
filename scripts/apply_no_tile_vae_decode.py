"""Set LTXVTiledVAEDecode to [1, 1, 1] (single-tile) on production workflows.

Last updated: 2026-04-27

Symptom it fixes: VAE decode wall time dominates the render
(`LTXVTiledVAEDecode [2,2,1]` cold-pass was 143s vs single-tile's
47s — 3x improvement).

Root cause: Tiled VAE decode pays per-tile prepare/stage overhead
that exceeds the activation-memory savings on 24GB cards. Each tile
emits a `Model VideoVAE prepared for dynamic VRAM loading. 1384MB
Staged.` cycle even when nothing offloads — that hook is roughly
30-40s per tile cold and ~3s warm. With 4 spatial tiles + 1 cold
pass, ComfyUI pays 4× that overhead. Going to `[1, 1, 1]` (one
tile, full-frame decode) keeps the LTX-aware decoder path but pays
the prepare cycle once instead of N times.

Empirical evidence (RUN_ID 20260427T195630Z_2b99 vs 20260427T201628Z_5710,
both at 832x448x497 on a 24GB sm89 card):

| Config                          | Cold  | Warm  |
|---------------------------------|------:|------:|
| `LTXVTiledVAEDecode [2,2,1]`    | 143.6s| 11.8s |
| `LTXVTiledVAEDecode [1,1,1]`    |  47.4s| 10.4s |   ← winner
| `LTXVSpatioTemporalTiledVAEDecode` (temporal-only, 9 tiles) | 61.4s | 13.3s |

Conclusion: single-tile beats temporal-tile beats spatial-tile on
this workload. Threshold matters less than per-tile overhead.

Fix: This script sets `horizontal_tiles=1, vertical_tiles=1,
overlap=1` on every active `LTXVTiledVAEDecode` node in production
workflows. Bypassed (mode=4) and dead nodes (no consumers) are left
alone. Validator's `overlap >= 1` minimum is honored.

Compatibility:
  - Schema-additive: doesn't change node type, only widget values.
  - Reversible via `--revert` (sets back to `[2, 2, 1]` canonical).
  - **VRAM requirement: 24GB.** On 16GB cards, single-tile decode
    of 832x448x497 will likely OOM. Revert if so.
  - No interaction with other apply scripts.

Usage:
    uv run --group dev python scripts/apply_no_tile_vae_decode.py
    uv run --group dev python scripts/apply_no_tile_vae_decode.py --revert
    uv run --group dev python scripts/apply_no_tile_vae_decode.py --dry-run

Idempotent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

# widgets_values layout for LTXVTiledVAEDecode:
#   [horizontal_tiles, vertical_tiles, overlap, last_frame_fix, working_device, working_dtype]
NO_TILE = [1, 1, 1, True, "auto", "auto"]
CANONICAL = [2, 2, 1, True, "auto", "auto"]


def _is_active(node: dict) -> bool:
    return node.get("mode", 0) != 4


def _has_downstream_consumers(wf: dict, node_id: int) -> bool:
    """An active LTXVTiledVAEDecode node is a write target only if
    something consumes its IMAGE output. Dead nodes (no consumers)
    are ComfyUI-skipped at runtime and not worth touching."""
    node = next((n for n in wf["nodes"] if n["id"] == node_id), None)
    if node is None:
        return False
    for out in node.get("outputs") or []:
        if out.get("links"):
            return True
    return False


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    decoders = ed.find_nodes_by_type("LTXVTiledVAEDecode")
    if not decoders:
        return "skip (no LTXVTiledVAEDecode node)"

    target = NO_TILE if not revert else CANONICAL
    other  = CANONICAL if not revert else NO_TILE

    changes = []
    for n in decoders:
        if not _is_active(n):
            continue  # bypassed; leave widgets as user configured
        if not _has_downstream_consumers(ed.wf, n["id"]):
            continue  # dead node; ComfyUI skips at runtime
        wv = n.get("widgets_values") or []
        if list(wv) == list(target):
            continue  # already at target
        if list(wv) != list(other):
            return f"skip (#{n.get('id')} widgets_values={wv} doesn't match canonical or no-tile)"
        n["widgets_values"] = list(target)
        verb = "reverted to [2,2,1]" if revert else "set [1,1,1]"
        changes.append(f"#{n.get('id')} {verb}")

    if not changes:
        return "already reverted" if revert else "no change (already at [1,1,1])"
    if dry_run:
        return "would " + "; ".join(changes)
    ed.save(wf_path)
    return "; ".join(changes)


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} no-tile VAE decode across example_workflows/...")
    fail = 0
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.json")):
        status = _apply_one(wf_path, revert, dry_run)
        print(f"  {wf_path.name}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--revert", action="store_true",
                    help="Restore the canonical [2,2,1] tiled decode.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
