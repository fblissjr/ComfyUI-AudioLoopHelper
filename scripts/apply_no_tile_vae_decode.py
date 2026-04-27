"""Set LTXVTiledVAEDecode to [1,1,1] (single-tile, no actual tiling) +
remove dead preview-decode chain on production workflows.

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

Note on terminology: "single-tile" means horizontal_tiles=1,
vertical_tiles=1 — the LTXVTiledVAEDecode node still runs (we keep
the LTX-aware decoder path), it just decodes the whole frame in one
pass instead of splitting spatially. The earlier "no-tile" framing
conflated config with node-class swap; we never swap the node.

Empirical evidence (RUN_ID 20260427T195630Z_2b99 vs 20260427T201628Z_5710,
both at 832x448x497 on a 24GB sm89 card):

| Config                                                      | Cold  | Warm  |
|-------------------------------------------------------------|------:|------:|
| `LTXVTiledVAEDecode [2,2,1]`                                | 143.6s| 11.8s |
| `LTXVTiledVAEDecode [1,1,1]`                                |  47.4s| 10.4s |   ← winner
| `LTXVSpatioTemporalTiledVAEDecode` (temporal-only, 9 tiles) |  61.4s| 13.3s |

Conclusion: single-tile beats temporal-tile beats spatial-tile on
this workload. Threshold matters less than per-tile overhead.

Two coordinated changes:

1. **Set widgets to `[1, 1, 1]`** on every active LTXVTiledVAEDecode
   node with downstream consumers. Bypassed (mode=4) and dead nodes
   (no consumers, or consumer is bypassed) are left alone — they
   don't run anyway.

2. **Remove the dead preview-decode chain** (#1318 LTXVTiledVAEDecode
   + #560 VHS_VideoCombine, the latter bypassed via mode=4). This
   pair is a leftover from earlier per-iteration-decode architecture
   that was deprecated in favor of the once-at-end #1604 decode but
   never removed. ComfyUI skips them at runtime regardless; this
   removes them structurally so future readers don't get confused
   by stale `[2,2,1]` configs on dead nodes.

Compatibility:
  - Reversible via `--revert` for the widget config (sets back to
    `[2, 2, 1]`). **Removed nodes stay removed**: use `git checkout`
    to restore them if you genuinely need the preview-decode chain.
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

from workflow_utils import WorkflowEditor, is_active

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

# widgets_values layout for LTXVTiledVAEDecode:
#   [horizontal_tiles, vertical_tiles, overlap, last_frame_fix, working_device, working_dtype]
SINGLE_TILE = [1, 1, 1, True, "auto", "auto"]
CANONICAL_TILED = [2, 2, 1, True, "auto", "auto"]

# Dead preview-decode chain. Pre-2026-04-27 architecture decoded a
# preview every iteration; loop now decodes once at the end via
# #1604. These two were left behind, with #560 bypassed and #1318
# either dead-end or feeding the bypassed #560.
DEAD_PREVIEW_DECODE_NODE = 1318
DEAD_PREVIEW_COMBINE_NODE = 560


def _has_live_consumer(wf: dict, node_id: int) -> bool:
    """A node's output is live iff at least one consumer is unbypassed."""
    node = next((n for n in wf["nodes"] if n["id"] == node_id), None)
    if node is None:
        return False
    consumer_ids: set[int] = set()
    for out in node.get("outputs") or []:
        for lid in out.get("links") or []:
            link = next((l for l in wf["links"] if isinstance(l, list) and l[0] == lid), None)
            if link is not None:
                consumer_ids.add(link[3])
    for cid in consumer_ids:
        consumer = next((n for n in wf["nodes"] if n["id"] == cid), None)
        if consumer and is_active(consumer):
            return True
    return False


def _set_widgets_to_single_tile(ed: WorkflowEditor, revert: bool) -> tuple[list[str], list[str]]:
    """Phase 1: widget config. Returns (changes, skipped) message lists."""
    target = SINGLE_TILE if not revert else CANONICAL_TILED
    other = CANONICAL_TILED if not revert else SINGLE_TILE

    changes: list[str] = []
    skipped: list[str] = []
    for n in ed.find_nodes_by_type("LTXVTiledVAEDecode"):
        if not is_active(n):
            continue  # bypassed; user choice
        if not _has_live_consumer(ed.wf, n["id"]):
            continue  # dead-end or fed-only-bypassed; ComfyUI skips
        wv = n.get("widgets_values") or []
        if list(wv) == list(target):
            continue
        if list(wv) != list(other):
            skipped.append(
                f"#{n.get('id')} unexpected widgets_values={wv} "
                f"(expected {other} or {target}; reconcile manually)"
            )
            continue
        n["widgets_values"] = list(target)
        verb = "reverted to [2,2,1]" if revert else "set [1,1,1] (single-tile)"
        changes.append(f"#{n.get('id')} {verb}")
    return changes, skipped


def _remove_dead_preview_chain(ed: WorkflowEditor) -> list[str]:
    """Phase 2: structural cleanup. Remove #1318 + #560 chain when present
    and dead. Idempotent (removing already-absent nodes is a no-op).
    Not reverted — once removed, stays removed (use git checkout to restore)."""
    removed: list[str] = []
    for nid in (DEAD_PREVIEW_DECODE_NODE, DEAD_PREVIEW_COMBINE_NODE):
        node = next((n for n in ed.wf["nodes"] if n["id"] == nid), None)
        if node is None:
            continue  # already absent
        # Sanity: only remove if dead at runtime. #560 must be bypassed.
        # #1318 must have no live consumers (OK if all consumers are
        # bypassed nodes — they don't run either).
        if nid == DEAD_PREVIEW_COMBINE_NODE:
            if is_active(node):
                continue  # Combine node is unbypassed → load-bearing, leave
        else:  # DEAD_PREVIEW_DECODE_NODE
            if _has_live_consumer(ed.wf, nid):
                continue  # Decode has a live consumer → leave
        ed.remove_node_and_links(nid)
        removed.append(f"removed #{nid} ({node.get('type')})")
    return removed


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    if not ed.find_nodes_by_type("LTXVTiledVAEDecode"):
        return "skip (no LTXVTiledVAEDecode node)"

    changes, skipped = _set_widgets_to_single_tile(ed, revert)
    removed = _remove_dead_preview_chain(ed) if not revert else []

    parts = changes + removed
    if not parts and not skipped:
        return "already reverted" if revert else "no change (already clean)"
    if not parts and skipped:
        return "; ".join(f"skip ({s})" for s in skipped)
    if dry_run:
        msg = "would " + "; ".join(parts)
        if skipped:
            msg += "; " + "; ".join(f"skip ({s})" for s in skipped)
        return msg
    ed.save(wf_path)
    msg = "; ".join(parts)
    if skipped:
        msg += "; " + "; ".join(f"skip ({s})" for s in skipped)
    return msg


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert widgets to [2,2,1]' if revert else 'apply single-tile + dead-chain cleanup'}"
    else:
        action = "Reverting widgets to [2,2,1]" if revert else "Applying single-tile + dead-chain cleanup"
    print(f"{action} across example_workflows/...")
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
                    help="Restore widget config to [2,2,1] (does NOT re-add removed dead-chain nodes; use git checkout for that).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
