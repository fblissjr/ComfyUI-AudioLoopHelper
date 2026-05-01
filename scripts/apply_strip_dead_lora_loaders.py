"""apply_strip_dead_lora_loaders.

Last updated: 2026-04-30

Symptom it fixes: the canonical baseline workflow
`example_workflows/audio-loop-music-video_latent.json` ships three
bypassed (mode=4) LoRA-loader nodes that form dead scaffolding:

    #1625 LoraLoaderModelOnly       title "ID-LoRA File (audio-conditioned identity)"
    #1626 LTXICLoRALoaderModelOnly  title "IC-LoRA File (visual reference adapter)"
    #1627 LoraLoaderModelOnly       title "Style/Generic LoRA"  (placeholder filename)

All three are bypassed → MODEL passes through unchanged → they produce
no output. They imply (via title + type) that the workflow does
ID-LoRA / IC-LoRA / style-LoRA when in fact it doesn't until the user
manually un-bypasses and re-targets the placeholder. This is misleading
UI clutter and a maintenance liability.

Root cause: scaffolding placeholders left in the canonical from earlier
exploration of the LoRA-loader chain. Phase 0a's IC-LoRA wiring
(`scripts/apply_iclora_initial_render.py`) does NOT reuse these — it
adds a new `LTXICLoRALoaderModelOnly` and `LTXAddVideoICLoRAGuide`
elsewhere in the graph. So the scaffolding was never load-bearing.

Fix: detect the three-node pattern via (id, type, title) triple AND
mode=4 AND the canonical filename per node, then splice the upstream
MODEL source (currently feeding #1625.0 via link 3080) directly to the
downstream sink (currently consumed off #1627.0 via link 3083), and
remove the three nodes plus their internal links. Strict matching
preserves user customizations: if a user un-bypassed any of the three
or renamed any title, the script skips that workflow.

Compatibility:
  - Independent of F2/F3/F4/F5/F6/F7/F8/F9/F10. Touches only the
    bypassed scaffolding chain that no live consumer depends on.
  - `apply_iclora_initial_render.py` adds new IC-LoRA nodes via
    `add_top_level_node()` (not by reusing #1626). Phase 0a still
    works after this strip.
  - `apply_iclora_video_reference.py` (companion to this cleanup)
    expects the canonical to be post-strip; it pre-flight-checks for
    the three dead-scaffolding nodes and refuses to fork from a still-
    polluted canonical.

Usage:
    uv run --group dev python scripts/apply_strip_dead_lora_loaders.py
    uv run --group dev python scripts/apply_strip_dead_lora_loaders.py --revert
    uv run --group dev python scripts/apply_strip_dead_lora_loaders.py --dry-run

Idempotent. Re-run reports "no change" without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

_Classification = Literal[
    "scaffolding_present", "already_stripped", "partial_or_mismatch", "collision_only",
]


# Canonical scaffolding signatures. Strict triple (id, type, title) plus
# mode=4 plus expected filename in widgets_values[0] — all must match
# for the node to be considered dead scaffolding.
_SCAFFOLDING = (
    {
        "id": 1625,
        "type": "LoraLoaderModelOnly",
        "title": "ID-LoRA File (audio-conditioned identity)",
        "lora_file": "LTX-2.3-ID-LoRA-CelebVHQ-3K/lora_weights.safetensors",
        "pos": [-300, 5500],
        "size": [360, 90],
    },
    {
        "id": 1626,
        "type": "LTXICLoRALoaderModelOnly",
        "title": "IC-LoRA File (visual reference adapter)",
        "lora_file": "MergeGreen_IC-lora_ltx2.3.safetensors",
        "pos": [-300, 5630],
        "size": [360, 90],
    },
    {
        "id": 1627,
        "type": "LoraLoaderModelOnly",
        "title": "Style/Generic LoRA",
        "lora_file": "your_style_lora.safetensors",
        "pos": [-300, 5760],
        "size": [360, 90],
    },
)

_SCAFFOLDING_IDS = tuple(s["id"] for s in _SCAFFOLDING)

# Canonical chain endpoints — needed for revert and for the rebridge.
# These are derived from the canonical layout; if the surrounding graph
# has been heavily restructured, the script falls back to whatever link
# is actually feeding/consuming the chain ends.
_DEFAULT_UPSTREAM_SOURCE = (503, 0)   # LTX2SamplingPreviewOverride.0
_DEFAULT_DOWNSTREAM_SINK = (572, 0)   # SetNode "model".0


def _is_scaffolding_node(node: dict, sig: dict) -> bool:
    """All-or-nothing match against a scaffolding signature."""
    if node.get("id") != sig["id"]:
        return False
    if node.get("type") != sig["type"]:
        return False
    if node.get("title") != sig["title"]:
        return False
    if node.get("mode") != 4:  # must still be bypassed
        return False
    widgets = node.get("widgets_values") or []
    if not widgets or widgets[0] != sig["lora_file"]:
        return False
    return True


def _find_chain_endpoints(ed: WorkflowEditor):
    """Return (upstream_src_id, upstream_src_slot, downstream_tgt_id, downstream_tgt_slot)
    for the dead-scaffolding chain, or None if the chain shape is unexpected."""
    head_node = ed.find_node(_SCAFFOLDING_IDS[0])
    head_inputs = head_node.get("inputs") or []
    if not head_inputs or head_inputs[0].get("link") is None:
        return None
    head_link_id = head_inputs[0]["link"]

    tail_node = ed.find_node(_SCAFFOLDING_IDS[-1])
    tail_outputs = tail_node.get("outputs") or []
    if not tail_outputs:
        return None
    tail_link_ids = list(tail_outputs[0].get("links") or [])
    if not tail_link_ids:
        return None

    # Resolve source of head_link_id and consumer(s) of tail_link_ids.
    upstream_src = None
    consumers = []  # list of (link_id, tgt_id, tgt_slot, dtype)
    for link in ed.wf["links"]:
        if not isinstance(link, list) or len(link) < 6:
            continue
        lid, src, src_slot, tgt, tgt_slot, dtype = link
        if lid == head_link_id:
            upstream_src = (src, src_slot, dtype)
        if lid in tail_link_ids:
            consumers.append((lid, tgt, tgt_slot, dtype))
    if upstream_src is None or not consumers:
        return None
    return upstream_src, consumers


def _strip(ed: WorkflowEditor) -> str:
    """Remove the three scaffolding nodes and rebridge the chain. Returns a status string."""
    endpoints = _find_chain_endpoints(ed)
    if endpoints is None:
        return "skip (chain endpoints not detectable)"
    upstream_src, consumers = endpoints
    src_id, src_slot, dtype = upstream_src

    # Remove the three scaffolding nodes (and their internal links).
    for nid in _SCAFFOLDING_IDS:
        ed.remove_node_and_links(nid)

    # Rebridge: add a direct link from upstream src to each consumer.
    for _lid, tgt_id, tgt_slot, _dt in consumers:
        ed.add_link(src_id, src_slot, tgt_id, tgt_slot, dtype)

    return f"stripped (rebridged node{src_id}.{src_slot} -> {len(consumers)} consumer(s))"


def _restore(ed: WorkflowEditor) -> str:
    """Inverse of _strip. Re-create the three nodes in series and rebridge through them."""
    # Find the rebridged direct link(s) replacing the chain. Heuristic:
    # any link where (src_id, src_slot) == _DEFAULT_UPSTREAM_SOURCE and
    # (tgt_id, tgt_slot) == _DEFAULT_DOWNSTREAM_SINK with dtype MODEL.
    src_id, src_slot = _DEFAULT_UPSTREAM_SOURCE
    sink_id, sink_slot = _DEFAULT_DOWNSTREAM_SINK
    bridge_links = [
        link for link in ed.wf["links"]
        if isinstance(link, list) and len(link) >= 6
        and link[1] == src_id and link[2] == src_slot
        and link[3] == sink_id and link[4] == sink_slot
        and link[5] == "MODEL"
    ]
    if not bridge_links:
        return "skip (no canonical rebridge link present; nothing to revert)"

    # Re-create the three nodes
    for sig in _SCAFFOLDING:
        if ed.has_node(sig["id"]):
            return f"skip (node #{sig['id']} already present; nothing to restore)"
        node = {
            "id": sig["id"],
            "type": sig["type"],
            "pos": list(sig["pos"]),
            "size": list(sig["size"]),
            "flags": {},
            "order": 0,
            "mode": 4,
            "inputs": [{"name": "model", "type": "MODEL", "link": None}],
            "outputs": [{"name": "model", "type": "MODEL", "links": []}],
            "properties": {"Node name for S&R": sig["type"]},
            "widgets_values": [sig["lora_file"], 1.0],
            "title": sig["title"],
        }
        if sig["type"] == "LTXICLoRALoaderModelOnly":
            node["outputs"].append({
                "name": "latent_downscale_factor",
                "type": "FLOAT",
                "links": [],
            })
        ed.add_node(node)

    # Remove the bridge link(s) and rewire chain through the restored nodes.
    for bridge in bridge_links:
        ed.remove_link(bridge[0])

    # Build the chain: src -> 1625 -> 1626 -> 1627 -> sink (and any other consumers)
    ids = list(_SCAFFOLDING_IDS)
    ed.add_link(src_id, src_slot, ids[0], 0, "MODEL")
    ed.add_link(ids[0], 0, ids[1], 0, "MODEL")
    ed.add_link(ids[1], 0, ids[2], 0, "MODEL")
    # All consumers that were on the bridge link(s) now read from the tail node.
    for bridge in bridge_links:
        ed.add_link(ids[2], 0, bridge[3], bridge[4], "MODEL")

    return f"reverted (re-added 3 scaffolding nodes; rewired through {len(bridge_links)} consumer(s))"


def _classify(ed: WorkflowEditor) -> _Classification:
    matches = []
    present = []
    for sig in _SCAFFOLDING:
        try:
            n = ed.find_node(sig["id"])
        except ValueError:
            n = None
        if n is None:
            continue
        present.append(sig["id"])
        if _is_scaffolding_node(n, sig):
            matches.append(sig["id"])

    if len(matches) == 3:
        return "scaffolding_present"
    if not present:
        return "already_stripped"
    if not matches and present:
        # All present-by-id but type/title/mode/file mismatched
        return "collision_only"
    return "partial_or_mismatch"


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    classification = _classify(ed)

    if revert:
        if classification == "scaffolding_present":
            return "already reverted (scaffolding present)"
        if classification != "already_stripped":
            return f"skip ({classification})"
        if dry_run:
            return "would revert (re-add 3 scaffolding nodes through canonical bridge)"
        status = _restore(ed)
        if status.startswith("skip"):
            return status
        ed.save(wf_path)
        return status

    # Forward apply
    if classification == "already_stripped":
        return "no change (already stripped)"
    if classification == "collision_only":
        return "no change (id collision but type/title differ — user nodes preserved)"
    if classification == "partial_or_mismatch":
        return "skip (partial scaffolding or user customization detected)"
    # scaffolding_present
    if dry_run:
        return "would strip (3 dead-scaffolding LoRA loaders + chain rebridge)"
    status = _strip(ed)
    if status.startswith("skip"):
        return status
    ed.save(wf_path)
    return status


def _iter_workflow_paths(workflows_dir: Path):
    yield from sorted(workflows_dir.glob("*.json"))
    experimental = workflows_dir / "experimental"
    if experimental.is_dir():
        yield from sorted(experimental.glob("*.json"))


def apply(revert: bool, dry_run: bool, workflows_dir: Path) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} apply_strip_dead_lora_loaders across {workflows_dir}/...")
    fail = 0
    for wf_path in _iter_workflow_paths(workflows_dir):
        try:
            rel = wf_path.relative_to(REPO_ROOT)
        except ValueError:
            rel = wf_path
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
                    help="Re-add the three scaffolding nodes in series.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    ap.add_argument("--workflows-dir", default=str(WORKFLOWS_DIR),
                    help="Directory of workflow JSONs to sweep (default: example_workflows/)")
    args = ap.parse_args()
    return apply(args.revert, args.dry_run, Path(args.workflows_dir))


if __name__ == "__main__":
    sys.exit(main())
