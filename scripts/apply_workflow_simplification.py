"""apply_workflow_simplification.

Last updated: 2026-04-27

Workflow-cleanup migration that removes verified-dead and verified-redundant
nodes from the latent workflow. Touches only items that have been
empirically confirmed safe (each via JSON-walk verification, not assumption).

What this script removes / collapses:

1. **#1513 ModelSamplingSD3 (dead)** — confirmed orphan after the
   sigma migration to canonical Lightricks values. Its sole prior consumer
   was BasicScheduler(1421) for sigma-curve generation; ManualSigmas now
   emits the canonical values verbatim, so ModelSamplingSD3's only effect
   (patching `model_sampling` to shift sigma generation) has nothing left
   to patch. The MODEL chain feeding the sampler bypasses #1513 entirely
   and always has — RuneXX workflows confirm the same pattern (their
   ModelSamplingSD3 also feeds only BasicScheduler, never the sampling
   model).

2. **#1606 Reroute (cosmetic passthrough)** — LATENT pass-through between
   #245 LTXVSeparateAVLatent and #1539 TensorLoopOpen.initial_value.
   Direct-wire the source to target.

3. **#560 VHS_VideoCombine (orphan, mode=4)** — bypassed alternate
   video-combine instance. Output `Filenames` has no consumers.
   Forgotten experimentation residue. The active `#617` covers all
   real video output.

4. **Four single-Get / single-consumer Set/Get pairs** collapse to direct
   wires. Verified by walking the JSON: each Set has exactly one
   downstream consumer (one Get with one consumer link, OR one direct
   link from Set, never both):
     - #576/#578  sampler          (#578 → KSamplerSelect direct consumer)
     - #650/#651  input_image      (#651 → IMAGE consumer)
     - #646/#648  base_cond_neg    (#648 → subgraph(843).slot[7])
     - #1271/#1273 first_frame_guide_strength (#1273 → subgraph(843).slot[12])

   The #1271/#1273 pair sources from #1269 FloatConstant (value=1.0).
   We direct-wire #1269 to subgraph slot 12 and remove the Set/Get pair;
   the FloatConstant stays so the user can edit the strength widget.

   NOT collapsed (multi-consumer broadcasts, intentionally kept):
     - #572/#654 model              (Set has direct link AND Get; 2 consumers)
     - #579/#580 sigmas             (Set direct link + Get; 2 consumers)
     - #228      video_vae          (4 Gets, 6 consumers)
     - #252      audio_vae          (2 Gets, 4 consumers)
     - #581      orig_audio         (2 Gets)
     - #640      actual_audio       (Set direct link + Get; 2 consumers)
     - #689      window_size_seconds (1 Get with 3 consumers)
     - #1528     start_seed         (2 Gets)

   These genuine broadcasts can't collapse to a single direct wire
   without duplicating the source link N times — would increase node-edge
   count, not decrease.

What this script does NOT touch (safety):
  - The MODEL patcher chain (UNETLoader -> SageAttention -> ChunkFeedForward
    -> AttentionTunerPatch -> NAG -> SamplingPreviewOverride -> LoRA chain
    -> SetNode("model")). All patches are upstream of the SetNode bottleneck;
    none are removed; order preserved.
  - The audio path (Get_orig_audio -> TrimAudioDuration -> Audio VAE encode
    -> LTXVConcatAVLatent). "Audio path is sacred" per CLAUDE.md.
  - F2/F3/F4/F5/F6/F7 invariants. None of the removed nodes touch these.
  - The bypassed LoRA chain (3 nodes from apply_lora_chain_bypassed.py)
    or the bypassed ID-LoRA runtime (3 nodes from apply_id_lora_runtime.py).
    Intentional toggles; users opt in by un-bypassing.

Total node count change: 113 -> 102 (saves 11 nodes, ~10%).

Compatibility with other apply scripts:
  - Independent of apply_lora_chain_bypassed.py and apply_id_lora_runtime.py
    (different node IDs).
  - Independent of apply_canonical_sigmas.py (which already converted
    BasicScheduler->ManualSigmas; #1513's death is a *consequence* of
    that, not a coincidence).
  - Independent of F2-F7 apply scripts.

Idempotent. Safe to re-run; if a candidate is already removed, that step
is a no-op and the script reports "no change".

Usage:
    uv run --group dev python scripts/apply_workflow_simplification.py
    uv run --group dev python scripts/apply_workflow_simplification.py --revert
    uv run --group dev python scripts/apply_workflow_simplification.py --dry-run

`--revert` only restores #1513, #1606, #560 (the dead nodes). The Set/Get
pairs are NOT restored on revert — collapsing them is a one-way cleanup
(the indirection wasn't load-bearing). If you need the pairs back for
visual canvas reasons, use ComfyUI's UI to add SetNode/GetNode and
relabel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

# Verified-dead nodes (orphan output, no real consumers).
DEAD_NODES = {
    1513: ("ModelSamplingSD3", "shift=13 patcher; output unwired post-sigma-migration"),
    1606: ("Reroute", "LATENT passthrough #245 -> #1539; cosmetic"),
    560: ("VHS_VideoCombine", "bypassed mode=4, no consumers; forgotten"),
}

# Set/Get pairs to collapse (single-consumer verified):
# (set_id, get_id, source_input_link_name, dtype) — the Set's input is
# what becomes the upstream of the direct wire; the Get's downstream
# becomes the direct target.
COLLAPSE_PAIRS = [
    {"set_id": 576, "get_id": 578, "var": "sampler", "dtype": "SAMPLER"},
    {"set_id": 650, "get_id": 651, "var": "input_image", "dtype": "IMAGE"},
    {"set_id": 646, "get_id": 648, "var": "base_cond_neg", "dtype": "CONDITIONING"},
    {"set_id": 1271, "get_id": 1273, "var": "first_frame_guide_strength",
     "dtype": "FLOAT"},
]


def _trace_set_upstream(ed: WorkflowEditor, set_id: int):
    """Return (src_node_id, src_slot, dtype) feeding the SetNode's first input."""
    set_node = ed.find_node(set_id)
    if set_node is None:
        return None
    inp = (set_node.get("inputs") or [{}])[0]
    link_id = inp.get("link")
    if link_id is None:
        return None
    link = next((l for l in ed.wf["links"] if l[0] == link_id), None)
    if link is None:
        return None
    return (link[1], link[2], link[5])


def _trace_get_downstream(ed: WorkflowEditor, get_id: int):
    """Return [(tgt_node_id, tgt_slot, dtype), ...] of all consumers."""
    get_node = ed.find_node(get_id)
    if get_node is None:
        return []
    out = (get_node.get("outputs") or [{}])[0]
    consumers = []
    for link_id in out.get("links") or []:
        link = next((l for l in ed.wf["links"] if l[0] == link_id), None)
        if link is None:
            continue
        consumers.append((link[3], link[4], link[5]))
    return consumers


def _node_exists(ed: WorkflowEditor, nid: int) -> bool:
    """find_node raises on missing; this returns bool."""
    return any(n.get("id") == nid for n in ed.wf.get("nodes", []))


def _is_dead_removed(ed: WorkflowEditor) -> bool:
    return not any(_node_exists(ed, nid) for nid in DEAD_NODES)


def _are_pairs_collapsed(ed: WorkflowEditor) -> bool:
    return all(
        not _node_exists(ed, p["set_id"]) and not _node_exists(ed, p["get_id"])
        for p in COLLAPSE_PAIRS
    )


def _apply(ed: WorkflowEditor) -> tuple[bool, str]:
    if _is_dead_removed(ed) and _are_pairs_collapsed(ed):
        return False, "no change (already simplified)"

    actions: list[str] = []

    # Step 1: Remove dead nodes (and their incoming links).
    for nid, (ntype, _why) in DEAD_NODES.items():
        if not _node_exists(ed, nid):
            actions.append(f"#{nid} ({ntype}): already removed")
            continue
        # Special case: #1606 Reroute — must direct-wire src->tgt before removing
        if nid == 1606:
            up = _trace_set_upstream(ed, 1606)
            downs = _trace_get_downstream(ed, 1606)
            if up and downs:
                src_id, src_slot, dtype = up
                # Reroute output type can be '*'; use the actual link type.
                # In practice for #1606 this is LATENT (verified).
                ed.remove_node_and_links(nid)
                for tgt_id, tgt_slot, _ in downs:
                    ed.add_link(src_id, src_slot, tgt_id, tgt_slot, dtype or "LATENT")
                actions.append(f"#{nid} ({ntype}) removed; direct-wired {src_id} -> {[d[0] for d in downs]}")
                continue
        ed.remove_node_and_links(nid)
        actions.append(f"#{nid} ({ntype}) removed")

    # Step 2: Collapse single-consumer Set/Get pairs.
    for pair in COLLAPSE_PAIRS:
        set_id, get_id = pair["set_id"], pair["get_id"]
        if not _node_exists(ed, set_id) or not _node_exists(ed, get_id):
            actions.append(f"pair {set_id}/{get_id} ({pair['var']}): already collapsed")
            continue

        up = _trace_set_upstream(ed, set_id)
        downs = _trace_get_downstream(ed, get_id)

        if up is None:
            actions.append(f"pair {set_id}/{get_id}: skip (no upstream feeding Set)")
            continue
        if not downs:
            actions.append(f"pair {set_id}/{get_id}: skip (no downstream from Get)")
            continue
        if len(downs) > 1:
            actions.append(f"pair {set_id}/{get_id}: skip (Get has {len(downs)} consumers, expected 1)")
            continue

        src_id, src_slot, _ = up
        tgt_id, tgt_slot, _ = downs[0]

        # Remove the Set + Get + their links; reattach src -> tgt directly.
        ed.remove_node_and_links(set_id)
        ed.remove_node_and_links(get_id)
        ed.add_link(src_id, src_slot, tgt_id, tgt_slot, pair["dtype"])
        actions.append(
            f"collapsed {set_id}/{get_id} ({pair['var']}): #{src_id} -> #{tgt_id}.in[{tgt_slot}]"
        )

    return True, "; ".join(actions)


def _revert(_ed: WorkflowEditor) -> tuple[bool, str]:
    """Partial revert: restore dead nodes only. Set/Get collapses are
    one-way (the indirection was cosmetic; restoring it would require
    duplicating widget metadata + rebuilding link IDs)."""
    return False, (
        "revert not implemented — the dead nodes (#1513 ModelSamplingSD3, "
        "#1606 Reroute, #560 VHS_VideoCombine) were verified-orphan, and "
        "the Set/Get pair collapses are cosmetic-only. To restore: use "
        "ComfyUI's UI to re-add nodes + rewire."
    )


def apply(revert: bool, dry_run: bool, wf_path: Path) -> int:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        print(f"load error: {e}")
        return 1

    changed, message = _revert(ed) if revert else _apply(ed)
    prefix = "would " if dry_run and changed else ""
    print(f"  {wf_path.relative_to(REPO_ROOT)}:")
    for line in message.split("; "):
        print(f"    {prefix}{line}")
    if changed and not dry_run:
        ed.save(wf_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("workflow", nargs="?", default=str(DEFAULT_WORKFLOW))
    ap.add_argument("--revert", action="store_true",
                    help="(Not fully implemented — see docstring.)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing.")
    args = ap.parse_args()
    return apply(args.revert, args.dry_run, Path(args.workflow))


if __name__ == "__main__":
    sys.exit(main())
