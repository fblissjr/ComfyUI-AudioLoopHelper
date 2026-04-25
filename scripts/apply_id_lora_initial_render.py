"""apply_id_lora_initial_render.

Last updated: 2026-04-25

Stages an ID-LoRA / style-LoRA variant of the canonical latent workflow.
Inserts a single `LoraLoaderModelOnly` between
`LTX2SamplingPreviewOverride(503)` and `SetNode(572)` so the patched
MODEL flows to BOTH the initial render AND the loop subgraph (via the
existing Set/Get(654) -> LoopIterationStamp(1618) chain).

Sibling of `apply_iclora_initial_render.py` (Phase 0a). The mechanism is
deliberately simpler:

  - IC-LoRA needs `LTXAddVideoICLoRAGuide` because it conditions on a
    REFERENCE IMAGE injected into `guide_attention_entries`.
  - ID-LoRA / style LoRAs encode the identity or style in the LoRA
    weights themselves; no per-iter image guide is required. Just patch
    MODEL and let cross-attention pick up the new prior.

Same MODEL-fork question Phase 0a opened applies here: a LoRA-patched
MODEL flows through every loop iteration, not just the first. If you
observe that loop iters 2+ degrade vs the canonical baseline (artifacting,
washed-out features) the loader is the suspect; fork the MODEL via a
parallel SetNode if so. See `internal/ic_lora_assessment.md` D2 for the
optimistic-coupling rationale.

For amplifying identity / style adherence past the LoRA's trained
strength, layer TTC1 (CFG-analog amplification) on top — feed
`(positive_with_lora, positive_without_lora)` to `CFGGuider` as
`(positive, negative)`. The existing sampler computes the differential
per denoising step. The mechanism + canonical wiring lives in
`scripts/apply_ttc_init_guide_amplification_poc.py` and
`scripts/apply_ttc_iclora_amplification_poc.py`; the (with, without)
streams just need to be `(model_with_id_lora, model_without_id_lora)`
forks instead. Out of scope for this script (single-knob loader is the
right starting point); reference for follow-up.

Usage:
    uv run --group dev python scripts/apply_id_lora_initial_render.py
    uv run --group dev python scripts/apply_id_lora_initial_render.py \\
        --lora-file my_identity_lora.safetensors --strength 0.85
    uv run --group dev python scripts/apply_id_lora_initial_render.py --revert

Idempotent on the OUTPUT path; `--revert` deletes the staging file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor


# Source workflow nodes that must exist for the migration to apply.
# Insertion point mirrors Phase 0a: SetNode 572 is the join point that
# feeds both the initial CFGGuider and the loop subgraph's MODEL slot.
LTX2_PREVIEW_OVERRIDE_ID = 503        # MODEL output -> SetNode(572) "model"
SETNODE_MODEL_ID = 572                # Set "model" -> Get 654 -> initial CFGGuider AND loop subgraph

REQUIRED_SOURCE_NODES = (
    LTX2_PREVIEW_OVERRIDE_ID,
    SETNODE_MODEL_ID,
)

# ID-LoRA defaults. The placeholder filename matches the upstream
# `LTX-2_T2V_Distilled_wLora.json` convention ("your_*_lora.safetensors")
# so the unmodified staging file is loud about needing a real LoRA name
# before render. Override via --lora-file.
DEFAULT_ID_LORA_FILE = "your_id_lora.safetensors"
DEFAULT_ID_LORA_STRENGTH = 0.9

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/scratch/audio-loop-music-video_latent_id_lora.json"

# Node-title marker used as a robust idempotence signal — `LoraLoaderModelOnly`
# is a generic core node so type alone is not specific enough for sibling
# scripts that may also stack one (e.g. a future style-LoRA chain).
ID_LORA_TITLE = "ID-LoRA Loader (initial-render + loop chain)"


def _find_id_lora_node(ed: WorkflowEditor) -> dict | None:
    for n in ed.find_nodes_by_type("LoraLoaderModelOnly"):
        if n.get("title") == ID_LORA_TITLE:
            return n
    return None


def _already_migrated(ed: WorkflowEditor) -> bool:
    return _find_id_lora_node(ed) is not None


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = []
    for nid in REQUIRED_SOURCE_NODES:
        try:
            ed.find_node(nid)
        except ValueError:
            missing.append(nid)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source node(s) missing: {missing}. "
            "This script assumes the canonical latent workflow layout. "
            "If your workflow was edited, update the constants at the top "
            "of this script."
        )


def _add_id_lora_loader(ed: WorkflowEditor, lora_file: str, strength: float) -> int:
    """Insert LoraLoaderModelOnly between LTX2SamplingPreviewOverride(503)
    and SetNode(572). Single MODEL in -> single MODEL out; widget order
    is `[lora_name, strength_model]` (verified against upstream
    LTX-2_T2V_Distilled_wLora.json)."""
    nid = ed.add_top_level_node(
        node_type="LoraLoaderModelOnly",
        pos=[-2200, 1100],
        size=[360, 90],
        inputs=[
            {"name": "model", "type": "MODEL", "link": None},
        ],
        outputs=[
            {"name": "MODEL", "type": "MODEL", "links": []},
        ],
        widgets_values=[lora_file, strength],
        properties={
            "cnr_id": "comfy-core",
            "Node name for S&R": "LoraLoaderModelOnly",
        },
        title=ID_LORA_TITLE,
    )

    # Splice into the model chain: 503 -> SetNode(572) becomes 503 -> loader -> SetNode(572).
    existing = ed.find_link_to_slot(SETNODE_MODEL_ID, 0)
    if existing is None:
        raise SystemExit(
            f"SetNode({SETNODE_MODEL_ID}).MODEL has no inbound link; "
            "workflow shape is unexpected."
        )
    old_link_id = existing[0]
    ed.remove_link(old_link_id)
    ed.add_link(LTX2_PREVIEW_OVERRIDE_ID, 0, nid, 0, "MODEL")
    ed.add_link(nid, 0, SETNODE_MODEL_ID, 0, "MODEL")
    return nid


def _migrate(input_path: Path, output_path: Path, lora_file: str, strength: float) -> None:
    # Preserve user edits: if the staging file is already migrated, bail
    # before overwriting. --revert deletes it for a fresh start.
    if output_path.exists() and input_path != output_path and _already_migrated(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
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

    print(f"{output_path.name}: applying ID-LoRA wiring...")
    loader_id = _add_id_lora_loader(ed, lora_file, strength)
    print(f"  added LoraLoaderModelOnly as node {loader_id} "
          f"(lora: {lora_file}, strength: {strength})")

    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    if lora_file == DEFAULT_ID_LORA_FILE:
        print(f"  2. Replace placeholder LoRA filename '{DEFAULT_ID_LORA_FILE}' on the loader node")
        print(f"     (or re-run with --lora-file <real_filename>).")
        print(f"  3. Open in ComfyUI and A/B render against the canonical baseline.")
    else:
        print(f"  2. Open in ComfyUI: load {output_path}")
        print(f"  3. A/B render against the canonical baseline (same seed, same prompts).")
    print( "  4. Look for: changed identity/style across the FULL run (initial + loop iters).")
    print( "     If iters 2+ degrade vs canonical: open MODEL-fork question (D2 in")
    print( "     internal/ic_lora_assessment.md). Fix by forking MODEL so the loop")
    print( "     subgraph reads from an unpatched chain.")
    print( "  5. To amplify identity adherence past trained strength, see")
    print( "     scripts/apply_ttc_iclora_amplification_poc.py for the (with, without)")
    print( "     CFGGuider wiring pattern; swap the conditional to MODEL forks.")


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
    ap.add_argument(
        "--input", default=DEFAULT_INPUT,
        help="Source workflow to fork from (default: %(default)s)",
    )
    ap.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output workflow path (default: %(default)s)",
    )
    ap.add_argument(
        "--lora-file", default=DEFAULT_ID_LORA_FILE,
        help="ID-LoRA / style LoRA filename in models/loras/ "
             "(default: %(default)s placeholder — replace before render)",
    )
    ap.add_argument(
        "--strength", type=float, default=DEFAULT_ID_LORA_STRENGTH,
        help="LoRA strength_model (default: %(default)s)",
    )
    ap.add_argument(
        "--revert", action="store_true",
        help="Delete the output staging file (does not touch --input).",
    )
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return

    _migrate(Path(args.input), output_path, args.lora_file, args.strength)


if __name__ == "__main__":
    main()
