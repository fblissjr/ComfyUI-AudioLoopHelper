"""Phase 1: wire kijai's PromptRelayEncode onto the INITIAL-RENDER path.

Last updated: 2026-04-24

Forks `example_workflows/audio-loop-music-video_latent.json` into a
staging copy under `internal/scratch/`, splicing in one
ComfyUI-PromptRelay node:

  - `PromptRelayEncode` between `LTX2AttentionTunerPatch(1523)`
    (MODEL fork-upstream point) and the initial `CFGGuider(153)`, and
    between `CLIPTextEncode(169)` and `LTXVConditioning(164).positive`
    on the CONDITIONING chain. `PromptRelayEncode` encodes the baseline
    global prompt concatenated with local per-segment prompts and patches
    the MODEL's `attn2` / `audio_attn2` forwards with a Gaussian temporal
    penalty on cross-attention keys.

MODEL fork rationale:
  - PromptRelayEncode installs `object_patches` via
    `model.add_object_patch(...)`. Per
    `docs/analysis/nag_object_patches_offload_asymmetry.md`, these are
    stripped by the TensorLoop offload/reload cycle. The patched MODEL
    must NOT reach the loop subgraph.
  - KJNodes `LTX2_NAG(508)` patches the same `attn2.forward` keys and
    would conflict with PromptRelayEncode's `_check_unpatched` guard, so
    the fork splits upstream of NAG. The initial-render branch runs
    WITHOUT NAG negative-prompt injection for Phase 1.
  - Loop subgraph continues to consume the existing
    `1523 -> 508 -> 503 -> SetNode(572) -> GetNode(654)` chain intact.

Conditioning rewire rationale:
  - PromptRelayEncode emits its own CONDITIONING by encoding
    `global_prompt + local_prompts`. Node 169's original output is
    replaced as the `LTXVConditioning(164).positive` source. Node 169 is
    left in the workflow (orphaned) so its widget remains the source of
    truth for manual sync back into `PromptRelayEncode.global_prompt`.
  - `frame_rate` stamping is preserved because `LTXVConditioning(164)`
    still sits downstream of `PromptRelayEncode`.

Defaults are deliberately coarse (two visibly-distinct segments) so the
A/B signal is unambiguous. Tune after the first render.

Usage:
    uv run --group dev python scripts/apply_prompt_relay_initial_render.py
    uv run --group dev python scripts/apply_prompt_relay_initial_render.py --dry-run
    uv run --group dev python scripts/apply_prompt_relay_initial_render.py --revert

Idempotent on the OUTPUT path. `--revert` deletes the staging file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor


# Canonical workflow node IDs the script keys off.
ATTENTION_TUNER_ID = 1523   # LTX2AttentionTunerPatch -- MODEL fork-upstream point
LTX2_NAG_ID = 508           # LTX2_NAG -- unchanged; stays on loop branch
SETNODE_MODEL_ID = 572      # SetNode Set_model -- unchanged; loop-branch storage
CFGGUIDER_ID = 153          # initial-render CFGGuider
LTXVCONDITIONING_ID = 164   # initial-render LTXVConditioning
CLIPTEXTENCODE_POS_ID = 169 # baseline positive prompt (orphaned post-apply)
DUAL_CLIP_LOADER_ID = 416   # DualCLIPLoader -- shared CLIP source
EMPTY_LATENT_ID = 344       # EmptyLTXVLatentVideo -- shape inference source

REQUIRED_SOURCE_NODES = (
    ATTENTION_TUNER_ID,
    LTX2_NAG_ID,
    SETNODE_MODEL_ID,
    CFGGUIDER_ID,
    LTXVCONDITIONING_ID,
    CLIPTEXTENCODE_POS_ID,
    DUAL_CLIP_LOADER_ID,
    EMPTY_LATENT_ID,
)

# Phase 1 defaults. Two gross-difference segments so the A/B is unambiguous.
DEFAULT_LOCAL_PROMPTS = (
    "wide establishing shot, full-body framing | "
    "tight closeup, shallow depth of field"
)
DEFAULT_SEGMENT_LENGTHS = ""   # empty -> PromptRelayEncode auto-distributes
DEFAULT_EPSILON = 0.001         # paper default; sharp boundaries

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/scratch/audio-loop-music-video_latent_prompt_relay_phase1.json"


def _already_migrated(ed: WorkflowEditor) -> bool:
    return bool(ed.find_nodes_by_type("PromptRelayEncode"))


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = ed.require_nodes(REQUIRED_SOURCE_NODES)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source node(s) missing: {missing}. "
            "This script assumes the canonical latent workflow layout."
        )


def _baseline_global_prompt(ed: WorkflowEditor) -> str:
    """Read Node 169's widget as the baseline positive prompt.

    CLAUDE.md: Node 169 prompt matches schedule 0:00 entry structurally.
    Phase 1 copies this string into PromptRelayEncode.global_prompt once
    at apply time. Future changes to Node 169 require re-running this
    script (or manually syncing PromptRelayEncode's widget).
    """
    node = ed.find_node(CLIPTEXTENCODE_POS_ID)
    widgets = node.get("widgets_values", [])
    if not widgets or not isinstance(widgets[0], str):
        raise SystemExit(
            f"CLIPTextEncode({CLIPTEXTENCODE_POS_ID}) has no string widget; "
            "cannot derive global_prompt."
        )
    return widgets[0]


def _add_prompt_relay_encode(ed: WorkflowEditor, global_prompt: str) -> int:
    return ed.add_top_level_node(
        node_type="PromptRelayEncode",
        pos=[-1800, 600],
        size=[420, 300],
        inputs=[
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "clip", "type": "CLIP", "link": None},
            {"name": "latent", "type": "LATENT", "link": None},
        ],
        outputs=[
            {"name": "model", "type": "MODEL", "links": []},
            {"name": "positive", "type": "CONDITIONING", "links": []},
        ],
        widgets_values=[
            global_prompt,
            DEFAULT_LOCAL_PROMPTS,
            DEFAULT_SEGMENT_LENGTHS,
            DEFAULT_EPSILON,
        ],
        properties={
            "cnr_id": "ComfyUI-PromptRelay",
            "Node name for S&R": "PromptRelayEncode",
        },
        title="Prompt Relay (initial-render)",
    )


def _wire_prompt_relay_inputs(ed: WorkflowEditor, relay_id: int) -> None:
    # MODEL forks upstream of LTX2_NAG(508) so its attn2-patch keys don't
    # collide with NAG's (PromptRelay's _check_unpatched would raise otherwise).
    ed.add_link(ATTENTION_TUNER_ID, 0, relay_id, 0, "MODEL")
    ed.add_link(DUAL_CLIP_LOADER_ID, 0, relay_id, 1, "CLIP")
    ed.add_link(EMPTY_LATENT_ID, 0, relay_id, 2, "LATENT")


def _reroute_outputs(ed: WorkflowEditor, relay_id: int) -> None:
    """Swing CFGGuider(153).model and LTXVConditioning(164).positive
    onto PromptRelayEncode's outputs."""
    ed.rewire_input(CFGGUIDER_ID, 0, relay_id, 0, "MODEL")
    ed.rewire_input(LTXVCONDITIONING_ID, 0, relay_id, 1, "CONDITIONING")


def _apply_ops(ed: WorkflowEditor) -> int:
    global_prompt = _baseline_global_prompt(ed)
    relay_id = _add_prompt_relay_encode(ed, global_prompt)
    _wire_prompt_relay_inputs(ed, relay_id)
    _reroute_outputs(ed, relay_id)
    return relay_id


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if output_path.exists() and input_path != output_path and _already_migrated(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
        return

    if dry_run:
        ed = WorkflowEditor(input_path)
        _assert_required_nodes_present(ed)
        global_prompt = _baseline_global_prompt(ed)
        print(f"would copy {input_path} -> {output_path}")
        print("would insert PromptRelayEncode with:")
        print(f"  model        <- Node {ATTENTION_TUNER_ID} (LTX2AttentionTunerPatch)")
        print(f"  clip         <- Node {DUAL_CLIP_LOADER_ID} (DualCLIPLoader)")
        print(f"  latent       <- Node {EMPTY_LATENT_ID} (EmptyLTXVLatentVideo)")
        print(f"  global_prompt = {global_prompt!r}")
        print(f"  local_prompts = {DEFAULT_LOCAL_PROMPTS!r}")
        print(f"would rewire CFGGuider({CFGGUIDER_ID}).model <- PromptRelayEncode.model")
        print(f"would rewire LTXVConditioning({LTXVCONDITIONING_ID}).positive "
              "<- PromptRelayEncode.positive")
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
    print(f"{output_path.name}: applying Phase 1 PromptRelay wiring...")
    relay_id = _apply_ops(ed)
    print(f"  added PromptRelayEncode as node {relay_id}")
    print(f"  locals: {DEFAULT_LOCAL_PROMPTS!r}")
    print(f"  epsilon: {DEFAULT_EPSILON}")

    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Open in ComfyUI: load {output_path}")
    print( "  3. A/B render against the canonical baseline (same seed, same prompts).")
    print( "  4. Verify per-iter sage trace unchanged vs canonical:")
    print( "       scripts/verify_sage_iteration_trace.sh")


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
