"""apply_p3_retake_edit_lora — wire IC-LoRA edit-anything pattern into retake.

Last updated: 2026-05-04

Stages a section-targeted retake-edit variant of the canonical retake
workflow at `internal/workflows/retake_edit.draft.json`. Adds:

  1. `LTXICLoRALoaderModelOnly` patching MODEL with the edit-anything
     LoRA (4-verb training: ADD / REMOVE / REPLACE / RESTYLE). Inserted
     between `AudioLoopHelperSageAttention` (#268) and `LTXVChunkFeedForward`
     (#504) — per the project gotcha "canonical order for compile-style
     patches: UNETLoader → ... → LTXICLoRALoaderModelOnly →
     <module-mutating node>". Module-mutating nodes (chunk-FFW + attn-tuner)
     call `model.state_dict()`; the IC-LoRA loader must precede them.

  2. `LTXVAddGuideMulti` (strength=1, frame_idx=0, num_guides=1) inserted
     in the conditioning + latent path. Image source: the same pixel
     stream feeding the existing `VAEEncode` (#1621) — the loaded source
     video. The node consumes positive/negative CONDITIONING from
     `LTXVConditioning` (#164) and the masked LATENT from
     `LatentTemporalMask` (#1622), and outputs re-conditioned positive/
     negative + a guide-baked LATENT to feed `SamplerCustomAdvanced` (#161).

  3. A `Note` node titled "Edit prompt usage" pointing the user to the
     existing positive `CLIPTextEncode` (#169) — that prompt becomes the
     edit instruction (e.g. `"change the jacket to red"` or `"remove
     the crowd member"`) per the LoRA's training distribution.

Symptom / motivation: P3 of the polish-passes design (D-EA-5 in
`internal/analysis/edit_anything_workflow_analysis.md`). Phase 3
`LatentTemporalMask` shipped 2026-04-25; the LoRA + IC-LoRA wiring
turns "re-roll the bad section" into "edit the bad section" with the
four edit verbs.

Root cause of the deferred state: gated on a user cfg=1 A/B render
confirming the four verbs land at distilled CFG=1. This script
delivers the WIRING; the gate is still open until the A/B lands.

Fix / change applied: the three node insertions described above plus
their links. No widget value changes on existing nodes.

Compatibility with other apply scripts:
  - Reads `example_workflows/audio-loop-music-video_retake.json`;
    does not mutate the source.
  - Idempotent (signature: an `LTXICLoRALoaderModelOnly` whose first
    widget value is the edit-anything LoRA filename means we've
    already run).
  - `--revert` deletes the staged output file.
  - `--dry-run` reports planned ops without writing.

Usage:
    uv run --group dev python scripts/apply_p3_retake_edit_lora.py
    uv run --group dev python scripts/apply_p3_retake_edit_lora.py --dry-run
    uv run --group dev python scripts/apply_p3_retake_edit_lora.py --revert

Default I/O:
    --input  example_workflows/audio-loop-music-video_retake.json
    --output internal/workflows/retake_edit.draft.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Required source nodes (canonical retake topology).
SAGE_NODE_ID = 268            # AudioLoopHelperSageAttention — predecessor in MODEL chain
CHUNK_FFW_NODE_ID = 504       # LTXVChunkFeedForward — successor in MODEL chain
LTX_COND_NODE_ID = 164        # LTXVConditioning — feeds positive/negative
LATENT_MASK_NODE_ID = 1622    # LatentTemporalMask — outputs masked latent
SAMPLER_NODE_ID = 161         # SamplerCustomAdvanced — consumes positive + latent
VAE_GETNODE_ID = 413          # GetNode "Get_video_vae" — VAE for the guide
PIXEL_SOURCE_NODE_ID = 1620   # GetVideoComponents — pixel source for image_1

REQUIRED_NODES = (
    SAGE_NODE_ID, CHUNK_FFW_NODE_ID, LTX_COND_NODE_ID,
    LATENT_MASK_NODE_ID, SAMPLER_NODE_ID, VAE_GETNODE_ID, PIXEL_SOURCE_NODE_ID,
)

# The edit-anything LoRA. Trained on ADD / REMOVE / REPLACE / RESTYLE
# prompt patterns; rank-128, ~9000 steps. User must have this on disk.
EDIT_ANYTHING_LORA = "ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors"

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_retake.json"
DEFAULT_OUTPUT = "internal/workflows/retake_edit.draft.json"


def _find_edit_iclora(ed: WorkflowEditor) -> dict | None:
    """Return the IC-LoRA loader patched with the edit-anything LoRA, if present."""
    for n in ed.find_nodes_by_type("LTXICLoRALoaderModelOnly"):
        wv = n.get("widgets_values") or []
        if wv and isinstance(wv[0], str) and EDIT_ANYTHING_LORA in wv[0]:
            return n
    return None


def _already_migrated(ed: WorkflowEditor) -> bool:
    return _find_edit_iclora(ed) is not None


def _apply(ed: WorkflowEditor, dry_run: bool) -> None:
    if _already_migrated(ed):
        print(f"  {ed.path.name}: edit-anything IC-LoRA already wired, skipping.")
        return

    missing = ed.require_nodes(REQUIRED_NODES)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source nodes missing {missing}. "
            "Source workflow may have drifted from the canonical retake topology."
        )

    if dry_run:
        print(f"  {ed.path.name}:")
        print(f"    would add LTXICLoRALoaderModelOnly between #{SAGE_NODE_ID} and #{CHUNK_FFW_NODE_ID}")
        print(f"    would add LTXVAddGuideMulti between #{LATENT_MASK_NODE_ID} and #{SAMPLER_NODE_ID}")
        print(f"    would add Note node explaining edit-prompt usage")
        return

    sage = ed.find_node(SAGE_NODE_ID)
    chunk_ffw = ed.find_node(CHUNK_FFW_NODE_ID)
    sampler = ed.find_node(SAMPLER_NODE_ID)

    # ---- 1. LTXICLoRALoaderModelOnly: insert between sage and chunk-FFW ----
    iclora_id = ed.add_top_level_node(
        node_type="LTXICLoRALoaderModelOnly",
        pos=[sage["pos"][0] + 320, sage["pos"][1]],
        size=[480, 102],
        inputs=[{"name": "model", "type": "MODEL", "link": None}],
        outputs=[
            {"name": "model", "type": "MODEL", "links": []},
            {"name": "latent_downscale_factor", "type": "FLOAT", "links": []},
        ],
        widgets_values=[EDIT_ANYTHING_LORA, 1.0],
        title="Edit-anything LoRA (P3)",
    )
    # Rewire chunk_ffw's MODEL input from sage → new iclora.
    chunk_ffw_model_slot = WorkflowEditor.find_input_slot(chunk_ffw, "model")
    sage_to_chunk = ed.find_link_to_slot(CHUNK_FFW_NODE_ID, chunk_ffw_model_slot)
    if sage_to_chunk is None:
        raise SystemExit(
            f"#{CHUNK_FFW_NODE_ID}.model has no inbound link — MODEL chain unexpectedly broken."
        )
    ed.rewire_input(iclora_id, 0, sage_to_chunk[1], sage_to_chunk[2], "MODEL")
    ed.rewire_input(CHUNK_FFW_NODE_ID, chunk_ffw_model_slot, iclora_id, 0, "MODEL")

    # ---- 2. LTXVAddGuideMulti: between LatentTemporalMask and sampler ----
    # Source positive/negative come from LTXVConditioning (#164).
    # The mask latent currently flows: #1622 → ... → #161.latent_image.
    # We hook into THAT path: new node consumes the inbound link to
    # SamplerCustomAdvanced.latent_image and feeds its latent output back.
    sampler_latent_slot = WorkflowEditor.find_input_slot(sampler, "latent_image")
    latent_to_sampler = ed.find_link_to_slot(SAMPLER_NODE_ID, sampler_latent_slot)
    if latent_to_sampler is None:
        raise SystemExit(
            f"#{SAMPLER_NODE_ID}.latent_image has no inbound link — retake topology broken."
        )

    guide_id = ed.add_top_level_node(
        node_type="LTXVAddGuideMulti",
        pos=[sampler["pos"][0] - 380, sampler["pos"][1] - 80],
        size=[330, 130],
        inputs=[
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
            {"name": "latent", "type": "LATENT", "link": None},
            {"name": "num_guides.image_1", "type": "IMAGE", "shape": 7, "link": None},
        ],
        outputs=[
            {"name": "positive", "type": "CONDITIONING", "links": []},
            {"name": "negative", "type": "CONDITIONING", "links": []},
            {"name": "latent", "type": "LATENT", "links": []},
        ],
        widgets_values=["1", 0, 1.0],  # num_guides=1, frame_idx=0, strength=1
        title="Edit guide (image_1 = source pixels)",
    )

    # Wire LTXVAddGuideMulti inputs.
    # positive/negative: from LTXVConditioning (#164) outputs 0/1.
    ed.rewire_input(guide_id, 0, LTX_COND_NODE_ID, 0, "CONDITIONING")
    ed.rewire_input(guide_id, 1, LTX_COND_NODE_ID, 1, "CONDITIONING")
    # vae: from existing VAE GetNode.
    ed.rewire_input(guide_id, 2, VAE_GETNODE_ID, 0, "VAE")
    # latent: from whatever currently feeds the sampler's latent_image (was the mask path).
    ed.rewire_input(guide_id, 3, latent_to_sampler[1], latent_to_sampler[2], "LATENT")
    # image_1: source pixels (same stream feeding VAEEncode).
    ed.rewire_input(guide_id, 4, PIXEL_SOURCE_NODE_ID, 0, "IMAGE")

    # Reroute sampler's latent_image input to the guide's latent output.
    ed.rewire_input(SAMPLER_NODE_ID, sampler_latent_slot, guide_id, 2, "LATENT")

    # ---- 3. Note node explaining edit-prompt usage ----
    ed.add_top_level_node(
        node_type="Note",
        pos=[sampler["pos"][0] - 380, sampler["pos"][1] + 80],
        size=[330, 100],
        inputs=[],
        outputs=[],
        widgets_values=[
            "P3 retake-edit usage:\n"
            "  * Set the retake range on LatentTemporalMask (start, end, fps).\n"
            "  * Set the positive prompt (CLIPTextEncode) to an edit verb:\n"
            "    add / remove / replace / restyle. The LoRA was trained on\n"
            "    these four verb patterns.\n"
            "  * Distilled CFG=1 — verb landing is gated on user A/B."
        ],
    )

    print(
        f"  {ed.path.name}: inserted #{iclora_id} (LTXICLoRALoaderModelOnly) + "
        f"#{guide_id} (LTXVAddGuideMulti) + Note node"
    )


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    """Stage `output_path` from `input_path`, then apply the migration.

    Idempotent: skips if the output already has the migration applied.
    To re-sync the draft with upstream bug fixes to the source workflow,
    run `--revert` first then re-apply.
    """
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if output_path.exists() and input_path != output_path:
        existing = WorkflowEditor(output_path)
        if _already_migrated(existing):
            print(
                f"  {output_path.relative_to(REPO_ROOT)}: already migrated, skipping. "
                "Run --revert then re-apply to pull upstream bug fixes from source."
            )
            return

    if not dry_run:
        if input_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            print(f"  copied {input_path.relative_to(REPO_ROOT)} -> {output_path.relative_to(REPO_ROOT)}")

    target = output_path if not dry_run else input_path
    ed = WorkflowEditor(target)
    _apply(ed, dry_run=dry_run)
    if not dry_run:
        ed.save()


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path.relative_to(REPO_ROOT)}")
    else:
        print(f"{output_path.relative_to(REPO_ROOT)} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    in_path = (REPO_ROOT / args.input).resolve()
    out_path = (REPO_ROOT / args.output).resolve()

    if args.revert:
        _revert(out_path)
        return

    _migrate(in_path, out_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
