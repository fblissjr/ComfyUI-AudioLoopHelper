#!/usr/bin/env python3
"""Build example_workflows/experimental/keyframe-i2v_audio-ic-lora.json.

Forks the shipped single-pass audio IC-LoRA workflow
(`example_workflows/audio-ic-lora_single-pass.json`) and splices in the
`KeyframeGuidesTimeSpaced` node so keyframe images drive the picture while
the audio IC-LoRA reference still steers audio attributes. The model
generates audio+video jointly in one distilled pass.

WHY this ordering (see the workflow MarkdownNote + the agent risk note):

  Latent plane (forced):
    EmptyLTXVLatentVideo(#344)  ->  KeyframeGuidesTimeSpaced.latent
    KeyframeGuidesTimeSpaced.latent_out  ->  LTXVConcatAVLatent(#350).video_latent
  The keyframe node calls core `LTXVAddGuide.append_keyframe`, which RAISES
  on a combined AV latent (shape[1] != 128). So it MUST run on the
  video-only `EmptyLTXVLatentVideo` BEFORE the AV concat. Its output video
  latent (samples + noise_mask grown along the frame axis dim=2) then feeds
  the concat's video_latent input. `LTXVConcatAVLatent` wraps video+audio
  as a per-stream NestedTensor (independent shapes), so a guide-extended
  video stream composes fine.

  Conditioning plane:
    LTXVConditioning(#164)  ->  KeyframeGuidesTimeSpaced.positive/negative
    KeyframeGuidesTimeSpaced.positive/negative  ->  LTXAddAudioICLoRAGuideAdvanced(#1996)
  The keyframe node stamps `keyframe_idxs` (+ guide-attention entries) onto
  the conditioning; the audio IC-LoRA guide stamps `ref_audio`. ORTHOGONAL
  keys, merged non-destructively by `conditioning_set_values`, so order
  between them doesn't collide on the dict. Keyframe node goes AFTER
  LTXVConditioning so the conditioning already carries frame_rate=25 (the
  node reads it to warn on an output_fps mismatch).

  Decode plane (crop the appended guide frames):
    sampler(#1845) -> LTXVSeparateAVLatent(#1827).video_latent ->
    LTXVCropGuides.latent -> LTXVTiledVAEDecode(#1995).latents
  `append_keyframe` appended N guide frames at the tail of the video latent;
  they must be cropped after sampling. `LTXVCropGuides` slices a single
  video-stream latent (dim=2), so it runs on the SEPARATED video latent
  (it can't slice the AV NestedTensor). Its conditioning inputs come from
  #1996's outputs (which carry keyframe_idxs); its conditioning outputs are
  unused here (we only need the latent crop).

Keyframe image source: a `VHS_LoadVideo` at force_rate=1 (one keyframe per
second) feeds KeyframeGuidesTimeSpaced.images; seconds_per_keyframe=1.0,
output_fps=25 (matches LTXVConditioning.frame_rate).

This file is staged into example_workflows/experimental/ because it CANNOT
be GPU-tested in this environment — it is scaffolded-but-not-runtime-
validated. Structural integrity + audit are clean; the joint
keyframe-guide + audio-IC-LoRA + AV-latent path is the runtime unknown the
human must verify on GPU.

CLI:
    uv run --group dev python scripts/build_keyframe_i2v_audio_iclora.py
    uv run --group dev python scripts/build_keyframe_i2v_audio_iclora.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import WorkflowEditor, EXAMPLE_WORKFLOWS_DIR  # type: ignore

BASE = EXAMPLE_WORKFLOWS_DIR / "audio-ic-lora_single-pass.json"
OUT = EXAMPLE_WORKFLOWS_DIR / "experimental" / "keyframe-i2v_audio-ic-lora.json"

# Base-workflow node ids (verified against the shipped JSON).
N_EMPTY_VIDEO = 344       # EmptyLTXVLatentVideo (video-only latent)
N_CONCAT = 350            # LTXVConcatAVLatent
N_LTXVCOND = 164          # LTXVConditioning (stamps frame_rate=25)
N_AUDIO_GUIDE = 1996      # LTXAddAudioICLoRAGuideAdvanced
N_SEPARATE = 1827         # LTXVSeparateAVLatent (post-sampler)
N_DECODE = 1995           # LTXVTiledVAEDecode (video)
N_VIDEO_VAE = 1559        # VAELoaderKJ (video VAE)
N_SAMPLER = 1853          # KSamplerSelect

NOTE_TEXT = (
    "## Keyframe-driven i2v + Audio IC-LoRA (single pass)\n\n"
    "Forked from `audio-ic-lora_single-pass.json`. Keyframe images drive the "
    "PICTURE; the audio IC-LoRA reference steers AUDIO attributes; the model "
    "generates audio+video jointly in one distilled 8-step pass.\n\n"
    "### Wiring (the splice)\n"
    "- **Latent:** `EmptyLTXVLatentVideo` -> **KeyframeGuidesTimeSpaced** -> "
    "`LTXVConcatAVLatent.video_latent`. The keyframe node appends guide frames "
    "to the VIDEO-ONLY latent (core `append_keyframe` rejects a combined AV "
    "latent), so it sits BEFORE the AV concat.\n"
    "- **Conditioning:** `LTXVConditioning` -> **KeyframeGuidesTimeSpaced** -> "
    "`LTXAddAudioICLoRAGuideAdvanced` -> `CFGGuider`. Keyframe node stamps "
    "`keyframe_idxs`; audio guide stamps `ref_audio` — orthogonal keys, no "
    "collision.\n"
    "- **Decode:** sampler -> `LTXVSeparateAVLatent` -> **LTXVCropGuides** -> "
    "video decode. Crops the appended keyframe guide-frames off the video "
    "latent tail (operates on the separated single-stream video latent).\n\n"
    "### Keyframe source\n"
    "`VHS_LoadVideo` at **force_rate=1** = one keyframe/second. "
    "`seconds_per_keyframe=1.0`, `output_fps=25` (matches "
    "`LTXVConditioning.frame_rate`). Keyframe[0] at frame 0 = the i2v init; "
    "later keyframes interpolate. Size `EmptyLTXVLatentVideo.length` for the "
    "clip; keyframes past the latent end are dropped (see `placement_info`).\n\n"
    "### COMPOSITION RISK — UNVERIFIED ON GPU\n"
    "Structurally the keyframe-guide and audio-IC-LoRA mechanisms touch "
    "DISJOINT state: the keyframe node grows the video latent + stamps "
    "`keyframe_idxs`; the audio guide only attaches `ref_audio` tokens and "
    "never touches the latent. They merge on the conditioning dict without "
    "key collision, and `LTXVConcatAVLatent` keeps video/audio as independent "
    "NestedTensor streams (so a guide-extended video stream is shape-legal).\n\n"
    "What is NOT verifiable without a render: whether the MODEL attends "
    "coherently to BOTH the keyframe guide frames (which carry their own RoPE "
    "positions via `keyframe_idxs`) AND the audio `ref_audio` tokens in the "
    "SAME forward pass on the distilled CFG=1 base. The audio IC-LoRA was "
    "trained on a no-keyframe path; keyframe anchoring may compete with the "
    "audio reference for cross-attention budget, or the guide-frame RoPE "
    "offsets may interact with the audio token positions. **Test on GPU: "
    "start with strength=1.0 keyframes + a short clip; if audio attributes "
    "wash out, lower keyframe `strength`; if the picture ignores keyframes, "
    "the paths likely don't co-attend and this needs a model-side change.**"
)


def already_built(ed: WorkflowEditor) -> bool:
    return bool(ed.find_nodes_by_type("KeyframeGuidesTimeSpaced"))


def build(dry_run: bool) -> None:
    if not BASE.exists():
        raise SystemExit(f"Base workflow not found: {BASE}")
    ed = WorkflowEditor(BASE)

    # Pre-flight: confirm the base shape we splice into.
    for nid in (N_EMPTY_VIDEO, N_CONCAT, N_LTXVCOND, N_AUDIO_GUIDE, N_SEPARATE, N_DECODE, N_VIDEO_VAE, N_SAMPLER):
        ed.find_node(nid)  # raises ValueError if missing

    if already_built(ed):
        print("KeyframeGuidesTimeSpaced already present — nothing to do (idempotent).")
        return

    # --- 1. Clean the orphan output-cache links the sibling UI re-save left. ---
    pruned = ed.prune_orphan_output_links()
    print(f"pruned {pruned} orphan output-cache link ids")

    # --- 1b. Canonical distilled sampler: euler (NOT euler_ancestral*). The
    # base file ships euler_ancestral_cfg_pp; CLAUDE.md's distilled 8-step path
    # mandates plain euler + the canonical sigmas (already correct here). ---
    sampler = ed.find_node(N_SAMPLER)
    if sampler.get("widgets_values") != ["euler"]:
        print(f"  fixed KSamplerSelect: {sampler.get('widgets_values')} -> ['euler']")
        sampler["widgets_values"] = ["euler"]

    # --- 2. Keyframe image source: VHS_LoadVideo @ 1 fps. ---
    kf_video = ed.add_top_level_node(
        node_type="VHS_LoadVideo",
        pos=[-200, 1100],
        size=[270, 310],
        inputs=[],
        outputs=[
            ed.out("IMAGE", "IMAGE"),
            ed.out("frame_count", "INT"),
            ed.out("audio", "AUDIO"),
            ed.out("video_info", "VHS_VIDEOINFO"),
        ],
        widgets_values={
            "video": "keyframes.mp4",
            "force_rate": 1,          # one keyframe per second
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": 0,      # load all keyframes
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "format": "LTXV",
            "videopreview": {"hidden": False, "paused": False, "params": {
                "filename": "keyframes.mp4", "type": "input", "format": "video/mp4",
                "force_rate": 1, "frame_load_cap": 0, "skip_first_frames": 0,
                "select_every_nth": 1,
            }},
        },
        title="Keyframe source (VHS_LoadVideo, force_rate=1)",
    )

    # --- 3. KeyframeGuidesTimeSpaced node. ---
    # inputs: vae, positive, negative, latent, images, output_fps, seconds_per_keyframe, strength
    # outputs: positive, negative, latent, placement_info
    kf = ed.add_top_level_node(
        node_type="KeyframeGuidesTimeSpaced",
        pos=[200, 1100],
        size=[330, 230],
        inputs=[
            ed.io_in("vae", "VAE"),
            ed.io_in("positive", "CONDITIONING"),
            ed.io_in("negative", "CONDITIONING"),
            ed.io_in("latent", "LATENT"),
            ed.io_in("images", "IMAGE"),
            ed.widget_in("output_fps", "FLOAT"),
            ed.widget_in("seconds_per_keyframe", "FLOAT"),
            ed.widget_in("strength", "FLOAT"),
        ],
        outputs=[
            ed.out("positive", "CONDITIONING"),
            ed.out("negative", "CONDITIONING"),
            ed.out("latent", "LATENT"),
            ed.out("placement_info", "STRING"),
        ],
        widgets_values=[25.0, 1.0, 1.0],  # output_fps, seconds_per_keyframe, strength
        title="Keyframe Guides (Time-Spaced)",
    )

    # --- 4. LTXVCropGuides on the decode side. ---
    crop = ed.add_top_level_node(
        node_type="LTXVCropGuides",
        pos=[200, 1400],
        size=[260, 100],
        inputs=[
            ed.io_in("positive", "CONDITIONING"),
            ed.io_in("negative", "CONDITIONING"),
            ed.io_in("latent", "LATENT"),
        ],
        outputs=[
            ed.out("positive", "CONDITIONING"),
            ed.out("negative", "CONDITIONING"),
            ed.out("latent", "LATENT"),
        ],
        widgets_values=[],
        title="Crop Keyframe Guides (before video decode)",
    )

    # --- 5. MarkdownNote documenting the workflow + the composition risk. ---
    ed.add_top_level_node(
        node_type="MarkdownNote",
        pos=[-200, 1450],
        size=[420, 560],
        inputs=[],
        outputs=[],
        widgets_values=[NOTE_TEXT],
        title="Keyframe i2v + Audio IC-LoRA — read me",
    )

    # === Rewire ===
    # Latent: EmptyLTXVLatentVideo -> KF.latent ; KF.latent_out -> Concat.video_latent
    # (rewire_input replaces whatever currently feeds Concat.video_latent, slot 0)
    ed.add_link(N_EMPTY_VIDEO, 0, kf, 3, "LATENT")          # #344.out0 -> KF.latent (in slot 3)
    ed.rewire_input(N_CONCAT, 0, kf, 2, "LATENT")           # KF.latent (out slot 2) -> Concat.video_latent

    # KF.vae <- video VAE (resize+encode keyframes); same VAE used for decode.
    ed.add_link(N_VIDEO_VAE, 0, kf, 0, "VAE")               # #1559.out0 -> KF.vae

    # KF images <- keyframe video.
    ed.add_link(kf_video, 0, kf, 4, "IMAGE")                # VHS.IMAGE -> KF.images (in slot 4)

    # Conditioning: LTXVConditioning -> KF.pos/neg ; KF.pos/neg -> AudioGuide.pos/neg
    ed.add_link(N_LTXVCOND, 0, kf, 1, "CONDITIONING")       # #164.pos -> KF.positive (in slot 1)
    ed.add_link(N_LTXVCOND, 1, kf, 2, "CONDITIONING")       # #164.neg -> KF.negative (in slot 2)
    ed.rewire_input(N_AUDIO_GUIDE, 0, kf, 0, "CONDITIONING")  # KF.positive (out slot 0) -> #1996.positive
    ed.rewire_input(N_AUDIO_GUIDE, 1, kf, 1, "CONDITIONING")  # KF.negative (out slot 1) -> #1996.negative

    # Decode: Separate.video_latent -> CropGuides.latent -> Decode.latents
    # CropGuides conditioning from the audio-guide outputs (carry keyframe_idxs).
    ed.add_link(N_AUDIO_GUIDE, 0, crop, 0, "CONDITIONING")  # #1996.pos -> crop.positive
    ed.add_link(N_AUDIO_GUIDE, 1, crop, 1, "CONDITIONING")  # #1996.neg -> crop.negative
    ed.add_link(N_SEPARATE, 0, crop, 2, "LATENT")           # #1827.video_latent -> crop.latent
    ed.rewire_input(N_DECODE, 1, crop, 2, "LATENT")         # crop.latent (out slot 2) -> #1995.latents

    # Final orphan sweep (rewire_input drops links cleanly, but be safe + idempotent).
    ed.prune_orphan_output_links()

    if dry_run:
        print(f"[dry-run] would write {OUT}")
        print(f"  + VHS_LoadVideo (#{kf_video}), KeyframeGuidesTimeSpaced (#{kf}), "
              f"LTXVCropGuides (#{crop}) + MarkdownNote")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ed.save(OUT)
    print(f"wrote {OUT}")
    print(f"  KeyframeGuidesTimeSpaced=#{kf}  VHS_LoadVideo=#{kf_video}  LTXVCropGuides=#{crop}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="show changes without writing")
    args = ap.parse_args()
    build(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
