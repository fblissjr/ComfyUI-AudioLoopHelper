"""apply_ltx_i2v_tiled_optimizations.

Last updated: 2026-05-11

Stages a portability-tuned variant of a single-stage LTX 2.3 I2V
workflow that uses a tiled-sampler upscale path. Reads the source
workflow, writes the optimized output to a sibling path. Does NOT
mutate `example_workflows/`.

Target shape (not a looping workflow — most loop-only F-pair audits
skip):

  LoadImage -> ImageResizeKJv2 -> EmptyLTXVLatentVideo
            -> LTXVImgToVideoInplaceKJ -> LTXVAddGuide
            -> STGGuiderAdvanced (model fork A: anchor)
            -> SamplerCustomAdvanced
            -> LTXVSeparateAVLatent -> LTXVCropGuides
            -> LTXVLatentUpsamplerTiled (2x)
            -> LTXVImgToVideoInplaceKJ -> LTXVConcatAVLatent
            -> a tiled-sampler node
            (model fork B: text-attention amplifier)
            -> VAEDecode -> VHS_VideoCombine

Optimizations applied (the portable ones — they travel from our
audio-loop discipline to any single-stage LTX 2.3 I2V pipeline):

  1. Strip dead audio re-encode branch.
     `LTXVAudioVAEEncode -> SetLatentNoiseMask` wired but the mask
     output has zero consumers. Pulls in a `SolidMask` and a
     `GetNode` of the audio VAE that have no other consumers
     either. All four nodes are removed.

  2. Insert AudioLoopHelperSageAttention on both MODEL forks.
     The workflow forks the model into an anchor chain and a
     text-attention-amplifier chain. Sage attaches AFTER the last
     LoRA loader on each fork so the LoRA weight patches apply
     against the unpatched attention module first (canonical
     order: LoRA loaders -> module-mutating attention patch ->
     state-dict-reading consumer).

  3. Replace both `VAEDecode` nodes with `LTXVTiledVAEDecode
     [1,1,1,true,"auto","auto"]`. Per the gotchas in root
     CLAUDE.md, single-tile decode on 24GB+ is ~3x faster than the
     default [2,2,1] tile shape and avoids OOM at the post-2x-
     upscale latent volume.

     SLOT-ORDER GOTCHA: `VAEDecode` inputs = [samples, vae];
     `LTXVTiledVAEDecode` inputs = [vae, latents]. A bare type
     swap dangles both links.

  4. Replace `ImageResizeKJv2 (lanczos, single-pass)` with
     `LTXSmartImageResize`. At >2x linear reduction, single-pass
     lanczos aliases; the aliasing reads as motion cues to LTX
     2.3's cross-attention. SmartResize stages adaptively
     (bicubic+antialias for intermediates, lanczos only at the
     final stage). Postmortem:
     `internal/analysis/smart_resize_quantization_postmortem.md`.

  5. Insert `LTXVPreprocess(img_compression=18)` on the init-image
     path. F2-mirror for non-loop workflows: matches Lightricks's
     training-time conditioning compression. Feeds only the three
     init-image *guide* consumers (anchor `reference_image`,
     `LTXVAddGuide.image`, `LTXVImgToVideoInplaceKJ.image_1`).
     Preview / setnode / dim-derive consumers keep the raw resized
     image.

Phase 3 -- keeper-config promotions from the A/B matrix (2026-05-11
visual A/B: arm2 selected over arm0/arm1):

  9. Replace the source 14-pt sigma curve on `ManualSigmas #527`
     with the canonical distilled 9-pt curve from root CLAUDE.md.
  10. Swap first-pass `KSamplerSelect #520` to `euler`. The source
      `euler_ancestral` is a distilled-path rule violation; injects
      stochastic noise the distilled LoRA wasn't trained for.
  11. Remap `LTXLatentAnchorAware #731.cache_at_step` from 6 to 5
      so the anchor's matching cache fires at the same sigma slot
      (~0.91) on the canonical curve that it did at (~0.81) on the
      source curve.
  12. Remap `STGGuiderAdvanced #653` sigma list + cfg/stg/rescale/
      layers widget tables onto the 9-pt curve. 14:13:13:13:14
      entry layout becomes 9:8:8:8:9. Preserves the 2-step cfg/stg
      warmup at the top of the curve.
  13. Remove the RES4LYF `Sigmas Easing #652` node entirely and
      direct-wire `ManualSigmas #527 -> SamplerCustomAdvanced #510`.
      Reasoning: RES4LYF is designed for non-distilled pipelines
      where sigma-curve shape is a free knob. LTX 2.3's distilled
      LoRA is trained at the canonical 8-step sigmas; warping the
      curve fights the training. Visual A/B confirmed arm2 (no
      easing) >= arm0/arm1 on the test render.

Phase 2 -- redundancy / DRY cleanup (no behavioral change to the
live render path):

  6. Remove the empty `Power Lora Loader (rgthree)` pass-through.
     Its widget set carries zero LoRAs (empty `{}` entries) and its
     CLIP output is unused. Rewires its two MODEL consumers
     directly to the upstream checkpoint loader.

  7. Remove the dangling `Set_seed` SetNode. Its `seed` variable
     has no `GetNode` consumer; the SetNode is dead weight. The
     rgthree seed itself is live (it drives `RandomNoise.noise_seed`
     via input wire) so it stays.

  8. Set the output-path `AudioAdjustVolume` widget to -3 dB to
     match the preview-path node. The source workflow had one path
     at -3 and the other at 0 (no-op); the asymmetry has no obvious
     intent and -3 dB on both paths matches the preview behavior
     the workflow author already chose.

  8b. Normalize the `LoadImage` default filename to `ref_image.jpg`
      so the shipped drafts don't carry the source workflow's
      user-specific input filename. To render, drop your reference
      image into ComfyUI's `input/` directory as `ref_image.jpg`
      (or pick another file via the LoadImage widget UI).

Out-of-scope (deferred -- risky or workflow-author-tuned):

  - The `euler_ancestral` / `euler_ancestral_cfg_pp` samplers
    violate the distilled-path rule, but the 14-pt sigma curve
    appears custom-tuned to the workflow's distilled LoRA.
    Swapping samplers requires A/B validation.
  - `Sigmas Easing` mid-chain curve warp; the source workflow's
    own Note labels this "don't mess with this."
  - LTXFramePlanner SSoT migration; structurally invasive on a
    non-loop workflow that already has working dim derivation.

Usage:
    uv run --group dev python scripts/apply_ltx_i2v_tiled_optimizations.py --input <path>
    uv run --group dev python scripts/apply_ltx_i2v_tiled_optimizations.py --input <path> --output <path>
    uv run --group dev python scripts/apply_ltx_i2v_tiled_optimizations.py --input <path> --revert
    uv run --group dev python scripts/apply_ltx_i2v_tiled_optimizations.py --input <path> --dry-run

`--input` is required (no default — avoids leaking source-specific
paths into the public script). `--output` defaults to the canonical
draft location `internal/workflows/ltx_i2v_tiled_optimized.draft.json`
(gitignored per `internal/workflows/README.md`). Idempotent on the
OUTPUT path. `--revert` deletes the output staging file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

# Source workflow node IDs we depend on.
ID_LAST_LORA_A = 723         # LTX2LoraLoaderAdvanced -- feeds anchor
ID_ANCHOR_731 = 731          # a model-anchor wrapper
ID_LAST_LORA_B = 718         # LTX2LoraLoaderAdvanced -- feeds amplifier
ID_AMPLIFIER_753 = 753       # a text-attention-amplifier wrapper
ID_RESIZE_531 = 531          # ImageResizeKJv2
ID_ADDGUIDE_767 = 767        # LTXVAddGuide
ID_INPLACE_772 = 772         # LTXVImgToVideoInplaceKJ (init-image inplace)
ID_VAEDECODE_PREVIEW = 552   # VAEDecode (first-pass preview)
ID_VAEDECODE_FINAL = 740     # VAEDecode (post-tile-sampler)
ID_DEAD_AUDIO_ENCODE = 274   # LTXVAudioVAEEncode (dead branch)
ID_DEAD_NOISE_MASK = 275     # SetLatentNoiseMask (dead branch)
ID_DEAD_AUD_VAE_GET = 279    # GetNode (feeds only the dead encode)
ID_DEAD_SOLID_MASK = 630     # SolidMask (feeds only the dead mask)

ID_CHECKPOINT = 646          # CheckpointLoaderSimple
ID_POWER_LORA = 557          # Power Lora Loader (rgthree) -- empty pass-through
ID_FORK_A_FIRST = 722        # first LoRA in fork A (anchor chain)
ID_FORK_B_FIRST = 719        # first LoRA in fork B (amplifier chain)
ID_DEAD_SET_SEED = 621       # Set_seed SetNode with no GetNode consumer
ID_OUTPUT_AUDIO_VOL = 598    # AudioAdjustVolume on output path (widget=0, asymmetric)
ID_LOAD_IMAGE_REF = 773      # LoadImage for the i2v init reference image

DEFAULT_REF_IMAGE_FILENAME = "ref_image.jpg"

# Phase 3 (keeper) node ids.
ID_MANUAL_SIGMAS = 527       # first-pass ManualSigmas
ID_KSAMPLER = 520            # first-pass KSamplerSelect
ID_ANCHOR = 731              # LTXLatentAnchorAware (alias of ID_ANCHOR_731)
ID_STG_GUIDER = 653          # STGGuiderAdvanced
ID_SIGMAS_EASING = 652       # RES4LYF Sigmas Easing -- to be removed
ID_SAMPLER_CUSTOM = 510      # SamplerCustomAdvanced (consumer of the eased curve)

# Canonical distilled curve + STG remap (matches arm1/arm2 from the A/B).
CANONICAL_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
KEEPER_STG_CFG = "2, 1.5, 1, 1, 1, 1, 1, 1"
KEEPER_STG_STG_SCALE = "2, 1.5, 1, 1, 1, 1, 1, 1"
KEEPER_STG_RESCALE = "1, 1, 1, 1, 1, 1, 1, 1"
KEEPER_STG_LAYERS = "[9999], [9999], [9999], [9999], [9999], [9999], [9999], [9999], [9999]"
KEEPER_ANCHOR_CACHE_STEP = 5

REQUIRED_SOURCE_NODES = (
    ID_LAST_LORA_A,
    ID_ANCHOR_731,
    ID_LAST_LORA_B,
    ID_AMPLIFIER_753,
    ID_RESIZE_531,
    ID_ADDGUIDE_767,
    ID_INPLACE_772,
    ID_VAEDECODE_PREVIEW,
    ID_VAEDECODE_FINAL,
)


def _already_migrated(ed: WorkflowEditor) -> bool:
    # Sage = Phase 1 marker; absence of `Sigmas Easing` = Phase 3 marker.
    # Both required so a draft from before the Phase 3 promotion gets
    # re-migrated rather than short-circuited.
    has_sage = bool(ed.find_nodes_by_type("AudioLoopHelperSageAttention"))
    has_easing = bool(ed.find_nodes_by_type("Sigmas Easing"))
    return has_sage and not has_easing


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = ed.require_nodes(REQUIRED_SOURCE_NODES)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source node(s) missing: {missing}. "
            "This script assumes the LTX 2.3 I2V tiled-sampler workflow layout "
            "documented in the module docstring."
        )


def _strip_dead_audio_branch(ed: WorkflowEditor) -> None:
    """Remove the dead audio re-encode branch.

    `LTXVAudioVAEEncode -> SetLatentNoiseMask` produces a masked
    audio latent that nothing consumes. The `SolidMask` feeds only
    the noise mask; the audio-VAE `GetNode` feeds only the encode.
    All dead.
    """
    for nid in (
        ID_DEAD_NOISE_MASK,
        ID_DEAD_AUDIO_ENCODE,
        ID_DEAD_SOLID_MASK,
        ID_DEAD_AUD_VAE_GET,
    ):
        if ed.has_node(nid):
            ed.remove_node_and_links(nid)
            print(f"  removed dead node #{nid}")


def _insert_sage_attention(ed: WorkflowEditor) -> None:
    """Insert two AudioLoopHelperSageAttention nodes -- one per MODEL fork.

    Order: last LoRA loader -> SAGE -> consumer (anchor or amplifier).
    """
    sage_widget = ["auto_mask_aware", True, 1024]

    anchor = ed.find_node(ID_ANCHOR_731)
    sage_a_pos = [anchor["pos"][0] + 280, anchor["pos"][1] - 90]
    sage_a_id = ed.add_top_level_node(
        node_type="AudioLoopHelperSageAttention",
        pos=sage_a_pos,
        size=[330, 130],
        inputs=[WorkflowEditor.io_in("model", "MODEL")],
        outputs=[WorkflowEditor.out("model", "MODEL")],
        widgets_values=sage_widget,
        title="Sage Attention (anchor chain)",
    )
    ed.rewire_input(sage_a_id, 0, ID_LAST_LORA_A, 0, "MODEL")
    ed.rewire_input(ID_ANCHOR_731, 0, sage_a_id, 0, "MODEL")
    print(f"  inserted SAGE_A #{sage_a_id} between #{ID_LAST_LORA_A} and #{ID_ANCHOR_731}")

    amp = ed.find_node(ID_AMPLIFIER_753)
    sage_b_pos = [amp["pos"][0] - 380, amp["pos"][1]]
    sage_b_id = ed.add_top_level_node(
        node_type="AudioLoopHelperSageAttention",
        pos=sage_b_pos,
        size=[330, 130],
        inputs=[WorkflowEditor.io_in("model", "MODEL")],
        outputs=[WorkflowEditor.out("model", "MODEL")],
        widgets_values=sage_widget,
        title="Sage Attention (amplifier chain)",
    )
    ed.rewire_input(sage_b_id, 0, ID_LAST_LORA_B, 0, "MODEL")
    ed.rewire_input(ID_AMPLIFIER_753, 0, sage_b_id, 0, "MODEL")
    print(f"  inserted SAGE_B #{sage_b_id} between #{ID_LAST_LORA_B} and #{ID_AMPLIFIER_753}")


def _slot0_consumers(ed: WorkflowEditor, src_node: int) -> list[tuple[int, int]]:
    """Return (tgt_node, tgt_slot) for every link from `src_node`'s output slot 0."""
    return [(L[3], L[4]) for L in ed.find_links_from(src_node) if L[2] == 0]


def _swap_vae_decode_to_tiled(ed: WorkflowEditor, target_id: int) -> None:
    """Replace a `VAEDecode` node with `LTXVTiledVAEDecode [1,1,1,true,auto,auto]`.

    Input slot order differs:
      - `VAEDecode`:           [samples, vae]
      - `LTXVTiledVAEDecode`:  [vae, latents]
    """
    old = ed.find_node(target_id)
    old_pos = list(old["pos"])

    samples_link = ed.find_link_to_slot(target_id, 0)
    vae_link = ed.find_link_to_slot(target_id, 1)
    if samples_link is None or vae_link is None:
        raise SystemExit(f"VAEDecode #{target_id} missing required inputs.")
    samples_src, samples_src_slot = samples_link[1], samples_link[2]
    vae_src, vae_src_slot = vae_link[1], vae_link[2]

    downstream = _slot0_consumers(ed, target_id)
    ed.remove_node_and_links(target_id)

    new_id = ed.add_top_level_node(
        node_type="LTXVTiledVAEDecode",
        pos=old_pos,
        size=[270, 130],
        inputs=[
            WorkflowEditor.io_in("vae", "VAE"),
            WorkflowEditor.io_in("latents", "LATENT"),
        ],
        outputs=[WorkflowEditor.out("image", "IMAGE")],
        widgets_values=[1, 1, 1, True, "auto", "auto"],
        title="LTXV Tiled VAE Decode",
    )
    ed.add_link(vae_src, vae_src_slot, new_id, 0, "VAE")
    ed.add_link(samples_src, samples_src_slot, new_id, 1, "LATENT")
    for tgt, tgt_slot in downstream:
        ed.add_link(new_id, 0, tgt, tgt_slot, "IMAGE")
    print(f"  swapped VAEDecode #{target_id} -> LTXVTiledVAEDecode #{new_id} "
          f"(downstream IMAGE consumers={len(downstream)})")


def _swap_resize_to_smart(ed: WorkflowEditor) -> int:
    """Replace `ImageResizeKJv2` with `LTXSmartImageResize`.

    Carries width/height/crop_position from the source widgets. The
    KJv2 IMAGE output (slot 0) maps 1-to-1 to SmartResize's image
    output; the KJv2 width/height/mask output slots are unused.
    """
    old = ed.find_node(ID_RESIZE_531)
    old_pos = list(old["pos"])
    old_widgets = old["widgets_values"]
    target_w = int(old_widgets[0])
    target_h = int(old_widgets[1])
    # KJv2 crop_position widget is at index 5.
    crop_pos = old_widgets[5]
    if crop_pos not in ("center", "top", "bottom", "left", "right"):
        crop_pos = "center"

    image_in_link = ed.find_link_to_slot(ID_RESIZE_531, 0)
    if image_in_link is None:
        raise SystemExit(f"ImageResizeKJv2 #{ID_RESIZE_531} missing image input.")
    image_src, image_src_slot = image_in_link[1], image_in_link[2]

    downstream = _slot0_consumers(ed, ID_RESIZE_531)
    ed.remove_node_and_links(ID_RESIZE_531)
    new_id = ed.add_top_level_node(
        node_type="LTXSmartImageResize",
        pos=old_pos,
        size=[290, 156],
        inputs=[
            WorkflowEditor.io_in("image", "IMAGE"),
            WorkflowEditor.widget_in("width", "INT"),
            WorkflowEditor.widget_in("height", "INT"),
            WorkflowEditor.widget_in("keep_proportion", "BOOLEAN"),
            WorkflowEditor.widget_in("crop_position", "COMBO"),
        ],
        outputs=[
            WorkflowEditor.out("image", "IMAGE"),
            WorkflowEditor.out("width", "INT"),
            WorkflowEditor.out("height", "INT"),
        ],
        widgets_values=[target_w, target_h, True, crop_pos],
        title="LTX Smart Image Resize",
    )
    ed.add_link(image_src, image_src_slot, new_id, 0, "IMAGE")
    for tgt, tgt_slot in downstream:
        ed.add_link(new_id, 0, tgt, tgt_slot, "IMAGE")
    print(f"  swapped ImageResizeKJv2 #{ID_RESIZE_531} -> LTXSmartImageResize #{new_id} "
          f"(downstream consumers={len(downstream)}, target={target_w}x{target_h})")
    return new_id


def _drop_empty_power_lora_loader(ed: WorkflowEditor) -> None:
    """Remove the empty `Power Lora Loader (rgthree)` and bypass it.

    The pass-through wraps the checkpoint MODEL with zero LoRAs
    actually loaded (widget set is `[{}, header, {}, ""]`). Its
    CLIP output is unused. The two MODEL consumers (first LoRA in
    each fork) get rewired straight to `CheckpointLoaderSimple`.
    """
    if not ed.has_node(ID_POWER_LORA):
        return
    pll = ed.find_node(ID_POWER_LORA)
    widgets = pll.get("widgets_values") or []
    # The rgthree Power Lora Loader stores per-LoRA dicts with keys like
    # {"on": True, "lora": "...", "strength": ...}; presence of any such key
    # means a real LoRA is configured and removal would silently drop weights.
    for w in widgets:
        if isinstance(w, dict) and any(k in w for k in ("lora", "on", "strength")):
            print("  Power Lora Loader carries real LoRA entries; skipping removal.")
            return

    consumers = _slot0_consumers(ed, ID_POWER_LORA)
    ed.remove_node_and_links(ID_POWER_LORA)
    for tgt, tgt_slot in consumers:
        ed.add_link(ID_CHECKPOINT, 0, tgt, tgt_slot, "MODEL")
    print(f"  removed empty Power Lora Loader #{ID_POWER_LORA}; rewired {len(consumers)} MODEL consumer(s) -> #{ID_CHECKPOINT}")


def _drop_dead_set_seed(ed: WorkflowEditor) -> None:
    """Remove the `Set_seed` SetNode whose variable has no GetNode consumer."""
    if not ed.has_node(ID_DEAD_SET_SEED):
        return
    for n in ed.wf["nodes"]:
        if n.get("type") == "GetNode":
            wv = n.get("widgets_values") or []
            if wv and wv[0] == "seed":
                print(f"  Set_seed has a live GetNode #{n['id']}; leaving in place.")
                return
    ed.remove_node_and_links(ID_DEAD_SET_SEED)
    print(f"  removed dead SetNode #{ID_DEAD_SET_SEED} (no GetNode consumes 'seed')")


def _normalize_load_image_default(ed: WorkflowEditor) -> None:
    """Set `LoadImage` filename widget to a generic placeholder.

    The source workflow carries a user-specific input filename. The
    keeper drafts ship with `ref_image.jpg` so a reader's first action
    is to drop their actual reference image at `input/ref_image.jpg`
    rather than work around a stranger's filename.
    """
    if not ed.has_node(ID_LOAD_IMAGE_REF):
        return
    n = ed.find_node(ID_LOAD_IMAGE_REF)
    wv = list(n.get("widgets_values") or [])
    if not wv:
        return
    old = wv[0]
    if old == DEFAULT_REF_IMAGE_FILENAME:
        return
    wv[0] = DEFAULT_REF_IMAGE_FILENAME
    n["widgets_values"] = wv
    print(f"  LoadImage #{ID_LOAD_IMAGE_REF} default: {old!r} -> {DEFAULT_REF_IMAGE_FILENAME!r}")


def _apply_keeper_config(ed: WorkflowEditor) -> None:
    """Phase 3: bake in arm2's wins from the A/B matrix.

    Canonical curve + euler + matched anchor/STG remap + remove the
    RES4LYF `Sigmas Easing` node entirely. After this Phase 3 step
    the baseline draft no longer imports RES4LYF; the variants
    script's `arm1` and `arm2` are now identity-vs-baseline (kept
    in the dispatch for back-compat but produce no diff).
    """
    # Widget edits
    ed.find_node(ID_MANUAL_SIGMAS)["widgets_values"] = [CANONICAL_SIGMAS]
    ed.find_node(ID_KSAMPLER)["widgets_values"] = ["euler"]
    anchor = ed.find_node(ID_ANCHOR)
    anchor["widgets_values"][1] = KEEPER_ANCHOR_CACHE_STEP
    stg = ed.find_node(ID_STG_GUIDER)
    stg["widgets_values"][2] = CANONICAL_SIGMAS
    stg["widgets_values"][3] = KEEPER_STG_CFG
    stg["widgets_values"][4] = KEEPER_STG_STG_SCALE
    stg["widgets_values"][5] = KEEPER_STG_RESCALE
    stg["widgets_values"][6] = KEEPER_STG_LAYERS
    print(f"  baked canonical curve + euler + anchor remap + STG remap onto nodes "
          f"#{ID_MANUAL_SIGMAS}, #{ID_KSAMPLER}, #{ID_ANCHOR}, #{ID_STG_GUIDER}")

    # Remove the Sigmas Easing node and direct-wire its source to its consumer
    if ed.has_node(ID_SIGMAS_EASING):
        easing_in_link = ed.find_link_to_slot(ID_SIGMAS_EASING, 0)
        if easing_in_link is None:
            raise SystemExit(f"Sigmas Easing #{ID_SIGMAS_EASING} has no sigmas input.")
        src_node, src_slot = easing_in_link[1], easing_in_link[2]
        # Capture downstream consumers of the easing output (slot 0).
        downstream = [(L[3], L[4]) for L in ed.find_links_from(ID_SIGMAS_EASING) if L[2] == 0]
        ed.remove_node_and_links(ID_SIGMAS_EASING)
        for tgt, tgt_slot in downstream:
            ed.add_link(src_node, src_slot, tgt, tgt_slot, "SIGMAS")
        print(f"  removed `Sigmas Easing` #{ID_SIGMAS_EASING}; "
              f"direct-wired #{src_node}.SIGMAS -> {len(downstream)} consumer(s) "
              f"(RES4LYF dep dropped from the pipeline)")


def _symmetrize_output_audio_volume(ed: WorkflowEditor) -> None:
    """Set the output-path AudioAdjustVolume to -3 dB (matches preview)."""
    if not ed.has_node(ID_OUTPUT_AUDIO_VOL):
        return
    n = ed.find_node(ID_OUTPUT_AUDIO_VOL)
    wv = list(n.get("widgets_values") or [])
    if not wv or wv[0] == -3:
        return
    old = wv[0]
    wv[0] = -3
    n["widgets_values"] = wv
    print(f"  set AudioAdjustVolume #{ID_OUTPUT_AUDIO_VOL} dB widget: {old} -> -3 (symmetric with preview)")


def _insert_preprocess_on_init_path(ed: WorkflowEditor, resize_id: int) -> None:
    """Insert LTXVPreprocess(img_compression=18) between SmartResize and the three init-image *guide* consumers.

    Consumers that get the preprocessed image:
      - a model-anchor wrapper.reference_image (slot 1)
      - LTXVAddGuide.image                   (slot 4)
      - LTXVImgToVideoInplaceKJ.image_1      (slot 2)

    Consumers that keep the raw resized image:
      - ResizeImageMaskNode, PreviewImage, SetNode (resized_image)
    """
    resize_node = ed.find_node(resize_id)
    pos = [resize_node["pos"][0] + 320, resize_node["pos"][1]]
    pre_id = ed.add_top_level_node(
        node_type="LTXVPreprocess",
        pos=pos,
        size=[270, 60],
        inputs=[WorkflowEditor.io_in("image", "IMAGE")],
        outputs=[WorkflowEditor.out("output_image", "IMAGE")],
        widgets_values=[18],
        title="LTXV Preprocess (img_compression=18)",
    )
    ed.add_link(resize_id, 0, pre_id, 0, "IMAGE")

    guide_consumers = (
        (ID_ANCHOR_731, 1, "reference_image"),
        (ID_ADDGUIDE_767, 4, "image"),
        (ID_INPLACE_772, 2, "image_1"),
    )
    for tgt_node, tgt_slot, slot_name in guide_consumers:
        ed.rewire_input(tgt_node, tgt_slot, pre_id, 0, "IMAGE")
        print(f"  rewired #{tgt_node}.{slot_name} <- #{pre_id} (preprocessed)")
    print(f"  inserted LTXVPreprocess #{pre_id} (img_compression=18) for 3 init-guide consumers")


DEFAULT_OUTPUT = "internal/workflows/ltx_i2v_tiled_optimized.draft.json"


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if output_path.exists() and input_path != output_path and _already_migrated(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
        return

    if dry_run:
        ed = WorkflowEditor(input_path)
        _assert_required_nodes_present(ed)
        print(f"would copy {input_path} -> {output_path}")
        print("would apply Phase 1 (portable):")
        print("  - strip dead audio re-encode branch")
        print("  - insert 2x AudioLoopHelperSageAttention (auto_mask_aware, skip<1024)")
        print("  - swap 2x VAEDecode -> LTXVTiledVAEDecode [1,1,1,true,auto,auto]")
        print("  - swap ImageResizeKJv2 -> LTXSmartImageResize")
        print("  - insert LTXVPreprocess(img_compression=18) for 3 init-guide consumers")
        print("would apply Phase 2 (DRY):")
        print("  - remove empty Power Lora Loader (rgthree) pass-through")
        print("  - remove dangling Set_seed SetNode (no GetNode consumes 'seed')")
        print("  - set output-path AudioAdjustVolume to -3 dB (match preview)")
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

    # Phase 1 -- portable optimizations
    _strip_dead_audio_branch(ed)
    _insert_sage_attention(ed)
    _swap_vae_decode_to_tiled(ed, ID_VAEDECODE_PREVIEW)
    _swap_vae_decode_to_tiled(ed, ID_VAEDECODE_FINAL)
    new_resize_id = _swap_resize_to_smart(ed)
    _insert_preprocess_on_init_path(ed, new_resize_id)

    # Phase 2 -- redundancy / DRY cleanup
    _drop_empty_power_lora_loader(ed)
    _drop_dead_set_seed(ed)
    _symmetrize_output_audio_volume(ed)
    _normalize_load_image_default(ed)

    # Phase 3 -- keeper-config promotions from the A/B matrix
    _apply_keeper_config(ed)

    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Audit:         uv run --group dev python scripts/audit_workflows.py {output_path}")
    print(f"  3. Load in ComfyUI and A/B against the source workflow.")


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
    ap.add_argument("--input", required=True,
                    help="Path to the source workflow JSON.")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"Output draft path (default: {DEFAULT_OUTPUT}).")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied/changed without writing.")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.revert:
        _revert(output_path)
        return

    _migrate(input_path, output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
