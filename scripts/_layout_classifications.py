"""Shared functional-column bindings for canonical audio-loop workflows.

Last updated: 2026-05-08

The single place where node id -> functional column ("inputs", "models",
"sampler", etc.) is recorded for the audio-loop workflow family. Apply
scripts that need to lay out a workflow consume this table and map each
functional column to their own group key.

Splits the previously-duplicated `apply_intro_workflow.SOURCE_NODE_GROUPS`
and `apply_layout_polish_audio_loop_latent.NODE_GROUPS` into one source.
When a new node type is added to the canonical, update here once.

Functional columns
------------------

Stable names — apply scripts depend on them.

  inputs       — user-facing inputs (audio, image, seed, planner, schedule)
  models       — DiT / VAEs / CLIP / sage attention / advanced patches
  loras        — LoRA loaders + IC-LoRA loader + ID-LoRA reference audio
  cond         — conditioning chain (CLIPTextEncode, ZeroOut, NAG, LTXVConditioning, ConditioningSelectByIteration)
  sampler      — distilled sampler chain (KSamplerSelect, CFGGuider, ManualSigmas, RandomNoise)
  loop         — TensorLoopOpen/Close + AudioLoopController/Planner + subgraph invoker + LoopIterationStamp
  output       — VHS_VideoCombine + final decode + latent concat + bypassed upsamplers
  preencode    — audio pre-encode + init render path (LTXVAudioVAEEncode, LTXVConcatAVLatent, etc.)
  iclora_ref   — IC-LoRA reference video chain (VHS_LoadVideo + ImageResizeKJv2 + LTXVPreprocess)

Per-script overrides
--------------------

Apply scripts compose `SHARED_NODE_FUNCTIONS` with their own overrides.
Example: `apply_layout_polish_audio_loop_latent.py` pins specific input
nodes to a REQUIRED tier instead of inheriting the default `inputs` →
COMMON-tier mapping. The override dict is the only per-script edit
needed; the shared table provides functional defaults.

Reference
---------
- `scripts/_layout_grid.py` — the layout primitives (LayoutSpec, apply_layout)
- `docs/reference/workflow_layout_helpers.md` — full pattern doc
- `scripts/apply_intro_workflow.py` — historical seed; consumes this module
"""

from __future__ import annotations

# Canonical functional column for each classified node id in the
# audio-loop family workflows. Compose with per-script overrides.
SHARED_NODE_FUNCTIONS: dict[int, str] = {
    # --- inputs ---
    565:  "inputs",        # LoadAudio
    567:  "inputs",        # TrimAudioDuration (Song Trim)
    581:  "inputs",        # SetNode Set_orig_audio (collapsed pill)
    604:  "inputs",        # GetNode Get_orig_audio (collapsed pill)
    444:  "inputs",        # LoadImage
    1527: "inputs",        # INTConstant start_seed
    1528: "inputs",        # SetNode Set_start_seed (collapsed pill)
    1269: "inputs",        # FloatConstant (image strength)
    1533: "inputs",        # Note (vocal sep)
    568:  "inputs",        # MelBandRoFormerModelLoader (BYPASSED)
    569:  "inputs",        # MelBandRoFormerSampler (BYPASSED)
    640:  "inputs",        # SetNode Set_actual_audio
    641:  "inputs",        # GetNode Get_actual_audio (collapsed pill)
    601:  "inputs",        # TrimAudioDuration (Initial-Render Audio Trim, 10s)

    # --- models ---
    414:  "models",        # UNETLoader
    1537: "models",        # VAELoaderKJ (video)
    228:  "models",        # SetNode Set_video_vae (collapsed pill)
    1538: "models",        # VAELoaderKJ (audio)
    252:  "models",        # SetNode Set_audio_vae (collapsed pill)
    416:  "models",        # DualCLIPLoader
    268:  "models",        # AudioLoopHelperSageAttention
    504:  "models",        # LTXVChunkFeedForward
    1523: "models",        # LTX2AttentionTunerPatch
    503:  "models",        # LTX2SamplingPreviewOverride

    # --- loras ---
    1635: "loras",         # LTXICLoRALoaderModelOnly
    572:  "loras",         # SetNode Set_model
    1631: "loras",         # TrimAudioDuration (ID-LoRA Reference Slice)
    1632: "loras",         # LTXVReferenceAudio (ID-LoRA initial)
    1633: "loras",         # LTXVReferenceAudio (ID-LoRA loop)

    # --- cond ---
    1634: "cond",          # LTXFramePlanner (matches intro's grouping; polish overrides to REQUIRED tier)
    169:  "cond",          # CLIPTextEncode (initial-render prompt) — historical; absent from current canonical
    507:  "cond",          # CLIPTextEncode (negative)
    420:  "cond",          # ConditioningZeroOut
    1615: "cond",          # TimestampPromptScheduleBatchEncode (matches intro; polish overrides to REQUIRED)
    508:  "cond",          # LTX2_NAG
    164:  "cond",          # LTXVConditioning (initial)
    1616: "cond",          # ConditioningSelectByIteration

    # --- sampler ---
    1421: "sampler",       # ManualSigmas
    1422: "sampler",       # VisualizeSigmasKJ
    1423: "sampler",       # PreviewImage (sigma viz)
    579:  "sampler",       # SetNode Set_sigmas (collapsed pill)
    154:  "sampler",       # KSamplerSelect
    1322: "sampler",       # RandomNoise
    153:  "sampler",       # CFGGuider
    161:  "sampler",       # SamplerCustomAdvanced

    # --- loop ---
    1582: "loop",          # AudioLoopController
    1560: "loop",          # AudioLoopPlanner
    1539: "loop",          # TensorLoopOpen
    1540: "loop",          # TensorLoopClose
    843:  "loop",          # subgraph invoker
    1618: "loop",          # LoopIterationStamp
    1563: "loop",          # PreviewAny (Iteration Timestamps)
    1586: "loop",          # PreviewAny

    # --- output ---
    1604: "output",        # LTXVTiledVAEDecode (Final)
    1591: "output",        # LTXVLatentUpsampler (BYPASSED)
    1589: "output",        # LatentUpscaleModelLoader (BYPASSED)
    1605: "output",        # LatentConcat (Prepend Initial Render)
    1587: "output",        # LTXVConditioning (Loop, BYPASSED)
    617:  "output",        # VHS_VideoCombine

    # --- preencode (audio pre-encode + init render path) ---
    2009: "preencode",     # LTXVAudioVAEEncode (full song)
    2010: "preencode",     # SetNode Set_full_audio_latent
    2011: "preencode",     # GetNode Get_full_audio_latent
    566:  "preencode",     # LTXVAudioVAEEncode (initial 10s)
    570:  "preencode",     # SetLatentNoiseMask
    571:  "preencode",     # SolidMask (collapsed pill)
    344:  "preencode",     # EmptyLTXVLatentVideo
    531:  "preencode",     # LTXVImgToVideoInplaceKJ
    1617: "preencode",     # VAEEncode (init image -> guide latent)
    350:  "preencode",     # LTXVConcatAVLatent
    245:  "preencode",     # LTXVSeparateAVLatent
    381:  "preencode",     # LTXVCropGuides
    445:  "preencode",     # ImageResizeKJv2 (init)
    446:  "preencode",     # LTXVPreprocess (init)

    # --- iclora_ref ---
    1636: "iclora_ref",    # VHS_LoadVideo (BYPASSED)
    1637: "iclora_ref",    # ImageResizeKJv2 (ref-video)
    1638: "iclora_ref",    # LTXVPreprocess (ref-video)

    # --- pill reroutes (Get/Set) routed by consumer-side function ---
    254:  "preencode",     # Get_audio_vae (audio-encode side)
    599:  "preencode",     # Get_audio_vae
    413:  "preencode",     # Get_video_vae
    236:  "preencode",     # Get_video_vae
    619:  "output",        # Get_video_vae (decoder side)
    1598: "output",        # Get_video_vae
    582:  "preencode",     # Get_orig_audio
    580:  "sampler",       # Get_sigmas
    654:  "loop",          # Get_model (subgraph invoker)
    1529: "sampler",       # Get_start_seed
    1530: "sampler",       # Get_start_seed
}


def compose(function_to_group: dict[str, str], *, overrides: dict[int, str] | None = None) -> dict[int, str]:
    """Map every shared node id to a group key via the function-to-group table,
    applying per-script overrides last.

    Apply scripts call this with:
      - `function_to_group`: their own mapping from functional column name
        ("inputs", "models", ...) to their group keys ("1_inputs", "G_INPUTS").
      - `overrides`: optional `{node_id: group_key}` dict that pins specific
        nodes to a group ignoring their functional column. Used by scripts
        with tier sub-groups inside a column (e.g. polish's REQUIRED tier).
    """
    overrides = overrides or {}
    out: dict[int, str] = {}
    for nid, fn in SHARED_NODE_FUNCTIONS.items():
        if nid in overrides:
            out[nid] = overrides[nid]
            continue
        if fn in function_to_group:
            out[nid] = function_to_group[fn]
    return out


__all__ = ["SHARED_NODE_FUNCTIONS", "compose"]
