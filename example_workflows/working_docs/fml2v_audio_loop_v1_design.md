# fml2v_var_d_audio_loop V1 — design

Last updated: 2026-05-16

## Goal

Add tensor-loop iteration to the `fml2v_var_d_audio_input` benchmark workflow so it can render a full-length song with audio-video sync (replacing the current 4-second-audio + 15-second-video hardcode). Output lives at `example_workflows/experimental/fml2v_var_d_audio_loop.json`.

V1 scope: **single-pass loop (pass1 only)** at base resolution. Pass2 + spatial upscaler kept in workflow but bypassed by default — can be re-enabled later as Option C per-iter or moved to a follow-up workflow (Option B) like canonical's `build_upscale_workflow.py` pattern.

## Design decisions (accepted)

| # | Decision | Choice |
|---|---|---|
| 1 | Sage variant | `AudioLoopHelperSageAttention` (this repo's `nodes_sage.py`), `mode="auto"`, `skip_under_seq_len=1024` |
| 2 | VAEs | Canonical LTX23 VAEs (not benchmark's LTX2 files) |
| 3 | Image preprocess | `LTXSmartImageResize` + `LTXVPreprocess(img_compression=18)` (canonical pattern, fixes quantization aliasing) |
| 4 | Window | `target_seconds=19.88s` at fps=25 (canonical, empirically validated as motion sweet spot) |
| 5 | fps | 25 globally (Lightricks shipped workflows + 8n+1 latent boundary) |
| 6 | NAG positioning | Inside loop body, downstream of `LoopIterationStamp` (re-executes per iter; defends against any per-iter patch-loss across offload) |
| 7 | Conditioning path | `TimestampPromptScheduleBatchEncode` active by default + bypassed parallel `CLIPTextEncode` for static-prompt A/B |
| 8 | Topology | **Flat canvas (no subgraph)** — easier debug, more modularity |
| 9 | nag_cond_video | Direct wire from top-level `CLIPTextEncode` to in-loop `LTX2_NAG.nag_cond_video` (no subgraph slot indirection) |
| 10 | Model patch chain | All patches (Sage, ChunkFFN, AttnTuner, NAG) INSIDE loop body downstream of `LoopIterationStamp` — re-execute per iter |
| 11 | Output assembly | `LatentConcat` + F14 trim chain + RunIdPrefix → VHS_VideoCombine |
| 12 | LoopConfigValidator | Yes — pre-run safety check |
| 13 | Audit compat | Validated by subagent — works at WARN-level with some F-checks gracefully skipping |
| pass2/upscale | Kept in workflow, bypassed (`mode=4`) | User can re-enable for experiments |
| static A/B prompt source | Bypassed `CLIPTextEncode` wired to same `PrimitiveStringMultiline` as benchmark | A/B uses same prompt text |
| Multi-frame anchoring | Initial render only (`LTXVAddGuideMulti` with first/middle/last frames); loop body uses single `LTXVAddLatentGuide` for trailing init anchor | Option A — cheap, matches canonical |

## Flat-canvas topology

Loop boundary demarcated by `TensorLoopOpen` (start) and `TensorLoopClose` (end). Everything between them on top-level canvas. ComfyUI executor re-runs nodes whose inputs change per iter; `LoopIterationStamp.current_iteration` is the transitive root for per-iter re-execution.

```
                         OUTSIDE LOOP (executes once)
┌──────────────────────────────────────────────────────────────────┐
│  Loaders (UNET, DualCLIP, VAEs, Upscale-bypassed)                │
│  Pre-encode:                                                     │
│    - LoadAudio → TrimAudioDuration → LTXVAudioVAEEncode          │
│      → Set_full_audio_latent                                     │
│    - CLIPTextEncode (positive bypassed-static)                   │
│    - TimestampPromptScheduleBatchEncode (positive active)        │
│    - CLIPTextEncode (nag_cond_video, "still image with no...")   │
│    - CLIPTextEncode (negative for LTXVConditioning slot)         │
│  LTXFramePlanner (fps=25, target_seconds=19.88)                  │
│  AudioLoopPlanner (iter-INDEPENDENT outputs)                     │
│  AudioLoopController (iter-DEPENDENT outputs; current_iter wire) │
│  FloatConstants: overlap_seconds=2.0, first_frame_guide_strength │
│  Init image: LoadImage → LTXSmartImageResize → LTXVPreprocess    │
│  Multi-frame: LoadImage(middle), LoadImage(last)                 │
│  Two ConditioningSelectByIteration (initial + loop)              │
│  LTXVConditioning (frame_rate=25 stamp on initial cond)          │
│  LoopConfigValidator                                             │
│                                                                  │
│  INITIAL RENDER:                                                 │
│    LTXVImgToVideoInplaceKJ (frame-0 anchor)                      │
│    LTXVAddGuideMulti (first+middle+last frames)                  │
│    LTXVConcatAVLatent (with full audio)                          │
│    CFGGuider + RandomNoise + SamplerCustomAdvanced               │
│    LTXVSeparateAVLatent → LTXVCropGuides → Set_initial_latent    │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  TensorLoopOpen.previous_value ← Get_initial_latent              │
│  TensorLoopOpen.iterations_in ← AudioLoopPlanner.total_iterations│
└──────────────────────────────────────────────────────────────────┘
                                  ↓
                     INSIDE LOOP (re-executes per iter)
┌──────────────────────────────────────────────────────────────────┐
│  LoopIterationStamp (transitive re-exec root)                    │
│    ↓ MODEL                                                       │
│  AudioLoopHelperSageAttention (re-applies via transformer_opts)  │
│    ↓                                                             │
│  LTXVChunkFeedForward (re-applies via object_patches)            │
│    ↓                                                             │
│  LTX2AttentionTunerPatch (bypassed by default; re-applies)       │
│    ↓                                                             │
│  LTX2_NAG (re-applies; consumes nag_cond_video from top-level)   │
│    ↓ MODEL → CFGGuider.model                                     │
│                                                                  │
│  AudioLatentSlice (per-iter slice of pre-encoded full song)      │
│  LatentContextExtract (overlap from prev iter)                   │
│  LatentOverlapTrim                                               │
│  LTXVAudioVideoMask (canonical: start=end=window, audio frozen)  │
│  LTXVAddLatentGuide (per-iter trailing init anchor at latent_-1) │
│  LTXVCropGuidesNoLatent → CFGGuider.positive/negative (F3)       │
│  LTXVCropGuides (with latent) → LTXVAdainLatent.latents (F3)     │
│  LTXVConcatAVLatent (audio frozen + video iter)                  │
│  RandomNoise (seeded by AudioLoopController.iteration_seed)      │
│  SamplerCustomAdvanced                                           │
│  LTXVSeparateAVLatent → LTXVAdainLatent (color anchor) → out     │
│  IterationCleanup                                                │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  TensorLoopClose.processed                                       │
│  TensorLoopClose.stop ← AudioLoopController.should_stop          │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
                         OUTSIDE LOOP (executes once)
┌──────────────────────────────────────────────────────────────────┐
│  LatentConcat (initial latent + iter outputs)                    │
│  TrimVideoLatentToAudio (F14, snap-UP)                           │
│  LTXVTiledVAEDecode [1,1,1,true,"auto","auto"]                   │
│  TrimImageBatchToAudio (F14, exact-length residue trim)          │
│  RunIdPrefix → VHS_VideoCombine.filename_prefix (F15)            │
│  VHS_VideoCombine (frame_rate=25, with audio mux)                │
└──────────────────────────────────────────────────────────────────┘
```

## SetNode/GetNode namespace (preserved + extended)

Preserved from benchmark: `vae`, `vae_audio`, `vae_tiny`, `upscale_model`, `clip`, `model`, `width`, `height`, `fps`, `frames`, `firstframe`, `firstframe_resized`, `middleframe`, `middleframe_resized`, `lastframe`, `lastframe_resized`, `firstframe_strength`, `middleframe_strength`, `lastframe_strength`, `negative`, `final_video`, `final_audio`.

Added for loop: `full_audio_latent`, `initial_latent`, `stride_seconds`, `audio_duration`, `current_iteration`, `iteration_seed`, `overlap_latent_frames`, `start_index`, `iterations_total`, `reference_latent` (for AdaIN anchor).

Removed from benchmark: `model_with_lora` (collapsed into single `model` bus since LoRA is bypassed), `model_nag` (NAG moves inside loop, no separate bus needed), `positive_guider2`, `negative_guider2`, `latent_audio` (replaced by `full_audio_latent` slice pattern).

## Pruning from benchmark

| Drop | Reason |
|---|---|
| `LTX2SamplingPreviewOverride` | Slow per-iter; preview during iteration adds VRAM |
| `LTXVScheduler #2` (orphaned) | Dead — `ManualSigmas` wins |
| `ManualSigmas #5` (orphaned) | Dead — `ManualSigmas #215` (9-value) is active |
| `LTXVEmptyLatentAudio #9` (bypassed) | Replaced by pre-encode pattern |
| `PathchSageAttentionKJ #226` (bypassed) | Drop — we use `AudioLoopHelperSageAttention` |
| `LTX2MemoryEfficientSageAttentionPatch #227` (bypassed) | Drop — same |
| `Set_latent_audio` chain (#2297-#2301, hardcoded 4s) | Replaced by full-song pre-encode |
| Static `CLIPTextEncode #16` | Replaced by schedule encoder + bypassed parallel CLIP |
| `Set_model_with_lora`, `Set_model_nag` | Namespace collapse |
| Various MarkdownNote tutorial nodes | Keep one summary; drop verbose tutorial blobs |

Kept-bypassed (user can re-enable):
- `LoraLoaderModelOnly #186` (distilled LoRA)
- `LTX2AttentionTunerPatch` (inside loop; bypassed default)
- `LatentUpscaleModelLoader #182` + pass2 sampler chain (Option C upscale-per-iter staging)

## Wiring details

### Model patch chain (inside loop, downstream of LoopIterationStamp)

```
UNETLoader (outside loop, runs once)
  → [SetNode "model_unpatched" — top-level]
        ↓
        [GetNode inside loop boundary, after TLO]
        ↓
        LoopIterationStamp.model  (current_iteration changes per iter)
        ↓
        AudioLoopHelperSageAttention  (model_options["transformer_options"]["optimized_attention_override"])
        ↓
        LTXVChunkFeedForward  (object_patches: transformer_blocks.{N}.ff.forward)
        ↓
        LTX2AttentionTunerPatch  (bypassed; if active: object_patches: transformer_blocks.{N}.forward)
        ↓
        LTX2_NAG  (object_patches: transformer_blocks.{N}.attn2.forward + audio_attn2.forward)
          ← nag_cond_video from top-level CLIPTextEncode
        ↓
        CFGGuider.model
```

**Rationale**: every patch is downstream of LoopIterationStamp's iter-changing input → executor re-runs them per iter → fresh `add_object_patch` calls every iter → patches guaranteed present regardless of comfy-aimdo offload behavior.

### Audio path

```
LoadAudio (full song, top-level)
  → TrimAudioDuration (clamp to song length; not 4s hardcode)
  → Set_actual_audio
       ↓
       Get_actual_audio (used by AudioLoopController, AudioLoopPlanner, initial render audio chain)
       ↓
       LTXVAudioVAEEncode (encode once, fp=25 latents/sec internally)
       → Set_full_audio_latent
            ↓
            (inside loop) Get_full_audio_latent
                          AudioLatentSlice.full_audio_latent
                          AudioLatentSlice.source_seconds ← Get_audio_duration (from AudioLoopController)
                          AudioLatentSlice.start_index    ← AudioLoopController.start_index
                          AudioLatentSlice.duration       ← LTXFramePlanner.actual_seconds (window)
                          → audio_latent (per-iter slice)
                          → LTXVAudioVideoMask (start=end=window, freeze)
                          → LTXVConcatAVLatent (audio_latent slot)
```

### Conditioning path (with A/B switch)

```
PrimitiveStringMultiline #2103 (PROMPT, used for both encoder paths)
  → CLIPTextEncode (BYPASSED, mode=4)  ──┐ user manually re-wires for A/B
                                         │
  → TimestampPromptScheduleBatchEncode  ──┴── [active by default]
       (clip + schedule + stride + duration + frame_rate=25)
       → cond_list
            → ConditioningSelectByIteration #LOOP (current_iter from TLO) → loop CFGGuider.positive
            → ConditioningSelectByIteration #INIT (current_iter UNWIRED, defaults 0) → LTXVConditioning #164.positive → initial CFGGuider.positive

CLIPTextEncode #NEGATIVE (blurry, oversaturated, ...) → Set_negative
  → Get_negative → LTXVConditioning #164.negative → initial CFGGuider.negative
  → Get_negative → loop body LTX2_NAG.nag_cond_audio (optional)
  → (also feeds loop CFGGuider.negative if not zeroed out)

CLIPTextEncode #NAG_COND_VIDEO ("still image with no motion, ...")
  → loop body LTX2_NAG.nag_cond_video  (direct wire on flat canvas)
```

### Loop math wiring

```
LTXFramePlanner widgets [width=960, height=544, target_seconds=19.88, fps=25]
  → fps_int → AudioLoopController.fps, AudioLoopPlanner.fps
  → actual_seconds → AudioLoopController.window_seconds, AudioLoopPlanner.window_seconds
  → width/height/frames → EmptyLTXVLatentVideo, smart resize targets

FloatConstant overlap_seconds=2.0
  → AudioLoopController.overlap_seconds
  → AudioLoopPlanner.overlap_seconds  (shared single source)

AudioLoopPlanner (iter-INDEPENDENT outputs)
  → total_iterations → TensorLoopOpen.iterations_in  (F5)
  → stride_seconds → TimestampPromptScheduleBatchEncode.stride_seconds
  → audio_duration → TimestampPromptScheduleBatchEncode.audio_duration

AudioLoopController (iter-DEPENDENT outputs)
  in: current_iteration ← TensorLoopOpen.current_iteration
       audio (audio_actual)
       window_seconds, overlap_seconds, fps, base_seed=42
  out: start_index → AudioLatentSlice.start_index (inside loop)
       should_stop → TensorLoopClose.stop
       audio_duration → AudioLatentSlice.source_seconds (inside loop)
       iteration_seed → RandomNoise.noise_seed (inside loop)
       overlap_latent_frames → LatentContextExtract.frame_count (inside loop)
       overlap_seconds → LTXVAudioVideoMask.video_start_time (inside loop)
```

### Sampler chain (shared by initial + loop)

```
KSamplerSelect (euler_ancestral_cfg_pp)  ← benchmark fidelity
  → both initial render sampler AND loop body sampler

ManualSigmas (canonical 9-value: "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
  → both samplers

CFGGuider with cfg=1
```

### Initial render path

```
LoadImage (firstframe) → LTXSmartImageResize → LTXVPreprocess(18)
  → Set_firstframe (+ resized variant)

EmptyLTXVLatentVideo (width/height/length from FramePlanner)
  → LTXVImgToVideoInplaceKJ (vae=video_vae, image=firstframe, strength=1.0)
  → LTXVAddGuideMulti (image_1=firstframe[0,0.7], image_2=lastframe[-1,1.0], image_3=middleframe[mid,0.3])
  → LTXVConcatAVLatent (with full audio from LTXVAudioVAEEncode)
  → SamplerCustomAdvanced (init CFGGuider, init RandomNoise seed=base_seed)
  → LTXVSeparateAVLatent
  → LTXVCropGuides (with latent — F2/F3 init branch)
  → Set_initial_latent  +  Set_reference_latent (for AdaIN)
```

### Output assembly

```
LatentConcat (samples1=Get_initial_latent, samples2=TensorLoopClose.output, dim="t")
  → TrimVideoLatentToAudio (fps from FramePlanner; snap-UP)
  → LTXVTiledVAEDecode [1,1,1,true,"auto","auto"]
  → TrimImageBatchToAudio (fps from FramePlanner)
  → Set_final_video

(audio for output)
LTXVAudioVAEDecode (from final assembled latent's audio portion via separate)
  → Set_final_audio  (or just pass through Get_actual_audio if mask=0 preserves bit-identity)

RunIdPrefix → VHS_VideoCombine.filename_prefix
VHS_VideoCombine (images=Get_final_video, audio=Get_final_audio, frame_rate=25)
```

## Per-iter cost estimate

- Inside-loop model patch re-execution: ~4× `model.clone()` (shallow), ~240 `add_object_patch` calls total, 2 small Linear projections for NAG text projection.
- Per Agent 3's analysis: well under 1% of iter wall-time (sampling dominates).
- Pre-iter audio re-encode is ELIMINATED (encoded once outside loop, sliced per iter).
- CLIP runs ONCE outside loop (per-iter schedule entries pre-encoded).

## Audit invariants — coverage on flat canvas

Per audit-compat subagent:

**Works as-is**: `graph_acyclic`, `link_integrity`, `widget_shape`, F2 (preprocess), F4, F5, F6, F7, F8, F14, F15, F16, F18, decoder/resolution/latent_volume invariants, controller widget shape checks.

**Gracefully skips (no false-OK)**: F3 (cropguides_symmetry), F12 (iclora wiring), F17 (cfg_guider walker), `audio_latent_slice_*`, `ltx2_nag_reaches_loop`, `prompt_relay_leak` — these early-return when `wf["definitions"]["subgraphs"]` is empty. Coverage gap, no danger.

**New invariants worth adding later** (not V1 blockers):
1. `model_patch_chain_lives_between_TLO_and_TLC`
2. `loop_body_nodes_between_tlo_tlc` (catch stranded patches that re-execute but go nowhere)
3. `alc_outputs_dont_escape_loop` (explicit; was implicit via subgraph boundary)

## Build approach

Write `scripts/build_fml2v_audio_loop.py` — from-scratch builder using `WorkflowEditor`. Reads benchmark `fml2v_var_d_audio_input.json` for benchmark-specific structures (multi-frame anchoring, prompt source, output mux) and canonical `audio-loop-music-video_latent.json` for loop-spine references. Produces a fresh flat-canvas variant.

Estimated complexity: ~70 nodes, ~110 links, ~600-800 lines of Python.

## Risks / unknowns

1. **`PrimitiveStringMultiline #2103` as shared prompt source**: keeping benchmark's multi-line prompt as input to BOTH the bypassed static `CLIPTextEncode` AND the schedule encoder requires the schedule encoder to accept a single-entry string. `TimestampPromptScheduleBatchEncode.schedule` widget format: lines like `"0:00+: <prompt>"`. The benchmark's prompt is a plain string; user must convert if they want schedule format. Mitigation: ship with `0:00+: video of a man dancing and singing` as default and a MarkdownNote on the schedule format.

2. **The two-pass benchmark's `LatentUpscaleModelLoader` + pass2 chain stays bypassed in V1.** Re-enabling for per-iter Option C would need additional plumbing (the upscaler output needs to feed pass2's `latent_image`, and the pass2 output needs to assemble correctly). V1 doesn't wire this; V1.5 adds it if desired.

3. **Audit gaps on F3/F17/F12 are real.** The variant won't catch loop-CFGGuider-missing-cropguides on the new flat canvas until those checks adapt. Mitigation: experimental/ placement means WARN-level is acceptable; flag in CLAUDE.md "Pending review" for followup.

4. **Per-iter patch re-execution overhead** is documented as <1% but unverified at scale. If iter wall-time noticeably increases vs canonical subgraph variant, the patch chain may need to move BACK outside the loop. Empirically verify after V1 ships.

5. **`AdainLatent` reference source on flat canvas**: canonical's subgraph pulls `reference_latent` from a subgraph input slot. On flat canvas, this is a direct wire from `Set_reference_latent` (outside loop) to `LTXVAdainLatent.reference_latents` (inside loop). Need to verify AdainLatent accepts the latent shape AdaIN expects.

## Followups (not V1)

- Add `model_patch_chain_lives_between_TLO_and_TLC` audit invariant
- Add `alc_outputs_dont_escape_loop` audit invariant
- F3, F17 walkers refactored to use a "loop-body locator" abstraction
- V1.5 / V2: re-enable two-pass upscale (Option C per-iter or Option B deferred-workflow)
- If V1 ships well: promote flat-canvas pattern to canonical workflow variant (test render quality first)
