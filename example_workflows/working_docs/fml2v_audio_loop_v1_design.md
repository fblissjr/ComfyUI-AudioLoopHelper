# fml2v_var_d_audio_loop V1 — design

Last updated: 2026-05-24

> **V2 ARCHITECTURAL PIVOT (2026-05-19).** Live-render verification of
> V1's two-pass-refine + spatial-upsample loop body produced yellow
> tiger-stripe artifacts in loop iters regardless of guide strengths or
> AdaIN state. Four diagnostic apply-scripts (`apply_fml2v_smoke_toggle`,
> `apply_fml2v_no_pass2_blend`, `apply_fml2v_pass1_recovery`,
> `apply_fml2v_inplace_kj_p1`) converged on the root cause: the loop
> body never consumed prev-iter content. `TLO.previous_value` had no
> downstream consumer; every iter started from `EmptyLatent` in the
> middle frames, and 8 distilled steps cannot invent coherent middle
> frames from soft anchors alone.
>
> **Pivot to "option 3": full-res two-pass refine + `LatentContextExtract`
> from prev iter, no spatial upsampler.** Staged via
> `scripts/apply_fml2v_option3_context_extract.py`. Resolves both
> architectural deltas in the V1 banner below — ContextExtract now in
> the chain, single resolution end-to-end (so the half-res / full-res
> mismatch that motivated dropping ContextExtract in V1 disappears).
>
> **Per-iter keyframe re-anchoring SUPERSEDED + RELOCATED (2026-05-24).**
> The keyframe feature no longer lives on the fml2v flat-canvas build at
> all. It was rebuilt on the proven canonical subgraph and ships as
> `example_workflows/audio-loop-music-video_latent_keyframe.json`
> (generator `scripts/apply_keyframe_iter_anchor.py`). Design record:
> `example_workflows/working_docs/keyframe_iter_anchor_design.md`. The
> `apply_fml2v_iter_keyframe.py` script and the in-loop `noise_mask=0`
> writer node design referenced in earlier drafts were removed —
> `LTXIterKeyframeSchedule` is now an OUTSIDE-loop selector feeding the
> canonical's existing `guide_latent` (hard lock comes from
> `first_frame_guide_strength=1.0`, not an in-loop mask write).
>
> The option3 pivot below (ContextExtract + single-resolution loop body)
> is what proved the canonical's loop-body design is the right spine —
> which is why the keyframe feature was moved off fml2v entirely. fml2v
> remains an experimental two-pass-refine record; not the keyframe home.
>
> Original V1 banner + recipe follow. V1 two-pass + 960×512 + no-ContextExtract
> are preserved for context but superseded by V2.

> **STATUS: BUILD COMPLETE as of 2026-05-18.** All six phases of
> `scripts/build_fml2v_audio_loop.py` implemented; workflow at
> `example_workflows/experimental/fml2v_var_d_audio_loop.json`
> (~239k bytes). Audit: 33 OK / 3 WARN / 0 ERR (WARNs are intentional
> two-pass design — pass2 `euler_cfg_pp` + pass1 `euler_ancestral_cfg_pp`
> + 4-value pass2 ManualSigmas, all deviation from single-stage
> canonical). Containment: clean via `scripts/verify_loop_containment.py`.
>
> **Two architectural deltas vs the design below** (don't revert these
> when re-reading the doc; the recipe text is preserved for intent +
> rationale):
>
> 1. **Resolution: 960×512, not 960×544.** Two-pass refine requires
>    div-by-64 base so half-res pass1 stays div-by-32 and the 2× upsampler
>    returns to a dim matching the init-render output (Phase 6's
>    `LatentConcat` welds across iters). 544 fails (272 not div-32 at
>    half-res); 512 is the closest div-64. Touches `LTXFramePlanner` +
>    `LTXSmartImageResize` widgets.
> 2. **`LatentContextExtract` removed from loop pass1.** The doc's
>    topology shows ContextExtract→OverlapTrim→mask on the INPUT side
>    (mirroring canonical single-pass). That doesn't translate to
>    two-pass: `TLO.previous_value` is full-res (post-pass2 upsample)
>    while pass1 samples at half-res, so the direct context→mask wire
>    shape-mismatches with no clean latent-space downscale. Removed
>    entirely; cross-iter continuity now comes from (a) fixed init-image
>    anchor via `LTXVAddLatentGuide` trailing frame at strength 0.7,
>    (b) frozen audio driving motion via cross-attention, (c) prompt-
>    schedule evolution. `LatentOverlapTrim` moved to OUTPUT side
>    (post-AdaIN_final, pre-IterationCleanup) per canonical placement —
>    trims the overlap region from per-iter output so adjacent windows
>    don't double-up at Phase 6's `LatentConcat`. **Open question for
>    live render**: does this trade-off produce visible seams between
>    iters? V2 enhancement is to re-add ContextExtract on the pass2
>    (full-res) side instead, or add a downscale node for pass1.
>
> **How to re-verify on a fresh clone**:
> 1. `uv run --group dev python scripts/build_fml2v_audio_loop.py` — all six phases log + write ~239k workflow.
> 2. `uv run --group dev python scripts/audit_workflows.py example_workflows/experimental/fml2v_var_d_audio_loop.json` — 33 OK / 3 WARN / 0 ERR expected.
> 3. `uv run --group dev python scripts/verify_loop_containment.py` — should report "OK: every active iter-dependent node reaches TensorLoopClose."
>
> **What's NOT done**: live-render verification in ComfyUI. Audit +
> containment script verify it structurally; only a real render
> confirms (a) prompt validation succeeds, (b) latent shapes flow
> across init→pass1→upsample→pass2→assembly, (c) the no-ContextExtract
> trade-off doesn't produce visible iter-boundary seams.
>
> Original design recipe follows. The topology diagrams reflect the
> doc's intent; the actual build is per the script. Both line up
> modulo the two deltas above.

## Goal

Add tensor-loop iteration to the `fml2v_var_d_audio_input` benchmark workflow so it can render a full-length song with audio-video sync (replacing the current 4-second-audio + 15-second-video hardcode). Output lives at `example_workflows/experimental/fml2v_var_d_audio_loop.json`.

V1 scope (REVISED 2026-05-16 after re-reading `workflow_quality_delta_analysis.md`): **Two-pass refine + spatial upsample IN the loop body (Option B from earlier analysis)**, flat canvas, multi-keyframe (first+last only) per iter. Earlier V1 plan was single-pass-with-pass2-bypassed; revised to include pass2 because `workflow_quality_delta_analysis.md` ranks two-pass refine as the *dominant* predicted quality lever — the main reason B looks visibly better than A. Shipping V1 without it means losing the quality differentiator that motivates this work. Middle-keyframe per iter deferred to V2 as the next easy extension (~5-7 added nodes).

## Why Option B (two-pass refine in the loop body)

Predicted quality lever ranking (merged from the archived
`workflow_quality_delta_analysis.md`, A=canonical loop vs B=benchmark fml2v):

1. **Two-pass refine + spatial upsample (DOMINANT)** — half-res commit
   → `LTXVLatentUpsampler` (2× spatial) → 3-step refine (sigmas
   `0.85, 0.725, 0.4219, 0.0`). The main reason B looks visibly better than A.
2. **`euler_cfg_pp` / `euler_ancestral_cfg_pp` samplers + NAG at cfg=1 (STRONG SECONDARY)** — at `cfg=1` vanilla `euler` IGNORES the negative path, so canonical A's NAG is effectively decorative. `_cfg_pp` engages it. So NAG in B does real work; NAG in A doesn't.
3. **Multi-keyframe anchoring (MEANINGFUL)** — middle keyframe reduces identity drift. B uses N=3 (first/mid/last) at pass1; N=2 (first/last) at pass2.
4. **Multi-image init resize pipeline (SMALL)** — V1 uses `LTXSmartImageResize` (anti-aliased).

V1 captures #1 and #2 directly. #3 in V1 is **first+last only** (matches B's pass2 shape); middle keyframe deferred to V2 (~5-7 added nodes — easy extension).

**What's NOT a quality lever** (verified — don't re-litigate):
- `auto` vs `auto_mask_aware` sage mode — masked path doesn't fire on audio-loop workflows
- `LTX2AttentionTunerPatch` active vs bypassed — identity widgets, no-op math
- LoRA strength 0.5 vs 0.6 — both bypassed in canonical
- GGUF model loaders — bypassed

## Sampler chain — two-pass inside the loop body

**Pass 1 (denoise at HALF resolution):**
- `KSamplerSelect`: `euler_ancestral_cfg_pp` (B's choice; preserves benchmark fidelity)
- `ManualSigmas`: canonical 9-value 8-step distilled chain
- `CFGGuider cfg=1` (consumes in-loop NAG-patched model)
- Latent volume: width/2 × height/2 — derive via `ComfyMathExpression a/2` from `LTXFramePlanner.width`/`height`

**Between passes (still inside loop body):**
- `LTXVSeparateAVLatent` (discard audio half — upscaler is video-only)
- `LTXVCropGuides`
- `LTXVLatentUpsampler` (2× spatial in latent space; uses `Get_upscale_model`)
- `LTXVAddGuideMulti` (pass2, N=2: first+last keyframe images at indices 0 and -1)
- `LTXVConcatAVLatent` (re-attach the same audio latent from Pass 1 input)

**Pass 2 (refine at FULL resolution):**
- `KSamplerSelect`: `euler_cfg_pp`
- `ManualSigmas`: `"0.85, 0.7250, 0.4219, 0.0"` (3-step refine starting at σ=0.85)
- `CFGGuider cfg=1` (same in-loop NAG-patched model as Pass 1)
- F3 symmetry: BOTH guiders' positive/negative must flow through `LTXVCropGuides` before reaching the sampler.

**Audio handling across the two passes**: audio attaches to Pass 1 input via `LTXVConcatAVLatent` (with `noise_mask=0` freeze on the audio half from `AudioLatentSlice` → `LTXVAudioVideoMask`). After Pass 1, `LTXVSeparateAVLatent` strips audio (upscaler is video-only). The same audio latent (cached in a local SetNode within the loop body) re-attaches before Pass 2 via a second `LTXVConcatAVLatent`. Per CLAUDE.md, this is also the canonical pattern in `build_upscale_workflow.py`.

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
│  [LatentContextExtract REMOVED — see status banner; full-res     │
│   prev-iter vs half-res pass1 shape-mismatches in two-pass build]│
│  [LatentOverlapTrim moved to OUTPUT side — see status banner]    │
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
LTXFramePlanner widgets [width=960, height=512, target_seconds=19.88, fps=25]
  (note: 512 not 544 — two-pass refine requires div-64 base; see status banner)
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
       overlap_latent_frames → LatentOverlapTrim.overlap_latent_frames (output-side, post-AdaIN; ContextExtract removed — see status banner)
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
