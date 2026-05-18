Last updated: 2026-05-15

# Subgraph chain reference — `audio-loop-music-video_latent.json`

> **Partially stale (added 2026-05-16)**: scattered references to fps=24 as canonical OR `first_frame_guide_strength=1.0` are pre-2026-05-16. **Today's canonical: fps=25, first_frame_guide_strength=0.7, target_seconds=19.88.** Body content is otherwise current.

Authoritative reference for the loop-body subgraph (definitions id `b4973d68-09b9-4da5-9845-38ad62ae9aca`, name `extension`) invoked by top-level node `#843`. Source: `example_workflows/audio-loop-music-video_latent.json`. 18 internal nodes, 52 internal links, 20 input slots, 1 output slot.

Companion docs: `docs/reference/audio_loop_controller.md` (controller semantics + iter-state cycle topology), `docs/reference/pipeline_flow_latent.md` (F2/F3 symmetry trace), `docs/reference/sampler_reference.md` (sigma chain + sampler choice rationale).

## 1. Subgraph IO table

The invoker `#843` is type `b4973d68-09b9-4da5-9845-38ad62ae9aca` (subgraph instance, not a registered ComfyNode). Each invoker input slot corresponds 1:1 to a `definitions.subgraphs[0].inputs[]` schema entry. Internal nodes pull from the input distributor (virtual id `-10`); the single internal output is `IterationCleanup #2007` feeding the output collector (virtual id `-20`).

Iteration-state column legend:
- **iter** = value changes every iteration (transitively reaches `current_iteration`)
- **planner** = derived from `LTXFramePlanner` / `AudioLoopPlanner` — fixed per render
- **fixed** = loaded once outside loop (model/VAE/static widget)

| Slot | Name | Type | Top-level source (via `#843`) | Internal consumer | Iter-state |
|---|---|---|---|---|---|
| 0  | `sampler` | SAMPLER | `#154 KSamplerSelect "euler"` → link 3125 | `#573 SamplerCustomAdvanced.sampler` | fixed |
| 1  | `sigmas` | SIGMAS | `#580 GetNode "Get_sigmas"` → link 2035 (canonical 8-step distilled chain) | `#573 SamplerCustomAdvanced.sigmas` | fixed |
| 2  | `model` | MODEL | `#1618 LoopIterationStamp` → link 3055 | `#644 CFGGuider.model` | iter (stamp writes `transformer_options["iteration"]` per pass — `nodes.py::LoopIterationStamp._stamp_impl`) |
| 3  | `vae` | VAE | `#619 GetNode "Get_video_vae"` → link 2037 | `#1519 LTXVAddLatentGuide.vae`, `#1640 LTXAddVideoICLoRAGuide.vae` (bypassed) | fixed |
| 4  | `previous_latent` | LATENT | `#1539 TensorLoopOpen.previous_value` → link 2869 | `#2004 LatentContextExtract.latent` | iter (tensor-loop framework feeds last iter's output here) |
| 5  | `video_end_time` | FLOAT | `#1634 LTXFramePlanner.actual_seconds` (slot 3) → link 3139 | `#606 LTXVAudioVideoMask` (slots 3, 4, 5 all wired here — `video_end_time` AND `audio_start_time/audio_end_time`), `#2012 AudioLatentSlice.duration_seconds`. F3 audit reference. | planner |
| 6  | `positive` | CONDITIONING | `#1633 LTXVReferenceAudio.positive` (mode=4 bypassed → passes through `#1616 ConditioningSelectByIteration`) → link 3121 | `#1519 LTXVAddLatentGuide.positive` | iter (`ConditioningSelectByIteration` selects per-iter from the pre-encoded schedule batch — `TimestampPromptScheduleBatchEncode` is one-shot outside loop, this picks by `current_iteration`) |
| 7  | `negative` | CONDITIONING | `#1633 LTXVReferenceAudio.negative` (bypassed → passes through `#164 LTXVConditioning.negative` zeroed) → link 3123 | `#1519 LTXVAddLatentGuide.negative` | fixed (zeroed neg, runtime-inert at CFG=1; see root CLAUDE.md "wired-correctly but runtime-inert" gotcha) |
| 8  | `guide_latent` | LATENT | `#1617 VAEEncode "init image → guide latent"` → link 3052 | `#1519 LTXVAddLatentGuide.guiding_latent` (init-image anchor per iter at `latent_idx=-1`) | fixed (init encoded ONCE outside loop) |
| 9  | `audio_vae` | VAE | `#599 GetNode "Get_audio_vae"` → link 2044 | `#598 LTXVAudioVAEEncode.audio_vae` (mode=4 bypassed) | fixed |
| 10 | `audio` | AUDIO | `#641 GetNode "Get_actual_audio"` → link 2045 | `#600 TrimAudioDuration.audio` (mode=4 bypassed) | fixed |
| 11 | `start_index` | FLOAT | `#1582 AudioLoopController.start_index` (out[0]) → link 2944 | `#2012 AudioLatentSlice.start_seconds`, `#600 TrimAudioDuration.start_index` (bypassed) | **iter** (ALC outputs are iter-dependent per root CLAUDE.md — `current_iteration` transitively reaches every ALC output) |
| 12 | `num_guides.strength_1` | FLOAT | `#1269 FloatConstant "first_frame_guide_strength" = 1.0` → link 3128 | `#1519 LTXVAddLatentGuide.strength` | fixed |
| 13 | `noise_seed` | INT | `#1582 AudioLoopController.iteration_seed` (out[3]) → link 2941 | `#574 RandomNoise.noise_seed` | **iter** (base_seed + current_iteration) |
| 14 | `num_frames` | INT | `#1582 AudioLoopController.overlap_latent_frames` (out[6]) → link 2954 | `#2004 LatentContextExtract.overlap_latent_frames`, `#2005 LatentOverlapTrim.overlap_latent_frames` | iter (ALC output; quantized stride math) |
| 15 | `reference_latent` | LATENT | `#245 LTXVSeparateAVLatent.video_latent` (initial render's separated video latent) → link 3005 | `#2006 LTXVAdainLatent.reference` | fixed (init render's latent — color reference for AdaIN) |
| 16 | `video_start_time` | FLOAT | `#1582 AudioLoopController.overlap_seconds` (out[7]) → link 3007 | `#606 LTXVAudioVideoMask.video_start_time` | **iter** (effective quantized overlap — see `docs/reference/audio_loop_controller.md`) |
| 17 | `reference_video` | IMAGE | `#1638 LTXVPreprocess "Preprocess ref-video (F2 symmetric)"` → link 3147 | `#1639 GetImageRangeFromBatch.images` → `#1640 LTXAddVideoICLoRAGuide` (mode=4 bypassed by default) | fixed |
| 18 | `full_audio_latent` | LATENT | `#2011 GetNode "Get_full_audio_latent"` → link 3162 | `#2012 AudioLatentSlice.latent` | fixed (audio VAE-encoded ONCE outside loop) |
| 19 | `source_seconds` | FLOAT | `#1582 AudioLoopController.audio_duration` (out[2]) → link 3172 | `#2012 AudioLatentSlice.source_seconds` | **iter** (sourced through ALC — same iter-state propagation as slots 11/13/14/16) |

**Output slot 0** `extended_latent: LATENT` (link 3042) ← `#2007 IterationCleanup.latent` → `TensorLoopClose.value_in` at top level.

## 2. Internal node-by-node walkthrough (execution order)

Order field from `definitions.subgraphs[0].nodes[*].order`. Two nodes share `order=0` (sampler `#573` and `AudioLatentSlice #2012`) — they're independent leaves of the DAG.

### `#573 SamplerCustomAdvanced` — order 0, mode 0

WHY: the per-iter video sampler. Distilled 8-step euler path (sampler + sigmas come from outside the subgraph so the whole render shares one source of truth).
- `noise` ← `#574 RandomNoise.NOISE`
- `guider` ← `#644 CFGGuider.GUIDER`
- `sampler` ← `SG_INPUT_SLOT[0]` (top-level `KSamplerSelect "euler"`)
- `sigmas` ← `SG_INPUT_SLOT[1]` (top-level canonical sigmas)
- `latent_image` ← `#583 LTXVConcatAVLatent.latent` (the masked AV-packed latent)
- `output:LATENT` → `#596 LTXVSeparateAVLatent.av_latent` (raw post-sampler AV latent, still with audio + video frames concatenated)
- `denoised_output` unused

### `#2012 AudioLatentSlice` — order 0, mode 0, title "Slice full-audio latent for this iter"

WHY: pulls the iter-window-sized audio latent out of the pre-encoded full-song audio latent. Audio is VAE-encoded once outside the loop (slot 18), then sliced here per iter so the loop body doesn't pay re-encode cost. Iter-dependent through `start_index` (slot 11) and `duration_seconds` (slot 5; planner constant — duration is fixed, only start moves).
- `latent` ← `SG_INPUT_SLOT[18]` (full_audio_latent)
- `source_seconds` ← `SG_INPUT_SLOT[19]` (from ALC.audio_duration)
- `start_seconds` ← `SG_INPUT_SLOT[11]` (from ALC.start_index)
- `duration_seconds` ← `SG_INPUT_SLOT[5]` (from FramePlanner.actual_seconds)
- `LATENT` → `#606 LTXVAudioVideoMask.audio_latent`

### `#574 RandomNoise` — order 1, mode 0

WHY: per-iter noise tensor seeded by ALC's `iteration_seed`. Reused across the sampler chain inside this iter (one noise per iter).
- `noise_seed` ← `SG_INPUT_SLOT[13]` (ALC.iteration_seed)
- `NOISE` → `#573.noise`

### `#583 LTXVConcatAVLatent` — order 2, mode 0

WHY: packs video + audio latents into the NestedTensor format LTX 2.3's masked self-attn expects. Audio path is sacred (root CLAUDE.md) — this is the canonical concat node.
- `video_latent` ← `#1640 LTXAddVideoICLoRAGuide.latent` (when bypassed, passes through `#1519`'s latent slot; effectively `#1519 LTXVAddLatentGuide.latent`)
- `audio_latent` ← `#606 LTXVAudioVideoMask.audio_latent`
- `latent` → `#573.latent_image`

### `#596 LTXVSeparateAVLatent` — order 3, mode 0

WHY: undoes the concat after the sampler so video frames can be processed independently downstream. Audio latent output is unused — audio is FROZEN, audio frames have `noise_mask=0`, so the post-sampler audio is bit-identical to pre-sampler and we discard it.
- `av_latent` ← `#573.output`
- `video_latent` → `#2008 LTXVCropGuides.latent`
- `audio_latent` → unused

### `#598 LTXVAudioVAEEncode` — order 4, **mode 4 (bypassed)**

Legacy per-iter audio VAE encode path. Pre-2026-04 the loop encoded audio every iter; replaced by the once-outside-loop encode + `AudioLatentSlice` (#2012). Kept bypassed for revert capability; bypass passes inputs to outputs of same TYPE only (none match cleanly) → dead-ends silently.

### `#600 TrimAudioDuration` — order 5, **mode 4 (bypassed)**

Companion to `#598`. Same legacy path. Bypassed.

### `#606 LTXVAudioVideoMask` — order 6, mode 0

WHY: builds the `noise_mask` for this iter's AV-concat latent. Audio frames get mask=0 (frozen — never regenerated). Video frames get mask values driven by `video_start_time` (the overlap region) so the overlap latents are context (mask=0), the new region is fresh (mask=1). **Wiring is intentional**: `audio_start_time = audio_end_time = video_end_time` (all wired to `SG_INPUT_SLOT[5]`) — the empty audio range keeps audio fixed. Don't change (root CLAUDE.md).
- `video_latent` ← `#2004 LatentContextExtract.context` (the overlap-region context from prev iter)
- `audio_latent` ← `#2012 AudioLatentSlice.LATENT`
- `video_start_time` ← `SG_INPUT_SLOT[16]` (ALC.overlap_seconds — the boundary between context and fresh)
- `video_end_time`, `audio_start_time`, `audio_end_time` ← all from `SG_INPUT_SLOT[5]` (FramePlanner.actual_seconds)
- `video_latent` → `#1519 LTXVAddLatentGuide.latent`
- `audio_latent` → `#583 LTXVConcatAVLatent.audio_latent`

Widget values: `[25, 1, 10, 10, 10, "pad", "add"]` (fps, num_audio_streams, three time defaults, audio_strategy, mask_op).

### `#644 CFGGuider` — order 7, mode 0

WHY: wraps positive/negative conditioning into a GUIDER for `SamplerCustomAdvanced`. CFG=1 (distilled path; effectively no CFG math — negative slot is wired-correctly-but-runtime-inert per root CLAUDE.md, but `CFGGuider` validates both slots so the chain stays).
- `model` ← `SG_INPUT_SLOT[2]` (LoopIterationStamp-clone of the top-level model)
- `positive` ← `#655 LTXVCropGuidesNoLatent.positive` **(F3 enforcement)**
- `negative` ← `#655 LTXVCropGuidesNoLatent.negative` **(F3 enforcement)**
- `GUIDER` → `#573.guider`
- widget `[cfg=1]`

### `#655 LTXVCropGuidesNoLatent` — order 8, mode 0 — **F3 enforcement (CONDITIONING half)**

WHY: F3 invariant — loop `CFGGuider.positive/negative` MUST flow through a CropGuides chain (mirrors the initial path's `#164 → #381 → #153`). This is the CONDITIONING-only variant: strips guide metadata from conditioning before it reaches CFGGuider. The companion `LATENT` half is `#2008` below — pair was introduced by `scripts/apply_split_cropguides.py` to break a loop cycle (CFGGuider ← CropGuides ← SeparateAV ← Sampler ← CFGGuider). The audit `loop_cropguides_symmetry` hardcodes node IDs `{644, 655, 1519}`.
- `positive` ← `#1640 LTXAddVideoICLoRAGuide.positive` (when bypassed → passes through `#1519.positive`)
- `negative` ← `#1640.negative` (when bypassed → passes through `#1519.negative`)
- `positive` → `#644.positive`
- `negative` → `#644.negative`

### `#1519 LTXVAddLatentGuide` — order 9, mode 0

WHY: anchors every iter to the init image via `guiding_latent` at `latent_idx=-1` (last latent frame) with `strength=1.0` from `#1269 FloatConstant`. This is the per-iter init-anchor that prevents identity drift across iterations. `first_frame_guide_strength` knob (slot 12) — lower for expressivity at the cost of cross-iter drift.
- `vae` ← `SG_INPUT_SLOT[3]`
- `positive` ← `SG_INPUT_SLOT[6]` (per-iter selected conditioning)
- `negative` ← `SG_INPUT_SLOT[7]` (zeroed)
- `latent` ← `#606 LTXVAudioVideoMask.video_latent` (masked context)
- `guiding_latent` ← `SG_INPUT_SLOT[8]` (VAE-encoded init image, once outside loop)
- `strength` ← `SG_INPUT_SLOT[12]`
- `positive` → `#1640.positive`, also fanout to `#655.positive` / `#2008.positive` via `#1640` when bypassed
- `negative` → `#1640.negative`
- `latent` → `#1640.latent` (passes to `#583` when `#1640` bypassed)
- widget `[-1, 1]` (`keyframe_idx=-1`, internal strength scaling)

### `#1639 GetImageRangeFromBatch` — order 10, mode 0, title "Slice ref-video for this iter"

WHY: KJNodes utility — slices a window of ref-video frames for this iter from the (preprocessed) ref-video batch. Drives the IC-LoRA video-reference path. Active even though its consumer `#1640` is bypassed by default (deferred-execution pattern; only matters when user un-bypasses #1640).
- `images` ← `SG_INPUT_SLOT[17]` (preprocessed ref-video)
- `masks` unconnected
- `IMAGE` → `#1640.image`
- widget `[0, 25]` (start, length)

### `#1640 LTXAddVideoICLoRAGuide` — order 11, **mode 4 (bypassed)** — title "IC-LoRA Guide (video reference)"

WHY: F12 video-reference IC-LoRA — adds the sliced ref-video as an IC-LoRA guide inside the subgraph between `#1519` and the F3 cropguides chain. Bypassed in canonical; un-bypass loader + guide + ref-video to enable. When bypassed, passes inputs to outputs of same TYPE only — pos/neg conditioning + latent pass through unchanged.

### `#2004 LatentContextExtract` — order 12, mode 0, title "Context Extract"

WHY: extracts the trailing `overlap_latent_frames` from the previous iter's output latent. Replaces raw `LTXVSelectLatents` because it auto-strips `noise_mask` (matching `VAEEncode` behavior). Reference: `nodes.py::LatentContextExtract` — "extracts last N latent frames as context for the next loop iteration."
- `latent` ← `SG_INPUT_SLOT[4]` (TensorLoopOpen.previous_value)
- `overlap_latent_frames` ← `SG_INPUT_SLOT[14]` (ALC.overlap_latent_frames)
- `context:LATENT` → `#606 LTXVAudioVideoMask.video_latent`
- widget default `[4]`

### `#2005 LatentOverlapTrim` — order 13, mode 0, title "Overlap Trim"

WHY: trims the leading `overlap_latent_frames` from the sampler's output — these were the context region, the rest is the new content for this iter. Strips noise_mask. Reference: `nodes.py::LatentOverlapTrim` — "trims first N latent frames (overlap) from sampler output. Keeps new content only."
- `latent` ← `#2006 LTXVAdainLatent.LATENT`
- `overlap_latent_frames` ← `SG_INPUT_SLOT[14]`
- `trimmed:LATENT` → `#2007 IterationCleanup.latent`
- widget default `[4]`

### `#2006 LTXVAdainLatent` — order 14, mode 0, title "AdaIN Color Correction"

WHY: latent-space color correction — matches the first/second moments of `latents` to `reference` (the initial-render's separated video latent). Prevents cross-iter color drift. Applied **before** `LatentOverlapTrim` so the AdaIN works on the full sampler output (incl. overlap region) before trim.
- `latents` ← `#2008 LTXVCropGuides.latent` (LATENT half of F3 split)
- `reference` ← `SG_INPUT_SLOT[15]`
- `LATENT` → `#2005.latent`
- widget `[0.2, false]` (strength, optional flag)

### `#2007 IterationCleanup` — order 15, mode 0

WHY: LATENT passthrough that runs PyTorch allocator hygiene (`gc.collect()` + `torch.cuda.empty_cache()` in `"always"` mode). Placed at the subgraph output so every iter ends with clean allocator state — reduces fragmentation across the loop. Reference: `nodes.py::IterationCleanup` — comfy-aimdo recommends per-run allocator flush.
- `latent` ← `#2005.trimmed`
- `latent` → `SG_OUTPUT_SLOT[0]` (extended_latent)
- widget `["always"]`

### `#2008 LTXVCropGuides` — order 16, mode 0, title "CropGuides (LATENT-only — split)" — **F3 enforcement (LATENT half)**

WHY: companion to `#655` — same F3 invariant, LATENT half of the split. Strips guide metadata from the post-sampler latent before AdaIN. Pair introduced to break the loop cycle that single `LTXVCropGuides` placement would have closed.
- `positive` ← `#1640.positive`
- `negative` ← `#1640.negative`
- `latent` ← `#596 LTXVSeparateAVLatent.video_latent` (sampler output, video half)
- `latent` → `#2006 LTXVAdainLatent.latents`
- pos/neg outputs unused (CONDITIONING already routed via `#655` parallel)

## 3. Insertion point for a refine pass (Variant 2a)

**Where to tap the latent for refine input**:

The structurally appropriate tap point is **after `#2008 LTXVCropGuides` (LATENT output)** — i.e. the same latent that today flows into `#2006 LTXVAdainLatent.latents`. Reasoning:

- `#573 SamplerCustomAdvanced.output` is the raw post-sampler AV-concat latent (still has audio frames concatenated, still carries guide metadata).
- `#596 LTXVSeparateAVLatent` strips the audio half.
- `#2008 LTXVCropGuides` strips the guide metadata. **This is the first latent in the chain that's structurally a clean video latent ready for another sampler.**
- After `#2006 LTXVAdainLatent` is also viable (and arguably better — color-corrected before refine). After `#2005 LatentOverlapTrim` is **wrong** for refine: trimming removes the overlap context that the refine sampler needs to keep the seam coherent.

**Recommended chain**: `#2008 → #2006 LTXVAdainLatent → [NEW refine sampler] → #2005 LatentOverlapTrim → #2007 IterationCleanup`. AdaIN runs on pass-1 output; refine sampler refines the color-corrected latent; trim happens after refine so the overlap context is fed to the refine sampler too.

**Alternative**: `#2008 → [NEW refine sampler] → #2006 AdaIN → #2005 trim → #2007`. AdaIN-after-refine if you want the refine pass's high-frequency detail preserved without AdaIN's smoothing of it.

**Inputs the new `SamplerCustomAdvanced` would need**:

| Input | Source recommendation | Why |
|---|---|---|
| `noise` | **NEW `RandomNoise`** with seed = `ALC.iteration_seed + offset` (e.g. wire through a `SimpleCalculatorKJ` adding a constant) | Reusing `#574`'s NOISE risks double-sampling identical noise; a per-pass seed offset gives independence while staying deterministic |
| `guider` | **Reuse `#644 CFGGuider`** | Same conditioning, same CFG=1; no need to duplicate. CFGGuider's GUIDER output already fans out trivially |
| `sampler` | **NEW `KSamplerSelect` with `euler_cfg_pp`** | B's pass-2 choice; non-ancestral preserves determinism. Per root CLAUDE.md no `euler_ancestral*` for distilled paths |
| `sigmas` | **NEW `ManualSigmas`** with `"0.909375, 0.725, 0.421875, 0.0"` (3 steps from the lower tail of the canonical 8-step chain) | Aligns with the distilled chain's lower-noise region — refines without re-introducing high-noise denoise work |
| `latent_image` | Output of `#2006 LTXVAdainLatent` (or `#2008` if AdaIN-after-refine) | See chain reasoning above |

**F3 implication**: the refine pass's `CFGGuider` re-use of `#644` means the F3 invariant continues to hold structurally (positive/negative still flow `#1519 → #1640 → #655 → #644`). **If you instead add a separate CFGGuider for the refine pass**, the F3 audit will NOT catch a missing CropGuides chain on it — `loop_cropguides_symmetry` hardcodes node ID `644`. New CFGGuider node IDs are invisible to that audit. Either reuse `#644` (recommended) or extend the audit to enumerate all CFGGuider nodes in the subgraph.

**noise_mask implications**: the `noise_mask` set by `#606 LTXVAudioVideoMask` is attached to the AV-concat latent before `#573`. After `#596 LTXVSeparateAVLatent` and `#2008 LTXVCropGuides`, the noise_mask is **stripped** (LatentContextExtract / LatentOverlapTrim explicitly `s.pop("noise_mask", None)`; LTXVCropGuides similarly strips guide-related metadata). So:

- **The refine sampler operates on a video-only latent with no noise_mask** — it samples the whole video latent freely. This is correct for a refine pass since the audio is no longer in the tensor and the overlap context is preserved by `LatentOverlapTrim` running *after* refine.
- **You do NOT need to re-establish the audio mask** for the refine pass. Audio is never in the refine sampler's input — it was stripped at `#596` and is re-concatenated only via `#583` for the *next* iter's pass-1 sampler. The refine pass is video-only by topology.
- **Important**: if you want refine to respect the overlap region as fixed context (i.e. only re-sample the new content), you'd need to re-attach a noise_mask before the refine sampler. Without that, refine will re-sample the overlap latents too — this MAY visibly alter the seam continuity that pass 1 spent effort matching. Consider testing both: (a) refine without mask (simpler; trust AdaIN + overlap-trim to maintain continuity), (b) refine with a fresh noise_mask via a new `LTXVAudioVideoMask`-like node or a manual mask setter that flags the leading `overlap_latent_frames` as mask=0.

**Per-iter wall-time impact estimate**:

Pass 1 is 9 sigma values → 8 effective steps; pass 2 is 4 sigma values → 3 effective steps. Naive ratio: refine adds 3/8 = **37.5% more sampler-time** per iter at equal per-step cost. But:
- Refine operates on a video-only latent (post-separate), so the per-step cost is roughly the same as pass 1's video-attn cost minus the audio-attn cost. Audio frames are a small fraction of the AV-concat for LTX 2.3 (audio VAE emits ~25 latents/sec vs ~3 video latents/sec at 24fps), so per-step cost is ~80-90% of pass-1 per-step.
- Realistic estimate: **+30-34% per iter wall-time** from sampling alone (3/8 × 0.85). Add ~1-2% for the extra `RandomNoise` + node-graph overhead.
- Caveat: this is sampler-only. If your full pipeline is sampler-bound (true on 24GB+ with `LTXVTiledVAEDecode [1,1,1]`), this is the end-to-end delta. If you're VAE-decode-bound or upstream-bound, the delta drops proportionally.

Record this prior in writing before the first A/B measurement (root CLAUDE.md "record the prior BEFORE the measurement").

## 4. F-pair invariant checklist for refine-pass insertion

Adding internal subgraph nodes (no schema/slot change to `#843`'s 20-input / 1-output surface) does NOT touch the top-level link graph. Implications per audit:

| Audit | Reads subgraph internals? | Affected by adding internal refine sampler nodes? |
|---|---|---|
| `graph_acyclic` | **No** — top-level only by design (`audit_workflows.py::_check_graph_acyclic`, comment: "Top-level only: subgraph internals share the global node ID space with top-level, so merging both into one graph yields false positives") | Safe — internal cycles are framework-managed by tensor-loop |
| `iterations_autowired` (F5) | Reads `AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in` (top-level) | Safe — no top-level wiring change |
| `frame_planner_present` (F8) | Top-level `LTXFramePlanner` presence + wiring | Safe |
| `planner_no_stride_input` (F7) | Top-level edges between Controller/Planner | Safe |
| `alc_seed_legacy_name` (F4) | Top-level `AudioLoopController` schema | Safe |
| `alc_widget_drift` (F6) | Top-level `AudioLoopController` widget shape | Safe |
| F2 `loop_guide_preprocess_symmetry` | Top-level (initial + loop `LTXVPreprocess(img_compression=18)` parity) | Safe — refine pass is post-sampler, doesn't touch preprocess |
| F3 `loop_cropguides_symmetry` | **Yes** — reads `sg["links"]` for `target_id=644` pos/neg origins. **Hardcoded to node IDs `{644, 655, 1519}`** | **Audit gap**: if the refine pass adds its own CFGGuider node (not reusing `#644`), the F3 audit will NOT validate that the new guider has a CropGuides chain on its inputs. Mitigation: reuse `#644` for the refine guider (recommended), OR file a follow-up to extend `loop_cropguides_symmetry` to enumerate all CFGGuider nodes in `sg["nodes"]` (not just id 644) and assert each has CropGuides on its CONDITIONING inputs |
| `cond_metadata_types` | Validates `frame_rate` stamp on loop-body CONDITIONING-producing nodes (root CLAUDE.md: "Loop-body CONDITIONING must carry frame_rate") | **Watch**: if the refine pass introduces any new CONDITIONING-producing node inside the subgraph (it shouldn't if you reuse `#644`), that node must stamp `frame_rate` via `node_helpers.conditioning_set_values`. Reusing `#644` keeps the existing stamping intact |
| `link_integrity` | **Yes** — subgraph link bidirectional consistency | Safe if using `WorkflowEditor.add_subgraph_link` / `rewire_subgraph_input` (canonical edit path enforces this) |
| `widget_shape` | Top-level widget value counts | Safe |

**Other root-CLAUDE.md rules to honor**:

- **Don't ship two schema changes touching the same iteration-state plane in one session.** The refine-pass insertion doesn't change any node's schema (no new ALC outputs, no new Planner outputs); it just adds an internal sampler. Single-plane: safe.
- **`AudioLoopController` outputs are iteration-state-dependent**; this is fine because the refine pass lives entirely inside the subgraph, downstream of the inputs that already carry iter state.
- **`KSamplerSelect "euler_cfg_pp"`** is the named sampler — verify it's a registered string in the running ComfyUI's sampler list. The KSamplerSelect widget options are populated from `comfy.samplers.KSampler.SAMPLERS`; `euler_cfg_pp` is in core comfy. (No new node-helper node.)
- **WorkflowEditor**: edits MUST go through `WorkflowEditor.add_subgraph_link` etc. — three link representations (top-level array, node body links, subgraph dict links + linkIds) stay in sync only via the editor.

## Related references

- `nodes.py::AudioLoopController` — output slot semantics (slot indices 0-7)
- `nodes.py::LatentContextExtract` / `nodes.py::LatentOverlapTrim` — auto-strips `noise_mask`
- `nodes.py::LoopIterationStamp` — `transformer_options["iteration"]` stamp
- `nodes.py::IterationCleanup` — allocator hygiene per iter
- `nodes_audio_latent_slice.py::AudioLatentSlice` — per-iter slice of pre-encoded audio
- `scripts/audit_workflows.py::_check_graph_acyclic` — top-level only
- `scripts/audit_workflows.py` F3 block (`loop_cropguides_symmetry`) — hardcoded to ids `{644, 655, 1519}`
- `docs/reference/audio_loop_controller.md` — controller iter-state propagation
- `docs/reference/pipeline_flow_latent.md` — F2/F3 invariant trace
- `docs/reference/sampler_reference.md` — 8-step distilled sigma chain rationale
