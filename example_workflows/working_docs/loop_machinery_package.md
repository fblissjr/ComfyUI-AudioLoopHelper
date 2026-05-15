Last updated: 2026-05-15

# Loop machinery package — what to graft from canonical onto variant B

Authoritative inventory of the **top-level audio-loop infrastructure** in `example_workflows/audio-loop-music-video_latent.json` (canonical) and the loop-body **subgraph definition object**. This is the COMPLEMENT to `example_workflows/working_docs/subgraph_chain_reference.md` (which documents subgraph internals); read that first.

Source: `example_workflows/audio-loop-music-video_latent.json` — 99 top-level nodes, 139 top-level links, 1 subgraph (`b4973d68-09b9-4da5-9845-38ad62ae9aca` "extension", 18 internal nodes, 52 internal links, 20 inputs, 1 output).

> **Note (2026-05-15)**: Widget snapshots below reflect the live JSON, which still uses `fps=25` and `frames=497`. Canonical fps for LTX 2.3 is **24** (training distribution; `comfy/ldm/lightricks/av_model.py:866`). The widget tables update when the workflow-JSON sweep lands. Audio-VAE `25` (audio latents per second) is independent of video fps and stays.

Target (B): `example_workflows/benchmark_workflows/fml2v_var_b_first_keyframe_only.json` — 156 top-level nodes, 126 top-level links, 2 subgraphs. Note: B already has Set/Get variables for `model`, `model_with_lora`, `model_nag`, `clip`, `vae`, `vae_audio`, `vae_tiny`, `upscale_model`, `firstframe`, `width`, `height`, `fps`, `frames`, `final_video`, `final_audio`. See §1.9 for the collision/rename map.

## §1 Top-level loop infrastructure (categorized)

### 1.1 Controller + Planner + tensor loop + subgraph invoker

The control plane. Every audio-loop workflow needs all of these.

| Node | id | type | mode | widgets `[value, ...]` | Role |
|---|---|---|---|---|---|
| AudioLoopController | 1582 | `AudioLoopController` | 0 | `[1, 19.88, 2, 344096342624943, 25]` (control_after_generate?, window_seconds default, num_loops default, base_seed, fps_default — actual values driven by inputs) | Stride/seed/overlap engine. **All 8 outputs are iter-dependent in the executor DAG** (root CLAUDE.md). |
| AudioLoopPlanner | 1560 | `AudioLoopPlanner` | 0 | `[19.88, 2, 25, 0, ""]` (window_seconds, overlap_seconds, fps, ?, summary) | Reports `total_iterations`, `stride_seconds`, `audio_duration`, `summary`. Used OUTSIDE the loop. |
| LTXFramePlanner | 1634 | `LTXFramePlanner` | 0 | `[832, 448, 20, 25]` (width, height, target_seconds, fps) | **Single source of truth for dimension config** (root CLAUDE.md). Outputs: `width`, `height`, `frames`, `actual_seconds`, `fps_int`, `fps_float`, `latent_volume`, `status`, `summary`. |
| TensorLoopOpen | 1539 | `TensorLoopOpen` | 0 | `["iterations", 50, 0]` (mode, max_iterations safety cap, initial_count) | Loop head. Outputs `flow_control`, `previous_value` (LATENT), `accumulated_count`, `current_iteration`. |
| TensorLoopClose | 1540 | `TensorLoopClose` | 0 | `[True, "disabled"]` (early_exit, ?) | Loop tail. |
| Subgraph invoker | 843 | `b4973d68-09b9-4da5-9845-38ad62ae9aca` | 0 | `[]` | The loop-body. 20 input slots / 1 output (LATENT). Full IO map: `subgraph_chain_reference.md §1`. |
| LoopIterationStamp | 1618 | `LoopIterationStamp` | 0 | `[0]` | Per-iter `transformer_options["iteration"]` stamp on the MODEL. **Outside the subgraph** — sits between `LTXVReferenceAudio #1633 → #843.model`. Its `current_iteration` input is sourced from `TensorLoopOpen.current_iteration` (closes the iter-state plane for the model edge). |
| AudioDuration | — | — | — | — | **NOT PRESENT in canonical.** `audio_duration` is sourced directly from `AudioLoopController.audio_duration` (out[2]) and `AudioLoopPlanner.audio_duration` (out[3]) without an `AudioDuration` helper node. |

**Controller inputs** (`#1582`):
- `audio` ← `#567 TrimAudioDuration.AUDIO` (the song-trim node)
- `current_iteration` ← `#1539 TensorLoopOpen.current_iteration`
- `window_seconds` ← `#1634 LTXFramePlanner.actual_seconds`
- `base_seed` ← `#1529 GetNode("start_seed").INT`
- `fps` ← `#1634 LTXFramePlanner.fps_int`
- `overlap_seconds` ← `#2013 FloatConstant("overlap_seconds")=2.0`

**Planner inputs** (`#1560`):
- `audio` ← `#567 TrimAudioDuration.AUDIO`
- `window_seconds` ← `#1634 LTXFramePlanner.actual_seconds`
- `fps` ← `#1634 LTXFramePlanner.fps_int`
- `overlap_seconds` ← `#2013 FloatConstant("overlap_seconds")` (SAME instance as controller — F-pair convention: `apply_overlap_seconds_single_source.py`)

**TensorLoopOpen inputs**: `initial_value` ← `#245 LTXVSeparateAVLatent.video_latent`, `iterations_in` ← `#1560 AudioLoopPlanner.total_iterations` (F5).

**TensorLoopClose inputs**: `flow_control` ← `#1539`, `processed` ← `#843` (subgraph output), `stop` ← `#1582 AudioLoopController.should_stop`.

### 1.2 Prompt encoder schedule

**Two `ConditioningSelectByIteration` instances** — one inside subgraph at slot 6 (already covered by `subgraph_chain_reference.md §1`), and one outside at top level for the initial render.

| Node | id | type | mode | widgets | Role |
|---|---|---|---|---|---|
| TimestampPromptScheduleBatchEncode | 1615 | `TimestampPromptScheduleBatchEncode` | 0 | `["0:00+: video of a man dancing", 17.92, 180, True, 25]` (schedule_text, stride_seconds_default, max_tokens, ?, fps) | Pre-encodes whole prompt schedule outside loop. **CLIP must not enter the loop body** (root CLAUDE.md). Inputs: `clip` ← `#416 DualCLIPLoader.CLIP`, `stride_seconds` ← `#1560 AudioLoopPlanner.stride_seconds`, `audio_duration` ← `#1560 AudioLoopPlanner.audio_duration`. Output: `conditioning_list`. |
| ConditioningSelectByIteration (initial) | 2021 | `ConditioningSelectByIteration` | 0 | `[0]` (fallback iteration index) | Title: `"Initial render conditioning (from schedule[0])"`. Picks schedule entry 0 for the initial render's `Node 169` rule (root CLAUDE.md: "Node 169 prompt matches schedule 0:00 entry"). Inputs: `conditioning_list` ← `#1615.conditioning_list`, `current_iteration` is **unwired** (defaults to widget value `0`). Output → `#164 LTXVConditioning.positive` AND `#420 ConditioningZeroOut.conditioning`. |
| ConditioningSelectByIteration (loop) | 1616 | `ConditioningSelectByIteration` | 0 | `[0]` | Same node type, INSIDE-loop wiring. Inputs: `conditioning_list` ← `#1615.conditioning_list`, `current_iteration` ← `#1539 TensorLoopOpen.current_iteration`. Output → `#1633 LTXVReferenceAudio.positive` (bypassed; passes through to subgraph slot 6). |

**Both reference the SAME `#1615.conditioning_list` output** — one CLIP encode pass shared by both the initial render and the loop. This is the F-pair `batch_encode` invariant.

### 1.3 Audio infrastructure (top-level)

Audio is VAE-encoded ONCE outside the loop, then sliced per iter inside via `AudioLatentSlice` (subgraph node `#2012`).

| Node | id | type | mode | widgets | Role |
|---|---|---|---|---|---|
| LoadAudio | 565 | `LoadAudio` | 0 | `["example_audio.mp3", None, None]` | Source. |
| TrimAudioDuration (song trim) | 567 | `TrimAudioDuration` | 0 | `[0, 600]` (start_index, duration) | Title: `"Song Trim (full song by default — set start_index > 0 to skip intro)"`. **Audit-paired**: `apply_fix_source_audio_trim_defaults.py` set safe `[0, 600]` after `[5, 300]` ate the first 5 seconds of every song silently (root CLAUDE.md "Widget defaults that DROP user content must be opt-in"). Output feeds: `#1582.audio`, `#1560.audio`, `#2009.audio` (full-song VAE encode), `#566.audio` (initial-render VAE encode, indirectly via `#601`). |
| TrimAudioDuration (initial-render audio trim) | 601 | `TrimAudioDuration` | 0 | `[0, 10]` (start_index, duration default — overridden by input) | Title: `"Initial-Render Audio Trim (10s context)"`. `duration` input is wired to `#1634 LTXFramePlanner.actual_seconds`. The "10s" in title refers to the legacy widget default; effective value comes from FramePlanner. |
| LTXVAudioVAEEncode (full-song) | 2009 | `LTXVAudioVAEEncode` | 0 | `[]` | Title: `"Full-song Audio VAE Encode (pre-encode pattern)"`. Inputs: `audio` ← `#567.AUDIO`, `audio_vae` ← `#254 GetNode("audio_vae").VAE`. Output → `#2010 SetNode("full_audio_latent")`. |
| LTXVAudioVAEEncode (initial-render) | 566 | `LTXVAudioVAEEncode` | 0 | `[]` | Initial-render audio encode. Inputs: `audio` ← `#601.AUDIO`, `audio_vae` ← `#254`. Output → `#570 SetLatentNoiseMask.samples`. **Separate instance from #2009** — initial render needs 10s of audio latent for the first window; loop reads pre-encoded full-song latent. |
| SetNode "full_audio_latent" | 2010 | `SetNode` | 0 | `["full_audio_latent"]` | Captures the pre-encoded full-song audio latent. |
| GetNode "full_audio_latent" | 2011 | `GetNode` | 0 | `["full_audio_latent"]` | Feeds subgraph slot 18 via top-level link 3162. |
| MelBandRoFormerModelLoader | 568 | bypassed | 4 | `["MelBandRoformer_fp32.safetensors"]` | Vocal-separation path (bypassed by default per `apply_melband_default_off.py`). |
| MelBandRoFormerSampler | 569 | bypassed | 4 | `[]` | Companion to #568. |

**Audio Set/Get vars**: `actual_audio` (used by subgraph slot 10), `orig_audio` (used by `TrimVideoLatentToAudio` + `TrimImageBatchToAudio` + `VHS_VideoCombine.audio`). Both source from `#565 LoadAudio` indirectly (`#640 SetNode("actual_audio")` ← `#567`; `#581 SetNode("orig_audio")` ← `#565`).

### 1.4 Initial render assembly (outside loop)

The first render before the loop iterates. Produces the LATENT that becomes (a) `TensorLoopOpen.initial_value` and (b) the `LatentConcat` prepend at the end.

| Node | id | type | mode | widgets | Role |
|---|---|---|---|---|---|
| LoadImage | 444 | `LoadImage` | 0 | `["reference_image.png", "image"]` | Init image source. |
| LTXSmartImageResize | 445 | `LTXSmartImageResize` | 0 | `[832, 448, True, "top"]` | Multi-stage lanczos resize (root CLAUDE.md: avoids the >2× linear alias). Inputs: `width` / `height` ← `#1634 LTXFramePlanner`. |
| LTXVPreprocess (init) | 446 | `LTXVPreprocess` | 0 | `[18]` (img_compression) | **F2 anchor — init side.** Both initial-render and loop ref-video branches must share `LTXVPreprocess(img_compression=18)`. |
| EmptyLTXVLatentVideo | 344 | `EmptyLTXVLatentVideo` | 0 | `[832, 448, 497, 1]` (width, height, length, batch — all overridden by inputs) | Inputs: `width`, `height` ← `#1634`; `length` ← `#1634.frames`. |
| LTXVImgToVideoInplaceKJ | 531 | `LTXVImgToVideoInplaceKJ` | 0 | `["1", 1, 0]` (image_indices, strength, ?) | Writes encoded init into frame 0. Inputs: `vae` ← `#413 GetNode("video_vae")`, `latent` ← `#344`, `num_images.image_1` ← `#446 LTXVPreprocess`. |
| SolidMask | 571 | `SolidMask` | 0 | `[0, 512, 512]` (value, width, height) | mask=0 source for audio frame. |
| SetLatentNoiseMask | 570 | `SetLatentNoiseMask` | 0 | `[]` | Inputs: `samples` ← `#566 LTXVAudioVAEEncode`, `mask` ← `#571`. **Audio frames mask=0 → frozen** (root CLAUDE.md). |
| LTXVConcatAVLatent (initial) | 350 | `LTXVConcatAVLatent` | 0 | `[]` | Inputs: `video_latent` ← `#531`, `audio_latent` ← `#570`. **Audio path is sacred** (root CLAUDE.md). |
| RandomNoise (initial) | 1322 | `RandomNoise` | 0 | `[0, "fixed"]` (seed_default, control) | Inputs: `noise_seed` ← `#1530 GetNode("start_seed").INT`. |
| ManualSigmas | 1421 | `ManualSigmas` | 0 | `["1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"]` | **Canonical 8-step distilled sigmas.** No `ModelSamplingSD3` shift, no `euler_ancestral*` (root CLAUDE.md). Set into `Set_sigmas` (#579), used by both initial sampler AND subgraph slot 1 via `Get_sigmas` (#580). |
| KSamplerSelect | 154 | `KSamplerSelect` | 0 | `["euler"]` | Shared by initial sampler AND subgraph slot 0. |
| LTX2SamplingPreviewOverride | 503 | `LTX2SamplingPreviewOverride` | 0 | `[8]` | Wraps the MODEL to override sampling-preview frequency. Sits between `#508 LTX2_NAG` → `#2014 LoraLoaderModelOnly` in the model chain. |
| LTXVConditioning | 164 | `LTXVConditioning` | 0 | `[25]` (frame_rate) | Stamps `frame_rate` on positive/negative CONDITIONING. Inputs: `positive` ← `#2021 ConditioningSelectByIteration`, `negative` ← `#420 ConditioningZeroOut`, `frame_rate` ← `#1634.fps_float`. |
| ConditioningZeroOut | 420 | `ConditioningZeroOut` | 0 | `[]` | Inputs: `conditioning` ← `#2021`. Output → `#164.negative`. Runtime-inert at CFG=1 but `CFGGuider` validates both slots (root CLAUDE.md). |
| LTXVReferenceAudio (initial; bypassed) | 1632 | `LTXVReferenceAudio` | 4 | `[3, 0, 1]` | Title: `"LTXV Reference Audio (ID-LoRA initial render)"`. Bypassed → passes `model`, `positive`, `negative` through. Inputs: `model` ← `#572 SetNode("model")`, `positive` ← `#164.positive`, `negative` ← `#164.negative`. |
| CFGGuider (initial) | 153 | `CFGGuider` | 0 | `[1]` (cfg) | Inputs: `model` ← `#1632`, `positive` ← `#1632`, `negative` ← `#1632`. **NOTE**: positive/negative do NOT flow through `LTXVCropGuides` on the init side — they flow directly from `#164 LTXVConditioning` through bypassed `#1632`. F3 is enforced INSIDE the subgraph (`#655` for loop CFGGuider), not on the initial CFGGuider. The "init-side F3 symmetry" the spec mentions is **structurally not present** in canonical for CONDITIONING; only the LATENT half of CropGuides (`#381`) runs on the init side. |
| SamplerCustomAdvanced (initial) | 161 | `SamplerCustomAdvanced` | 0 | `[]` | Inputs: `noise` ← `#1322`, `guider` ← `#153`, `sampler` ← `#154`, `sigmas` ← `#579 SetNode("sigmas")` via link 2628, `latent_image` ← `#350 LTXVConcatAVLatent`. |
| LTXVSeparateAVLatent | 245 | `LTXVSeparateAVLatent` | 0 | `[]` | Separates initial render output. Inputs: `av_latent` ← `#161`. Outputs: `video_latent` → `#381 LTXVCropGuides.latent` AND `#1539 TensorLoopOpen.initial_value` AND `#843 reference_latent` (subgraph slot 15). |
| LTXVCropGuides (initial) | 381 | `LTXVCropGuides` | 0 | `[]` | Strips guide metadata from initial-render LATENT before concat. Inputs: `positive` ← `#164.positive`, `negative` ← `#164.negative`, `latent` ← `#245.video_latent`. Output `latent` → `#1605 LatentConcat.samples1`. (pos/neg outputs unused.) |

### 1.5 F2 init-image preprocess (the symmetric anchor)

Per root CLAUDE.md F2: **both initial render and ref-video branches share `LTXVPreprocess(img_compression=18)`**.

- `#446 LTXVPreprocess` (initial side, widgets `[18]`) — feeds `#531 LTXVImgToVideoInplaceKJ` AND `#1617 VAEEncode` (per-iter guide latent).
- `#1638 LTXVPreprocess` (ref-video side, widgets `[18]`, title `"Preprocess ref-video (F2 symmetric)"`) — feeds subgraph slot 17 (`reference_video`).

**Audit reference**: `audit_workflows.py` F2 check at ~line 1221+ (`loop_guide_preprocess_symmetry` for ref-video). The legacy F2 check for `Set_input_image #650` is dead code in canonical (no `#650` node).

### 1.6 F3 cropguides chain (initial vs loop)

- **Initial side**: `#381 LTXVCropGuides` strips LATENT only. Its `positive`/`negative` outputs are unused. The initial `CFGGuider #153` sources positive/negative directly from `#164 LTXVConditioning` via bypassed `#1632`.
- **Loop side (inside subgraph)**: `#655 LTXVCropGuidesNoLatent` (CONDITIONING half) + `#2008 LTXVCropGuides` (LATENT half) — pair introduced by `apply_split_cropguides.py` to break the loop cycle. Audit at `audit_workflows.py:529-582` checks `#644 CFGGuider.positive/negative ← #655`. **Hardcoded to subgraph node IDs `{644, 655, 1519}`** (see §4 for build-script implication).

The asymmetry between init and loop is intentional: the init render runs once before the loop closes its cycle, so the init's CFGGuider doesn't need the split.

### 1.7 Latent post-loop assembly

The output chain from `TensorLoopClose` → final mp4.

| Node | id | type | mode | widgets | Role |
|---|---|---|---|---|---|
| LatentConcat (prepend initial) | 1605 | `LatentConcat` | 0 | `["t"]` (dim) | Inputs: `samples1` ← `#381 LTXVCropGuides.latent` (initial render's stripped LATENT), `samples2` ← `#1540 TensorLoopClose.output` (loop accumulator). |
| TrimVideoLatentToAudio | 2028 | `TrimVideoLatentToAudio` | 0 | `[25]` (fps_default) | **F14 latent half.** Inputs: `latent` ← `#1605`, `audio` ← `#604 GetNode("orig_audio")`, `fps` ← `#1634.fps_int`. |
| LTXVTiledVAEDecode | 1604 | `LTXVTiledVAEDecode` | 0 | `[1, 1, 1, True, "auto", "auto"]` | Title: `"Final VAE Decode (once)"`. **Single-tile (24GB+)** per root CLAUDE.md. Inputs: `vae` ← `#1598 GetNode("video_vae")`, `latents` ← `#2028`. |
| TrimImageBatchToAudio | 2029 | `TrimImageBatchToAudio` | 0 | `[25]` (fps_default) | **F14 image half.** Eliminates silence-at-end from iter overshoot. Inputs: `images` ← `#1604`, `audio` ← `#604`, `fps` ← `#1634.fps_int`. |
| RunIdPrefix | 2026 | `RunIdPrefix` | 0 | `["audio-loop-music-video_latent", "%Y%m%d_%H%M%S"]` (base_name, timestamp_fmt) | **F15.** Outputs: `video_prefix` → `#617.filename_prefix`; `latent_prefix` → `#2027 SaveLatent.filename_prefix`. |
| SaveLatent | 2027 | `SaveLatent` | 4 (bypassed) | `["latents/segment"]` | Bypassed-by-default toggle. User flips to `mode=0` to capture the assembled latent for the latent-load upscale path. Inputs: `samples` ← `#1605 LatentConcat`. |
| VHS_VideoCombine | 617 | `VHS_VideoCombine` | 0 | `{frame_rate: 25, loop_count: 0, filename_prefix: "LTX-2", format: "video/h264-mp4", pix_fmt: "yuv420p", crf: 19, save_metadata: True, trim_to_audio: True, pingpong: False, save_output: True, videopreview: {...}}` | Final encoder. Inputs: `images` ← `#2029`, `audio` ← `#604 GetNode("orig_audio")`, `filename_prefix` ← `#2026.video_prefix`. |

### 1.8 Sage attention + model chain

The model passes through this chain BEFORE entering the loop:

```
#414 UNETLoader
  → #268 AudioLoopHelperSageAttention [auto, True, 1024]
  → #504 LTXVChunkFeedForward [2, 4096]
  → #1523 LTX2AttentionTunerPatch ["", 1, 1, 1, 1, True]
  → #508 LTX2_NAG [11, 0.25, 2.5, True]  (nag_cond_video ← #507 CLIPTextEncode)
  → #503 LTX2SamplingPreviewOverride [8]
  → #2014 LoraLoaderModelOnly (mode 4, "Distill LoRA")
  → #2015 LoraLoaderModelOnly (mode 4, "Style or ID LoRA")
  → #1635 LTXICLoRALoaderModelOnly (mode 4, "IC-LoRA Loader (video reference)")
  → #572 SetNode "model"
```

Then GET'd by `#654 GetNode("model")` → `#1633 LTXVReferenceAudio (bypassed)` → `#1618 LoopIterationStamp` → subgraph slot 2.

**Canonical order constraint** (root CLAUDE.md gotcha): "`UNETLoader → ... → LTXICLoRALoaderModelOnly → <module-mutating node> → SetNode "model"`" — IC-LoRA loader must come BEFORE `state_dict()`-reading patches.

**Sage attention widget shape**: `[mode, attention_compile, skip_under_seq_len]` = `["auto_mask_aware", True, 1024]`. The `1024` (skip-under-seq-len) is set by `apply_skip_under_seq_len.py`.

### 1.9 Named-variable (Set/Get) map

**Canonical Set vars (8)**: `audio_vae`, `sigmas`, `start_seed`, `model`, `video_vae`, `actual_audio`, `orig_audio`, `full_audio_latent`.

**Canonical Get vars (used)**: `audio_vae` (3 instances), `video_vae` (3), `model` (1), `sigmas` (1), `start_seed` (2), `actual_audio` (1), `orig_audio` (1), `full_audio_latent` (1).

**B's existing Set vars (collision/compatibility check)**:

| Variable | Canonical uses | B uses | Action for graft |
|---|---|---|---|
| `model` | yes (post-mutation MODEL) | yes (B's own model chain) | **COLLISION** — B's `Set_model` writes a different model. Rename canonical's to `model_loop` OR rebuild model chain on the B side and reuse B's `Set_model`. |
| `audio_vae` / `vae_audio` | `audio_vae` | `vae_audio` (B uses underscore-reversed name) | **NAMING DRIFT** — pick one. Canonical's GetNode `#254` reads `"audio_vae"`; B's `#172` writes `"vae_audio"`. Either rename canonical's reads to `vae_audio` or rewrite B's writes to `audio_vae`. |
| `video_vae` / `vae` | `video_vae` | `vae` | **NAMING DRIFT** — same issue. |
| `vae_tiny` | not used | yes | safe; canonical doesn't read this |
| `clip` | not used (canonical wires `#416 DualCLIPLoader.CLIP` directly to `#1615`) | yes | safe |
| `start_seed` | yes (INT, seed source) | not used | **ADD** to B (canonical's `#1527 INTConstant "start_seed" widgets=[42]` + `#1528 SetNode`). |
| `sigmas` | yes (SIGMAS) | not used (B uses `LTXVScheduler` inline) | **ADD** sigmas Set/Get if reusing canonical's pattern, OR rewire subgraph slot 1 to B's sigma source. |
| `actual_audio` | yes (AUDIO, feeds subgraph slot 10) | not used | **ADD**. |
| `orig_audio` | yes (AUDIO, feeds post-loop trims + VHS_VideoCombine.audio) | not used | **ADD**. |
| `full_audio_latent` | yes (LATENT, feeds subgraph slot 18) | not used | **ADD**. |
| `firstframe`, `width`, `height`, `fps`, `frames`, `final_video`, `final_audio`, `upscale_model`, `model_with_lora`, `model_nag`, `firstframe_strength`, `lastframe_strength`, `middleframe`, `lastframe`, `positive_guider`, `negative_guider`, `positive_guider2`, `negative_guider2`, `latent_audio`, `firstframe_resized`, `middleframe_resized`, `lastframe_resized`, `middleframe_count`, `middleframe_strength`, `lastframe_strength`, `negative` | not used | yes | B's own infrastructure — leave alone. |

**Decision needed before grafting**: align Set/Get naming. Recommended: keep B's names (`vae` for video, `vae_audio` for audio) and rewrite canonical's reads. This avoids touching B's existing wiring.

### 1.10 LTX2_NAG

Single instance: `#508 LTX2_NAG`, widgets `[11, 0.25, 2.5, True]` = `[nag_scale, nag_alpha, nag_tau, inplace]`. **`nag_scale=11` is aggressive for distilled** (root CLAUDE.md: "dial to 3-7 if initial render freezes"). Sits between `#1523 LTX2AttentionTunerPatch` and `#503 LTX2SamplingPreviewOverride`. `nag_cond_video` ← `#507 CLIPTextEncode` (negative prompt: `"still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone, microphone"`). `nag_cond_audio` is unwired.

### 1.11 LTX2SamplingPreviewOverride

`#503`, widgets `[8]` (preview_period_steps). Wraps MODEL to override the per-step sampling-preview frequency. Position: between `LTX2_NAG` and the LoRA loaders. Outputs MODEL → `#2014 LoraLoaderModelOnly`.

## §2 The subgraph definition object

The subgraph is a single dict at `wf["definitions"]["subgraphs"][0]`. Top-level keys: `id`, `version`, `state`, `revision`, `config`, `name`, `inputNode`, `outputNode`, `inputs`, `outputs`, `widgets`, `nodes`, `groups`, `links`, `extra`.

- **id**: `"b4973d68-09b9-4da5-9845-38ad62ae9aca"` (UUID — the invoker `#843`'s `type` matches this).
- **name**: `"extension"`.
- **inputs[]**: 20 slots. Full schema in `subgraph_chain_reference.md §1` (slot/name/type/source/consumer/iter-state table). Slot order is load-bearing — ComfyUI's loader matches by `inputs[].name` but the slot-distributor (virtual id `-10`) uses index.
- **outputs[]**: 1 slot — `{name: "extended_latent", type: "LATENT"}`. Backed by `#2007 IterationCleanup.latent` via internal link 3042.
- **nodes[]**: 18 entries. Types covered in `subgraph_chain_reference.md §2`. Includes `#573 SamplerCustomAdvanced`, `#574 RandomNoise`, `#583 LTXVConcatAVLatent`, `#596 LTXVSeparateAVLatent`, `#598 LTXVAudioVAEEncode` (bypassed), `#600 TrimAudioDuration` (bypassed), `#606 LTXVAudioVideoMask`, `#644 CFGGuider`, `#655 LTXVCropGuidesNoLatent`, `#1519 LTXVAddLatentGuide`, `#1639 GetImageRangeFromBatch`, `#1640 LTXAddVideoICLoRAGuide` (bypassed), `#2004 LatentContextExtract`, `#2005 LatentOverlapTrim`, `#2006 LTXVAdainLatent`, `#2007 IterationCleanup`, `#2008 LTXVCropGuides`, `#2012 AudioLatentSlice`.
- **links[]**: 52 entries. Internal link shape is `{id, origin_id, origin_slot, target_id, target_slot, type}` (dict, NOT array — top-level links are arrays). Distributor virtual id `-10`, output collector `-20`.

### Does `WorkflowEditor.add_subgraph` exist?

**No.** `scripts/workflow_utils.py` has:
- `get_subgraph(index)` (read)
- `find_subgraph_invoker(sg_index)` / `find_subgraph_node` / `find_subgraph_link` / `find_subgraph_link_to_slot` (read)
- `add_subgraph_link` / `remove_subgraph_link` / `add_subgraph_node` / `rewire_subgraph_input` (mutate WITHIN an existing subgraph)

**There is no helper to clone or insert an entire subgraph definition object.** The build script will need to:
1. Read canonical's `wf["definitions"]["subgraphs"][0]` as an opaque dict.
2. Splice it into the new workflow as `new_wf["definitions"]["subgraphs"][0]` (creating `definitions.subgraphs` if absent; B already has 2 subgraphs, so the new entry would be `subgraphs[2]` or replace the array).
3. Add the invoker top-level node `#843` (or a fresh id) with `type` = the UUID, and 20 input slots + 1 output slot matching the subgraph's IO.
4. Wire all 20 top-level inputs (§3) and the 1 output (LATENT → `TensorLoopClose.processed`).

**Note**: B already has 2 subgraphs (`subgraphs[0]` and `subgraphs[1]` — likely from B's first-keyframe / guider helpers). The graft adds a third. ComfyUI matches invoker nodes to subgraph definitions by `type` (UUID), so the order in the `subgraphs[]` array doesn't matter for runtime — but be careful if any build code assumes `subgraphs[0]` is THE loop.

**Recommended approach**: write a `_splice_subgraph_definition(target_wf, source_wf, src_index=0)` helper in the build script. Deep-copy the source's `subgraphs[src_index]` dict, append to `target_wf["definitions"]["subgraphs"]`. Internal link IDs and node IDs in the subgraph share the global ID space with top-level nodes (root CLAUDE.md gotcha: "subgraph internals share the global node ID space with top-level"), so you must renumber if any IDs collide with B's top-level. Check B for collisions on IDs `{573, 574, 583, 596, 598, 600, 606, 644, 655, 1519, 1639, 1640, 2004, 2005, 2006, 2007, 2008, 2012}` and internal link IDs `1573-3162` (~52 ids in that range).

## §3 Boundary contract — wires the build script must create

The 20 subgraph input slots and 1 output (full IO table: `subgraph_chain_reference.md §1`). Top-level edges THE BUILD SCRIPT MUST CREATE:

### Inputs into the subgraph invoker `#843`

| Slot | Name | Source (top-level) |
|---|---|---|
| 0 | sampler | `#154 KSamplerSelect.SAMPLER` (also feeds `#161` initial sampler) |
| 1 | sigmas | `#580 GetNode("sigmas").SIGMAS` (sourced from `#579 SetNode("sigmas") ← #1421 ManualSigmas`) |
| 2 | model | `#1618 LoopIterationStamp.model` (the MODEL passes through `LoopIterationStamp` immediately before entering the subgraph) |
| 3 | vae | `#619 GetNode("video_vae").VAE` (or B's `vae` GetNode after naming alignment) |
| 4 | previous_latent | `#1539 TensorLoopOpen.previous_value` |
| 5 | video_end_time | `#1634 LTXFramePlanner.actual_seconds` |
| 6 | positive | `#1633 LTXVReferenceAudio.positive` (bypassed → passes through `#1616 ConditioningSelectByIteration`) |
| 7 | negative | `#1633 LTXVReferenceAudio.negative` (bypassed → passes through `#164 LTXVConditioning.negative`) |
| 8 | guide_latent | `#1617 VAEEncode("init image → guide latent").LATENT` |
| 9 | audio_vae | `#599 GetNode("audio_vae").VAE` |
| 10 | audio | `#641 GetNode("actual_audio").AUDIO` |
| 11 | start_index | `#1582 AudioLoopController.start_index` |
| 12 | num_guides.strength_1 | `#1269 FloatConstant("first_frame_guide_strength")=1.0` |
| 13 | noise_seed | `#1582 AudioLoopController.iteration_seed` |
| 14 | num_frames | `#1582 AudioLoopController.overlap_latent_frames` (out[6] — **the spec calls this slot 14 / "num_frames"; not slot 12**) |
| 15 | reference_latent | `#245 LTXVSeparateAVLatent.video_latent` (initial render's video LATENT) |
| 16 | video_start_time | `#1582 AudioLoopController.overlap_seconds` (out[7]) |
| 17 | reference_video | `#1638 LTXVPreprocess.output_image` (F2-symmetric ref-video) |
| 18 | full_audio_latent | `#2011 GetNode("full_audio_latent").LATENT` |
| 19 | source_seconds | `#1582 AudioLoopController.audio_duration` (out[2]) |

### Outputs from `#843`

| Slot | Name | Consumer |
|---|---|---|
| 0 | extended_latent (LATENT) | `#1540 TensorLoopClose.processed` |

### Non-subgraph boundary wires

The "control plane" wires not covered by the IO table:

- `#1634 LTXFramePlanner.width` → `#344 EmptyLTXVLatentVideo.width`, `#445 LTXSmartImageResize.width`
- `#1634 LTXFramePlanner.height` → `#344.height`, `#445.height`
- `#1634 LTXFramePlanner.frames` → `#344.length`
- `#1634 LTXFramePlanner.actual_seconds` → `#601 TrimAudioDuration("Initial-Render").duration`, `#1582 AudioLoopController.window_seconds`, `#1560 AudioLoopPlanner.window_seconds`, `#843 video_end_time` (covered above)
- `#1634 LTXFramePlanner.fps_int` → `#1582 AudioLoopController.fps`, `#1560 AudioLoopPlanner.fps`, `#2028 TrimVideoLatentToAudio.fps`, `#2029 TrimImageBatchToAudio.fps`
- `#1634 LTXFramePlanner.fps_float` → `#164 LTXVConditioning.frame_rate`
- `#1560 AudioLoopPlanner.total_iterations` → `#1539 TensorLoopOpen.iterations_in` (**F5 invariant**)
- `#1560 AudioLoopPlanner.stride_seconds` → `#1615 TimestampPromptScheduleBatchEncode.stride_seconds`
- `#1560 AudioLoopPlanner.audio_duration` → `#1615.audio_duration`
- `#1582 AudioLoopController.should_stop` → `#1540 TensorLoopClose.stop`
- `#1539 TensorLoopOpen.current_iteration` → `#1582 AudioLoopController.current_iteration`, `#1616 ConditioningSelectByIteration.current_iteration`, `#1618 LoopIterationStamp.current_iteration`
- `#1539 TensorLoopOpen.initial_value` ← `#245 LTXVSeparateAVLatent.video_latent`
- `#1540 TensorLoopClose.output` → `#1605 LatentConcat.samples2`
- `#381 LTXVCropGuides.latent` → `#1605 LatentConcat.samples1` (prepend initial render)
- `#2013 FloatConstant("overlap_seconds")=2.0` → `#1582.overlap_seconds`, `#1560.overlap_seconds` (single-source per `apply_overlap_seconds_single_source.py`)
- `#1615 TimestampPromptScheduleBatchEncode.conditioning_list` → `#2021 (initial)` AND `#1616 (loop)` ConditioningSelectByIteration

### Set/Get var bridges

- `#1530 GetNode("start_seed").INT` → `#1322 RandomNoise.noise_seed` (initial), `#1582 AudioLoopController.base_seed`. Source: `#1527 INTConstant=42` → `#1528 SetNode("start_seed")`.
- `#640 SetNode("actual_audio") ← #567 TrimAudioDuration` → `#641 GetNode("actual_audio")` → subgraph slot 10.
- `#581 SetNode("orig_audio") ← #567 TrimAudioDuration.AUDIO` → `#582/#604 GetNode("orig_audio")` → `#2028/#2029.audio` and `#617.audio`. Both `actual_audio` (#640) and `orig_audio` (#581) source from the SAME `#567` output — they're two named handles on the trimmed song. `#1582 AudioLoopController.audio` is wired directly to `#567` (not via a GetNode).

## §4 Audit invariants this build touches

Per `scripts/CLAUDE.md` F-pair convention. The build must pass `scripts/audit_workflows.py` (exit 0 = no ERR).

| Audit ID | What it checks | Build action |
|---|---|---|
| `graph_acyclic` | Top-level DAG only (subgraph internals not merged). | Verify no cycles between Controller / Planner / FramePlanner / TensorLoop / subgraph invoker. |
| `link_integrity` | Bidirectional link consistency. | Use `WorkflowEditor.add_link` / `add_subgraph_link` exclusively. |
| `widget_shape` | Widget value counts match schema. | Don't hand-construct widget arrays. |
| F2 `loop_guide_preprocess_symmetry` (ref-video at line ~1221) | `LTXVPreprocess(img_compression=18)` present for ref-video branch. | Include `#1638` (or its rebuilt equivalent). |
| F2 `preprocess_symmetry` (line 508) | Hardcoded to `#650 Set_input_image` — dead code in canonical (no `#650`). | Safe; no `#650` to break. |
| F3 `loop_cropguides_symmetry` (line 538) | **Hardcoded to subgraph node IDs `{644, 655, 1519}`**. Checks `sg_links` for `target_id=644 target_slot=1/2` with `origin_id=655`. | **CRITICAL — see below.** |
| F4 `alc_seed_legacy_name` | `AudioLoopController` schema uses `base_seed`, not `seed`. | Use canonical's controller as-is. |
| F5 `iterations_autowired` | `AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in`. | Wire it (covered in §3). |
| F6 `alc_widget_drift` | `AudioLoopController` widgets shape correct (no leftover `randomize`). | Use canonical's controller widgets unchanged. |
| F7 `planner_no_stride_input` | `AudioLoopPlanner` does NOT receive `stride_seconds` input (would close a cycle). | Don't wire any edge into `#1560.stride_seconds` (the slot is output-only). |
| F8 `frame_planner_present` | `LTXFramePlanner` exists and its outputs feed `EmptyLTXVLatentVideo` etc. | Include `#1634` and wire per §3. |
| F12 `iclora_video_reference` | IC-LoRA video-ref symmetry (subgraph node `#1640` + loader `#1635`). | Bypassed-by-default chain; include for shape conformance even if disabled. |
| F14 `trim_video_latent_to_audio` / `trim_image_batch_to_audio` | `#2028` / `#2029` spliced into post-loop chain. | Include both. |
| F15 `run_id_layout` | `RunIdPrefix` wired to `VHS_VideoCombine.filename_prefix`. | Include `#2026`. |
| `cond_metadata_types` | Loop-body CONDITIONING-producing nodes stamp `frame_rate`. | `TimestampPromptScheduleBatchEncode` stamps it; preserved if you reuse `#1615`. |

### F3 hardcoded-IDs implication

`audit_workflows.py:538` does `if {644, 655, 1519}.issubset(sg_node_ids):` then reads `sg_links` for `target_id=644 target_slot=1/2`. The check is **gated on the IDs existing** — if your build script renumbers subgraph nodes (e.g. because of collision with B's top-level ID space), the F3 audit will silently **skip** rather than ERR. That's a hidden audit gap.

**Three options for the build script**:

1. **Preserve canonical subgraph node IDs verbatim.** Splice the subgraph definition with no renumbering. Requires checking B's top-level IDs don't collide with `{573, 574, 583, 596, 598, 600, 606, 644, 655, 1519, 1639, 1640, 2004, 2005, 2006, 2007, 2008, 2012}`. **Recommended** — least audit drift.
2. **Renumber and extend the audit.** Update `audit_workflows.py` F3 block to look up nodes by *title* or *type* instead of hardcoded IDs (e.g. `find sg node where title startswith "CropGuides (CONDITIONING half)"` for `#655`). Costs a follow-up apply-script + audit-pair change. Out of scope for a graft.
3. **Renumber and accept the audit gap.** Build still works, F3 silently no-ops. **NOT recommended** — it eliminates the only check protecting against the cropguides cycle regression.

**Recommendation**: option 1. Renumbering should happen only on collision, and B's top-level IDs are mostly in the `0-2300` range with gaps; the canonical subgraph IDs `{573, 574, 583, 596, 598, 600, 606, 644, 655, 1519, ...}` should be checked against B's full ID set before splicing.

The same hardcoded-IDs concern applies to top-level audit references:
- F2 line 508: `next((n for n in wf["nodes"] if n["id"] == 650), None)` — dead in canonical, safe.
- F3 line 538: subgraph `{644, 655, 1519}` — handled above.
- F14/F15: type-driven (find by `type=="TrimVideoLatentToAudio"` etc.), no ID hardcoding.

Other audits are type-driven and survive renumbering.

## References

- `example_workflows/working_docs/subgraph_chain_reference.md` — subgraph internals (IO table, internal nodes, internal links, F3 invariant details)
- `docs/reference/audio_loop_controller.md` — controller iter-state propagation + widget shape
- `docs/reference/pipeline_flow_latent.md` — F2/F3 trace
- `docs/reference/frame_planner_reference.md` — dimension SSoT
- `docs/reference/sampler_reference.md` — 8-step distilled sigma rationale
- `scripts/audit_workflows.py` — live audit-check source (F-pair invariants)
- `scripts/workflow_utils.py` — `WorkflowEditor` API (no `add_subgraph` helper)
- `scripts/CLAUDE.md` — apply-script + WorkflowEditor conventions
