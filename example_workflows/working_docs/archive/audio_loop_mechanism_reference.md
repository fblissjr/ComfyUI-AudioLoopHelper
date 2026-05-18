# Audio loop mechanism reference

> **STALE WARNING (added 2026-05-16)**: this doc was written 2026-05-15 when fps=24 was framed as canonical. **The fps 25→24 sweep was reverted on 2026-05-16; canonical inference fps is now 25** (matches Lightricks shipped workflows + 8n+1 latent boundary). Canonical `first_frame_guide_strength=0.7`, canonical `target_seconds=19.88`. Full postmortem: `internal/analysis/fps_24_partial_reading_postmortem.md` (private clone only). Re-read body claims through that lens.

Last updated: 2026-05-15

Canonical "how the loop works" reference for the LTX 2.3 audio-driven music-video pipeline. Anchors against `example_workflows/audio-loop-music-video_latent.json` and the runtime classes in `nodes.py` / `ComfyUI-NativeLooping_testing/nodes.py`. Read this once before touching any of: stride math, `noise_mask` boundaries, init-image symmetry, prompt routing, or the sampler chain.

> **Note (2026-05-15)**: LTX 2.3 was trained at 24fps and `LTXVConditioning.frame_rate=24` is the canonical default (`comfy/ldm/lightricks/av_model.py:866`). Examples below still cite the pre-2026-05-15 `fps=25` widget snapshot to match the live workflow JSON; re-derive `window_px = round(window_seconds * fps)` and downstream values at `fps=24` once the workflow-JSON migration lands. Audio-VAE math (16kHz / 160 / 4 = 25 audio latents per second) is INDEPENDENT of video fps and unaffected.

## Scope and disambiguation

- **What this covers**: the `TensorLoopOpen` + `TensorLoopClose` extension loop around a sampler subgraph, fed by `AudioLoopController` math. The shipped workflow generates arbitrarily long audio-conditioned video by iterating one fixed-size window and advancing audio by an integer-latent stride.
- **Not the same as `LTXVLoopingSampler`** (upstream ComfyUI-LTXVideo). That sampler rejects AV NestedTensor latents and is video-only. Rationale and source line: `docs/reference/ltxv_looping_sampler_reference.md`.
- **Not the same as `_image*.json` workflows.** The image-loop variants decode VAE every iteration. The latent loop here decodes once at the end. `docs/reference/pipeline_flow_latent.md` covers full node-by-node detail; this doc is the loop-mechanics summary.
- **Iteration indexing**: `current_iteration` is **1-based** at the `TensorLoopOpen` interface (see `_temporal_frame_count` callers and `nodes.py::AudioLoopController` widget order). The initial render runs **outside** the loop and is conceptually "iteration 0".

## Top-level data flow

```
                                                  +-- LatentConcat --+
initial render (iter 0, outside loop) -----------+                   |
   EmptyLTXVLatentVideo #344                                          |
   -> LTXVImgToVideoInplaceKJ #531                                    |
   -> LTXVConcatAVLatent #350 ----+                                   |
   -> SamplerCustomAdvanced #161  |                                   |
   -> LTXVSeparateAVLatent #245 --+--> #381 LTXVCropGuides ----------->|---> VAEDecodeTiled #1604 -> VHS_VideoCombine #617
                                  |                                   |
                                  +--> TensorLoopOpen #1539  (initial_value)
                                                |
                                  +-------------+-----------------+
                                  |   loop body (Extension #843)  |
                                  |   per iter: ContextExtract,   |
                                  |   AudioMask, AddLatentGuide,  |
                                  |   ConcatAV, Sampler,          |
                                  |   SeparateAV, CropGuides,     |
                                  |   OverlapTrim                 |
                                  +-------------+-----------------+
                                                |
                                                v
                                        TensorLoopClose #1540
                                        (accumulate=True, overlap=disabled)
                                                |
                                                +----------------------> samples2 of LatentConcat
```

## 1. The tensor loop primitive

Source: `ComfyUI-NativeLooping_testing/nodes.py::TensorLoopOpen` (line 34) and `TensorLoopClose` (line 122). The shipped workflow uses one `TensorLoopOpen #1539` and one `TensorLoopClose #1540`.

### What flows around the loop

`TensorLoopOpen` carries one typed value across iterations (`MatchType.Template("data", allowed_types=[Image, Mask, Latent])`). For the latent music-video workflow, the carried type is `LATENT`. Per iteration:

- `previous_value` (LATENT): on iter 1 = `initial_value` (the initial-render output from `#245 LTXVSeparateAVLatent.video_latent`). On iter N>1 = the previous iter's `processed` value from `TensorLoopClose`.
- `current_iteration` (INT, 1-based): consumed by `AudioLoopController #1582`, `ConditioningSelectByIteration #1616`, `LoopIterationStamp #1618`, `IterationCleanup #2007`.
- `accumulated_count` (INT): unwired in the shipped workflow.
- `flow_control` (FLOW_CONTROL): control token wired to `TensorLoopClose.flow_control`.

### `should_stop` semantics

`TensorLoopClose.stop` is **checked AFTER the loop body executes** (see the `_ConditionalSelect` block at `nodes.py::TensorLoopClose.execute` line 225). Critical consequence: the iteration that triggers `should_stop=True` still runs to completion. `AudioLoopController.execute` clamps `start_index` to `max(0.0, audio_duration - 0.5)` precisely to keep ≥0.5s of audio (>1024 samples) for the mel spectrogram on that final overshoot iter — without the clamp the loop body would crash on the last pass. Source: `nodes.py::AudioLoopController.execute` line 784-786.

The loop terminates when **any** of these is true:
1. `should_stop=True` (audio exhausted): wired from `#1582 AudioLoopController.should_stop` to `#1540 TensorLoopClose.stop` via raw link.
2. The mode-widget iteration count is exhausted: widget is `["iterations", 50, 0]` on `#1539` — would cap at 50, but `iterations_in` wire supersedes (see below).
3. (Unused here) `mode=total_frames` accumulator hits its target.

### `iterations_in` source

Per root CLAUDE.md: `AudioLoopPlanner.total_iterations -> TensorLoopOpen.iterations_in` is wired in every shipped workflow. Audit `iterations_autowired` (F5) enforces this. Verified for this workflow:

```
link 3069: src #1560 slot 1 (AudioLoopPlanner.total_iterations) -> dst #1539 slot 1 (TensorLoopOpen.iterations_in) type=INT
```

When `iterations_in > 0` it overrides the mode widget (see `TensorLoopOpen.execute` line 98). `AudioLoopPlanner.total_iterations` is `floor(audio_duration / stride_seconds)` capped at 200 (see `_compute_tile_count` in `nodes.py:288`). The mode widget value `50` is fallback only — non-load-bearing in shipped state.

Short-test override: drag in an `INTConstant` and rewire to `iterations_in`. Recipe in `docs/guides/debugging_guide.md`.

### `TensorLoopClose` accumulation

Widgets on `#1540`: `[True, "disabled"]` → `accumulate=True`, `overlap=disabled`. Each iteration's `processed` LATENT is appended to an internal `_AccumulateNode` accumulator. On loop exit, `_AccumulationToImageBatch` concatenates along the temporal dim (dim=2 for video latents and `NestedTensor` AV latents — see `_get_cat_dim` line 454).

The loop-body output type is plain video LATENT (not AV NestedTensor): `LatentOverlapTrim #2005` returns `s["samples"] = video[:, :, trim:]` after `LTXVSeparateAVLatent` has already split the AV bundle. Audio is re-encoded fresh every iteration from the source audio — there is no per-iter audio carry-over.

## 2. AudioLoopController and AudioLoopPlanner: the stride math

Source: `nodes.py::_compute_loop_geometry` (line 251) — shared by both classes. Both compute stride from the same primitives (`window_seconds`, `overlap_seconds`, `fps`); both produce the same `stride_seconds` value. There is **no wire** between them. This is intentional — the 2026-04-26 controller→planner edge created a dependency cycle and was retired. Audit `planner_no_stride_input` (F7) prevents reintroduction.

### Source of inputs

- `window_seconds` and `fps` come from `LTXFramePlanner #1634` (`actual_seconds` and `fps_int`). Widget values `[832, 448, 20, 25]` snap to `frames=497, actual_seconds=19.88, fps=25`. The planner is the single source of truth for dimensions; audit `frame_planner_present` (F8) enforces.
- `overlap_seconds` is a widget on each consumer (planner = 2, controller = 2 in the shipped workflow).
- `audio` is shared from `#567 TrimAudioDuration` (the song-trim downstream of `LoadAudio`).

### The math, concretely

Implementation: `nodes.py::_compute_loop_geometry` lines 251-285. Given `window_seconds`, `overlap_seconds`, `fps`:

```
window_px       = max(1, round(window_seconds * fps))
overlap_px      = max(0, round(overlap_seconds * fps))
window_latents  = (window_px - 1) // 8 + 1               # LTX video VAE: (L-1)/8 + 1
overlap_latents = (overlap_px - 1) // 8 + 1   if overlap_px > 0 else 0
   if overlap_latents >= window_latents:                  # auto-clamp
      overlap_latents = window_latents - 1
new_latents     = window_latents - overlap_latents
stride_px       = new_latents * 8
stride_seconds  = stride_px / fps
effective_overlap_pixel_frames = window_px - stride_px
effective_overlap_seconds      = (window_px - stride_px) / fps
```

For the shipped widget defaults (`window=19.88, overlap=2, fps=25`):

```
window_px  = round(19.88 * 25) = 497
overlap_px = round(2.00 * 25)  = 50
window_latents  = (497 - 1) // 8 + 1 = 63
overlap_latents = (50  - 1) // 8 + 1 = 7
new_latents     = 63 - 7              = 56
stride_px       = 56 * 8              = 448
stride_seconds  = 448 / 25            = 17.92
effective_overlap_seconds = (497 - 448) / 25 = 1.96
```

Hence `TimestampPromptScheduleBatchEncode #1615` widgets `[..., 17.92, 180, True, 25]` — its `stride_seconds` widget is the cached effective stride (180 = audio_duration cap).

### Why integer-latent counts, not widget seconds

LTX 2.3 video VAE compresses 8 pixel frames → 1 latent frame. If audio stride is computed from the widget `overlap_seconds=2` directly (= 17.88s stride), but the latent decoder emits 448 pixel frames per iter (= 17.92s of video), audio drifts forward by 0.04s/iter — multiplied across 10 iterations that's a half-second lip-sync error. By deriving stride from `new_latents * 8 / fps` you guarantee the audio advance per iter matches exactly what the decoder will emit. See `docs/reference/audio_loop_controller.md` lines 38-48 for the same derivation.

### `(L-1) % 8 == 0` constraint

This is the LTX video VAE temporal-compression constraint: encoder formula `latent = (pixel - 1) // 8 + 1` (silently floors invalid `length` in `comfy_extras/nodes_lt.py::EmptyLTXVLatentVideo`). `LTXFramePlanner` snaps `frames` DOWN to the nearest valid value at `nodes.py::_snap_frames` line 1510. The loop math reuses this same formula on both `window_px` and `overlap_px`, which is what makes `new_latents * 8` a clean integer number of pixel frames per iter.

### `total_iterations` computation

`AudioLoopPlanner.execute` line 1390: `auto_iterations = _compute_tile_count(audio_duration, stride_seconds)`. Implementation (`nodes.py:288`):

```
return max(1, min(int(audio_duration // stride_seconds), 200))
```

Uses `floor(audio_duration / stride)` so the last iter's START is within audio bounds. Note: the last iter's WINDOW can extend up to `window − stride` seconds past audio end. `TrimImageBatchToAudio #2029` (F14) clips this overshoot from the saved mp4. Rationale at `nodes.py:288-314` — bounding by `floor((audio − window) / stride)` would lose audio coverage at the end of the song, strictly worse than trimming the trailing video tail.

If the user sets `AudioLoopPlanner.max_iterations > 0`, the planner caps to `min(max_iterations, auto_iterations)` for short test runs.

### Per-iter outputs that feed the loop body

`AudioLoopController.execute` (lines 766-800) emits 8 outputs. Iteration-dependent (recomputed per loop pass):

- `start_index` (FLOAT): `current_iteration * stride_seconds`, clamped to `max(0, audio_duration - 0.5)`. Wired to subgraph slot 11 → `TrimAudioDuration #600` inside the subgraph (BYPASSED in shipped — see §3).
- `should_stop` (BOOLEAN): `(current_iteration + 1) * stride_seconds >= audio_duration`. Raw link to `TensorLoopClose #1540`.
- `iteration_seed` (INT): `base_seed + current_iteration`. Wired to subgraph slot 13 → `RandomNoise #574`.

Iteration-independent in value but still flowing through controller every iter (they depend on widget inputs and audio only, but ComfyUI's DAG recomputes them because `current_iteration` is a controller input):

- `audio_duration`, `stride_seconds`, `overlap_frames`, `overlap_latent_frames`, effective `overlap_seconds`.

**Critical cycle-avoidance rule**: anything OUTSIDE the loop that needs `stride_seconds` or `audio_duration` must source from `AudioLoopPlanner`, not `AudioLoopController`. Reason: the controller transitively depends on `current_iteration`, so feeding its outputs into the initial-render conditioning closes a cycle through `TensorLoopOpen`. The shipped workflow respects this: `TimestampPromptScheduleBatchEncode #1615` is fed `stride_seconds` and `audio_duration` from `AudioLoopPlanner #1560` (links 3187, 3188), NOT from the controller. Audit `graph_acyclic` catches violations.

## 3. Inside the loop body: subgraph "extension" #843

Subgraph id `b4973d68-09b9-4da5-9845-38ad62ae9aca`. The shipped extension subgraph has 17 internal nodes (some bypassed for the audio path — see below). External I/O:

### External inputs (from `TensorLoopOpen` and outer workflow)

| Slot | Name | Type | Source |
|---|---|---|---|
| 0 | sampler | SAMPLER | `#578 Get_sampler` (← `#154 KSamplerSelect`) |
| 1 | sigmas | SIGMAS | `#580 Get_sigmas` (← `#1421 ManualSigmas`) |
| 2 | model | MODEL | `#654 Get_model` |
| 3 | vae | VAE | `#619 Get_video_vae` |
| 4 | previous_latent | LATENT | `#1539 TensorLoopOpen.previous_value` |
| 5 | video_end_time | FLOAT | `#691 Get_window_size_seconds` (19.88) |
| 6 | positive | CONDITIONING | `#1616 ConditioningSelectByIteration` (loop body) |
| 7 | negative | CONDITIONING | `#648 Get_base_cond_neg` |
| 8 | guide_latent | LATENT | top-level `VAEEncode` of init image |
| 9 | audio_vae | VAE | `#599 Get_audio_vae` |
| 10 | audio | AUDIO | `#641 Get_actual_audio` |
| 11 | start_index | FLOAT | `#1582 AudioLoopController.start_index` |
| 12 | num_guides.strength_1 | FLOAT | `#1273 Get_first_frame_guide_strength` (1.0) |
| 13 | noise_seed | INT | `#1582 AudioLoopController.iteration_seed` |
| 14 | num_frames (overlap_latent_frames) | INT | `#1582 AudioLoopController.overlap_latent_frames` |

External output (single): `extended_latent` (LATENT) → `#1540 TensorLoopClose.processed`.

### Per-iter `noise_mask` setup

Three nodes cooperate to enforce "audio frozen, overlap context frozen, new video generated". Reference: `docs/reference/noise_mask_semantics.md`.

1. **`LatentContextExtract #2004`** (`nodes.py:1965`) takes the previous iteration's full latent and slices the last `overlap_latent_frames=7` frames as context. **Strips `noise_mask`** (`s.pop("noise_mask", None)` at line 2006). This strip is mandatory: `LTXVAudioVideoMask` uses `existing_mask_mode: "add"` which `max(existing, new)`-merges stale masks, corrupting region semantics.

2. **`LTXVAudioVideoMask #606`** (KJNodes) creates a fresh mask map. Widgets in shipped workflow: `[25, 1, 10, 10, 10, "pad", "add"]` = `[video_fps, video_start_time(proxy), audio_start_time, audio_end_time, video_end_time, max_length, existing_mask_mode]`. The pattern `audio_start_time == audio_end_time` (= 10 in widget; overridden by input wire from slot 5 = 19.88) creates an **empty audio mask range** → audio frames all get `noise_mask=0` (fixed). Video context frames get `noise_mask=0`; new video frames get `noise_mask=1`. **Don't change this wiring** — it's documented as intentional in root CLAUDE.md.

3. **`LatentOverlapTrim #2005`** (`nodes.py:2011`) after sampling, trims the first `overlap_latent_frames=7` frames (the overlap region that was regenerated as fixed-context anyway) and again strips `noise_mask`. Output is clean new-content latent for accumulation by `TensorLoopClose`.

`LatentTemporalMask` (`nodes.py:2493`) is the inverse pattern used by the retake workflow (mask=1 inside a time range, 0 outside). Not used in the music-video latent workflow.

### Prompt routing (CLIP must never enter the loop body)

`docs/reference/timestamp_prompt_schedule_batch_encode.md` is canonical. The mechanism in this workflow:

- **Outside the loop**: `#1615 TimestampPromptScheduleBatchEncode` consumes `clip` (Gemma 3) once, plus `stride_seconds` and `audio_duration` from `AudioLoopPlanner #1560`. Widget: `["0:00+: video of a man dancing", 17.92, 180, True, 25]`. Emits a `LIST[CONDITIONING]` (one per expected iteration + 1 headroom entry). Stamps `frame_rate=25.0` on every entry via `node_helpers.conditioning_set_values` — required to prevent identity drift iter-over-iter.
- **Inside the loop body**: `#1616 ConditioningSelectByIteration` (`nodes.py:1212`) plucks the per-iter entry from the list by `current_iteration` (wired from `TensorLoopOpen.current_iteration`). Clamps to `[0, len-1]` to absorb the headroom and stay safe on overshoot.
- **Module-level cache**: `_BATCH_ENCODE_CACHE` is an LRU(4) keyed on `(id(clip), schedule, stride_seconds, audio_duration, snap_boundaries, frame_rate)`. Survives ComfyUI framework-level cache invalidation; dies on process restart.

Why this matters: when CLIP loads inside the loop body, ComfyUI ModelPatcher's `object_patches` closures (which NAG uses to capture `nag_cond_video`) silently go stale across the CLIP-load-triggered DiT eviction. Iter 2+ runs against stale captures → NAG disengages silently → microphones / anatomy regressions / style drift returning after iter 1. Mechanism walkthrough: `docs/analysis/nag_object_patches_offload_asymmetry.md`.

The legacy `TimestampPromptSchedule` (`nodes.py:803`) is still defined for back-compat but emits a deprecation warning via `_warn_legacy_use` on every execute. Bypassed `#1558` node in the workflow file is from before the 2026-04-22 migration; not consulted at runtime.

### Per-iter init-image guide (`LTXVAddLatentGuide #1519`)

The init image is **encoded once** at the top level via `VAEEncode`, then fed into the subgraph as `guide_latent` (slot 8). Inside the subgraph:

- `LTXVAddLatentGuide #1519` widgets `[-1, 1]` = `[latent_idx, strength]`. `latent_idx=-1` positions the guide before the first frame (RoPE-style temporal positioning). `strength` widget is overridden by the wire from slot 12 (`first_frame_guide_strength`, default 1.0 via `#1269 FloatConstant`).
- Inputs: `vae`, `positive`, `negative` (from `LTXVAudioVideoMask`-masked video latent and current conditioning), `latent` (the masked video latent), `guiding_latent` (the slot-8 init-image latent), `strength` (slot 12).
- Outputs: modifies `positive`/`negative` with guide attention entries (`keyframe_idxs`), appends guide frames to the latent's temporal dim, extends `noise_mask` with strength-derived values.

**`first_frame_guide_strength` semantics**: 1.0 = max identity stability (init pinned hard, minimal motion between iter-start overlap and iter-end anchor — both = init image). Lower for music-video expressivity at the cost of cross-iter identity drift: 0.5 soft anchor, 0.3 visible drift, 0.0 no anchor. Source: root CLAUDE.md "Init image conditioning + IC-LoRA paths" section.

After sampling, `LTXVCropGuides #2008` strips the appended guide frame from the latent (and corresponding `keyframe_idxs` from conditioning); `LatentOverlapTrim #2005` strips the overlap region from the front.

### Other loop-body nodes in the shipped subgraph

- `LoopIterationStamp #1618` (top-level, `nodes.py:3454`): stamps `transformer_options["iteration"]` on the model clone. Wired between `Get_model` and `LTX2_NAG` / sampler. Lets the sage tracer (`nodes_sage.py::_iter_from_kwargs`) attribute per-iter kernel work to a loop pass.
- `IterationCleanup #2007` (subgraph, `nodes.py:3398`): LATENT passthrough that runs `gc.collect() + torch.cuda.empty_cache()` per iter. Mode `always` in shipped. Recommended by comfy-aimdo to prevent allocator fragmentation across iterations.
- `LTXVAdainLatent #2006` (subgraph): AdaIN color correction per iter. Widgets `[0.2, False]` (strength=0.2, disabled-by-default flag off). Cheap insurance against color drift; orthogonal to loop math.
- `LTXVCropGuidesNoLatent #655` (subgraph): conditioning-side crop (strips `keyframe_idxs` from positive/negative WITHOUT touching the latent). Pairs with the split `LTXVCropGuides #2008` for the latent side. Together they preserve F3 symmetry — see §5.

### What exits the loop body

`LatentOverlapTrim #2005.trimmed` → subgraph output slot 0 → `TensorLoopClose #1540.processed`. The output is a plain video LATENT (not AV NestedTensor — `LTXVSeparateAVLatent #596` discards the audio path; each iter re-encodes audio from source).

## 4. Initial render outside the loop

The initial render produces iteration 0 — the first window of the video. It runs OUTSIDE the loop because (a) it has no "previous iteration latent" to extract context from and (b) ComfyUI evaluates downstream conditioning before upstream sampling, so initial-render conditioning must already be in the graph before loop entry.

### `EmptyLTXVLatentVideo #344 → LTXVImgToVideoInplaceKJ #531 → LTXVConcatAVLatent #350 → SetLatentNoiseMask #570 ⟶ SamplerCustomAdvanced #161`

- **`EmptyLTXVLatentVideo #344`** widgets `[832, 448, 497, 1]` — width, height, length, batch. Length=497 satisfies `(L-1) % 8 == 0` (= 62 latent frames @ 8:1). Source dims wired from `LTXFramePlanner #1634` outputs (`width`, `height`, `frames`). Creates a zero-init latent of the target shape.
- **`LTXVImgToVideoInplaceKJ #531`** widgets `[1, 1, 0]` = `[num_images, strength_1, index_1]`. VAE-encodes the preprocessed init image and writes it at frame 0 of the empty latent, attaching a denoise mask: frame 0 = mask=0 (locked), all others = mask=1 (regenerate). Inputs: video VAE, the empty latent, the preprocessed image (`#446 LTXVPreprocess` output).
- **`LTXVConcatAVLatent #350`** wraps video latent + masked audio latent (from `#570 SetLatentNoiseMask`) into the AV NestedTensor the sampler consumes.
- **`SetLatentNoiseMask #570` + `SolidMask #571`** apply a value=0 mask to the audio latent — audio frames are FROZEN as the real encoded song, not regenerated. Audio path is sacred (root CLAUDE.md).

This chain runs once at workflow execution, **before** the loop opens. Its output feeds:
1. `TensorLoopOpen.initial_value` (link 3124, slot 0) → becomes `previous_value` on iter 1
2. `LTXVCropGuides #381` → `LatentConcat #1605` samples1 (the prepend prefix for final concat)
3. `LTXVTiledVAEDecode #1318` (active, ungated preview decode of the initial window)

### `TrimAudioDuration #601` widgets `[0, 10]` — the "10s context" for initial render

Title in workflow JSON: "Initial-Render Audio Trim (10s context)". `start_index=0`, `duration=10`. This trims the song down to its FIRST 10 SECONDS for the initial-render audio path. The trimmed audio is fed to `#566 LTXVAudioVAEEncode` → `#570 SetLatentNoiseMask` → `#350 LTXVConcatAVLatent`.

Why 10 seconds, not the full 19.88s window? The initial render's audio doesn't need to fill the entire window — `LTXVAudioVideoMask` is not in the initial-render chain (it lives only in the loop body subgraph), so the audio latent here is just any AV-pair that the sampler can co-denoise to lock the video frame 0 to its corresponding audio. 10s is enough audio context for the first iteration's cross-attention to anchor on, while avoiding encoding the full song before the loop has started. The loop iterations beyond iter 0 re-trim audio fresh per iter via `#600 TrimAudioDuration` (currently bypassed in the shipped subgraph — see §3 caveat about `AudioLatentSlice #2012` which has taken over that role for the latent path).

### `Node 169 == schedule[0]` rule

Background: `nodes.py::TimestampPromptScheduleBatchEncode._prepare_sections` and `_build_prompt_for_section` are the shared parser; the initial-render text encode (`CLIPTextEncode #169`) must produce a CONDITIONING **byte-exact** with what the batch encoder emits for `current_iteration=0`. If they diverge, the conditioning the user sees at iter 0 (initial render) differs from what `ConditioningSelectByIteration` plucks at iter 0 in the loop — visible style discontinuity at the iter-0 → iter-1 transition.

In the shipped workflow this is handled by routing `#2021 ConditioningSelectByIteration (Initial render conditioning (from schedule[0]))` between the batch encoder and `#164 LTXVConditioning`. Verified wiring:

```
link 3184: src #1615 (batch encoder) slot 0 -> dst #2021 (initial-render selector) slot 0
link 3185: src #2021 slot 0 -> dst #164 (LTXVConditioning) slot 0    (positive)
link 3186: src #2021 slot 0 -> dst #420 (ConditioningZeroOut)         (negative path)
```

The selector's `current_iteration` widget is 0 (it's not wired). Result: the initial render gets the schedule's first entry directly from the batch encoder. There is no separate `CLIPTextEncode #169` consumed for the initial render in the latent.json workflow — the text encode happens inside `TimestampPromptScheduleBatchEncode` and the same encoded CONDITIONING serves iter 0 (initial) and iter 0 (loop selector).

This is the structural form of the "Node 169 prompt matches schedule 0:00 entry" rule from root CLAUDE.md: by routing both paths through the same encoder output, byte-exact equivalence is automatic.

## 5. F2 / F3 symmetry rules

These are MANDATORY for the init-image path. Background: `docs/reference/pipeline_flow_latent.md`. F-pair convention: `docs/reference/f_pair_convention.md`.

### F2: both initial-render and loop branches share `LTXVPreprocess(img_compression=18)`

Both branches must consume the **same `LTXVPreprocess` output**, not raw `ImageResizeKJv2` output. Different preprocessing → different pixel statistics → different VAE encoding → photoreal drift / subject replacement across the iter-0 → iter-1 boundary.

Verified in the shipped workflow:
- **Initial render branch**: `#446 LTXVPreprocess` (widget `[18]`) → `#531 LTXVImgToVideoInplaceKJ.image_1` (consumes the preprocessed image directly).
- **Loop branch**: same `#446 LTXVPreprocess` → `#650 Set_input_image` → `Get_input_image` → top-level `VAEEncode` → subgraph slot 8 (`guide_latent`) → `#1519 LTXVAddLatentGuide.guiding_latent`.

Same source node (`#446`), same widget `img_compression=18` (Lightricks default). Audit/migration: `scripts/apply_loop_guide_preprocess_symmetry.py`.

The workflow file also has `#1638 LTXVPreprocess` (widget `[18]`, title "Preprocess ref-video (F2 symmetric)") for the IC-LoRA video-reference path (F12). It is the same preprocessing applied to a reference video for that bypassed feature.

### F3: loop `CFGGuider` positive/negative flow through `LTXVCropGuides`

The loop body's `CFGGuider #644` must receive its positive/negative from `LTXVAddLatentGuide #1519` outputs that have been routed through `LTXVCropGuides` (or a CropGuides-pair on the cond and latent sides). Skipping this drops `keyframe_idxs` cleanup and corrupts cross-iter conditioning state — identity drift across iterations.

Verified: inside subgraph, `#1519.positive` → `#644 CFGGuider.positive` (via link 2830) AND `#1519.positive` → `#655 LTXVCropGuidesNoLatent.positive` (via link 2832). The split-CropGuides setup (`#655` for cond-only, `#2008` for latent-only) is the F3-compliant form for the AV path because the standard `LTXVCropGuides` would touch the AV NestedTensor in ways that break the audio carry — splitting it lets cond cleanup happen without touching the AV latent until after `LTXVSeparateAVLatent #596` has split it.

Both F2 and F3 holding is the gate against the photoreal-drift / identity-drift footgun.

## 6. Sampler chain — distilled 8-step path

Verify the latent.json file matches the canonical distilled path. Reference: `docs/reference/sampler_reference.md`.

| Setting | Shipped value | Verified | Notes |
|---|---|---|---|
| `ManualSigmas` (`#1421`) | `"1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"` | YES | Bit-exact `DISTILLED_SIGMA_VALUES` from `coderef/LTX-2/.../utils/constants.py`. 8-step plateau + collapse curve. |
| `KSamplerSelect` (`#154`) | `"euler"` | YES | Deterministic Euler, matches `coderef/LTX-2/.../samplers.py::euler_denoising_loop`. NOT `euler_ancestral*`. |
| `CFGGuider.cfg` (`#153` initial, `#644` loop) | `1` (both) | YES | NAG handles guidance; CFG branch wired but inert at cfg=1. |
| `ModelSamplingSD3` | ABSENT | YES | No flow-matching shift node in the active model chain. The bypassed `#1513` may still exist in the workflow file but is not on the active path. |

Why `ManualSigmas`, not `BasicScheduler linear_quadratic 8 1`: the canonical hand-tuned `DISTILLED_SIGMA_VALUES` are the spec (what the distilled checkpoint was trained to denoise); the linear-quadratic parameterization only approximates them. Migration: `scripts/apply_canonical_sigmas.py`.

Why no `ModelSamplingSD3`: distilled checkpoint was trained on the fixed sigma curve without flow-matching shift. Adding shift moves off the trained distribution. The bypassed `#1513` node in older workflow forks is a relic; it should be removed via `scripts/apply_strip_sd3_shift_node.py`.

Decode: `LTXVTiledVAEDecode #1604` widgets `[320, 240, 32, 16]`. On 24GB+ the recommended single-tile config is `[1, 1, 1, true, "auto", "auto"]` — ~3× faster cold-pass than `[2, 2, 1]`. Shipped widgets in this file represent the multi-tile fallback for ≤16GB; consider running `scripts/apply_no_tile_vae_decode.py` on a 24GB+ system.

## 7. Loop-body invariants (one-page summary)

For an LLM reader who needs the load-bearing rules without re-deriving them:

1. **CLIP outside the loop body, always.** Pre-encode via `TimestampPromptScheduleBatchEncode #1615`; pluck per-iter via `ConditioningSelectByIteration #1616`. Audit: `prompt_schedule`.
2. **`AudioLoopPlanner #1560.total_iterations → TensorLoopOpen #1539.iterations_in`**, in every shipped workflow. Audit `iterations_autowired` (F5). Verified via link 3069 in this file.
3. **Stride from integer-latent counts, not widget seconds.** `_compute_loop_geometry` is the canonical formula. Don't bypass.
4. **`AudioLoopController` outputs are iteration-dependent in the DAG.** Anything outside the loop that needs `stride_seconds`/`audio_duration` sources from `AudioLoopPlanner`, not the controller. Audit `graph_acyclic`.
5. **No `controller → planner` stride wire.** Both nodes derive stride locally from the same primitives. Audit `planner_no_stride_input` (F7).
6. **`noise_mask` boundary discipline**: `LatentContextExtract` strips on read, `LTXVAudioVideoMask` creates fresh `{audio:0, video:1, overlap:0}`, `LatentOverlapTrim` strips on emit. Bypass either strip and stale masks contaminate the next iter via `existing_mask_mode: "add"`.
7. **F2 + F3 symmetry**: both initial-render and loop branches consume the same `LTXVPreprocess` output; loop `CFGGuider` positive/negative flow through `LTXVCropGuides` (split into cond/latent halves for the AV path). Audit pair via `apply_loop_guide_preprocess_symmetry.py`.
8. **Sampler is distilled 8-step**: `ManualSigmas` (hand-tuned values), `KSamplerSelect euler`, `CFGGuider cfg=1`, NO `ModelSamplingSD3`, NO `euler_ancestral*`.
9. **`TensorLoopClose.stop` is checked AFTER the body.** `AudioLoopController.start_index` clamps to `audio_duration - 0.5s` to keep the overshoot iter from crashing on the mel spectrogram's >1024-sample requirement.
10. **Loop-body CONDITIONING-producing nodes must stamp `frame_rate`** via `node_helpers.conditioning_set_values`. Missing → identity drift iter-over-iter. AST guard: `tests/test_node_schemas.py`.
11. **`Node 169 == schedule[0]` structural rule**: handled in this workflow by routing both the initial-render and loop-iter-0 conditioning through the same batch encoder (`#1615 → #2021 → #164` for initial; `#1615 → #1616` for loop). Byte-exact equivalence is automatic when the path is shared.

## 8. Citations

- Workflow: `example_workflows/audio-loop-music-video_latent.json`
- Runtime classes:
  - `ComfyUI-NativeLooping_testing/nodes.py::TensorLoopOpen` (line 34), `TensorLoopClose` (line 122)
  - `nodes.py::_compute_loop_geometry` (line 251), `_compute_tile_count` (line 288), `_audio_duration` (line 317)
  - `nodes.py::AudioLoopController` (line 634), `AudioLoopPlanner` (line 1270)
  - `nodes.py::TimestampPromptScheduleBatchEncode` (line 1008), `ConditioningSelectByIteration` (line 1212)
  - `nodes.py::LatentContextExtract` (line 1965), `LatentOverlapTrim` (line 2011), `LatentTemporalMask` (line 2493)
  - `nodes.py::LoopIterationStamp` (line 3454), `IterationCleanup` (line 3398)
  - `nodes.py::LTXFramePlanner` (~line 1505), `_snap_frames` (line 1510)
- Reference docs:
  - `docs/reference/audio_loop_controller.md` — stride math + cycle history
  - `docs/reference/timestamp_prompt_schedule_batch_encode.md` — CLIP-outside-the-loop mechanism
  - `docs/reference/noise_mask_semantics.md` — boundary strip discipline
  - `docs/reference/frame_planner_reference.md` — dimension SSoT
  - `docs/reference/sampler_reference.md` — distilled 8-step justification
  - `docs/reference/pipeline_flow_latent.md` — full node-by-node trace
  - `docs/architecture_overview.md` — entry point + sister-doc map
  - `docs/reference/debug_tools.md` — F-pair audit inventory
  - `docs/analysis/nag_object_patches_offload_asymmetry.md` — why CLIP-in-loop silently breaks NAG
- Scripts:
  - `scripts/audit_workflows.py` — F-series audit checks
  - `scripts/apply_canonical_sigmas.py` — distilled sigma migration
  - `scripts/apply_strip_sd3_shift_node.py` — remove flow-matching shift
  - `scripts/apply_loop_guide_preprocess_symmetry.py` — F2 enforcement
  - `scripts/apply_no_tile_vae_decode.py` — 24GB+ single-tile decode
  - `scripts/apply_trim_image_batch_to_audio.py` — F14 audio-overshoot trim
