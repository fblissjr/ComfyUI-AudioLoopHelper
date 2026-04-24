# ComfyUI-AudioLoopHelper

## TLDR

ComfyUI nodes that automate loop timing and audio analysis for full-length
music video generation with LTX 2.3. The main node (AudioLoopController)
reads audio duration from the tensor, computes stride from window + overlap,
outputs start_index / should_stop / audio_duration / iteration_seed /
stride_seconds / overlap_frames / overlap_latent_frames / overlap_seconds.

## Architecture

Two files: `nodes.py` (core loop nodes) and `nodes_analysis.py` (runtime
audio analysis). Uses ComfyUI's extension API (`ComfyExtension`,
`io.ComfyNode`). Single entry point: `comfy_entrypoint()` in nodes.py
imports analysis nodes from nodes_analysis.py.

Core nodes (nodes.py):
- `AudioLoopController` -- core: 8 outputs. overlap_seconds (slot 7) auto-wires to LTXVAudioVideoMask video_start_time.
- `TimestampPromptScheduleBatchEncode` -- PRIMARY path. Pre-encodes every per-iteration prompt OUTSIDE the loop; emits `conditioning_list` + `iteration_count`. Dedup means each unique prompt encodes once. Pairs with `ConditioningSelectByIteration`. Eliminates the CLIP offload cycle that silenced NAG iter 2+ (see "CLIP must not enter the loop body" in Critical constraints, and `docs/analysis/nag_object_patches_offload_asymmetry.md`). Carries its own module-level LRU (`_BATCH_ENCODE_CACHE`) keyed on `(id(clip), schedule, stride_seconds, audio_duration, snap_boundaries)` + an `IS_CHANGED` hook: required because `AudioLoopController` (upstream of the batch encoder's `stride_seconds` + `audio_duration`) re-executes per iteration, invalidating ComfyUI's framework-level cache and forcing re-encode of every unique prompt per loop pass unless we memoize ourselves.
- `ConditioningSelectByIteration` -- indexes the pre-encoded `conditioning_list` by `current_iteration`. Runs inside the loop; no CLIP reference. Clamps on overshoot/negative; empty list raises.
- `TimestampPromptSchedule` -- LEGACY emit-strings path. Still exported, not wired in the shipped `_latent.json` (superseded by batch encode). Useful for workflows that explicitly want per-iter re-encode.
- `ConditioningBlend` -- lerps two conditioning tensors (works with LTX Gemma 3 and CLIP). Not wired by default after 2026-04-22; kept for workflows that need per-iteration cross-fade via the legacy TimestampPromptSchedule path.
- `AudioLoopPlanner` -- displays iteration timeline for planning
- `AudioDuration` -- extracts duration/sample_rate from audio tensor
- `LatentContextExtract` -- extracts tail latent frames + strips noise_mask
- `LatentOverlapTrim` -- trims overlap latent frames + strips noise_mask
- `StripLatentNoiseMask` -- low-level noise_mask removal utility
- `LatentTemporalMask` -- retake support. Writes a noise_mask to a video latent so only `[start_time, end_time]` regenerates on a re-sample; rest stays fixed as context. Latent-frame math: `start_latent = int(t*fps/8)`, `end_latent = int(end*fps/8) + 1`. Reversed or zero-width ranges yield an all-zero mask (no-op). Port of `TemporalRegionMask.apply_to` from `coderef/LTX-2/.../retake.py`. Tests in `tests/test_retake_nodes.py`.
- `KeyframeImageSchedule` -- per-iteration keyframe image selection from timestamp schedule (like TimestampPromptSchedule but for images). Outputs image/next_image/blend_factor. **Still uses the legacy spike-blend path** (no `snap_boundaries` widget yet); sub-stride `blend_seconds` produces jitter. Phase 1.5 follow-up in the plan.
- `VideoFrameExtract` -- extracts frame from reference video at current iteration's timestamp for video-to-video conditioning
- `ImageBlend` -- pixel-space lerp of two images by a factor. Pairs with KeyframeImageSchedule for smooth keyframe transitions.
- `CachedTextEncode` -- LEGACY. Drop-in replacement for CLIPTextEncode with LRU cache keyed by `(id(clip), text)`. Not wired in `_latent.json` after 2026-04-22 — the batch encoder runs CLIP once for the whole schedule, which is strictly better than per-iter caching. Kept for workflows that still use `TimestampPromptSchedule` inside the loop.
- `IterationCleanup` -- LATENT passthrough that runs `gc.collect()` + `torch.cuda.empty_cache()`. Place near subgraph output to reduce allocator fragmentation between iterations. Modes: always / gpu_only / never.
- `LoopIterationStamp` -- MODEL passthrough that writes `transformer_options["iteration"] = int(current_iteration)`. Insert between the patch chain and the subgraph invoker, feeding `TensorLoopOpen.current_iteration` into it. Per-iteration tracers (sage JSONL, profiler) read the stamp to attribute work to a loop pass. Additive: preserves `optimized_attention_override` and any other transformer_options keys. Wiring helper: `scripts/apply_iteration_stamp.py [--all]`. Tests in `tests/test_iteration_stamp.py`.
- `AudioLoopHelperSageAttention` (in `nodes_sage.py`) -- sage attention patch with pytorch fallback, `ON_CLEANUP` handler, opt-in JSONL telemetry (env: `AUDIOLOOPHELPER_SAGE_TRACE`), and arch-filtered mode combo. **Default `auto_mask_aware`** — masked cross-attn → `sageattn_qk_int8_pv_fp16_triton`, unmasked self-attn → sage `auto` (fp8++ on Ada). Routing exists because sage's CUDA kernels don't implement mask support (MaskMode is `{kNone, kCausal}`; `attn_mask` is silently dropped via kwargs); only triton has a masked path. Stateless per-call dispatch — no offload/loop state risk. Tracer emits both `mode` and `effective_mode` so routing can be verified post-hoc. Swap KJ's `PathchSageAttentionKJ` in/out via `scripts/apply_audioloophelper_sage.py [--all] [--revert]`; change mode via `scripts/apply_sage_mode.py <mode|mask_aware>` (handles both node types). Full reference: `docs/reference/sage_attention.md`. Patch-chain analysis: `internal/analysis/sage_attention_analysis.md`. Backlog: `internal/design/sage_backlog.md`. Tests: `tests/test_sage_node.py`.
- `ProfileBegin` / `ProfileIterStep` / `ProfileEnd` -- three-node `torch.profiler` integration. **Opt-in via `scripts/apply_profiling_nodes.py`**; example workflows do NOT ship with profile nodes. `record_function` spans on hot nodes are runtime-gated via `_profile_span()` so they're zero-overhead (`nullcontext`) when no profiler is active. See `docs/guides/profiling_guide.md`.

Analysis nodes (nodes_analysis.py, torchaudio only):
- `AudioPitchDetect` -- per-iteration F0 detection, vocal presence, male/female classification. Outputs FLOAT/BOOLEAN only.

Key helper functions: `_audio_duration`, `_parse_timestamp` ("M:SS" or bare seconds),
`_format_timestamp` (preserves sub-second; NOT same as `_fmt_ts()` in analyze_audio_features.py
which truncates). Schedule parsing uses generic `_parse_schedule_generic` /
`_match_schedule_generic` / `_match_schedule_with_next_generic` parameterized by
value converter and default. Thin wrappers: `_parse_schedule` / `_match_schedule` (str),
`_parse_image_schedule` / `_match_image_schedule` (int via `_safe_int`).

## Key patterns

- AUDIO type: `{"waveform": Tensor, "sample_rate": int}`. Duration = `waveform.shape[-1] / sample_rate`.
- **Stride is derived from integer-latent counts**, not from `window - overlap` seconds. `AudioLoopController` computes `new_latent_frames = window_latents - overlap_latents`, then `stride_seconds = new_latent_frames * 8 / fps`. The user's `overlap_seconds` widget is a TARGET; the node outputs the EFFECTIVE quantized overlap. This guarantees audio advances per iteration by the same number of real pixel frames the video decoder emits, so lip-sync cannot drift regardless of overlap value. Prior to 2026-04-20 the node used `stride = window - overlap` directly, which accumulated ~0.04s/iter drift at overlap=2 and ~0.12s/iter at overlap=4 because of the sequence-start-vs-mid-sequence latent interpretation mismatch. Tests: `tests/test_audio_loop_controller.py`.
- start_index is clamped so at least 0.5s of audio always remains (prevents mel crash on final iteration).
- TimestampPromptSchedule only runs in loop iterations, NOT the initial render. Node 169 handles initial ~20s.
- **Prompt changes cause style drift even at CFG 1.0.** Default mitigation: `snap_boundaries=True` + identical subject across entries. For cross-fade at visible seams, set `blend_seconds ≥ stride_seconds` (sub-stride values are auto-clamped — they can't produce smooth ramps at iteration resolution).
- LTX 2.3 uses Gemma 3 text encoder (NOT CLIP). Format: `[tensor, {"attention_mask": mask}]`, no pooled_output. Standard ConditioningAverage won't work.
- **Audio path is sacred.** Audio enters via `LTXVAudioVAEEncode -> LTXVConcatAVLatent`. Never feed visualizations into the video latent stream.
- Video VAE formula: `latent = (pixel - 1) // 8 + 1`. NOT `pixel // 8`.
- **mask=0 means "fixed context"** in LTX noise masks. Audio latent with mask=0 keeps real song. mask=1 regenerates from noise, destroying lip sync.
- LTXVLoopingSampler CANNOT support AV latents (2 architectural root blockers — temporal-schedule mismatch between audio/video latent densities, and model cross-attention trained on joint-not-tiled AV; 3 type-system cascades follow from those). TensorLoop is the correct approach. See `docs/analysis/ltx23_gaps_analysis.md`.
- **Guide chaining works**: Multiple `LTXVAddLatentGuide` or KJNodes' `LTXVAddGuideMulti` calls accumulate guides via `keyframe_idxs` metadata. `LTXVCropGuides` counts all guides and strips them correctly. Path to multi-guide-per-window without custom nodes.

## Critical constraints

- **Never feed audio visualizations into video latent stream** -- DiT generates heatmap-looking frames.
- **Never change LTXVAudioVideoMask (Node 606) wiring** -- audio_start_time = audio_end_time = window_size is intentional (empty mask range keeps audio fixed).
- **Use LatentContextExtract/LatentOverlapTrim** instead of raw LTXVSelectLatents in latent-space subgraph -- they strip noise_mask automatically.
- **Node 169 prompt MUST match schedule's 0:00 entry** to avoid visual discontinuity at ~20s. Enforced structurally: `get_node_169_prompt` and `_generate_subject_schedule` both call `_build_prompt_for_section` via the SAME subdivision (`_prepare_sections`), so the first schedule entry is byte-exact to Node 169.
- **Every generated prompt MUST contain "singing"** (or "are singing together" for multi-subject). LTX 2.3's audio-video joint cross-attention drives lip sync off the action verb; generic "performing" loses the signal. Enforced in `_SECTION_MODIFIERS` and `_build_action_phrase`; checked by `test_prompts_always_include_singing_verb_with_subject`.
- **Always use WorkflowEditor** from `scripts/workflow_utils.py` for subgraph edits. Manual JSON surgery breaks links.
- **Distilled 8-step sampling reverse-engineered to pure-pipeline sigmas.** The shipped chain (`BasicScheduler linear_quadratic, 8, 1` + `ModelSamplingSD3 shift=13` + `KSamplerSelect euler` + `CFGGuider CFG=1`) produces bit-exact `DISTILLED_SIGMAS` from `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:16` = `[1.0, 0.994, 0.988, 0.981, 0.975, 0.909, 0.725, 0.422, 0.0]`. Verify via `VisualizeSigmasKJ` before touching any part of this chain. Decoder: `LTXVTiledVAEDecode [2, 2, 1, true, "auto", "auto"]`. Don't use `euler_ancestral` or `euler_ancestral_cfg_pp` — plateau at σ≈0.99 for 5 steps amplifies re-noise → iteration drift. Full sampler walkthrough with ComfyUI + MultimodalGuider source references: `docs/reference/sampler_reference.md`.
- **LTX 2.3 audio-video cross-attention is photoreal-trained.** Illustrated / painterly / 3D-render inits progressively drift toward photoreal across loop iterations ("broadway musical" failure mode). `Style: illustrated.` at CFG=1 is too weak to overcome the trained prior. First-line fix: match init-image style family to training distribution (use cinematic / photoreal init). Structural fix (not yet built): multi-image-guide per iteration via KJNodes' `LTXVAddGuideMulti` to re-anchor illustrated style mid-iteration.
- **LTX2_NAG widgets `[nag_scale, nag_alpha, nag_tau, inplace]`** per source `ComfyUI-KJNodes/nodes/ltxv_nodes.py:452-459`. Attention math: `positive×nag_scale − negative×(nag_scale−1)` (line 351-353). No `skip_blocks` parameter exists (an earlier stale CLAUDE.md note had this wrong). KJNodes default is `[11, 0.25, 2.5, True]` but scale=11 is aggressive for LTX 2.3 distilled — with a "still image / deformed / duplicate" negative, extrapolation can blow out-of-distribution and produce zero-update sampler steps (frozen latent). If the initial render freezes or the sampler stalls, dial scale to 3-7 and verify motion returns. Use `scripts/trace_node_source.py <workflow> 508` to print the authoritative schema before editing.
- **ComfyUI-LTXVideo upstream `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json` runs the distilled LoRA on the FULL 22B model**, not the merged distilled checkpoint. Its 15-step `LTXVScheduler` + `MultimodalGuider` stack is NOT authoritative for `ltx-2.3-22b-distilled-1.1.safetensors`. The authoritative distilled path is 8 fixed sigmas + `SimpleDenoiser` (no guidance) per `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/distilled.py`. Don't copy upstream's 15-step sampling stack when running the merged checkpoint.
- **LTX 2.3 resolution must be divisible by 32** (single-stage, our case) or **64** (distilled two-stage) per `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:325`. ComfyUI doesn't enforce; off-grid resolutions silently degrade output. `scripts/validate_workflow_resolution.py` checks. Current default `832x448` (div-by-64).
- **Audio is FROZEN in our workflow** (`noise_mask=0` on audio latent via `SolidMask` + `SetLatentNoiseMask`). LTX's official i2v/t2v prompting guide tells writers to weave audio descriptions alongside actions (`her voice echoes`, `brass swells`, `snare firing`) — that guidance assumes the model is GENERATING audio. We're not; we're providing fixed audio and asking the model to produce video that syncs to it. Text descriptions of audio double-signal what the audio latent already carries and can over-crank visual intensity at music beats. **Strip music/instrumentation references from schedule prompts; keep diegetic ambient sounds** (`wind`, `thunder`, `rain` when present in the visual scene). Action verbs and canonical camera phrases do all the lifting; audio cross-attention binds to the frozen audio latent alone. Validated across `music_prompt*` and `action_prompt*` case studies in `docs/examples/`.
- **Length widget constraint**: `EmptyLTXVLatentVideo.length` must satisfy `(length - 1) % 8 == 0` so pixel frames map cleanly to LTX latent frames. Valid: `1, 9, 17, ..., 241, 249, 257, ..., 497, 505, ...`. Invalid values get auto-rounded UP by ComfyUI silently. If changing `length`, also update `window_size_seconds` (node 688 FloatConstant) to `length / fps` exactly — a mismatch reintroduces the same integer-latent drift the `AudioLoopController` fix eliminated. For 20-iter rapid-cut runs use `length=249 → window=9.96s → stride=8.0s`; for default use `length=497 → window=19.88s → stride=17.92s`.
- **`snap_boundaries=True` (default) means changing `overlap_seconds` does NOT require re-authoring the prompt schedule.** The node rounds schedule timestamps to the actual iteration stride at runtime. One-widget change; no mental math on the grid.
- **CLIP must not enter the loop body.** Per-iteration `CachedTextEncode` forces CLIP into VRAM on every iteration, which triggers `load_models_gpu → free_memory → model.detach(unpatch_weights=True)` (`comfy/model_management.py:664, 690, 764-766`) and evicts the DiT. On reload, `patch_model()` re-injects `object_patches` closures but ComfyUI's `model_patches_to(device)` (`comfy/model_patcher.py:561-580`) migrates `transformer_options["patches"]` only — NOT `object_patches`. LTX2_NAG's captured `nag_cond_video` tensor lives in an `object_patches` closure (KJNodes `ltxv_nodes.py:500-501`) and goes stale across the offload/reload round-trip. Net effect: NAG silently disengages after iter 1 and every suppressed class leaks back (microphones, `still image with no motion`, `deformed hands`, `duplicate character`, style drift). Fix: pre-encode the whole schedule ONCE outside the loop via `TimestampPromptScheduleBatchEncode`, select per-iteration via `ConditioningSelectByIteration`. Mechanism + operational rule: `docs/reference/nag_technical_reference.md`. Forensic root-cause + line-level `model_patcher.py` trace: `docs/analysis/nag_object_patches_offload_asymmetry.md`.
- **Loop-body CONDITIONING must carry `frame_rate` metadata** (default 25.0). The initial-render path stamps it via `LTXVConditioning`; the subgraph's positive path does NOT pass through that node. `TimestampPromptScheduleBatchEncode` stamps it internally via `{**meta, "frame_rate": float(frame_rate)}` on every emitted CONDITIONING. Any NEW CONDITIONING-producing node wired to the loop body must do the same (inline or via `node_helpers.conditioning_set_values(cond, {"frame_rate": 25.0})`) — without the stamp, LTX 2.3's temporal scaling diverges between initial window and loop iterations → identity drift + hallucinated objects (microphones) escalating iter-over-iter regardless of NAG scale or prompt content. Sibling asymmetry (NAG object_patches offload) at `docs/analysis/nag_object_patches_offload_asymmetry.md`.
- **Batch widget edits across example workflows:** `python3 -c` one-liner that loads each JSON, mutates `widgets_values` on the target node type, and writes back with `json.dumps(wf, indent=2) + "\n"`. Preserves structure cleanly.

## ComfyUI gotchas

- **Workflow JSON has two link representations**: node body `"link"` fields AND the `"links"` array. Both must stay in sync.
- Link array: `[link_id, source_node, source_output, target_node, target_input, type]`
- Node `"mode": 0` = active, `"mode": 4` = bypassed.
- **PrimitiveNode cannot feed DynamicCombo sub-inputs.** Set values directly on the widget.
- **TensorLoopClose checks should_stop AFTER the loop body executes.** Handle edge-case inputs gracefully.
- **After changing define_schema(), users must delete and re-add the node in UI.** JSON slot indices are baked at save time.
- **Removing a subgraph component input shifts all higher slot indices.** Decrement `origin_slot` references.
- **ComfyUI execution engine evaluates downstream conditioning graphs before upstream sampling.** Extra nodes in conditioning path can corrupt initial render.
- **torchaudio detect_pitch_frequency on silence gives false positives.** Gate with RMS energy check (< 0.005).
- **`LTXVPreprocess img_compression=0` SKIPS preprocessing entirely** (see `comfy_extras/nodes_lt.py:577-588`). Feeds pristine init image to i2v → model treats as "stay exactly here" → frozen first frames. Use 18 (Lightricks upstream 2.3 value) or 35 (comfy-core default).
- Pyright `reportIncompatibleMethodOverride` on `execute()` is a false positive.
- Module constants must be defined BEFORE functions that reference them (project convention).
- **Scrub workflows before open-sourcing:** filenames, paths, UUIDs, image previews, creative prompts.
- **TensorLoop framework-cache invalidation is transitive.** Any node whose inputs transitively depend on `TensorLoopOpen.current_iteration` (e.g. anything wired through `AudioLoopController`) gets framework-cache-invalidated every iteration even if its OUTPUT values don't change. If you want "run once per workflow" behavior, memoize INSIDE the node on an `id(clip)`-keyed LRU + add `IS_CHANGED` (see `TimestampPromptScheduleBatchEncode` pattern), OR break the dependency chain by wiring from non-loop-dependent sources.
- **`LTXVConcatAVLatent` looks buggy at a glance but isn't.** The `execute()` does `output.update(video_latent); output.update(audio_latent)` (which appears to overwrite the video noise_mask with the audio mask), then IMMEDIATELY overwrites `output["noise_mask"]` and `output["samples"]` with proper `NestedTensor((video, audio))` wrappers. Net result is correct. Source: `<comfyui_extras>/nodes_lt.py:619-651`. Do not chase this — one exploration agent this session misread the control flow and filed a false-positive bug.
- Validate workflow JSON after edits: `python3 -c "import json; json.load(open('file.json'))"`
- **Subgraph schema changes force a UI re-add.** Adding/removing/renaming a subgraph input or output changes the JSON slot indices that ComfyUI bakes into saved node positions. Users who already have the workflow open must delete the subgraph node from the canvas and re-add it for the new schema to take effect. Document this loudly in any apply script that mutates `sg["inputs"]` / `sg["outputs"]` — currently relevant to the planned IC-LoRA Phase 0b migration; not yet triggered by any shipped script.

## Init image conditioning path

- **Initial render** embeds the init image at frame 0 via
  `#531 LTXVImgToVideoInplaceKJ` (KJNodes; encodes image + writes into
  frame 0 of the blank latent in-place, noise_mask locks it).
- **Loop iterations** anchor to the init image via a top-level
  `VAEEncode` → subgraph input slot 8 (LATENT, `guide_latent`) →
  `#1519 LTXVAddLatentGuide.guiding_latent`. `latent_idx=-1` per
  `ComfyUI-LTXVideo/latents.py:411-420` means "conditioning is on the
  frame BEFORE this window" — the model sees the init image as the
  frame immediately preceding the current iteration's window.
- **The init image is VAE-encoded ONCE** at top level (one encode per
  workflow run), not per iteration. The old shape re-encoded every
  loop iteration inside the subgraph — vestigial from the IMAGE-loop
  era — and was removed 2026-04-23 via `scripts/apply_vae_and_cleanup.py`.
- **Subgraph input slot 8 semantics changed 2026-04-23.** Old: IMAGE
  `num_guides.image_1`. New: LATENT `guide_latent`. Workflows saved
  before this date will need re-migration (the apply script is
  idempotent, safe to re-run).

## Subgraph editing

- ALWAYS use WorkflowEditor from `scripts/workflow_utils.py`.
- Structural graph-walking helpers on `WorkflowEditor`: `find_subgraph_invoker(sg_index=0)` (returns the top-level node whose `type` matches the subgraph UUID), `find_input_slot(node, name)` (named-slot lookup), `find_link_to_slot(tgt_node, tgt_slot)` (slot-filtered variant of `find_links_to`). Use these instead of hand-rolling iteration over `wf["nodes"]` / `wf["links"]` in apply scripts.
- **`remove_link` and `remove_subgraph_link` REBIND the target list** (via filter comprehension), they don't mutate in place. Any local variable holding `ed.wf["links"]` or `sg["links"]` goes stale after the first call — re-fetch before appending, or use the editor's methods which re-resolve internally.
- Top-level links: array format `[id, src_node, src_slot, tgt_node, tgt_slot, type]`
- Subgraph internal links: dict format `{id, origin_id, origin_slot, target_id, target_slot, type}`
- Subgraph defs at `wf['definitions']['subgraphs'][0]` with keys: `nodes`, `links`, `inputs`, `outputs`, `widgets`.
- Distributor node ID = -10. Output collector = -20. Both are VIRTUAL -- they do not appear in `sg["nodes"]`. Links with `origin_id == -10` terminate at the distributor; links with `target_id == -20` terminate at `sg["outputs"][target_slot]`.
- **Node-output slots use `"links"` (plural, list). Subgraph boundary entries (`sg["inputs"][i]`, `sg["outputs"][i]`) use `"linkIds"`.** These are two schemas for two locations -- do not conflate them.
- When adding a link into a subgraph output, also update `sg["outputs"][target_slot]["linkIds"]` -- `WorkflowEditor.remove_subgraph_link` handles input-side `linkIds` but output-side is currently manual.
- DynamicCombo widgets: `[num_items, strength_1, strength_2, ..., index_1, index_2, ...]` -- strengths FIRST, then indices. NOT interleaved.

## Testing

```bash
uv run --group dev --group analysis python -m pytest tests/ -v --rootdir=.
```
- `__init__.py` guards ComfyUI-only import with try/except for pytest.
- `nodes.py` has try/except for `comfy_api` with `_IOStub`/`_Passthrough` fallback for test imports.
- `tests/conftest.py` adds `scripts/` + `tests/` to sys.path. Shared MODEL-patcher fakes live in `tests/_fakes.py` (`from _fakes import FakeModelPatcher` / `FakeModelWithCallbacks`; `clone()` deepcopies to mirror production `deepcopy_list_dict`). Don't use `conftest` as the import target -- the root `./conftest.py` (which holds `collect_ignore` for ComfyUI-only imports) shadows `tests/conftest.py` for `from conftest import X`.
- `tests/test_audio_features.py` -- offline analysis (style flag, companion-animal detection, diversity tiers, montage, subdivision, LLM system prompt rules, JSON export shape)
- `tests/test_audio_analysis_nodes.py` -- runtime AudioPitchDetect (9 tests)
- `tests/test_audio_loop_controller.py` -- AudioLoopController integer-latent stride invariants (20 tests — zero-drift across overlap values 0-5s, effective-vs-target overlap reporting, edge cases)
- `tests/test_keyframe_nodes.py` -- KeyframeImageSchedule, VideoFrameExtract, ImageBlend (28 tests)
- `tests/test_cache_nodes.py` -- CachedTextEncode LRU, IterationCleanup modes (13 tests)
- `tests/test_profile_nodes.py` -- ProfileBegin/ProfileIterStep/ProfileEnd disabled paths (7 tests)
- `tests/test_schedule_snapping.py` -- TimestampPromptSchedule snap + raised-cosine blend (20 tests)
- `tests/test_batch_encode.py` -- `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` parity, dedup, clamp, end-to-end integration asserting CLIP encode called exactly once per unique prompt (12 tests)
- `tests/test_decoder_validator.py` -- DR1 decoder widget alignment (6 tests)
- `tests/test_workflows.py` -- workflow JSON structural validation (parametrized over all example_workflows/*.json)
- Total: 220 tests (2026-04-22).
- **Test shape for memoization fixes.** When a fix's correctness depends on "node runs once across the loop", the red test MUST exercise REPEATED `execute()` calls with identical inputs — single-call tests can't detect framework-cache-invalidation. Canonical shape: `tests/test_batch_encode.py::TestBatchEncoderCaching`.
- **`id()`-keyed module caches need an autouse clear-fixture in their test files.** `_COND_CACHE`, `_BATCH_ENCODE_CACHE` key on `id(clip)`. Production CLIP is 15 GB and never recycles; pytest's tiny `FakeCLIP` gets GC'd rapidly and Python address recycling produces ghost hits across tests. Pattern: autouse fixture calling `nodes._CACHE_NAME.clear()` before + after each test (see `tests/test_batch_encode.py:17`).

## Dependencies

Companion custom nodes (not imported, used alongside in workflows):
- **Sage attention fork at `<sage_fork_repo>/`** -- our fork of `woct0rdho/SageAttention` (which forks `thu-ml/SageAttention`). Kernel-side changes land there; consumer-side routing is in `nodes_sage.py`. Rebuild with `<sage_fork_repo>/build.sh` (hardened to install into explicit `VIRTUAL_ENV`). Cross-repo state in `internal/design/sage_backlog.md`; fork has its own CHANGELOG + "Open work" backlog.
- ComfyUI-NativeLooping_testing -- TensorLoopOpen/Close
- ComfyUI-LTXVideo -- LTXVAddLatentGuide, LTXVCropGuides, LTXVPreprocess, LTXVTiledVAEDecode (default decoder in our example workflows — spatial-only tiling, no temporal-tile seams; swap via `scripts/apply_ltx_decoder.py --revert` if you need the generic VAEDecodeTiled fallback)
- ComfyUI-KJNodes -- Set/Get nodes, FloatConstant, LTX2_NAG, LTXVImgToVideoInplaceKJ, LTXVAddGuideMulti (multi-guide, up to 20), LTXVAddGuidesFromBatch
- ComfyUI-VideoHelperSuite -- VHS_VideoCombine
- ComfyUI-MelBandRoFormer -- vocal separation (hardcoded `dim=384, depth=6, num_stems=1`). **Bypassed by default** (`mode=4`) in every shipped workflow as of 2026-04-22, AND `Set_actual_audio` is explicitly wired from `TrimAudioDuration`(567) directly rather than through the bypassed sampler — explicit wiring makes the graph readable without relying on ComfyUI's bypass-passthrough slot mapping. Re-enabling separation is a two-step manual edit: flip either MelBand node back to `mode=0` AND re-route the `Set_actual_audio` link through `MelBandRoFormerSampler`(569) output 0. Worthwhile for clean-vocal tracks where lip-sync benefits; overhead for instrumental-only or speech-only. Re-apply defaults via `scripts/apply_melband_default_off.py`.

### Dependency boundary
- **Offline scripts** (scripts/): librosa allowed via optional `analysis` dep group.
- **Runtime nodes** (nodes.py, nodes_analysis.py): torchaudio only, zero extra deps. All outputs FLOAT or INT.
- AudioPitchDetect.vocal_fraction wires directly to ConditioningBlend.blend_factor for audio-reactive blending.

## Audio analysis scripts

- `scripts/analyze_audio.py` -- ffmpeg-only energy/structure detection (no Python deps)
- `scripts/analyze_audio_features.py` -- librosa: BPM, key, vocal F0, structure, JSON for LLM prompt generation
- `scripts/spectrogram_to_reference.py` -- renders a Mel spectrogram as a PNG frame sequence for IC-LoRA spectrogram-as-reference experiments (Phase 2.0 PoC). Pure-function core: `compute_mel_log`, `prepare_mel_for_render` (global normalization runs ONCE, preserves beat-amplitude signal — do not switch to per-frame), `render_frame`, `render_spectrogram_frame` (test wrapper). Render modes: `raw` / `normalized` / `blurred` (default) / `edge_detected`. Output lands in `internal/scratch/spectrogram_runs/<ts>/` with `metadata.json` + wiring README. Design + iteration ladder: `internal/design/spectrogram_reference_design.md`. Tests: `tests/test_spectrogram_lib.py`.
- CLI flags: `--scene-diversity <tier><sub>` (default `2a`; tiers 1-6 performance_live → avant_garde; sub-letters add mood bundles). `--montage` (orthogonal; ~12s dwell, emotional-arc language, Arcane-style pacing). `--style <cinematic|realistic|illustrated|painterly|animated|none>` (default `cinematic`; match init-image style family to avoid photoreal drift — see Critical Constraints).
- Long sections auto-subdivided (~20s default, ~12s in montage mode) so a 3-min song yields 7+ entries instead of 4-5.
- JSON export (`-j`) includes `llm_system_prompt` with HARD RULES R1-R9 (R9 = snap timestamps to `stride_seconds` grid; R7 = canonical camera list + no-dolly-out), an INFERENCE block (init image commits style/palette/setting/subjects — schedule drives camera/body/lighting/cuts/arc), tier semantics, and three worked examples. `workflow_context` surfaces `style`, `scene_diversity`, `scene_diversity_tier_name`, `scene_diversity_mood_bundle`, `montage`, `overlap_seconds_target` (widget value) + `overlap_seconds_effective` (post-integer-latent quantization), and `stride_seconds` (effective, matches `AudioLoopController`'s quantized stride). Paste into Claude/Gemini.
- Full guide: `docs/guides/audio_analysis_guide.md`; LLM integration: `docs/guides/prompt_workflow_end_to_end.md` (System prompt reference §)

## Debugging workflow regressions

Compare against known-working workflow JSON (keep copies in `internal/scratch/`).
Change ONE setting at a time. Run `uv run --group dev --group analysis python -m pytest tests/test_workflows.py --rootdir=.` after every edit. `scripts/validate_workflow_resolution.py` additionally checks LTX-compliant div-by-32/64 dimensions.
LTX-2_00032.json and LTX-2_00040.json are confirmed working (April 9, 2026). For session-specific symptom→fix recipes see `docs/guides/debugging_guide.md`.

### Debug tools (when a workflow regression needs forensics)

- `uv run --group dev python scripts/audit_workflows.py [--verbose]` — health audit across every `example_workflows/*.json`: sage node+mode, `LoopIterationStamp` presence, batch-encode vs legacy prompt path, distilled sigma chain (`linear_quadratic 8 1` + `shift=13` + `euler` + `cfg=1`, STG exception handled), resolution div-32, `(L-1)%8==0`, `LTXVPreprocess img_compression >= 18`, `LTXVTiledVAEDecode` preferred. Exits 1 on any ERR; WARNs don't fail. Run after any bulk workflow edit; cheap (<100 ms).
- `uv run --group dev python scripts/trace_node_source.py <workflow> <node_id> --include-inputs` — resolve any node to its Python source (AST-extracted class body + bounded call graph) + workflow-level wiring. Flags `add_object_patch` closures, captured tensors, mode=4 bypasses, widget overrides on wired inputs. **Run this before trusting any widget annotation in CLAUDE.md** — saved this session from a stale `LTX2_NAG` widget-order error.
- `uv run --group dev python scripts/analyze_workflow_dag.py <workflow> --format <ascii|mermaid|dot|json>` — topo-sorted execution order + DAG rendering. `--subgraph 0` pulls loop-body internals into the same graph. Answers "what runs in what order" without executing anything.
- `COMFYUI_EXEC_LOG=/tmp/exec.jsonl python <comfyui>/main.py` — runtime per-node JSONL log (start/end/error events, timings, input/output tensor shapes). Installed from `exec_logger.py` via `__init__.py`. Zero overhead when env var is unset. Use when "which node is frozen/slow/crashing" is the question.
- **All debug outputs land in `internal/analysis/runs/`** (gitignored) when you use `--save-run` (DAG analyzer), `COMFYUI_EXEC_LOG=auto` (exec logger), or the default `ProfileBegin.output_dir` (torch.profiler traces land under `runs/profiler/<timestamp>/`). Timestamped filenames so successive runs can be diffed. Shared helper `timestamped_run_path()` in `scripts/workflow_utils.py` — use it when adding a new debug tool.
- **Apply scripts take an optional workflow path.** `apply_batch_encode_fix.py`, `apply_melband_default_off.py`, `apply_profiling_nodes.py` all default to `example_workflows/audio-loop-music-video_latent.json` but accept a CLI arg — useful for staging changes on `internal/scratch/` test workflows first.
- **Experimental apply scripts stage to `internal/scratch/<base>_<feature>_<phase>.json`.** `apply_iclora_initial_render.py` is the canonical example: `--input` defaults to canonical `example_workflows/_latent.json`, `--output` defaults to `internal/scratch/audio-loop-music-video_latent_iclora_phase0a.json`. Idempotent on the output path (re-running on an already-migrated file is a no-op); `--revert` deletes the staging file. Promotion to `example_workflows/` follows the "feature ships AND stabilizes" rule — validation criteria are feature-specific and documented in `internal/PLAN.md` + `internal/ic_lora_assessment.md`.
- `scripts/verify_sage_iteration_trace.sh [path]` — reads the latest sage JSONL under `internal/analysis/runs/sage/` (lexicographic sort on filename; `timestamped_run_path()` format bakes the timestamp into the name). Emits one JSON doc with `summary`, `stamp_missing`, and `per_iteration` (grouped by iter with per-kernel + fallback counts). Answers backlog item 7: "does the sage override survive model offload across loop iterations?" `AUDIOLOOPHELPER_SAGE_TRACE=auto` is the default in `<comfyui>/start.sh` so every render emits a trace; unset once the verification closes.
- **When iter-over-iter drift appears and NAG-scale toggling + prompt-content changes don't fix it, trace CONDITIONING paths in parallel.** Walk both the initial-render sampler's conditioning chain and the loop-body sampler's chain back to their CLIP encoders. Asymmetries (one path passing through `LTXVConditioning` + the other not; different `frame_rate` values; CLIP-producing node inside the loop subgraph) are load-bearing bugs. The 2026-04-23 `frame_rate` metadata regression was found this way.

## Documentation conventions

- **Active planning lives in gitignored `internal/`.** Feature roadmaps,
  validation plans, Phase-N decision trees → `internal/PLAN.md` (or
  equivalent). Promote a doc to `docs/` only when the feature ships
  AND stabilizes. This prevents the "stale plan orphaned in user docs"
  class that produced `docs/PLAN.md` pre-2026-04-21.
- **Case studies live in pairs.** Unscrubbed working version in
  `internal/prompts/` or `internal/postmortem_*.md`; scrubbed public
  version in `docs/examples/` or `docs/guides/debugging_guide.md` as a "Case
  studies" entry. Privacy scrub removes filenames, absolute paths
  under `/home/`, username, UUIDs, real asset names.
- **Breaking changes trigger a docs sweep.** When a change alters a
  formula, value, or constraint that's referenced in prose (not just
  code), add the old-value phrase to
  `scripts/validate_docs_consistency.py`'s `STALE_PATTERNS`. The test
  in `tests/test_docs_consistency.py` fails loudly until every stale
  claim is updated or marked with a `HISTORICAL_MARKERS` substring
  ("pre-YYYY-MM-DD", "(continuous seconds)", etc.).
- **Last-updated date at top of every doc.** Format: `Last updated: YYYY-MM-DD`.
- **Trim public + archive full** for reference docs that grow past
  ~1000 lines but still carry widget-level detail someone might need.
  Public version in `docs/reference/` becomes a summary; full version
  moves to `internal/archive/<name>_full.md` (gitignored). Applied to
  `pipeline_flow_image.md` 2026-04-23; `pipeline_flow_latent.md` is
  the next candidate.
- **CHANGELOG historical entries stay as-is.** When moving or renaming
  docs, do NOT rewrite historical `CHANGELOG.md` paths — they reflect
  truth at the time of the original entry. Only add a new `[Unreleased]`
  entry describing the reorg.
- **internal skill state is gitignored.** Any Claude Code hooks/agents/skills placed there are local-only. Team-shareable automations belong in `scripts/` or a tracked plugin.

## Documentation index

Layout: `docs/README.md` is the authoritative task-first nav — start there for anything not in the entry points below. `docs/guides/` = how-to, `docs/reference/` = deep-dive, `docs/analysis/` = research/postmortems, `docs/examples/` = scrubbed case studies.

### Entry points
- `docs/README.md` -- nav index (task → doc).
- `docs/architecture_overview.md` -- **START HERE.** Single-entry-point reference covering our workflow, ComfyUI core execution, ComfyUI-LTXVideo + KJNodes layers, native LTX-2 portability, CLIP path, sampler + mask routing, known bugs, extension playbook. Navigable in one pass with cross-references to every deeper doc.

### Guides (how-to, in `docs/guides/`)
- `docs/guides/prompt_workflow_end_to_end.md` -- complete pipeline: init image -> VLM -> audio analysis -> LLM -> workflow
- `docs/guides/prompt_creation_guide.md` -- prompt rules, variation patterns (A/B/C), sampler tuning, examples
- `docs/guides/audio_analysis_guide.md` -- offline/runtime analysis, AudioPitchDetect wiring patterns
- `docs/guides/debugging_guide.md` -- symptom-first troubleshooting. First stop when output looks wrong.
- `docs/guides/profiling_guide.md` -- torch.profiler three-node integration (opt-in)
- (removed 2026-04-23: upscale design doc moved to `internal/design/upscale_workflow_design.md` — promote back to `docs/guides/upscale_guide.md` when the workflow actually ships)
- `docs/reference/ltxv_looping_sampler_reference.md` -- LTXVLoopingSampler structural reference (video-only; trimmed from the prior build guide — we don't recommend building this for music video)

### Reference (deep-dive, in `docs/reference/`)
- `docs/reference/ltx23_model_reference.md` -- image guides, latent volume, VAE conversion, AdaIN, conditioning path, noise_mask, dual workflow, extension subgraph, upscaling
- `docs/reference/sampler_reference.md` -- `euler` vs `euler_ancestral` vs `euler_ancestral_cfg_pp` with ComfyUI + MultimodalGuider source code walkthrough; why `euler` is mandatory for our loop architecture and why upstream's `euler_ancestral_cfg_pp` is wrong for merged distilled-1.1
- `docs/reference/nag_technical_reference.md` -- NAG (Normalized Attention Guidance)
- `docs/reference/pipeline_flow_image.md` -- IMAGE workflow node-by-node trace
- `docs/reference/pipeline_flow_latent.md` -- LATENT workflow node-by-node trace
- `docs/reference/ltx23_prompt_system_prompts.md` -- official i2v/t2v system prompts (raw Lightricks; project-specific rules live in `docs/guides/prompt_creation_guide.md`)
- `docs/reference/standup_system_prompt.md` -- LLM system prompt for standup/dialogue schedule generation (renamed from `docs/system_prompt.md` 2026-04-23)

### Analysis and research
- `docs/analysis/ltx23_gaps_analysis.md` -- capability gaps, LTXVLoopingSampler AV incompatibility
- `docs/analysis/audio_in_prompt_research.md` -- community lip-sync prompting research (consolidated 2026-04-23; framed for when it applies vs when our frozen-audio workflow diverges)
- `docs/analysis/ltx2_native_conditioning_analysis.md` -- 3 conditioning types, MultiModalGuiderFactory per-sigma guidance
- `docs/analysis/ltx_desktop_conditioning_analysis.md` -- ModalitySpec, TemporalRegionMask (retake), frozen modality
- `docs/analysis/comfyui_ltxvideo_multiframe_guide_analysis.md` -- guide chaining, LTXVAddGuide* hierarchy
- `docs/analysis/kjnodes_multiframe_guide_analysis.md` -- LTXVAddGuideMulti (up to 20 guides), LTXVAddGuidesFromBatch
- `docs/analysis/nag_object_patches_offload_asymmetry.md` -- why CLIP cannot enter the loop body (the 2026-04-22 root cause behind per-iter microphones/anatomy/motion regressions)

### Prompt schedule examples (public)
- `docs/examples/README.md` -- index of all case studies with patterns-that-transfer summary
- `docs/examples/music_prompt*.md` -- vocal-driven music videos (v1-v3 illustrated→cinematic)
- `docs/examples/action_prompt*.md` -- instrumental action sequences (v1-v6, including v5's 20-iter rapid-cut architecture and v6's frozen-audio insight)
- `docs/examples/prompt_comedy*.md` -- standup/dialogue (v4 introduced "Cut to ..." iteration-boundary technique; v5 covers unusual-character init-image adaptation)
- Scrubbed copies; working versions with actual asset names live in `internal/prompts/` (gitignored)

### Example workflows
- `example_workflows/audio-loop-music-video_image.json` -- IMAGE loop (per-iteration AdaIN)
- `example_workflows/audio-loop-music-video_latent.json` -- LATENT loop (per-iteration AdaIN). Primary working baseline. As of 2026-04-22 uses `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` for schedule handling (the per-iter `CachedTextEncode`+`ConditioningBlend` chain was removed because CLIP-in-loop was silencing NAG; see Critical constraints). Migrate other variants via `scripts/apply_batch_encode_fix.py`.
- `example_workflows/audio-loop-music-video_latent_keyframe.json` -- LATENT loop + per-iteration keyframe image schedule via `KeyframeImageSchedule` + `ImageBlend` (instead of constant `Get_input_image`).
- `example_workflows/audio-loop-music-video_latent_stg.json` -- LATENT loop with STG-hybrid sampling: preserves authoritative distilled-1.1 sigma schedule (`linear_quadratic, 8, 1` + shift=13) but swaps `CFGGuider` for `MultimodalGuider` + `GuiderParameters` (cfg=2, stg=1 on both modalities) for STG quality lift. `cfg=2` rather than the originally-designed `cfg=1` because `MultimodalGuider` has an unbound-variable bug at `multimodal_guider.py:269` when `cfg=1.0` on both modalities (uncond branch not run, `noise_pred_neg` referenced anyway). NAG bypassed. Use `KSamplerSelect: euler` (not `euler_ancestral_cfg_pp`) — ancestral re-noise compounds iteration drift in the loop architecture regardless of CFG++; see `docs/reference/sampler_reference.md`. Built via `scripts/apply_stg_hybrid_package.py`. A/B target against the baseline `_latent.json`.
- `example_workflows/audio-loop-music-video_image_adain_perstep.json` -- IMAGE + per-step AdaIN (experimental)
- `example_workflows/upscale-loop-output.json` -- separate upscale workflow (when built)

### Reference codebases (read-only, for comparing implementations)
- `coderef/LTX-2/` -- LTX-2 native. Conditioning types at `packages/ltx-core/src/ltx_core/conditioning/types/`, pipelines at `packages/ltx-pipelines/src/ltx_pipelines/`
- `coderef/LTX-Desktop/` -- Lightricks Desktop app. A2V / retake / IC-LoRA at `backend/services/`
- `<comfyui_custom_nodes>/ComfyUI-LTXVideo/` -- ComfyUI LTX integration. Guide nodes at `guide.py`, `latents.py`, samplers at `looping_sampler.py`

### Internal (gitignored)
- `internal/PLAN.md` -- active planning roadmap (Phase 1 validation pending)
- `internal/postmortem_v0408_session.md` -- debugging history (6 issues)
- `internal/postmortem_v0409_latent_rework.md` -- latent rework (5 issues, noise_mask root cause)
- `internal/audio_analysis_evolution.md` -- original critique of the audio→prompt pipeline (heuristic vs learned gap)
- `internal/prompts/` -- unscrubbed working versions of the `docs/examples/` case studies plus legacy prompt drafts
- `internal/analysis/` -- deep-dive investigations (frozen initial render, NAG object_patches asymmetry, reference-workflow comparison). `internal/analysis/runs/` holds timestamped debug-tool outputs from `analyze_workflow_dag.py --save-run`, `COMFYUI_EXEC_LOG=auto`, and `ProfileBegin`.
