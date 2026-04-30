# ComfyUI-AudioLoopHelper

Last updated: 2026-04-30

ComfyUI nodes that automate loop timing + audio analysis for full-length music video generation with LTX 2.3. Core pattern: `AudioLoopController` drives stride from integer latent counts, audio is frozen via `noise_mask=0`, prompts pre-encoded once outside the loop (CLIP must never enter the loop body). **Start here:** `docs/architecture_overview.md`; task-first nav at `docs/README.md`.

## Architecture

Runtime files: `nodes.py` (core loop), `nodes_analysis.py` (torchaudio audio analysis), `nodes_sage.py` (sage attention), `nodes_validation.py` (config validator). Entry point: `comfy_entrypoint()` in `nodes.py`.

Core nodes (per-node role + wiring in each class's docstring; full reference at `docs/reference/ltx23_model_reference.md`):

- **Loop spine**: `AudioLoopController`, `LoopIterationStamp`, `IterationCleanup`, `AudioLoopPlanner`, `AudioDuration`
- **Prompt schedule**: `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` (current) / `TimestampPromptSchedule` + `CachedTextEncode` (legacy; don't wire in loop body)
- **Keyframe schedule**: `KeyframeLatentScheduleBatchEncode` + `LatentSelectByIteration` (current — VAE-encodes once outside loop) / `KeyframeImageSchedule` + `ImageBlend` (legacy; per-iter VAE)
- **Latent ops**: `LatentContextExtract`, `LatentOverlapTrim`, `StripLatentNoiseMask`, `LatentTemporalMask` (retake)
- **Image path**: `KeyframeImageSchedule`, `ImageBlend`, `VideoFrameExtract`
- **Conditioning blend**: `ConditioningBlend` (works with Gemma 3 + CLIP)
- **Attention + profiling**: `AudioLoopHelperSageAttention` (default `auto_mask_aware`), `ProfileBegin`/`IterStep`/`End`
- **Step-skipping cache**: `LTXVideoEasyCache` (experimental, default off). Patches LTX denoiser via `WrappersMP.DIFFUSION_MODEL`. Single threshold knob; `cache_device` offload to CPU optional. Reference: `nodes_easycache.py`. Telemetry/privacy story: `docs/reference/telemetry_and_tracing.md`.

Analysis (`nodes_analysis.py`, torchaudio only): `AudioPitchDetect` → F0 + vocal-fraction; pairs directly with `ConditioningBlend.blend_factor`.

## Key patterns

- `AUDIO = {"waveform": Tensor, "sample_rate": int}`. Duration = `waveform.shape[-1] / sample_rate`.
- **Stride derived from integer-latent counts**, not widget seconds: `stride_seconds = (window_latents - overlap_latents) * 8 / fps`. The `overlap_seconds` widget is a TARGET; node emits EFFECTIVE quantized overlap. Eliminates lip-sync drift across overlap values. Tests: `tests/test_audio_loop_controller.py`.
- `start_index` clamps so ≥0.5s audio remains on final iter (prevents mel crash).
- LTX 2.3 text encoder is Gemma 3, NOT CLIP. Format: `[tensor, {"attention_mask": mask}]`, no pooled.
- **Audio path is sacred.** `LTXVAudioVAEEncode → LTXVConcatAVLatent`; never feed visualizations into the video latent.
- Video VAE formula: `latent = (pixel - 1) // 8 + 1`. Not `pixel // 8`.
- `noise_mask=0` = fixed context; `mask=1` = regenerate. Audio is 0; video is 1.
- Guide chaining: multiple `LTXVAddLatentGuide` / `LTXVAddGuideMulti` (up to 20) accumulate via `keyframe_idxs`; `LTXVCropGuides` strips them.
- **CFG-analog amplification of any conditional contribution**: feed `(positive_with_X, positive_without_X)` to `CFGGuider` as `(positive, negative)`. Existing sampler computes `eps = eps_without + cfg * (eps_with - eps_without)` per step. Zero new sampler code; generalizes beyond IC-LoRA to any conditional (style LoRAs, identity LoRAs, per-reference ablation). Distinct from control-vector / concept-slider techniques (static directions) — this is dynamic per-step differential. Canonical POC: `scripts/apply_ttc_iclora_amplification_poc.py`. Landscape: `internal/analysis/iclora_landscape_analysis.md` §TTC.

## Critical constraints

- **Never feed audio visualizations into video latent** — heatmap frames result.
- **`LTXVAudioVideoMask` (Node 606) wiring is intentional** — `audio_start_time = audio_end_time = window_size` (empty range keeps audio fixed). Don't change.
- **Use `LatentContextExtract` / `LatentOverlapTrim`**, not raw `LTXVSelectLatents` — they strip `noise_mask` automatically.
- **Node 169 prompt matches schedule 0:00 entry** structurally (`_build_prompt_for_section` via shared `_prepare_sections`; byte-exact).
- **Every prompt must contain "singing"** (or "are singing together"). LTX 2.3 audio-video cross-attention binds lip sync to the action verb.
- **Use `In a [shot], [camera]` continuation framing for non-first entries — NOT `Cut to a ...`.** Lightricks's official LTX 2.3 system prompt explicitly trains the model to treat scene-cut language as a discontinuation directive, fighting the loop's continuity mechanisms (`LTXVAddLatentGuide latent_idx=-1` + `LatentContextExtract` 1s overlap). Convention retracted 2026-04-25. Canonical guide: `docs/guides/prompt_creation_guide.md` §5.1.
- **Always use `WorkflowEditor`** (`scripts/workflow_utils.py`) for JSON edits.
- **Distilled 8-step sigmas**: `ManualSigmas "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"` + `ModelSamplingSD3 shift=13` + `KSamplerSelect euler` + `CFGGuider cfg=1`. Decoder: `LTXVTiledVAEDecode [1,1,1,true,"auto","auto"]` on **24GB+** (single-tile, ~3× faster cold-pass than [2,2,1]); fall back to [2,2,1] on ≤16GB. The 9 sigma values are Lightricks's hand-tuned `DISTILLED_SIGMA_VALUES` from `coderef/ID-LoRA-2.3/packages/ltx-pipelines/utils/constants.py` — what their distilled checkpoint was trained to denoise. Pre-2026-04-27 we used `BasicScheduler linear_quadratic 8 1` which approximated this curve parametrically; migration: `scripts/apply_canonical_sigmas.py`. **Don't use `euler_ancestral*`** — Lightricks's own distilled inference uses plain `EulerDiffusionStep` (`coderef/ID-LoRA-2.3/.../diffusion_steps.py`), and the 4-step plateau near σ≈0.99 amplifies ancestral re-noise enough to bleed across our TensorLoop iteration boundaries. Full walkthrough: `docs/reference/sampler_reference.md`.
- **Illustrated inits drift toward photoreal across iterations** (cross-attention is photoreal-trained). Match init-image style family; or re-anchor via `LTXVAddGuideMulti` per iteration.
- **LTX2_NAG widgets** `[nag_scale, nag_alpha, nag_tau, inplace]`. KJNodes default `scale=11` is aggressive for distilled — dial to 3-7 if initial render freezes. Always verify schema via `scripts/trace_node_source.py <wf> 508`. Reference: `docs/reference/nag_technical_reference.md`.
- **Don't copy upstream's 15-step sampling** from `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`. Authoritative distilled path: 8 fixed sigmas per `coderef/LTX-2/.../distilled.py`.
- **Resolution div-by-32** (single-stage) or **div-by-64** (two-stage). `scripts/audit_workflows.py` checks.
- **Audio is FROZEN in our workflow.** Strip music/instrumentation references from schedule prompts; keep diegetic sounds only. Rationale: `docs/analysis/audio_in_prompt_research.md`; case studies in `internal/prompts/` (gitignored).
- **Dimension config flows from `LTXFramePlanner` (single source of truth).** It snaps `target_width`/`target_height` to div-32, `target_seconds` to `(frames - 1) % 8 == 0`, computes latent volume vs the artifact ceiling, and emits matched `frame_rate` / `window_seconds` / `fps` outputs. All shipped workflows wire it to `EmptyLTXVLatentVideo` (width/height/length), `ImageResizeKJv2` (width/height), `LTXVConditioning` (frame_rate), `AudioLoopController`/`AudioLoopPlanner` (window_seconds, fps), and the subgraph (`video_end_time`). Migration: `scripts/apply_frame_planner_consolidation.py`. Audit: `frame_planner_present` (ERR if missing on production workflows; WARN on experimental forks). The previous manual rule "match `window_size_seconds = length / fps` exactly" is now machine-enforced.
- **`snap_boundaries=True`** (default) lets `overlap_seconds` change without schedule re-authoring.
- **CLIP must not enter the loop body.** Pre-encode via `TimestampPromptScheduleBatchEncode`; `object_patches` don't survive the offload/reload → silent NAG disengagement iter 2+. Mechanism: `docs/analysis/nag_object_patches_offload_asymmetry.md`.
- **Loop-body CONDITIONING must carry `frame_rate`** (default 25.0). Batch encoder stamps it; any new CONDITIONING-producing loop-body node must too (via `node_helpers.conditioning_set_values`). Missing → identity drift + hallucinated objects iter-over-iter.
- **Bake new topology constraints into `scripts/audit_workflows.py`.** Every fix that ships an apply script should ship a matching audit check (ERR status with a `Run scripts/apply_X.py` remediation pointer). Canonical pairs: F2 (`preprocess_symmetry`), F3 (`loop_cropguides_symmetry`), F4 (`alc_seed_legacy_name`), F5 (`iterations_autowired`), F6 (`alc_widget_drift`), F7 (`planner_no_stride_input`), F8 (`frame_planner_present`), F9 (`ltx2_nag_reaches_loop`), F10 (`vae_decode_no_tile` — WARN-level since [2,2,1] is safe fallback for ≤16GB), F11 (`dead_lora_loader_scaffolding_absent` — strips bypassed `#1625/#1626/#1627` LoRA placeholders from canonical), F12 (`iclora_video_reference_guide_in_loop_with_cropguides` + `iclora_loader_present_when_guide_present` + `iclora_ref_video_preprocess_symmetry` — paired with `apply_iclora_video_reference.py` for in-loop video-reference IC-LoRA wiring; mirrors F2/F3 patterns onto the ref-video chain). Prevents silent regression of fixes a sibling branch might revert.
- **VAE decode is single-tile `[1,1,1]` on 24GB cards.** Tiled decode pays per-tile prepare/stage overhead (`Model VideoVAE prepared for dynamic VRAM loading.`) that exceeds activation savings on large-VRAM cards. Empirical (832×448×497, 24GB sm89, 2026-04-27): `[2,2,1]` cold = 143s, `[1,1,1]` cold = 47s, `LTXVSpatioTemporalTiledVAEDecode` cold = 61s. Single-tile wins on cold AND warm. Apply: `scripts/apply_no_tile_vae_decode.py` (idempotent + reversible). Audit: `vae_decode_no_tile`. **Revert if on ≤16GB** — single-tile decode of 832×448×497 may OOM there.
- **A schema rename is not enough — strip leftover widget values too.** When `apply_alc_seed_rename.py` (1f6b830) renamed `seed` → `base_seed` to defuse the `control_after_generate` dropdown trap, it updated `inputs[].name` but did NOT prune the leftover `'randomize'` string at `widgets_values[4]` that the old dropdown had baked in. The frontend stops re-attaching the dropdown (no input named `seed`/`noise_seed` anymore) so nothing rewrites the widget — but ComfyUI's backend still pops widgets positionally from the saved list, and 6 saved values into 5 schema slots shifts `'randomize'` into the `fps` slot, INT-parse fails. Companion strip migration: `scripts/apply_strip_alc_control_after_generate.py`. Audit: `alc_widget_drift`. Diagnosed 2026-04-27.
- **Don't ship two schema changes that both touch the same iteration-state plane in one session.** The 2026-04-26 `apply_iterations_autowire.py` (`AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in`) created a back-edge against the existing `AudioLoopController.stride_seconds → AudioLoopPlanner.stride_seconds` and `TensorLoopOpen.current_iteration → AudioLoopController.current_iteration` edges. ComfyUI's prompt validator rejected the workflow with "Dependency cycle detected" before any node ran. Fix: `AudioLoopPlanner` now derives stride internally via `_compute_loop_geometry` (matching controller), eliminating the controller→planner edge. Migration: `scripts/apply_planner_break_stride_cycle.py`. Audit: `planner_no_stride_input`. Lesson: when adding an auto-wire that closes a control loop, walk every existing edge between the involved nodes and confirm none of them produces a cycle. Diagnosed 2026-04-27.
- **Authoritative LTX 2.3 prompting evidence**: `docs/reference/ltx23_prompt_system_prompts.md:44, 56, 93` (Lightricks's own i2v + t2v system prompts: "DO NOT describe scene cuts", "Inaccurate descriptions may cause scene cuts"). What retracted our `Cut to` convention 2026-04-25. Check before relitigating any prompt-rule debate.
- **Never name an INT widget exactly `"seed"` or `"noise_seed"`.** ComfyUI's frontend auto-attaches a `control_after_generate` dropdown to those literal names, which silently mutates the saved widget value across runs even when the input is wired (link supersedes widget at execute time, but the mutated widget still gets serialized — saved JSONs drift across renders despite reproducible runtime seeds). Use `base_seed`, `seed_in`, etc. Guard: `tests/test_node_schemas.py::test_no_seed_or_noise_seed_named_inputs` AST-walks every `io.*.Input(...)` call. Diagnosed 2026-04-26 in `internal/analysis/id_lora_ablation_and_seed_widget_audit.md`.
- **Iterations auto-track audio length.** `AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in` is wired in every shipped workflow (added 2026-04-26 via `scripts/apply_iterations_autowire.py` + an upstream `ComfyUI-NativeLooping_testing` schema patch that made `iterations_in` a wireable optional input). User puts in any audio, loop runs exactly the iterations needed. For short tests, drag in an `INTConstant` and rewire — recipe in `docs/guides/debugging_guide.md`. Audit: `audit_workflows.py::iterations_autowired` (ERR if unwired in shipped workflows).

## ComfyUI gotchas

- Workflow JSON has two link representations: node-body `"link"` fields AND top-level `"links"` array. Both must sync.
- Link array: `[link_id, src, src_slot, tgt, tgt_slot, type]`.
- **Workflow JSON references inputs by NAME, not slot index.** Each node's `inputs[]` entry stores `{"name": ..., "type": ..., "widget": {"name": ...}, "link": ...}`; ComfyUI matches the saved name to the schema's input list when reattaching wires. So a bare schema rename (e.g. `"seed"` → `"base_seed"`) without a paired migration script that rewrites `inputs[].name` and `widget.name` in every saved JSON will dangle every existing wire on the renamed input. Canonical migration: `scripts/apply_alc_seed_rename.py`.
- `"mode": 0` = active, `"mode": 4` = bypassed. **Bypass passes inputs to outputs of same TYPE only**; inputs with no matching-type output dead-end silently. E.g., bypassing `LTXAddVideoICLoRAGuide` leaves its `image` input unconsumed. Verify truly-inert bypass by swapping the upstream input and byte-diffing outputs (`md5sum` on sampled frames, `wave` on decoded audio).
- **`workflow_utils.is_active(node)`** is the canonical bypass check (`mode != 4`). Use it instead of inline `node.get("mode", 0) != 4` — 5 call sites across `audit_workflows.py`, `apply_no_tile_vae_decode.py`, `apply_melband_default_off.py`. The bare integer obscures that `4` means bypass.
- **Dead-node detection requires live-consumer check, not link-count check.** A node with output links can still be runtime-dead if every consumer is `mode=4` (bypassed). Pattern: walk consumer ids, return True only if at least one consumer satisfies `is_active`. Inline `any(o.get("links") for o in n.get("outputs"))` misses bypassed-consumer chains (e.g. `#1318 → bypassed-#560` was effectively dead despite having a link). See `apply_no_tile_vae_decode.py::_has_live_consumer`.
- **ComfyUI exposes the active prompt's id via `comfy_execution.utils.get_executing_context().prompt_id`** (a contextvar, not `transformer_options`). For per-prompt telemetry / routing, lazy-import in the call path (try/except ImportError so non-ComfyUI test environments don't break). Pattern at `nodes_sage.py:541-559`.
- `PrimitiveNode` can't feed `DynamicCombo` sub-inputs — set on the widget directly.
- `TensorLoopClose` checks `should_stop` AFTER the body; handle edge inputs.
- **Subgraph schema changes force a UI re-add** (slot indices baked at save time). Same for any `define_schema()` change.
- Removing a subgraph input shifts higher slot indices — decrement `origin_slot` refs.
- ComfyUI evaluates downstream conditioning before upstream sampling → extra nodes in conditioning path can corrupt initial render.
- **`CLIPTextEncode(169) → ConditioningZeroOut(420) → LTXVConditioning(164).negative → CFGGuider(153).negative` chain is wired-correctly but runtime-inert at `CFG=1`** (sampler computes `eps = eps_positive` only). Don't try to remove it — `CFGGuider` validates both `positive` and `negative` input slots; removing 169 or 420 unwires CFGGuider and breaks the workflow.
- `torchaudio.detect_pitch_frequency` on silence → false positives. Gate with RMS > 0.005.
- `LTXVPreprocess img_compression=0` SKIPS preprocessing (frozen first frames). Use 18 (Lightricks) or 35 (core).
- Pyright `reportIncompatibleMethodOverride` on `execute()` is a false positive.
- **`LTXVConcatAVLatent` isn't buggy.** `output.update(video); output.update(audio)` gets overwritten by a proper `NestedTensor` assignment on the next line. Don't chase.
- Validate after edits: `python3 -c "import json; json.load(open('file.json'))"`.
- **`Path.glob()` returns `[]` cleanly on a missing directory.** Drop the `if not source_dir.is_dir(): return False` pre-check before `source_dir.glob(...)` — the empty-result guard a few lines down already covers both "dir absent" and "dir present, no match." TOCTOU + extra stat for nothing. Caught 2026-04-26 in `harness.py::_locate_and_link_output_mp4`.
- **New node modules** that need `comfy_api` / `comfy.patcher_extension` imports define inline `_Passthrough` / `_IOStub` / `override` fallbacks under a `try: from comfy_api.latest import io / except ImportError:` block. See `nodes_sage.py` and `nodes_easycache.py`. Two consumers is the minimum threshold for extracting to a shared helper; factor out only if a third node needs the same stubs.
- **LTX denoiser-level wrapping** uses `model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, fn)`. Supported wrapper API; not a monkey patch. Reference: `nodes_easycache.py`. Cleaner than patching `BasicTransformerBlock.forward` directly.
- **Always `git status --short` before `git commit`**. Pre-staged files (privacy_guard hook, linter mutations, half-finished prior work) get swept into your commit otherwise; the commit title then misrepresents the content.
- Scrub workflows before open-sourcing: filenames, paths, UUIDs, previews, creative prompts.
- **TensorLoop framework-cache invalidation is transitive.** Any node downstream of `current_iteration` re-executes per iter. Memoize via `id()`-keyed LRU + `IS_CHANGED` (see `TimestampPromptScheduleBatchEncode`). Module-level caches (`_BATCH_ENCODE_CACHE`, `_COND_CACHE`) die on ComfyUI restart — they're plain dicts, no persistence.
- **LTX has no image VAE encode node.** Decode variants exist (`LTXVTiledVAEDecode`, `LTXVSpatioTemporalTiledVAEDecode`); audio has `LTXVAudioVAEEncode`. For image→latent, use core `VAEEncode` — even Lightricks' reference workflows do.
- **KJNodes ships `GetImageRangeFromBatch` (batch slicer: `start_index`, `num_frames`, `images` → IMAGE) and `SimpleCalculatorKJ` (expression-string math, Int/Float/Bool outputs).** Compose these before building custom slicer or math nodes — `apply_iclora_video_reference.py` uses `GetImageRangeFromBatch` inside the loop subgraph for IC-LoRA reference window selection. Grep `ComfyUI-KJNodes/__init__.py` registry before designing new utility nodes.

## Init image conditioning path

- **Initial render**: `#531 LTXVImgToVideoInplaceKJ` writes encoded init into frame 0; `noise_mask=0` locks it.
- **Loop iterations**: top-level `VAEEncode → subgraph slot 8 (guide_latent) → #1519 LTXVAddLatentGuide` with `latent_idx=-1` (conditioning on the frame BEFORE the window). Init encoded ONCE.
- **F2 — Preprocess symmetry (MANDATORY)**: both paths consume `#446 LTXVPreprocess(img_compression=18)` output. Wiring: `#445 ImageResizeKJv2 → #446 → { #531 (initial), #650 Set_input_image (loop guide) }`. Skipping `#446` on the loop branch is the photoreal-drift footgun — cross-attention reasserts its "singing woman with microphone" prior iter-over-iter. Apply: `scripts/apply_loop_guide_preprocess_symmetry.py`. Audit: `audit_workflows.py::preprocess_symmetry`.
- **F3 — Cropguides symmetry (MANDATORY)**: loop `#644 CFGGuider` positive/negative CONDITIONING must come from `#655 LTXVCropGuides`, NOT `#1519 LTXVAddLatentGuide` directly — mirrors initial path's `#164 → #381 → #153`. Bypassing `#655` leaves guide-keyframe metadata to accumulate iter-over-iter, producing subtle identity drift even after F2 is fixed. Apply: `scripts/apply_loop_cropguides_symmetry.py`. Audit: `audit_workflows.py::loop_cropguides_symmetry`. Recipes for both in `docs/guides/debugging_guide.md`.
- Full trace: `docs/reference/pipeline_flow_latent.md`.

## Video-reference IC-LoRA path (F12)

- **Companion to F2/F3**, not a replacement. Adds an IC-LoRA guide (cameraman, outpaint, union-control, etc.) **inside the subgraph**, downstream of `#1519 LTXVAddLatentGuide` and upstream of the F3 cropguides chain. Every iteration sees the IC-LoRA effect; sidesteps the Phase-0a MODEL-fork question (no iters with patched MODEL but no attached guide).
- **Topology**: top-level `VHS_LoadVideo → ImageResizeKJv2 → LTXVPreprocess(val=18) → SetNode "ref-video-frames" → subgraph IMAGE input slot`; subgraph `[reference_video] → GetImageRangeFromBatch (KJNodes) → LTXAddVideoICLoRAGuide.image`. Guide CONDITIONING outputs flow through the existing F3 cropguides chain.
- **F2 ref-video symmetry**: ref-video chain MUST include `LTXVPreprocess(val=18)` — same val as init image. Without it, ref-video frames hit different edge statistics than init image → cross-attention drift across iters. Audit: `iclora_ref_video_preprocess_symmetry`.
- **F3 ref-video symmetry**: guide CONDITIONING outputs MUST reach `CFGGuider` via `LTXVCropGuides[NoLatent]`. Audit: `iclora_video_reference_guide_in_loop_with_cropguides`.
- **Static is the default; sliding is a widget setting on the same wiring** (`GetImageRangeFromBatch.start_index=0` for static; rewire from `SimpleCalculatorKJ` for sliding). Switching modes doesn't require a graph rebuild.
- **Use upstream `LTXAddVideoICLoRAGuide` unchanged** — per-iter VAE encode of the sliced ref-video (~25 frames at output resolution) is acceptable until profiling shows otherwise. Don't fork.
- **`iclora.py` constraint**: `frame_idx` must be 0 (single-frame special case) or 1 mod 8 — otherwise rounded down. Default: 1.
- **`GetImageRangeFromBatch` (KJNodes)** = `(start_index, num_frames, images) → IMAGE`. Composes with `SimpleCalculatorKJ` for any future sliding-mode math (`int(round(a*b))`). No project-side slicer node needed.
- Apply: `scripts/apply_iclora_video_reference.py`. Pre-flight refuses if Step 0 (`apply_strip_dead_lora_loaders.py`) hasn't run. Decisions: `internal/ic_lora_assessment.md` D19–D23. Reference workflow that inspired the pattern: `internal/ref_workflows/ltx2.3-ic-lora-cameraman.json` (single-pass, not looped).

## Subgraph editing

- ALWAYS use `WorkflowEditor`. Top-level helpers: `find_node`, `has_node`, `require_nodes`, `find_link_to_slot(tgt, slot)`, `add_link`, `remove_link`, `rewire_input(tgt, slot, new_src, new_src_slot, dtype)`, `find_links_to/from`. Subgraph: `find_subgraph_invoker`, `find_subgraph_node`, `find_subgraph_link`, `find_subgraph_link_to_slot(tgt, slot)`, `add_subgraph_link`, `remove_subgraph_link`, `rewire_subgraph_input` (mirrors top-level rewire). `find_input_slot` works on both. **Don't hand-roll link lookups or rewires** — `find_link_to_slot` replaces the `next(lk for lk in ed.wf["links"] if lk[0] == link_id)` pattern; `rewire_input` / `rewire_subgraph_input` replace the `remove_link` + `add_link` splice.
- **Scaffold new apply scripts from `scripts/templates/`**. Two templates (`apply_script_all_workflows.py` for in-place edits, `apply_script_staged_variant.py` for experimental staging). Both include the canonical `--revert`, `--dry-run`, idempotence, and `require_nodes` guards. HyDE pattern: `apply_X.py --dry-run | audit_workflows.py` verifies a hypothetical state before committing to it.
- **`remove_link` rebinds the target list** via filter — locals holding `ed.wf["links"]` go stale. Use editor methods or re-fetch.
- Top-level links are array `[id, src, src_slot, tgt, tgt_slot, type]`; subgraph internal links are dict `{id, origin_id, origin_slot, target_id, target_slot, type}`. Subgraph def at `wf['definitions']['subgraphs'][0]`.
- Distributor `-10` / output collector `-20` are virtual — not in `sg["nodes"]`. Their slot indices map 1-to-1 with `sg["inputs"]` / `sg["outputs"]` order — useful when rewiring `CFGGuider` slots to/from the subgraph boundary (e.g. TTC1 init-guide POC: `CFGGuider.negative <- (-10, slot 6)` = "positive" raw, before `LTXVAddLatentGuide`).
- Output slots use `"links"` (plural list); subgraph boundary entries use `"linkIds"`. Don't conflate.
- DynamicCombo widgets: `[num, strength_1..N, index_1..N]` — strengths FIRST, not interleaved.
- **Apply-script pre-flight chaining**: when one migration logically depends on another, the dependent script's pre-flight should detect the pre-requisite's signature and refuse with an actionable message ("Run scripts/apply_X.py first"). Reference: `apply_iclora_video_reference.py` refuses if `#1625/#1626/#1627` are still present (Step 0 strip unrun).
- **`scripts/_apply_helpers.py` is for RAW-orjson fork-and-strip scripts only** (debug-tool stability when `WorkflowEditor` itself is suspect) — NOT a general utility module. Apply scripts that use `WorkflowEditor` (the canonical path) don't import from it. Confirm by reading its docstring before extracting helpers there.

## Testing

```bash
uv run --group dev --group analysis python -m pytest tests/ -v --rootdir=.
```

Add `--group experiments` if running the autoresearch contract tests
locally (`tests/test_autoresearch.py`); they gracefully skip without it
on fresh public clones that don't have duckdb installed.

- CI runs on push/PR to main (`.github/workflows/ci.yml`): pytest + `scripts/audit_workflows.py` + docs-consistency tests.
- `__init__.py` guards ComfyUI imports for pytest; `nodes.py` has `_IOStub`/`_Passthrough` fallback.
- `tests/conftest.py` adds `scripts/` + `tests/` to `sys.path`. Shared fakes: `tests/_fakes.py` (`FakeModelPatcher`, `FakeModelWithCallbacks`). Root `./conftest.py` has `collect_ignore` — shadows `tests/conftest.py` for `from conftest import X`.
- **Memoization fixes need REPEATED-call tests.** Single-call tests can't detect framework-cache-invalidation. Canonical shape: `tests/test_batch_encode.py::TestBatchEncoderCaching`.
- **Schema invariant tests need AST parsing, not runtime introspection.** When ComfyUI isn't loaded (pytest), `define_schema()` returns `_Passthrough` stubs and `schema.inputs` isn't iterable. Walk `io.*.Input(...)` calls via `ast` instead. Canonical: `tests/test_node_schemas.py::test_no_seed_or_noise_seed_named_inputs`.
- **`id()`-keyed caches need autouse clear-fixtures.** `FakeCLIP` gets GC'd rapidly; Python address recycling produces ghost hits. Production cache keys now include `type(clip).__name__` as cheap cross-class insurance; test fixtures still required.
- **`_LLM_SYSTEM_PROMPT` rewrites need test-invariant check first.** 8 tests in `tests/test_audio_features.py::TestFormatJsonReport::test_llm_system_prompt_*` assert specific load-bearing substrings: `is singing` / `are singing together`, `verbatim`/`identical`/`exactly`, all 6 tier names, `montage` + `emotional`, `dolly out`, `present progressive`, `frozen`, `init image` + `do not re-describe`, style-family examples (`comic` / `graphic-novel` / `animated` / `live-action`). Read these before rewriting the system prompt — a substring you remove silently breaks the test.
- **Degenerate-input metric branches need a distinct status, not sentinel "ok" values.** When an extractor handles a degenerate case (e.g. n=1 frame in `subject_consistency` → no comparisons possible → all sims trivially 1.0), returning `status: "ok"` with sentinel numbers pollutes downstream `WHERE status = 'ok'` aggregations — a degenerate render scores identically to a perfect one. Add a distinct `Literal` status (e.g. `single_frame`) so queries can exclude. Caught 2026-04-26 in `subject_consistency.py` simplify pass; same shape as the `trace_empty` / `trace_missing` / `decode_failed` distinctions that already exist in `sage_summary.py`.
- **Apply-script tests that need pre-migration state**: have the fixture `shutil.copy2(CANONICAL, dst)` then invoke the apply script's own `--revert` to restore. Keeps fixture state in lockstep with the script's own understanding of "before"; avoids a separate fixture-baseline file that drifts when the canonical changes. Reference: `tests/test_apply_strip_dead_lora_loaders.py::canonical_copy`.

## Dependencies

- **Runtime nodes** (`nodes*.py`): torchaudio only. All outputs FLOAT or INT.
- **Offline scripts** (`scripts/`): `analysis` optional group (librosa, scipy, Pillow).
- **Experiment runner** (`internal/autoresearch/`): `experiments` optional group (`duckdb`, `httpx`).
- **Phase 2.1 perceptual metrics** (`internal/autoresearch/metrics/`): `metrics` optional group (`torchvision`, `transformers>=4.46`, `opencv-python-headless`, `numpy`, `scipy`). Each metric module gates its heavy imports under try/except so the module loads cleanly on a public clone without the group; missing-deps path returns `*_status: "model_unavailable"`. Active metrics: `wall_time` (placeholder), `sage_summary`, `subject_consistency` (DINOv3, gated on HF), `av_consistency` (PE-AV-16-frame, Apache-2.0). Originally-planned `lip_sync` (AV-HuBERT) + `seam_continuity` (STREAM) subsumed by `av_consistency`. Reference codebases under `coderef/` (read-only): `dinov3/`, `perception_models/`, `sam-audio/`.

Companion custom nodes (used alongside, not imported):
- `<sage_fork_repo>/` — our SageAttention fork. Cross-repo state: `internal/design/sage_backlog.md`.
- `ComfyUI-NativeLooping_testing` (TensorLoopOpen/Close), `ComfyUI-LTXVideo`, `ComfyUI-KJNodes`, `ComfyUI-VideoHelperSuite`, `ComfyUI-MelBandRoFormer` (bypassed by default in shipped workflows; re-enable via two-step manual edit per `scripts/apply_melband_default_off.py`).

## Audio analysis scripts

- `scripts/analyze_audio.py` — ffmpeg-only energy/structure detection, zero Python deps.
- `scripts/analyze_audio_features.py` — librosa: BPM, key, F0, structure, JSON export for LLM (`--scene-diversity`, `--montage`, `--style`). Full guide: `docs/guides/audio_analysis_guide.md`; end-to-end: `docs/guides/prompt_workflow_end_to_end.md`. **Works on generated audio too**: extract via `ffmpeg -i <mp4> -vn -acodec pcm_s16le <wav>` then analyze — primary tool for comparing source vs. generated audio features.
- `scripts/spectrogram_to_reference.py` — Mel spectrogram → PNG frame sequence for IC-LoRA spectrogram-as-reference (Phase 2.0 PoC). **Global normalization runs ONCE in `prepare_mel_for_render`** (do NOT switch to per-frame — washes out beat-amplitude). **Dual-use**: primary use is reference rendering; diagnostic use is visualizing generated audio via `--audio <wav>` (Claude can't hear mp4 streams; spectrograms of the output make audio behavior reviewable). Supports `--colormap {gray,viridis,spectrum}`; B&W triggers vintage-broadcast audio priors in LTX 2.3 — use color for V2A experiments. Design + iteration ladder: `internal/design/spectrogram_reference_design.md`.

## Debug & migration tooling

Full reference: **`docs/reference/debug_tools.md`** — inspection scripts (`audit_workflows.py`, `analyze_workflow_dag.py`, `trace_node_source.py`, sage telemetry), apply-script conventions (three-tier staging, idempotence, scratch-build pattern, `_apply_helpers.py`), runtime telemetry paths, RUN_ID artifact correlation, and the **canonical first-pass when a workflow won't run** (tail comfyui log → audit → DAG → trace node → exec log). Symptom-first quality troubleshooting: `docs/guides/debugging_guide.md`. Or invoke `/diagnose-workflow` for the canonical first-pass as a single command.

- **`scripts/audit_workflows.py [path...]`** — default sweeps `example_workflows/` (+ audited subset of `experimental/`); pass paths to audit a staged scratch file or any other JSON. Use this when validating an apply-script-produced file in `internal/scratch/`.

Two non-negotiable rules from that reference:
- **Bake new topology constraints into `audit_workflows.py`.** Every fix that ships an apply script ships a matching audit check (ERR + `Run scripts/apply_X.py` remediation). Canonical pairs: F2 (`preprocess_symmetry`), F3 (`loop_cropguides_symmetry`), F4 (`alc_seed_legacy_name`), F5 (`iterations_autowired`), F6 (`alc_widget_drift`), F7 (`planner_no_stride_input`), F11 (`dead_lora_loader_scaffolding_absent`), F12 (`iclora_video_reference_guide_in_loop_with_cropguides` + sibling iclora checks). Full table: `docs/reference/debug_tools.md`. Plus three generic audit invariants (`graph_acyclic`, `widget_shape`, `link_integrity`) and one AST test (`tests/test_node_schemas.py::test_keyframe_idxs_cleared_to_none_not_empty_list`) that catch CLASSES of drift without per-bug rules.
- **Iter-over-iter drift** → trace CONDITIONING paths in parallel (initial vs loop). Asymmetries (missing `LTXVConditioning`, `frame_rate` mismatch, CLIP in subgraph) are load-bearing bugs.

## Working with Claude across sessions

- **Check sibling-session backlogs (`internal/design/*_backlog.md`) before executing stale PLAN items** that touch defaults. Stale items can silently regress decisions another session has since made.
- **Run `/simplify` after non-trivial code changes.** Three-agent review (reuse / quality / efficiency) catches data-flow correctness bugs that shape-only tests miss.
- **Verify a new model via its paper, not its name.** SAM-Audio reads as "audio-conditioned visual segmentation" but is actually audio source separation (arxiv:2512.18099). Run `paper_search` or fetch the README before designing a metric/feature around it. Cost ~30s; saves an entire session of building against the wrong assumption. Caught 2026-04-26 during Phase 2.1 SAM-Audio evaluation.
- **Promote helpers at the 3rd call site, not the 2nd.** Reviewer consensus across multiple `/simplify` passes. Prevents premature extraction: two sites can share a short inline pattern without paying the abstraction cost; by the third, the pattern is load-bearing and the name-plus-tests earn their keep.
- **`PLAN.md` (or feature design doc) is the spec.** When red TDD tests disagree with the spec formula, fix the test — the spec wins unless you explicitly update PLAN first.
- **Decisions-index pattern**: DECISION / WHY / CONTEXT triples, grouped by feature. Template at `internal/ic_lora_assessment.md §6.5`. Roll up any feature >3 commits to avoid re-deriving rationale from git log.
- **LTX 2.3 audio-feature seed variance is ~±20 BPM** for equivalent electronic-genre conditioning. Single-seed comparisons between configs are noise; multi-seed (3-5 per config) needed to detect audio-effect changes. Ref: `docs/experiments/exp_2026-04-24_spectrogram_iclora_v2a.md` §Inferences.
- **Record the prior in writing BEFORE the measurement.** Even a rough Amdahl derivation ("attention is ~30% of step time × 2.62× kernel speedup ≈ 1.24× gen speedup") commits a prediction the result can grade against. "Did the prior hold?" is more useful than "what was the number?" when the measurement lands. Canonical: bilateral pre-bench briefs at `internal/brief_for_*.md` (gitignored).
- **Bulk `replace_all` AFTER adding prose to the same file is dangerous.** A retraction note or callout that *quotes* the pattern being replaced (e.g. a note saying "used to use `Cut to a [shot]...`" followed by `replace_all "Cut to a " → "In a "`) sweeps your own freshly-added prose, corrupting the annotation. Add prose annotations AFTER bulk pattern edits, or scope the pattern (only schedule-line matches, only inside fenced blocks). Caught 2026-04-25 in Cut-to retraction; only `/simplify` review caught it.
- **Sibling-session commit race**: when multiple Claude sessions run concurrently, one's `git add` + `git commit` can sweep another's staged changes into its commit with the wrong message. Verify your own commits via `git log -- <file>` (not `git log -1`); the CHANGELOG entry inside the commit is the durable record if the message lies.
- **Measure the boundary you actually patch, not the boundary your model predicts.** Sage's per-call timing exposes the attention row (8.2% of wall on production audio-loop), but sage's int8 amortization reaches into FFN-adjacent sampler work too — empirical e2e is **1.22×**, +17 points above the strict-attention Amdahl prediction (~1.05×). Both priors (sage-claude's 1.05-1.10×, mine 1.05-1.10× revised) were wrong same direction; we anchored on attention-fraction × kernel-speedup without measuring the sampler-fraction empirically first. Lesson: when an optimization patches a chunk of an inner loop, measure the inner loop's wall time, not just the patched call's elapsed time. Caught 2026-04-27 post-bench. Canonical e2e number lives in sage-fork `CHANGELOG.md` v0.5.1 + `VISION.md` item-3 status.
- **Sage e2e is shape-invariant relative to VAE-decode choice.** Sage doesn't patch VAE. When testing alternate VAE decoders (`LTXVTiledVAEDecode` vs `LTXVSpatioTemporalTiledVAEDecode` vs single-tile), the per-arm wall-time ratio in sage-fork's bench is confounded by VAE cold-start; only the per-node `exec.jsonl` aggregate of the decoder row is the load-bearing read. Don't chase a sage_off/sage_on wall ratio change across decoder variants — that's measuring decoder, not sage.
- **Use `--warmup always` (not `--warmup auto`) on sage-fork's `bench_e2e_ltx.py` when the workflow shape changes between runs.** `--warmup auto` detects "ComfyUI process is generally warm" via recent sage-trace mtime, but doesn't know per-shape autotune + activation buffers are still cold when the workflow changes (different decoder, different sampler config, different model load). Symptom: arm 1 pays new-shape cold-start, arm 2 doesn't, raw `wall_off / wall_on` ratio gets the arm-order bias we spent 2026-04-27 debugging. Future sage-fork improvement (deferred): record `workflow_path/hash` per-run + only auto-skip when prior run used same workflow.
- **Project-level `settings.json` hook config is loaded once at session start** — deleting a hook script mid-session leaves the cached config trying to run a now-missing file, blocking every Write/Edit until session restart. Workaround: use Bash (not Write/Edit) to make any post-deletion edits; `/reload-plugins` doesn't clear the cached project settings, only marketplace plugin contents. Caught 2026-04-30 during the home-grown-`privacy_guard.py` retirement.
- **Marketplace plugin cache lags behind merged plugin changes**, even after `/reload-plugins`. The plugin's `install-git-hooks.sh` bakes the script path into the wrapper at install time; that path points at the cached version (`<HOME>/.claude/plugins/cache/<marketplace>/<plugin>/<version>/...`) which won't change until the cache refreshes. To pick up freshly-merged plugin changes immediately, re-run `install-git-hooks.sh` from a workspace clone of the plugin repo — the wrapper then points at the workspace source, no cache dance needed. Tradeoff: workspace deletion breaks the hook. Caught 2026-04-30 when the path-privacy 0.1.4 false-positive fix was blocked by a 0.1.1-cache hook.

## Documentation conventions

- **Active planning lives in gitignored `internal/`.** Promote to `docs/` only when feature ships AND stabilizes.
- **Don't reference `internal/log/` from public-facing docs** — session logs are timestamped/personal. Other `internal/` subdirs (`analysis/`, `design/`, `ic_lora_assessment.md`, `action_items_for_*.md`) are fine to reference from `docs/` if no private prompts/paths leak.
- **Case studies live in `internal/prompts/` (gitignored, unscrubbed).** Public guides distill patterns inline rather than linking out — the parallel scrubbed-copy convention was retired 2026-04-25 (parallel maintenance burden, scrub-leak risk, no confirmed external readership). Reference internal prompt runs from `docs/` only via paraphrase, never via filename.
- **Public docs written for GitHub readers, not our local state.** No "already on disk" / "we use X locally" framing; use `<comfyui_models>` / `/path/to/model` placeholders and list file sources (Hugging Face slugs, upstream repos). `internal/` docs can assume our local state.
- **Breaking changes trigger docs sweep** — add stale phrase to `scripts/validate_docs_consistency.py::STALE_PATTERNS`; `tests/test_docs_consistency.py` fails until fixed.
- **Last-updated date at top of every doc** (`Last updated: YYYY-MM-DD`).
- **Trim public + archive full** for reference docs >1000 lines. Public in `docs/reference/` → summary; full → `internal/archive/` (gitignored).
- **`.claude/` harness is tracked, NOT gitignored.** Agents/skills/hooks/`settings.json` are shared via git so contributors get the same automation. Per-user state lives in `*.local.*` files (`settings.local.json`, `<repo-root>/.path-privacy.local.json`, `skills/cross-repo-handoff/`) which ARE gitignored. Full conventions for editing the harness: **`.claude/CLAUDE.md`**. Audit baseline (when present): `internal/analysis/harness_analysis.md` (gitignored). Drift protection: schedule a periodic re-audit via `/schedule` — routine IDs are per-account.
- **Path-privacy enforcement comes from the `path-privacy` plugin** (in the `fb-claude-skills` marketplace), not from in-repo hooks. Plugin provides PreToolUse Write/Edit blocking + git pre-commit/commit-msg hooks + SessionStart directive + `find-external-paths.sh` (audit) + `scrub-paths.sh` (apply fixes with diff preview). Per-repo suggestion config lives at `<repo-root>/.path-privacy.local.json` (gitignored). Install the plugin's git hooks once per clone via `bash <plugin-root>/skills/path-privacy/skills/path-privacy/scripts/install-git-hooks.sh`.

## Documentation layout

Public docs: `docs/README.md` (task-first nav) → `docs/guides/` (how-to), `docs/reference/` (deep-dive — incl. `docs/reference/environment.md`, the env-var registry), `docs/analysis/` (research/postmortems on shipped code), `docs/experimental/` (scaffolded-but-not-validated features paired with workflows in `example_workflows/experimental/`), `docs/experiments/` (per-experiment logs: hypothesis → setup → observations → inferences → next; convention in `docs/experiments/README.md`). `docs/architecture_overview.md` is the single-entry-point architecture reference.

Reference codebases (read-only): `coderef/LTX-2/` (LTX-2 native), `coderef/LTX-Desktop/` (Lightricks Desktop), `<comfyui_custom_nodes>/ComfyUI-LTXVideo/` (ComfyUI LTX integration).

Example workflows (`example_workflows/`): seven shipped — `_image.json`, `_image_adain_perstep.json`, `_latent.json` (primary), `_latent_keyframe.json`, `_latent_stg.json`, `_latent_validator.json`, `_retake.json` (regenerate one section of a prior generation; built by `scripts/apply_audio_loop_retake.py`). All on `AudioLoopHelperSageAttention auto_mask_aware`. Validate via `scripts/audit_workflows.py`.

Claude Code harness (`.claude/`, mostly tracked):
- `.claude/CLAUDE.md` — conventions for editing harness contents (hook authoring, agent/skill rules, privacy abstraction, settings split). Read before adding/modifying anything under `.claude/`.
- `.claude/README.md` — human-oriented contributor overview.
- `.claude/agents/` — 3 subagents (`workflow-validator`, `conditioning-path-auditor`, `ltx-constraints-auditor`). Privacy-scrubbing now comes from the `path-privacy` plugin.
- `.claude/skills/` — 10 user-invokable workflows. (Privacy scrub now via the plugin's `scrub-paths.sh`.)
- `.claude/hooks/` — `doc_date_check.py` (PostToolUse), `check_memo_inbox.sh` (SessionStart). Privacy enforcement now via the `path-privacy` plugin's hooks.
- `.claude/settings.json` — shared hook wiring; uses `${CLAUDE_PROJECT_DIR}` for portability.
- `.claude/settings.local.json` (gitignored) — per-user permissions + ComfyUI-loader smoke test.
- `<repo-root>/.path-privacy.local.json` (gitignored) — literal-substring suggestion config consumed by the `path-privacy` plugin.

Internal (gitignored):
- `internal/PLAN.md` — active roadmap.
- `internal/TODO.md` — step-by-step "what to do next" with checkbox sections, when present. Updated by Claude on demand.
- `internal/ic_lora_assessment.md` — IC-LoRA phases + decisions index (D1–D18).
- `internal/design/*.md` — long-term designs (`spectrogram_reference_design`, `sage_backlog`, `upscale_workflow_design`).
- `internal/autoresearch/` — Karpathy-autoresearch-style experiment-runner framework adapted for LTX video. Agent edits `apply.py`; harness orchestrates; tracker is DuckDB; metric extractors live under `metrics/`. Brief: `internal/autoresearch/program.md`. Public-facing test contract: `tests/test_autoresearch.py`.
- `internal/scripts/` — canonical sources for files that deploy out-of-repo (`start.sh` → `<comfyui>/start.sh`; `sage_fork_build.sh` → `<sage-fork>/build.sh`). Edit here, push via `internal/scripts/sync_to_deployed.sh`. README at `internal/scripts/README.md`.
- `internal/postmortem_*.md`, `internal/prompts/`, `internal/analysis/` — debugging history, unscrubbed case studies, deep dives.
