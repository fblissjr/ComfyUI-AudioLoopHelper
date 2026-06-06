# ComfyUI-AudioLoopHelper

Last updated: 2026-06-03

ComfyUI nodes that automate loop timing + audio analysis for full-length music video generation with LTX 2.3. Core pattern: `AudioLoopController` drives stride from integer latent counts, audio is frozen via `noise_mask=0`, prompts pre-encoded once outside the loop (CLIP must never enter the loop body). **Start here:** `docs/architecture_overview.md`; task-first nav at `docs/README.md`.

## Contents

1. [Commands](#commands) — pytest, audit, common apply-script invocation
2. [Architecture](#architecture) — files, nodes, entry points
3. [Critical constraints](#critical-constraints) — split by topic
4. [ComfyUI gotchas](#comfyui-gotchas)
5. [Init image conditioning + IC-LoRA paths](#init-image-conditioning--ic-lora-paths)
6. [Working with Claude across sessions](#working-with-claude-across-sessions)
7. [Documentation conventions](#documentation-conventions)
8. [Pending review](#pending-review) — capture-then-review staging

## Commands

```bash
# Full test suite
uv run --group dev --group analysis python -m pytest tests/ -v --rootdir=.

# Workflow audit (sweeps example_workflows/ + audited subset of experimental/)
uv run --group dev python scripts/audit_workflows.py

# Loop containment check (per-file; also runs under audit_workflows as F-pair)
uv run --group dev python scripts/verify_loop_containment.py <workflow.json>

# Apply script — typical shape (every apply_*.py supports these)
uv run --group dev python scripts/apply_<X>.py            # apply
uv run --group dev python scripts/apply_<X>.py --dry-run  # show changes
uv run --group dev python scripts/apply_<X>.py --revert   # undo
```

Add `--group experiments` for autoresearch contract tests (`tests/test_autoresearch.py`); they skip on clones without duckdb. Subtree CLAUDE.md files cover deeper conventions: working in `scripts/`, `tests/`, or `internal/autoresearch/` loads the matching subtree CLAUDE.md automatically.

## Architecture

Runtime node modules: `nodes.py` (core loop), `nodes_analysis.py` (torchaudio audio analysis), `nodes_sage.py` (sage attention), `nodes_validation.py` (config validator), `nodes_audio_iclora.py` (audio-reference IC-LoRA: loaders + guides + Compose Reference Audio), `nodes_audio_latent_slice.py`, `nodes_easycache.py`, `nodes_ffn.py`, `nodes_regional_compile.py`. `audio_reference_shaping.py` is the reference-windowing engine backing `nodes_audio_iclora` (no node class). Entry point: `comfy_entrypoint()` in `nodes.py`.

Core nodes (per-node role + wiring in each class's docstring; full reference at `docs/reference/ltx23_model_reference.md`):

- **Loop spine**: `AudioLoopController`, `LoopIterationStamp`, `IterationCleanup`, `AudioLoopPlanner`, `AudioDuration`
- **Prompt schedule**: `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` (current) / `TimestampPromptSchedule` + `CachedTextEncode` (legacy; don't wire in loop body)
- **Keyframe schedule**: `KeyframeLatentScheduleBatchEncode` + `LatentSelectByIteration` (timestamp-driven; current — VAE-encodes once outside loop) / `KeyframeImageSchedule` + `ImageBlend` (legacy; per-iter VAE) / `LTXIterKeyframeSchedule` (iter-index SELECTOR — per-iter keyframe latent by **1-based** `target_iters` → subgraph `guide_latent` → `LTXVAddLatentGuide`; `first_frame_guide_strength=1.0` = HARD lock. Ships `example_workflows/audio-loop-music-video_latent_keyframe.json`; design `example_workflows/working_docs/keyframe_iter_anchor_design.md`; gen `scripts/apply_keyframe_iter_anchor.py`) / single-pass, no loop: `KeyframeGuidesTimeSpaced` (`docs/guides/keyframe_fill_single_pass.md`)
- **Latent ops**: `LatentContextExtract`, `LatentOverlapTrim`, `LatentTemporalMask` (retake; `edge_taper_seconds` for soft boundary), `AudioTemporalMask` (audio analog of `LatentTemporalMask`), `LatentSeamZoneMask` (multi-band mask centered on iteration boundaries — pairs with `scripts/diagnose_overlap_seams.py`), `LatentFrameCount` (sizes empty audio latent for upscale + seam), `TrimImageBatchToAudio` (F14), `TrimVideoLatentToAudio` (A/B staged), `LTXHeadTrim` (image+audio composite — drops first N latent-frames' pixel + matching audio span; default 0 = no-op), `RunIdPrefix` (F15), `PreDecodeCleanup` (unloads models+pins before the full-song final decode — loop family ONLY, never single-pass/battery; F-pair `apply_pre_decode_cleanup.py`)
- **Attention + profiling + blend**: `AudioLoopHelperSageAttention` (default `auto` as of 2026-05-15 via `scripts/apply_sage_mode_auto.py`; `auto_mask_aware` was prior default — runtime-equivalent on audio-loop workflows since the masked path doesn't fire there, and `auto` is what benchmark workflows need), `ProfileBegin`/`IterStep`/`End`, `ConditioningBlend` (works with Gemma 3 + CLIP)
- **Step-skipping cache**: `LTXVideoEasyCache` (experimental, default off)
- **Dimension SSoT**: `LTXFramePlanner` — see `docs/reference/frame_planner_reference.md`

Analysis (`nodes_analysis.py`, torchaudio only): `AudioPitchDetect` → F0 + vocal-fraction; pairs with `ConditioningBlend.blend_factor`.

## Key patterns

- `AUDIO = {"waveform": Tensor, "sample_rate": int}`. Duration = `waveform.shape[-1] / sample_rate`.
- **Stride from integer-latent counts**, not widget seconds. `overlap_seconds` widget is a target; node emits the effective quantized overlap. Prevents lip-sync drift across overlap-widget changes. Math + wiring + audits: `docs/reference/audio_loop_controller.md`.
- LTX 2.3 text encoder is Gemma 3, NOT CLIP. Format: `[tensor, {"attention_mask": mask}]`, no pooled.
- Video VAE formula: `latent = (pixel - 1) // 8 + 1`. Not `pixel // 8`.
- `noise_mask=0` = fixed context; `mask=1` = regenerate. Audio frames always 0, video 1, overlap context 0. Setters/strippers + decision table: `docs/reference/noise_mask_semantics.md`.
- Guide chaining: multiple `LTXVAddLatentGuide` / `LTXVAddGuideMulti` (up to 20) accumulate via `keyframe_idxs`; `LTXVCropGuides` strips them.
- **CFG-analog amplification**: feed `(positive_with_X, positive_without_X)` to `CFGGuider` to amplify any conditional via existing CFG math. POC: `scripts/apply_ttc_iclora_amplification_poc.py`. Mechanism + decision table + failure modes: `docs/reference/cfg_analog_amplification.md`.

## Critical constraints

### Audio + latent topology

- **Audio path is sacred.** `LTXVAudioVAEEncode → LTXVConcatAVLatent`; never feed visualizations into the video latent (heatmap frames result).
- **`LTXVAudioVideoMask` (Node 606) wiring is intentional** — `audio_start_time = audio_end_time = window_size` (empty range keeps audio fixed). Don't change.
- **Audio is FROZEN.** Strip music/instrumentation references from schedule prompts; keep diegetic sounds only. Rationale: `docs/analysis/audio_in_prompt_research.md`.
- **Use `LatentContextExtract` / `LatentOverlapTrim`**, not raw `LTXVSelectLatents` — they strip `noise_mask` automatically.
- **`AudioLoopController` outputs are iteration-dependent in the executor DAG.** Its `current_iteration` input transitively reaches every output. Anything OUTSIDE the loop that needs `stride_seconds` / `audio_duration` (e.g. initial-render conditioning fed from `TimestampPromptScheduleBatchEncode`) must source them from `AudioLoopPlanner` — otherwise closes a cycle through `TensorLoopOpen`. Audit catches it as `graph_acyclic` ERR.

### Sampler + sigma chain

- **Distilled 8-step path.** `ManualSigmas "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"` + `KSamplerSelect euler` + `CFGGuider cfg=1`. **No flow-matching shift node** (no `ModelSamplingSD3`). **No `euler_ancestral*`.** Full walkthrough + Lightricks evidence: `docs/reference/sampler_reference.md`. Migration: `scripts/apply_canonical_sigmas.py`, `scripts/apply_strip_sd3_shift_node.py`.
- **VAE decode — loop family**: `LTXVSpatioTemporalTiledVAEDecode [1,1,63,7,true,"cpu","float16"]` — temporal chunks in LATENT frames whose stride (t_len−t_overlap = 56 latents = 17.92s at canonical 19.88/2) is bit-exact with the iteration stride (chunk seams land on iteration boundaries), bounding decode RAM at any song length; widgets derive per-workflow from `nodes._compute_loop_geometry`. `"cpu","float16"` accumulator is LOAD-BEARING (`"auto","auto"` = full-video fp32 buffer, kernel-OOMs ≥4-min songs). Apply: `scripts/apply_ltx_decoder.py --spatiotemporal`; CI gate: `scripts/validate_workflow_decoder.py` (link-traces the controller's window/overlap — its widgets are stale placeholders). **Single-pass/short-clip** stays `LTXVTiledVAEDecode [1,1,1,true,"cpu","float16"]` (no song-length exposure). Mechanism: `docs/reference/benchmarking_memory_pressure.md`.
- **Don't copy upstream's 15-step sampling** from `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`. Authoritative distilled path: 8 fixed sigmas.

### Conditioning + prompts

- **Verb choice drives cross-attention; generic verbs dilute it; token budget is shared.** LTX 2.3 binds the visible action to the prompt's verb — not "singing"-specific (works with `is dancing`, `is playing <instrument>`, etc. when the verb matches the audio). Generic verbs (`performing`, `vocalizing`) dilute it. Prompt tokens compete with audio + image cross-attention; **concise > verbose, especially with i2v init** (carries scene/style for free); without i2v, text does more work. Not a hard "must contain singing" rule. Full guidance: `docs/guides/prompt_creation_guide.md`.
- **Use `In a [shot], [camera]` continuation framing for non-first entries — NOT `Cut to a ...`.** Lightricks's official LTX 2.3 system prompt explicitly trains the model to treat scene-cut language as a discontinuation directive. Guide: `docs/guides/prompt_creation_guide.md`. Evidence: `docs/reference/ltx23_prompt_system_prompts.md`.
- **Node 169 prompt matches schedule 0:00 entry** structurally (`_build_prompt_for_section` via shared `_prepare_sections`; byte-exact).
- **CLIP must not enter the loop body.** Pre-encode via `TimestampPromptScheduleBatchEncode` outside; `ConditioningSelectByIteration` plucks per-iter inside. Mechanism + cache + failure modes: `docs/reference/timestamp_prompt_schedule_batch_encode.md`.
- **Loop-body CONDITIONING must carry `frame_rate=25.0`.** `LTXVConditioning.frame_rate` scales the temporal pos embed at `comfy/ldm/lightricks/av_model.py:866`; batch encoder stamps it, new CONDITIONING-producing loop-body nodes must too via `node_helpers.conditioning_set_values`. Missing → identity drift iter-over-iter. **Canonical LTX 2.3 inference fps = 25** (Lightricks's shipped ComfyUI-LTXVideo example workflows set `LTXVConditioning.frame_rate=25` across T2V/I2V distilled + full; V2V uses 24 to preserve source-video fps; 8n+1 latent boundary aligns cleanly at 25). The 24.0 in `coderef/LTX-2/ltx-pipelines/.../constants.py` is a Python-library placeholder, not canonical. `LIST_WIDGET_NODES` allowlist coverage enforced by `tests/test_node_schemas.py::test_apply_fps_default_covers_all_fps_bearing_widgets`.
- **Illustrated inits drift toward photoreal across iterations** (cross-attention is photoreal-trained). Match init-image style family; or re-anchor via `LTXVAddGuideMulti` per iteration.
- **LTX2_NAG widgets** `[nag_scale, nag_alpha, nag_tau, inplace]`. KJNodes default `scale=11` is aggressive for distilled — dial to 3-7 if initial render freezes. Reference: `docs/reference/nag_technical_reference.md`.

### Resolution + dimensions

- **Dimension config flows from `LTXFramePlanner` (single source of truth).** All shipped workflows wire its outputs to consumers. See `docs/reference/frame_planner_reference.md` for snap rules, the latent-volume **VRAM advisory** (no hard ceiling — `OK`/`HIGH_VRAM` vs LTX-2's HQ default 32,130; rebased 2026-06-03), wiring map, migration. Audit: `frame_planner_present` (F8).
- **The ONLY hard resolution/length rules are grid alignment** (video VAE): **div-by-32** spatial (single-stage; **div-by-64** two-stage), `(frames-1)%8==0` temporal. There is no model-side latent-volume/token ceiling — high volume = more VRAM, not artifacts. Audio VAE has NO divisibility rule (`latents/sec` from checkpoint config, runtime uses `ceil`; our audio nodes infer the rate empirically from the encoded latent's `T` — never hard-code `round(duration*25)`). Stride↔frames matches the reference (`stride_px = new_latents*8`).
- **Resolution div-by-32** (single-stage) or **div-by-64** (two-stage). Two-stage = anything with `LTXVLatentUpsampler` + half-res pass1 → 2x → full-res pass2 (e.g. fml2v variants). For two-stage, init-render dim must equal pass2 output dim so downstream `LatentConcat` welds across iters — 960x544 fails (272 not div-32 at half-res); 960x512 works. `scripts/audit_workflows.py` checks the div constraint.
- **`snap_boundaries=True`** (default) lets `overlap_seconds` change without schedule re-authoring.
- **Iterations auto-track audio length.** `AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in` is wired in every shipped workflow. For short tests, drag in an `INTConstant` and rewire — recipe in `docs/guides/debugging_guide.md`. Audit: `iterations_autowired` (F5).

### Workflow JSON discipline

- **Always use `WorkflowEditor`** (`scripts/workflow_utils.py`) for JSON edits. Apply-script + audit-pair conventions: see `scripts/CLAUDE.md`.
- **Never name an INT widget exactly `"seed"` or `"noise_seed"`.** ComfyUI's frontend auto-attaches a `control_after_generate` dropdown to those names, silently mutating saved widget values across runs. Use `base_seed`, `seed_in`. Guard: `tests/test_node_schemas.py::test_no_seed_or_noise_seed_named_inputs`. Audit: `alc_seed_legacy_name` (F4).
- **Schema renames must strip leftover widget values too.** ComfyUI's backend pops widgets positionally — a rename without paired widget-value strip shifts stale strings into the wrong slot. Audit: `alc_widget_drift` (F6). Widget-order spec + failure modes: `docs/reference/audio_loop_controller.md`.
- **New schema-input DEFAULTS must be no-op when added to existing nodes.** ComfyUI's loader fills missing widget values from the schema default at workflow-load time. Non-zero default = silent behavior change for every saved workflow on next reload. Default to 0.0 / `False` / `""` (whatever's no-op) and opt in via widget. Reference: `LatentTemporalMask.edge_taper_seconds` (default 0.0 = bit-identical hard mask). AST guard pattern: `tests/test_node_schemas.py::test_latent_temporal_mask_edge_taper_default_is_zero`.
- **Don't ship two schema changes touching the same iteration-state plane in one session.** Walk every edge between involved nodes; confirm none closes a cycle. Audit: `planner_no_stride_input` (F7). Cycle topology: `docs/reference/audio_loop_controller.md`.
- **Bake new topology constraints into `audit_workflows.py`.** Apply script + audit-check pair (F-pair convention). Inventory: `docs/reference/debug_tools.md`. How-to: `docs/reference/f_pair_convention.md`.

## ComfyUI gotchas

- **LTX 2.3 cross-attn passes `mask=None`.** `BasicTransformerBlock.attn2` calls with `attention_mask=None`; sage's `auto_mask_aware` mask-routing is defensive only on current LTX workflows.
- **`nn.Module.__setattr__` auto-registers Module-typed attributes as submodules.** Don't store sentinels via setattr on Module wrappers — `state_dict()` recurses on the same tensor twice. Use the official wrapper API or stash in a non-Module dict.
- **`CallbacksMP.ON_CLEANUP` fires after EVERY model invocation, not at model-unload.** Safe for per-call state RESET, destroys per-load state (compile caches, residency tunes).
- **ComfyUI v3 `io.ComfyNode` clones lock class attrs.** Any `cls.X = Y` / `cls.X += 1` raises `AttributeError` at execute time on the executor's locked clone. pytest's `_IOStub` fake doesn't lock so unit tests miss this — use module-level state for per-call counters or shared state across executes. Pattern: `_INSPECTOR_CALL_COUNTERS` in `nodes.py` (above `IterPatchInspector`).
- **comfy-aimdo's `0 patches attached` log line is misleading.** It counts the legacy `.patches` dict only; modern-API patches (LoRA via `.object_patches`, sage/NAG/AttentionTuner via `transformer_options`) survive offload but don't show in that counter. Verify in-flight via `IterPatchInspector` (`nodes.py`); drop it in the loop-body model chain.
- **Nodes that call `model.state_dict()` constrain model-mutation order.** Canonical order for compile-style patches: `UNETLoader → ... → LTXICLoRALoaderModelOnly → <module-mutating node> → SetNode "model"`.
- **LTX audio VAE always runs fp32 in ComfyUI; the video VAE runs bf16.** `comfy/sd.py` gives the LTX audio VAE `working_dtypes=[torch.float32]` and upcasts the (bf16) weights; `VAELoaderKJ` *ignores* its `weight_dtype` widget for audio VAEs (the audio branch passes no dtype to `VAE()`). So fp32 audio-reference encode-parity with training holds **automatically** — every audio-VAE weight file is bf16, fp32 is *compute*, not a file. Don't tell users to cast/get an "fp32 audio VAE" for the in-ComfyUI path (only matters if encoding the reference outside ComfyUI).
- **Eager top-level imports of non-core deps must be in `requirements.txt`** or ComfyUI Manager installs nothing on a clean clone and the *entire pack* fails to load (a top-of-module `import orjson` in `nodes_sage` did exactly this). Don't list torch/torchaudio (comfy ships them; pinning clobbers a user's CUDA build); `sageattention` stays optional (Sage nodes fall back to pytorch).
- **Workflow JSON has two link representations:** node-body `"link"` fields AND top-level `"links"` array. Both must sync. Link array: `[link_id, src, src_slot, tgt, tgt_slot, type]`.
- **Workflow JSON references inputs by NAME, not slot index.** A bare schema rename without a paired migration script that rewrites `inputs[].name` and `widget.name` will dangle every existing wire.
- `"mode": 0` = active, `"mode": 4` = bypassed. Bypass passes inputs to outputs of same TYPE only; non-matching inputs dead-end silently. Use `workflow_utils.is_active(node)` (canonical bypass check). Dead-node detection requires live-consumer check, not link-count check.
- **ComfyUI exposes the active prompt's id via `comfy_execution.utils.get_executing_context().prompt_id`** (a contextvar, not `transformer_options`). Lazy-import in the call path. Pattern at `nodes_sage.py:541-559`.
- `PrimitiveNode` can't feed `DynamicCombo` sub-inputs — set on the widget directly.
- `TensorLoopClose` checks `should_stop` AFTER the body; handle edge inputs.
- **Subgraph schema changes force a UI re-add** (slot indices baked at save time). Removing a subgraph input shifts higher slot indices.
- ComfyUI evaluates downstream conditioning before upstream sampling → extra nodes in conditioning path can corrupt initial render.
- **`CLIPTextEncode(169) → ConditioningZeroOut(420) → LTXVConditioning(164).negative → CFGGuider(153).negative` chain is wired-correctly but runtime-inert at `CFG=1`.** Don't try to remove it — `CFGGuider` validates both slots.
- `torchaudio.detect_pitch_frequency` on silence → false positives. Gate with RMS > 0.005.
- `LTXVPreprocess img_compression=0` SKIPS preprocessing (frozen first frames). Use 18 (Lightricks) or 35 (core).
- **`LTXVConcatAVLatent` isn't buggy.** Two `output.update(...)` lines are dead writes; the `NestedTensor` assignment that follows is load-bearing. Don't chase. Full investigation: `internal/postmortem_concat_av_latent_investigation.md` (private clone only).
- **TensorLoop framework-cache invalidation is transitive.** Any node downstream of `current_iteration` re-executes per iter. Memoize via `id()`-keyed LRU + `IS_CHANGED`.
- **LTX has no image VAE encode node.** For image→latent, use core `VAEEncode`.
- **KJNodes ships `GetImageRangeFromBatch` and `SimpleCalculatorKJ`.** Compose these before building custom slicer or math nodes. Grep `ComfyUI-KJNodes/__init__.py` registry before designing new utility nodes. Sibling-repo nodes also leak widget defaults into saved workflows (e.g. `LTXVAudioVideoMask.video_fps`); apply-scripts must allowlist them.
- **`--cache-none` is fatal for TensorLoop/looping workflows.** Maps to `NullCache`; the execution-inversion loop then re-runs ALL non-contained upstream nodes (text-encode, audio + keyframe VAE, model re-patch) EVERY iter → N× slowdown. `nodynvram`/`safe`/`minimal` modes set it (benchmark only); use `default` for loop renders. Fix: `docs/reference/debug_tools.md`.
- **Always `git status --short` before `git commit`.** Pre-staged files get swept into your commit otherwise. Scrub workflows before open-sourcing.
- **Concurrent unstaged edits get bundled into your commit.** If a file shows ` M` before you edit it, `git add` stages prior changes + yours. `git stash` first, or `git diff --cached -- <file>` post-stage to verify scope; when a sibling session's work must stay in-tree, stage only your hunks (filter `git diff` by `@@` context line, `git apply --cached`).
- **fp8 distilled checkpoint layout**: `<weight>.weight_scale` (rank-0 F32) = per-tensor scalar; `<weight>.comfy_quant` (U8) = JSON metadata blob, NOT a scale tensor. Block layout (44/48 fp8, bookend `[0,1,46,47]` bf16) + detail: `internal/reference/sage_optimization_landscape.md` (private clone only).

## Init image conditioning + IC-LoRA paths

- **Initial render**: `#531 LTXVImgToVideoInplaceKJ` writes encoded init into frame 0; `noise_mask=0` locks it.
- **Loop iterations**: top-level `VAEEncode → subgraph slot 8 → #1519 LTXVAddLatentGuide latent_idx=-1`. Init encoded ONCE.
- **F2 + F3 are MANDATORY symmetry rules** for the init-image path: both initial and loop branches share the same `LTXVPreprocess(img_compression=18)` output (F2); loop `CFGGuider` positive/negative flow through `LTXVCropGuides` (F3). Skipping either is the photoreal-drift / identity-drift footgun. Full trace + apply scripts: `docs/reference/pipeline_flow_latent.md`.
- **F12 video-reference IC-LoRA** (companion to F2/F3): IC-LoRA guide inside the subgraph between `#1519` and the F3 cropguides chain; F2/F3 symmetry rules extend to the ref-video chain. F2/F3 background: `docs/reference/pipeline_flow_latent.md`. Baked into the canonical latent.json (bypassed by default; un-bypass loader + guide + ref-video to enable). Design record: `scripts/archive/apply_iclora_video_reference.py`. Decisions: `internal/ic_lora_assessment.md` D19–D23 (private clone only).
- **F14 + F15 output handling**: `VHS_VideoCombine.images` must be fed by `TrimImageBatchToAudio` (eliminates silence-at-end from iter overshoot since ffmpeg `-shortest` doesn't truncate `-c:v copy`); `.filename_prefix` must be fed by `RunIdPrefix` (per-render folder clustering, WARN-level). Same `apply_run_id_layout.py` adds a bypassed `SaveLatent` toggle wired to `LatentConcat #1605` for the LoadLatent upscale chain. Guide: `docs/guides/upscale_guide.md`.
- **i2v init-image resize precision matters.** Single-pass lanczos at >2× linear reduction aliases (model reads as motion cues → spurious zoom/dolly); naive multi-stage PIL-backed lanczos stacks float32→uint8 quantization rounds (banding noise → same motion-cue effect). `LTXSmartImageResize` solves both: stages adaptively, uses bicubic+antialias for intermediates and lanczos only at the final stage. Stage planner: `nodes._compute_resize_stages`. Postmortem: `internal/analysis/smart_resize_quantization_postmortem.md` (private clone only).
- **`first_frame_guide_strength` (`FloatConstant #1269`) is the per-iter init-anchor strength** for `LTXVAddLatentGuide #1519`. Default `1.0` = max identity stability, minimal motion freedom. Lower for expressivity at the cost of cross-iter drift: `0.5` soft anchor, `0.3` visible drift, `0.0` no anchor.

## Working with Claude across sessions

- **GPU contention check before any bench/render.** `mtime` of `data/runs/*/*/sage.jsonl` (per-prompt routing) within last few minutes ⇒ a sibling-repo render is likely active. Ask before starting GPU work.
- **Infra-level regressions (OOM / perf / crashes): date the symptom against the ComfyUI checkout's pull history FIRST** (`git -C <comfyui> reflog`). The substrate under this pack moves upstream-fast; "worked last month" comparisons are cross-substrate. Worked example: `internal/analysis/oom_kill_cascade_2026-06/` (private clone only).
- **ComfyUI console log path**: `coderef/comfy_user/comfyui_8188.log` (current) + `coderef/comfy_user/comfyui_8188.prev.log` (previous render). Symlinked to ComfyUI's user dir; available without piping `start.sh` output to `tee`. Tail this when diagnosing a render that already crashed.
- **`AUDIOLOOPHELPER_PER_PROMPT=1` is default in `start_experiment.sh`** (since 2026-05-01). Artifacts route under `data/runs/${RUN_ID}/${prompt_id}/`. Reader scripts auto-detect both layouts.
- **Run `/simplify` after non-trivial code changes.** Its multi-agent review (reuse / simplification / efficiency / altitude) catches data-flow issues shape-only tests miss.
- **Behavioral-regression debugging starts with workflow diff, not code.** Extract embedded workflows from the working + broken PNGs via `scripts/extract_workflow_from_png.py <png> -o <json>` (positional arg, no `--workflow` flag) and structural-diff. Saved `.latent` files embed the EXECUTED prompt graph too (safetensors `__metadata__.prompt`; parse with stdlib json — Python emits NaN, orjson rejects). Code-level analysis after.
- **Magic string vs semantic literal**: extract opaque magic (`"v2"`, `"7"`) eagerly; leave semantic literals (`"DEBUG"`, `"private clone"`, `"singing"`) inline until 3rd call site or drift risk. `/simplify` reviewers flag both as "magic strings" — false-positive class on semantic literals.
- **Grep `internal/design/`, `internal/analysis/`, and the gitignored `PLAN.md` BEFORE forming recommendations** on any topic that smells like prior territory (trigger words: upscale, retake, IC-LoRA, sigma profile, edit anything, spatial, polish, seam). Most apparent "session-1 discoveries" are previously analyzed; re-derivation costs 2-3 turns. Subagent briefs should include "check internal/ first" — agents see the filesystem but not session context.
- **Sage v0.5.5+/v0.6 deliberation state** lives at `internal/reference/sage_optimization_landscape.md` (private clone only). Read it before responding to sage-fork claude memos or recommending v0.6 FFN-fusion next-steps; carries decision-gate state, three candidates with per-workload reach, fp8 vs bf16 four-branch matrix, trace measurement methodology.
- **Verify a new model via its paper, not its name.** Run `paper_search` / fetch README. Cost ~30s.
- **Promote helpers at the 3rd call site, not the 2nd.** Two sites can share inline; the third earns the abstraction.
- **`PLAN.md` (or feature design doc) is the spec.** When red TDD tests disagree with the spec formula, fix the test — the spec wins unless you explicitly update PLAN first.
- **Decisions-index pattern**: DECISION / WHY / CONTEXT triples, grouped by feature. Template at `internal/ic_lora_assessment.md` (private clone only). Roll up any feature >3 commits.
- **LTX 2.3 audio-feature seed variance is ~±20 BPM** for equivalent electronic-genre conditioning. Single-seed comparisons are noise; multi-seed (3-5 per config) needed.
- **Record the prior in writing BEFORE the measurement.** A rough Amdahl derivation commits a prediction the result can grade against. "Did the prior hold?" is more useful than "what was the number?"
- **Measure the boundary you actually patch**, not the boundary your model predicts. Sage's e2e gain is workload-dependent — don't generalize one workload's kernel ratio to another; non-Amdahl-gap claims require both arms directly measured. Bench numbers + retired framings: `internal/analysis/empirical_bench_findings.md` (private clone only).
- **Disprove-test before multi-day commits.** For "X% of sampler is Y" claims that would shift a multi-day decision, run the cheapest falsifying test first (interpretation-side variant of synthetic-vs-in-pipeline). Two profiler-data traps — nested `key_averages()` rows, and chrome-trace `cat=cpu_op` `dur` ≠ serial wall-time — so anchor on kernel rows. Detail: `scripts/analyze_torch_profile.py`.
- **Check sibling-session backlogs (`internal/design/*_backlog.md`) before executing stale PLAN items** that touch defaults.
- **Project-level `settings.json` hook config is loaded once at session start** — deleting a hook script mid-session leaves the cached config trying to run a missing file, blocking every Write/Edit until session restart. Workaround: use Bash for post-deletion edits.
- **Marketplace plugin cache lags behind merged plugin changes.** To pick up freshly-merged plugin changes immediately, re-run the plugin's `install-git-hooks.sh` from a workspace clone of the plugin repo.
- **Cross-repo coordination**: sister forks `sage-fork` + `LTX-2` (trainer at `coderef/LTX-2/`, fork CLAUDE.md inside); companion `comfy-workbench`. Memo channels: `cross-repo-handoff` + `cross-repo-handoff-ltx2`. Rule: ONE parent claude session at a time (two-claudes produce conflicting roadmap edits). Taxonomy: `internal/design/sister_repo_taxonomy.md` (private clone only).

## Documentation conventions

Generic doc rules (last-updated dates, lowercase filenames, document the "why", session-log location) live in the `/dev-conventions:doc-conventions` skill. Project-specific rules below.

- **Active planning lives in gitignored `internal/`.** Promote to `docs/` only when feature ships AND stabilizes.
- **Don't reference `internal/log/` from public-facing docs** — session logs are timestamped/personal. Other `internal/` subdirs are fine to reference if no private prompts/paths leak.
- **Case studies live in `internal/prompts/` (gitignored, unscrubbed).** Public guides distill patterns inline. Reference internal prompt runs from `docs/` only via paraphrase, never via filename.
- **Public docs written for GitHub readers, not local state.** Use `<comfyui_models>` / `/path/to/model` placeholders.
- **Breaking changes trigger docs sweep** — add stale phrase to `scripts/validate_docs_consistency.py::STALE_PATTERNS`; `tests/test_docs_consistency.py` fails until fixed.
- **Trim public + archive full** for reference docs >1000 lines. Public summary in `docs/reference/`; full → `internal/archive/` (gitignored).
- **Path-privacy enforcement comes from the `path-privacy` plugin** (in the `fb-claude-skills` marketplace). Per-repo suggestion config lives at `<repo-root>/.path-privacy.local.json` (gitignored). Install hooks once per clone.
- **Wiki direction**: `docs/reference/` is evolving toward Karpathy-style atomic notes (uniform shape per `docs/reference/frame_planner_reference.md`). Lint mode: `tests/test_claude_md_budget.py` catches orphans + broken pointers.

## Documentation layout

Public: `docs/README.md` (task-first nav) → `docs/guides/` (how-to), `docs/reference/` (deep-dive — incl. `docs/reference/environment.md`, the env-var registry; `docs/reference/frame_planner_reference.md`), `docs/analysis/` (research/postmortems on shipped code), `docs/experimental/`, `docs/experiments/`. Architecture entry point: `docs/architecture_overview.md`.

Reference codebases (read-only): `coderef/LTX-2/`, `coderef/LTX-Desktop/`, ComfyUI-LTXVideo upstream.

Example workflows (`example_workflows/`, all run `AudioLoopHelperSageAttention auto`): `_latent` (default loop), `_av_inversion` (video→audio dialogue replacement), `_keyframe` (per-iter re-anchoring; `target_iters` ships pre-filled 1,2,3, re-spread per song), `_retake`, `audio_reactive_loop`, `audio-ic-lora_single-pass` (single-pass audio-reference IC-LoRA). More variants in `experimental/`; retired ones in `archive/`. Validate via `scripts/audit_workflows.py`.

Subtree CLAUDE.md files (auto-loaded when working in that subtree):
- `scripts/CLAUDE.md` — apply-script conventions, audit invariants, WorkflowEditor patterns.
- `tests/CLAUDE.md` — pytest invocation, AST patterns, fakes hierarchy.
- `internal/autoresearch/CLAUDE.md` — experiment-runner framework (target-agnostic; gitignored, only loads on private clone).
- `.claude/CLAUDE.md` — harness conventions + CLAUDE.md governance policy.

Internal (gitignored): `internal/PLAN.md`, `internal/TODO.md`, `internal/ic_lora_assessment.md`, `internal/design/*.md` (long-term designs), `internal/autoresearch/`, `internal/scripts/` (out-of-repo deploy sources), `internal/postmortem_*.md`, `internal/prompts/`, `internal/analysis/`, `internal/log/log_YYYY-MM-DD.md` (session logs).

## Pending review (last drained: 2026-06-02)
- (none pending)
