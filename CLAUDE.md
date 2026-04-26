# ComfyUI-AudioLoopHelper

Last updated: 2026-04-26

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
- **Distilled 8-step sigmas**: `BasicScheduler linear_quadratic 8 1` + `ModelSamplingSD3 shift=13` + `KSamplerSelect euler` + `CFGGuider cfg=1`. Decoder: `LTXVTiledVAEDecode [2,2,1,true,"auto","auto"]`. Don't use `euler_ancestral*`. Full walkthrough: `docs/reference/sampler_reference.md`.
- **Illustrated inits drift toward photoreal across iterations** (cross-attention is photoreal-trained). Match init-image style family; or re-anchor via `LTXVAddGuideMulti` per iteration.
- **LTX2_NAG widgets** `[nag_scale, nag_alpha, nag_tau, inplace]`. KJNodes default `scale=11` is aggressive for distilled — dial to 3-7 if initial render freezes. Always verify schema via `scripts/trace_node_source.py <wf> 508`. Reference: `docs/reference/nag_technical_reference.md`.
- **Don't copy upstream's 15-step sampling** from `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`. Authoritative distilled path: 8 fixed sigmas per `coderef/LTX-2/.../distilled.py`.
- **Resolution div-by-32** (single-stage) or **div-by-64** (two-stage). `scripts/audit_workflows.py` checks.
- **Audio is FROZEN in our workflow.** Strip music/instrumentation references from schedule prompts; keep diegetic sounds only. Rationale: `docs/analysis/audio_in_prompt_research.md`; case studies in `internal/prompts/` (gitignored).
- **`EmptyLTXVLatentVideo.length` satisfies `(length - 1) % 8 == 0`.** Match `window_size_seconds = length / fps` exactly. Rapid-cut: `length=249`; default: `length=497`. To derive length + matching `window_size_seconds` from a desired duration without hand-math, use `LTXAVTools.LTXFrameCalculator(seconds, fps) → (frames, latent, actual_seconds)`.
- **`snap_boundaries=True`** (default) lets `overlap_seconds` change without schedule re-authoring.
- **CLIP must not enter the loop body.** Pre-encode via `TimestampPromptScheduleBatchEncode`; `object_patches` don't survive the offload/reload → silent NAG disengagement iter 2+. Mechanism: `docs/analysis/nag_object_patches_offload_asymmetry.md`.
- **Loop-body CONDITIONING must carry `frame_rate`** (default 25.0). Batch encoder stamps it; any new CONDITIONING-producing loop-body node must too (via `node_helpers.conditioning_set_values`). Missing → identity drift + hallucinated objects iter-over-iter.
- **Bake new topology constraints into `scripts/audit_workflows.py`.** Every fix that ships an apply script should ship a matching audit check (ERR status with a `Run scripts/apply_X.py` remediation pointer). Canonical pairs: F2 (`preprocess_symmetry`), F3 (`loop_cropguides_symmetry`), F4 (`alc_seed_legacy_name`), F5 (`iterations_autowired`). Prevents silent regression of fixes a sibling branch might revert.
- **Authoritative LTX 2.3 prompting evidence**: `docs/reference/ltx23_prompt_system_prompts.md:44, 56, 93` (Lightricks's own i2v + t2v system prompts: "DO NOT describe scene cuts", "Inaccurate descriptions may cause scene cuts"). What retracted our `Cut to` convention 2026-04-25. Check before relitigating any prompt-rule debate.
- **Never name an INT widget exactly `"seed"` or `"noise_seed"`.** ComfyUI's frontend auto-attaches a `control_after_generate` dropdown to those literal names, which silently mutates the saved widget value across runs even when the input is wired (link supersedes widget at execute time, but the mutated widget still gets serialized — saved JSONs drift across renders despite reproducible runtime seeds). Use `base_seed`, `seed_in`, etc. Guard: `tests/test_node_schemas.py::test_no_seed_or_noise_seed_named_inputs` AST-walks every `io.*.Input(...)` call. Diagnosed 2026-04-26 in `internal/analysis/id_lora_ablation_and_seed_widget_audit.md`.
- **Iterations auto-track audio length.** `AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in` is wired in every shipped workflow (added 2026-04-26 via `scripts/apply_iterations_autowire.py` + an upstream `ComfyUI-NativeLooping_testing` schema patch that made `iterations_in` a wireable optional input). User puts in any audio, loop runs exactly the iterations needed. For short tests, drag in an `INTConstant` and rewire — recipe in `docs/guides/debugging_guide.md`. Audit: `audit_workflows.py::iterations_autowired` (ERR if unwired in shipped workflows).

## ComfyUI gotchas

- Workflow JSON has two link representations: node-body `"link"` fields AND top-level `"links"` array. Both must sync.
- Link array: `[link_id, src, src_slot, tgt, tgt_slot, type]`.
- **Workflow JSON references inputs by NAME, not slot index.** Each node's `inputs[]` entry stores `{"name": ..., "type": ..., "widget": {"name": ...}, "link": ...}`; ComfyUI matches the saved name to the schema's input list when reattaching wires. So a bare schema rename (e.g. `"seed"` → `"base_seed"`) without a paired migration script that rewrites `inputs[].name` and `widget.name` in every saved JSON will dangle every existing wire on the renamed input. Canonical migration: `scripts/apply_alc_seed_rename.py`.
- `"mode": 0` = active, `"mode": 4` = bypassed. **Bypass passes inputs to outputs of same TYPE only**; inputs with no matching-type output dead-end silently. E.g., bypassing `LTXAddVideoICLoRAGuide` leaves its `image` input unconsumed. Verify truly-inert bypass by swapping the upstream input and byte-diffing outputs (`md5sum` on sampled frames, `wave` on decoded audio).
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
- **New node modules** that need `comfy_api` / `comfy.patcher_extension` imports define inline `_Passthrough` / `_IOStub` / `override` fallbacks under a `try: from comfy_api.latest import io / except ImportError:` block. See `nodes_sage.py` and `nodes_easycache.py`. Two consumers is the minimum threshold for extracting to a shared helper; factor out only if a third node needs the same stubs.
- **LTX denoiser-level wrapping** uses `model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, fn)`. Supported wrapper API; not a monkey patch. Reference: `nodes_easycache.py`. Cleaner than patching `BasicTransformerBlock.forward` directly.
- **Always `git status --short` before `git commit`**. Pre-staged files (privacy_guard hook, linter mutations, half-finished prior work) get swept into your commit otherwise; the commit title then misrepresents the content.
- Scrub workflows before open-sourcing: filenames, paths, UUIDs, previews, creative prompts.
- **TensorLoop framework-cache invalidation is transitive.** Any node downstream of `current_iteration` re-executes per iter. Memoize via `id()`-keyed LRU + `IS_CHANGED` (see `TimestampPromptScheduleBatchEncode`). Module-level caches (`_BATCH_ENCODE_CACHE`, `_COND_CACHE`) die on ComfyUI restart — they're plain dicts, no persistence.
- **LTX has no image VAE encode node.** Decode variants exist (`LTXVTiledVAEDecode`, `LTXVSpatioTemporalTiledVAEDecode`); audio has `LTXVAudioVAEEncode`. For image→latent, use core `VAEEncode` — even Lightricks' reference workflows do.

## Init image conditioning path

- **Initial render**: `#531 LTXVImgToVideoInplaceKJ` writes encoded init into frame 0; `noise_mask=0` locks it.
- **Loop iterations**: top-level `VAEEncode → subgraph slot 8 (guide_latent) → #1519 LTXVAddLatentGuide` with `latent_idx=-1` (conditioning on the frame BEFORE the window). Init encoded ONCE.
- **F2 — Preprocess symmetry (MANDATORY)**: both paths consume `#446 LTXVPreprocess(img_compression=18)` output. Wiring: `#445 ImageResizeKJv2 → #446 → { #531 (initial), #650 Set_input_image (loop guide) }`. Skipping `#446` on the loop branch is the photoreal-drift footgun — cross-attention reasserts its "singing woman with microphone" prior iter-over-iter. Apply: `scripts/apply_loop_guide_preprocess_symmetry.py`. Audit: `audit_workflows.py::preprocess_symmetry`.
- **F3 — Cropguides symmetry (MANDATORY)**: loop `#644 CFGGuider` positive/negative CONDITIONING must come from `#655 LTXVCropGuides`, NOT `#1519 LTXVAddLatentGuide` directly — mirrors initial path's `#164 → #381 → #153`. Bypassing `#655` leaves guide-keyframe metadata to accumulate iter-over-iter, producing subtle identity drift even after F2 is fixed. Apply: `scripts/apply_loop_cropguides_symmetry.py`. Audit: `audit_workflows.py::loop_cropguides_symmetry`. Recipes for both in `docs/guides/debugging_guide.md`.
- Full trace: `docs/reference/pipeline_flow_latent.md`.

## Subgraph editing

- ALWAYS use `WorkflowEditor`. Top-level helpers: `find_node`, `has_node`, `require_nodes`, `find_link_to_slot(tgt, slot)`, `add_link`, `remove_link`, `rewire_input(tgt, slot, new_src, new_src_slot, dtype)`, `find_links_to/from`. Subgraph: `find_subgraph_invoker`, `find_subgraph_node`, `find_subgraph_link`, `find_subgraph_link_to_slot(tgt, slot)`, `add_subgraph_link`, `remove_subgraph_link`, `rewire_subgraph_input` (mirrors top-level rewire). `find_input_slot` works on both. **Don't hand-roll link lookups or rewires** — `find_link_to_slot` replaces the `next(lk for lk in ed.wf["links"] if lk[0] == link_id)` pattern; `rewire_input` / `rewire_subgraph_input` replace the `remove_link` + `add_link` splice.
- **Scaffold new apply scripts from `scripts/templates/`**. Two templates (`apply_script_all_workflows.py` for in-place edits, `apply_script_staged_variant.py` for experimental staging). Both include the canonical `--revert`, `--dry-run`, idempotence, and `require_nodes` guards. HyDE pattern: `apply_X.py --dry-run | audit_workflows.py` verifies a hypothetical state before committing to it.
- **`remove_link` rebinds the target list** via filter — locals holding `ed.wf["links"]` go stale. Use editor methods or re-fetch.
- Top-level links are array `[id, src, src_slot, tgt, tgt_slot, type]`; subgraph internal links are dict `{id, origin_id, origin_slot, target_id, target_slot, type}`. Subgraph def at `wf['definitions']['subgraphs'][0]`.
- Distributor `-10` / output collector `-20` are virtual — not in `sg["nodes"]`. Their slot indices map 1-to-1 with `sg["inputs"]` / `sg["outputs"]` order — useful when rewiring `CFGGuider` slots to/from the subgraph boundary (e.g. TTC1 init-guide POC: `CFGGuider.negative <- (-10, slot 6)` = "positive" raw, before `LTXVAddLatentGuide`).
- Output slots use `"links"` (plural list); subgraph boundary entries use `"linkIds"`. Don't conflate.
- DynamicCombo widgets: `[num, strength_1..N, index_1..N]` — strengths FIRST, not interleaved.

## Testing

```bash
uv run --group dev --group analysis python -m pytest tests/ -v --rootdir=.
```

- CI runs on push/PR to main (`.github/workflows/ci.yml`): pytest + `scripts/audit_workflows.py` + docs-consistency tests.
- `__init__.py` guards ComfyUI imports for pytest; `nodes.py` has `_IOStub`/`_Passthrough` fallback.
- `tests/conftest.py` adds `scripts/` + `tests/` to `sys.path`. Shared fakes: `tests/_fakes.py` (`FakeModelPatcher`, `FakeModelWithCallbacks`). Root `./conftest.py` has `collect_ignore` — shadows `tests/conftest.py` for `from conftest import X`.
- **Memoization fixes need REPEATED-call tests.** Single-call tests can't detect framework-cache-invalidation. Canonical shape: `tests/test_batch_encode.py::TestBatchEncoderCaching`.
- **`id()`-keyed caches need autouse clear-fixtures.** `FakeCLIP` gets GC'd rapidly; Python address recycling produces ghost hits. Production cache keys now include `type(clip).__name__` as cheap cross-class insurance; test fixtures still required.
- **`_LLM_SYSTEM_PROMPT` rewrites need test-invariant check first.** 8 tests in `tests/test_audio_features.py::TestFormatJsonReport::test_llm_system_prompt_*` assert specific load-bearing substrings: `is singing` / `are singing together`, `verbatim`/`identical`/`exactly`, all 6 tier names, `montage` + `emotional`, `dolly out`, `present progressive`, `frozen`, `init image` + `do not re-describe`, style-family examples (`comic` / `graphic-novel` / `animated` / `live-action`). Read these before rewriting the system prompt — a substring you remove silently breaks the test.

## Dependencies

- **Runtime nodes** (`nodes*.py`): torchaudio only. All outputs FLOAT or INT.
- **Offline scripts** (`scripts/`): `analysis` optional group (librosa, scipy, Pillow).

Companion custom nodes (used alongside, not imported):
- `<sage_fork_repo>/` — our SageAttention fork. Cross-repo state: `internal/design/sage_backlog.md`.
- `ComfyUI-NativeLooping_testing` (TensorLoopOpen/Close), `ComfyUI-LTXVideo`, `ComfyUI-KJNodes`, `ComfyUI-VideoHelperSuite`, `ComfyUI-MelBandRoFormer` (bypassed by default in shipped workflows; re-enable via two-step manual edit per `scripts/apply_melband_default_off.py`).

## Audio analysis scripts

- `scripts/analyze_audio.py` — ffmpeg-only energy/structure detection, zero Python deps.
- `scripts/analyze_audio_features.py` — librosa: BPM, key, F0, structure, JSON export for LLM (`--scene-diversity`, `--montage`, `--style`). Full guide: `docs/guides/audio_analysis_guide.md`; end-to-end: `docs/guides/prompt_workflow_end_to_end.md`. **Works on generated audio too**: extract via `ffmpeg -i <mp4> -vn -acodec pcm_s16le <wav>` then analyze — primary tool for comparing source vs. generated audio features.
- `scripts/spectrogram_to_reference.py` — Mel spectrogram → PNG frame sequence for IC-LoRA spectrogram-as-reference (Phase 2.0 PoC). **Global normalization runs ONCE in `prepare_mel_for_render`** (do NOT switch to per-frame — washes out beat-amplitude). **Dual-use**: primary use is reference rendering; diagnostic use is visualizing generated audio via `--audio <wav>` (Claude can't hear mp4 streams; spectrograms of the output make audio behavior reviewable). Supports `--colormap {gray,viridis,spectrum}`; B&W triggers vintage-broadcast audio priors in LTX 2.3 — use color for V2A experiments. Design + iteration ladder: `internal/design/spectrogram_reference_design.md`.

## Debug tools

- `scripts/audit_workflows.py [--verbose]` — health audit across all `example_workflows/`: sage, batch-encode, sigma chain, resolution, `(L-1)%8`, preprocess, decoder, F2/F3 symmetry. Exits 1 on ERR. Run after bulk edits. **Intentionally `WorkflowEditor`-independent** — raw `orjson.loads` + inline link scans. Debug tool must stay usable when the editor it audits has a bug; don't DRY these scans against `WorkflowEditor` helpers.
- `scripts/trace_node_source.py <wf> <id> --include-inputs` — resolve any node to AST-extracted source + wiring. Flags `object_patches`, captured tensors, bypasses, widget overrides. **Run before trusting any widget annotation.**
- `scripts/analyze_workflow_dag.py <wf> --format <ascii|mermaid|dot|json>` — topo-sorted execution order.
- `COMFYUI_EXEC_LOG=/tmp/exec.jsonl python <comfyui>/main.py` — runtime per-node JSONL log. Zero overhead unset.
- `scripts/verify_sage_iteration_trace.sh` — diff per-iter sage kernel counts. `AUDIOLOOPHELPER_SAGE_TRACE=auto` is default in `<comfyui>/start.sh`.
- `scripts/sage_telemetry_summary.py --sage-log <path> [--exec-log <path>]` — outside-ComfyUI aggregator. Per-(kernel, mask) median/p90/count + Phase 0 gate verdict. Reads only; does not write.
- **Telemetry / privacy reference**: `docs/reference/telemetry_and_tracing.md` — what the two tracers capture (and don't), where files land, retention, on/off, why prompt text can leak via the exec logger but not via the sage tracer.
- Debug artifacts land in `internal/analysis/runs/` via `timestamped_run_path()` / `timestamped_run_dir()` (`scripts/workflow_utils.py`) by default. With `RUN_ID` env var set (auto-generated by `start_experiment.sh` at the repo root), every logger writes to `data/runs/${RUN_ID}/<category>.jsonl` instead — single shared correlation key across `exec.jsonl`, `sage.jsonl`, `profiler/`, and the VHS output mp4 (when the experiment harness mutates `filename_prefix` to embed the run id). Without RUN_ID, each logger stamps its own filename from `time.time()` at startup and the three drift apart by seconds. Helpers: `scripts/workflow_utils.py::run_artifact_path` (single-file artifacts) / `run_artifact_dir` (multi-file artifacts) / `_current_run_id` (the single `RUN_ID` reader; route every other env-var read for this var through it).
- **Env-var registry**: `docs/reference/environment.md` enumerates every env var the codebase reads, its default behavior, who sets it, and who reads it. **DRY rule**: each env var is read at exactly one helper call site. Audit: `grep -rn 'os.environ\|os.getenv' --include='*.py' .`
- Symptom-first recipes: `docs/guides/debugging_guide.md`.
- **Iter-over-iter drift** → trace CONDITIONING paths in parallel (initial vs loop). Asymmetries (missing `LTXVConditioning`, `frame_rate` mismatch, CLIP in subgraph) are load-bearing bugs.

### Apply scripts

- Default: mutate `example_workflows/audio-loop-music-video_latent.json` in place (accept optional path).
- **Three-tier staging**: `internal/scratch/` (exploratory, gitignored) → `example_workflows/experimental/` (cross-machine reviewable; opt-in to audit via `EXPERIMENTAL_AUDITED_FILES` allowlist in `audit_workflows.py`) → `example_workflows/` (production, "ships AND stabilizes" per `internal/PLAN.md`). Apply scripts are idempotent; `--revert` deletes the staged file. POCs that intentionally break a production invariant (e.g. F3 asymmetry) ship a paired audit check that dispatches on a node-title prefix and ERRs only if the rewire is damaged. Canonical TTC1 pair: `apply_ttc_init_guide_amplification_poc.py` + `ttc1_init_guide_amplification` check.
- **Scratch-build apply scripts** use `WorkflowEditor.from_scratch(output_path)` + `add_top_level_node` + `add_link` — returns an empty-skeleton editor with fresh uuid + reset `last_node_id` / `last_link_id`. No parallel `Builder` class needed. Canonical: `scripts/apply_spectrogram_iclora_minimal.py`.
- **Shared apply-script helpers live in `scripts/_apply_helpers.py`** (`add_link`, `find_node`, `remove_node_and_links`, `find_link_to_slot`, `next_id`, etc.). Import with aliases to preserve call-site names; don't re-define inline.
- **Idempotence**: `md5sum` before + after re-run must match. Guard with `if _is_already_built(wf): return` to avoid burning `last_node_id` on strip-then-readd.
- **Sweep orphan virtual GetNodes** after fork-and-strip. A GetNode whose `widgets_values[0]` matches no live SetNode is orphaned; ComfyUI tolerates it at runtime but it clutters the graph and the dead-wire audit will WARN. Add the ID to `STRIP_IDS` with a categorical comment. Detect via: `[n["id"] for n in wf["nodes"] if n["type"]=="GetNode" and not (n.get("outputs",[{}])[0].get("links") or [])]`.

## Working with Claude across sessions

- **Check sibling-session backlogs (`internal/design/*_backlog.md`) before executing stale PLAN items** that touch defaults. Stale items can silently regress decisions another session has since made.
- **Run `/simplify` after non-trivial code changes.** Three-agent review (reuse / quality / efficiency) catches data-flow correctness bugs that shape-only tests miss.
- **Promote helpers at the 3rd call site, not the 2nd.** Reviewer consensus across multiple `/simplify` passes. Prevents premature extraction: two sites can share a short inline pattern without paying the abstraction cost; by the third, the pattern is load-bearing and the name-plus-tests earn their keep.
- **`PLAN.md` (or feature design doc) is the spec.** When red TDD tests disagree with the spec formula, fix the test — the spec wins unless you explicitly update PLAN first.
- **Decisions-index pattern**: DECISION / WHY / CONTEXT triples, grouped by feature. Template at `internal/ic_lora_assessment.md §6.5`. Roll up any feature >3 commits to avoid re-deriving rationale from git log.
- **LTX 2.3 audio-feature seed variance is ~±20 BPM** for equivalent electronic-genre conditioning. Single-seed comparisons between configs are noise; multi-seed (3-5 per config) needed to detect audio-effect changes. Ref: `docs/experiments/exp_2026-04-24_spectrogram_iclora_v2a.md` §Inferences.
- **Bulk `replace_all` AFTER adding prose to the same file is dangerous.** A retraction note or callout that *quotes* the pattern being replaced (e.g. a note saying "used to use `Cut to a [shot]...`" followed by `replace_all "Cut to a " → "In a "`) sweeps your own freshly-added prose, corrupting the annotation. Add prose annotations AFTER bulk pattern edits, or scope the pattern (only schedule-line matches, only inside fenced blocks). Caught 2026-04-25 in Cut-to retraction; only `/simplify` review caught it.
- **Sibling-session commit race**: when multiple Claude sessions run concurrently, one's `git add` + `git commit` can sweep another's staged changes into its commit with the wrong message. Verify your own commits via `git log -- <file>` (not `git log -1`); the CHANGELOG entry inside the commit is the durable record if the message lies.

## Documentation conventions

- **Active planning lives in gitignored `internal/`.** Promote to `docs/` only when feature ships AND stabilizes.
- **Don't reference `internal/log/` from public-facing docs** — session logs are timestamped/personal. Other `internal/` subdirs (`analysis/`, `design/`, `ic_lora_assessment.md`, `action_items_for_*.md`) are fine to reference from `docs/` if no private prompts/paths leak.
- **Case studies live in `internal/prompts/` (gitignored, unscrubbed).** Public guides distill patterns inline rather than linking out — the parallel scrubbed-copy convention was retired 2026-04-25 (parallel maintenance burden, scrub-leak risk, no confirmed external readership). Reference internal prompt runs from `docs/` only via paraphrase, never via filename.
- **Public docs written for GitHub readers, not our local state.** No "already on disk" / "we use X locally" framing; use `<comfyui_models>` / `/path/to/model` placeholders and list file sources (Hugging Face slugs, upstream repos). `internal/` docs can assume our local state.
- **Breaking changes trigger docs sweep** — add stale phrase to `scripts/validate_docs_consistency.py::STALE_PATTERNS`; `tests/test_docs_consistency.py` fails until fixed.
- **Last-updated date at top of every doc** (`Last updated: YYYY-MM-DD`).
- **Trim public + archive full** for reference docs >1000 lines. Public in `docs/reference/` → summary; full → `internal/archive/` (gitignored).
- **internal skill state is gitignored** — local Claude Code automations only.

## Documentation layout

Public docs: `docs/README.md` (task-first nav) → `docs/guides/` (how-to), `docs/reference/` (deep-dive), `docs/analysis/` (research/postmortems on shipped code), `docs/experimental/` (scaffolded-but-not-validated features paired with workflows in `example_workflows/experimental/`), `docs/experiments/` (per-experiment logs: hypothesis → setup → observations → inferences → next; convention in `docs/experiments/README.md`). `docs/architecture_overview.md` is the single-entry-point architecture reference.

Reference codebases (read-only): `coderef/LTX-2/` (LTX-2 native), `coderef/LTX-Desktop/` (Lightricks Desktop), `<comfyui_custom_nodes>/ComfyUI-LTXVideo/` (ComfyUI LTX integration).

Example workflows (`example_workflows/`): seven shipped — `_image.json`, `_image_adain_perstep.json`, `_latent.json` (primary), `_latent_keyframe.json`, `_latent_stg.json`, `_latent_validator.json`, `_retake.json` (regenerate one section of a prior generation; built by `scripts/apply_audio_loop_retake.py`). All on `AudioLoopHelperSageAttention auto_mask_aware`. Validate via `scripts/audit_workflows.py`.

Internal (gitignored):
- `internal/PLAN.md` — active roadmap.
- `internal/ic_lora_assessment.md` — IC-LoRA phases + decisions index (D1–D18).
- `internal/design/*.md` — long-term designs (`spectrogram_reference_design`, `sage_backlog`, `upscale_workflow_design`).
- `internal/postmortem_*.md`, `internal/prompts/`, `internal/analysis/` — debugging history, unscrubbed case studies, deep dives.
