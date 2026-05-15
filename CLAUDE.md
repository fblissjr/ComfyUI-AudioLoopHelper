# ComfyUI-AudioLoopHelper

Last updated: 2026-05-12

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

# Apply script — typical shape (every apply_*.py supports these)
uv run --group dev python scripts/apply_<X>.py            # apply
uv run --group dev python scripts/apply_<X>.py --dry-run  # show changes
uv run --group dev python scripts/apply_<X>.py --revert   # undo
```

Add `--group experiments` for autoresearch contract tests (`tests/test_autoresearch.py`); they skip on clones without duckdb. Subtree CLAUDE.md files cover deeper conventions: working in `scripts/`, `tests/`, or `internal/autoresearch/` loads the matching subtree CLAUDE.md automatically.

## Architecture

Runtime files: `nodes.py` (core loop), `nodes_analysis.py` (torchaudio audio analysis), `nodes_sage.py` (sage attention), `nodes_validation.py` (config validator). Entry point: `comfy_entrypoint()` in `nodes.py`.

Core nodes (per-node role + wiring in each class's docstring; full reference at `docs/reference/ltx23_model_reference.md`):

- **Loop spine**: `AudioLoopController`, `LoopIterationStamp`, `IterationCleanup`, `AudioLoopPlanner`, `AudioDuration`
- **Prompt schedule**: `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` (current) / `TimestampPromptSchedule` + `CachedTextEncode` (legacy; don't wire in loop body)
- **Keyframe schedule**: `KeyframeLatentScheduleBatchEncode` + `LatentSelectByIteration` (current — VAE-encodes once outside loop) / `KeyframeImageSchedule` + `ImageBlend` (legacy; per-iter VAE)
- **Latent ops**: `LatentContextExtract`, `LatentOverlapTrim`, `LatentTemporalMask` (retake; `edge_taper_seconds` for soft boundary), `LatentSeamZoneMask` (multi-band mask centered on iteration boundaries — pairs with `scripts/diagnose_overlap_seams.py`), `LatentFrameCount` (sizes empty audio latent for upscale + seam), `TrimImageBatchToAudio` (F14), `TrimVideoLatentToAudio` (A/B staged), `LTXHeadTrim` (image+audio composite — drops first N latent-frames' pixel + matching audio span; default 0 = no-op), `RunIdPrefix` (F15)
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
- **VAE decode**: `LTXVTiledVAEDecode [1,1,1,true,"auto","auto"]` on **24GB+** (single-tile, ~3× faster cold-pass than [2,2,1]); fall back to [2,2,1] on ≤16GB. Apply: `scripts/apply_no_tile_vae_decode.py`. Empirical timings + audit details in `docs/reference/sampler_reference.md`.
- **Don't copy upstream's 15-step sampling** from `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`. Authoritative distilled path: 8 fixed sigmas.

### Conditioning + prompts

- **Verb choice drives cross-attention; generic verbs dilute it. Token budget is shared.** LTX 2.3's audio-video cross-attention binds the visible action to the verb in the prompt — but it's not "singing"-specific (confirmed working with `dancing` and other action verbs when the verb matches what the audio implies). Pick the verb for the action you want: `is singing` / `are singing together` for vocal performance, `is dancing` for movement, `is playing <instrument>` for instrumental. Generic verbs (`performing`, `vocalizing`) dilute the signal. Prompt tokens compete with audio + image cross-attention for budget; **concise > verbose, especially with i2v init** (which carries scene/style for free). Without i2v, text has to do more work and may need more length. Decide where your constraints live. Retracted as a hard "must contain singing" rule 2026-05-04.
- **Use `In a [shot], [camera]` continuation framing for non-first entries — NOT `Cut to a ...`.** Lightricks's official LTX 2.3 system prompt explicitly trains the model to treat scene-cut language as a discontinuation directive. Convention retracted 2026-04-25. Guide: `docs/guides/prompt_creation_guide.md`. Evidence: `docs/reference/ltx23_prompt_system_prompts.md`.
- **Node 169 prompt matches schedule 0:00 entry** structurally (`_build_prompt_for_section` via shared `_prepare_sections`; byte-exact).
- **CLIP must not enter the loop body.** Pre-encode via `TimestampPromptScheduleBatchEncode` outside; `ConditioningSelectByIteration` plucks per-iter inside. Mechanism + cache + failure modes: `docs/reference/timestamp_prompt_schedule_batch_encode.md`.
- **Loop-body CONDITIONING must carry `frame_rate`** (default 24.0; LTX 2.3 trained at 24fps — `LTXVConditioning.frame_rate` scales the temporal pos embed at `comfy/ldm/lightricks/av_model.py:866`). Batch encoder stamps it; new CONDITIONING-producing loop-body nodes must too via `node_helpers.conditioning_set_values`. Missing → identity drift iter-over-iter.
- **Illustrated inits drift toward photoreal across iterations** (cross-attention is photoreal-trained). Match init-image style family; or re-anchor via `LTXVAddGuideMulti` per iteration.
- **LTX2_NAG widgets** `[nag_scale, nag_alpha, nag_tau, inplace]`. KJNodes default `scale=11` is aggressive for distilled — dial to 3-7 if initial render freezes. Reference: `docs/reference/nag_technical_reference.md`.

### Resolution + dimensions

- **Dimension config flows from `LTXFramePlanner` (single source of truth).** All shipped workflows wire its outputs to consumers. See `docs/reference/frame_planner_reference.md` for snap rules, latent-volume ceiling, wiring map, migration. Audit: `frame_planner_present` (F8).
- **Resolution div-by-32** (single-stage) or **div-by-64** (two-stage). `scripts/audit_workflows.py` checks.
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
- **Nodes that call `model.state_dict()` constrain model-mutation order.** Canonical order for compile-style patches: `UNETLoader → ... → LTXICLoRALoaderModelOnly → <module-mutating node> → SetNode "model"`.
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
- Validate after edits: `python3 -c "import json; json.load(open('file.json'))"`.
- **TensorLoop framework-cache invalidation is transitive.** Any node downstream of `current_iteration` re-executes per iter. Memoize via `id()`-keyed LRU + `IS_CHANGED`.
- **LTX has no image VAE encode node.** For image→latent, use core `VAEEncode`.
- **KJNodes ships `GetImageRangeFromBatch` and `SimpleCalculatorKJ`.** Compose these before building custom slicer or math nodes. Grep `ComfyUI-KJNodes/__init__.py` registry before designing new utility nodes.
- **No `.py` edits to ANY file in this package while a render is in flight.** ComfyUI-HotReloadHack reloads the entire package on any `.py` change, invalidating Inductor autotune state. CPU-only edits to docs / scripts / `internal/scratch/*.json` / non-package files are safe.
- **Always `git status --short` before `git commit`.** Pre-staged files get swept into your commit otherwise. Scrub workflows before open-sourcing.
- **Concurrent unstaged edits get bundled into your commit.** When a file you intend to edit already shows ` M` in `git status`, `git add <file>` stages BOTH the prior unstaged changes AND your edits — your commit captures both. Either `git stash` first, or `git diff --cached -- <file>` post-stage to verify scope when you only meant to land your own edits.

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
- **`AUDIOLOOPHELPER_PER_PROMPT=1` is default in `start_experiment.sh`** (since 2026-05-01). Artifacts route under `data/runs/${RUN_ID}/${prompt_id}/`. Reader scripts auto-detect both layouts.
- **Run `/simplify` after non-trivial code changes.** Three-agent review (reuse / quality / efficiency) catches data-flow correctness bugs that shape-only tests miss.
- **Behavioral-regression debugging starts with workflow diff, not code.** Extract embedded workflows from the working + broken PNGs via `scripts/extract_workflow_from_png.py <png> -o <json>` (positional arg, no `--workflow` flag) and structural-diff. Code-level analysis after.
- **Magic string vs semantic literal**: extract opaque magic (`"v2"`, `"7"`) eagerly; leave semantic literals (`"DEBUG"`, `"private clone"`, `"singing"`) inline until 3rd call site or drift risk. `/simplify` reviewers flag both as "magic strings" — false-positive class on semantic literals.
- **Grep `internal/design/`, `internal/analysis/`, and the gitignored `PLAN.md` BEFORE forming recommendations** on any topic that smells like prior territory (trigger words: upscale, retake, IC-LoRA, sigma profile, edit anything, spatial, polish, seam). Most apparent "session-1 discoveries" are previously analyzed; re-derivation costs 2-3 turns. Subagent briefs should include "check internal/ first" — agents see the filesystem but not session context.
- **Verify a new model via its paper, not its name.** Run `paper_search` / fetch README. Cost ~30s.
- **Promote helpers at the 3rd call site, not the 2nd.** Two sites can share inline; the third earns the abstraction.
- **`PLAN.md` (or feature design doc) is the spec.** When red TDD tests disagree with the spec formula, fix the test — the spec wins unless you explicitly update PLAN first.
- **Decisions-index pattern**: DECISION / WHY / CONTEXT triples, grouped by feature. Template at `internal/ic_lora_assessment.md` (private clone only). Roll up any feature >3 commits.
- **LTX 2.3 audio-feature seed variance is ~±20 BPM** for equivalent electronic-genre conditioning. Single-seed comparisons are noise; multi-seed (3-5 per config) needed.
- **Record the prior in writing BEFORE the measurement.** A rough Amdahl derivation commits a prediction the result can grade against. "Did the prior hold?" is more useful than "what was the number?"
- **Measure the boundary you actually patch**, not the boundary your model predicts. Sage's e2e gain is workload-dependent and per-workload Amdahl-captured: 1.22× on audio_loop_latent (8.2% wall-time attention share, v0.5.1 bench), 1.41× on iclora (42% CUDA-time attention share, 3.08× per-call kernel speedup; predicted 1.39× matched within 1.4%). Don't generalize one workload's kernel ratio to another. **Non-Amdahl-gap claims require both arms directly measured** — the prior "+17pt FFN-adjacent reach" framing was a single-data-point inference (sage-off attention never traced) and was retired 2026-05-07 across three falsified hypotheses (FFN-adjacent / launch-overhead / cache-footprint). Specific bench numbers: `internal/analysis/empirical_bench_findings.md` (private clone only).
- **torch profiler op-dispatch and kernel rows are nested, not parallel.** `key_averages()` lists `namespace::op_name` and `void kernel<...>` as separate entries; their Self CUDA times are nested (op self-CUDA covers the kernel work plus small auxiliary kernels). Summing double-counts. Anchor on kernel rows for CUDA time; treat op rows as call-site metadata. Mismatched call counts between op and kernel rows (e.g. 4608 vs 3456) are the tell — a single op can't fire <1 kernel/call on average.
- **Check sibling-session backlogs (`internal/design/*_backlog.md`) before executing stale PLAN items** that touch defaults.
- **Project-level `settings.json` hook config is loaded once at session start** — deleting a hook script mid-session leaves the cached config trying to run a missing file, blocking every Write/Edit until session restart. Workaround: use Bash for post-deletion edits.
- **Marketplace plugin cache lags behind merged plugin changes.** To pick up freshly-merged plugin changes immediately, re-run the plugin's `install-git-hooks.sh` from a workspace clone of the plugin repo.
- **Cross-repo coordination**: sister repos split two ways — sister fork (current: `sage-fork`, patches an upstream library) vs companion umbrella (`comfy-workbench`, cross-workload meta-harness + glue). Taxonomy + bootstrap + decision table: `internal/design/sister_repo_taxonomy.md` (private clone only). Bilateral memo channel: `cross-repo-handoff` skill.

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

Example workflows: ten shipped on `AudioLoopHelperSageAttention auto`. Validate via `scripts/audit_workflows.py`.

Subtree CLAUDE.md files (auto-loaded when working in that subtree):
- `scripts/CLAUDE.md` — apply-script conventions, audit invariants, WorkflowEditor patterns.
- `tests/CLAUDE.md` — pytest invocation, AST patterns, fakes hierarchy.
- `internal/autoresearch/CLAUDE.md` — experiment-runner framework (target-agnostic; gitignored, only loads on private clone).
- `.claude/CLAUDE.md` — harness conventions + CLAUDE.md governance policy.

Internal (gitignored): `internal/PLAN.md`, `internal/TODO.md`, `internal/ic_lora_assessment.md`, `internal/design/*.md` (long-term designs), `internal/autoresearch/`, `internal/scripts/` (out-of-repo deploy sources), `internal/postmortem_*.md`, `internal/prompts/`, `internal/analysis/`, `internal/log/log_YYYY-MM-DD.md` (session logs).

## Pending review (last drained: 2026-05-10)

<!--
Capture-then-review staging area. New findings (via `#`-key or otherwise)
land HERE, not inline above. Drained on each curation pass: most demote
to internal/ archive, some promote to scripts/audit_workflows.py or a
test, few earn a slot in the curated body. Policy: .claude/CLAUDE.md
"CLAUDE.md governance". Update the "last drained" date above when you
finish a curation pass.
-->

- **Memory benchmarks must track offload pressure, not just peak allocation.** ComfyUI's dynamic VRAM loader (`comfy-aimdo`) shuffles model weights to CPU under memory pressure — renders go slower instead of crashing, so snapshot-allocation gates miss the actual cost. Full methodology + tooling pointers (`scripts/bench_aimdo_vram.py`, `scripts/analyze_sage_traces.py`) + A/B recipe + failure modes at `docs/reference/benchmarking_memory_pressure.md`. Related gotcha: audio-loop workflows cannot exercise LTX 2.3's masked self-attn path — something in the conditioning route (likely `LTXVCropGuides` or `LTXVConcatAVLatent`'s NestedTensor packing) strips `guide_attention_entries` before `_process_input` builds the mask. For masked-path code (kernel benches, mask-correctness checks), use `example_workflows/benchmark_workflows/fml2v_sage_masked_attn_benchmark.json` instead.
