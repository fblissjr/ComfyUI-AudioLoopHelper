# ComfyUI-AudioLoopHelper

Last updated: 2026-05-04

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
- **Latent ops**: `LatentContextExtract`, `LatentOverlapTrim`, `StripLatentNoiseMask`, `LatentTemporalMask` (retake)
- **Conditioning blend**: `ConditioningBlend` (works with Gemma 3 + CLIP)
- **Attention + profiling**: `AudioLoopHelperSageAttention` (default `auto_mask_aware`), `ProfileBegin`/`IterStep`/`End`
- **Step-skipping cache**: `LTXVideoEasyCache` (experimental, default off)
- **Dimension SSoT**: `LTXFramePlanner` — see `docs/reference/frame_planner_reference.md`

Analysis (`nodes_analysis.py`, torchaudio only): `AudioPitchDetect` → F0 + vocal-fraction; pairs with `ConditioningBlend.blend_factor`.

## Key patterns

- `AUDIO = {"waveform": Tensor, "sample_rate": int}`. Duration = `waveform.shape[-1] / sample_rate`.
- **Stride from integer-latent counts**, not widget seconds: `stride_seconds = (window_latents - overlap_latents) * 8 / fps`. `overlap_seconds` widget is a TARGET; node emits EFFECTIVE quantized overlap. Eliminates lip-sync drift. Tests: `tests/test_audio_loop_controller.py`.
- LTX 2.3 text encoder is Gemma 3, NOT CLIP. Format: `[tensor, {"attention_mask": mask}]`, no pooled.
- Video VAE formula: `latent = (pixel - 1) // 8 + 1`. Not `pixel // 8`.
- `noise_mask=0` = fixed context; `mask=1` = regenerate. Audio is 0; video is 1.
- Guide chaining: multiple `LTXVAddLatentGuide` / `LTXVAddGuideMulti` (up to 20) accumulate via `keyframe_idxs`; `LTXVCropGuides` strips them.
- **CFG-analog amplification of any conditional**: feed `(positive_with_X, positive_without_X)` to `CFGGuider` as `(positive, negative)`. Sampler computes `eps = eps_without + cfg * (eps_with - eps_without)` per step. Generalizes to style LoRAs, identity LoRAs, per-reference ablation. POC: `scripts/apply_ttc_iclora_amplification_poc.py`. Landscape: `internal/analysis/iclora_landscape_analysis.md`.

## Critical constraints

### Audio + latent topology

- **Audio path is sacred.** `LTXVAudioVAEEncode → LTXVConcatAVLatent`; never feed visualizations into the video latent (heatmap frames result).
- **`LTXVAudioVideoMask` (Node 606) wiring is intentional** — `audio_start_time = audio_end_time = window_size` (empty range keeps audio fixed). Don't change.
- **Audio is FROZEN.** Strip music/instrumentation references from schedule prompts; keep diegetic sounds only. Rationale: `docs/analysis/audio_in_prompt_research.md`.
- **Use `LatentContextExtract` / `LatentOverlapTrim`**, not raw `LTXVSelectLatents` — they strip `noise_mask` automatically.

### Sampler + sigma chain

- **Distilled 8-step path.** `ManualSigmas "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"` + `KSamplerSelect euler` + `CFGGuider cfg=1`. **No flow-matching shift node** (no `ModelSamplingSD3`). **No `euler_ancestral*`.** Full walkthrough + Lightricks evidence: `docs/reference/sampler_reference.md`. Migration: `scripts/apply_canonical_sigmas.py`, `scripts/apply_strip_sd3_shift_node.py`.
- **VAE decode**: `LTXVTiledVAEDecode [1,1,1,true,"auto","auto"]` on **24GB+** (single-tile, ~3× faster cold-pass than [2,2,1]); fall back to [2,2,1] on ≤16GB. Apply: `scripts/apply_no_tile_vae_decode.py`. Empirical timings + audit details in `docs/reference/sampler_reference.md`.
- **Don't copy upstream's 15-step sampling** from `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`. Authoritative distilled path: 8 fixed sigmas.

### Conditioning + prompts

- **Verb choice drives cross-attention; generic verbs dilute it. Token budget is shared.** LTX 2.3's audio-video cross-attention binds the visible action to the verb in the prompt — but it's not "singing"-specific (confirmed working with `dancing` and other action verbs when the verb matches what the audio implies). Pick the verb for the action you want: `is singing` / `are singing together` for vocal performance, `is dancing` for movement, `is playing <instrument>` for instrumental. Generic verbs (`performing`, `vocalizing`) dilute the signal. Prompt tokens compete with audio + image cross-attention for budget; **concise > verbose, especially with i2v init** (which carries scene/style for free). Without i2v, text has to do more work and may need more length. Decide where your constraints live. Retracted as a hard "must contain singing" rule 2026-05-04.
- **Use `In a [shot], [camera]` continuation framing for non-first entries — NOT `Cut to a ...`.** Lightricks's official LTX 2.3 system prompt explicitly trains the model to treat scene-cut language as a discontinuation directive. Convention retracted 2026-04-25. Guide: `docs/guides/prompt_creation_guide.md`. Evidence: `docs/reference/ltx23_prompt_system_prompts.md`.
- **Node 169 prompt matches schedule 0:00 entry** structurally (`_build_prompt_for_section` via shared `_prepare_sections`; byte-exact).
- **CLIP must not enter the loop body.** Pre-encode via `TimestampPromptScheduleBatchEncode`; `object_patches` don't survive the offload/reload → silent NAG disengagement iter 2+. Mechanism: `docs/analysis/nag_object_patches_offload_asymmetry.md`.
- **Loop-body CONDITIONING must carry `frame_rate`** (default 25.0). Batch encoder stamps it; any new CONDITIONING-producing loop-body node must too (via `node_helpers.conditioning_set_values`). Missing → identity drift + hallucinated objects iter-over-iter.
- **Illustrated inits drift toward photoreal across iterations** (cross-attention is photoreal-trained). Match init-image style family; or re-anchor via `LTXVAddGuideMulti` per iteration.
- **LTX2_NAG widgets** `[nag_scale, nag_alpha, nag_tau, inplace]`. KJNodes default `scale=11` is aggressive for distilled — dial to 3-7 if initial render freezes. Reference: `docs/reference/nag_technical_reference.md`.

### Resolution + dimensions

- **Dimension config flows from `LTXFramePlanner` (single source of truth).** All shipped workflows wire its outputs to consumers. See `docs/reference/frame_planner_reference.md` for snap rules, latent-volume ceiling, wiring map, migration. Audit: `frame_planner_present` (F8).
- **Resolution div-by-32** (single-stage) or **div-by-64** (two-stage). `scripts/audit_workflows.py` checks.
- **`snap_boundaries=True`** (default) lets `overlap_seconds` change without schedule re-authoring.
- **Iterations auto-track audio length.** `AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in` is wired in every shipped workflow. For short tests, drag in an `INTConstant` and rewire — recipe in `docs/guides/debugging_guide.md`. Audit: `iterations_autowired` (F5).

### Workflow JSON discipline

- **Always use `WorkflowEditor`** (`scripts/workflow_utils.py`) for JSON edits. Apply-script + audit-pair conventions: see `scripts/CLAUDE.md`.
- **Never name an INT widget exactly `"seed"` or `"noise_seed"`.** ComfyUI's frontend auto-attaches a `control_after_generate` dropdown to those literal names, which silently mutates the saved widget value across runs even when the input is wired. Use `base_seed`, `seed_in`, etc. Guard: `tests/test_node_schemas.py::test_no_seed_or_noise_seed_named_inputs`.
- **A schema rename is not enough — strip leftover widget values too.** Example: F4 `seed`→`base_seed` rename also stripped a leftover `'randomize'` that ComfyUI's positional widget pop would otherwise have shifted into the `fps` slot. Companion: `scripts/apply_strip_alc_control_after_generate.py`. Audit: `alc_widget_drift` (F6).
- **Don't ship two schema changes that touch the same iteration-state plane in one session.** When adding an auto-wire that closes a control loop, walk every existing edge between the involved nodes and confirm none of them produces a cycle. Audit: `planner_no_stride_input` (F7).
- **Bake new topology constraints into `audit_workflows.py`.** Every fix that ships an apply script ships a matching audit check. F-pair inventory + remediation pointers: `docs/reference/debug_tools.md`.

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
- **`LTXVConcatAVLatent` isn't buggy.** Two `output.update(...)` lines are dead writes; the `NestedTensor` assignment that follows is load-bearing. Don't chase. Full investigation: `internal/postmortem_concat_av_latent_investigation.md`.
- Validate after edits: `python3 -c "import json; json.load(open('file.json'))"`.
- **TensorLoop framework-cache invalidation is transitive.** Any node downstream of `current_iteration` re-executes per iter. Memoize via `id()`-keyed LRU + `IS_CHANGED`.
- **LTX has no image VAE encode node.** For image→latent, use core `VAEEncode`.
- **KJNodes ships `GetImageRangeFromBatch` and `SimpleCalculatorKJ`.** Compose these before building custom slicer or math nodes. Grep `ComfyUI-KJNodes/__init__.py` registry before designing new utility nodes.
- **No `.py` edits to ANY file in this package while a render is in flight.** ComfyUI-HotReloadHack reloads the entire package on any `.py` change, invalidating Inductor autotune state. CPU-only edits to docs / scripts / `internal/scratch/*.json` / non-package files are safe.
- **Always `git status --short` before `git commit`.** Pre-staged files get swept into your commit otherwise. Scrub workflows before open-sourcing.

## Init image conditioning + IC-LoRA paths

- **Initial render**: `#531 LTXVImgToVideoInplaceKJ` writes encoded init into frame 0; `noise_mask=0` locks it.
- **Loop iterations**: top-level `VAEEncode → subgraph slot 8 → #1519 LTXVAddLatentGuide latent_idx=-1`. Init encoded ONCE.
- **F2 + F3 are MANDATORY symmetry rules** for the init-image path: both initial and loop branches consume the SAME `LTXVPreprocess(img_compression=18)` output (F2); loop CFGGuider positive/negative come from `LTXVCropGuides`, not `LTXVAddLatentGuide` directly (F3). Skipping either is the photoreal-drift / identity-drift footgun. Apply: `scripts/apply_loop_guide_preprocess_symmetry.py` + `scripts/apply_loop_cropguides_symmetry.py`. Full trace: `docs/reference/pipeline_flow_latent.md`.
- **F12 video-reference IC-LoRA**: companion to F2/F3, adds an IC-LoRA guide inside the subgraph between `#1519` and the F3 cropguides chain. F2 + F3 ref-video symmetry rules apply to the ref-video chain too. Apply + decisions + flag reference: `scripts/apply_iclora_video_reference.py` + `internal/ic_lora_assessment.md` D19–D23.

## Working with Claude across sessions

- **GPU contention check before any bench/render.** `mtime` of `data/runs/*/*/sage.jsonl` (per-prompt routing) within last few minutes ⇒ a sibling-repo render is likely active. Ask before starting GPU work.
- **`AUDIOLOOPHELPER_PER_PROMPT=1` is default in `start_experiment.sh`** (since 2026-05-01). Artifacts route under `data/runs/${RUN_ID}/${prompt_id}/`. Reader scripts auto-detect both layouts.
- **Run `/simplify` after non-trivial code changes.** Three-agent review (reuse / quality / efficiency) catches data-flow correctness bugs that shape-only tests miss.
- **Verify a new model via its paper, not its name.** Run `paper_search` / fetch README before designing around assumptions. Cost ~30s; saves entire sessions.
- **Promote helpers at the 3rd call site, not the 2nd.** Two sites can share inline; the third earns the abstraction.
- **`PLAN.md` (or feature design doc) is the spec.** When red TDD tests disagree with the spec formula, fix the test — the spec wins unless you explicitly update PLAN first.
- **Decisions-index pattern**: DECISION / WHY / CONTEXT triples, grouped by feature. Template at `internal/ic_lora_assessment.md`. Roll up any feature >3 commits.
- **LTX 2.3 audio-feature seed variance is ~±20 BPM** for equivalent electronic-genre conditioning. Single-seed comparisons are noise; multi-seed (3-5 per config) needed.
- **Record the prior in writing BEFORE the measurement.** A rough Amdahl derivation commits a prediction the result can grade against. "Did the prior hold?" is more useful than "what was the number?"
- **Measure the boundary you actually patch**, not the boundary your model predicts. Sage e2e was +17 points above the strict-attention Amdahl prediction because int8 amortization reaches into FFN-adjacent sampler work. Specific bench numbers: `internal/analysis/empirical_bench_findings.md`.
- **Check sibling-session backlogs (`internal/design/*_backlog.md`) before executing stale PLAN items** that touch defaults.
- **Project-level `settings.json` hook config is loaded once at session start** — deleting a hook script mid-session leaves the cached config trying to run a missing file, blocking every Write/Edit until session restart. Workaround: use Bash for post-deletion edits.
- **Marketplace plugin cache lags behind merged plugin changes.** To pick up freshly-merged plugin changes immediately, re-run the plugin's `install-git-hooks.sh` from a workspace clone of the plugin repo.
- **Cross-repo coordination**: when an optimization target lives in a sister repo (current: sage-fork), use the `cross-repo-handoff` skill (bilateral memo files, seen-marker discipline). Pattern reusable for any future sister-repo co-optimization.

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

Example workflows: eight shipped on `AudioLoopHelperSageAttention auto_mask_aware`. Validate via `scripts/audit_workflows.py`.

Subtree CLAUDE.md files (auto-loaded when working in that subtree):
- `scripts/CLAUDE.md` — apply-script conventions, audit invariants, WorkflowEditor patterns.
- `tests/CLAUDE.md` — pytest invocation, AST patterns, fakes hierarchy.
- `internal/autoresearch/CLAUDE.md` — experiment-runner framework (target-agnostic).
- `.claude/CLAUDE.md` — harness conventions + CLAUDE.md governance policy.

Internal (gitignored): `internal/PLAN.md`, `internal/TODO.md`, `internal/ic_lora_assessment.md`, `internal/design/*.md` (long-term designs), `internal/autoresearch/`, `internal/scripts/` (out-of-repo deploy sources), `internal/postmortem_*.md`, `internal/prompts/`, `internal/analysis/`, `internal/log/log_YYYY-MM-DD.md` (session logs).

## Pending review (last drained: 2026-05-04)

<!--
Capture-then-review staging area. New findings (via `#`-key or otherwise)
land HERE, not inline above. Drained on each curation pass: most demote
to internal/ archive, some promote to scripts/audit_workflows.py or a
test, few earn a slot in the curated body. Policy: .claude/CLAUDE.md
"CLAUDE.md governance". Update the "last drained" date above when you
finish a curation pass.
-->

(empty)
