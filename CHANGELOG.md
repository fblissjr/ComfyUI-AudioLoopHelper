# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).
This project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (workflow layout)
- **Loop subgraph input slot 8 type changed: IMAGE → LATENT.** Formerly
  `num_guides.image_1`, now `guide_latent`. The per-iteration
  `VAEEncode` inside the subgraph (previously re-encoding the same
  init image every iteration) is replaced by one top-level `VAEEncode`
  that encodes the init image exactly once at workflow setup. The
  subgraph's `LTXVAddLatentGuide` now pulls its `guiding_latent`
  directly from the slot-8 input. Saves one video VAE encode per
  loop iteration. **Breaking for saved workflows** with the old
  subgraph shape — re-migrate via
  `scripts/apply_vae_and_cleanup.py` (idempotent).
- **Dead-code cleanup across LATENT example workflows.** Removed:
  `#1590 VAEEncode` + `#1597 LTXVTiledVAEDecode` (mode=4 skeleton from
  the deferred in-workflow upscale path; input unwired),
  `Reroute #618` (dead), `Note #1585` (stale copy of a prompt schedule
  from pre-batch-encode days; schedule now lives on
  `TimestampPromptScheduleBatchEncode.schedule`).
- **Initial-render preview decode upgraded.** `#1318 VAEDecode` →
  `LTXVTiledVAEDecode` (widgets `[2, 2, 1, True, "auto", "auto"]`).
  Defensive: avoids OOM on the preview path when resolution is bumped.
- Applied to: `_latent.json`, `_latent_keyframe.json`, `_latent_stg.json`,
  `_latent_validator.json`, `_image_adain_perstep.json`. The reference
  `_image.json` legacy workflow is untouched.

### Fixed
- **Frame-rate metadata asymmetry between initial render and loop iterations.**
  `TimestampPromptScheduleBatchEncode` emitted raw CLIP conditioning with
  no `frame_rate` metadata, while the initial render's positive
  conditioning passed through `LTXVConditioning` (stamps
  `frame_rate=25`) and the loop body's negative conditioning did too
  (sourced from `Set_base_cond_neg` downstream of the same
  `LTXVConditioning`). Net: loop-iter positive was the only path
  without `frame_rate`, which made the model's temporal scaling
  inconsistent between the initial window and subsequent iterations.
  Symptom: identity drift + hallucinated objects (microphones, altered
  faces) escalating iter-over-iter regardless of `nag_scale` or prompt
  content. Introduced by the 2026-04-22 batch-encode migration
  (moving CLIP out of the loop accidentally dropped the
  `LTXVConditioning` wrapping that used to live in the per-iter chain).
  Fix: `TimestampPromptScheduleBatchEncode` now stamps
  `{'frame_rate': frame_rate}` on each encoded CONDITIONING internally,
  matching `LTXVConditioning`'s behavior. New `frame_rate` widget
  (default 25.0) on the node; cache key includes it; shipped example
  workflows updated to expose the widget. Four tests in
  `tests/test_batch_encode.py::TestFrameRateMetadata` lock the invariant
  in place (default stamp, override, metadata preservation, cache-key
  invalidation). Behavior confirmed against ComfyUI core
  `comfy_extras/nodes_lt.py:416-436` (`LTXVConditioning` is just a
  thin `node_helpers.conditioning_set_values` call stamping frame_rate).

### Changed
- **Docs cleanup round 2 (2026-04-23 PM).** After the structural
  reorg, user critique surfaced ~10 content-level opportunities.
  Eight shipped in one batch:
  - **NAG reference doc de-black-boxed.**
    `docs/reference/nag_technical_reference.md` rewritten 144 → 237
    lines. Five new sections: cross-attention mechanism (why attn2
    not attn1), closure-capture pattern with tensor lifecycle,
    audio_attn2 subsection, NAG×CFG layer interaction,
    troubleshooting. Stale `nag_scale=11` default replaced with
    "start at 5, A/B 3-7 for distilled-1.1." Prominent operational
    constraint callout: "CLIP must not enter the loop body."
    Source citations verified against `ltxv_nodes.py` and
    `model_patcher.py` at current HEAD (agent-assisted). Top-note
    cross-link added to the `nag_object_patches_offload_asymmetry.md`
    postmortem pointing readers to the reference doc first.
  - **LTXVLoopingSampler gap analysis tightened.** Prior "5 blocking
    issues" framing at equal weight replaced with "2 architectural
    root blockers (temporal-schedule mismatch, cross-attention
    trained on joint-not-tiled AV) + 3 type-system cascades." The
    three cascades vanish once NestedTensor is unbound before
    tiling; the root blockers require research-grade work. Citation
    sites updated: `CLAUDE.md:57`, `docs/architecture_overview.md:429`.
  - **LTXVLoopingSampler build guide trimmed and moved.**
    `docs/guides/latent_loop_build_guide.md` (261 lines, detailed
    widget tables for a workflow we don't recommend building for
    music video) → `docs/reference/ltxv_looping_sampler_reference.md`
    (~100 lines, structural reference + pointer to upstream
    ComfyUI-LTXVideo README for widget detail).
  - **Audio-in-prompt docs consolidated + re-contextualized.**
    `docs/analysis/audio_in_prompt_analysis.md` (83 lines) +
    `docs/analysis/audio_in_prompt_guide_notebooklm.md` (36 lines)
    → single `docs/analysis/audio_in_prompt_research.md` (258
    lines). Top framing section explains *when this applies* (audio-
    generating workflows) vs *when it doesn't* (our frozen-audio
    i2v workflow — be concise not detailed). Eight concrete example
    prompts preserved.
  - **Prompt-workflow + LLM-generation-guide merged.**
    `docs/analysis/llm_prompt_generation_guide.md` (~50% duplicative,
    miscategorized in `analysis/`) merged into
    `docs/guides/prompt_workflow_end_to_end.md`. Unique content
    (INFERENCE block, R1-R8 hard rules, ambition tier semantics)
    preserved as a "System prompt reference" section.
    Variation-pattern and troubleshooting sections (duplicative
    with `prompt_creation_guide.md`) dropped with a cross-link.
  - **Lightricks system-prompts framing rewrite.** Banner on
    `docs/reference/ltx23_prompt_system_prompts.md` replaced from
    "Historical reference" → "Why this doc exists / Why our
    workflow diverges / How to use." Explains that the raw
    Lightricks prompts are training-distribution references;
    because LTX 2.3 was trained jointly on audio+video with shared
    cross-attention, our frozen-audio + i2v workflow works better
    with *concise and less detailed* prompts than those raw
    upstream prompts recommend.
  - **Upscale design doc moved to internal/.**
    `docs/guides/upscale_guide.md` →
    `internal/design/upscale_workflow_design.md`. The workflow it
    describes doesn't exist yet; per CLAUDE.md's convention, active
    planning lives in `internal/`. Promotes back to `docs/guides/`
    when the workflow ships.
  - **pipeline_flow_image.md trimmed.** 1923-line node-by-node trace
    of the reference-only IMAGE workflow → 112-line summary (data
    flow + diffs vs the LATENT primary path). Full original archived
    to `internal/archive/pipeline_flow_image_full.md` (gitignored)
    for legacy `_image.json` runs.
  - **Indices refreshed.** `docs/README.md` task-first index +
    per-folder tables and `CLAUDE.md` "Documentation index" section
    both updated to match the new state.
  - Plan + detailed session log in `internal/log/log_2026-04-23.md`.
- **Reorganized `docs/`** into a task-first structure so the right
  doc is obvious from the task. New layout:
  - `docs/README.md` — task-first nav index (new).
  - `docs/architecture_overview.md` — stays at root (advertised
    "START HERE" in `CLAUDE.md`).
  - `docs/guides/` — task-oriented how-to (`prompt_workflow_end_to_end`,
    `prompt_creation_guide`, `audio_analysis_guide`, `debugging_guide`,
    `profiling_guide`, `upscale_guide`, `latent_loop_build_guide`).
  - `docs/reference/` — technical deep-dive (`ltx23_model_reference`,
    `ltx23_prompt_system_prompts`, `nag_technical_reference`,
    `pipeline_flow_image`, `pipeline_flow_latent`, `sampler_reference`,
    `standup_system_prompt`).
  - `docs/analysis/` and `docs/examples/` — unchanged.
  Moved via `git mv` so blame history is preserved. All cross-refs in
  `docs/`, `CLAUDE.md`, root `README.md`, and `nodes.py` updated to the
  new paths. `CHANGELOG.md` historical entries left at their original
  paths (they reflect truth at the time).
- **Renamed `docs/system_prompt.md` → `docs/reference/standup_system_prompt.md`.**
  The bare name collided with `ltx23_prompt_system_prompts.md`; the
  new name states the scope (standup / dialogue) and the role (LLM
  system prompt, not an LTX i2v system prompt).
- **Removed two dead references to `docs/subgraph_latent_rework_guide.md`**
  (file never existed in the current repo). The ref in
  `docs/guides/latent_loop_build_guide.md` now points to
  `docs/architecture_overview.md` + `docs/reference/pipeline_flow_latent.md`.
  The CLAUDE.md entry was dropped.

### Fixed
- **Batch encoder was re-executing per loop iteration (post-ship patch
  of the offload fix below).** `TimestampPromptScheduleBatchEncode`'s
  `stride_seconds` / `audio_duration` inputs come from
  `AudioLoopController`, which itself depends on `current_iteration` —
  so ComfyUI's framework cache invalidated the batch encoder every
  iteration, forcing it to re-encode every unique prompt. User
  observed N `Model LTXAVTEModel_ prepared` lines per loop pass on an
  N-entry schedule. Added module-level LRU
  (`_BATCH_ENCODE_CACHE`) and `IS_CHANGED` classmethod so the batch
  encoder short-circuits on value-stable inputs regardless of
  framework-cache churn. 2 new tests exercising cache-hit and
  invalidation paths.
- **Prompt schedule regression iter 2+ (microphones/style-drift/anatomy
  glitches).** With `TimestampPromptSchedule` active, suppressed
  classes returned after iteration 1. Root cause: per-iteration
  `CachedTextEncode` forced CLIP into VRAM each iteration, which
  triggered `load_models_gpu` → `free_memory` → DiT eviction. On
  reload, LTX2_NAG's `object_patches` closure still pointed at the GPU
  `nag_cond_video` tensor from before the offload, but ComfyUI's
  `model_patches_to()` does not migrate `object_patches` the way it
  migrates `transformer_options["patches"]` (asymmetry at
  `comfy/model_patcher.py:561-580` vs `object_patches` not touched).
  Net effect: NAG silently disengaged after iteration 1 and every
  class it was suppressing leaked back simultaneously — microphones,
  "still image with no motion", "deformed hands", "duplicate
  character/twin/clone", style drift toward photoreal from
  illustrated inits. Fixed by moving all prompt encoding OUTSIDE the
  loop; CLIP now loads exactly once per run. Migrate existing
  workflows via `scripts/apply_batch_encode_fix.py`.

### Changed
- **Every shipped music-video workflow migrated to the batch-encode
  path.** Previously only `_latent.json` had been rewired; now
  `_image.json`, `_image_adain_perstep.json`, `_latent_keyframe.json`,
  `_latent_stg.json`, and `_latent_validator.json` all use
  `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration`.
  `scripts/apply_batch_encode_fix.py` now also sweeps orphaned
  `CachedTextEncode_AudioLoop` / `TimestampPromptSchedule` nodes by
  type (the ID-based pass missed variant-specific IDs — e.g. 'Next
  Prompt Encode' was 1604 in `_image.json` vs 1607 elsewhere).
- **MelBand vocal separation bypassed by default in every workflow**
  (`mode=4` on `MelBandRoFormerModelLoader` and `MelBandRoFormerSampler`)
  AND `Set_actual_audio` explicitly wired from `TrimAudioDuration(567)`
  directly rather than through the bypassed sampler. Explicit wiring
  makes the graph readable without relying on bypass-passthrough slot
  mapping. Apply via `scripts/apply_melband_default_off.py`.
- **`example_workflows/audio-loop-music-video_latent.json` rewired.**
  Primary working baseline now uses the batch-encode path. The inner
  subgraph is unchanged; the only rewire is on the positive
  conditioning feed into `positive` (subgraph input slot 6).

### Added
- **`TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration`
  in `nodes.py`.** The batch encoder runs ONCE outside the loop and
  pre-encodes every per-iteration prompt (deduplicated) into a
  `conditioning_list`. The selector runs inside the loop, indexes the
  list by `current_iteration`, has no CLIP dependency, and cannot
  trigger the offload cycle. Parity with `TimestampPromptSchedule`
  for `snap_boundaries=True/False`. 12 new tests
  (`tests/test_batch_encode.py`).
- **`scripts/apply_batch_encode_fix.py`.** Idempotent workflow
  migration: swaps the legacy `TimestampPromptSchedule` +
  `CachedTextEncode_AudioLoop` x2 + `ConditioningBlend` chain for the
  new batch encoder + selector. Also removes dead wiring discovered
  during the offload-bug investigation: `Set_guider` (stored, never
  Get'd) and `Set_base_cond_pos` + its two `Get_base_cond_pos` nodes
  (stored/retrieved but downstream-unconsumed).
- **`LoopConfigValidator` node (`nodes_validation.py`) + new example
  workflow `audio-loop-music-video_latent_validator.json`.** Shows the
  exact integer-latent math `AudioLoopController` performs and flags
  configurations likely to cause silent failures: invalid length
  (`(length-1) % 8 != 0`), length vs `window*fps` disagreement,
  resolution not div-by-32/64, effective overlap drift from target
  (with reachable values near target), iteration seams landing on
  prompt-schedule boundaries (the `action_prompt6` failure pattern),
  thin-context-on-short-window, audio too short. Outputs diagnostic
  text plus `ok: BOOLEAN` / `warnings: INT` / `errors: INT` /
  `effective_stride_seconds: FLOAT` for downstream gating. 24 new
  tests. Rebuild via `uv run python scripts/apply_config_validator.py`.
- **`_compute_loop_geometry` helper + `LoopGeometry` NamedTuple in
  `nodes.py`.** Single source of truth for the integer-latent stride
  math so `AudioLoopController` and `LoopConfigValidator` cannot drift
  apart.
- **`scripts/validate_docs_consistency.py` + `tests/test_docs_consistency.py`.**
  Grep-based guard against known-stale phrases in `docs/`. Catches the
  pre-2026-04-20 stride formula (`stride = window − overlap`) and raw
  pre-fix stride values (`17.88` / `16.88` / `15.88`) from re-entering
  public prose after the integer-latent quantization fix landed.
  Historical-marker list (`(not 17.88)`, `pre-2026-04-20`, `(continuous
  seconds)`) lets legitimate fix-verification callouts through. 11 new
  tests wired into the standard pytest run; `main()` exit code makes
  the script CI-friendly standalone.
- **`docs/debugging_guide.md` — "Case studies: architectural lessons"
  section.** Three public case studies promoted from internal
  postmortems: stale `noise_mask` leaking across iterations (why
  `LatentContextExtract` / `LatentOverlapTrim` exist), loop-body
  past-end-of-data handling (why `AudioLoopController` clamps
  `start_index`), and shared-conditioning-ancestry execution order
  (why conditioning-path edits need a known-working diff). Each has a
  consistent symptom → investigation → root cause → structural fix →
  transferable lesson shape.
- **`CLAUDE.md` — "Documentation conventions" section.** Four rules:
  active planning docs live in gitignored `internal/`, promote to
  `docs/` only when shipped and stable; case studies come in scrubbed
  public + unscrubbed internal pairs; breaking changes trigger a docs
  sweep via the new validator; last-updated date at top of every doc.
  Prevents the "stale plan orphaned in user docs" class that produced
  `docs/PLAN.md`.
- **`docs/examples/prompt_comedy5.md`.** New public case study on
  adapting the v4 schedule structure to an init image whose subject is
  outside LTX's typical training distribution (oversized cranium,
  floral-patterned clothing). Explains why the subject-block rewrite is
  mandatory (text/image conflict destroys identity anchoring) and how
  to guard against commit-phase head-shape normalization.
- **`--style` flag on `analyze_audio_features.py`.** Choices:
  `cinematic` (default, photoreal anchor), `realistic`, `illustrated`
  (painterly / animated inits), `painterly` (digital painting inits),
  `animated`, `none` (omit prefix entirely when the init image strongly
  commits style). Threads through `_build_prompt_for_section`,
  `get_node_169_prompt`, `generate_schedule_suggestion`,
  `format_json_report`, `format_markdown_report`, and surfaces in
  `workflow_context.style` so LLM-driven schedule generation can apply
  R5 correctly. Fixes the "subject drifts toward photoreal" failure
  mode on non-photoreal init images that `Style: cinematic.` was
  causing.

### Changed
- **Docs audit: stale stride values corrected across 7 docs.** The
  2026-04-20 integer-latent fix changed the effective stride from the
  pre-fix continuous-seconds values (`17.88`, `16.88`, `15.88` at
  overlap 2/3/4) to integer-latent quantized values (`17.92`, `16.96`,
  `16.00`). Propagation lagged in `debugging_guide.md` (6 locations),
  `docs/examples/prompt_comedy{1,2,3,4}.md`, `docs/system_prompt.md`,
  and `docs/prompt_workflow_end_to_end.md`. All updated. Historical
  comparison callouts (`(not 17.88)`, pre-fix section headings) kept
  as-is for the fix-verification narrative.
- **Merged `docs/ltxv_looping_sampler_settings.md` into
  `docs/latent_loop_build_guide.md`.** Both documented the same
  AV-incompatible `LTXVLoopingSampler`; one combined doc replaces the
  pair. Full parameter tuning tables (`temporal_overlap_cond_strength`,
  `adain_factor`, `temporal_overlap`, `temporal_tile_size`) retained
  inline.
- **Moved `docs/PLAN.md` → `internal/PLAN.md`.** Active Phase 1/2/3
  roadmap with pending user-validation work belongs in gitignored
  internal notes, not user-facing docs. Adds a `ACTIVE — Phase 1
  validation pending` banner at the top.
- **Historical banners on `docs/pipeline_flow_image.md` and
  `docs/ltx23_prompt_system_prompts.md`.** Flags the IMAGE loop as
  reference-only (LATENT is primary per CLAUDE.md) and flags the raw
  Lightricks i2v/t2v system prompts as historical (current schedule
  construction starts at `prompt_creation_guide.md` /
  `analysis/llm_prompt_generation_guide.md`).
- **`CLAUDE.md` documentation index pruned.** Removed
  `ltxv_looping_sampler_settings.md` entry (merged), extended the
  prompt_comedy entry to cover v5, and repointed the internal-versions
  pointer at the new `internal/prompts/` subfolder.
- **Canonical camera phrasings in generator output.** `_DYNAMIC_CAMERA_BEATS`
  now only emits byte-exact LTX 2.3 camera keywords from the README
  table (`static camera, locked off shot`, `dolly in, camera pushing
  forward`, `jib up, camera rising up`, `focus shift, rack focus`,
  etc.). Off-list phrasings like `slow dolly in`, `slight focus shift`,
  `slow jib up` are removed — LTX does not reliably follow
  non-canonical camera direction.
- **No more dolly-out in generator output.** `_SECTION_MODIFIERS.OUTRO`
  framing changed from `"In a wide shot, dolly out, camera pulling back"`
  to `"In a close-up, static camera, locked off shot"`. Dolly-out
  shrinks the face over an 18s sampler pass and loses lip-sync
  cross-attention signal; held close-up + audio fade is the safer
  outro. `_DYNAMIC_CAMERA_BEATS.OUTRO` also static-only. Worked
  examples in `_LLM_SYSTEM_PROMPT` updated to match.
- **LLM system prompt rules tightened.** R5 now permits style variation
  based on `workflow_context.style` + explicitly allows `Cut to ...`
  language on entries after the first. R7 rewritten as the canonical
  camera list + hard-bans dolly-out even on OUTRO. R9 references
  `stride_seconds` as the EFFECTIVE stride post-integer-latent
  quantization with the example grid updated to stride=17.92 (the
  actual stride at window=19.88, overlap=2).
- **`workflow_context` JSON fields updated.** `overlap_seconds` split
  into `overlap_seconds_target` (user widget value) and
  `overlap_seconds_effective` (post-quantization — what the node
  actually uses). `stride_seconds` now computed via
  `_effective_stride_seconds` helper that mirrors
  `nodes.AudioLoopController.execute`. New `style` field.

### Fixed
- **Audio VAE loader crash after comfy core commit `ad94d472`** —
  resolved upstream. That commit refactored
  `comfy.ldm.lightricks.vae.audio_vae.AudioVAE` so `__init__` takes
  only `metadata` (weights load via the `VAE` wrapper in
  `comfy/sd.py`). ComfyUI-KJNodes `VAELoaderKJ` still called
  `AudioVAE(sd, metadata)` and crashed with `TypeError:
  AudioVAE.__init__() takes 2 positional arguments but 3 were
  given`. All 5 example workflows use `VAELoaderKJ` for both VAE
  loaders (node ids 1537 video, 1538 audio), so audio-VAE init
  crashed. **Kijai shipped the upstream fix** (KJNodes `6ec4d67`
  "Fix audio VAE for latest comfy") that adapts detection logic and
  dual-paths against both old and new `AudioVAE` APIs. Example
  workflows stay on `VAELoaderKJ` — kijai's implementation has the
  right standalone-LTXV-VAE loading logic that comfy core's generic
  `VAELoader` is still missing (core's LTX detection in
  `comfy/sd.py:616` targets older LTXV 1.x keys, not LTX 2.3).
  `scripts/apply_audio_vae_fix.py` kept in-repo as an emergency
  lever: `--revert` returns to `VAELoaderKJ` (current default);
  running without `--revert` swaps to core `VAELoader` for users
  stuck on a broken KJNodes + post-`ad94d472` comfy combo.
- **`AudioLoopController` lip-sync drift at higher overlap values.** Stride
  was previously computed as `window_seconds - overlap_seconds` (continuous
  seconds), but each iteration's trimmed latent contributes exactly
  `new_latent_frames * 8` pixels to the final decoded video (integer-latent
  quanta). That mismatch accumulated ~0.04s/iter drift at overlap=2 and
  ~0.12s/iter at overlap=4 — up to ~1.3s total desync over a 10-iteration
  3-minute loop, manifesting as lip-sync going completely off. Stride is
  now derived from integer-latent counts so audio advance per iteration
  exactly matches video pixel advance; lip-sync stays aligned for any
  overlap value. User's `overlap_seconds` widget is now a TARGET; outputs
  reflect the EFFECTIVE quantized value (e.g. widget 2.0 → effective 1.96s,
  widget 4.0 → effective 3.88s). Default widget value raised from 1.0 → 2.0
  in `AudioLoopController` schema + both latent example workflows. New
  test file `tests/test_audio_loop_controller.py` (20 tests) locks in
  zero-drift invariants across overlap values 0-5s and iteration counts.

### Changed
- **Default sampler in example workflows is now `euler` (was `euler_ancestral`).**
  On LTX 2.3 distilled at 8 steps with `linear_quadratic, 8, 1`, the schedule
  plateaus at σ ≈ 0.99 for 5 steps before collapsing. Ancestral re-noise scales
  with sigma, so `euler_ancestral` injects near-maximum noise during those
  warmup steps and the 3-step commit phase doesn't have runway to average it
  out — that compounds into subject-identity drift across loop iterations.
  Deterministic `euler` integrates the same ODE cleanly. All 4 example
  workflows updated; `docs/pipeline_flow_*.md`, `docs/prompt_creation_guide.md`,
  and `docs/latent_loop_build_guide.md` updated to match.
- **Default decoder in example workflows is now `LTXVTiledVAEDecode` from
  `ComfyUI-LTXVideo`** (Phase DR1 of the decoder-reliability plan). Spatial-only
  tiling — no temporal tiling at all, so no possibility of mid-video temporal
  tile seams regardless of `AudioLoopController.overlap_seconds`. Eliminates
  the fragile stride-alignment coordination invariant between decoder widgets
  and loop overlap. All 6 `VAEDecodeTiled` instances across the 4 example
  workflows swapped.
- **`scripts/apply_ltx_decoder.py`** (new): idempotent patcher that swaps
  `VAEDecodeTiled` → `LTXVTiledVAEDecode` across workflows. `--revert` flag
  restores the generic decoder with stride-aligned widgets. Round-trip is
  byte-identical. Uses `WorkflowEditor` from `scripts/workflow_utils.py`.
- **`scripts/validate_workflow_decoder.py`** (new): check-only validator.
  Verifies each workflow is either on `LTXVTiledVAEDecode` (preferred) or
  on `VAEDecodeTiled` with widgets aligned to the iteration stride derived
  from `AudioLoopController`. Emits actionable warnings with expected widget
  values when misaligned. Exits non-zero on failure so it can wire into CI.
- `docs/debugging_guide.md`: demoted stride-alignment invariant to fallback-
  only status. LTX decoder is now primary recommendation; stride table
  retained for users who must stay on `VAEDecodeTiled` for VRAM or legacy
  reasons.
- `CLAUDE.md`: noted `LTXVTiledVAEDecode` is the default decoder.

- **Aligned `VAEDecodeTiled` temporal stride with iteration stride across all
  4 example workflows.** Widget values `[512, 64, 64, 8]` → `[512, 64, 512, 64]`.
  Old default produced tile stride 2.24s → ~80 mid-video seams on a 3-min
  run. New values give tile stride 17.92s, matching the iteration stride of
  17.88s at `window_seconds=19.88, overlap_seconds=2`. Decoder seams now
  co-locate with iteration seams (~10 total seam positions instead of ~90).
  Empirically confirmed via test runs. See `docs/debugging_guide.md` for the
  maintenance invariant that must hold if `overlap_seconds` ever changes.
- Shortened negative prompt across all 4 workflows: dropped `"fourth
  character"` (vestige from multi-subject music video defaults, irrelevant
  for most use cases). New text:
  `still image with no motion, subtitles, deformed facial features, extra
  limbs, disfigured hands, duplicate character, twin, clone`

### Added
- `scripts/preprocess_audio_for_ltx.py`: CLI audio preprocessor. Applies a
  5-stage EQ + loudnorm chain tuned for LTX 2.3's audio VAE characteristics
  (16 kHz internal, n_fft=1024, mel_hop=160): HP 80 Hz, −3 dB @ 200 Hz
  (de-boom), −2 dB @ 400 Hz (de-box), +4 dB @ 3 kHz (presence lift — F2/F3
  formants), +3 dB @ 6.5 kHz (sibilance shelf — recover fricatives for
  cross-attention). Outputs WAV to avoid MP3 inter-sample-peak overshoot.
  Prints a before/after spectral-balance + SNR + level table. Addresses
  recurring bass-heavy / dull-sibilance issues in non-music source material
  that hurt lip sync. Requires ffmpeg + the `analysis` dep group.
- `docs/debugging_guide.md`: symptom-first troubleshooting guide covering
  the six layers of quality issues in this pipeline (prompt, decoder tiles,
  iteration seams, schedule-boundary mix, audio, model-intrinsic), with
  per-symptom diagnostic paths, five controlled diagnostic experiments,
  known-good baselines for standup and music variants, and a "things that
  look like bugs but aren't" section. Serves as the troubleshooting
  landing page; cross-references all related docs.

### Fixed
- **`TimestampPromptSchedule` blend_seconds jitter (Phase 1 of the structural
  fix).** Pre-fix, `blend_seconds` was sampled once per loop iteration at
  `current_iteration * stride_seconds`. Values smaller than the loop stride
  (e.g. the then-documented recommendation of 5.0 with stride ~17.88)
  produced a single-iteration "spike" in blend_factor surrounded by zero-
  blend iterations — visible as jitter on one ~18s segment of video. The
  old tooltip prose recommended 5.0, actively leading users into the
  failure mode.
  - New `snap_boundaries` widget on `TimestampPromptSchedule` (default
    `True`) rounds every schedule boundary to the nearest integer multiple
    of `stride_seconds` via new `_snap_schedule_to_iterations` helper.
    Every iteration now runs on exactly one pure prompt — no mid-iteration
    mixed conditioning.
  - New raised-cosine blend ramp (formula: `0.5 * (1 - cos(π * dt))`)
    centered on each boundary, spanning `±blend_seconds/2`. Smooth in
    derivative across multiple iterations when `blend_seconds ≥ stride`.
  - Sub-stride `blend_seconds` (`0 < x < stride_seconds`) is now
    auto-clamped upward to `stride_seconds` with a one-time console
    warning. Smaller values mathematically cannot produce smooth ramps at
    iteration resolution.
  - Legacy "spike" blend preserved behind `snap_boundaries=False` for
    backcompat. `KeyframeImageSchedule` continues to use the spike path
    (no `snap_boundaries` widget yet; candidate for Phase 1.5 follow-up).
  - `_LLM_SYSTEM_PROMPT` gained rule **R9**: schedule timestamps must
    fall on integer multiples of `workflow_context.stride_seconds` so the
    LLM emits pre-snapped schedules (the runtime snap is a safety net).
    Same rule added to `docs/system_prompt.md` for the standup variant.
  - 20 new tests in `tests/test_schedule_snapping.py` covering snap math,
    raised-cosine ramp, auto-clamp, spike backcompat, and node integration.
  - Docs updated: `docs/prompt_creation_guide.md` no longer recommends
    `blend_seconds=5` (it was the worst-case value); documents the clamp,
    the new widget, and the cross-fade recipe. `internal/prompt_comedy1.md`
    corrected with a v1→v2 note.

### Added
- Prompt generation rework in `scripts/analyze_audio_features.py`:
  - Unified `_build_prompt_for_section` is the single source of truth for
    Node 169 and the first schedule entry; they are now byte-exact equal
    by construction (no drift possible). Enforced by new tests.
  - Every prompt now contains an explicit "singing" verb. Single-subject
    uses "is singing ..."; multi-subject (detected via heuristic:
    "two/three/four/both/and/duo/pair" + plural nouns) uses "are singing
    together ...". This keeps LTX 2.3's audio-video joint cross-attention
    signal intact — generic "is performing" kills lip sync.
  - Long sections (>30s normal, >18s in montage mode) are subdivided into
    ~20s chunks (~12s in montage) via `_subdivide_long_sections` so a
    3-minute song produces 7+ entries instead of 4-5, with each dwell
    matching the iteration window.
  - Scene-diversity taxonomy: `--scene-diversity <tier><sub>` with tiers
    1-6 (performance_live → avant_garde, mapped to internal/prompt*.md
    patterns) and sub-letters for mood bundles (3a urban night, 3b
    natural outdoor, etc.). Default: `2a`.
  - `--montage` orthogonal flag: shortens dwell, adds emotional-arc
    language ("the feeling building", "catharsis arriving"), Arcane-style
    music-drives-narrative pacing. Layers on any tier 2-6.
  - Rewritten `_LLM_SYSTEM_PROMPT` with strict schema, hard rules R1-R8,
    an INFERENCE block telling the LLM what the init image already
    encodes (style / palette / setting / subject appearance — DO NOT
    re-describe) vs what the schedule should drive (camera / body /
    lighting shifts / cuts / arc — describe these), tier/sub-letter
    semantics, and three worked examples (single, multi-character,
    montage).
  - `workflow_context` in JSON export now surfaces `scene_diversity`,
    `scene_diversity_tier_name`, `scene_diversity_mood_bundle`, and
    `montage` so the LLM knows the target ambition level.
- `scripts/remove_profiling_nodes.py`: idempotent inverse of
  `apply_profiling_nodes.py`. Round-trip (remove → apply → remove) is
  structurally identity-preserving.
- 15 new tests in `tests/test_audio_features.py` covering singing-verb
  enforcement, multi-subject detection, subdivision behavior, diversity
  tiers, sub-letter mood bundles, montage dwell, inference block in the
  LLM system prompt, and the ambition-tier / montage semantics.

### Changed
- Profiler nodes are now OPT-IN. No example workflow ships with
  ProfileBegin / ProfileIterStep / ProfileEnd wired in. Users who want
  to profile run `scripts/apply_profiling_nodes.py`, run their workflow,
  then `scripts/remove_profiling_nodes.py`. `docs/profiling_guide.md`
  updated to reflect this.
- `torch.profiler.record_function` calls in `CachedTextEncode`,
  `IterationCleanup`, `LatentContextExtract`, and `LatentOverlapTrim`
  are now gated by `_profile_span()` — returns a singleton
  `nullcontext()` when no profiler is active, so instrumented nodes have
  zero overhead in the common case. When profiling IS active, the
  spans appear in the trace as before.
- `TimestampPromptSchedule` (1558) and `AudioLoopPlanner` (1560)
  un-bypassed (mode 4 → 0) in all four example workflows so users get
  the full feature set by default. Bypassed-by-default was an artifact
  of development.

### Added
- End-to-end profiling via three coordinated nodes:
  - `ProfileBegin_AudioLoop`: starts `torch.profiler` before the loop. All
    settings live here (enabled toggle, output dir, warmup/active iteration
    counts, CPU/memory/shapes/flops flags).
  - `ProfileIterStep_AudioLoop`: placed inside the subgraph; calls
    `profiler.step()` to mark iteration boundaries. Zero widgets.
  - `ProfileEnd_AudioLoop`: placed after the loop; stops the profiler and
    writes `trace.json` + `summary.txt` + `memory_timeline.html` to a
    timestamped subdir of `output_dir`.
  - All three become zero-overhead passthroughs when `enabled=False` or
    bypassed. `torch.profiler.record_function` spans added to
    `CachedTextEncode`, `IterationCleanup`, `LatentContextExtract`, and
    `LatentOverlapTrim` so the trace shows named spans for our hot paths.
- `scripts/profile_summary.py`: re-run categorized summary on any saved
  trace without re-running the workflow. Uses orjson per project convention.
- `docs/profiling_guide.md`: user-facing guide for placing the three
  profile nodes, reading the output, and interpreting categorized kernel
  breakdowns.
- 7 new tests in `tests/test_profile_nodes.py` covering disabled-path
  passthroughs, one-time warning behavior, and three-node coordination.

### Changed
- `pyproject.toml`: added `orjson>=3.9` to the `dev` dependency group
  (used by `scripts/profile_summary.py`).

- Two new nodes for reducing per-iteration overhead:
  - `CachedTextEncode_AudioLoop`: drop-in replacement for `CLIPTextEncode`
    with an LRU cache keyed on `(id(clip), text)`. Skips Gemma 3 encoding
    when the same prompt is reused across iterations (common when a
    schedule range spans multiple iterations). Bounded at 20 entries.
  - `IterationCleanup`: LATENT passthrough that calls `gc.collect()` and
    `torch.cuda.empty_cache()` between iterations to reduce allocator
    fragmentation. Three modes: `always` (default), `gpu_only`, `never`.
- `scripts/apply_perf_improvements.py`: idempotent patch script that
  swaps in-loop `CLIPTextEncode` nodes to `CachedTextEncode_AudioLoop` and
  inserts `IterationCleanup` after `LatentOverlapTrim` in the subgraph.
  Applied to `audio-loop-music-video_latent.json`,
  `audio-loop-music-video_latent_keyframe.json`, and
  `audio-loop-music-video_image.json` (CachedTextEncode only -- image
  workflow's subgraph output is IMAGE-typed, no IterationCleanup).
- 13 new tests in `tests/test_cache_nodes.py` covering cache hits/misses,
  LRU eviction, and IterationCleanup mode behavior.

- Three new nodes for per-iteration visual conditioning:
  - `KeyframeImageSchedule`: timestamp-to-image-index schedule, outputs
    image/next_image/blend_factor/current_time/image_index. Mirrors
    `TimestampPromptSchedule` pattern for images.
  - `VideoFrameExtract`: pulls a frame from a reference video batch at
    the current iteration's timestamp. Enables video-to-video style transfer.
  - `ImageBlend`: pixel-space lerp of two images by a factor. Pairs with
    `KeyframeImageSchedule` for smooth keyframe transitions.
- New workflow variant `example_workflows/audio-loop-music-video_latent_keyframe.json`
  (UNTESTED): latent workflow with KeyframeImageSchedule + ImageBlend wired to
  the subgraph init_image. Different reference images per song section.
- `scripts/build_keyframe_workflow.py`: generates the keyframe workflow from
  the base latent workflow via `WorkflowEditor`. Reusable pattern for variants.
- Four analysis reports in `docs/analysis/` (ltx2_native, ltx_desktop,
  comfyui_ltxvideo, kjnodes multi-frame guide capabilities) — surface what
  we can borrow from each codebase for future phases.
- `docs/PLAN.md`: decision-tree plan for Phase 1 validation + conditional
  Phase 2 (multi-guide subgraph) and Phase 3 (retake node) next steps.
- 28 new tests in `tests/test_keyframe_nodes.py` (schedule parsing, matching,
  blend computation, node execute()).

- `docs/prompt_workflow_end_to_end.md`: complete end-to-end walkthrough from
  init image preparation through VLM description extraction, audio analysis,
  LLM schedule generation, and workflow insertion. Includes exact VLM prompts
  for single and multi-person scenes.
- `docs/ltx23_model_reference.md`: extracted LTX 2.3 model behavior reference
  (image guides, latent volume, VAE conversion, AdaIN, conditioning path,
  noise_mask, dual workflow support, extension subgraph, upscaling)

### Changed
- `nodes.py`: schedule helpers deduplicated into generic functions
  (`_parse_schedule_generic`, `_match_schedule_generic`,
  `_match_schedule_with_next_generic`) parameterized by value converter and
  default. Prompt (str) and image (int) variants become thin wrappers.
  Net -13 lines.
- `nodes.py`: added try/except guard around `comfy_api.latest` import with
  `_IOStub`/`_Passthrough` fallback so helper functions and execute() methods
  are testable outside ComfyUI runtime (matches pattern in `nodes_analysis.py`).
- CLAUDE.md reorganized for progressive disclosure (481 -> 154 lines).
  Deep implementation details moved to `docs/ltx23_model_reference.md`.
  CLAUDE.md now focuses on architecture, key patterns, critical constraints,
  gotchas, and a categorized documentation index.
- Removed `docs/latent_loop_workflow_guide.md` (redundant with
  `docs/latent_loop_build_guide.md`, had confusing supersession header)
- Moved raw analysis artifacts to `internal/analysis/` (comfyui_ltxvideo_raw,
  ltx2_native_raw, ltx_desktop_raw)
- Moved superseded `workflow_pipeline_trace.md` to `internal/`
- Added cross-references between docs: VLM extraction prompts linked from
  audio_analysis_guide and llm_prompt_generation_guide; variation patterns
  linked to full examples in prompt_creation_guide; multi-person rules
  linked from end-to-end guide to prompt_creation_guide
- Enhanced JSON export (`-j`): includes `workflow_context` (trim, window, stride,
  subject, image description) and `llm_system_prompt` with all 17 prompt engineering
  rules for the i2v + frozen audio loop workflow. Paste directly into Claude/Gemini.
- New CLI args: `--window`, `--overlap`, `--image-desc` for workflow timing context
- `get_node_169_prompt()`: script now outputs a separate "Node 169" section showing
  exactly what to paste into the initial CLIPTextEncode (matches first schedule entry)
- TimestampPromptSchedule + ConditioningBlend fully wired in all 3 example workflows:
  prompt -> CLIPTextEncode A -> ConditioningBlend.a, next_prompt -> CLIPTextEncode B ->
  ConditioningBlend.b, blend_factor -> ConditioningBlend. Extension subgraph input 6
  rewired from static GetNode to ConditioningBlend output.
- `scripts/patch_scheduling_wiring.py`: one-shot patch script for wiring scheduling
- `docs/analysis/llm_prompt_generation_guide.md`: complete guide for LLM-assisted
  prompt schedule generation with system prompt, user template, and examples
- Per-iteration AdaIN color correction (LTXVAdainLatent) inside Extension subgraph
  for all workflows. Normalizes each iteration's latent statistics against the
  initial render. factor=0.2 default, bypassable. Prevents progressive darkening.
- Per-step AdaIN workflow variant (`audio-loop-music-video_image_adain_perstep.json`).
  Adds LTXVPerStepAdainPatcher to model chain for denoising-time correction.
- `overlap_seconds` output on AudioLoopController (slot 7). Automatically wires
  to LTXVAudioVideoMask video_start_time inside the Extension subgraph. No more
  manual sync when changing overlap.
- Multi-character prompting guide in docs/prompt_creation_guide.md
- AudioLoopPlanner now shows initial render time range with "[uses static prompt,
  not schedule]" annotation, making it clear the schedule only applies to loop iterations
- AudioPitchDetect node: per-iteration vocal pitch detection using torchaudio
  (median F0, has_vocals, is_male_range, is_female_range, vocal_fraction).
  Wire to MelBandRoFormer separated vocals for clean signal.
- `nodes_analysis.py`: separate file for audio analysis runtime nodes
- `_slice_audio_window()` shared helper for extracting iteration audio windows
- `scripts/analyze_audio_features.py`: librosa-based music feature extraction
  (BPM, key detection, chromagram, mel spectrogram, vocal F0, structure segmentation).
  Outputs JSON (for LLM prompt generation), markdown report, and PNG visualizations.
- `--subject` flag on analyze_audio_features.py: generates full LTX 2.3 prompt
  templates with section-appropriate camera, lighting, and energy modifiers.
  Copy-pasteable into TimestampPromptSchedule.
- `pyproject.toml` with `analysis` and `dev` dependency groups
- `tests/test_audio_features.py`: 24 tests for offline feature extraction
- `tests/test_audio_analysis_nodes.py`: 9 tests for runtime AudioPitchDetect
- `tests/conftest.py`: pytest path configuration for scripts/ imports
- `conftest.py` (root): prevents pytest from importing ComfyUI-only `__init__.py`
- LatentContextExtract node: extracts tail latent frames + strips noise_mask
- LatentOverlapTrim node: trims overlap latent frames + strips noise_mask
- StripLatentNoiseMask node: low-level utility for noise_mask removal
- ScheduleToMultiPrompt node: converts TimestampPromptSchedule to MultiPromptProvider format
- overlap_latent_frames output on AudioLoopController
- Latent workflow variant (`audio-loop-music-video_latent.json`) -- UNTESTED
- Pipeline flow documentation for both workflow variants (`docs/pipeline_flow_*.md`)
- Workflow validator agent supporting both image and latent workflows
- `scripts/workflow_utils.py` and `scripts/test_workflow_integrity.py` (moved from
  internal/scripts/ for open-source distribution)
- CHANGELOG.md

### Changed
- `__init__.py` now guards ComfyUI-only import with try/except (allows pytest to run)
- Renamed workflows: removed date-based versioning (v0408/v0409), now
  `audio-loop-music-video_image.json` and `audio-loop-music-video_latent.json`
- AudioLoopController now outputs 7 values (added overlap_latent_frames)
- overlap_latent_frames uses correct formula: `(pixel-1)//8+1` not `pixel//8`
- LatentOverlapTrim clamps overlap to prevent empty tensor edge case

### Fixed
- Latent workflow: overlap_latent_frames now dynamically wired from
  AudioLoopController through subgraph to LatentContextExtract and
  LatentOverlapTrim (was hardcoded to 4)
- Latent workflow: subgraph input 14 now receives latent-space frames
  (from slot 6) instead of pixel-space frames (from slot 5)
- AudioLoopController overlap_latent_frames tooltip: references correct
  downstream nodes instead of stale LTXVSelectLatents
