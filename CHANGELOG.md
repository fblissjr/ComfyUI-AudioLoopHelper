# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).
This project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Companion-repo coordination.** Root `CLAUDE.md` and `README.md` now
  document the two-bucket sister-repo scheme (sister fork vs companion
  umbrella). Adds a second sister to the workflow alongside the existing
  attention fork.

### Changed
- **`scripts/` cleanup pass.** Moved 10 baked-in / superseded apply
  scripts to `scripts/archive/`. Added `scripts/archive/CLAUDE.md`
  (per-script inventory: original purpose + reason archived) and an
  inventory section to `scripts/CLAUDE.md` (what each remaining
  script does today, who calls it, dedup analysis). Audit + budget
  + docs-consistency suites pass; no remediation pointers broken.
- **`scripts/audit_workflows.py` calibration for non-loop workflows.**
  Added `_is_loop_workflow(by_type)` helper that detects loop topology
  by presence of any `TensorLoopOpen` / `TensorLoopClose` /
  `AudioLoopController` node. Loop-only invariants (`sage_node`,
  `iteration_stamp`, `frame_planner_present`, `prompt_schedule`, F2/F3
  init-image symmetry, `_check_prompt_relay_wiring`,
  `_check_ltx2_nag_reaches_loop`, `_check_iclora_video_reference_wiring`,
  `_check_audio_latent_slice_*`, `_check_initial_render_audio_duration_wired`,
  `_check_overlap_seconds_single_source`,
  `_check_vhs_video_combine_frame_rate_parity`) silently skip on
  workflows without that topology. Generic invariants
  (`graph_acyclic`, `link_integrity`, `widget_shape`, resolution /
  length / volume checks, `_check_no_sd3_shift_node`) and the retake
  check stay ungated. The `manual_sigmas` warning message reworded
  from a raw widget dump to "non-canonical sigma profile" so a
  deliberately-low-σ tail (e.g. the 3-step refine in upscale and
  seam-refinement workflows) reads as informational rather than
  broken. Effect: the new post-loop polish workflows
  (`internal/workflows/upscale_loop_output.draft.json`,
  `seam_zone_refinement.draft.json`) audit clean (0 ERR). Existing
  loop workflows unchanged.

- **Workflow consolidation.** Three shipped variants merged into one:
  `audio-loop-music-video_latent_intro.json`,
  `audio-loop-music-video_latent_iclora.json`, and
  `audio-loop-music-video_latent_iclora_audio_pre_encode.json` were all
  strict-subset shapes of the intro variant. Renamed intro →
  `audio-loop-music-video_latent.json` (the new canonical default) and
  removed the other two. The new default ships with pre-encoded audio,
  IC-LoRA scaffolding bypassed (un-bypass to enable visual reference
  adapters), two LoRA loaders bypassed (un-bypass distill LoRA when
  running base ltx-2.3 dev), and the 9-group two-row layout. Apply
  scripts whose source/output workflows were removed
  (`apply_audio_latent_pre_encode.py`, `apply_iclora_video_reference.py`)
  retired with a "baked into canonical" docstring note. Apply scripts
  whose `DEFAULT_TARGETS` referenced the removed files updated to point
  at the new default. Tests for the retired scripts removed (audit pairs
  cover the topology invariants).

### Fixed
- **`overlap_seconds` divergence footgun across workflows.** Both
  `AudioLoopController.overlap_seconds` and
  `AudioLoopPlanner.overlap_seconds` shipped as widget-only with default
  2. Direct controller→planner wiring re-introduces F7's cycle. Fix:
  shared `FloatConstant("overlap_seconds")` with no upstream inputs,
  wired to both. Apply script:
  `apply_overlap_seconds_single_source.py`. Audit pair:
  `overlap_seconds_single_source` (ERR if both consumers don't share a
  source). Applied to 8 shipped workflows + 1 experimental POC.

- **Audit-only enforcement: `VHS_VideoCombine.frame_rate` parity with
  `LTXFramePlanner.fps`.** `VHS_VideoCombine.frame_rate` lives in a
  dict-shaped `widgets_values` (not a converted input), so it can't be
  cleanly auto-wired. Audit check `vhs_frame_rate_matches_planner` ERRs
  when the two values diverge. Manual remediation only — edit the
  widget value to match the planner.

- **Lip-sync drift in pre-encode workflows.** Three `AudioLatentSlice`
  widgets (`source_seconds`, `start_seconds`, `duration_seconds`) and
  `#601 TrimAudioDuration.duration` were widget-driven defaults that
  silently mismatched per-song audio geometry, causing each iteration's
  audio slice to misalign with the corresponding video frames. All four
  now wire to their canonical sources:
  `AudioLoopController.audio_duration`, `start_index`, and
  `LTXFramePlanner.actual_seconds`. The bug had been latent since the
  pre-encode topology shipped — only songs coincidentally ~300s with
  default everything appeared to work. Verified end-to-end on user
  render. Apply scripts:
  `apply_audio_latent_slice_source_seconds_autowire.py`,
  `apply_audio_latent_slice_iter_wiring_fix.py`,
  `apply_initial_render_audio_duration_autowire.py`. Audit pairs ERR
  with concrete remediation pointers on regression
  (`audio_latent_slice_source_seconds_wired`,
  `audio_latent_slice_iter_wiring`,
  `initial_render_audio_duration_wired`).

### Added
- **`LatentTemporalMask.edge_taper_seconds`** — optional cosine taper at
  retake-section boundaries. Default `0.0` keeps the historical
  hard-mask output bit-identical (regression guard); non-zero values
  ramp the noise_mask `0 → 1` over the taper window at the leading
  edge and `1 → 0` at the trailing edge so a downstream inpainting
  sampler blends the regenerated region into surrounding context
  instead of hitting a hard step. Taper width is clamped to half the
  retake range so leading and trailing ramps never overlap. Existing
  retake workflows pick up the new input slot transparently
  (ComfyUI's loader fills missing widget values from the schema
  default). Behavioral coverage: `tests/test_retake_nodes.py` (5 new
  tests). AST guard: `tests/test_node_schemas.py::test_latent_temporal_mask_edge_taper_default_is_zero`
  catches rename, removal, default change, and accidental duplication
  across node modules.

- **`scripts/apply_lanczos_init_preprocess.py`** — applies a
  supersample-then-decimate two-stage lanczos preprocess in front of
  the init-image resize node. With a single-pass downscale from a
  much-larger source, residual aliasing on faces, text, and fine
  textures shows in the encoded latent. The two-stage pass —
  supersample to 2× target via lanczos, then decimate via the existing
  target-dim resize — gives the second pass's anti-alias kernel
  enough samples to integrate properly. No-op when source ≤ target.
  Stages the variant to `internal/workflows/loop_with_lanczos_preprocess.draft.json`;
  does not mutate shipped workflows. Idempotent (signature-checked
  via node title), `--revert`, `--dry-run`. Drift-sync flow:
  `--revert && apply` regenerates the draft from the current source,
  picking up any upstream bug fixes automatically.

- **`scripts/build_upscale_workflow.py`** — builds the post-loop
  spatial-upscale workflow from scratch (24 nodes, 32 links) at
  `internal/workflows/upscale_loop_output.draft.json`. Topology:
  `VHS_LoadVideo → VAEEncode → LTXVLatentUpsampler (2×) →
  LTXVImgToVideoConditionOnly → LTXVConcatAVLatent → SamplerCustomAdvanced
  (3-step σ-tail [0.85, 0.7250, 0.4219, 0.0], euler, CFG=1) →
  LTXVSeparateAVLatent → LTXVCropGuides → LTXVTiledVAEDecode →
  VHS_VideoCombine`. Original audio passes through directly to
  combine without re-encoding. Idempotent; `--dry-run` prints the
  node table without writing, `--revert` deletes the output. Loader
  names + sigma profile + frame rate centralized as constants at the
  top of the script.

- **`LatentSeamZoneMask`** — node companion to `LatentTemporalMask`.
  Writes a multi-band `noise_mask` centered on each internal iteration
  boundary in an assembled loop output latent. Boundaries derive from
  the same integer-latent counts the loop ran with: `stride =
  window_latents - overlap_latents`, seams at `[stride, 2*stride, ...,
  (N-1)*stride]`. Optional `edge_taper_seconds > 0` cosine-ramps the
  outer edges of each band so a downstream low-σ corrective sampler
  blends seam-zone regenerations into frozen context. Default
  `iteration_count=1` writes an all-zero mask (no-op) so the node is
  safe to drop onto a workflow before configuration. Coverage:
  8 behavioral tests in `tests/test_retake_nodes.py::TestLatentSeamZoneMask`
  (single-iteration no-op, multi-band shape, taper ramps, half-band
  clamp, default-is-hard, samples-preservation, zero-stride raises,
  band-clipped-at-edges). AST guard:
  `tests/test_node_schemas.py::test_latent_seam_zone_mask_iteration_count_default_is_one`
  asserts the `iteration_count` default stays at 1 inside the
  `LatentSeamZoneMask` class body — catches a default change that
  would surprise users by writing a non-zero mask on first drop.
  Registered in `comfy_entrypoint()` so it appears in the ComfyUI
  category browser. Pairs with `scripts/diagnose_overlap_seams.py` —
  use the diagnostic to decide whether seam-zone refinement is needed
  on a given render before wiring this node into a corrective
  workflow.

- **`scripts/build_seam_refinement_workflow.py`** — builds the
  post-loop seam-zone refinement workflow from scratch (22 nodes, 27
  links) at `internal/workflows/seam_zone_refinement.draft.json`.
  Topology: `VHS_LoadVideo → VAEEncode → LatentSeamZoneMask →
  LTXVConcatAVLatent (empty audio) → SamplerCustomAdvanced (3-step
  σ-tail [0.85, 0.7250, 0.4219, 0.0], euler, CFG=1) →
  LTXVSeparateAVLatent → LTXVCropGuides → LTXVTiledVAEDecode →
  VHS_VideoCombine`. Original audio passes through to combine without
  re-encoding. Idempotent; `--dry-run` prints node table, `--revert`
  deletes the output. Loader names + sigma profile + frame rate
  centralized as constants at the top of the script. Pairs with
  `scripts/diagnose_overlap_seams.py`: run the diagnostic on a real
  render first to confirm boundary-zone artifacts exist above the
  noise floor before configuring this workflow.

- **`docs/reference/debug_tools.md`** — adds rows for
  `diagnose_overlap_seams.py` (Inspection scripts), a new "Workflow
  build scripts" section covering `build_upscale_workflow.py`, and a
  "Selected staged-variant apply scripts" subsection covering
  `apply_lanczos_init_preprocess.py` + `apply_p3_retake_edit_lora.py`.
  Pre-existing apply-script conventions section unchanged.

- **`docs/README.md`** — task-first nav adds a pointer to
  `build_upscale_workflow.py` and a seam-artifact entry under "My
  output looks wrong / workflow won't run" pointing at
  `diagnose_overlap_seams.py`.

- **`scripts/apply_p3_retake_edit_lora.py`** — wires the section-targeted
  retake-edit pattern into a copy of the canonical retake workflow.
  Adds `LTXICLoRALoaderModelOnly` patching MODEL with the edit-anything
  LoRA (4-verb training: add / remove / replace / restyle), inserted
  between `AudioLoopHelperSageAttention` and `LTXVChunkFeedForward` per
  the canonical compile-style-patch order (LoRA loader must precede
  module-mutating nodes that call `model.state_dict()`). Adds
  `LTXVAddGuideMulti` (strength=1, frame_idx=0, num_guides=1) between
  `LatentTemporalMask` and `SamplerCustomAdvanced` so the sampler sees
  a guide-baked latent re-conditioned against the source pixels, and
  pulls positive/negative CONDITIONING through the same multi-guide
  node. Adds a Note node documenting that the existing positive
  CLIPTextEncode becomes the edit instruction. Stages output to
  `internal/workflows/retake_edit.draft.json`; does not mutate shipped
  workflows. Idempotent (signature-checked via the LoRA filename in
  the loader's first widget value), `--revert`, `--dry-run`. Promotion
  gated on a user cfg=1 A/B render confirming the four edit verbs
  land at distilled CFG=1.

- **`scripts/diagnose_overlap_seams.py`** — Phase A diagnostic for
  iteration-boundary artifacts in assembled loop output latents. Runs
  the per-frame ghost-residual `|f[t] - (f[t-1] + f[t+1]) / 2|`,
  inverts and normalizes to a ghost score (HIGH = ghost-like), and
  reports top-K frames plus per-seam-band scores derived from
  `--iteration-count`, `--window-latents`, and `--overlap-latents`.
  Includes a noise-floor baseline (median ghost score) so boundary-
  zone scores read against it. Pure CPU analysis on a saved latent
  tensor; gating evidence for Phase B (a `LatentSeamZoneMask` node +
  experimental corrective workflow). If real renders show
  boundary-zone scores well above the noise floor, build Phase B.

- **`example_workflows/audio-loop-music-video_latent_intro.json`** —
  shipped "intro" variant built atop the pre-encode workflow. Two
  `LoraLoaderModelOnly` nodes bypassed-by-default (Distill + Style;
  Distill pre-filled at strength 0.5), IC-LoRA chain bypassed, 9-group
  two-row layout for self-documenting discovery, five `Note` nodes
  annotating usage and Hugging Face model sources. Built by
  `scripts/apply_intro_workflow.py`.

- **`example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json`** —
  shipped variant of the iclora workflow with the audio-latent pre-encode
  topology applied. Validation render 2026-05-01 measured **−12.8s wall
  vs the apples-to-apples baseline** (`bench_profile2` adjusted for the
  ~178s of Profile* node overhead it carries that this workflow doesn't):
  381.7s → 368.94s, **−3.4% wall**. Breakdown matches design prediction:
  `LTXVAudioVAEEncode` count drops from 6 → 2 (saves 1.16s on encode
  itself), `IterationCleanup` drops from 9.67s × 5 to 1.28s × 5 (saves
  8.4s — the per-iter AudioVAE re-stage cost we predicted from console-
  log analysis), plus various per-iter savings on adjacent nodes. The
  smoking-gun confirmation: AudioVAE no longer re-stages between sampler
  invocations because the loop body no longer needs it. **Real win,
  matches estimate, ships as a shipped variant.** Audit clean (29 OK /
  1 WARN pre-existing latent_volume / 0 ERR). Promote to canonical iclora
  variant after a second confirmation render.

- **`scripts/apply_audio_latent_pre_encode.py`** — implements the
  audio-latent pre-encode topology designed at
  `internal/design/audio_latent_pre_encode_design.md`. Replaces the
  per-iter `LTXVAudioVAEEncode` + `TrimAudioDuration` subgraph chain
  with a one-shot full-song encode + per-iter `AudioLatentSlice`. Saves
  ~8.5s/render of audio re-encode (~1.7s × 5 loop iters) plus per-iter
  AudioVAE re-stage cost (5-15s/render of `Model AudioVAE prepared for
  dynamic VRAM loading` console-log overhead). Realistic estimate: 8-15s
  total wall savings on the iclora workflow — empirically confirmed at
  −12.8s on validation render. Stages to
  `internal/scratch/audio-loop-music-video_latent_audio_pre_encode.json`;
  shipped variant promoted to `example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json`.
  CLI flags `--source-seconds` (default 300, matches upstream
  `TrimAudioDuration` widget) and `--window-seconds` (default 17.92,
  matches `LTXFramePlanner` widget) bake into the `AudioLatentSlice`
  widget values. Subgraph schema changes force a UI delete-and-re-add
  of the loop subgraph node per CLAUDE.md. `#598 LTXVAudioVAEEncode` and
  `#600 TrimAudioDuration` are bypassed (`mode=4`) rather than deleted —
  cleaner UI, supports easy `--revert`. 13 unit tests at
  `tests/test_apply_audio_latent_pre_encode.py`. Validation render
  pending; reference_video VAE re-stage is still per-iter (not addressed
  by this script — separate spike if needed).

### Changed
- Retracted "every prompt must contain 'singing'" hard-rule. Confirmed
  working with non-singing verbs (e.g. dancing) when the verb matches
  the audio's action class. Reframed across `README.md`, `CLAUDE.md`,
  `docs/guides/`, the `ltx-constraints-auditor` agent, and
  `analyze_audio_features.py`'s LLM system prompt as a token-budget
  principle: pick a concrete verb that matches the visible action;
  generic verbs (`performing`, `vocalizing`) dilute cross-attention;
  concise > verbose because tokens compete with audio + image alignment
  in LTX 2.3's cross-attention budget.

### Empirical negative result
- **`LTXVideoRegionalCompile` mode="default" delivers null effect on the iclora
  workflow.** Bench 2026-05-01 (`ab_compile_default_v2` vs `bench_profile2`):
  555.85s vs 559.90s wall (-0.7%, within noise); 260.83s vs 261.51s sampler
  (-0.3%). Why: LTX 2.3's FFN is already fp8-quantized matmul (`sm89_xmma_gemm`),
  Inductor's "default" mode does kernel fusion but there's not much left to
  fuse around an already-fused fp8 matmul. The 42% launch overhead from the
  bench is dominated by ATTENTION + memcpy + sync launches, not FFN compile-
  amenable work. mode="reduce-overhead" (cudagraph_trees) might still help —
  bench variant ready at `internal/scratch/audio-loop-music-video_latent_iclora_bench_compile_reduce.json`
  — not yet tested. The node stays shipped as an experimental tool for future
  workflows that may benefit (e.g. larger DiT models with non-pre-fused FFN).

### Added
- **`AudioLatentSlice` (`nodes_audio_latent_slice.py`)** — slice an audio
  LATENT by source-timeline seconds. Companion to the audio-latent
  pre-encode pattern (design doc: `internal/design/audio_latent_pre_encode_design.md`).
  The current loop subgraph re-encodes the windowed audio slice via
  `LTXVAudioVAEEncode` each iteration (~1.7s × 5 loop iters = ~8.5s/render
  of pure encode + ~5-15s of `AudioVAE` re-stage overhead per console-log
  observation). Encoding the full song's audio latent ONCE outside the
  loop and slicing in latent space per-iter eliminates both costs.
  Latent rate inferred empirically from `latent.shape[temporal_dim] /
  source_seconds` — works regardless of audio VAE's mel hop_length /
  autoencoder downscale. Slicing math handles edge cases: out-of-range
  start clamps safely (returns at least 1 frame), zero duration yields
  a 1-frame slice. 21 unit tests at `tests/test_audio_latent_slice.py`.
  Workflow integration (apply script + subgraph schema change for the
  full pre-encode topology) is the bigger ~3.5hr piece, deferred until
  bench data shows the chunk + regional-compile candidates aren't enough.

- **`LTXVideoRegionalCompile` (experimental node, `nodes_regional_compile.py`)** —
  spike implementation of regional `torch.compile` per-block FFN on the LTX-2.3
  transformer. Compiles `transformer_blocks[i].ff` across all 48 blocks, leaves
  attention paths in eager dispatching to sage's `optimized_attention_override`.
  This is the canonical PyTorch + Diffusers pattern for diffusion DiTs: compile
  the static-shape compute modules, exclude attention because attention
  dispatchers (sage's pybind kernels graph-break Inductor and produce rtol
  drift on torch 2.11 per N5 spike 2026-05-01). Targets the 42% launch-overhead
  bucket from clean chrome trace 2026-05-01. Two modes: "default" (kernel
  fusion only, ~5-10% e2e estimate) and "reduce-overhead" (also enables
  cudagraph_trees, ~13-25% e2e estimate per FasterCache/PyGraph research, +1-2GB
  VRAM). Connect AFTER the LTX checkpoint loader and BEFORE other patches;
  order vs sage doesn't matter (orthogonal hook surfaces — sage on attn1/attn2,
  this on ff). Mutates shared `transformer_blocks[i].ff` in place; cleanup
  callback restores originals on unload, with sentinel-attribute detection so
  re-applies refresh rather than double-wrap. 8 unit tests at
  `tests/test_regional_compile.py`. Validation render pending. Surfaced by
  SOTA research agent 2026-05-01 (regional compile is the documented
  Diffusers-team recipe, not exotic).

- **Sliding-mode flag on `scripts/apply_iclora_video_reference.py`** — Phase 2
  of the IC-LoRA video-reference roadmap. `--ref-mode {static,sliding}` (default
  `static` preserves prior behavior). Sliding mode inserts a `SimpleCalculatorKJ`
  in the loop subgraph computing `start_index = round(video_start_time * ref_fps)`,
  rewires `GetImageRangeFromBatch.start_index` from widget to wired INT input, and
  the slicer's reference-video window advances with each loop iteration instead
  of statically reusing the same slice. Useful for long songs where a single
  ref-video window doesn't carry enough motion variation.
  Companion flag `--ref-fps INT` (default 25) is the single source of truth: it's
  baked into BOTH `VHS_LoadVideo.force_rate` (controls reference-video resampling)
  AND the calculator's expression (controls per-iter index advancement). Setting
  one without the other is impossible by construction. 3 new tests; all 21
  existing IC-LoRA tests still pass; both modes audit-clean (29 OK / 1 WARN
  pre-existing latent-volume informational / 0 ERR). To switch modes after a
  prior apply, run `--revert` then re-apply with the new mode.

### Changed
- **Retired the legacy `profile_output/` profile artifact path.** The runtime
  default for `ProfileBegin.output_dir` was migrated to
  `internal/analysis/runs/profiler` some time ago, but the legacy path lingered
  in `__init__.py`'s startup-cleanup loop, in test fixtures, in two doc files,
  and in `.gitignore`. Cleared all references in one sweep:
  `__init__.py::_clear_profiler_run_artifacts()` (renamed from
  `_clear_stale_profile_output()`) now targets a single canonical path via
  `shutil.rmtree(..., ignore_errors=True) + mkdir(exist_ok=True)` (~2× faster
  than the prior per-child iterdir loop). `tests/test_profile_nodes.py`
  updated to use the runtime default in its `enabled=False` test stubs.
  `scripts/profile_summary.py` docstring example, `docs/guides/profiling_guide.md`
  (3 references), and `docs/reference/telemetry_and_tracing.md` updated to
  reflect the canonical path. `nodes.py:2660` widget tooltip now also calls
  out the `data/runs/${RUN_ID}/profiler` override behavior when launched via
  `start_experiment.sh`. Empty `profile_output/` directory removed; gitignore
  entry dropped. No render-time behavior change — this is a hygiene-only
  cleanup that closes the migration started when the per-RUN_ID layout shipped.

### Fixed
- **Stripped 3× stale `linkIds: 3004` from `#1519 LTXVAddLatentGuide.outputs[2]`**
  on `audio-loop-music-video_latent.json`, `_keyframe.json`, and `_stg.json`.
  The denormalized `linkIds` field referenced a link that no longer exists in
  the subgraph's `links` array — caught by `audit_workflows.py::link_integrity`.
  Audit went 12 WARN → 9 WARN, 0 ERR. The remaining 1 link_integrity WARN is
  on `init_guide_amplification_poc.json` (experimental scratch variant,
  regenerated by its own apply script — out of scope).

### Added (continued)
- **`example_workflows/audio-loop-music-video_latent_iclora.json`** — promoted
  the IC-LoRA video-reference variant to a shipped example workflow. Mirrors
  the wiring used to render `numa.mp4` (LTX-2_00248.png snapshot): canonical
  audio-loop pipeline + `LTXICLoRALoaderModelOnly` (Cseti cameraman LoRA at
  strength 1.0) + `VHS_LoadVideo → ImageResizeKJv2 → LTXVPreprocess(val=18) →
  SetNode → subgraph reference_video IMAGE input → GetImageRangeFromBatch →
  LTXAddVideoICLoRAGuide` inside the loop subgraph, downstream of the F3
  cropguides chain. 84 nodes, 121 links. Audit: 29 OK / 1 WARN (pre-existing
  `latent_volume` near-edge soft ceiling) / 0 ERR. Users substitute their own
  `--reference-video` MP4 and IC-LoRA file when loading.

### Removed
- **`ModelSamplingSD3 shift=13` from all shipped workflows that had it** — 8 of our
  workflows (5 production variants + 3 experimental forks) shipped with this node;
  the canonical did not. Verified against Lightricks's reference distilled
  pipeline (`coderef/ID-LoRA/ID-LoRA-2.3/packages/ltx-pipelines/src/ltx_pipelines/distilled.py:106-112`)
  and their official 2.3 distilled example workflows: **no flow-matching shift
  is applied between sigma scheduling and denoising**. The `DISTILLED_SIGMA_VALUES`
  are the final sampling schedule, fed directly to `euler_denoising_loop`.
  `ModelSamplingSD3` was a borrowed-from-SD3 holdover that distorted the
  sigma-to-timestep mapping (`t' = 13t / (1 + 12t)`) for a checkpoint trained
  on raw sigmas. Bonus finding during the strip: in all 8 instances the node's
  output was already wired to nothing (`outputs[0].links == []`) — the node was
  effectively dead. Strip is pure cleanup, no behavior change at render time.
  Migration: `scripts/apply_strip_sd3_shift_node.py` (in-place, idempotent,
  `--revert`able). Audit `model_sampling_shift` semantic flipped: now WARNs
  when `ModelSamplingSD3` is present on production workflows. CLAUDE.md L45
  retracted accordingly. Full audit + evidence: `internal/analysis/ltx23_sigma_shift_audit.md`.

### Added
- **Video-reference IC-LoRA wiring (`scripts/apply_iclora_video_reference.py`)** —
  in-loop `LTXAddVideoICLoRAGuide` driven by a reference video clip, mirroring
  the canonical pattern from popular LTX 2.3 IC-LoRA HuggingFace repos
  (cameraman, outpaint, union-control). Splices `LTXICLoRALoaderModelOnly` on
  the top-level MODEL chain + `VHS_LoadVideo → ImageResizeKJv2 →
  LTXVPreprocess(val=18) → SetNode → subgraph IMAGE input` for the ref-video
  preprocessing chain (F2 symmetric with the init-image path) +
  `GetImageRangeFromBatch` (KJNodes) + `LTXAddVideoICLoRAGuide` inside the loop
  subgraph, downstream of `#1519 LTXVAddLatentGuide` and upstream of
  `LTXVCropGuides[NoLatent]` (F3 symmetric). Static is the default widget
  configuration of the sliding-capable wiring (`start_index=0`, `num_frames=25`);
  switching to sliding mode is widget edits, not graph rebuild. Reference workflow
  pattern: `internal/ref_workflows/ltx2.3-ic-lora-cameraman.json`. Decisions:
  `internal/ic_lora_assessment.md` D19–D23.
- **Three audit checks for video-reference IC-LoRA topology** (F12) —
  `iclora_video_reference_guide_in_loop_with_cropguides` (ERR: guide CONDITIONING
  outputs must reach CFGGuider via cropguides),
  `iclora_loader_present_when_guide_present` (ERR: guide implies loader on top-level),
  `iclora_ref_video_preprocess_symmetry` (ERR: `LTXVPreprocess(val=18)` must be
  present on the ref-video chain). Fire only when `LTXAddVideoICLoRAGuide` is in
  the subgraph; bail fast on workflows without it.
- **`audit_workflows.py [path...]`** — positional path argument for auditing
  arbitrary workflow JSONs (e.g. staged scratch files in `internal/scratch/`).
  No-args still sweeps `example_workflows/` + audited `experimental/` subset.

### Changed
- **Stripped dead bypassed LoRA-loader scaffolding from the canonical**
  (`scripts/apply_strip_dead_lora_loaders.py`). Three nodes (`#1625 LoraLoaderModelOnly`
  "ID-LoRA File", `#1626 LTXICLoRALoaderModelOnly` "IC-LoRA File", `#1627
  LoraLoaderModelOnly` "Style/Generic LoRA" with placeholder filename) were
  inert (mode=4 → MODEL passes through unchanged) but suggested behavior the
  workflow didn't have. Strict triple-match `(id, type, title, mode=4, file)`
  preserves user customizations. Audit check `dead_lora_loader_scaffolding_absent`
  (F11, ERR) prevents regression. Bridge link rebridges `#503.0 → #572.0`
  directly (skipping the dead chain).

### Added
- **Phase 2.1 perceptual metric: `av_consistency`** — Perception
  Encoder PE-AV-16-frame (`facebook/pe-av-large-16-frame`,
  arxiv:2512.19687, **Apache-2.0**) joint audio-video-text embedding.
  v0 reports a single cosine similarity: AV embedding vs the
  fixture's `init_positive` text — measuring how well the rendered
  video+audio matches its target prompt. Replaces the
  originally-planned lip_sync (AV-HuBERT) + seam_continuity (STREAM)
  pair with a unified A/V-aware extractor; Apache license is a
  meaningful upgrade over those gated/restrictive options. 16
  evenly-spaced frames per video + audio extracted from the same
  mp4. Two-tier model loading: tries `transformers` first
  (`pe_audio_video` model_type, needs ≥4.51), falls back to
  `perception_models` package; reports `model_unavailable` if
  neither resolves. 7 new test cases under
  `tests/test_autoresearch.py::TestAvConsistency`. V1 (deferred):
  per-iteration AV embeddings + drift trajectory.
- **Output mp4 discovery in harness**:
  `_locate_and_link_output_mp4(run_id, run_dir, source_dir=None)`
  scans `COMFYUI_OUTPUT_DIR` for `LTX-2_${run_id}_*.mp4` and symlinks
  the first lexicographic match into `data/runs/${run_id}/output.mp4`
  so video-content metrics (`subject_consistency`, future style /
  lip_sync / aesthetic) can read a stable per-run path. Soft failure
  when env unset or no mp4 matches: caller still records the row,
  metric reports `*_status: video_missing`. Idempotent re-link. New
  `COMFYUI_OUTPUT_DIR` env var documented in
  `docs/reference/environment.md` with privacy note (typically points
  outside the repo). 7 new tests under
  `tests/test_autoresearch.py::TestLocateAndLinkOutputMp4` covering
  None/missing/unmatched/single-match/multi-match/idempotence/env-fallback.
- **Phase 2.1 perceptual metric: `subject_consistency`** — DINOv3
  (`facebook/dinov3-vitb16-pretrain-lvd1689m`, arxiv:2508.10104)
  cosine similarity of per-frame embeddings against the first frame
  (the init-image anchor). Reports `mean/min/max_to_anchor` and a
  linear `drift_slope`. Answers the ID-LoRA "is it doing anything?"
  question that started the experiment-runner arc — a mean-cos-sim
  delta between LoRA-on and LoRA-off renders of the same fixture.
  First metric in the heavy-dep tier; introduces a new `metrics`
  optional dep group (`torchvision`, `transformers>=4.46`,
  `opencv-python-headless`, `numpy`, `scipy`). DINOv3 is **gated** on
  Hugging Face — a clone needs `huggingface-cli login` or `HF_TOKEN`
  in the environment. Heavy imports are gated so the module loads
  cleanly on a public clone without the group installed (returns
  `subject_consistency_status: "model_unavailable"`).
  Public-facing test contract:
  `tests/test_autoresearch.py::TestSubjectConsistency` (10 cases — all
  status branches + helper-function unit tests with synthetic
  embeddings; no model download required). Sets the import-gating +
  module-cache pattern that SigLIP-2 (style) and AV-HuBERT (lip-sync)
  will follow. Future v1 (queued): SAM 3.1 (`facebook/sam3.1`,
  arxiv:2511.16719) to mask the subject across frames before
  embedding — strips background noise from the metric.
- **First non-placeholder Phase 2.1 metric extractor:
  `sage_summary`** — reads `data/runs/${RUN_ID}/sage.jsonl` per
  render, aggregates kernel distribution + fallback count + total
  attention time + distinct shapes + arch tag, lands in the tracker's
  `metrics` JSON column and on disk at
  `data/runs/${RUN_ID}/metrics.json`. Validates the metric-extractor
  contract for non-trivial extractors (the previous `wall_time` was a
  placeholder). Cross-pollinates with `scripts/sage_telemetry_summary.py`'s
  aggregation logic; per-run instead of per-CLI-invocation.
  Public-facing test contract: `tests/test_autoresearch.py::TestSageSummary`
  (5 cases — trace-missing, trace-empty, ok-aggregation,
  fallback-to-effective-mode, blank-lines-and-decode-errors).
  Implementation gitignored in `internal/autoresearch/metrics/sage_summary.py`.
- **Public env-var registry** at `docs/reference/environment.md`. Lists
  every env var the codebase reads (`RUN_ID`, `COMFYUI_EXEC_LOG`,
  `COMFYUI_EXEC_LOG_SHAPE_LIMIT`, `AUDIOLOOPHELPER_SAGE_TRACE`,
  `COMFYUI_API_URL`) with default behavior, set/read responsibility,
  and an audit command. **DRY rule** documented + enforced: each var is
  read at exactly one helper call site. `RUN_ID` reads now centralized
  through `scripts/workflow_utils.py::_current_run_id` after a
  Phase 1b follow-up refactor of `ProfileBegin.execute` that previously
  read `RUN_ID` inline.
- **`scripts/workflow_utils.py::run_artifact_dir(subdir="")`** —
  companion to `run_artifact_path` for multi-file artifacts (profiler
  trace.json + summary.txt + memory_timeline.html, frame sequences).
- **Sage tracer stamps `prompt_id`** on every per-call JSONL row.
  `nodes_sage.py::SageTracer.emit` accepts `prompt_id: str | None = None`;
  `make_sage_override` reads it from `kwargs["transformer_options"]`
  via a new `_prompt_id_from_kwargs()` helper. Same "absent means
  unknown" contract as `dispatched_kernel`. Closes the cross-repo ask
  from the sage-fork bench harness — direct prompt-id-keyed filtering
  replaces fence-post-prone timestamp windowing. Tests:
  `tests/test_sage_node.py` "8. prompt_id stamping" (4 cases).
- **Experiment-runner framework scaffold** at `internal/autoresearch/`
  (gitignored; promotes to top-level when stabilized). DuckDB tracker
  (`tracker.py`), one-cycle harness (`harness.py`), agent-edit stub
  (`apply.py`), placeholder wall-time metric (`metrics/wall_time.py`),
  fixture format, agent brief (`program.md`), one-shot wrapper
  (`run.sh`). Public surface: new `experiments` dep group
  (`duckdb`, `httpx`) + `tests/test_autoresearch.py` (19 cases;
  gracefully skips on fresh public clones where `internal/` isn't
  present). Real perceptual metric modules (DINO-v2, SigLIP-2,
  AV-HuBERT, STREAM, VideoScore, palette EMD) arrive in Phase 2.1.
- Foundation pieces below ship together because all three remove a
  correlation/reproducibility gap the experiment runner needs.

- **Foundation for an experiment-runner framework: seed rename, RUN_ID
  propagation, iterations auto-wiring.** Three pieces shipped together;
  each removes a correlation/reproducibility gap surfaced by the eight-
  render ID-LoRA ablation where it was impossible to tell whether the
  LoRA was actually patching the model flow.
  - `AudioLoopController.seed` → `base_seed` (`nodes.py:482`). ComfyUI's
    frontend auto-attaches a `control_after_generate` dropdown to any
    INT widget literally named `"seed"` or `"noise_seed"`, which mutates
    saved widget values across runs even when the input is wired (the
    link supersedes the widget at execute time, but the widget still
    serializes — saved JSONs drift across renders despite reproducible
    runtime seeds). Migration: `scripts/apply_alc_seed_rename.py`.
    Audit: `audit_workflows.py::alc_seed_legacy_name`. Invariant test:
    `tests/test_node_schemas.py::test_no_seed_or_noise_seed_named_inputs`
    AST-walks every `io.*.Input(...)` call site.
  - **`RUN_ID` env var** propagated to every logger via
    `scripts/workflow_utils.py::run_artifact_path`. With `RUN_ID` set
    (auto-generated by the new `start_experiment.sh` wrapper, format
    `${ISO8601_UTC}_${rand4}`), `exec_logger.py`, `nodes_sage.py`, and
    `nodes.py::ProfileBegin` all write to `data/runs/${RUN_ID}/<cat>.<ext>`.
    Single shared correlation key across `exec.jsonl`, `sage.jsonl`,
    `profiler/`, and the VHS output mp4 (when the harness mutates
    `filename_prefix`). Without `RUN_ID`, all loggers fall back to the
    legacy timestamped paths under `internal/analysis/runs/`.
    `sage_telemetry_summary.py` and `verify_sage_iteration_trace.sh` now
    search both layouts and pick newest by mtime.
  - **`AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in`**
    auto-wired in every shipped workflow. Loop count auto-tracks input
    audio length; user no longer touches the iterations widget. Required
    a backward-compatible upstream patch to `ComfyUI-NativeLooping_testing`
    that adds the optional wireable `iterations_in` Int input. For short
    test renders, drag in an `INTConstant` and rewire — recipe in
    `docs/guides/debugging_guide.md`. Migration:
    `scripts/apply_iterations_autowire.py`. Audit:
    `audit_workflows.py::iterations_autowired` (ERR if unwired in a
    shipped workflow, WARN on non-canonical sources).
  - Round-trip test: `tests/test_run_id_propagation.py`.
  - Diagnosis writeup: `internal/analysis/id_lora_ablation_and_seed_widget_audit.md`.
- **`scripts/extract_workflow_from_png.py`** — Pillow-based extractor for
  ComfyUI's embedded `workflow` and `prompt` tEXt chunks. Single-file or
  batch (`-d <dir>`); pretty-printed via orjson. Used to recover the
  eight ablation workflow JSONs from their PNG previews.

### Changed
- **`start_experiment.sh` (this repo's root) now owns all three
  telemetry env-var exports** (`RUN_ID`, `AUDIOLOOPHELPER_SAGE_TRACE`,
  `COMFYUI_EXEC_LOG`). The companion ComfyUI launcher
  (`<comfyui>/start.sh`) is back to a vanilla launcher with no
  plugin-specific knowledge — anything experiment-related layers on
  top via `start_experiment.sh`. Backward-compatible at the env-var
  level (all three vars still honored if exported anywhere). Migration
  impact: a user who previously ran plain `./start.sh` and implicitly
  got telemetry will now get vanilla ComfyUI; restore via
  `start_experiment.sh`.
- **Sage tracer reads `sageattention.get_last_dispatched_kernel()`.**
  Sage-fork ships a thread-local that records the resolved kernel name
  (one of `KNOWN_KERNEL_NAMES`: `fp16_triton`, `fp16_cuda`,
  `fp16_cuda(fp16)`, `fp16_cuda++`, `fp8_cuda`, `fp8_cuda(fp32+fp32)`,
  `fp8_cuda++`, `fp8_cuda_sm90`, `varlen_triton`) for the most recent
  `sageattn*` call. The override reads it immediately after `sage_fn`
  returns and stamps `dispatched_kernel` into the per-call trace row.
  Defensive: cached `getattr` at module import; absent symbol degrades
  silently (older sageattention installs / pre-upgrade traces continue
  to use the routing-table mirror in the summary script).
- **Summary script prefers `dispatched_kernel` over inference.**
  `aggregate()` precedence: `row['dispatched_kernel']` > `effective_
  mode` + routing-table mirror via `arch`. Trust real-data values
  above operator overrides. Empty / None values fall through to the
  mirror (treats "thread-local unset" same as "field missing").
- **`kernel_source_counts` trace-freshness signal.** Each row's kernel
  resolution path is bucketed into `sage_telemetry` /
  `mirror_inferred` / `unknown`; counts surfaced in the table header
  as one line: `attribution: N sage_telemetry / M mirror_inferred /
  K unknown`. Operator instantly sees whether the gate verdict came
  from real telemetry (post-upgrade trace) or post-hoc inference
  (pre-upgrade trace). Three buckets always present (zeros included)
  so consumers don't need defensive `.get()`.
- **Sage tracer self-describing arch field.** `nodes_sage.py` detects
  `sm<MAJ><MIN>_cuda<MAJ>_<MIN>` once at `SageTracer.__init__` and
  stamps it into a one-time `event="header"` row plus every per-call
  row. Lets the summary script resolve `auto` → actual kernel without
  a `--arch` flag — traces are self-describing across hosts.
- **`scripts/sage_telemetry_summary.py` per-prompt grouping.** New
  `PromptWindow` / `parse_exec_log_windows()` / `assign_prompt_id()`
  / `aggregate_per_prompt()`. CLI auto-switches to per-prompt mode
  when the exec log spans >1 `prompt_id`; rows whose ts falls outside
  every window go to a `"unknown"` bucket (counted, not dropped).
  Replaces the broken `sum-of-ksampler-durations-across-all-prompts`
  denominator that produced wildly inconsistent pct values
  (4.27 / 17.28 / 3.16% for the same trace).
- **`scripts/sage_telemetry_summary.py --arch`, `--use-sage-span`,
  routing-table mirror.** `infer_kernel(effective_mode, has_mask,
  arch)` mirrors the subset of `sageattention/core.py::sageattn` that
  the consumer's call pattern reaches (sm89/CUDA12.8, no smooth_k, no
  LSE, head_dim ∈ {64,120,128}): masked → `fp16_triton`, unmasked
  → `fp8_cuda++`. Documented as a stopgap until sage-fork ships
  `get_last_dispatched_kernel()`. `--use-sage-span` uses
  `(max_ts - min_ts)` of the sage rows as a self-contained
  denominator. Arch precedence: `--arch` > per-row `arch` field >
  local autodetect.
- **`LTXVCropGuidesNoLatent` node** — CONDITIONING-only variant of upstream
  `LTXVCropGuides`. Strips `keyframe_idxs` from positive/negative without
  taking or producing a LATENT (eliminates the wasted `latent["samples"]
  .clone()` on the F3 path). `apply_split_cropguides.py` now upgrades
  `#655` to this type during apply (and reverts cleanly).
- **`WorkflowEditor.add_subgraph_node`** — subgraph-side counterpart to
  `add_top_level_node`. `add_subgraph_link` extended to handle the
  virtual `-10` input distributor and `-20` output collector. Three
  hand-rolling sites migrated: `apply_split_cropguides.py`,
  `apply_perf_improvements.py`, `apply_profiling_nodes.py`.
- **`LTXResolutionFromAspect` node + `_compute_ltx_resolution` helper +
  `scripts/calc_ltx_resolution.py` CLI.** Resolves a target aspect ratio
  + long edge to LTX 2.3-valid (W, H), classifies the latent volume
  against the `docs/reference/ltx23_model_reference.md` ceiling, and
  short-edge-snaps DOWN to bias toward the safe side of the artifact
  threshold (matches users' empirical 832x448 vs the volume-bursting
  true-16:9 832x480). Shared math between node, CLI, and audit. Tests
  in `tests/test_ltx_resolution.py`.
- **`scripts/apply_canonical_resolution_fix.py`.** Brings every shipped
  production workflow's `EmptyLTXVLatentVideo` widget into spec with the
  reference doc's latent-volume ceiling AND with `ImageResizeKJv2`'s
  actual target. Pre-fix: `[704, 704, 497]` (volume 30,492 — ~25% over
  the 24,570 artifact ceiling). Post-fix: `[832, 448, 497]` (22,932 —
  in NEAR_EDGE territory, which is users' actual operating point).
  Idempotent + `--revert`. Mismatch had been committed for at least 8
  prior commits; users were editing dimensions in UI before each render.
- **`audit_workflows.py` latent-volume check.** ERRs when
  `(W/32)*(H/32)*((L-1)/8+1)` exceeds 24,570 with a remediation
  pointer to `apply_canonical_resolution_fix.py`. WARNs above 20,000.
  Per CLAUDE.md "bake topology constraints into audit": fix and audit
  ship as a pair so a sibling branch can't silently regress the fix.

- **Loop-subgraph cycle through `LTXVCropGuides` resolved by splitting
  it into two instances.** Recent ComfyUI's strict cycle detector
  rejects the canonical `CFGGuider ← CropGuides ← SeparateAV ← Sampler
  ← CFGGuider` loop at prompt-validation time; users got the sampler
  pass but no `.mp4`. Fix wires `CropGuides(655)` purely from
  `LTXVAddLatentGuide(1519)` (CONDITIONING role, F3 honored) and adds
  a second `LTXVCropGuides` (titled `"CropGuides (LATENT-only —
  split)"`) reading `SeparateAV(596).video_latent` for the post-
  sampling crop into AdainLatent. No new node code (uses upstream
  CropGuides twice). Apply via `scripts/apply_split_cropguides.py`;
  `audit_workflows.py::cropguides_split_topology` ERRs if reverted.
- **`exec_logger.py` async-cache wrapper.** Recent ComfyUI made
  `HierarchicalCache.get` a coroutine; the wrapper at line 220 called
  it without `await`, generating an unawaited-coroutine warning at
  every node execution. Now uses `inspect.iscoroutine` to await iff the
  return is a coroutine (compatible with both pre- and post-async
  ComfyUI versions).

### Changed
- **Public-facing prompt case studies retired (`docs/examples/`
  removed).** The parallel scrubbed-copy convention was a real
  maintenance burden (every internal prompt edit required mirroring
  + scrubbing) and a recurring scrub-leak risk vector with no
  confirmed external readership. Pattern guidance is now distilled
  inline in `docs/guides/prompt_creation_guide.md` §12 (six scenario
  families with load-bearing rules, no file pointers); concrete runs
  remain in `internal/prompts/`. Cleaned up 21 references across 9
  files (`prompt_creation_guide.md`, `CLAUDE.md`, `debugging_guide.md`,
  `README.md`, `nodes_validation.py`, `ltx23_prompt_system_prompts.md`,
  `experimental/README.md`, `audio_in_prompt_research.md`). CLAUDE.md
  "Case studies in pairs" convention rewritten accordingly.

### Added (continued)
- **Second TTC1 amplification POC, no IC-LoRA required.**
  `scripts/apply_ttc_init_guide_amplification_poc.py` stages
  `example_workflows/experimental/init_guide_amplification_poc.json` —
  forks the production audio-loop latent workflow and rewires
  `CFGGuider(644).negative` inside the loop subgraph to read from the
  subgraph input distributor's positive slot (before
  `LTXVAddLatentGuide`). Demonstrates that the CFG-analog amplification
  mechanism generalizes per-conditional: same recipe as
  `apply_ttc_iclora_amplification_poc.py`, different upstream node, no
  IC-LoRA in the graph. Pairs with a new `ttc1_init_guide_amplification`
  audit check that recognizes the deliberate F3 asymmetry on the
  negative branch as intentional and ERRs if the rewire is damaged.
  `audit_workflows.py` now scans an explicit allowlist of experimental
  POC files (`EXPERIMENTAL_AUDITED_FILES`) so the new variant is
  CI-checked alongside production workflows. Public docs framing
  updated (`README.md`, `docs/experiments/README.md`,
  `docs/experiments/exp_2026-04-24_spectrogram_iclora_v2a.md`) to make
  clear that TTC1 is a generalized inference-time technique and IC-LoRA
  is just the first wiring.

### Changed
- **Hygiene bundle 2026-04-25** — small/low-risk items closing flags from
  prior sessions. (1) `LTXFrameCalculator` (LTXAVTools) referenced from
  CLAUDE.md `(L-1)%8==0` constraint and `docs/guides/retake_guide.md` so
  users can derive valid frame counts without hand-math. (2) Outer-path
  `CLIPTextEncode(169) → ConditioningZeroOut(420) → LTXVConditioning(164)
  → CFGGuider(153).negative` chain documented in CLAUDE.md gotchas as
  wired-correctly but runtime-inert at `cfg=1` (cannot be removed —
  CFGGuider validates both input slots). Closes the 2026-04-22 "dead
  code?" flag with no-op outcome. (3) Orphan virtual GetNodes stripped:
  `_latent_keyframe.json` cleared one (`Set_input_image`/`Get_input_image`
  pair, vestigial after Phase 1 keyframe-batch-encode migration);
  `_retake.json` cleared eleven (added to `apply_audio_loop_retake.py`
  STRIP_IDS — re-running the script produces a workflow with zero
  orphan GetNodes). (4) `VideoFrameExtract` docstring updated to
  document its retained-but-unwired status so future sessions don't
  mistake it for dead code. (5) NAG `object_patches` analysis note
  audited as already adequately linked (CLAUDE.md + docs/README +
  architecture_overview); no further follow-up needed.
- **`scripts/apply_spectrogram_iclora_minimal.py` rewritten as a
  production-fork builder.** Previous scratch-built 25-node workflow
  produced chroma-static output because LTX 2.3 distilled needs the
  full production patch chain (sage → chunk-FF → tuner → NAG →
  preview-override → ModelSamplingSD3). The new script forks
  `example_workflows/audio-loop-music-video_latent.json`, strips loop
  infrastructure (TensorLoopOpen/Close, subgraph, AudioLoopController,
  batch-encode, MelBand, LoadAudio + TrimAudioDuration), strips the
  audio-freeze chain (`LTXVAudioVAEEncode` + `SetLatentNoiseMask`),
  and adds `LTXVEmptyLatentAudio` + `LTXVAudioVAEDecode` so the
  sampler generates both video AND audio (V2A round-trip test).
  `LTXICLoRALoaderModelOnly` + `LTXAddVideoICLoRAGuide` inserted on
  the initial-render path with the spectrogram mp4 as IC-LoRA IMAGE
  input via `LoadVideo` + `GetVideoComponents`. Bypasses
  `LatentConcat(1605)` "Prepend Initial Render" whose second input
  dangles after the loop strip; wires `LTXVCropGuides(381).latent →
  LTXVTiledVAEDecode(1604)` directly. Topology verified via
  `scripts/analyze_workflow_dag.py`. Output:
  `example_workflows/experimental/spectrogram_iclora_minimal.json`
  (78 nodes, 66 links).
- **`docs/experimental/spectrogram_iclora_tutorial.md` rewritten** to
  reflect the production-fork topology. Drops the "scratch-built
  minimal" framing. Adds §Extensions section covering
  `ComfyUI-LTXAVTools` nodes now available (`LTXFrameCalculator`,
  `LTXDimensionCalculator`, `LTXVAddAudioLatentGuide`,
  `LTXAudioLatentTrim/Pad`, `LTXVAVLoopingSampler`, `LTXDetailSigmas`).
  Notes prompt-simplicity guidance ("descriptive prompts break
  things, let other modalities drive"). Notes with/without sage A/B
  as a variant worth running. Troubleshooting section maps actual
  failure modes we hit during testing (chroma noise → patch chain
  missing; empty mp4 → LatentConcat dangling; audio silent → LTX's
  audio head limits) to recovery actions.

### Added
- **Phase 1 keyframe-latent batch encode shipped.** New
  `KeyframeLatentScheduleBatchEncode` (top-level) +
  `LatentSelectByIteration` (loop body) pair mirrors the
  conditioning-side `TimestampPromptScheduleBatchEncode +
  ConditioningSelectByIteration` shape shipped 2026-04-22. VAE encodes
  each unique keyframe image exactly once per generation regardless of
  how many iterations share it; the legacy
  `KeyframeImageSchedule + ImageBlend + per-iter VAEEncode` chain is
  retired on the latent-keyframe workflow. Module-level
  `_KEYFRAME_LATENT_CACHE` (LRU, bounded) plus `IS_CHANGED` classmethod
  prevent re-encoding on the AudioLoopController-driven framework
  re-execution. Migration:
  `scripts/apply_keyframe_batch_encode.py` (idempotent, supports
  `--dry-run` and `--revert`); rewires
  `example_workflows/audio-loop-music-video_latent_keyframe.json` to
  the new pattern. 17 new tests in
  `tests/test_keyframe_batch_encode.py` covering encode-once invariant,
  cache hit, cache invalidation, identity stability across shared
  iterations, out-of-bounds and negative index clamps, and integration
  parity with the legacy `KeyframeImageSchedule` per-iter output. Both
  nodes registered in `comfy_entrypoint`. CLAUDE.md updated with a
  dedicated "Keyframe schedule" core-nodes line.
- **Retake workflow shipped (Phase 3, Option A).** New
  `example_workflows/audio-loop-music-video_retake.json` regenerates a
  `[start_time, end_time]` window of a previously-generated video without
  re-rendering the rest. Built by `scripts/apply_audio_loop_retake.py`
  (forks production `audio-loop-music-video_latent.json`, strips 34
  loop/audio/init-image nodes, adds `LoadVideo + GetVideoComponents +
  VAEEncode + LatentTemporalMask`, rewires sampler/decoder/audio
  passthrough). Audio passes through unchanged from the source mp4 via
  `VHS_VideoCombine.audio` (Option A — no AV cross-attention during
  retake). User guide at `docs/guides/retake_guide.md`. A-vs-B design
  rationale in `internal/design/retake_workflow_design.md`; Phase 3.5
  (AV-aware retake) parked pending lip-sync drift signal from real
  retake renders. `scripts/audit_workflows.py` extended with three
  retake-specific checks (`retake_temporal_mask_present`,
  `retake_audio_passthrough`, `retake_no_loop_nodes`) gated on
  filename match; loop-only checks (`iteration_stamp`,
  `prompt_schedule`) skipped for retake workflows via new
  `_is_retake(name)` helper.
- **`LatentTemporalMask` node (`nodes.py`)** — retake support. Writes a
  `noise_mask` to a video latent so only `[start_time, end_time]`
  regenerates on a re-sample; rest stays fixed as context. Latent-frame
  math: `start_latent = int(t*fps/8)`, `end_latent = int(end*fps/8)+1`,
  clamped to the latent length. Reversed or zero-width ranges yield an
  all-zero mask (no-op) rather than raising — safer for UI widget drift.
  Mask shape matches samples `[B,C,F,H,W]` (no broadcast shortcut,
  consistent with upstream `LTXVSetAudioVideoMaskByTime`). Port of
  `TemporalRegionMask.apply_to` from
  `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/retake.py`.
  11 tests in `tests/test_retake_nodes.py`.
- **`scripts/audit_workflows.py`** — health audit across every
  `example_workflows/*.json` (sage node+mode, `LoopIterationStamp`,
  batch-encode pattern, distilled sigma chain, resolution div-32,
  `(L-1)%8==0`, `LTXVPreprocess img_compression>=18`,
  `LTXVTiledVAEDecode` preferred). Exits 1 on any ERR, 0 on WARN-only.
  All six shipped workflows currently 66 OK / 0 WARN / 0 ERR.
- **`scripts/apply_iclora_initial_render.py`** — Phase 0a IC-LoRA wiring
  for the latent loop workflow. Forks `example_workflows/audio-loop-music-video_latent.json`
  into `internal/scratch/audio-loop-music-video_latent_iclora_phase0a.json`
  with `LTXICLoRALoaderModelOnly` (LoRA-patches MODEL for both initial
  render and loop body — the open MODEL-fork question Phase 0a is
  designed to answer empirically) and `LTXAddVideoICLoRAGuide` on the
  initial-render conditioning + latent path. Loop-body `base_cond_neg`
  Set source intentionally untouched so loop iterations see the
  unmodified conditioning baseline. Idempotent + `--revert` supported.
  Companion assessment in `internal/ic_lora_assessment.md` evaluates the
  full sibling-Claude IC-LoRA ideas list (`internal/IC-LORA_IDEAS.md`)
  tier-by-tier against this project's architecture; downgrades the doc's
  Tier-S "proxy-render-as-reference" flagship as misaligned with our
  init-image commitment pattern, and promotes the doc's Tier-C
  spectrogram-as-canny idea to Tier A given our frozen-audio thesis.
- **`EXAMPLE_WORKFLOWS_DIR` constant in `scripts/workflow_utils.py`** —
  retires three duplicate inline definitions across `audit_workflows.py`,
  `validate_workflow_resolution.py`, `tests/test_workflows.py`.

### Changed
- **`AudioLoopHelperSageAttention` default mode is now `auto_mask_aware`**
  (was `auto`). Routes masked cross-attn calls to
  `sageattn_qk_int8_pv_fp16_triton` and unmasked self-attn calls to
  sage `auto`. Necessary because sage's INT8-QK-FP8/FP16-PV CUDA kernels
  don't implement mask support — `MaskMode` is `{kNone, kCausal}` only;
  `attn_mask` passed via kwargs is silently dropped, contaminating
  attention with padded positions. Measurement in
  `<sage_fork_repo>/tests/test_sageattn_ltx_shapes.py` shows rtol 0.26–0.94
  vs SDPA across seq_kv 32–1024 on LTX cross-attn shapes (scaling ∝
  1/seq_kv, consistent with "mask ignored"); triton stays clean at
  rtol ≈ 0.039. Stateless per-call routing — no offload-state risk
  beyond the base override. Tracer JSONL rows now include an
  `effective_mode` field so a trace can confirm the routing is firing.
  6 new tests.
- **Shipping `_latent*.json` workflows swapped from KJNodes'
  `PathchSageAttentionKJ` to `AudioLoopHelperSageAttention`** via
  `scripts/apply_audioloophelper_sage.py [--all]`. Reversible with
  `--revert`. `scripts/apply_sage_mode.py` extended to handle both
  node types; `mask_aware` added as shorthand for `auto_mask_aware`.

### Added
- **`LoopIterationStamp` node** (`nodes.py`). MODEL passthrough that
  writes `transformer_options["iteration"] = int(current_iteration)`.
  Wired between the patch chain (sage / NAG / tuner) and the subgraph
  invoker; auto-inserted by `scripts/apply_iteration_stamp.py [--all]`.
  Unblocks offload-asymmetry verification: `AudioLoopHelperSageAttention`'s
  JSONL tracer groups rows by `iter` so silent sage disengagement on
  iter 2+ (the NAG-asymmetry sibling risk) becomes detectable. Additive
  -- preserves `optimized_attention_override` and other transformer_options
  keys. 6 tests in `tests/test_iteration_stamp.py`.
- **`AudioLoopHelperSageAttention` node** (`nodes_sage.py`). An
  AudioLoopHelper-native alternative to KJNodes'
  `PathchSageAttentionKJ` with three properties the KJ node lacks:
  (1) try/except pytorch fallback on sage exceptions with deduplicated
  logging per `(shape, mode, error)`, (2) `CallbacksMP.ON_CLEANUP`
  handler so the override is removed on model unload, (3) opt-in
  per-call JSONL telemetry to `internal/analysis/runs/sage/` gated by
  the `AUDIOLOOPHELPER_SAGE_TRACE` env var (zero overhead when unset).
  Mode combo is filtered to modes the detected GPU can actually run
  (no Blackwell-only footguns on Ada). See
  `internal/analysis/sage_attention_analysis.md` for the patch-chain
  analysis that motivates this node and
  `internal/design/sage_backlog.md` for deferred mask-aware routing /
  baselining work. 15 tests in `tests/test_sage_node.py`.

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
- **`exec_logger.py` chained-wrapping across module reloads.** Six+
  near-duplicate exec_log files all cutting off at the exact same
  instant with byte-identical trailing content traced to a missing
  reload guard: `_INSTALLED` resets to `False` when ComfyUI reloads
  the custom node, `install()` runs again, captures the previously-
  wrapped `_exec_mod.execute` as `original`, and adds another sink in
  front of it — N reloads produce N sinks all writing to N files.
  Fix: stamp `_audioloophelper_wrapped = True` sentinel on
  `_exec_mod.execute` itself; `install()` checks for it before
  wrapping. Sentinel survives module reloads (lives on the function
  object), correctly disappears when ComfyUI replaces `execute`
  wholesale (re-install proceeds). Tests in `tests/test_exec_logger.py`.
- **`scripts/sage_telemetry_summary.py` denominator broken for
  multi-prompt exec logs.** `total_wall_us_from_exec_log` summed
  every ksampler `end` event regardless of prompt — with loops +
  multiple queued prompts in the same exec log, the denominator was
  ~5× too big. CLI now auto-switches to per-prompt grouping when
  >1 prompt_id is present in the exec log; single-prompt exec log
  uses that prompt's wall window; legacy sum-of-ksampler-durations
  is the last-resort fallback only. See Added entry above for the
  per-prompt grouping API.
- **Sage gate cross-section "unmasked_fp8++" never populated.** The
  consumer-side tracer records `effective_mode="auto"` for unmasked
  calls because `_route_mask_aware()` returns the literal string
  `"auto"` (the actual kernel is chosen inside `sageattention.
  sageattn()`, opaque from outside). Without inference, the gate's
  `(fp8_cuda++, False)` cross-section was permanently empty. Fix in
  the summary script: `infer_kernel()` post-hoc rewrites `auto` to
  `fp8_cuda++` on sm89/CUDA12.8 (mirroring sage's actual routing
  table for our call pattern). See Added entry above.
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
