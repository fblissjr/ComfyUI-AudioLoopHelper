Last updated: 2026-05-05

# docs/ — ComfyUI-AudioLoopHelper documentation

**If you're new to this repo, start at [`architecture_overview.md`](architecture_overview.md).**
It's the single-pass walkthrough of the full stack (our workflow →
ComfyUI core → ComfyUI-LTXVideo → KJNodes → LTX-2 model) and links
into every deeper doc.

This README is the task-first nav index — "I want to do X, which doc has
the answer?" Use it when you already know what you're looking for.

---

## Directory layout

```
docs/
├── README.md                 ← you are here (task-first nav)
├── architecture_overview.md  ← START HERE (single-entry walkthrough)
├── guides/                   ← task-oriented "how do I …"
├── reference/                ← technical deep-dive "how does X work"
├── analysis/                 ← research, postmortems, comparative code study
├── experimental/             ← scaffolded-but-not-validated workflow tutorials
└── experiments/              ← per-experiment logs (hypothesis → setup → results)
```

**Why this split:**
- `guides/` = you want to *do* something. Prose is action-oriented.
- `reference/` = you want to *understand* something. Prose is structural.
- `analysis/` = one-shot investigations (postmortems, competitor-code
  comparisons, decision docs). Frozen in time; re-read for context.
- `experimental/` pairs with `example_workflows/experimental/`.
- `experiments/` follows the convention in `experiments/README.md`.

Case studies of actual prompt schedules live unscrubbed in gitignored
`internal/prompts/` — public guides distill patterns inline rather than
linking out (parallel-scrubbed-copy convention retired 2026-04-25).

---

## Task-first index

### "I want to build or modify a workflow"
- End-to-end pipeline (init image → VLM → audio analysis → LLM → schedule → workflow): [`guides/prompt_workflow_end_to_end.md`](guides/prompt_workflow_end_to_end.md)
- Set dimensions / aspect ratio / per-iteration window length (`LTXFramePlanner`): [`reference/frame_planner_reference.md`](reference/frame_planner_reference.md)
- Loop pacing — how stride / overlap / iteration count get computed (`AudioLoopController`): [`reference/audio_loop_controller.md`](reference/audio_loop_controller.md)
- The LATENT-loop workflow, node by node: [`reference/pipeline_flow_latent.md`](reference/pipeline_flow_latent.md)
- LTXVLoopingSampler structural reference (video-only; NOT for music video): [`reference/ltxv_looping_sampler_reference.md`](reference/ltxv_looping_sampler_reference.md)
- Fix one section of a previously generated video (retake): [`guides/retake_guide.md`](guides/retake_guide.md)
- Bench / A-B procedure (sage variants, profiling arms): [`guides/bench_workflow_guide.md`](guides/bench_workflow_guide.md)
- Post-loop spatial upscale (2×, 3-step σ-tail refine; staged draft): `scripts/build_upscale_workflow.py` — see `reference/debug_tools.md` "Workflow build scripts"

### "I want to write a prompt schedule"
- Project-specific rules + variation patterns: [`guides/prompt_creation_guide.md`](guides/prompt_creation_guide.md)
- LLM-mediated schedule generation: [`guides/prompt_workflow_end_to_end.md`](guides/prompt_workflow_end_to_end.md)
- Pre-encoding the schedule outside the loop (entity reference): [`reference/timestamp_prompt_schedule_batch_encode.md`](reference/timestamp_prompt_schedule_batch_encode.md)
- Standup / dialogue system prompt (music variant ships embedded in analyzer JSON): [`reference/standup_system_prompt.md`](reference/standup_system_prompt.md)
- Raw Lightricks i2v/t2v system prompts (historical reference): [`reference/ltx23_prompt_system_prompts.md`](reference/ltx23_prompt_system_prompts.md)

### "I want to analyze audio / wire audio-reactive nodes"
- Offline + runtime analysis, AudioPitchDetect: [`guides/audio_analysis_guide.md`](guides/audio_analysis_guide.md)

### "My output looks wrong / workflow won't run"
- **First stop**: [`guides/debugging_guide.md`](guides/debugging_guide.md) — symptom → first-check table
- Canonical first-pass when validation fails: [`reference/debug_tools.md`](reference/debug_tools.md) (or invoke `/diagnose-workflow`)
- Iteration-boundary seam artifacts (per-frame ghost residual scan): `scripts/diagnose_overlap_seams.py` — see `reference/debug_tools.md` "Inspection scripts"
- Iter-over-iter drift / heatmap frames / lost continuity (`noise_mask` semantics): [`reference/noise_mask_semantics.md`](reference/noise_mask_semantics.md)
- ModelPatcher offload asymmetry (why CLIP cannot enter the loop body): [`analysis/nag_object_patches_offload_asymmetry.md`](analysis/nag_object_patches_offload_asymmetry.md)
- Sampler choice (why `euler` is mandatory): [`reference/sampler_reference.md`](reference/sampler_reference.md)
- NAG deep dive (mechanism + loop-body constraint + troubleshooting): [`reference/nag_technical_reference.md`](reference/nag_technical_reference.md)
- LTXVLoopingSampler AV incompatibility, capability gaps: [`analysis/ltx23_gaps_analysis.md`](analysis/ltx23_gaps_analysis.md)

### "I want to profile performance"
- `torch.profiler` opt-in three-node integration: [`guides/profiling_guide.md`](guides/profiling_guide.md)
- Telemetry / tracing (sage trace, exec log, summary aggregator): [`reference/telemetry_and_tracing.md`](reference/telemetry_and_tracing.md)
- Sage attention node + mask-aware routing: [`reference/sage_attention.md`](reference/sage_attention.md)

### "I need LTX 2.3 model internals"
- Image guides, latent volume math, VAE conversion, AdaIN, noise_mask, conditioning path: [`reference/ltx23_model_reference.md`](reference/ltx23_model_reference.md)
- LTX-2 native conditioning types + `MultiModalGuiderFactory` per-sigma guidance: [`analysis/ltx2_native_conditioning_analysis.md`](analysis/ltx2_native_conditioning_analysis.md)
- LTX-Desktop `ModalitySpec`, `TemporalRegionMask` (retake), frozen-modality semantics: [`analysis/ltx_desktop_conditioning_analysis.md`](analysis/ltx_desktop_conditioning_analysis.md)

### "I want to add multi-frame guides (KJNodes / ComfyUI-LTXVideo)"
- Guide chaining, `LTXVAddLatentGuide` hierarchy: [`analysis/comfyui_ltxvideo_multiframe_guide_analysis.md`](analysis/comfyui_ltxvideo_multiframe_guide_analysis.md)
- `LTXVAddGuideMulti` (up to 20 guides), `LTXVAddGuidesFromBatch`: [`analysis/kjnodes_multiframe_guide_analysis.md`](analysis/kjnodes_multiframe_guide_analysis.md)

### "I want to understand lip-sync / frozen-audio prompting"
- Community research on lip-sync prompting + when it applies vs when our frozen-audio workflow diverges: [`analysis/audio_in_prompt_research.md`](analysis/audio_in_prompt_research.md)

### "I need env-vars / runtime knobs"
- Environment-variable registry: [`reference/environment.md`](reference/environment.md)

### "I want to amplify a conditional contribution at inference time"
- CFG-analog amplification pattern (IC-LoRA, style LoRAs, identity LoRAs, attention guidance): [`reference/cfg_analog_amplification.md`](reference/cfg_analog_amplification.md)

### "I want to add a new workflow-mutation fix"
- F-pair convention (apply script + audit check + pre-flight chaining): [`reference/f_pair_convention.md`](reference/f_pair_convention.md)
- Live F-pair inventory + apply-script three-tier staging: [`reference/debug_tools.md`](reference/debug_tools.md)
- Apply-script conventions in detail: [`../scripts/CLAUDE.md`](../scripts/CLAUDE.md)

---

## File reference (alphabetical, per folder)

### Root
| File | Purpose |
|---|---|
| `architecture_overview.md` | Full-stack single-entry walkthrough. Advertised "START HERE" in `CLAUDE.md`. |
| `README.md` | This index. |

### `guides/` — task-oriented how-to
| File | When to read |
|---|---|
| `audio_analysis_guide.md` | Running offline analysis; wiring `AudioPitchDetect`. |
| `bench_workflow_guide.md` | Sage A/B procedure + bench-variant apply scripts. |
| `debugging_guide.md` | Output looks wrong → symptom → first-check. |
| `profiling_guide.md` | Placing `ProfileBegin`/`IterStep`/`End` for a torch.profiler run. |
| `prompt_creation_guide.md` | Project-specific prompt rules + variation patterns. |
| `prompt_workflow_end_to_end.md` | Init image → VLM → audio → LLM → schedule. |
| `retake_guide.md` | Regenerate one `[start, end]` window of a prior render. |

### `reference/` — technical deep-dive
| File | When to read |
|---|---|
| `_atomic_note_template.md` | (For authors.) The shape every new reference note follows + 5-step ingest checklist + anti-pattern list. |
| `audio_loop_controller.md` | `AudioLoopController` — loop pacing; stride from integer latents; F4/F5/F6/F7 audits. |
| `cfg_analog_amplification.md` | Inference-time pattern — `(positive_with_X, positive_without_X) → CFGGuider` for amplifying any conditional. |
| `debug_tools.md` | Canonical first-pass when a workflow won't run; audit invariant table; apply-script three-tier staging; RUN_ID artifact correlation. |
| `environment.md` | Environment-variable registry (sage trace, exec log, per-prompt routing, etc.). |
| `f_pair_convention.md` | Apply-script + audit-check pairing convention; pre-flight chaining; how to add a new F-pair. |
| `frame_planner_reference.md` | `LTXFramePlanner` — single-source-of-truth dimension config; snap rules + wiring + F8 audit. |
| `ltx23_model_reference.md` | Image guides, latent volume, VAE conversion, AdaIN, noise_mask, conditioning path. |
| `noise_mask_semantics.md` | `noise_mask=0`/`1` semantics; setters + strippers; loop-body discipline; failure modes. |
| `timestamp_prompt_schedule_batch_encode.md` | Pre-encodes prompts outside the loop; pairs with `ConditioningSelectByIteration`; prevents NAG silent disengagement. |
| `ltx23_prompt_system_prompts.md` | Raw Lightricks i2v/t2v system prompts + why our frozen-audio + i2v workflow prefers concise prompts. |
| `ltxv_looping_sampler_reference.md` | Video-only structural reference for `LTXVLoopingSampler`. We don't recommend building this for music video (AV-incompatible). |
| `nag_technical_reference.md` | LTX2_NAG — attention math, widgets, closure-capture mechanism, NAG×CFG composition, troubleshooting. |
| `pipeline_flow_latent.md` | LATENT workflow node-by-node trace — the primary working baseline. |
| `sage_attention.md` | `AudioLoopHelperSageAttention` — parameters, arch-filtered mode combo, mask-aware routing (default `auto_mask_aware`), JSONL telemetry schema. |
| `sampler_reference.md` | `euler` vs `euler_ancestral` vs `euler_ancestral_cfg_pp` with ComfyUI + MultimodalGuider source walkthrough. |
| `standup_system_prompt.md` | LLM system prompt for standup / dialogue schedule generation. |
| `telemetry_and_tracing.md` | What's captured (and what isn't), output paths, retention, end-to-end aggregator workflow. |

### `analysis/` — research, postmortems, comparative study
| File | What it covers |
|---|---|
| `audio_in_prompt_research.md` | Community research on lip-sync prompting, with framing about when it applies vs when our frozen-audio workflow diverges. |
| `comfyui_ltxvideo_multiframe_guide_analysis.md` | Guide chaining; `LTXVAddLatentGuide*` hierarchy. |
| `kjnodes_multiframe_guide_analysis.md` | `LTXVAddGuideMulti` (≤20 guides); `LTXVAddGuidesFromBatch`. |
| `ltx23_gaps_analysis.md` | Capability gaps; `LTXVLoopingSampler` AV incompatibility. |
| `ltx2_native_conditioning_analysis.md` | LTX-2 native conditioning types; `MultiModalGuiderFactory` per-sigma guidance. |
| `ltx_desktop_conditioning_analysis.md` | LTX-Desktop `ModalitySpec`; `TemporalRegionMask` retake; frozen-modality semantics. |
| `nag_object_patches_offload_asymmetry.md` | Root cause for "CLIP cannot enter the loop body." The 2026-04-22 postmortem behind `TimestampPromptScheduleBatchEncode`. |

### `experimental/` — scaffolded-but-not-validated tutorials
Paired with workflows in `example_workflows/experimental/`. See
`experimental/README.md` for the index.

### `experiments/` — per-experiment logs
Hypothesis → setup → observations → inferences → next steps. Convention
in `experiments/README.md`.

---

## Contributing

- Every new doc: include `Last updated: YYYY-MM-DD` as the first line.
- Filenames: lowercase with underscores. No spaces, no camelCase.
- Write the "why" — decisions, alternatives considered, constraints
  that fixed the shape of the solution. Not just the "what."
- Task-oriented "how do I do X" → `guides/`.
- Deep structural reference "how does X work" → `reference/`.
- One-shot research / postmortem / competitor-code comparison → `analysis/`.
- When you add a doc, add it to this README's task-first index AND
  to `CLAUDE.md`'s "Documentation index" section.
- Breaking changes that alter a formula / value / constraint
  referenced in prose: add the stale phrase to
  `scripts/validate_docs_consistency.py`'s `STALE_PATTERNS` and run
  `uv run --group dev --group analysis python -m pytest tests/test_docs_consistency.py`.
