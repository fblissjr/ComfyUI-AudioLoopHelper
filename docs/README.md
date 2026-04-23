Last updated: 2026-04-23 (docs cleanup round 2)

# docs/ — ComfyUI-AudioLoopHelper documentation

**If you're new to this repo, start at [`architecture_overview.md`](architecture_overview.md).**
It's the single-pass walkthrough of the full stack (our workflow →
ComfyUI core → ComfyUI-LTXVideo → KJNodes → LTX-2 model) and links
into every deeper doc.

This README is the task-first nav index — it answers the question
"I want to do X, which doc has the answer?" Use it when you already
know what you're looking for.

---

## Directory layout

```
docs/
├── README.md                     ← you are here (task-first nav)
├── architecture_overview.md      ← START HERE (single-entry walkthrough)
├── guides/                       ← task-oriented "how do I …"
├── reference/                    ← technical deep-dive "how does X work"
├── analysis/                     ← research, postmortems, comparative code study
└── examples/                     ← scrubbed prompt-schedule case studies
```

**Why this split:**
- `guides/` = you want to *do* something (build a workflow, write a
  prompt schedule, debug, profile). Prose is action-oriented.
- `reference/` = you want to *understand* something (why euler over
  euler_ancestral; what NAG does; what node 606 is wired to). Prose
  is structural.
- `analysis/` = one-shot investigations (postmortems, competitor-code
  comparisons, decision docs). Frozen in time; re-read for context.
- `examples/` = actual schedules that ran, with what worked and what
  broke. Patterns transfer; specific assets are scrubbed.

---

## Task-first index

### "I want to build or modify a workflow"
- End-to-end pipeline (init image → VLM → audio analysis → LLM → schedule → workflow): [`guides/prompt_workflow_end_to_end.md`](guides/prompt_workflow_end_to_end.md)
- The LATENT-loop workflow, node by node: [`reference/pipeline_flow_latent.md`](reference/pipeline_flow_latent.md)
- The IMAGE-loop workflow, node by node (reference-only now): [`reference/pipeline_flow_image.md`](reference/pipeline_flow_image.md)
- LTXVLoopingSampler structural reference (video-only; NOT for music video): [`reference/ltxv_looping_sampler_reference.md`](reference/ltxv_looping_sampler_reference.md)
- Upscale workflow: not yet shipped; design doc at `internal/design/upscale_workflow_design.md`

### "I want to write a prompt schedule"
- Project-specific rules + variation patterns (A/B/C): [`guides/prompt_creation_guide.md`](guides/prompt_creation_guide.md)
- LLM-mediated schedule generation (rules R1-R8 + tier semantics): [`guides/prompt_workflow_end_to_end.md`](guides/prompt_workflow_end_to_end.md)
- Standup / dialogue system prompt (music-variant ships embedded in the analyzer JSON): [`reference/standup_system_prompt.md`](reference/standup_system_prompt.md)
- Raw Lightricks i2v/t2v system prompts (historical reference): [`reference/ltx23_prompt_system_prompts.md`](reference/ltx23_prompt_system_prompts.md)
- Real case studies to copy from: [`examples/README.md`](examples/README.md)

### "I want to analyze audio / wire audio-reactive nodes"
- Offline + runtime analysis, AudioPitchDetect: [`guides/audio_analysis_guide.md`](guides/audio_analysis_guide.md)

### "My output looks wrong"
- **First stop**: [`guides/debugging_guide.md`](guides/debugging_guide.md) — symptom → first-check table
- ModelPatcher offload asymmetry (why CLIP cannot enter the loop body, aka "NAG silently disengages after iter 1"): [`analysis/nag_object_patches_offload_asymmetry.md`](analysis/nag_object_patches_offload_asymmetry.md)
- Sampler choice (why `euler` is mandatory, why upstream's `euler_ancestral_cfg_pp` is wrong for merged distilled-1.1): [`reference/sampler_reference.md`](reference/sampler_reference.md)
- NAG deep dive (mechanism + operational loop-body constraint + troubleshooting): [`reference/nag_technical_reference.md`](reference/nag_technical_reference.md)
- LTXVLoopingSampler AV incompatibility, capability gaps: [`analysis/ltx23_gaps_analysis.md`](analysis/ltx23_gaps_analysis.md)

### "I want to profile performance"
- `torch.profiler` opt-in three-node integration: [`guides/profiling_guide.md`](guides/profiling_guide.md)

### "I need LTX 2.3 model internals"
- Image guides, latent volume math, VAE conversion, AdaIN, noise_mask, conditioning path, upscaling: [`reference/ltx23_model_reference.md`](reference/ltx23_model_reference.md)
- LTX-2 native conditioning types + `MultiModalGuiderFactory` per-sigma guidance: [`analysis/ltx2_native_conditioning_analysis.md`](analysis/ltx2_native_conditioning_analysis.md)
- LTX-Desktop `ModalitySpec`, `TemporalRegionMask` (retake), frozen-modality semantics: [`analysis/ltx_desktop_conditioning_analysis.md`](analysis/ltx_desktop_conditioning_analysis.md)

### "I want to add multi-frame guides (KJNodes / ComfyUI-LTXVideo)"
- Guide chaining, `LTXVAddLatentGuide` hierarchy: [`analysis/comfyui_ltxvideo_multiframe_guide_analysis.md`](analysis/comfyui_ltxvideo_multiframe_guide_analysis.md)
- `LTXVAddGuideMulti` (up to 20 guides), `LTXVAddGuidesFromBatch`: [`analysis/kjnodes_multiframe_guide_analysis.md`](analysis/kjnodes_multiframe_guide_analysis.md)

### "I want to understand lip-sync / frozen-audio prompting"
- Community research on lip-sync prompting + when to apply it: [`analysis/audio_in_prompt_research.md`](analysis/audio_in_prompt_research.md)
- Worked case: action-track schedule with and without audio descriptors: [`examples/action_prompt6.md`](examples/action_prompt6.md)

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
| `debugging_guide.md` | Output looks wrong → symptom → first-check. |
| `profiling_guide.md` | Placing `ProfileBegin`/`IterStep`/`End` for a torch.profiler run. |
| `prompt_creation_guide.md` | Project-specific prompt rules + variation patterns A/B/C. |
| `prompt_workflow_end_to_end.md` | Init image → VLM → audio → LLM → schedule. |

### `reference/` — technical deep-dive
| File | When to read |
|---|---|
| `ltx23_model_reference.md` | Image guides, latent volume, VAE conversion, AdaIN, noise_mask, conditioning path. |
| `ltx23_prompt_system_prompts.md` | Raw Lightricks i2v/t2v system prompts + why our frozen-audio + i2v workflow prefers concise-not-detailed prompts. |
| `ltxv_looping_sampler_reference.md` | Video-only structural reference for `LTXVLoopingSampler`. We don't recommend building this for music video (AV-incompatible). |
| `nag_technical_reference.md` | LTX2_NAG — attention math, widgets, closure-capture mechanism, NAG×CFG composition, troubleshooting. De-black-boxed 2026-04-23. |
| `pipeline_flow_image.md` | IMAGE workflow summary + diffs vs LATENT. Full node-by-node trace archived to `internal/archive/`. |
| `pipeline_flow_latent.md` | LATENT workflow node-by-node trace — the primary working baseline. |
| `sage_attention.md` | `AudioLoopHelperSageAttention` node — parameters, arch-filtered mode combo, fallback behavior, JSONL telemetry schema. Drop-in alternative to KJNodes' `PathchSageAttentionKJ` with fallback, cleanup, and observability. |
| `sampler_reference.md` | `euler` vs `euler_ancestral` vs `euler_ancestral_cfg_pp` with ComfyUI + MultimodalGuider source walkthrough. |
| `standup_system_prompt.md` | LLM system prompt for standup / dialogue schedule generation (music variant ships embedded in analyzer JSON). |

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

### `examples/` — prompt-schedule case studies
- `examples/README.md` indexes all case studies with a "patterns that transfer" summary.
- Music: `music_prompt1.md` … `music_prompt3.md` (illustrated → cinematic realism arc)
- Action / instrumental: `action_prompt1.md` … `action_prompt6.md` (v5 introduces 20-iter rapid-cut; v6 introduces "frozen audio" insight)
- Standup / dialogue: `prompt_comedy1.md` … `prompt_comedy5.md` (v4 introduces "Cut to …" technique; v5 covers unusual-character init adaptation)

---

## Contributing

- Every new doc: include `Last updated: YYYY-MM-DD` as the first line.
- Filenames: lowercase with underscores. No spaces, no camelCase.
- Write the "why" — decisions, alternatives considered, constraints
  that fixed the shape of the solution. Not just the "what."
- Task-oriented "how do I do X" → `guides/`.
- Deep structural reference "how does X work" → `reference/`.
- One-shot research / postmortem / competitor-code comparison → `analysis/`.
- Scrubbed case study → `examples/`; unscrubbed working copy lives
  in gitignored `internal/prompts/`.
- When you add a doc, add it to this README's task-first index AND
  to `CLAUDE.md`'s "Documentation index" section.
- Breaking changes that alter a formula / value / constraint
  referenced in prose: add the stale phrase to
  `scripts/validate_docs_consistency.py`'s `STALE_PATTERNS` and run
  `uv run --group dev --group analysis python -m pytest tests/test_docs_consistency.py`.
