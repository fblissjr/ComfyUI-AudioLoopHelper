# ComfyUI-AudioLoopHelper

<p align="center">
  <img src="assets/hero.webp" alt="ComfyUI-AudioLoopHelper" width="500">
</p>

Last updated: 2026-05-04

Custom ComfyUI nodes for full-length music video generation with LTX 2.3.
Drives loop timing from integer-latent counts, freezes audio via
`noise_mask=0`, pre-encodes prompts once outside the loop. Originally built this repo as a few helper nodes for experimenting with
[kijai's LTX 2.3 long-loop extension](https://github.com/kijai/ComfyUI-NativeLooping_testing/blob/main/ltx23_long_loop_extension_test.json) - thanks to Kijai for all his work, and for giving me some fun ideas to explore.

> Power-user repo. Assumes you know ComfyUI. Architecture nuance lives in
> `docs/architecture_overview.md`

## Quick start — the intro workflow

Open `example_workflows/audio-loop-music-video_latent_intro.json` in ComfyUI.
The workflow itself documents what to change via group titles, node titles,
and Note nodes. Five things to set:

1. **LoadAudio** — drop your song.
2. **LoadImage** — drop the init image (matches the first scene visually).
3. **start_seed** — any int.
4. **CLIPTextEncode (Node 169)** — initial-render prompt.
5. **TimestampPromptScheduleBatchEncode** — paste the schedule.

> **On prompt budget.** LTX 2.3's cross-attention has to share its
> token budget across text, audio coherence, and (with i2v) image
> coherence. Concise prompts usually win. Pick the verb that matches
> the visible action you want — `is singing` for vocal performance,
> `is dancing` for movement, `is playing <instrument>` for instrumental,
> etc. Generic verbs (`performing`, `vocalizing`) dilute the signal.
> Without an i2v init, text has to do more work and may need to be
> longer. With i2v, text should be tight. Pick where to spend your
> constraints.

For (4) + (5), generate copy-paste-ready text from `scripts/analyze_audio_features.py`:

```bash
uv sync --group analysis
uv run --group analysis python scripts/analyze_audio_features.py your_song.wav \
  --subject "your scene description" --trim 5
```

Run.

LoRAs and IC-LoRA scaffolding ship bypassed-by-default — un-bypass when
you need them. Layout, defaults, and bypass-toggle annotations are all in
the workflow itself.

## Dependencies

**Required custom nodes:**

| Repo | Provides |
|---|---|
| [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) | LTX 2.3 nodes (LTXVAddLatentGuide, LTXVCropGuides, LTXVPreprocess, IC-LoRA) |
| [ComfyUI-NativeLooping_testing](https://github.com/kijai/ComfyUI-NativeLooping_testing) | TensorLoopOpen / TensorLoopClose |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Set/Get nodes, LTX2_NAG, LTXVImgToVideoInplaceKJ, ImageResizeKJv2, GetImageRangeFromBatch, SimpleCalculatorKJ |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | VHS_LoadVideo, VHS_VideoCombine |

**Recommended if you have a 4090/Ada architecture:**

[fblissjr/SageAttention-ada](https://github.com/fblissjr/SageAttention-ada)
— our SageAttention fork. The shipped workflows wire
`AudioLoopHelperSageAttention` (`auto_mask_aware`, ~1.22× e2e speedup
on production iclora workload) which expects this build. **No build, or
incompatible hardware?** Bypass `AudioLoopHelperSageAttention` (set
`mode=4`) and either run with default attention or use KJNodes sage in its place.

**Optional:**

[ComfyUI-MelBandRoFormer](https://github.com/DrJKL/ComfyUI-MelBandRoFormer)
— vocal separation. Bypassed by default in shipped workflows. Tons of different model variations out on HF for this depending on your use case.

## Workflow variants

| File | Use when |
|---|---|
| `audio-loop-music-video_latent_intro.json` | **Default. Start here.** Pre-encoded audio, IC-LoRA bypassed, two LoRA loaders bypassed, layout grouped + Note-annotated. |
| `audio-loop-music-video_latent.json` | Same pipeline, no LoRA / IC-LoRA scaffolding. |
| `audio-loop-music-video_latent_iclora.json` | IC-LoRA enabled by default. |
| `audio-loop-music-video_latent_iclora_audio_pre_encode.json` | IC-LoRA + pre-encoded audio (~12.8s/render saved). |
| `audio-loop-music-video_latent_keyframe.json` | Per-section reference images. |
| `audio-loop-music-video_latent_validator.json` | Adds `LoopConfigValidator` + `PreviewAny`. |
| `audio-loop-music-video_latent_stg.json` | A/B target — Spatial-Temporal Guidance instead of CFG. |
| `audio-loop-music-video_image_adain_perstep.json` | Per-step AdaIN, per-iter VAE round-trip. Color-drift prevention. |
| `audio-loop-music-video_retake.json` | Regenerate a `[start, end]` window of an existing render. |

Experimental forks live in `example_workflows/experimental/` paired with
`docs/experiments/` run logs. Not on the shipped-promotion path.

## Audio feature analysis

`scripts/analyze_audio_features.py` extracts BPM, key, structure, F0, and
emits LTX-2.3-ready prompt schedules. Output has two clearly labeled
sections — the initial-render prompt (paste into Node 169) and the
per-iteration schedule (paste into `TimestampPromptScheduleBatchEncode`).

Common invocations:

```bash
# Subject-driven schedule generation
uv run --group analysis python scripts/analyze_audio_features.py song.wav \
  --subject "a woman in her 30s with dark hair singing in a basement workshop" --trim 5

# Pick an ambition tier (default 2a). All tiers in audio_analysis_guide.md.
uv run --group analysis python scripts/analyze_audio_features.py song.wav \
  --subject "..." --scene-diversity 3b

# JSON export for LLM-assisted schedule generation
uv run --group analysis python scripts/analyze_audio_features.py song.wav \
  --subject "..." -j analysis.json
```

Full reference: [`docs/guides/audio_analysis_guide.md`](docs/guides/audio_analysis_guide.md).
End-to-end LLM workflow: [`docs/guides/prompt_workflow_end_to_end.md`](docs/guides/prompt_workflow_end_to_end.md).
Prompt-authoring rules: [`docs/guides/prompt_creation_guide.md`](docs/guides/prompt_creation_guide.md).

## Validation + debugging

When a workflow fails to validate or produces wrong output:

```bash
# Audit shipped workflows (named topology checks + generic invariants)
uv run --group dev python scripts/audit_workflows.py

# Audit one file
uv run --group dev python scripts/audit_workflows.py example_workflows/audio-loop-music-video_latent_intro.json

# DAG topo-sort if audit is clean but it still fails
uv run --group dev python scripts/analyze_workflow_dag.py \
  example_workflows/audio-loop-music-video_latent.json --format ascii
```

Or invoke `/diagnose-workflow` for the canonical first-pass.

Full reference: [`docs/reference/debug_tools.md`](docs/reference/debug_tools.md).
Symptom-first quality troubleshooting: [`docs/guides/debugging_guide.md`](docs/guides/debugging_guide.md).

## Local logging + profiling (off by default)

These are local-only debugging instruments that **this plugin** ships. Both
default to off, both write only to plain JSONL files on your own disk
(under gitignored `data/runs/${RUN_ID}/` when launched via
`start_experiment.sh`; under gitignored `internal/analysis/runs/` as a
legacy fallback when `RUN_ID` is unset), and **none of this code makes
any network call or sends data anywhere**. There is no telemetry endpoint,
no analytics service, no "anonymous usage data." It's local file I/O for
your own profiling and bench-analysis. Anything ComfyUI itself does at
runtime is upstream behavior unrelated to this plugin.

Two opt-in instruments + one offline aggregator:

- `AUDIOLOOPHELPER_SAGE_TRACE` — our writer in `nodes_sage.py`. Per-attention-call JSONL when set.
- `COMFYUI_EXEC_LOG` — **our** monkey-patch on ComfyUI's `execute()` (defined in `exec_logger.py`); installs only when the env var is set, no-op otherwise. The env var name has the `COMFYUI_` prefix because it controls our patch on a ComfyUI internal — the patch itself is plugin code.
- `scripts/sage_telemetry_summary.py` — offline aggregator. Reads JSONL files; never writes anything; runs outside ComfyUI.

All three off when env vars are unset. What gets captured + the privacy posture: [`docs/reference/telemetry_and_tracing.md`](docs/reference/telemetry_and_tracing.md).

## Layout

```
nodes*.py             runtime nodes (entry: comfy_entrypoint() in nodes.py)
scripts/              apply scripts + audit + analysis utilities
docs/                 public docs — task-first nav at docs/README.md
example_workflows/    shipped workflow variants
internal/             gitignored design + analysis + experiment notes
.claude/              shared Claude Code harness (subagents, skills, hooks)
```

Architecture overview: [`docs/architecture_overview.md`](docs/architecture_overview.md).
Per-node API + wiring: each runtime class's docstring + [`docs/reference/ltx23_model_reference.md`](docs/reference/ltx23_model_reference.md).
Project conventions for editing this repo: [`CLAUDE.md`](CLAUDE.md).

## License

See [LICENSE](LICENSE).
