# ComfyUI-AudioLoopHelper

<p align="center">
  <img src="assets/hero.webp" alt="ComfyUI-AudioLoopHelper" width="500">
</p>

Last updated: 2026-05-08

Custom ComfyUI nodes for full-length music video generation with LTX 2.3.
Drives loop timing from integer-latent counts, freezes audio via
`noise_mask=0`, pre-encodes prompts once outside the loop. Originally built this repo as a few helper nodes for experimenting with
[kijai's LTX 2.3 long-loop extension](https://github.com/kijai/ComfyUI-NativeLooping_testing/blob/main/ltx23_long_loop_extension_test.json) - thanks to Kijai for all his work, and for giving me some fun ideas to explore.

> Power-user repo. Assumes you know ComfyUI. Architecture nuance lives in
> `docs/architecture_overview.md`

## Quick start

Open `example_workflows/audio-loop-music-video_latent.json` in ComfyUI.
The workflow itself documents what to change via group titles, node titles,
and Note nodes. Four things to set:

1. **LoadAudio** — drop your song.
2. **LoadImage** — drop the init image. Any size; auto-resized adaptively. Matches the first scene visually.
3. **start_seed** — any int.
4. **TimestampPromptScheduleBatchEncode** — paste the schedule. The initial-render prompt is read from the `0:00` entry (no separate node).

Optional knob: **`first_frame_guide_strength`** (`FloatConstant #1269`). Default `1.0` pins init image to every iter's last frame for max identity stability. Lower (`0.5`/`0.3`) for music-video expressivity at the cost of cross-iter identity drift.

> **On prompt budget.** LTX 2.3's cross-attention has to share its
> token budget across text, audio coherence, and (with i2v) image
> coherence. Concise prompts usually win. Pick the verb that matches
> the visible action you want — `is singing` for vocal performance,
> `is dancing` for movement, `is playing <instrument>` for instrumental,
> etc. Generic verbs (`performing`, `vocalizing`) dilute the signal.
> Without an i2v init, text has to do more work and may need to be
> longer. With i2v, text should be tight. Pick where to spend your
> constraints.

For (4), generate copy-paste-ready text from `scripts/analyze_audio_features.py`:

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

**Companion repos:**

This project coordinates with two kinds of sister repos:

| Repo | Bucket | Role |
|---|---|---|
| [fblissjr/SageAttention-ada](https://github.com/fblissjr/SageAttention-ada) | Sister fork | SageAttention fork with mask-aware routing tuned for LTX 2.3 cross-attention (active; recommended on Ada — RTX 4090) |
| [fblissjr/audio-loop-lab](https://github.com/fblissjr/audio-loop-lab) | Companion umbrella | Workload glue across upstream libs — torchao filter functions for the LTX DiT, cross-library bench harness (scaffold pending) |

The split is by upstream lineage: **forks** patch an upstream library's internals (small surface, rebase tax accepted); **umbrellas** build *on top of* upstream libraries (no lineage, free to grow modularly).

**SageAttention-ada specifics:** the shipped workflows wire `AudioLoopHelperSageAttention` (`auto_mask_aware`, ~1.22× e2e speedup on production iclora workload) which expects this build. **No build, or incompatible hardware?** Bypass `AudioLoopHelperSageAttention` (set `mode=4`) and either run with default attention or use KJNodes sage in its place.

**Optional:**

[ComfyUI-MelBandRoFormer](https://github.com/DrJKL/ComfyUI-MelBandRoFormer)
— vocal separation. Bypassed by default in shipped workflows. Tons of different model variations out on HF for this depending on your use case.

## Workflow variants

| File | Use when |
|---|---|
| `audio-loop-music-video_latent.json` | **Default. Start here.** Pre-encoded audio, IC-LoRA scaffolding bypassed, two LoRA loaders bypassed, 9-group two-row layout, Note-annotated. Un-bypass IC-LoRA chain to enable visual reference adapters; un-bypass distill LoRA when running base ltx-2.3 dev. |
| `audio-loop-music-video_latent_keyframe.json` | Per-section reference images. |
| `audio-loop-music-video_latent_validator.json` | Adds `LoopConfigValidator` + `PreviewAny`. |
| `audio-loop-music-video_latent_stg.json` | A/B target — Spatial-Temporal Guidance instead of CFG. |
| `audio-loop-music-video_image_adain_perstep.json` | Per-step AdaIN, per-iter VAE round-trip. Color-drift prevention. |
| `audio-loop-music-video_retake.json` | Regenerate a `[start, end]` window of an existing render. |

Experimental forks live in `example_workflows/experimental/` paired with
`docs/experiments/` run logs. Not on the shipped-promotion path.

## Audio feature analysis

`scripts/analyze_audio_features.py` extracts BPM, key, structure, F0, and
emits an LTX-2.3-ready timestamp-prompt schedule. Paste the whole schedule
into `TimestampPromptScheduleBatchEncode`; the initial-render prompt comes
from its `0:00+` entry automatically.

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
uv run --group dev python scripts/audit_workflows.py example_workflows/audio-loop-music-video_latent.json

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
