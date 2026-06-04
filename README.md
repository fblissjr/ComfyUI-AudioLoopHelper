# ComfyUI-AudioLoopHelper

<p align="center">
  <img src="assets/hero.webp" alt="ComfyUI-AudioLoopHelper" width="500">
</p>

Last updated: 2026-06-03

Custom ComfyUI nodes for full-length music video generation with LTX 2.3.
Drives loop timing from integer-latent counts, freezes audio via
`noise_mask=0`, pre-encodes prompts once outside the loop. Originally built this repo as a few helper nodes for experimenting with
[kijai's LTX 2.3 long-loop extension](https://github.com/kijai/ComfyUI-NativeLooping_testing/blob/main/ltx23_long_loop_extension_test.json) - thanks to Kijai for all his work, and for giving me some fun ideas to explore.

> Power-user repo. Assumes you know ComfyUI.
> **Docs hub: [`docs/README.md`](docs/README.md)** — the task-first index ("I want to do X, which doc?").
> Single-pass architecture walkthrough: [`docs/architecture_overview.md`](docs/architecture_overview.md).

## Quick start

Open `example_workflows/audio-loop-music-video_latent.json` in ComfyUI.
The workflow documents itself via group titles, node titles, and Note nodes.
Four things to set:

1. **LoadAudio** — drop your song.
2. **LoadImage** — drop the init image (any size; auto-resized adaptively; matches the first scene visually).
3. **start_seed** — any int.
4. **TimestampPromptScheduleBatchEncode** — paste the schedule. The initial-render prompt is read from the `0:00` entry.

Generate the schedule from your song:

```bash
uv sync --group analysis
uv run --group analysis python scripts/analyze_audio_features.py your_song.wav \
  --subject "your scene description" --trim 5
```

Run. LoRAs + IC-LoRA scaffolding ship bypassed-by-default — un-bypass when you
need them; knobs (like `first_frame_guide_strength`: 1.0 = max identity lock,
lower for expressivity) are annotated in the workflow itself. For the rest:

- Prompt authoring — verb choice, token budget, continuation framing: [`docs/guides/prompt_creation_guide.md`](docs/guides/prompt_creation_guide.md)
- Audio analysis — all flags, scene-diversity tiers, JSON export: [`docs/guides/audio_analysis_guide.md`](docs/guides/audio_analysis_guide.md)
- End-to-end LLM schedule workflow (init image → VLM → schedule): [`docs/guides/prompt_workflow_end_to_end.md`](docs/guides/prompt_workflow_end_to_end.md)

## Dependencies

**Required custom nodes:**

| Repo | Provides |
|---|---|
| [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) | LTX 2.3 nodes (LTXVAddLatentGuide, LTXVCropGuides, LTXVPreprocess, IC-LoRA) |
| [ComfyUI-NativeLooping_testing](https://github.com/kijai/ComfyUI-NativeLooping_testing) | TensorLoopOpen / TensorLoopClose |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Set/Get nodes, LTX2_NAG, LTXVImgToVideoInplaceKJ, ImageResizeKJv2, and more |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | VHS_LoadVideo, VHS_VideoCombine |

**Sage attention:** shipped workflows wire `AudioLoopHelperSageAttention` in
`auto` mode, built for the sister fork
[fblissjr/SageAttention-ada](https://github.com/fblissjr/SageAttention-ada)
(recommended on Ada / RTX 40xx). No build, or incompatible hardware? Bypass the
node (`mode=4`) or swap in KJNodes sage. Deep dive: [`docs/reference/sage_attention.md`](docs/reference/sage_attention.md).

**Optional:** [ComfyUI-MelBandRoFormer](https://github.com/DrJKL/ComfyUI-MelBandRoFormer)
for vocal separation (bypassed by default). Companion repo:
[fblissjr/comfy-workbench](https://github.com/fblissjr/comfy-workbench)
(shared tooling + conventions across my ComfyUI work).

## Workflow variants

Shipped at top-level `example_workflows/`:

| File | What it does | Detail |
|---|---|---|
| `audio-loop-music-video_latent.json` | **Default — start here.** Full-length music video: i2v init + your full audio track frozen; loops overlapping windows so the video tracks the song end-to-end. | [`docs/architecture_overview.md`](docs/architecture_overview.md) |
| `audio-loop-music-video_latent_av_inversion.json` | **Video → audio.** Dialogue replacement / voice-clone dub over held footage. | [`docs/guides/dialogue_replacement_guide.md`](docs/guides/dialogue_replacement_guide.md) |
| `audio-loop-music-video_latent_keyframe.json` | **Per-section keyframe re-anchoring** — combats drift on long renders; scene changes synced to song structure. | [`example_workflows/working_docs/keyframe_iter_anchor_design.md`](example_workflows/working_docs/keyframe_iter_anchor_design.md) |
| `audio-loop-music-video_retake.json` | **Regenerate one section** — re-roll a `[start, end]` window, rest held as fixed context. | [`docs/guides/retake_guide.md`](docs/guides/retake_guide.md) |
| `audio_reactive_loop.json` | **Audio-driven motion** — init image animated so its motion tracks the (frozen) audio. | [`docs/experimental/audio_reactive_workflows.md`](docs/experimental/audio_reactive_workflows.md) |
| `audio-ic-lora_single-pass.json` | **Audio-reference IC-LoRA (single pass)** — steer a render from a reference audio clip. Pairs with our trained adapter: [Audio-Only-Context on Hugging Face](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context). | [`docs/audio_iclora/index.md`](docs/audio_iclora/index.md) |

More variants in `example_workflows/experimental/` (paired with run logs in
`docs/experiments/`; inventory in [`docs/experimental/README.md`](docs/experimental/README.md));
retired ones in `example_workflows/archive/`. Design notes for the shipped
variants live in [`example_workflows/working_docs/`](example_workflows/working_docs/).

## Validation + debugging

```bash
# topology checks + generic invariants across shipped workflows
uv run --group dev python scripts/audit_workflows.py
```

`/diagnose-workflow` is the canonical first-pass when something won't run.
Tooling reference: [`docs/reference/debug_tools.md`](docs/reference/debug_tools.md).
Symptom-first quality troubleshooting: [`docs/guides/debugging_guide.md`](docs/guides/debugging_guide.md).

## Local logging + profiling (off by default)

Two opt-in, env-var-gated instruments (`AUDIOLOOPHELPER_SAGE_TRACE`,
`COMFYUI_EXEC_LOG`) plus an offline aggregator. All write plain JSONL to your
own disk only — **no network calls, no telemetry endpoint, no "anonymous usage
data"; it's local file I/O for your own profiling.** What gets captured + the
full privacy posture: [`docs/reference/telemetry_and_tracing.md`](docs/reference/telemetry_and_tracing.md).

## Layout

```
nodes*.py             runtime nodes (entry: comfy_entrypoint() in nodes.py)
scripts/              apply scripts + audit + analysis utilities
docs/                 public docs — task-first nav at docs/README.md
example_workflows/    shipped workflow variants (+ working_docs/ design notes)
internal/             gitignored design + analysis + experiment notes
.claude/              shared Claude Code harness (subagents, skills, hooks)
```

Per-node API + wiring: each runtime class's docstring + [`docs/reference/ltx23_model_reference.md`](docs/reference/ltx23_model_reference.md).
Project conventions for editing this repo: [`CLAUDE.md`](CLAUDE.md).

## License

See [LICENSE](LICENSE).
