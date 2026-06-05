# ComfyUI-AudioLoopHelper

<p align="center">
  <img src="assets/hero.webp" alt="ComfyUI-AudioLoopHelper" width="500">
</p>

Last updated: 2026-06-05

> [!NOTE]
> **Experimental repo — it moves fast and changes often.** Treat these as
> experimental nodes: fork it, customize it, or point your agent of choice at
> the codebase and take the patterns you like (and skip the ones you don't).

Custom ComfyUI nodes for full-length music video generation with LTX 2.3.
Drives loop timing from integer-latent counts, freezes audio via
`noise_mask=0`, pre-encodes prompts once outside the loop. Originally built this repo as a few helper nodes for experimenting with
[kijai's LTX 2.3 long-loop extension](https://github.com/kijai/ComfyUI-NativeLooping_testing/blob/main/ltx23_long_loop_extension_test.json) - thanks to Kijai for all his work, and for giving me some fun ideas to explore.

## Trained adapters (audio-reference IC-LoRA)

Two cuts of the same 1000-step experiment, released as
[**LTX-2.3-22b-IC-LoRA-Audio-Only-Context**](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context)
— an IC-LoRA where the in-context reference is *only audio* (no image, no
video). Load either with the audio IC-LoRA nodes via the single-pass workflow
(see the [workflow table](#workflow-variants) below); background + node
mechanics: [`docs/audio_iclora/index.md`](docs/audio_iclora/index.md).

| Adapter | What it's supposed to do (in theory) | Suggested start |
|---|---|---|
| [`cross_modal_step_01000`](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context/blob/main/cross_modal_step_01000.safetensors) | Adapts the audio stack **and** the cross-modal bridges — gives the audio reference a direct path into the video stream. Default pick if you want the audio to move the picture (mannerisms, energy, expression timing). | Strength ~0.5 (working band ~0.3–0.75, reference-dependent). With the per-stream loader, push `bridge_strength` above `audio_strength` to amplify the audio→video coupling. |
| [`audio_only_step_01000`](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context/blob/main/audio_only_step_01000.safetensors) | Adapts the audio stack only — the reference shapes the generated **audio**; the video follows via the frozen base coupling. Subtler, more emergent video effect; perturbs the base video path least. | Strength ~0.5 (same band). No bridge keys, so `bridge_strength` is inert. |

Both are early proof-of-concepts: many variables influence the end result
(reference level/quality/content, prompt, seed), each cut has its own strengths,
weaknesses, and tradeoffs, and both may behave differently when retrained on a
better dataset. Reproduce or extend:
[data recipe](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context/blob/main/data_recipe.md) ·
[training config](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context/blob/main/train_config.yaml) ·
[LTX-2 train fork (audio-only IC-LoRA strategy)](https://github.com/fblissjr/LTX-2/tree/audio-guidance-iclora-vtv).

<sub>First proof-of-concept run (the pitch "Helium" probe): [LTX-2.3-22b-IC-LoRA-Helium](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Helium).</sub>

## Demos

Click play — **sound on**. The audio is frozen and *drives* the picture; the
motion is generated against the waveform, not added after.

https://github.com/user-attachments/assets/ed1657c5-35a7-48e0-b384-7e58b4aafafe

<sub>30s of one continuous **2:53 render** — a painted heart pulsing to a drum
loop. Workflow: [`audio_reactive_loop.json`](example_workflows/audio_reactive_loop.json) ·
[writeup](docs/experimental/audio_reactive_workflows.md)</sub>

<details>
<summary><b>▶ Audio-reactive, second take</b></summary>

https://github.com/user-attachments/assets/3760b301-8fc7-4b43-8f24-3128558cff69

<sub>Same drumbeat as the heart, different init — the stomps land on the beat.</sub>

</details>

<details>
<summary><b>▶ One 5-second audio clip, four takes — a sketch line drives the whole scene</b></summary>

https://github.com/user-attachments/assets/5464d767-85ba-4827-b448-5e93a2fd8871

https://github.com/user-attachments/assets/c3149832-b830-4013-8701-d562ee91eceb

https://github.com/user-attachments/assets/a96cc95c-3268-4d95-a45d-6e3c0d17dcbb

https://github.com/user-attachments/assets/e198faa1-4fbf-4e88-996d-7af7634e8144

<sub>Four generations from the **same ~5s comedy-sketch audio + one init image** —
the audio alone sets the performance: timing, delivery, gesture.</sub>

</details>

<details>
<summary><b>▶ Full-length music video — the default workflow</b></summary>

https://github.com/user-attachments/assets/51d96aaf-c938-4ae4-8c58-34524e8fe79d

<sub>Opening 30s of one continuous **2:51 render** from
[`audio-loop-music-video_latent.json`](example_workflows/audio-loop-music-video_latent.json) —
one song + one init image + a timestamp prompt schedule.</sub>

</details>

<details>
<summary><b>▶ One image, three songs — the audio reshapes everything</b></summary>

https://github.com/user-attachments/assets/80b02031-87f5-44fe-9448-da0021339494

https://github.com/user-attachments/assets/e276fb04-b411-4d38-b03b-2ee30122f7e7

https://github.com/user-attachments/assets/e3920e6b-9540-461b-9b65-0c1e7424c0d6

<sub>Three full renders through the default workflow from the **same single init
image**; the first two share essentially the same prompt (the astronaut run
differs). The audio is the change — and it redraws pacing, motion, and mood.</sub>

</details>

**Three ways in:**

- **"Just show me."** [Demos](#demos) above; more variants in the [workflow table](#workflow-variants) below. The [model-card examples](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context#examples) cover the audio-steering (IC-LoRA) side. To run one yourself: open the default workflow ([Quick start](#quick-start)), drop a song + an image, run.
- **"I want to use it."** Quick start below, then the **docs hub: [`docs/README.md`](docs/README.md)** — the task-first index ("I want to do X, which doc?"). Power-user repo; assumes you know ComfyUI.
- **"I want to verify, reproduce, or extend it."** Architecture walkthrough: [`docs/architecture_overview.md`](docs/architecture_overview.md), then per-node docstrings + [`docs/reference/`](docs/reference/). The audio IC-LoRA training story: [`docs/audio_iclora/index.md`](docs/audio_iclora/index.md) + the [trained adapters](#trained-adapters-audio-reference-ic-lora) above (data recipe + config + train fork). Invariants are enforced as code — the pytest suite and the workflow-topology audit (`scripts/audit_workflows.py`) run in CI.

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

**Optional:** [ComfyUI-MelBandRoFormer](https://github.com/DrJKL/ComfyUI-MelBandRoFormer)
for vocal separation (bypassed by default). Companion repo:
[fblissjr/comfy-workbench](https://github.com/fblissjr/comfy-workbench)
(shared tooling + conventions across my ComfyUI work).

## SageAttention fork (optional)

The sister fork [fblissjr/SageAttention-ada](https://github.com/fblissjr/SageAttention-ada)
is tested and optimized for **RTX 4090 / Ada** architectures. **You don't need
it** — unless you use our sage node: the shipped workflows wire
`AudioLoopHelperSageAttention` (`auto` mode), which expects this build. No
build, or different hardware? Bypass the node (`mode=4`) or swap in KJNodes
sage — everything else works without it. Deep dive:
[`docs/reference/sage_attention.md`](docs/reference/sage_attention.md).

## Workflow variants

Shipped at top-level `example_workflows/`:

| File | What it does | Detail |
|---|---|---|
| `audio-loop-music-video_latent.json` | **Default — start here.** Full-length music video: i2v init + your full audio track frozen; loops overlapping windows so the video tracks the song end-to-end. | [`docs/architecture_overview.md`](docs/architecture_overview.md) |
| `audio-loop-music-video_latent_av_inversion.json` | **Video → audio.** Dialogue replacement / voice-clone dub over held footage. | [`docs/guides/dialogue_replacement_guide.md`](docs/guides/dialogue_replacement_guide.md) |
| `audio-loop-music-video_latent_keyframe.json` | **Per-section keyframe re-anchoring** — combats drift on long renders; scene changes synced to song structure. | [`example_workflows/working_docs/keyframe_iter_anchor_design.md`](example_workflows/working_docs/keyframe_iter_anchor_design.md) |
| `audio-loop-music-video_retake.json` | **Regenerate one section** — re-roll a `[start, end]` window, rest held as fixed context. | [`docs/guides/retake_guide.md`](docs/guides/retake_guide.md) |
| `audio_reactive_loop.json` | **Audio-driven motion** — init image animated so its motion tracks the (frozen) audio. | [`docs/experimental/audio_reactive_workflows.md`](docs/experimental/audio_reactive_workflows.md) |
| `audio-ic-lora_single-pass.json` | **Audio-reference IC-LoRA (single pass)** — steer a render from a reference audio clip, using the [trained adapters](#trained-adapters-audio-reference-ic-lora) above. | [`docs/audio_iclora/index.md`](docs/audio_iclora/index.md) |
| `decode-latent-to-video.json` | **Crash recovery** — loop workflows save the assembled latent before the final decode; if a render dies there, this decodes the saved `.latent` to the finished video (copy it into ComfyUI's input dir first). | [`docs/reference/benchmarking_memory_pressure.md`](docs/reference/benchmarking_memory_pressure.md) |

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

## Local profiling (off by default)

Opt-in, env-var-gated JSONL instruments for your own performance debugging
(attention trace, exec log, offline summarizer). Local files only.
Details: [`docs/reference/telemetry_and_tracing.md`](docs/reference/telemetry_and_tracing.md).

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
