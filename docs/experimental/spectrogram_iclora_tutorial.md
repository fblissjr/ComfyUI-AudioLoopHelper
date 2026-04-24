Last updated: 2026-04-24

# Spectrogram-as-reference IC-LoRA — Phase 2.0 tutorial

Experimental. This tutorial shows how to test whether a Mel spectrogram of a song, rendered as a video sequence and fed as an IC-LoRA structural reference, makes LTX 2.3 generate video with beat-locked visual rhythm.

Architectural design + iteration ladder lives in `internal/design/spectrogram_reference_design.md` (gitignored internal doc). This file is the hands-on "how do I actually run it" guide.

---

## What you're testing

**Hypothesis:** LTX 2.3's IC-LoRA attention layer, trained on (canny-like structural reference, real video) pairs, will partially generalize if you feed it a spectrogram as the reference — producing video whose visual motion pulses with the music's beat structure.

**What this tests:** does an out-of-distribution structural signal (spectrogram ≠ canny edges) survive the IC-LoRA's VAE-compressed reference encoding enough to influence generation?

**What this does NOT test** (deferred to later Phases):
- Whether it works inside the full audio-loop architecture (Phase 2.3)
- Quantitative beat-sync scoring (Phase 2.2 via `scripts/measure_beat_sync.py`, not yet built)
- Per-iteration vocal/instrumental blending (Phase 2.4)

**Go/no-go gate:** does the spectrogram-referenced render show visibly more rhythm-aligned motion than the same prompt without IC-LoRA? Yes → Phase 2.1 mode-sweep. No / heatmap artifacts → Phase 2 retired with documented failure mode.

---

## Prerequisites

### Models + LoRAs

Paths below use `<comfyui_models>` as a placeholder for your ComfyUI `models/` directory. If you already run the main AudioLoopHelper workflow you'll have most of these; otherwise, download them from their respective sources (linked in the AudioLoopHelper main README + Lightricks' LTX 2.3 Hugging Face repos).

| File | Path | Size | Purpose |
|---|---|---|---|
| `ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` | `<comfyui_models>/diffusion_models/` | ~12 GB | Merged distilled diffusion MODEL |
| `gemma_3_12B_it_fpmixed.safetensors` | `<comfyui_models>/text_encoders/` | ~6 GB | Gemma 3 text encoder |
| `ltx-2.3_text_projection_bf16.safetensors` | `<comfyui_models>/text_encoders/` | small | LTX 2.3 text-projection head |
| `LTX23_video_vae_bf16.safetensors` | `<comfyui_models>/vae/` | ~600 MB | Video VAE |
| `LTX23_audio_vae_bf16.safetensors` | `<comfyui_models>/vae/` | ~80 MB | Audio VAE (for dummy silent audio latent) |
| `MergeGreen_IC-lora_ltx2.3.safetensors` | `<comfyui_models>/loras/` | ~1 GB | IC-LoRA adapter (community Union Control variant; Hugging Face: `MergeGreen/LTX-2.3-IC-LoRA`) |

**IC-LoRA rationale:** `LTXAddVideoICLoRAGuide` is a *generic* node that appends reference images to the sampler's latent + conditioning metadata — by itself, it doesn't teach the model what the reference means. The IC-LoRA file is what's trained on structural-reference → real-video pairs. Without the LoRA, the model gets the guide tokens but ignores them. The MergeGreen Union Control IC-LoRA is trained on canny/depth/pose-style references; this tutorial tests whether that training generalizes out-of-distribution to spectrograms.

**File-name compatibility:** if you have a different Union Control IC-LoRA (e.g. Lightricks' upstream `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`), edit the `ICLORA_FILE` constant at the top of `scripts/apply_spectrogram_iclora_minimal.py` and re-run it. Any Union-Control-trained IC-LoRA should work; the workflow wiring is file-agnostic.

### Tools

- `ffmpeg` on PATH (spectrogram → mp4 + audio dubbing).
- `uv` for Python.
- ComfyUI with `ComfyUI-LTXVideo`, `ComfyUI-KJNodes`, `ComfyUI-VideoHelperSuite` installed.
- A 4090 (or comparable) with ~18 GB VRAM for a 5s render at 832×448.

### Audio

A short, **drum-forward** clip of a song. 5–10 seconds is plenty for the PoC. Avoid:
- Ambient pads, sustained chords (no spectrogram variance → no signal)
- Vocal-only passages (less rhythmic structure in the spectrogram)

Lead with a drum break, a snare-heavy chorus, or a dance-track intro. Whatever the IC-LoRA has the best chance of showing rhythm-lock on.

---

## Step-by-step

### 1. Build the workflow (~1 second, one-time)

```bash
uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py
# → example_workflows/experimental/spectrogram_iclora_minimal.json
```

25-node workflow, scratch-built. Uses our production loader stack + `AudioLoopHelperSageAttention` with the mask-aware default. No API nodes. All files listed above. Idempotent; `--revert` deletes it.

### 2. Generate the spectrogram video (~10–30 seconds)

```bash
uv run --group analysis python scripts/spectrogram_to_reference.py \
    --audio /path/to/your/song.wav \
    --duration 5.0 \
    --emit-video
```

Output appears at `data/spectrogram_runs/<timestamp>/`:
- `spectrogram.mp4` — 121 frames at 25 fps, 832×448, near-lossless x264. **This is what you feed into ComfyUI.**
- `frame_XXXXX.png` — raw PNG sequence (ignore; same data as the mp4).
- `metadata.json` — all render params.
- `README.txt` — echoes wiring steps.

Default mode is `blurred` (Gaussian σ=1.5, natural-image contrast range) — the safest first test. Phase 2.1 will sweep other modes; for this tutorial, use the default.

### 3. Load the workflow + point LoadVideo at the mp4 (~30 seconds)

In ComfyUI:
1. Load `example_workflows/experimental/spectrogram_iclora_minimal.json`.
2. Find the `LoadVideo` node (titled "Spectrogram mp4 (REPLACE widget)"). Its widget says `REPLACE_WITH_SPECTROGRAM.mp4` — click and point it at the absolute path of the `spectrogram.mp4` from step 2.
3. (Optional) Edit the positive prompt. The default is a reasonable start:
   > "A drummer performing energetically on a dimly lit stage, warm stage lighting, shallow depth of field, cinematic. The performer's motion pulses with the music, confident and rhythmic."

### 4. Render the test pass (~60–90s on a 4090)

Queue. The workflow:
1. Loads models (first queue caches them; subsequent queues are fast)
2. Encodes the spectrogram mp4 → IMAGE batch via `GetVideoComponents`
3. IC-LoRA guide injects the sequence as structural reference
4. Samples 8 distilled steps → decodes → outputs silent mp4

### 5. Render the baseline for A/B (~60–90s)

Same workflow. Either:
- Set `LTXICLoRALoaderModelOnly.strength_model` to `0.0` (loader becomes pass-through, LoRA disengages), OR
- Set `LTXAddVideoICLoRAGuide.mode` to `4` via the node's right-click menu (bypassed — outputs pass upstream conditioning + latent unchanged)

Either produces the "no IC-LoRA, same prompt + seed" control. Queue, wait.

### 6. Dub the original audio (ffmpeg one-liner)

Your ComfyUI output lands under `<comfyui_output>/spectrogram_iclora_test_NNNNN.mp4` (silent — we intentionally didn't render audio for clarity).

```bash
ffmpeg -y \
    -i <comfyui_output>/spectrogram_iclora_test_00001.mp4 \
    -i /path/to/your/song.wav \
    -c:v copy -c:a aac -shortest \
    /tmp/spectrogram_run.mp4

ffmpeg -y \
    -i <comfyui_output>/spectrogram_iclora_test_00002.mp4 \
    -i /path/to/your/song.wav \
    -c:v copy -c:a aac -shortest \
    /tmp/baseline_run.mp4
```

Open both in a video player. Compare.

---

## Interpreting the result

| Observation (visual A/B) | Verdict | Next |
|---|---|---|
| Spectrogram version shows visibly more rhythm-aligned motion than baseline; both look reasonable | **YES — beat signal survives** | Phase 2.1: sweep `--mode` and `--blur-sigma`, find strongest config |
| Both look nearly identical; IC-LoRA has no visible influence | **Weak / no signal** | Re-run spectrogram with `--mode edge_detected` (closer to canny distribution). If still flat after that, retire Phase 2 |
| Spectrogram version shows dense horizontal bands / color tears / heatmap artifacts | **OOD break — model rejects reference** | Try `--mode blurred --blur-sigma 3.0` (heavier smoothing). If still ugly, retire Phase 2 with documented failure |
| Spectrogram version moves rhythmically but identity drifts (subject changes) | **Reference competes with prompt** | Re-run workflow with LoRA strength=0.5 or 0.7 (the loader widget). Retry A/B |

This is qualitative. Phase 2.2 will add an objective `beat_sync_score` metric.

---

## Troubleshooting

**"Failed to extract reference_downscale_factor from metadata"** in the ComfyUI log when the IC-LoRA loader runs: the MergeGreen LoRA's safetensors header may not declare this field. The loader falls back to `1.0`, which typically works for this LoRA. Not an error — just an information message. If generation looks degenerate, check `metadata_format` in the `.json` sidecar file next to the .safetensors.

**OOM at 832×448**: drop to `768×416` (both still div-32) via the `EmptyLTXVLatentVideo` widgets. Or drop length to `105` (`(105-1)%8 == 0`) for a 4.2-second clip.

**"LoadVideo failed"**: path problem or codec. The script emits h264-mp4 yuv420p — widely readable. If ComfyUI's LoadVideo can't read it, try `ffprobe <comfyui_output>/spectrogram.mp4` and compare to a known-working mp4 in your workflow.

**"Node `AudioLoopHelperSageAttention` not found"**: this is our custom node. Ensure `ComfyUI-AudioLoopHelper` is installed (you're in its repo, so this should already be true).

**The spectrogram mp4 looks like a static image**: your audio is too quiet or too steady. Boost volume or pick a more dynamic clip.

**You get a render but the "rhythm-aligned motion" is hard to tell from random camera shake**: compare 4-5 A/B pairs before concluding. Cherry-pick drum hits in the song and see if the video has visible energy at those timestamps.

---

## What's next

If the test works (qualitative yes on beat-sync):
- Phase 2.1: sweep render modes. Same workflow, regenerate the spectrogram with different `--mode` / `--blur-sigma` / `--window-seconds`.
- Phase 2.2: build `scripts/measure_beat_sync.py` for quantitative scoring.
- Phase 2.3: integrate into the full audio-loop architecture (`example_workflows/audio-loop-music-video_latent_iclora.json` gets built).
- Phase 2.4: pair with `AudioPitchDetect.vocal_fraction` for instrumental-vs-vocal-gated reference blending.

If the test doesn't work:
- Document the failure mode (heatmap? no influence? identity drift?) in `internal/design/spectrogram_reference_design.md` §8 kill switches.
- Phase 2 retires. Remaining Phase 2-ish work (`measure_beat_sync.py`, runtime node) is dropped.

---

## Related reading

- `internal/design/spectrogram_reference_design.md` — full architecture + 5-phase ladder + kill switches + decision log. The "why" behind all choices here.
- `internal/ic_lora_assessment.md §6.5 D14-D18` — topic-searchable decisions index for this track (blurred default, offline-first, global normalization, Pillow-in-analysis-not-runtime).
- `docs/reference/nag_technical_reference.md` — adjacent ("NAG also uses object_patches"). Relevant because IC-LoRA's `guide_attention_entries` has the same offload-asymmetry risk surface; verify via `scripts/verify_sage_iteration_trace.sh` on any longer-form integration test.
- `<comfyui_custom_nodes>/ComfyUI-LTXVideo/iclora.py` — source of `LTXAddVideoICLoRAGuide`, `LTXICLoRALoaderModelOnly`. Read if the runtime behavior surprises you.

### What is ComfyUI-LTXVideo?

Lightricks' official ComfyUI integration for LTX-2 / LTX-2.3. Provides the runtime wrappers around LTX's inference primitives: model/VAE/text-encoder loaders, the AV-joint latent concat/separate nodes, guide nodes (`LTXVAddLatentGuide`, `LTXAddVideoICLoRAGuide`), IC-LoRA loader, STG and APG guiders, sparse-track editor, tiled VAE decoder. Every `LTXV*` / `LTXAV*` / `LTXAddVideo*` / `LTXICLoRA*` node in any workflow comes from there. Treat it as read-only reference — our runtime nodes in `nodes.py` / `nodes_analysis.py` / `nodes_sage.py` are designed to compose with it, not modify it.
