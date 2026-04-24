Last updated: 2026-04-24

# Spectrogram-as-reference IC-LoRA — experimental test rig

Experimental. Tests whether feeding a Mel spectrogram of a song as the IMAGE reference to `LTXAddVideoICLoRAGuide` drives generated video motion that tracks the song's rhythm — and optionally whether LTX 2.3 can GENERATE audio from that visual encoding (V2A round-trip).

Architecture + long-term roadmap: `internal/design/spectrogram_reference_design.md`.

---

## What this workflow does

Forks the production `example_workflows/audio-loop-music-video_latent.json` initial-render path, strips the TensorLoop + subgraph + audio-freeze machinery, and inserts:

- `LTXICLoRALoaderModelOnly` between `LTX2SamplingPreviewOverride` and the MODEL-Set node
- `LTXAddVideoICLoRAGuide` on the initial-render conditioning + latent path, with the spectrogram mp4 as the IMAGE input
- `LTXVEmptyLatentAudio` feeding the AV concat (no audio input — sampler generates the audio)
- `LTXVAudioVAEDecode` on the separated audio latent → pipes into `VHS_VideoCombine.audio`

The output mp4 has **generated video + generated audio** baked in. You compare the generated audio against the original song to see what the spectrogram-driven generation "thinks" the audio should sound like.

Keeps the full production patch chain (sage → chunk-FF → tuner → NAG → preview-override → ModelSamplingSD3 shift=13), authoritative distilled sigmas (linear_quadratic 8 1), distilled I2V init via `LTXVImgToVideoInplaceKJ(reference_image.png)`. An earlier scratch-built "minimal" topology (fewer nodes, no patch chain) produced chroma noise in testing — the full patch chain is load-bearing for distilled LTX 2.3.

Post-build DAG verified via `scripts/analyze_workflow_dag.py`.

---

## Prerequisites

### Models + LoRAs

Paths use `<comfyui_models>` placeholder for your ComfyUI models directory.

| File | Path | Purpose |
|---|---|---|
| `ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` | `<comfyui_models>/diffusion_models/` | Merged distilled MODEL |
| `gemma_3_12B_it_fpmixed.safetensors` | `<comfyui_models>/text_encoders/` | Gemma 3 text encoder |
| `ltx-2.3_text_projection_bf16.safetensors` | `<comfyui_models>/text_encoders/` | LTX 2.3 text projection |
| `LTX23_video_vae_bf16.safetensors` | `<comfyui_models>/vae/` | Video VAE |
| `LTX23_audio_vae_bf16.safetensors` | `<comfyui_models>/vae/` | Audio VAE |
| Union Control IC-LoRA safetensors | `<comfyui_models>/loras/` | IC-LoRA adapter |

### IC-LoRA options

| LoRA | Source | Notes |
|---|---|---|
| `MergeGreen_IC-lora_ltx2.3.safetensors` | HF: `MergeGreen/LTX-2.3-IC-LoRA` | Community Union Control variant; used by default in the apply script |
| `ltx-2.3-22b-ic-lora-union-control.safetensors` | HF: `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` | First-party Lightricks Union Control; trained on `Lightricks/Canny-Control-Dataset` |
| `ltx-2.3-22b-ic-lora-motion-track-control.safetensors` | HF: `Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control` | Motion-Track variant; less likely to match spectrogram edge structure than Union Control |

Edit the `ICLORA_FILE` constant at the top of `scripts/apply_spectrogram_iclora_minimal.py` and re-run to swap.

### Custom nodes

- `ComfyUI-LTXVideo` (Lightricks official) — required
- `ComfyUI-KJNodes` — required (sage attention, Get/Set, etc.)
- `ComfyUI-VideoHelperSuite` (VHS) — required (LoadVideo, GetVideoComponents, VHS_VideoCombine)
- `ComfyUI-NativeLooping_testing` — required (production workflow has these nodes even though we strip the loop)
- `ComfyUI-LTXAVTools` — **optional but recommended** for `LTXFrameCalculator` + `LTXVAddAudioLatentGuide` (see §Extensions below)
- `ComfyUI-AudioLoopHelper` — you're in it

### Tools

- `ffmpeg` on PATH (spectrogram → mp4)
- `uv` for Python scripts

### Audio

A short, **drum-forward** clip. Avoid ambient pads, sustained tones, vocals-only — the spectrogram test needs visible rhythmic variance. 5–20 seconds is plenty.

---

## Step-by-step

### 1. Generate the spectrogram mp4

```bash
uv run --group analysis python scripts/spectrogram_to_reference.py \
    --audio /path/to/song.wav --start 0 --duration 20 \
    --mode edge_detected --emit-video
```

- `--mode edge_detected` produces sharp vertical beat transitions (closest to canny-trained IC-LoRA distribution)
- Use `--mode normalized` (middle-ground VAE-friendly contrast) or `blurred` (softest) as alternates for mode sweeps
- **Match the duration to your intended render length.** The default production render is ~20 seconds; if your spectrogram is only 5s, the remaining 15s of render has no IC-LoRA reference and will drift into pure text-driven output. Longer is fine — the guide just won't exceed the working latent's frame count.

Output lands in `data/spectrogram_runs/<timestamp>_<mode>/`:
- `spectrogram.mp4` — stitched PNG sequence via ffmpeg, near-lossless x264
- `metadata.json` — all render params
- `frame_*.png` — raw PNG sequence

Copy the mp4 to ComfyUI's input directory (`<comfyui_input>/`) so `LoadVideo` can find it.

### 2. Build the workflow

```bash
uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py
# → example_workflows/experimental/spectrogram_iclora_minimal.json
```

Forks production, strips loop, inserts IC-LoRA, switches audio to generated. ~78 nodes, DAG-verified. Idempotent; `--revert` deletes.

### 3. Open workflow + queue

1. Open `example_workflows/experimental/spectrogram_iclora_minimal.json` in ComfyUI.
2. Click the `LoadVideo` widget, set filename to your `spectrogram.mp4`.
3. **Keep prompts simple.** Inherited from production — probably already right. Lesson from music-video work: descriptive prompts break things; let init image + IC-LoRA reference drive the output. Avoid long adjective chains.
4. Optional: edit `EmptyLTXVLatentVideo.length` (via `PrimitiveNode(526)`) if you want a shorter/longer render. Valid values satisfy `(length - 1) % 8 == 0` (e.g. 121, 129, 249, 497). If you install `ComfyUI-LTXAVTools`, the `LTXFrameCalculator` node computes this for you from a target seconds + fps.
5. Queue.

### 4. Dub + compare (optional, if you want A/B against frozen-audio baseline)

The output mp4 already has generated audio. For comparison with the original song:

```bash
ffmpeg -y \
    -i <comfyui_output>/LTX-2_00001.mp4 \
    -i /path/to/song.wav \
    -c:v copy -c:a aac -shortest \
    /tmp/original_audio_dub.mp4
```

Then play both side-by-side.

---

## Variants worth running

### With vs without sage attention

Set `AudioLoopHelperSageAttention(268).mode` to `4` (bypassed) via right-click → Bypass, queue again, compare. Tells us whether sage's attention routing affects the IC-LoRA × spectrogram interaction. Per CLAUDE.md, `auto_mask_aware` is the production default and is preserved here; ablating it isolates its contribution.

### Different IC-LoRA files

Download Lightricks' first-party Union Control, edit `ICLORA_FILE` in `apply_spectrogram_iclora_minimal.py`, regenerate the workflow. Clean A/B against the community MergeGreen variant using otherwise identical inputs.

### Different render modes (Phase 2.1 sweep)

Re-run `spectrogram_to_reference.py` with `--mode raw` / `--mode normalized` / `--mode blurred` / `--mode edge_detected`. Each produces a distinctly-named output dir. Queue each and compare. Design doc's §Phase 2.1 covers what each mode tests.

### Frozen-audio variant (if you want to isolate IC-LoRA's visual effect)

Currently the apply script always switches to generated audio. For a frozen-audio variant (song stays, only video changes), the surgery would be: stop stripping `LTXVAudioVAEEncode(566)` + `SetLatentNoiseMask(570)`, don't add the `LTXVEmptyLatentAudio` / `LTXVAudioVAEDecode` / VHS-audio-rewire. Easy to fork the apply script. Left as a follow-up.

---

## Interpreting the result

| Observation | Verdict | Next |
|---|---|---|
| Video shows coherent subject with motion visibly tracking the beat; generated audio resembles rocks.wav in rhythmic structure even if not melodically | **Strong signal** — spectrogram encoding round-trips both visually and audibly | Quantify via Phase 2.2 `beat_sync_score`; promote from experimental |
| Video coherent with some rhythmic motion; generated audio is ambient / noise-like | **Partial signal** — visual path works, audio path is harder | Expected — LTX's audio head isn't trained for pure A2V without audio conditioning. Visual result is the win. |
| Video shows subject but no beat-lock; generated audio is random | **No signal** — IC-LoRA isn't picking up spectrogram structure | Try `--mode edge_detected`, raise IC-LoRA strength to 1.0, swap to Lightricks first-party LoRA |
| Output is noise / heatmap artifacts | **OOD break** — model rejects the reference | Check DAG with `analyze_workflow_dag.py`; try `--mode blurred --blur-sigma 3.0`; if still bad, retire Phase 2 per design doc §8 |

---

## Troubleshooting

**"Output is pure static / chroma noise."** Production's patch chain (sage → chunk-FF → tuner → NAG → preview-override → ModelSamplingSD3) is load-bearing for distilled LTX 2.3. If you've edited the workflow and stripped any of those nodes, re-generate from the apply script. Run `uv run --group dev python scripts/analyze_workflow_dag.py example_workflows/experimental/spectrogram_iclora_minimal.json --format ascii | tail -30` and verify every link into the sampler is connected.

**"Output is 0 seconds / empty mp4."** A node downstream of the sampler has a dangling input. Common cause: in past debugging a `LatentConcat(1605)` "Prepend Initial Render" had a dangling second input once the loop was stripped. The apply script strips it and rewires `LTXVCropGuides → LTXVTiledVAEDecode(1604)` directly; if you see this, re-run the apply script.

**"DAG analysis shows dead nodes."** Most are harmless Set/Get/Reroute orphans from the loop strip — they don't execute. The critical DEAD tell is when a VHS_VideoCombine or TiledVAEDecode output has no consumers. If those are dead, the video path is broken.

**"Audio is silent / inaudible."** Not a bug — LTX 2.3's audio head wasn't designed to generate audio from empty latent + text alone. It may produce silence, hiss, or partial sound. If you want audible audio, wire the real song audio back in (frozen-audio variant) or explore `LTXVAddAudioLatentGuide` from `ComfyUI-LTXAVTools` for true A2V conditioning.

**"Sampler takes forever / OOMs."** Check `EmptyLTXVLatentVideo` widget — production default is 497 frames (~20s at 25fps), requires ~18 GB VRAM. Drop to 121 or 249 to reduce. Also verify `LTXVPreprocess.img_compression >= 18` (CLAUDE.md gotcha).

---

## Extensions (ComfyUI-LTXAVTools)

If you install `ckinpdx/ComfyUI-LTXAVTools`, several nodes become available that improve this workflow:

- **`LTXFrameCalculator`** — snaps a target `seconds × fps` to the nearest valid `(N-1)%8==0` frame count. Replaces the hardcoded `PrimitiveNode(526)` length widget with a proper calculation. Returns `frame_count`, `latent_frames`, `actual_seconds`, plus `clean_*` variants for contamination-buffer handling.
- **`LTXDimensionCalculator`** — aspect-ratio-aware picker for div-by-64 resolutions; dynamic dropdown updates when ratio/orientation changes. Cleaner than our hardcoded 832×448.
- **`LTXVAddAudioLatentGuide`** — injects a raw audio latent as reference conditioning at NEGATIVE temporal RoPE positions (before t=0). Different mechanism than our `noise_mask=0` freeze approach; the audio conditions generation but doesn't appear in the output latent. Could be the cleaner way to do "song drives video without appearing in the AV NestedTensor."
- **`LTXAudioLatentTrim`** / **`LTXAudioLatentPad`** — direct 4D audio latent slicing `[B, C, T, F]`. Relevant if we build custom per-iteration audio windows for the loop architecture.
- **`LTXVAVLoopingSampler`** — native AV tiling sampler. CLAUDE.md currently states `LTXVLoopingSampler` can't do AV (video-only); this node claims to do AV via temporal+spatial tiling. Potential alternative to our TensorLoop architecture — worth evaluating if it produces coherent output on long-form music videos.
- **`LTXDetailSigmas`** — parametric distilled sigma schedule. Not recommended for production (our canonical `linear_quadratic 8 1` produces the authoritative distilled sigmas per `coderef/LTX-2/.../constants.py:16`). Useful for Phase 2.1 sigma-variation experiments.

Use `LTXFrameCalculator` + `LTXDimensionCalculator` for interactive UX in the workflow; `LTXVAddAudioLatentGuide` + `LTXVAVLoopingSampler` are candidates for architectural follow-ups in the broader project.

---

## Related reading

- `internal/design/spectrogram_reference_design.md` — full architecture + 5-phase iteration ladder + kill switches + decision log.
- `internal/ic_lora_assessment.md §6.5 D14–D18` — decisions index for the spectrogram track.
- `docs/reference/nag_technical_reference.md` — adjacent. `LTXAddVideoICLoRAGuide` uses `guide_attention_entries` which has similar offload-asymmetry concerns to NAG's `object_patches` (verify via `scripts/verify_sage_iteration_trace.sh` on any longer-form integration test).
- `scripts/analyze_workflow_dag.py` — static DAG analyzer. Run it on your edited workflow before queuing if you suspect a wiring bug.

### What is ComfyUI-LTXVideo?

Lightricks' official ComfyUI integration for LTX-2 / LTX-2.3. Provides the runtime wrappers around LTX's inference primitives. Every `LTXV*` / `LTXAV*` / `LTXAddVideo*` / `LTXICLoRA*` node in any workflow comes from there. Read-only reference.

### What is ComfyUI-LTXAVTools?

Community-built tools for LTX 2 AV workflows. Frame/dimension calculators, audio-latent trim/pad, A2V guide, AV looping sampler. Complementary to `ComfyUI-LTXVideo`; fills gaps the official nodes don't cover. Recommended.
