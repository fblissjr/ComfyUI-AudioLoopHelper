Last updated: 2026-05-15

# LTX 2.3 maximum video length per single sampler call

Research note answering: what caps frames-per-call on the 22B distilled
variant, and why the audio-loop workflow exists. Numbers are cited to
source files / line ranges in `coderef/` or to upstream paper / repo URLs.

## TL;DR

- **No hard transformer ceiling.** LTX 2.3 uses 3D RoPE with
  `positional_embedding_max_pos = [20, 2048, 2048]` (time, h, w). `max_pos`
  is a *normalization divisor* for RoPE fractional positions, not a hard
  index bound — you can exceed it, the model just extrapolates and quality
  degrades. Source: `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/rope.py:114-123`.
- **Practical empirical ceiling = the latent-volume budget.**
  `(W/32) * (H/32) * ((F-1)/8 + 1) ≲ 20,000–24,570` before artifacts /
  banding / grid patterns dominate. This is the *quality* cliff, derived
  empirically and codified at `nodes.py::LTXFramePlanner` and
  `docs/reference/ltx23_model_reference.md:46-51`.
- **Practical VRAM ceiling on 24GB.** At 832x448, 473 frames (~19.7s @
  24fps; canonical) ≈ 18 GB peak with sage attention + single-tile tiled VAE decode.
  Source: `docs/experimental/spectrogram_iclora_tutorial.md:168`,
  `docs/guides/upscale_guide.md:118`.
- **Lightricks ship 121 frames at 960x544** in every reference 2.3
  workflow (5.04s @ 24fps, the training-distribution framerate). LTX-Desktop UI caps user-
  visible duration at 20s @ 1080p/24fps, 10s @ 1440p/2160p, 10s @ 48/50fps.
- **Audio VAE has no independent length cap.** It runs at 25 latents/sec
  (`16000 / 160 / 4`) and `LTXVConcatAVLatent` packs it alongside video
  latents in a NestedTensor; the joint sequence length is what eats VRAM.
- **`(L-1) % 8 == 0` (a.k.a. "8k+1" frames)** is the video VAE temporal-
  compression formula: `latent = (pixel-1) // 8 + 1` because the encoder
  emits 1 latent frame for the first pixel frame, then 1 per 8 pixel
  frames. `SpatioTemporalScaleFactors.default = (time=8, width=32, height=32)`.

The audio-loop-music-video workflow exists because (a) the latent-volume
budget caps a single sampler call at roughly 20s @ 832x448 / 5s @ 1080p,
and (b) songs are 2-4 minutes. Looping the sampler with frozen audio
context extends video duration past the per-call ceiling without ever
exceeding the artifact regime.

## 1. Hard model ceiling (positional embeddings)

LTX 2.3 has no fixed positional-embedding ceiling that errors out. Its
RoPE uses *fractional positions* normalized by `max_pos`:

```python
# coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/rope.py:114-123
def get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (...)
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )
    return fractional_positions
```

Defaults from `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model.py:79`
and `model_configurator.py:55`:

| Stream | `positional_embedding_max_pos` | Order |
|---|---|---|
| Video | `[20, 2048, 2048]` | `[time_latent, height_latent, width_latent]` |
| Audio | `[20]` | `[time_latent]` |
| Cross-AV PE | `max(video[0], audio[0]) = 20` | shared time-axis |

The same constant is the default in `rope.py:190`. Interpretation:

- The "20" on the time axis is the divisor RoPE uses to convert raw
  latent-frame index into a fractional position in `[-1, 1]`. Past index
  20, fractional position grows beyond 1.0 and RoPE has to extrapolate.
- Lightricks's distilled training data sits at 121 pixel frames = 16 latent
  frames @ 24fps (= ratio of ~0.8 against the divisor 20), so the model
  has not seen positions ≫ 20.
- The `[20]` audio cap matches: `19.88s * 25 latents/s = 497 audio latents`,
  which already pushes audio fractional positions to ~24.85 — well past
  20. The model still runs, but the further out you go, the more the
  signal looks like out-of-distribution RoPE territory.

**There is no `assert idx < N` in the model that would refuse a long
sequence.** The ceiling is empirical (quality degrades), not enforced.

## 2. Practical VRAM ceiling

Source: `docs/reference/ltx23_model_reference.md:44-58`,
`docs/experimental/spectrogram_iclora_tutorial.md:168`,
`docs/guides/upscale_guide.md:118`.

### Latent-volume budget (quality cliff, not OOM)

```
latent_volume = (W / 32) * (H / 32) * ((F - 1) / 8 + 1)
```

Classification (codified in `nodes.py::LTXFramePlanner`):

| Status | Range | Behavior |
|---|---|---|
| `OK` | ≤ 20,000 | Artifact-free regime |
| `NEAR_EDGE` | 20,001 – 24,570 | Quality degrades near the cliff |
| `OVER_EDGE` | > 24,570 | Banding, grid patterns, color loss likely |

### Audio-loop default (832 x 448, 24fps)

- 832 / 32 * 448 / 32 = 26 * 14 = 364 spatial patches per latent frame.
- For F=497 (default): 364 * 63 = **22,932** → `NEAR_EDGE`. The shipped
  workflow runs right at the artifact cliff already.
- F=249 (10s): 364 * 32 = 11,648 → `OK`.
- F=121 (~5s, Lightricks default): 364 * 16 = 5,824 → `OK`.

**Measured peak VRAM**: `~18 GB` on 24GB cards at 832x448 / 497 frames
with sage attention + FP8 LoRAs + single-tile VAE decode
(`[1,1,1,true,"auto","auto"]`). Source:
`docs/experimental/spectrogram_iclora_tutorial.md:168` and
`docs/reference/sampler_reference.md:30`.

### 1024 x 576

- 1024 / 32 * 576 / 32 = 32 * 18 = 576 spatial patches per latent frame.
- F=121 (5s): 576 * 16 = 9,216 → `OK`.
- F=193 (~8s): 576 * 25 = 14,400 → `OK`.
- F=249 (~10s): 576 * 32 = 18,432 → `OK`.
- F=313 (~12.5s): 576 * 40 = 23,040 → `NEAR_EDGE`.
- F=393 (~15.7s): 576 * 50 = 28,800 → `OVER_EDGE`.
- Practical 1024x576 ceiling on 24GB: **unknown — needs measurement.**
  Latent-volume budget says ~12.5s max before artifacts; VRAM ceiling
  likely lower (no shipped workflow runs this resolution in this repo).

### 1280 x 720

- 1280 / 32 * 720 / 32 = 40 * 22.5. Note 720 % 32 = 16, so it does NOT
  satisfy LTX's div-by-32 rule. The closest valid heights are 704 (40*22)
  or 736 (40*23). Using 1280 x 704:
- 40 * 22 = 880 spatial patches per latent frame.
- F=121 (5s): 880 * 16 = 14,080 → `OK`.
- F=193 (~8s): 880 * 25 = 22,000 → `NEAR_EDGE`.
- F=249 (~10s): 880 * 32 = 28,160 → `OVER_EDGE`.
- Practical 1280x704 ceiling: roughly **5-8s** before artifacts.
  This matches LTX-Desktop's API spec (below) which caps 1080p+ at 10s
  even with FP8 quantization. VRAM ceiling on 24GB: **unknown — needs
  measurement** in this repo; LTX-Desktop API spec implies it's
  achievable on cloud hardware.
- On 48GB cards: **unknown — needs measurement.** Not benched in this
  repo. Doubling VRAM doesn't lift the latent-volume *artifact* cliff,
  only the *OOM* cliff. So 48GB lets you push closer to 24,570 latent
  volume comfortably, but past that the model degrades regardless.

### LTX-Desktop / Lightricks API caps (canonical user-facing maxes)

From `coderef/LTX-Desktop/backend/api_model_specs.py:30-72` (the LTX-2.3
Fast API spec, which is the closest analog to "what does Lightricks
themselves believe works"):

| Resolution | fps 24/25 | fps 48/50 |
|---|---|---|
| 1080p | up to 20s | up to 10s |
| 1440p | up to 10s | up to 10s |
| 2160p | up to 10s | up to 10s |

LTX-Desktop UI duration alias is `Literal[5, 6, 8, 10, 12, 14, 16, 18, 20]`
(`coderef/LTX-Desktop/backend/api_types.py:282`), default 5
(`api_types.py:315`). 20s @ 1080p/24fps is therefore Lightricks's *upper
bound for the in-app generator*, not just a default.

The audio-loop default of 19.88s @ 832x448 (497 frames @ 25fps pre-2026-05-15; re-derives to 473 frames = 19.708s @ canonical 24fps) is right at the same
edge as the 1080p/20s cap — same latent volume neighborhood (832x448
sits between 540p and 720p, F≈497 is the max that lands in NEAR_EDGE).

## 3. Audio VAE ceiling

Source: `coderef/LTX-2/packages/ltx-core/src/ltx_core/types.py:128-145`.

```python
latents_per_second = sample_rate / hop_length / audio_latent_downsample_factor
                   = 16000 / 160 / 4
                   = 25.0
```

The audio VAE has:

- No independent length cap. `AudioLatentShape.from_duration(duration=...)`
  simply produces `round(duration * 25)` frames.
- A separate positional embedding budget (`max_pos = [20]`,
  `model.py:92`). Same fractional-position story as the video stream —
  long audio extrapolates but doesn't error.
- Distinct latent topology: `(batch, channels=8, frames, mel_bins=16)`
  (`types.py:104-110`). Lives as a separate sub-tensor inside a
  `NestedTensor` that `LTXVConcatAVLatent` packs alongside the 5D video
  latent.

**Tie-in to video frame count**: not via the audio VAE itself, but via
the joint forward pass. The transformer sees `video_tokens +
audio_tokens` in self-attention (within each modality) and cross-AV
attention. Total sequence length = `(W/32)(H/32)((F-1)/8+1) + (audio_frames)`.
For the audio-loop default: 22,932 video tokens + 497 audio tokens ≈
23,429. VRAM pressure scales with sequence length squared in attention,
so audio adds <3% overhead on top of video at the audio-loop window
size.

LTX-Desktop's audio path enforces duration alignment to video:
`max_samples = round(num_frames / fps * sample_rate)`
(`coderef/LTX-Desktop/backend/services/a2v_pipeline/distilled_a2v_pipeline.py:214`).
That's a *trim*, not a model-side cap.

## 4. The (L-1) % 8 == 0 rule

Source: `coderef/LTX-2/packages/ltx-core/src/ltx_core/types.py:30,89-95`
and `<comfyui>/comfy_extras/nodes_lt.py:36`.

The temporal VAE compresses 8 pixel frames into 1 latent frame, BUT the
first pixel frame gets its own latent frame:

```python
# types.py:30
def default(cls) -> "SpatioTemporalScaleFactors":
    return cls(time=8, width=32, height=32)

# types.py:89-95 — upscale formula
return self._replace(
    channels=3,
    frames=(self.frames - 1) * scale_factors.time + 1,  # latent -> pixel
    height=self.height * scale_factors.height,
    width=self.width * scale_factors.width,
)

# nodes_lt.py:36 — encode formula (ComfyUI's EmptyLTXVLatentVideo)
latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, ...])
```

So:

- pixel_frames `F` -> latent_frames `(F - 1) // 8 + 1`
- For round-trip identity (pixel -> latent -> pixel) you need `F = 8k + 1`
  for some integer `k` ≥ 0. Examples: 1, 9, 17, 25, 33, 41, ..., 121, ...,
  249, ..., 497.
- Non-conforming `F` is **silently floored** by `EmptyLTXVLatentVideo`
  (the `// 8` integer divides truncates). User-typed length and actual
  rendered length can disagree by up to 7 frames without warning.

Lightricks call this the "8k+1 format" in their own docs:
`coderef/LTX-2/packages/ltx-pipelines/README.md:221`:

> Source video frame count must satisfy the 8k+1 format (e.g. 97, 193)
> and resolution must be multiples of 32.

LTX-Desktop's `_compute_num_frames` snaps to it explicitly:

```python
# coderef/LTX-Desktop/backend/handlers/video_generation_handler.py:384-386
@staticmethod
def _compute_num_frames(duration: int, fps: int) -> int:
    n = ((duration * fps) // 8) * 8 + 1
    return max(n, 9)
```

`LTXFramePlanner` in this repo applies the same snap (downward) so the
user-typed `target_seconds` always lands on a valid `F`. See
`docs/reference/frame_planner_reference.md:22`.

## 5. Lightricks reference workflow defaults

Source: `coderef/ComfyUI-LTXVideo/example_workflows/2.3/*.json` (read
via the `EmptyLTXVLatentVideo` widget values).

Every shipped Lightricks 2.3 reference workflow uses **the same default**:

```
[width=960, height=544, length=121, batch_size=1]
```

That holds for all of:

- `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`
- `LTX-2.3_T2V_I2V_Two_Stage_Distilled.json`
- `LTX-2.3_ICLoRA_Motion_Track_Distilled.json`
- `LTX-2.3_ICLoRA_Union_Control_Distilled.json`
- `LTX-2.3_ICLoRA_HDR_Distilled.json`
- `LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled.json`

**Why 121 frames at 960x544?**

- `121 = 15*8 + 1` (satisfies the 8k+1 rule). At 24fps that's 5.04s — the
  same as LTX-Desktop's default `duration=5` (`api_types.py:315`).
- 960x544 is the LTX standard 540p-ish landscape format (16:9 ≈ 1.76,
  actual 1.76). Width 960 = 30*32, height 544 = 17*32.
- Latent volume: 30 * 17 * 16 = **8,160**. Comfortably `OK` (< 20,000).
- Lightricks's distilled checkpoint params (`PipelineParams.num_frames =
  121`, `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:35`)
  match: this is the value the distilled pipeline ships against.
- The two-stage HQ variant defaults to `1088 // 2 = 544` height and
  `1920 // 2 = 960` width (same 960x544) at `num_inference_steps=15`
  (`constants.py:78-98`). Stage 2 upsamples to 1088x1920.

121 frames is "what works fast on a single GPU pass, with margin." The
audio-loop project pushes 4x past that (497 frames) because (a) the music
context demands ~20s per iteration to give the loop overlap enough room,
and (b) sage attention buys back the latency at that sequence length —
see `internal/analysis/empirical_bench_findings.md` for the bench math.

## Putting it together: why we loop

The audio-loop workflow generates 2-4 minute music videos. Per the
numbers above, the model can't do that in a single sampler call:

- **Latent-volume cliff at ~20s @ 832x448** (97% of the artifact ceiling).
- **VRAM cliff at ~20s @ 832x448 on 24GB** (~18 GB measured, leaving
  little headroom for FP8 LoRAs + VAE + Gemma).
- **Lightricks's own training distribution is ~5s** (121 frames @ 24fps).
  Pushing F much past 497 takes RoPE further out-of-distribution along
  the time axis (max_pos[0] = 20; 497/8+1 = 63 latent frames, fractional
  position 63/20 = 3.15).

So the loop holds resolution fixed (832x448), holds the per-iteration
length fixed (497 frames = 19.88s @ 25fps), and chains windows together
with `noise_mask=0` overlap context to extend duration indefinitely.
Each call stays inside both the artifact regime and the VRAM regime.

## Open questions ("unknown — needs measurement")

- Peak VRAM at 1024x576 and 1280x704 on 24GB / 48GB cards. No shipped
  workflow in this repo runs these resolutions; LTX-Desktop's API spec
  implies they're feasible on cloud hardware (likely H100 80GB or A100
  80GB based on the 1440p/2160p tier).
- Real-world quality cliff vs the codified 24,570 latent-volume number.
  The audio-loop default (22,932) is `NEAR_EDGE`, and the workflow ships
  it — implying the artifact regime is gradient, not sharp.
- Whether RoPE fractional-position extrapolation past `max_pos[0]=20` on
  the time axis contributes measurably to drift across loop iterations
  vs the photoreal-cross-attention drift documented in
  `docs/reference/ltx23_model_reference.md`. Not separable without a
  controlled test that holds everything else constant.

## File references

- `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model.py:78-93` — `positional_embedding_max_pos` defaults
- `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/rope.py:114-123,190` — RoPE fractional-position math
- `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py:55,63` — config loader
- `coderef/LTX-2/packages/ltx-core/src/ltx_core/types.py:19-34,89-95,128-145` — temporal scale factors + audio latents/sec
- `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:30-78` — `PipelineParams.num_frames=121` distilled default
- `coderef/LTX-2/packages/ltx-pipelines/README.md:221` — official "8k+1" wording
- `coderef/LTX-Desktop/backend/api_model_specs.py:30-72` — API-tier resolution/duration support matrix
- `coderef/LTX-Desktop/backend/api_types.py:281-316` — UI duration cap (max 20)
- `coderef/LTX-Desktop/backend/handlers/video_generation_handler.py:384-386` — `_compute_num_frames` 8k+1 snap
- `coderef/ComfyUI-LTXVideo/example_workflows/2.3/*.json` — `EmptyLTXVLatentVideo widgets=[960, 544, 121, 1]` in every shipped 2.3 workflow
- `<comfyui>/comfy_extras/nodes_lt.py:36` — `((L-1)//8)+1` silent floor in ComfyUI's EmptyLTXVLatentVideo
- `nodes.py::LTXFramePlanner` — local latent-volume classifier
- `docs/reference/frame_planner_reference.md` — snap rules + latent-volume thresholds
- `docs/reference/ltx23_model_reference.md:44-58` — latent-volume ceiling (24,570) source
- `docs/experimental/spectrogram_iclora_tutorial.md:168` — 18 GB VRAM measurement at 497 frames
- `docs/guides/upscale_guide.md:118` — VRAM peak measurement on 24GB
- `docs/analysis/ltx23_gaps_analysis.md:267-321` — why tiled spatial sampling can't extend a single call past the budget on AV
