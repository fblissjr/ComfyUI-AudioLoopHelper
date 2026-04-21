Last updated: 2026-04-17

# Standup Comedy — Example Prompt Schedule (v3)

Third iteration after v2 revealed that decoder tiling (not prompt
structure) was the dominant source of remaining jitter. v3 does three
things:

1. **Trims prompts to minimum viable** — drops audio descriptions
   (frozen audio, model hears it, text descriptions fight the
   cross-attention signal per R6). Drops repeated mood-bundle
   boilerplate.
2. **Restores specific standup verbs** per entry (v2 user-edit had
   collapsed them to generic "is speaking and performing", losing the
   phoneme-specific cross-attention signal per R1).
3. **Fixes the small `stand-up` / `a stand-up` inconsistency** that
   broke byte-exact subject match.

Includes schedules for **both** `overlap_seconds=2` and `=3`, because
changing overlap shifts the iteration grid and timestamps must snap
differently.

## Hard thinking: decoder choice (VAEDecodeTiled vs LTX-specific alternatives)

Investigated four decoder options. **Two tiers of fix**, both
documented here — the incremental widget-fix that shipped with v3,
and the structural swap planned as Phase DR1 in the plan file.

**Previously shipped (incremental fix, stride-alignment approach)**:
Generic `VAEDecodeTiled` with widgets tuned to match iteration stride.
Fixed the 2.24s mid-video seam cadence but relied on a fragile
coordination invariant between decoder widgets and
`AudioLoopController.overlap_seconds`.

**Currently shipped (Phase DR1, structural fix)**: All example
workflows now use `LTXVTiledVAEDecode` from `ComfyUI-LTXVideo`.
Spatial-only tiling, no temporal tiling at all. The stride invariant
is gone — overlap_seconds is now decoupled from decoder configuration.
`scripts/apply_ltx_decoder.py` and `scripts/validate_workflow_decoder.py`
shipped for node-type management and drift detection.

### Why `VAEDecodeTiled` produces seams

LTX 2.3's video VAE decoder has temporal attention spans within a
tile. When you tile temporally (current defaults: `temporal_size=64`,
`temporal_overlap=8`), adjacent tiles process independently — the
decoder's internal attention state does NOT carry across tile
boundaries. The overlap region (8 pixel frames ≈ 0.32 s at 25 fps)
gets blended but the attention-state mismatch between tiles produces
subtle visible shifts at every tile boundary:

- Tile stride (pixel frames) = `temporal_size − temporal_overlap` = `56`
- At 25 fps → **seams every 2.24 s**, matching user-reported jitter at
  t=3, 5, 6, 8, and continuing through the video.

This is intrinsic to how tiled temporal VAE decode works, not a bug in
ComfyUI's implementation. The fix is to **minimize or eliminate
temporal tiling**, not improve the blend.

### Why `VAEDecodeLoopKJ` doesn't solve our problem

Inspected source at
`ComfyUI-KJNodes/nodes/nodes.py:3069-3107`. It:

- Calls `vae.decode(latents)` on the FULL input latent in one pass
  (line 3088) — **no temporal tiling**.
- For videos with `overlap_latent_frames > 0`, additionally decodes a
  cross-boundary chunk (end + start latents concatenated) and splices
  it over the original boundary frames — designed to hide the
  **end-to-start loop seam** in looping animations (Wan 2,
  HunyuanVideo 1.5).

**Use case mismatch**: it's for *cyclically-looping* output videos
(seamless infinite loops) — NOT for our audio-driven 3-min stitched
video. Its seam-hiding logic applies at the END→START wrap, not at the
mid-video iteration boundaries we care about.

Additionally, calling `vae.decode()` non-tiled on a 3-min video
(~600 latent frames × 22×22 spatial × 128 channels × bf16) overruns a
24 GB GPU during intermediate activations. Raw peak estimate: 40-55 GB.
Don't use for 3-min material on a 4090.

### What actually fixes it

The LTX video VAE decode memory is bounded by spatial × temporal ×
channels × precision. On 24 GB with the 22B distilled model + other
state loaded, the largest temporal-tile footprint that fits is
roughly `temporal_size ≈ 512 pixel frames` at 704×704 spatial.

Three points on the spectrum:

| `temporal_size` | `temporal_overlap` | Tile stride (s) | Seams behavior |
|---|---|---|---|
| 64 (current default) | 8 | 2.24 | ~10 seams per iteration; every 2-3 s uniform pulsing |
| **512** (recommended) | **64** | **17.92** (≈ iteration stride) | **0 seams mid-iteration**; decode boundaries co-locate with iteration boundaries |
| 2048 (diagnostic only) | 128 | 76.8 | 2-3 total decode seams over a 3-min video, at novel positions (don't align with iterations) — useful for confirming decode is the cause, bad for production |

**`[512, 64, 512, 64]` is the right production answer.** It:

- Eliminates mid-iteration decode seams entirely (tile stride 17.92 s
  matches iteration stride 17.92 s at overlap=2 exactly, so each
  iteration fits inside one temporal decode tile).
- Decode boundaries coincide with iteration boundaries. The model
  already has seam-like behavior at iteration boundaries (latent
  hand-off between independently-sampled iterations); decoder seams
  now **add no new seam positions**, they just reinforce existing
  ones. Net: ~10 seam positions total instead of ~80+.
- Fits in VRAM on a 4090 with the 22B distilled model loaded
  (empirical evidence needed; if it OOMs, step down to `[512, 64,
  256, 32]` — one mid-iteration seam per iteration, still 8× fewer
  than baseline).

### `LTXVTiledVAEDecode` (from `ComfyUI-LTXVideo/tiled_vae_decode.py:11`)

Reading the source: decodes with **spatial tiling only**. Each spatial
tile receives the FULL temporal dimension — `vae.decode(tile_latents)`
where `tile_latents` spans the whole time axis but only part of the
spatial axis. Temporal attention stays continuous across the entire
video; **no temporal seams possible**.

Widgets:
- `horizontal_tiles`, `vertical_tiles` (default 1,1 — no spatial
  tiling)
- `overlap` (spatial overlap, in latent frames, default 1)
- `last_frame_fix` (works around an LTX-specific final-frame issue)
- `working_device` (can set "cpu" if VRAM-tight at cost of speed)
- `working_dtype`

For a 3-min 704×704 video at 2×2 spatial tiles: each tile decodes
352×352 × 4500 pixel frames × 3 channels × fp16 ≈ 3.5 GB raw + ~10 GB
activations. Fits in 24 GB with other state loaded.

This is the right answer IF VRAM permits. **No temporal seams, no
co-location hack, just clean decode.**

### `LTXVSpatioTemporalTiledVAEDecode` (from same file, line 274)

Adds temporal tiling on top, but with LTX-specific blending that's
materially better than the generic `VAEDecodeTiled`:

- `temporal_tile_length` in **latent frames** (at LTX's time_scale=8
  → `temporal_tile_length=16` ≈ 128 pixel frames ≈ 5 s per tile at
  25 fps). Default 16.
- `temporal_overlap` in **latent frames**. Default 1.
- **Proper weighted blending**: `frame_weights = linspace(0, 1,
  overlap_frames + 2)[1:-1]` applied as a ramp across the overlap
  region — smooth cross-fade in pixel space, not the generic
  "just blend the two tiles equally" approach.
- Drops the first frame of each non-initial chunk (correct handling
  of LTX's causal temporal structure).

With `temporal_tile_length=64` (latent frames) = 512 pixel frames =
20.48 s per tile → each iteration fits in one temporal tile. This
gets you the same iteration-aligned stride as my previous `[512, 64,
512, 64]` recommendation, but with LTX-tuned blending instead of
generic. Fallback for when spatial-only doesn't fit VRAM.

### Why LTX-specific beats generic

Generic `VAEDecodeTiled` doesn't know about LTX's VAE-specific
`downscale_index_formula` (time_scale_factor=8, spatial=32). It
tiles in pixel-frame units and blends symmetrically, which misses:
- The +1 first-frame offset in LTX's temporal latent structure
- The right overlap math for LTX's causal conv layers
- Dropping the first frame of continued chunks (needed to avoid
  double-counting in the overlap region)

The LTX-specific nodes implement all this correctly. The generic node
produces visible seams every 2.24s; the LTX-spatial-only node produces
none; the LTX-spatiotemporal with large temporal_tile_length produces
seams only at iteration-aligned boundaries.

### Summary of decoder options

| Option | Temporal seams | VRAM | Verdict |
|---|---|---|---|
| `VAEDecodeTiled [512, 64, 64, 8]` (current default) | ~10 per iteration (every 2.24 s) | Low | **Broken** — don't use for 3-min material |
| `VAEDecodeTiled [512, 64, 512, 64]` | 0 mid-iteration (tile stride ≈ iteration stride) | Medium | Workable workaround |
| **`LTXVTiledVAEDecode` with `horizontal_tiles=2, vertical_tiles=2, overlap=1`** | **None** (no temporal tiling) | Medium | **Best if VRAM permits — the right answer** |
| `LTXVSpatioTemporalTiledVAEDecode` with `temporal_tile_length=64, temporal_overlap=2` | 0 mid-iteration (tile aligns with iteration) | Low | Fallback when spatial-only OOMs |
| `VAEDecodeLoopKJ` | N/A for our use case (designed for looping videos) | High (no tiling) | **Don't use** — wrong tool |

### "Maybe tiling is a bad idea" — exactly right, and we have a clean way out

For our 3-min stitched video on a 24 GB 4090, temporal tiling is not
actually necessary. `LTXVTiledVAEDecode` with spatial-only tiling
gives us memory-safe decode without temporal seams. This is a
**node swap in the workflow** (replace node 1604's type from
`VAEDecodeTiled` → `LTXVTiledVAEDecode`), not a widget value fiddle.

Phase 2 of the blend-jitter plan had an implicit assumption that
temporal tiling was unavoidable and we'd need latent-space cross-fading
to smooth seams. With `LTXVTiledVAEDecode` that assumption is wrong —
no temporal seams means no cross-fading needed. Only iteration seams
(latent hand-off between sampler calls) remain, and those are a smaller
problem.

Update the plan post-this-run: Phase 2 scope may shrink to "smooth
iteration-boundary hand-offs" only, not "smooth decoder tile seams AND
iteration hand-offs."

## Inputs

- **Audio**: preprocessed WAV from `scripts/preprocess_audio_for_ltx.py`
  or `<your-assets>/norm2_from_webm_ltx.wav`.
- **Image**: `Performing_standup_at_202604171215.png` (LIVE STANDUP
  CHICAGO neon sign, tight framing).
- **Subject string (byte-exact across all entries)**:
  `a male standup comedian in a striped sweater at a stand-up comedy club`
  — shorter than v2's verbose identity block, but keeps the minimum
  identity anchor ("striped sweater") so text conditioning adds
  constraint beyond pure i2v image reliance.
- **Tier**: `1b` (performance_live, wide-stage — but the tight image
  makes the framing-bundle irrelevant so it's not templated in).
- **Montage**: off.

## Why v3 differs from v2

| Issue in v2 | v3 fix |
|---|---|
| Audio descriptions in most entries ("Ambient club sounds", "The crowd on the right mid-laugh") duplicated info the model already hears from the frozen audio; per R6 these fight cross-attention | Dropped all audio-content description. Keep only ambient sounds NOT in track (rare) and vocal-delivery qualifiers (occasional). |
| Mood-bundle `"Wide stage framing, warm stage wash"` repeated every entry — R3 says subject stays byte-exact, but the mood-bundle is technically non-subject boilerplate that just adds tokens | Reduced to `"Warm stage wash."` |
| Verbose identity description (`"short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater"`) was detailed but long | Shorter: `"a male standup comedian in a striped sweater"` — 8 words, still identity-anchored |
| Crowd descriptions ("A few crowd members on the right mid-laugh", "The crowd silhouetted in the foreground reacting") sometimes didn't match the image | Simplified to one occasional reference like `"crowd visible on the right"` |
| User-edit v2' collapsed action verbs to generic `"is speaking and performing"` — loses R1 lip-sync signal | Restored specific standup verbs per entry from R1 pool |
| `stand-up` vs `a stand-up` inconsistency across entries | Consistent `"at a stand-up comedy club"` everywhere |

v3 lines run ~40-60 words vs v2's ~70-90. Cleaner, faster to read,
less token budget spent on repeat boilerplate.

## Schedule — overlap = 2 (stride = 17.92)

Grid boundaries at `0:00, 0:17, 0:35, 0:53, 1:11, 1:29, 1:47, 2:05,
2:23, 2:41, 2:58+`. Matches current workflow default.

```
node_169_prompt: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is pausing for the laugh. Warm stage wash.

schedule:
0:00-0:17: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is pausing for the laugh. Warm stage wash.
0:17-0:35: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the setup. Warm stage wash. Delivery in a dry deadpan.
0:35-0:53: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is telling a joke. Warm stage wash. Brisk rhythmic delivery.
0:53-1:11: Style: cinematic. In a medium close-up, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the punchline. Warm stage wash. The crowd reacting.
1:11-1:29: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is mid-bit. Warm stage wash. Delivering in a low rhythmic tone.
1:29-1:47: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is smiling wryly, looking out into the audience. Warm stage wash.
1:47-2:05: Style: cinematic. In a medium close-up, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is telling a joke, gesturing sharply with his left hand. Warm stage wash.
2:05-2:23: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the punchline, leaning back slightly, pointing at a crowd member on the right. Warm stage wash.
2:23-2:41: Style: cinematic. In a medium close-up, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is leaning into the mic, building the final premise. Warm stage wash. Intense, rapid delivery.
2:41-2:58: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the final punchline, smiling wide. Warm stage wash. The crowd fully laughing.
2:58+: Style: cinematic. In a wide shot, slow dolly out, a male standup comedian in a striped sweater at a stand-up comedy club is reacting to the crowd, stepping back from the mic stand. Warm stage wash. Room tone settling.
```

## Schedule — overlap = 2 + node 567 trim = 5s (trimmed audio, stride 17.92)

With `node 567 TrimAudioDuration.start=5`, video-t=0 maps to routine-t=5.
Entry verbs need to shift to match what the audio actually plays at each
video-time window. Also the total trimmed duration is 179s instead of 184s,
so the final entry's boundary lands ~1s later.

**When to use this variant**: if your audio source has problematic
content in the first 5 seconds (cold-open applause that's too loud,
mic-clipping burst, room noise before the comedian starts) and you
want to skip it for cleaner generation starting at "comedian speaking"
rather than "crowd applauding."

Iteration grid unchanged (stride still 17.92) — the trim only shifts
WHICH audio plays at each video timestamp, not the timing of the loop
itself.

```
node_169_prompt: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the opening line. Warm stage wash.

schedule:
0:00-0:17: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the opening line. Warm stage wash.
0:17-0:35: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the setup. Warm stage wash. Delivery in a dry deadpan.
0:35-0:53: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is telling a joke. Warm stage wash. Brisk rhythmic delivery.
0:53-1:11: Style: cinematic. In a medium close-up, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the punchline. Warm stage wash. The crowd reacting.
1:11-1:29: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is mid-bit. Warm stage wash. Delivering in a low rhythmic tone.
1:29-1:47: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is telling a joke, looking out into the audience. Warm stage wash.
1:47-2:05: Style: cinematic. In a medium close-up, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the setup, gesturing sharply with his left hand. Warm stage wash.
2:05-2:23: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the punchline, leaning back slightly. Warm stage wash.
2:23-2:41: Style: cinematic. In a medium close-up, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is leaning into the mic, building the final premise. Warm stage wash. Intense, rapid delivery.
2:41-2:59: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the final punchline, smiling wide. Warm stage wash. The crowd fully laughing.
2:59+: Style: cinematic. In a wide shot, slow dolly out, a male standup comedian in a striped sweater at a stand-up comedy club is reacting to the crowd, stepping back from the mic stand. Warm stage wash. Room tone settling.
```

Key verb shifts from the untrimmed variant:
- **Entry 1**: `"is pausing for the laugh"` → `"is delivering the opening line"`. The cold-open applause is mostly gone at video-t=0; the comedian is already into speech.
- **Entry 6**: `"is smiling wryly, looking out into the audience"` → `"is telling a joke, looking out into the audience"`. The trimmed content puts this window in active speech, not a pause.
- **Final entry**: `2:58+` → `2:59+`. Total trimmed audio is 179s, 1s later than untrimmed.

## Schedule — overlap = 3 (stride = 16.96)

Different grid. Boundaries at `0:00, 0:16, 0:33, 0:50, 1:07, 1:24,
1:41, 1:58, 2:15, 2:31, 2:48+`. Use this schedule if you bumped
`overlap_seconds` to 3.0 per the iteration-seam fix guidance.

```
node_169_prompt: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is pausing for the laugh. Warm stage wash.

schedule:
0:00-0:16: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is pausing for the laugh. Warm stage wash.
0:16-0:33: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the setup. Warm stage wash. Delivery in a dry deadpan.
0:33-0:50: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is telling a joke. Warm stage wash. Brisk rhythmic delivery.
0:50-1:07: Style: cinematic. In a medium close-up, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the punchline. Warm stage wash. The crowd reacting.
1:07-1:24: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is mid-bit. Warm stage wash. Delivering in a low rhythmic tone.
1:24-1:41: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is smiling wryly, looking out into the audience. Warm stage wash.
1:41-1:58: Style: cinematic. In a medium close-up, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is telling a joke, gesturing sharply with his left hand. Warm stage wash.
1:58-2:15: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the punchline, leaning back slightly, pointing at a crowd member on the right. Warm stage wash.
2:15-2:31: Style: cinematic. In a medium close-up, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is leaning into the mic, building the final premise. Warm stage wash. Intense, rapid delivery.
2:31-2:48: Style: cinematic. In a medium shot, static camera, a male standup comedian in a striped sweater at a stand-up comedy club is delivering the final punchline, smiling wide. Warm stage wash. The crowd fully laughing.
2:48+: Style: cinematic. In a wide shot, slow dolly out, a male standup comedian in a striped sweater at a stand-up comedy club is reacting to the crowd, stepping back from the mic stand. Warm stage wash. Room tone settling.
```

## Workflow widget values (v3, post-DR1)

**The decoder is now `LTXVTiledVAEDecode` across all 4 example
workflows.** Shipped via `scripts/apply_ltx_decoder.py`. Everything
else stays from v2 (euler sampler, sage attention, etc.).

### Current decoder (LTXVTiledVAEDecode)

Node 1604 and node 1597 (all 4 workflows) are now `LTXVTiledVAEDecode`.
Widget values:

| Widget | Value |
|---|---|
| `horizontal_tiles` | 2 |
| `vertical_tiles` | 2 |
| `overlap` | 1 (latent frames) |
| `last_frame_fix` | true |
| `working_device` | auto |
| `working_dtype` | auto |

No stride-alignment concern — spatial-only tiling means no temporal
tile boundaries exist.

### Historical: VAEDecodeTiled widget fix (pre-DR1, fallback option)

Before DR1, these nodes were generic `VAEDecodeTiled` with widgets
changed from `[512, 64, 64, 8]` → `[512, 64, 512, 64]` to align tile
stride with iteration stride. That fix works but carries the
coordination invariant. To revert to this approach (e.g., for VRAM
constraints):

```bash
uv run python scripts/apply_ltx_decoder.py --revert
```

This restores `VAEDecodeTiled` with widgets `[512, 64, 512, 64]`
(tile stride 17.92s, exactly matches `overlap_seconds=2`'s iteration
stride of 17.92s). If you change `overlap_seconds` after this, you
must recompute widgets per the table in `docs/debugging_guide.md`
or risk re-introducing mid-video seams.

**Do NOT use**: `VAEDecodeLoopKJ` (wrong tool — designed for
seamlessly-looping short videos, does full non-tiled decode that
OOMs on 3-min material).

### Other widgets (unchanged from v2 unless noted)

| Widget | Node | Value | Change from v2? |
|---|---|---|---|
| decoder node type | **1604, 1597** | **`LTXVTiledVAEDecode`** | **YES — DR1 swap shipped** |
| `overlap_seconds` | AudioLoopController | `2.0` (use matching schedule) | Empirically confirmed the right value |
| `sampler_name` | 154 KSamplerSelect | `euler` | Unchanged |
| `shift` | 1513 ModelSamplingSD3 | `13` | Unchanged |
| Scheduler | 1421 BasicScheduler | `linear_quadratic, 8, 1` | Unchanged |
| CFG | 153 CFGGuider | `1.0` | Unchanged |
| NAG scale/tau/alpha/enabled | 508 LTX2_NAG | `11, 0.25, 2.5, true` | Unchanged |
| Sage attention | 268 PathchSageAttentionKJ | `sageattn_qk_int8_pv_fp16_triton` or `sageattn_qk_int8_pv_fp8_cuda` | User preference |
| `window_seconds` | 688 FloatConstant | `19.88` | Unchanged |
| `blend_seconds` | 1558 TimestampPromptSchedule | `0.0` | Unchanged |
| `snap_boundaries` | 1558 TimestampPromptSchedule | `true` | Unchanged |
| `start` (outer trim) | 567 TrimAudioDuration | `0` | Unchanged (pre-trimmed audio) |

### If LTXVTiledVAEDecode OOMs on your GPU

Shouldn't on a 24 GB 4090 with default `horizontal_tiles=2,
vertical_tiles=2`. If it does, step up tile counts (smaller tiles,
more of them): `3×3` or `4×4`. Or set `working_device="cpu"` to
decode on CPU at cost of speed.

If LTX decoder still OOMs, fall back via:
```bash
uv run python scripts/apply_ltx_decoder.py --revert
```

That restores `VAEDecodeTiled` with `[512, 64, 512, 64]`, aligned to
`overlap_seconds=2`. Further step-down for memory:
- `[512, 64, 384, 48]` — tile stride 13.44 s, few mid-iteration seams
- `[512, 64, 256, 32]` — tile stride 8.96 s, some mid-iteration seams
- `[512, 64, 128, 16]` — tile stride 4.48 s

Each step reintroduces some mid-iteration decoder seams.

## Negative prompt

Unchanged from v1/v2 standup-tuned version. See
`internal/prompt_comedy1.md#negative-prompt`.

## Observations for future refinement

- **Widget fix is fragile — structural fix needed (Phase DR1).** The
  current tuning `[512, 64, 512, 64]` works only at `overlap_seconds=2`.
  Any change to overlap silently breaks the stride alignment. DR1 in
  the plan file specs the `scripts/apply_ltx_decoder.py` swap to
  `LTXVTiledVAEDecode`, which eliminates the coordination invariant
  entirely (no temporal tiling → no stride to align). Entry criteria
  and exit criteria are in the plan.
- **`VAEDecodeLoopKJ` investigation was a dead end for us, but
  useful context.** It's the right tool for seamlessly-looping short
  videos (Wan 2, HunyuanVideo 1.5). Different problem space from
  audio-driven stitched video. Worth noting in docs so future
  decisions don't revisit this.
- **Phase 2 scope almost certainly shrinks after this run.** The
  `LatentOverlapCrossfade` spec assumed temporal tiling seams needed
  latent-space smoothing. With `LTXVTiledVAEDecode` there are no
  decoder seams. Only iteration-hand-off seams remain — smaller
  problem, maybe doesn't need the latent crossfade at all. Revisit
  the plan after v3 ships data.
- **Verb pool is finite.** For a 3-min routine with 11 entries, the
  standup verb pool of ~10 gets close to exhausted. For longer
  routines (5+ min, 15+ entries), expand the pool or accept some
  repetition. Not a blocker for typical use.
- **v3 dropped almost all crowd description.** That was deliberate —
  the image anchors crowd presence, we don't want text competing.
  Only occasional `"The crowd reacting"` or `"The crowd fully
  laughing"` remains where there's clear audio-content-independent
  visual direction. If crowd behavior in output doesn't respond to
  audio energy as expected, we can add more crowd direction later.
- **`debugging_guide.md` reflects current state correctly** — primary
  recommendation is `[512, 64, 512, 64]` for `VAEDecodeTiled` with
  the stride-alignment invariant documented. Will be updated as part
  of Phase DR1 to promote `LTXVTiledVAEDecode` as primary, demote the
  widget fix to fallback.
