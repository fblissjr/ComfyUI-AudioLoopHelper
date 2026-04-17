Last updated: 2026-04-17

# Debugging Guide: Quality Problems in the Audio-Loop Pipeline

Problem-first troubleshooting. You see X in your output — here's how
to diagnose and fix it.

Related docs:
- `prompt_creation_guide.md` — prompt rules + widget-value guidance
- `audio_analysis_guide.md` — offline analysis + runtime audio nodes
- `system_prompt.md` — LLM system prompt for schedule generation
- `profiling_guide.md` — performance profiling (opt-in)

## How quality problems layer in this pipeline

Most "the video looks wrong" issues in the audio-loop pipeline have
one of six root causes. They layer in this order from **most
perceptible** to **most subtle** — fix top-down:

1. **Prompt-level issues** (wrong subject string, wide framing, camera
   moves within a window, identity drift from different subjects per
   entry). Dominates perception; fix first.
2. **VAEDecodeTiled temporal seams** (every ~2 s if temporal_size is
   small). Looks like periodic pulsing or fine-grained jitter.
3. **Iteration-boundary seams** (every ~stride_seconds). Looks like
   a visible "cut" or identity hand-off every ~18 s.
4. **Schedule-boundary conditioning switches** (mid-iteration mixed
   prompts if `snap_boundaries=False`). Looks like a bad "blended"
   iteration between two good ones.
5. **Audio quality** (dull / bass-heavy / crowd-contaminated voice).
   Hurts lip sync specifically.
6. **Model-intrinsic noise** (NAG modulation, sampler stochasticity,
   latent temporal chunk boundaries). Often can't be fixed from our
   side.

Each layer is masked by the ones above it. So fixing a top layer
often reveals the next one. That's not a regression — it's progress.

---

## Symptom → first check (quick lookup)

| What you see | First suspect | Jump to |
|---|---|---|
| Fine-grained pulsing / jitter every ~2-3 s (all the way through) | VAEDecodeTiled temporal tiles too small | [Decode-tile seams](#decode-tile-seams) |
| Visible cut / identity jump every ~18 s | Iteration-boundary hand-off, `overlap_seconds` too low | [Iteration-boundary seams](#iteration-boundary-seams) |
| One "weird" iteration between two normal ones | Mid-iteration mixed conditioning | [Mid-iteration mix](#mid-iteration-mix) |
| Face drifts (becomes a different person) over 30-60 s | Identity drift from subject-string inconsistency or drift-compounding ancestral sampler | [Identity drift](#identity-drift) |
| Mouth doesn't match audio | Lip-sync failure (framing too wide, camera moves, poor audio) | [Lip sync failures](#lip-sync-failures) |
| Prompts feel muddy / lip shapes too arbitrary for consonants | Audio issue — weak presence band or no sibilance | [Audio quality issues](#audio-quality-issues) |
| Output prompts don't land at the times I wrote in the schedule | Runtime schedule snap to iteration grid | [Schedule timing surprises](#schedule-timing-surprises) |
| Wide shot wasn't what I wanted | Prompt said "wide shot" or "wide stage framing"; for standup/dialogue, keep it to medium/close-up | [Lip sync failures](#lip-sync-failures) |
| Big jump at the boundary where a prompt changes | Prompt delta too large, or blend_seconds mis-set | [Iteration-boundary seams](#iteration-boundary-seams) + [Blend_seconds pitfalls](#blend_seconds-pitfalls) |

---

## Debugging each symptom

### Decode-tile seams

**What it looks like**: periodic pulse / micro-flicker / micro-color-shift every ~2-3 seconds all the way through the video. Uniform rhythm, starts from near the beginning (t ≈ 2, 4, 6, 8, ...).

**Root cause**: `VAEDecodeTiled` (typically node 1604) decodes the
video latent in temporal tiles. Tile boundaries produce subtle seams
because the decoder doesn't perfectly reconstruct identical content
at the boundaries of adjacent tiles.

With widgets `[tile_size, overlap, temporal_size, temporal_overlap]`:
- Tile stride (pixel frames) = `temporal_size − temporal_overlap`
- At 25 fps, tile stride in seconds = `(temporal_size − temporal_overlap) / 25`
- A seam lands at every multiple of that stride.

Current workflow default is `[512, 64, 64, 8]` → `(64-8)/25 = 2.24s` per tile → a seam approximately every 2.24s. That's the symptom.

**Fix (production, shipped in default workflows)**: node 1604's widgets
are now `[512, 64, 512, 64]` across all example workflows. This gives
tile stride = `448/25 = 17.92s`, which **aligns decoder tile
boundaries with iteration-loop boundaries** (stride_seconds = 17.88 by
default at `overlap_seconds=2`). Decoder seams now co-locate with
iteration seams instead of adding new seam positions.

**Maintenance invariant**: if you change `overlap_seconds`, the
`VAEDecodeTiled` widgets must change too to preserve alignment.
Rule: `(temporal_size − temporal_overlap) / fps ≈ window_seconds − overlap_seconds`.
Specific values at `window_seconds=19.88, fps=25`:

| `overlap_seconds` | Iter stride | Target tile stride | `temporal_size, temporal_overlap` |
|---|---|---|---|
| **2.0 (default)** | **17.88 s** | **17.92 s** | **`512, 64`** |
| 3.0 | 16.88 s | 16.96 s | `480, 56` |
| 4.0 | 15.88 s | 15.92 s | `448, 50` |
| 1.0 | 18.88 s | 18.88 s | `544, 72` |

If you change overlap and forget to update the decoder, the tile and
iteration strides drift apart over the video. Empirical test on a 3-min
run confirmed this: `overlap=3` with decoder widgets `[512, 64, 512, 64]`
produces ~1s drift per iteration, re-introducing mid-iteration seams
that grow over time. Reverting overlap to 2 tightens alignment back to
0.04s and seams resolve.

Not currently enforced in code — a `scripts/check_stride_alignment.py`
pre-flight validator would be a worthwhile addition.

If `[512, 64, 512, 64]` OOMs: step down to `[512, 64, 256, 32]`
(tile stride 8.96s — one mid-iteration seam per iteration; still
~8× fewer seams than `temporal_size=64`).

**Diagnostic run (optional, if unsure)**: `[512, 64, 2048, 128]` — tile
stride ~77s, so for a 3-min video there's only 2-3 tile boundaries
total. If the every-2-seconds jitter disappears with this setting,
decoder tiling was the cause. VRAM-intensive; may OOM on 24 GB + 22B
model. If it does, try `[512, 64, 1024, 128]` or `[512, 64, 768, 96]`.

**Constraint**: ComfyUI clamps `overlap ≤ tile_size / 4` (both spatial
and temporal). `temporal_overlap=64` with `temporal_size=512` is valid
(`64 ≤ 128`). Going higher than `temporal_size/4` silently clamps and
can cause the symptom you're trying to fix.

### Iteration-boundary seams

**What it looks like**: identity or color hand-off every ~17.88 s
(at default `overlap_seconds=2.0`). Becomes more visible after
decoder-tile seams are fixed, because it was masked by them before.

**Root cause**: each loop iteration is an independent LTX sampler pass
using the previous iteration's tail as spatial/audio context. The
model doesn't perfectly reconstruct identical pixels at the transition
point, producing a small visible cut.

**First lever — increase `overlap_seconds`**:

| `overlap_seconds` | Stride | Iterations per 3 min | Trade-off |
|---|---|---|---|
| 2.0 (default) | 17.88 s | ~10 | Baseline |
| **3.0** | **16.88 s** | **~11** | **Recommended when iteration seams are visible** |
| 4.0 | 15.88 s | ~12 | Very smooth transitions, ~20% more compute than default |

More overlap = more context carryover = smoother hand-off, at cost of
~1 s less new content per iteration and slightly more compute.

**Second lever (Phase 2, parked)**: `LatentOverlapCrossfade` node — blends
the overlap region in latent space instead of trimming. Not yet
implemented; spec lives in the internal planning file (not in repo).

**If you bump `overlap_seconds`**: the iteration grid shifts (stride
changes from 17.88 → 16.88). Schedules pre-snapped to the old grid
will get runtime-snapped to the new grid. See [Schedule timing
surprises](#schedule-timing-surprises) for what that means.

### Mid-iteration mix

**What it looks like**: one iteration (~18s segment) looks visibly
different from the ones before and after — as if the model was
confused mid-generation.

**Root cause**: the pre-Phase-1 `blend_seconds` logic applied a single
blend_factor per iteration, producing spike blends when the iteration's
current_time happened to land near a schedule boundary. One iteration
ran on mixed conditioning (e.g., `0.28 × prompt_A + 0.72 × prompt_B`)
while neighbors ran on pure prompts — visible as a "weird" segment.

**Fix**: make sure `TimestampPromptSchedule.snap_boundaries = True`
(default in post-Phase-1 workflows). That snaps schedule boundaries to
iteration multiples so every iteration runs on exactly one pure prompt.
Also ensure `blend_seconds = 0` unless you explicitly want cross-fading.

**Phase-1 auto-clamp**: if you accidentally set `blend_seconds`
between 0 and `stride_seconds`, the runtime clamps it up to
`stride_seconds` with a one-time warning. If you see that warning,
your blend is being adjusted — read the log or change the value.

### Identity drift

**What it looks like**: the subject's face, hair, or clothing
subtly morphs across 30-60 s of video. By the 2-min mark the subject
looks like a different person than they did at 0:15.

**Root causes (in order of impact)**:

1. **Subject-string inconsistency across schedule entries.** The #1
   cause. If entry 1 says "blonde comedian in a striped shirt" and
   entry 2 says "comedian in blue and gray shirt," the text encoder
   produces different embeddings and the model drifts to match.
   Always use **byte-exact identical subject strings** across all
   entries. R3 in the LLM system prompt enforces this.

2. **Ancestral sampler stochasticity compounding.** `euler_ancestral`
   adds noise per step, which diverges across iterations. Use
   **`euler`** (deterministic, matches the LTX 2.3 distilled training
   regime) instead.

3. **Low `overlap_seconds`.** 1 s of overlap gives the model minimal
   context for the identity hand-off. 2 s is default; 3 s if drift is
   visible.

4. **Wide framing.** When the face is small in frame, identity detail
   is compressed into few pixels; tiny reconstruction errors compound.
   Keep framing to medium or medium-close-up throughout.

5. **In-iteration camera moves.** `dolly in`, `jib up`, `dolly out`
   within a single window make the face rescale/reposition across
   frames of one sampler pass. LTX has to re-establish identity each
   frame. Use `static camera, locked off shot` everywhere except the
   final OUTRO's fade-out dolly.

6. **Ancestral sampler x distilled model mismatch.** The 22B distilled
   LTX was DMD-distilled to match the teacher on a 1st-order Euler-like
   update rule. `euler_ancestral` injects noise at sigma levels the
   distillation wasn't trained to correct — compounds over iterations.

Fix by working top-down: subject-string first (check every schedule
entry has identical subject), then sampler choice, then overlap, then
framing/camera language.

### Lip sync failures

**What it looks like**: mouth movements don't match the audio's
phonemes. Particularly visible on fricatives (/s/, /sh/, /t/, /k/) —
the mouth makes vowel-like shapes for consonant sounds.

**Root causes (in order of impact)**:

1. **Face too small in frame.** Audio-video cross-attention needs
   mouth pixels to predict. A face covering 10% of frame has maybe
   8000 mouth pixels; LTX can establish phoneme correspondence. A
   face covering 1% of frame has ~80 mouth pixels; sync fails. Use
   medium-shot or medium-close-up framing. **Never use "wide shot" in
   prompts for speech-heavy content.**

2. **Camera moving during a window.** `dolly in`, `jib up`, `slow
   zoom`, `handheld sway` — all of these move/rescale the face across
   the window. LTX processes one window in a single sampler pass;
   if the mouth target shifts every frame, sync can't lock. Use
   `static camera, locked off shot`.

3. **Audio missing upper-band content.** LTX's audio VAE operates at
   16 kHz with n_fft=1024 and mel_hop=160 (~10 ms frames). It attends
   to mel-bin energy across ~0-8 kHz. If your source has no sibilance
   (4-8 kHz band), LTX has no signal for fricative mouth shapes —
   they're guessed arbitrarily. Run
   `scripts/preprocess_audio_for_ltx.py` or use the offline CLI to
   rebalance the spectrum; see [Audio quality
   issues](#audio-quality-issues).

4. **Subject drift.** If the subject string changes mid-schedule, the
   model re-interprets "who is singing/speaking" per boundary and has
   to re-establish mouth correspondence. Keep subject byte-exact.

5. **Using generic verbs instead of sync-driving ones.** "is
   performing," "is speaking," "is vocalizing" — these are weak. LTX's
   action-verb attention drives lip shape. Use:
   - Music: "is singing..." (single) / "are singing together..." (multi)
   - Standup: "is telling a joke," "is delivering the punchline",
     "is pausing for the laugh," etc. (see `docs/system_prompt.md`).
   - Dialogue: emotion-loaded verbs like "is pressing the point," "is
     softening." Avoid the too-generic "is speaking."

Fix top-down. #1 and #2 are usually the dominant problems on real
runs.

### Audio quality issues

**What it looks like**: (symptom expressed on video, not audio)
mouth makes reasonable vowel shapes but fails on consonants.
Specifically /s/ /sh/ /t/ /k/ look generic — mouth opens partially
regardless of the actual phoneme.

**Root cause**: audio source lacks content in the bands LTX's audio
VAE uses for phoneme discrimination. Typically:
- Bass-heavy spectrum (60-800 Hz dominates) → presence band is masked
- Dull sibilance (≤4-8 kHz has almost no energy) → fricatives
  invisible to upper mel bins
- Low SNR (<20 dB voice-to-noise) → noise floor competes with quiet
  consonants

**Diagnose**: run `scripts/analyze_audio_features.py your_file.mp3`
(or load in any spectrum analyzer). Check the 4-8 kHz band relative
to 300-800 Hz. If 4-8 kHz is more than ~20 dB below the loudest band,
you have dull sibilance — fricatives are effectively gone.

**Fix (offline CLI, exists today)**:
```bash
uv run --group analysis python scripts/preprocess_audio_for_ltx.py \
    input.m4a output.wav --trim-end 184
```
Applies a 5-stage EQ chain:
- HP 80 Hz (removes rumble)
- 200 Hz −3 dB (de-boom)
- 400 Hz −2 dB (de-box)
- 3 kHz +4 dB (presence / intelligibility)
- 6.5 kHz +3 dB (sibilance recovery)
- loudnorm to −16 LUFS, TP ceiling −2.0 dB

Outputs WAV (no MP3 re-encoding overshoot). Feed the processed file
into the workflow's audio input.

**Future**: this will be available as an in-workflow node
(`AudioPreprocessForLTX`) per Phase A1 of the parked audio
preprocessing track.

**Special case — crowd noise**: for standup or live recordings where
crowd laughter is mixed with speech, vocal separation BEFORE the LTX
audio encode can help. Current workflow uses MelBandRoformer with a
vocals model. A crowd-removal variant
(`mel_band_roformer_crowd_aufr33_viperx`) exists and shares
architecture. Using it is a Phase A2 task in the plan.

### Schedule timing surprises

**What it looks like**: you wrote a schedule with boundary at `1:15`
but the prompt change happens at `1:11` (or some other nearby time).
Or prompt boundaries aren't exactly where your audio's section
boundaries are.

**Root cause**: with `snap_boundaries=True` (default), the runtime
snaps every schedule boundary to the nearest integer multiple of
`stride_seconds = window_seconds − overlap_seconds`. Default at
`overlap=2.0`: stride = 17.88. At `overlap=3.0`: stride = 16.88.

With stride = 17.88:
- 1:15 = 75 s → 75 / 17.88 = 4.19 → rounds to 4 → 4 × 17.88 = 71.52 s = `1:11`
- So your 1:15 entry actually starts at 1:11.

This is **intentional** — it prevents mid-iteration mixed conditioning
(the jitter source Phase 1 fixed). But it means the widget text and
the actual behavior drift by up to ~9 seconds (half of stride) per
boundary.

**Fix options**:

1. **Accept and regenerate**: re-snap your schedule to the current
   stride grid before pasting into the widget. For stride=17.88, valid
   boundaries are 0:00, 0:17, 0:35, 0:53, 1:11, 1:29, 1:47, 2:05, ...
   Rule R9 in the LLM system prompt instructs the LLM to emit
   pre-snapped schedules.
2. **Accept and shift the interpretation**: leave the widget as-is,
   recognize that a CHORUS entry labeled 0:45-1:15 actually runs
   during the 0:36-1:11 segment. Content is still right; only timing
   is off by ~5 s.
3. **Disable snapping** (`snap_boundaries = False`): gets you exact
   timing but re-introduces mid-iteration mixed conditioning. Don't
   use this unless you really need sub-stride precision AND accept
   the jitter risk.

### Blend_seconds pitfalls

**What it looks like**: output feels "washed out" or "always in
transition"; prompts seem less distinct than they should be.

**Root cause**: `blend_seconds` that's too large dilutes each prompt
with adjacent ones. `blend_seconds ≥ 2 × stride_seconds` means you're
never running on a pure prompt anywhere in the timeline.

Historically (pre-Phase-1), `blend_seconds` < `stride_seconds`
produced jitter instead — sawtooth blend_factor per iteration. That
mode is now auto-clamped.

**Guidance**:
- `blend_seconds = 0` (default): hard switch at each iteration
  boundary. Clean when subject is identical across entries. **Use
  this for identity-anchored content (standup, podcast, music
  video).**
- `blend_seconds = stride_seconds` (~18): raised-cosine ramp spanning
  one iteration on each side of the boundary. Use if you see a
  visible seam at prompt transitions specifically (not decoder or
  identity-drift seams).
- `blend_seconds = 2 × stride_seconds` (~36): softer ramp, dilutes
  adjacent prompts. Rarely needed for our workflows.
- `blend_seconds` between 0 and `stride_seconds`: auto-clamped to
  stride with warning (don't do this on purpose).

---

## Diagnostic experiments

When you can't tell which layer is contributing a given symptom,
these controlled experiments isolate variables. Each is one-variable
runs at the same seed so differences are attributable.

### E1: Is it decoder tiles? (isolates decode layer)

Run with `VAEDecodeTiled` = `[512, 64, 2048, 128]` (effectively no
temporal tiling, or only 2-3 tiles over the whole video). Same seed,
same everything else. If mid-iteration jitter disappears or changes
position significantly, decoder tiles were the cause.

### E2: Is it model-intrinsic? (isolates noise sensitivity)

Run with a different RANDOM_SEED (node 1322), everything else identical.
If the seam TIMESTAMPS stay the same, the cause is deterministic (model
structure, decoder tiles, schedule boundaries). If they move, the cause
has a stochastic component (sampler, NAG, noise).

### E3: Is it audio-driven? (isolates audio conditioning)

Replace the audio source with a silent track of the same length, or
skip the audio conditioning path entirely (t2v mode). If jitter
disappears, it's audio-driven. If it persists, it's video-side.

### E4: Is it NAG modulation? (isolates attention guidance)

Set Node 508 `LTX2_NAG` enable → `false`. Run. If jitter changes,
NAG is contributing. (NAG typically improves outputs; disabling is a
diagnostic, not a fix.)

### E5: Is overlap_seconds the cause? (iteration hand-off)

Bump `overlap_seconds` from 2 → 3 while keeping everything else. If
iteration-boundary seams (at t ≈ 18, 36, 54, ...) become visibly
smoother, overlap was the dominant factor at those timestamps.

---

## Known-good baselines

**For standup / speech / dialogue** (no singing):

| Widget | Node | Value |
|---|---|---|
| `sampler_name` | 154 KSamplerSelect | `euler` (not `euler_ancestral`) |
| `shift` | 1513 ModelSamplingSD3 | `13` |
| scheduler | 1421 BasicScheduler | `linear_quadratic, 8, 1` |
| CFG | 153 CFGGuider | `1.0` (distilled model) |
| NAG scale/tau/alpha/enable | 508 LTX2_NAG | `11, 0.25, 2.5, true` |
| Sage attention mode | 268 PathchSageAttentionKJ | `sageattn_qk_int8_pv_fp8_cuda` or `sageattn_qk_int8_pv_fp16_triton` |
| `window_seconds` | 688 FloatConstant | `19.88` |
| `overlap_seconds` | AudioLoopController | `2.0` or `3.0` |
| `temporal_size, temporal_overlap` | 1604 VAEDecodeTiled (widgets 3-4) | `512, 64` (not `64, 8`) |
| `snap_boundaries` | 1558 TimestampPromptSchedule | `true` |
| `blend_seconds` | 1558 TimestampPromptSchedule | `0.0` |
| `start` (outer trim) | 567 TrimAudioDuration | `0` (for pre-trimmed audio) |
| Negative prompt | 507 CLIPTextEncode | standup-tuned (see `internal/prompt_comedy1.md`) |

**For music video** (singing):

Same as above except:
- Verb pool: "is singing..." / "are singing together..."
- Negative prompt: music-tuned defaults (see `example_workflows/*.json`)
- NAG settings: same (11, 0.25, 2.5, true)

### Node 1604 VAEDecodeTiled — widget meaning

Widgets are in **pixel frames** at the decoder output:

- `tile_size`: spatial tile dimension (pixels). Default 512.
- `overlap`: spatial overlap (pixels). Default 64. Constraint: ≤ tile_size/4.
- `temporal_size`: pixel frames per temporal tile. **Current example
  workflows ship with 512 (tile stride 17.92 s, aligned to iteration
  stride). If you change `overlap_seconds`, recompute per the
  maintenance invariant above.**
- `temporal_overlap`: pixel frames overlapped between adjacent
  temporal tiles. **64 when temporal_size=512**. Constraint:
  ≤ temporal_size/4.

At 25 fps, `temporal_size=512, temporal_overlap=64` gives tile stride
= `(512-64)/25 = 17.92 s`, which aligns with loop iteration boundaries
(17.88 s at default `overlap_seconds=2`).

---

## Things that look like bugs but aren't

- **"My schedule says 0:15 but the model changes at 0:17."** That's
  `snap_boundaries=True` doing its job. See [Schedule timing
  surprises](#schedule-timing-surprises).
- **"Warning: blend_seconds clamped to stride_seconds."** Expected
  and correct — the value you set can't produce smooth ramps at
  iteration resolution. Either use 0 or ≥ stride_seconds.
- **"The crowd reaction looks canned — same laugh each time."**
  Check whether your init image shows the SAME CROWD MEMBERS in fixed
  positions. LTX i2v anchors composition; if the init shows 3 people
  laughing, you'll see those 3 people laughing for 3 minutes. Not a
  bug — a feature of i2v.
- **"Some iteration boundaries are smoother than others."** Natural
  variance — the visibility of an iteration seam depends on how much
  the adjacent prompts differ AND how well LTX reconstructs the
  overlap. Not every boundary shows a seam.
- **"First iteration looks a bit different from subsequent ones."**
  The initial render (Node 169) is one pass of t2v-via-i2v; loop
  iterations carry over context from prior ones. A subtle
  "settling" effect in the first 1-2 seconds is normal.

---

## When to re-run vs change-and-re-run

Each run takes ~15-30 minutes (distilled 22B model at 8 steps). Budget
your iterations.

**Re-run identically** (no config change): only if you think the
issue might be stochastic. Fix the seed (Node 1322 widget value
`fixed`, same integer) and re-run. If the output is different, a seed
change would help; if identical, stochasticity isn't contributing.

**Change ONE thing and re-run**: the default diagnostic approach.
Keep a log of which change correlates with which observed
improvement.

**Change multiple things at once**: only when you're confident each
change is independent AND you won't need to diagnose further. For
example, the "known-good baseline" above is a bundled multi-change
configuration, but it's based on accumulated prior experiments.

---

## If you've tried everything and it still doesn't work

Re-read this list top-to-bottom and verify each box is actually
checked, not just "I think I set that." Common false-checks:

- Workflow loaded from a stale JSON that didn't include your widget
  changes. Save the workflow JSON to a new file before each run so
  you have a checkpoint.
- Node 169 prompt was edited but TimestampPromptSchedule's 0:00 entry
  wasn't — or vice versa. Verify they're **byte-exact identical**.
- ComfyUI UI shows different widget values from the JSON (cache
  mismatch). Reload the page or restart ComfyUI.
- Audio file was pre-trimmed externally but node 567 still has
  `start=5`, double-trimming your routine. Set 567's `start` to 0
  for pre-trimmed audio.
- `overlap_seconds` changed but the schedule wasn't re-snapped.
  Runtime snap will re-snap on load, but the displayed widget
  timestamps won't match until you regenerate.

If all boxes are checked and the symptom persists, it's likely in the
model-intrinsic layer — sampler steps, latent chunk structure,
audio VAE temporal resolution. These can't be fixed from our side
without LTX 2.3 model changes. Document the specific symptom + timestamp
in `internal/log/log_<date>.md` for future reference and move on.

---

## Cross-references

- Prompt rules + widget guidance: `docs/prompt_creation_guide.md`
- LLM system prompt (music + standup variants):
  `docs/system_prompt.md` + `scripts/analyze_audio_features.py`
- Audio analysis pipeline: `docs/audio_analysis_guide.md`
- LTX 2.3 model reference: `docs/ltx23_model_reference.md`
- Profiling opt-in: `docs/profiling_guide.md`
- Current plan + post-phase findings: internal planning file (not in repo)
- Standup example schedules: `internal/prompt_comedy1.md`,
  `internal/prompt_comedy2.md`
