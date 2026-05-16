Last updated: 2026-05-16

# Debugging Guide: Quality Problems in the Audio-Loop Pipeline

Problem-first troubleshooting. You see X in your output — here's how
to diagnose and fix it.

Related docs (all paths relative to repo root):
- `docs/guides/prompt_creation_guide.md` — prompt rules + widget-value guidance
- `docs/guides/audio_analysis_guide.md` — offline analysis + runtime audio nodes
- `docs/reference/standup_system_prompt.md` — LLM system prompt for schedule generation (standup variant)
- `docs/guides/profiling_guide.md` — performance profiling (opt-in)

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
| First few seconds have no motion / frozen frames | `LTXVPreprocess img_compression` widget is `0` — preprocessing is skipped, model treats pristine init as "stay here" | [Frozen first frames](#frozen-first-frames) |
| Illustrated init progressively becomes photoreal / "broadway musical" over iterations | LTX 2.3's audio-video cross-attention has photoreal-trained prior; `Style: illustrated.` at CFG=1 can't overcome it | [Style drift toward photoreal](#style-drift-toward-photoreal) |
| ~20% of generations: an unrelated photoreal woman (often holding a microphone) replaces the reference subject in later windows; iter 0 looks correct | Loop-guide branch skips `LTXVPreprocess` — anchors to raw resized init instead of the preprocessed one the initial render uses. Cross-attention drifts across the delta and reasserts its "singing woman" prior | [Preprocess asymmetry (F2)](#preprocess-asymmetry-f2) |
| Subtle identity feature drift (hair/clothing/face) over later windows with no microphone — subtler than F2, present even after the preprocess fix | Loop-body CFGGuider bypasses `LTXVCropGuides` on the CONDITIONING path; guide-keyframe metadata accumulates across iterations in ways the initial render never saw | [Loop-cropguides asymmetry (F3)](#loop-cropguides-asymmetry-f3) |
| Lip-sync desyncs progressively over 10 iterations | Integer-latent stride drift (fixed 2026-04-20 in `AudioLoopController`) | [Lip-sync drift over iterations](#lip-sync-drift-over-iterations) |
| Resolution-related sampling oddness | `ImageResizeKJv2` width/height not divisible by 32 (single-stage) or 64 (distilled) | [Resolution alignment](#resolution-alignment) |
| Items from the negative prompt (microphones, duplicate characters, etc.) reappear starting iter 2+ even though iter 1 is clean; or `Style: illustrated.` inits slide toward photoreal; or anatomy glitches (deformed hands, extra limbs) return after the first iteration — and the schedule-bypassed run is clean | CLIP loaded inside the loop body is evicting the DiT; LTX2_NAG's captured negative-conditioning tensor goes stale across the offload/reload round-trip (`object_patches` are not device-migrated by ComfyUI). Fixed 2026-04-22 by moving CLIP out of the loop via `TimestampPromptScheduleBatchEncode`. | Migrate the workflow: `uv run --group dev python scripts/apply_batch_encode_fix.py`. Full technical reference: `docs/analysis/nag_object_patches_offload_asymmetry.md`. |

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
- At the canonical 25 fps, tile stride in seconds = `(temporal_size − temporal_overlap) / 25`
- A seam lands at every multiple of that stride.

Current workflow default is `[512, 64, 64, 8]` → `(64-8)/25 = 2.24s` per tile → a seam approximately every 2.24s. That's the symptom.

**Fix (production, shipped in default workflows)**: node 1604 (and
node 1597) are now `LTXVTiledVAEDecode` (from `ComfyUI-LTXVideo`)
across all example workflows — **spatial-only tiling, no temporal
tiling at all**. No temporal tile boundaries exist, so there are no
mid-video decoder seams of any stride. This eliminates the class of
problem structurally.

Widgets on the LTX decoder: `[horizontal_tiles=2, vertical_tiles=2,
overlap=1, last_frame_fix=true, working_device="auto", working_dtype="auto"]`.

To apply or revert the swap against any workflow:

```bash
# Apply: VAEDecodeTiled → LTXVTiledVAEDecode
uv run python scripts/apply_ltx_decoder.py

# Revert: restore VAEDecodeTiled with stride-aligned widgets
uv run python scripts/apply_ltx_decoder.py --revert
```

Both directions are idempotent (re-run is a no-op). Round-trip is
byte-identical.

To check any workflow's decoder configuration:

```bash
uv run python scripts/validate_workflow_decoder.py
```

Warns on misaligned VAEDecodeTiled widgets; emits OK for
LTXVTiledVAEDecode regardless of overlap_seconds.

### Fallback: staying on generic VAEDecodeTiled

If you need to stay on the generic `VAEDecodeTiled` for any reason
(VRAM constraints where LTX's spatial-only tiling doesn't fit, legacy
workflow compatibility), you must keep widget values aligned with the
iteration stride. Rule: tile stride (pixel frames) must match the
effective iteration stride reported by `AudioLoopController`. Since the
2026-04-20 integer-latent fix, stride is quantized to LTX's 8-pixel
temporal boundary, so the widget math is:
`(temporal_size − temporal_overlap) / fps = stride_seconds` (where
`stride_seconds` is the controller's `stride_seconds` output, NOT
`window − overlap`).

Values below are at the canonical LTX 2.3 inference `fps=25` (`docs/reference/ltx23_model_reference.md` § "`frame_rate`: canonical inference value is 25"). A 2026-05-15 sweep briefly defaulted to `fps=24`; reverted to 25 on 2026-05-16 (production render validation pending). Same `window_seconds=19.88`:

| `overlap_seconds` (target) | Iter stride (@fps=25) | `temporal_size, temporal_overlap` |
|---|---|---|
| 1.0 | 18.88 s | `544, 72` |
| 2.0 | 17.92 s | `512, 64` |
| 3.0 | 16.96 s | `480, 56` |
| 4.0 | 16.00 s | `448, 48` |

If you change overlap and forget to update the decoder, tile and
iteration strides drift apart over the video — ~1s per iteration per
1s of overlap delta, re-introducing mid-iteration seams that grow
over time. Use `scripts/validate_workflow_decoder.py` to catch drift
early.

**The `LTXVTiledVAEDecode` swap eliminates this whole concern.** Prefer
the structural path unless VRAM forces the fallback.

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

**What it looks like**: identity or color hand-off every ~17.92 s
(at default `overlap_seconds=2.0`). Becomes more visible after
decoder-tile seams are fixed, because it was masked by them before.

**Root cause**: each loop iteration is an independent LTX sampler pass
using the previous iteration's tail as spatial/audio context. The
model doesn't perfectly reconstruct identical pixels at the transition
point, producing a small visible cut.

**First lever — increase `overlap_seconds`**:

| `overlap_seconds` | Stride | Iterations per 3 min | Trade-off |
|---|---|---|---|
| 2.0 (default) | 17.92 s | ~10 | Baseline |
| **3.0** | **16.96 s** | **~11** | **Recommended when iteration seams are visible** |
| 4.0 | 16.00 s | ~12 | Very smooth transitions, ~20% more compute than default |

More overlap = more context carryover = smoother hand-off, at cost of
~1 s less new content per iteration and slightly more compute.

**Second lever (Phase 2, parked)**: `LatentOverlapCrossfade` node — blends
the overlap region in latent space instead of trimming. Not yet
implemented; spec lives in the internal planning file (not in repo).

**If you bump `overlap_seconds`**: the iteration grid shifts (stride
changes from 17.92 → 16.96). Schedules pre-snapped to the old grid
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

5. **Using generic verbs instead of action-specific ones.** "is
   performing," "is speaking," "is vocalizing" — these are abstract
   enough that the model can't bind them to visible motion. LTX's
   action-verb cross-attention drives lip + body shape. Pick a
   concrete verb that matches the visible action:
   - Music (vocal): "is singing..." (single) / "are singing together..." (multi)
   - Music (dance / movement): "is dancing," "spins through the frame," etc.
   - Music (instrumental): "is playing <instrument>"
   - Standup: "is telling a joke," "is delivering the punchline",
     "is pausing for the laugh," etc. (see `docs/reference/standup_system_prompt.md`).
   - Dialogue: emotion-loaded verbs like "is pressing the point," "is
     softening." Avoid the too-generic "is speaking."

   Reframed 2026-05-04 from "must contain singing" — the verb is
   load-bearing, but it's the action-class binding that matters, not
   the literal word "singing".

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
snaps every schedule boundary to the nearest integer multiple of the
`stride_seconds` output from `AudioLoopController`. Stride is quantized
to LTX's 8-pixel temporal boundary (see the lip-sync drift section
below for the derivation). Canonical `fps=25`, `window=19.88`: `overlap=2.0`
→ stride = **17.92 s**; `overlap=3.0` → stride = **16.96 s** (the 2026-05-15 sweep to `fps=24` was reverted on 2026-05-16; for any other fps, re-derive via `new_latents * 8 / fps`).

With stride = 17.92:
- 1:15 = 75 s → 75 / 17.92 = 4.185 → rounds to 4 → 4 × 17.92 = 71.68 s ≈ `1:12`
- So your 1:15 entry actually starts at 1:12.

This is **intentional** — it prevents mid-iteration mixed conditioning
(the jitter source Phase 1 fixed). But it means the widget text and
the actual behavior drift by up to ~9 seconds (half of stride) per
boundary.

**Fix options**:

1. **Accept and regenerate**: re-snap your schedule to the current
   stride grid before pasting into the widget. For stride=17.92, valid
   boundaries are 0:00, 0:18, 0:36, 0:54, 1:12, 1:30, 1:47, 2:05, ...
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

### Frozen first frames

**Symptom**: the first 1-3 seconds of the video show no motion; the
init image appears to "hold" before movement begins.

**Root cause**: `LTXVPreprocess` (node 446) `img_compression` widget
is `0`. Looking at `comfy_extras/nodes_lt.py:577-588`:

```python
def preprocess(image, crf=29):
    if crf == 0:
        return image  # SKIPS preprocessing entirely
    ...  # Otherwise: JPEG-like compression added
```

LTX 2.3 is trained on conditioning images that have compression
artifacts. Feeding a pristine (uncompressed) image is out-of-
distribution; the model's response to "perfectly clean init" is to
hold exactly on it until the sigma schedule forces it to diverge.

**Fix**: set `LTXVPreprocess.img_compression` to `18` (Lightricks'
upstream 2.3 value in `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`).
`35` is the comfy-core generic LTX default if 18 feels too aggressive.

**Verification**: Re-run. The first frames should show ambient motion
(eye saccade, slight head sway, hair movement) instead of a held still.

### Preprocess asymmetry (F2)

**Symptom**: in ~1 out of 5 generations, an unrelated photoreal woman —
often holding a microphone — replaces the reference subject in later
loop windows. Iter 0 is correct; the drift compounds from iter 1 onward.
Overall generation quality is noticeably better 80% of the time, making
the regression easy to miss.

**Root cause**: the loop guide branch was picking up the RAW resized init
image (from `#445 ImageResizeKJv2`) instead of the preprocessed one the
initial render consumes (from `#446 LTXVPreprocess img_compression=18`).
Iter 0 locks in preprocessed stats via `#531 LTXVImgToVideoInplaceKJ`;
iters 1+ anchor via `#1519 LTXVAddLatentGuide` to the raw image. Cross-
attention (photoreal-trained) drifts across that delta iteration-over-
iteration and reasserts its "singing woman with microphone" prior —
the textbook fingerprint.

CLAUDE.md flags `img_compression=0` vs `18` as a frozen-first-frame
footgun; the loop branch was effectively running `=0` while initial
ran `=18`.

**Fix**:

```bash
uv run --group dev python scripts/apply_loop_guide_preprocess_symmetry.py
```

Applied to all six shipped workflows. After the fix both paths share
`#446 LTXVPreprocess` output:

```
#445 ImageResizeKJv2 → #446 LTXVPreprocess → { #531 (initial), #650 Set_input_image (loop guide) }
```

`--dry-run` previews the diff without writing; `--revert` undoes.

**Verification**: `uv run --group dev python scripts/audit_workflows.py`
must report `preprocess_symmetry OK` for every workflow. The audit check
was added specifically to prevent this regression recurring.

### Loop-cropguides asymmetry (F3)

**Symptom**: subtle iter-over-iter identity drift — small feature shifts
in hair, clothing, facial structure — concentrated in later windows. No
microphone (that's F2). Often shows up AFTER you've applied the F2 fix
and the worst regression is gone but something still feels slightly off
in long generations.

**Root cause**: the initial-render CONDITIONING path runs through
`#381 LTXVCropGuides` before `#153 CFGGuider` — guide-keyframe metadata
is stripped from CONDITIONING before the sampler sees it. Inside the
loop subgraph, `#655 LTXVCropGuides` exists with both inputs wired from
`#1519 LTXVAddLatentGuide`, but its CONDITIONING outputs are unconsumed.
`#644 CFGGuider` reads directly from `#1519[0,1]`, bypassing `#655`.
Cropped conditioning is computed every iteration, then discarded.
Guide metadata accumulates in CONDITIONING across N iterations in a way
the initial render never saw.

**Fix**:

```bash
uv run --group dev python scripts/apply_loop_cropguides_symmetry.py
```

After the fix, `#644` reads from `#655[0,1]` — topologically symmetric
to the initial path's `#164 → #381 → #153`.

`--dry-run` and `--revert` supported.

**Verification**: `audit_workflows.py` must report
`loop_cropguides_symmetry OK` across all workflows. The audit check was
added specifically to prevent this regression recurring.

**Related**: F2 is the preprocess-branch symmetry on the LATENT path;
F3 is the crop symmetry on the CONDITIONING path. They're independent
fixes addressing two different halves of the same broader constraint:
"loop-body DiT inputs must be preprocessed identically to the initial
render." When diagnosing identity drift, check both before reaching for
sampler/prompt changes.

### Style drift toward photoreal

**Symptom**: an illustrated / painterly init image progressively
becomes photoreal across the 10-iteration loop. "Human mouth
superimposed over illustration" early; "broadway musical staging" by
iteration 10.

**Root cause**: LTX 2.3's audio-video cross-attention was trained
predominantly on photoreal footage. The singing-mouth pathway has a
photoreal prior baked into the transformer weights. Text conditioning
at CFG=1 (`Style: illustrated.`) is too weak to overcome it. The
prior compounds across iterations because each iter's output becomes
next iter's context latent.

**First-line fix**: match init-image style family to training
distribution. Use a cinematic / photoreal init image and set
`--style cinematic` in the generator. This removes the gradient the
drift was running down.

**Partial mitigations** (if you want to stick with illustrated):

- Stack anti-photoreal terms in the negative prompt (node 507):
  `photorealistic, realistic skin, film grain, cinematic lighting,
  live-action footage, theatrical stage lighting, broadway musical`.
  The negative path specifically suppresses these concepts; effect
  is modest but free.
- Try the `_latent_stg.json` variant (STG quality lift preserves
  per-attention-block style identity better than NAG in some cases).
- Lower `overlap_seconds` has a secondary effect: each iteration
  starts more freshly from the init-image-anchored first frame,
  bleeding less photoreal accumulation. Tradeoff: more iteration
  seams, less subject continuity.

**Structural fix** (not yet built): multi-image-guide per iteration
via KJNodes' `LTXVAddGuideMulti`. Places the init image as a guide
at frames 0, 15, 30, 45... within each iteration window, constantly
re-anchoring style. Requires extension-subgraph surgery.

### Lip-sync drift over iterations

**Symptom**: lip-sync is tight in iterations 1-5 but visibly desyncs
by iteration 9-10. Worse at higher `overlap_seconds` values.

**Root cause (pre-2026-04-20)**: `AudioLoopController` computed
stride as `window_seconds - overlap_seconds` (continuous seconds)
but each iteration's trimmed latent contributes exactly
`new_latent_frames * 8` pixels to the final decoded video
(integer-latent quanta). At `overlap=2` this was 0.04s/iter drift;
at `overlap=4`, 0.12s/iter = ~1.3s cumulative desync over 10 iters.

**Fix (2026-04-20 onward)**: `AudioLoopController.execute` now
derives stride from integer-latent counts. The `overlap_seconds`
widget is a TARGET; outputs reflect the EFFECTIVE quantized value.
Audio advance per iteration exactly matches video pixel advance
regardless of overlap.

**How to verify the fix is active**: `AudioLoopPlanner` summary
should show `stride_seconds = 17.92` at `window=19.88, overlap=2`
(not 17.88). Locked in by `tests/test_audio_loop_controller.py`.

**If you still see drift post-fix**: check that your ComfyUI
instance has reloaded the updated `nodes.py`. ComfyUI caches custom
node code; restart ComfyUI entirely if you just pulled.

### Manually short-circuit iterations for a quick smoke test

**Symptom**: you want a fast 1-iteration or 3-iteration render to
verify a workflow change without waiting for the full audio length.

**Background**: as of 2026-04-26 every shipped workflow auto-wires
`AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in`,
so by default the loop runs exactly long enough to cover the input
audio. To cap it shorter, override the wire with a constant.

**Recipe** (in the ComfyUI canvas):

1. Drag in an `INTConstant` (KJNodes) and set its widget to the cap
   you want (e.g. `1` or `3`).
2. On the `TensorLoopOpen` node, find the `iterations_in` input
   (rendered as an optional cyan slot near the bottom of the inputs
   list).
3. Right-click the existing wire feeding `iterations_in` (it's coming
   from `AudioLoopPlanner.total_iterations`) and disconnect it.
4. Wire your `INTConstant.value` → `TensorLoopOpen.iterations_in`.
5. Render. The loop now caps at your constant regardless of audio
   length.

**Restore**: reconnect `AudioLoopPlanner.total_iterations` →
`TensorLoopOpen.iterations_in` and delete the constant. Or run
`scripts/apply_iterations_autowire.py` against the workflow file to
restore the canonical wiring.

**Audit**: `scripts/audit_workflows.py` will WARN (not ERR) when
`iterations_in` is wired from a non-AudioLoopPlanner source — this is
intentional, the WARN message says "OK if intentional (e.g.
experiment-tier override)."

For programmatic short tests, use the experiment harness — it copies
the workflow JSON to a temp path and rewires `iterations_in` to an
`INTConstant` of the desired tier (1 / 3 / N) without touching the
canonical workflow file.

---

### Resolution alignment

**Symptom**: subtle sampling artifacts or off-distribution output
that doesn't match LTX's usual quality.

**Root cause**: `ImageResizeKJv2` widget is at a resolution that
isn't divisible by 32 (single-stage requirement) or 64 (distilled
two-stage requirement, per `coderef/LTX-2/packages/ltx-pipelines/
src/ltx_pipelines/utils/helpers.py:325`).

**Fix**: set `ImageResizeKJv2` width and height to multiples of 64.
Common LTX-compliant resolutions: `832x448` (default, 1.857 aspect),
`832x576` (1.444), `896x512` (1.75), `1024x576` (exact 16:9).

**Validate**: run `uv run python scripts/validate_workflow_resolution.py`.
Exits non-zero if any workflow fails.

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

## Tier-1 smoke test for the experiment harness

End-to-end verification that the autoresearch framework actually
populates the tracker with real numbers. Run this once per
environment after install + when DINOv3 / PE-AV deps land.

**Pre-conditions** (one-time setup):

1. Install the experiments + metrics dep groups:
   ```
   uv sync --group experiments --group metrics
   ```
2. Hugging Face auth for DINOv3 (gated):
   ```
   huggingface-cli login   # or: export HF_TOKEN=hf_...
   ```
3. Pick a fixture and fill in real paths. Open
   `internal/autoresearch/fixtures/fixture_man_girl_guitar.json`
   and replace the `<TODO: ...>` strings with absolute paths to a
   real `.wav` and a real `.png` ComfyUI can see.
4. (Optional) harvest an API-format workflow JSON via
   `python3 scripts/extract_workflow_from_png.py <recent-VHS-png>
   --prompt` — required when `--dry-run` is OFF.

**Per-launch setup** (every shell session):

```bash
# Start ComfyUI with telemetry-enabled wrapper (auto-generates RUN_ID,
# enables sage tracer + exec logger).
./start_experiment.sh &

# Tell the harness where ComfyUI writes its mp4s.
export COMFYUI_OUTPUT_DIR=/path/to/your/comfyui/output
```

**Verify wiring before rendering** (cheap; ~1 sec):

```bash
uv run --group experiments --group metrics python -m \
    internal.autoresearch.harness \
    --fixture internal/autoresearch/fixtures/fixture_man_girl_guitar.json \
    --preflight
```

Exits 0 ("Preflight OK.") when fixture validates, env vars set, and
ComfyUI is reachable. Exits 1 with a list of issues otherwise — fix
each before continuing.

**Run the actual smoke test** (one render, ~1-2 min for tier 1):

```bash
uv run --group experiments --group metrics python -m \
    internal.autoresearch.harness \
    --fixture internal/autoresearch/fixtures/fixture_man_girl_guitar.json \
    --tier 1 \
    --api-workflow path/to/api_workflow.json \
    --description "tier-1 smoke test"
```

**Verify outputs**:

```bash
# Tracker row should show status='complete' with metrics populated.
uv run --group experiments python -c \
    "import duckdb; conn = duckdb.connect('internal/autoresearch/runs.duckdb', read_only=True); \
     print(conn.execute('SELECT run_id, status, primary_metric, primary_metric_value, metrics FROM runs ORDER BY ts DESC LIMIT 1').fetchall())"

# Run dir should contain the workflow snapshot, exec/sage jsonl, metrics.json,
# and a symlink to the rendered mp4. (Use the most-recent run dir if you
# opened a fresh shell — `RUN_ID` is per-launch.)
ls "data/runs/$(ls -t data/runs/ | head -1)/"
```

Look for:
- `subject_consistency_status: "ok"` with `mean_to_anchor` near 1.0
  (high = identity preserved against the init image).
- `av_consistency_status: "ok"` with `av_text_sim` somewhere in
  [0.2, 0.5] (depends on prompt; just verify it's not 0 or NaN).
- `sage_summary_status: "ok"` with the kernel distribution dict
  showing the kernels you expect (fp8_cuda, fp16_triton, etc.).

If any metric reports `model_unavailable`, the corresponding dep
group isn't synced or HF auth failed — fix and re-run preflight.
If `video_missing`, `COMFYUI_OUTPUT_DIR` is wrong or the mp4
filename pattern doesn't match `LTX-2_${RUN_ID}_*.mp4`.

## Known-good baselines

**For standup / speech / dialogue** (no singing):

| Widget | Node | Value |
|---|---|---|
| `sampler_name` | 154 KSamplerSelect | `euler` (not `euler_ancestral`) |
| `shift` | 1513 ModelSamplingSD3 | `13` |
| scheduler | 1421 BasicScheduler | `linear_quadratic, 8, 1` |
| CFG | 153 CFGGuider | `1.0` (distilled model) |
| NAG scale/alpha/tau/inplace | 508 LTX2_NAG | `11, 0.25, 2.5, true` (dial `nag_scale` to 3-7 for distilled-1.1 — see `docs/reference/nag_technical_reference.md`) |
| Sage attention mode | 268 AudioLoopHelperSageAttention | `auto_mask_aware` (default; routes masked cross-attn to triton) |
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
- NAG settings: same defaults; dial `nag_scale` to 3-7 for distilled-1.1

### Node 1604 VAEDecodeTiled — widget meaning

Widgets are in **pixel frames** at the decoder output:

- `tile_size`: spatial tile dimension (pixels). Default 512.
- `overlap`: spatial overlap (pixels). Default 64. Constraint: ≤ tile_size/4.
- `temporal_size`: pixel frames per temporal tile. Only relevant if
  you're on the generic `VAEDecodeTiled` fallback; LTX's decoder has
  no temporal tiling. Current example workflows use `LTXVTiledVAEDecode`
  by default.
- `temporal_overlap`: pixel frames overlapped between adjacent temporal
  tiles. Same caveat as above. Constraint: ≤ temporal_size/4.

At the canonical LTX 2.3 inference `fps=25` (`docs/reference/ltx23_model_reference.md` § "`frame_rate`: canonical inference value is 25"), `temporal_size=512, temporal_overlap=64` gives tile stride
= `(512-64)/25 = 17.92 s`. Re-derive the matching loop iteration stride from
`new_latents * 8 / fps` at `overlap_seconds=2` (integer-latent quantized, see
`AudioLoopController`).

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

## Model swap crash (`'NoneType' object has no attribute 'model_size'`)

**Symptom**: Render fails mid-execution with:

```
File "comfy/model_management.py", line 563, in model_offloaded_memory
    return self.model.model_size() - self.model.loaded_size()
AttributeError: 'NoneType' object has no attribute 'model_size'
```

May be preceded by `Exception ignored in: <finalize object ...>` from
`cleanup_models()`.

**Root cause**: ComfyUI core bug. After a large model (e.g. LTX 2.3
22B at 24 GB) is GC'd, its wrapper in
`comfy.model_management.current_loaded_models` survives with
`.model = None`. The next `free_memory()` call walks the list, hits
the stale entry, crashes computing its size. Most likely when
session-stale state piles up across multiple renders.

**First fix: restart ComfyUI.** Clears the registry. Then retry the
render.

**If it recurs across fresh sessions**: wire the `PurgeVRAM` node
between `SamplerCustomAdvanced` output and the next model-using node
(typically `LTXVTiledVAEDecode`). It prunes stale wrappers before
ComfyUI's own cleanup walks them. The node is registered as
"Purge VRAM (defensive)" under `utility/` in the node picker.
Pass-through LATENT — wire it inline; no parameters to set.

Not wired into the canonical workflow by default (the underlying
ComfyUI bug is rare on fresh sessions, and the node touches internals
we'd rather not depend on permanently). If the bug becomes load-
bearing for your workflows, ask for the apply-script wiring.

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

## Case studies: architectural lessons

Three past incidents whose lessons keep recurring. Each shows a class
of bug that won't surface on a local reproduction — you have to think
about it ahead of time.

### CS1: Stale `noise_mask` corrupts later iterations

**Context.** Latent-space loop rework (2026-04-09). Reimplemented the
extension subgraph to pass LATENT between iterations instead of IMAGE,
eliminating the per-iteration VAE round-trip. Initial render was
clean; iterations 2-N had a persistent ~5s lip-sync offset regardless
of `start_index` or `overlap_seconds`.

**Investigation.** Ruled out `TrimAudioDuration` math, `start_index`
clamp, audio VAE temporal alignment, mel hop. All matched v0408
(IMAGE loop, working). The only architectural difference was the
latent-vs-image context path.

**Root cause.** `VAEEncode` returns `{"samples": t}` and implicitly
DROPS any `noise_mask` key (`nodes.py:366`). `LTXVSelectLatents`, by
contrast, PRESERVES `noise_mask` when it slices
(`latents.py:83-84`). In the LATENT loop, each iteration's context
tail carried the PREVIOUS iteration's `noise_mask`.
`LTXVAudioVideoMask` then cloned the stale mask instead of creating a
fresh all-zeros one (`ltxv_nodes.py:246`). The sampler saw corrupted
mask semantics and the "audio is fixed" invariant (mask=0) leaked
into zones the next iteration shouldn't treat as fixed.

**Structural fix.** Two new nodes encapsulate the boundary hygiene:
`LatentContextExtract` (slice tail frames + strip mask) and
`LatentOverlapTrim` (skip first N frames + strip mask). Both produce
the same shape as `VAEEncode` — no mask key — so downstream
`LTXVAudioVideoMask` always creates a fresh mask.

**Transferable lesson.** Metadata-passing custom nodes inherit
everything the source dict holds. If a replacement path skips a node
that implicitly sanitized the dict (e.g. `VAEEncode` stripping
`noise_mask`), every downstream consumer needs an explicit
equivalent. When replacing a lossy step with a lossless one, audit
what the lossy step was hiding.

### CS2: Loop-body node must handle past-end-of-data

**Context.** 10-second audio file tested against a default
`window=19.88, overlap=2.0` loop. Sampler crashed on the first
iteration's `LTXVAudioVAEEncode` with:
`RuntimeError: padding (512, 512) at dimension 2 of input [1, 2, 1]`.

**Investigation.** Traced through `AudioLoopController`'s
`should_stop` logic. `should_stop = True` on iteration 0 (because
`next_start > audio_duration`), so the loop would exit after one
iteration — correct. But the body still ran once, and
`TrimAudioDuration` produced a 1-sample waveform for the final
window's audio slice. Mel STFT needs >1024 samples.

**Root cause.** `TensorLoopClose` checks `should_stop` AFTER the loop
body executes, not before. The body ALWAYS runs at least once, and
must tolerate whatever `start_index` the controller emits on that
last-but-unused iteration.

**Structural fix.** `AudioLoopController.execute` clamps
`start_index` so at least 0.5s of audio always remains:

```python
min_audio_seconds = 0.5
max_start = max(0.0, audio_duration - min_audio_seconds)
start_index = min(start_index, max_start)
```

**Transferable lesson.** Any node that emits an index / offset /
start time into a loop body must produce a value that keeps the body
executable, even on the iteration where `should_stop` is already
true. `TensorLoopClose` doesn't skip — it runs, then checks. Defensive
clamping at the emitting node is the correct location; "the consumer
shouldn't ask for bad input" is brittle.

### CS3: Extra conditioning node corrupts the initial render

**Context.** v0407 added `LTXVConditioning` (Node 1587) between
`CLIPTextEncode` and the Extension subgraph to propagate `frame_rate`
metadata (fixing a text2video-looking initial render from missing
metadata). That fix worked for the regression it targeted, but
introduced a new symptom: the 0-19.88s initial render had zero lip
sync. Loop iterations 2-N were fine.

**Investigation.** Compared against `LTX-2_00032.json`
(known-working, 2026-04-09). Same model, same sampler, same audio,
same image — only difference was Node 1587 in the conditioning path.

**Root cause.** ComfyUI's execution engine evaluates downstream
conditioning graphs before upstream sampling. Node 1587 sat in the
conditioning path feeding the Extension subgraph AND the initial-
render sampler. Its presence caused the conditioning graph (including
the Extension) to evaluate before the initial render's sampler ran,
corrupting the audio-video cross-attention state for iteration 0.

**Structural fix.** Bypass Node 1587. Wire `Get_base_cond_pos` /
`Get_base_cond_neg` directly to Extension #843. `frame_rate` metadata
was already being added upstream (via a different `LTXVConditioning`
in the initial-render path) — the Extension didn't need its own.

**Transferable lesson.** A conditioning-path node that "looks
harmless" can change graph evaluation order. If two samplers share a
conditioning ancestry, adding a node in the shared ancestor forces
the ancestor to evaluate sooner. Cross-check against a
known-working JSON (keep copies under `internal/scratch/`) on any
conditioning-path edit; diff the execution order, not just the node
graph.

---

## Cross-references

- Prompt rules + widget guidance: `docs/guides/prompt_creation_guide.md`
- LLM system prompt (standup variant; music variant is embedded in
  the analyzer's JSON export as `llm_system_prompt`):
  `docs/reference/standup_system_prompt.md` + `scripts/analyze_audio_features.py`
- Audio analysis pipeline: `docs/guides/audio_analysis_guide.md`
- LTX 2.3 model reference: `docs/reference/ltx23_model_reference.md`
- Profiling opt-in: `docs/guides/profiling_guide.md`
- Current plan + post-phase findings: `internal/PLAN.md` (not in repo)
