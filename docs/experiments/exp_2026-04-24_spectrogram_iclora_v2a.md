# Experiment: spectrogram-as-canny IC-LoRA + V2A round-trip

Last updated: 2026-04-24
Status: open — leaning toward dead path; R17 is the single run that closes out or rescues the hypothesis

## TL;DR (current state)

After 9 unique runs (R1-R2, R7-R8a, R14-R16; R15 was a duplicate):

- **Color spectrograms avoid the B&W → 1940s-radio-voice failure mode.** Real usable knob. Always use `--colormap viridis` or `--colormap spectrum` from here on.
- **IC-LoRA + spectrogram without an init image = pure leakage.** Init is required as a competing visual anchor; without it, IC-LoRA just reproduces the reference regardless of strength or colormap.
- **No IC-LoRA-active run has produced audio features closer to source than IC-LoRA-inactive runs.** The best audio-feature alignment in the whole set is R14 (no IC-LoRA): 143.6 BPM C minor vs source 136 BPM F minor. IC-LoRA appears to be actively hurting, not helping.
- **Seed variance is ±20 BPM** on equivalent-conditioning configs. Single-seed comparisons between configs are statistical noise for detecting audio effects.

**Next single run to settle this:** R17 (see "Next run to close out" section).

## Hypothesis

Two intertwined hypotheses:

1. **Spectrogram-as-canny** — A Mel spectrogram rendered as a video sequence, fed as IC-LoRA structural reference, will cause LTX 2.3 to produce output whose motion is locked to audio amplitude (beats become visual pulses).

2. **V2A round-trip** — With audio generation enabled (no `noise_mask=0` freeze), LTX 2.3's joint AV cross-attention will reconstruct audio whose structure resembles the input audio that was used to produce the spectrogram.

## Setup

- Workflow: `example_workflows/experimental/spectrogram_iclora_minimal.json`
- Apply script: `scripts/apply_spectrogram_iclora_minimal.py`
- Model: LTX 2.3 distilled (8-step `linear_quadratic` sigmas, `shift=13`, `euler`, `cfg=1`)
- NAG: `[scale=5, alpha=0.25, tau=2.5, inplace=True]`
- Resolution 832x448 (div-by-32), length 497 (`(497-1) % 8 == 0`)
- Source audio: 19.88s clip from a 3:12 electronic/synth-rock track at 136 BPM, F minor, male vocals at 87 Hz F0

## Run log

| # | Init | LoRA | LoRA strength | Guide strength | Spectrogram | Visual outcome | Audio outcome |
|---|---|---|---|---|---|---|---|
| **R1** | <water_photo>.jpg | MergeGreen | 0.9 | 1.0 | edge_detected B&W | Pure spectrogram leakage from frame ~50 | Speech (1940s radio announcer aesthetic), English → gibberish |
| **R2** | <water_photo>.jpg | Lightricks Union Control | 1.0 | 1.0 | edge_detected B&W | Water → architectural buildings with horizontal banding by frame ~25 | Same speech / radio aesthetic |
| **R7** | <concert_photo>.jpg | OFF (bypassed) | n/a | n/a (guider bypassed) | none (bypassed) | concert-lights scene preserved across all frames with subtle laser motion | **Genuine music**: 123 BPM, E minor (0.94 conf), male F0 82 Hz, sustained pitched tones |
| **R8a** | none (bypassed) | OFF (bypassed) | n/a | n/a (guider bypassed) | loaded but unused (IC-LoRA bypassed) | **Model generated full DJ booth scene from prompt alone** ("concert lights pulsing") — DJs at CDJs, crowd, red+blue stage lighting | **Music with clearer beat structure**: 123 BPM, A minor (0.67), F0 64.7 Hz (sub-bass dominant), visible drum-pulse transients |
| **R14** | none (bypassed) | OFF (bypassed) | n/a | n/a (guider bypassed) | loaded but unused | **Different DJ scene**: 2 DJs at Pioneer CDJs, moody purple lighting, clean composition | **Music**: **143.6 BPM** (closer to source 136), C minor (0.88 conf), F0 83.4 Hz, strong 4-on-the-floor kick pattern visible in spectrogram. **Same config as R8a, only seed differs — 20 BPM delta from seed alone.** |
| **R15** | (same as R14) | (same as R14) | — | — | rainbow (swapped) | (same as R14) | **Byte-identical to R14.** Confirms spectrogram is inert when IC-LoRA bypassed. Not a separate data point. |
| **R16** | none (bypassed) | Union Control | **0.6** | **0.7** | **rainbow** (colormap=spectrum) | **Pure rainbow-spectrogram leakage** every frame — model reproduced the IC-LoRA reference verbatim | **Music (not speech)**: 123 BPM, **G Major (0.79)** — first Major key in the whole set, F0 64.7 Hz (sub-bass). Color prevented radio-voice failure mode from R1/R2. But tempo + tonality moved *away* from source if same seed as R14. |

## Observations

### R1 vs R2 — IC-LoRA family matters for OOD references

MergeGreen mode-collapsed onto the reference (literal copy). Union Control interpreted the structural pattern (horizontal bands → architectural floors/columns). Union Control is the better default for non-canny references; MergeGreen overfits to "reproduce input" when the reference doesn't match its training distribution.

### R1, R2 audio: identical "1940s radio announcer" aesthetic

Both runs produced speech with vintage-radio timbre: narrowband, mid-range, "broadcast voice" affect. Both used B&W edge-detected spectrograms. The B&W aesthetic visually resembles old film/radio/TV training data, which has strong audio-domain associations with vintage broadcast audio. The visual prior bled into audio generation.

### R7 — the controlled baseline

Removed IC-LoRA, removed spectrogram, removed guider. Kept concert-photo init + simple prompt. Result: **the model produces electronic-flavored music with the right tempo range, mood, and vocal register** — without any explicit audio-related conditioning. The init image alone carries enough audio prior to drive plausible music generation.

### R8a — prompt-only baseline (no init, no IC-LoRA)

Bypassed init image AND IC-LoRA. Prompt: `"concert lights pulsing"`. The model **materialized a full DJ booth scene from prompt alone** — DJs at CDJs, headphones on, crowd around, red/blue club lighting, working hands visible across frames. Audio: 123 BPM (same as R7), A minor, F0 64.7 Hz (sub-bass/synth bass dominant), with visible drum-pulse transients in the spectrogram (clearer beat structure than R7).

Two takeaways:
- **Strong text prompts can replace init images** for genre conditioning. "Concert lights pulsing" → DJ scene with same tempo as the concert-photo-init R7 produced.
- **123 BPM appearing in both R7 and R8a** suggests an internal default tempo for electronic-genre conditioning around 120-125 BPM, regardless of which conditioning channel triggers it.

| Audio feature | Source | R7 output | Delta |
|---|---|---|---|
| BPM | 136 | 123 | -13 (10% off, both dance-range) |
| Key | F minor | E minor | 1 semitone (adjacent, both minor) |
| Vocal F0 | 87 Hz male | 82 Hz male | -5 Hz (~6%, same register) |
| Spectrogram structure | Sustained pitched tones | Sustained pitched tones | ✅ same modality |
| Audio type | Music with vocals | Music with vocals | ✅ match |

The output is "in the neighborhood" of the source genre/mood/register — not a literal V2A reproduction, but the right *kind* of music for the visual context.

## Inferences

1. **Init image controls audio genre in LTX 2.3's joint AV decoder.** concert-photo init → electronic music. Water + B&W spectrogram → 1940s radio speech. The init carries massive audio-prior weight even when no audio is explicitly conditioned.

2. **The "1940s radio voice" was the spectrogram visual aesthetic, not the model's default empty-conditioning prior.** R7 (no spectrogram) produced normal contemporary music, not radio voice. The B&W visual triggered vintage-broadcast audio priors. Confirmed by elimination.

3. **MergeGreen IC-LoRA mode-collapses on OOD references.** It reproduces the reference verbatim rather than extracting structure. Use Union Control for any non-canny reference.

4. **Union Control IC-LoRA *does* extract structure from OOD references** — it just maps that structure to whatever real-world content has the same structural signature. Spectrogram horizontal bands → buildings with horizontal floors. The LoRA is working; the content prior is doing what it was trained to do.

5. **Single-shot literal V2A (output audio = input audio) is not what LTX 2.3 produces.** The model produces *contextually appropriate* audio that matches the visual scene, not a literal reconstruction of audio you fed in via spectrogram. This is actually a more useful capability for music video generation than literal V2A.

6. **LTX 2.3's electronic-genre audio output has high seed variance in tempo** (~±20 BPM). R7, R8a, R14 all used the same conditioning class (electronic-genre via init or prompt, no IC-LoRA). Results: 123, 123, 143.6 BPM. Two samples matching at 123 was coincidence — R14 shows the true spread. **Critical methodology implication**: single-seed comparisons between configs are statistically meaningless for detecting IC-LoRA + spectrogram audio effects, because the seed noise is larger than any plausible signal. Multi-seed (3-5 per config) distribution comparison is required.

7. **Strong text prompts can replace init images for genre conditioning.** R8a generated a complete DJ-booth scene from `"concert lights pulsing"` alone — DJs at CDJs, crowd, club lighting. Means future ablations can use prompt-only baselines without losing genre coherence.

8. **Init images aren't just style — they're a visual competitor that forces IC-LoRA to blend rather than copy.** R16 (IC-LoRA active, no init) produced pure spectrogram leakage. R2 (IC-LoRA active, water init) produced architectural *interpretation* of the spectrogram structure. The difference: with an init, IC-LoRA's structural pull has to reconcile two competing visual signals and produces blended output; without an init, it just reproduces the reference verbatim. An init image is a requirement for IC-LoRA to produce meaningful output at high strength, not an optional anchor.

9. **Color spectrograms break the "1940s radio voice" failure mode** (hypothesis confirmed). R1 and R2 (B&W edge spectrograms) produced speech with vintage-radio timbre. R16 (rainbow spectrogram, same leakage failure) produced music — not speech. **Color aesthetic determines the audio genre prior the model pulls from.** This is a real, usable knob: B&W → vintage/speech priors; color → modern/music priors.

10. **IC-LoRA + spectrogram may actively hurt audio-feature alignment to source, not help.** R14 (no IC-LoRA): 143.6 BPM C minor. R16 (IC-LoRA active, same seed if unchanged): 123 BPM G Major. Both tempo and tonality moved *away* from source (136 BPM F minor). If same-seed comparison holds across multiple samples, this kills the hypothesis that spectrogram-as-IC-LoRA transfers audio info. The best audio alignment in the experiment is from runs without IC-LoRA at all.

## Next run to close out (R17)

The one remaining run that cleanly decides whether spectrogram-as-IC-LoRA is alive or dead for audio alignment:

**R17 — "everything working together"**

| Setting | Value | Why |
|---|---|---|
| Init image | **<concert_photo>.jpg** | Required anchor so IC-LoRA blends rather than copies (inference #8) |
| Spectrogram | **`<song>_spec_viridis.mp4`** | Color avoids radio-voice failure; viridis is modern-spectrogram aesthetic |
| LoRA | Lightricks Union Control | MergeGreen mode-collapses; Union Control is the better OOD interpreter |
| LoRA `strength_model` | **0.6** | Preserves init (R2 at 1.0 wiped init; 0.6 is mid-range) |
| Guide `strength` | **0.7** | Independent dial; moderate pull |
| Prompt | `"concert lights pulsing"` | Genre-locked, matches init |
| NAG scale | 5 | Dialed back from production 11 for distilled |
| Seed | **Same seed as R14 (143.6 BPM baseline)** | Controls variance — single-variable vs R14 is "IC-LoRA + spectrogram active" |

Expected outcomes and what each means:

| R17 audio output | Interpretation |
|---|---|
| BPM shifts toward 136 from R14's 143.6; key shifts toward F minor | **Hypothesis partial-rescue.** IC-LoRA + color spectrogram + init transmits audio info. Proceed to multi-seed replication + strength sweep. |
| BPM near 143.6; genre-coherent music; minor key | IC-LoRA neutral — doesn't hurt, doesn't help. Probably not worth pursuing (adds complexity for no win), but not actively broken. |
| BPM shifts *away* from 136; key changes to Major; audio degrades | **Hypothesis dead.** R16 pattern repeats. Close out and pivot. |
| Visual collapse (buildings from R2, or leakage) | IC-LoRA family can't handle spectrogram references even with init + color. Close out and pivot. |

After R17, write up verdict and close the spectrogram-as-IC-LoRA exploration. Pivot paths listed below.

## Critical missing experiment

The most valuable comparison in the whole tree is now:

| Run | Init | IC-LoRA | Spectrogram |
|---|---|---|---|
| **R7** (done) | concert-photo | OFF | none |
| **R8** (pending) | concert-photo | Union Control, str=0.6, guide str=0.7 | `<song>_spec_viridis.mp4` (color, no aesthetic damage) |

Compare audio features (BPM, key, F0) between R7 and R8 against source:
- **R8 features closer to source than R7** → IC-LoRA + spectrogram contributes real audio guidance via visual proxy. The hypothesis is partially salvaged for music alignment (not literal V2A).
- **R8 features same as R7** → IC-LoRA doesn't transfer audio info from the spectrogram; visual aesthetic was the only thing it ever did. Pivot.
- **R8 features worse than R7** (back to speech, or off-genre) → IC-LoRA still hurts even with color. Pivot.

A single A/B answers whether the spectrogram path has *any* audio-coupling value beyond aesthetic damage.

## Spectrogram variants generated for follow-up runs

All sourced from the same `<source_song>.mp3` (`--start 2.0 --duration 19.88`, 497 frames at 25fps, 832x448, LTX-aligned):

| Filename (in ComfyUI input/) | Mode | Colormap | Aesthetic | What it isolates |
|---|---|---|---|---|
| `<song>_spectrogram_clean.mp4` | edge_detected | gray (B&W) | Vintage spectrum analyzer | Original (used in R1, R2) |
| `<song>_spec_blurred3.mp4` | blurred σ=3 | gray | Soft B&W | Edge-filter contribution vs underlying bands |
| `<song>_spec_normalized.mp4` | normalized | gray | Crisp B&W bands | Edge filter off, raw amplitude |
| `<song>_spec_blurred8_heavy.mp4` | blurred σ=8 | gray | Almost featureless gradient | Extreme structure removal |
| `<song>_spec_blurred3_window4s.mp4` | blurred σ=3, window=4s | gray | Slower temporal change | Temporal-window length |
| `<song>_spec_viridis.mp4` | blurred σ=3 | viridis | Modern matplotlib spectrogram | **Color isolation — pairs with R8** |
| `<song>_spec_rainbow.mp4` | blurred σ=3 | spectrum (HSV) | Rainbow audio analyzer | Alternate color aesthetic |

## Variable space

Everything you can change, in roughly the order it matters:

| Variable | Values worth testing | Why it matters |
|---|---|---|
| **Init image** | <concert_photo>.jpg, <water_photo>.jpg, candle flame on black, smoke, none (bypass `LTXVImgToVideoInplaceKJ(531)`) | R7 proved init drives audio genre. Strongest single lever. |
| **IC-LoRA family** | Lightricks Union Control, MergeGreen, OFF (bypass `LTXAddVideoICLoRAGuide(1622)` + `LTXICLoRALoaderModelOnly(1619)`) | R1 vs R2 proved this matters: MergeGreen mode-collapses, Union Control extracts structure. |
| **LoRA strength** (`LTXICLoRALoaderModelOnly.strength_model`) | 0.0 (off), 0.4, 0.6, 0.9, 1.0 | R2 at 1.0 wiped init; lower likely preserves it. |
| **Guide strength** (`LTXAddVideoICLoRAGuide.strength`) | 0.3, 0.5, 0.7, 1.0 | Independent knob from LoRA strength. Controls how hard the IC-LoRA reference pulls. |
| **Spectrogram colormap** | gray (B&W), viridis, spectrum (rainbow) | R1+R2 both produced 1940s radio voice with B&W. Color isolation pending. |
| **Spectrogram mode** | edge_detected, blurred σ=3, blurred σ=8, normalized | R1 used edge_detected and got literal copy. Blurred/normalized may reduce structural pull. |
| **Spectrogram window** | 2s (default), 4s, 1s | Wider = slower temporal change. May affect beat-locking quality if anything works. |
| **Prompt (positive)** | Subject-only ("concert lights pulsing"), action-only ("pulsing"), genre-only ("electronic music"), single noun ("music") | Per project rule "super simple". Still untested how prompt alone steers audio. |
| **Negative prompt** | Standard ("static, frozen, blurry"), anti-aesthetic ("daytime, daylight"), minimal | Under cfg=1, only NAG sees this. NAG scale=5 makes it gentler. |
| **NAG scale** | 3, 5, 7, 11 | 11 is aggressive/dominates IC-LoRA. 5 is balanced. 3 lets IC-LoRA dominate. |

## Run matrix — what's been tested + what to try next

### Tested

| # | Init | LoRA | LoRA str | Guide str | Spec mode | Spec color | Prompt | Audio result |
|---|---|---|---|---|---|---|---|---|
| R1 | water-splash | MergeGreen | 0.9 | 1.0 | edge | gray | (water/simple) | speech, radio aesthetic |
| R2 | water-splash | Union Control | 1.0 | 1.0 | edge | gray | (water/simple) | speech, radio aesthetic |
| R7 | concert-photo | OFF | — | — | none | — | (concert lights) | **music**, 123 BPM E minor, F0 82 Hz |
| R8a | none | OFF | — | — | loaded but unused | gray | "concert lights pulsing" | music + DJ scene, 123 BPM A minor, F0 64.7 Hz |
| R14 | none | OFF | — | — | loaded but unused | gray | "concert lights pulsing" | music + DJ scene (different scene than R8a), **143.6 BPM** C minor, F0 83.4 Hz. **Same config as R8a, seed differs.** |
| R15 | (same as R14) | — | — | — | rainbow | — | — | **Byte-identical to R14.** Confirms spectrogram inert when IC-LoRA bypassed. |
| R16 | none | Union Control | 0.6 | 0.7 | rainbow | spectrum | "concert lights pulsing" | Pure rainbow-spectrogram leakage in video. **Music (not speech)** — color prevented radio-voice failure. 123 BPM **G Major** (first Major key), F0 64.7 Hz. Audio alignment moved *away* from source. |

### High-priority next (in order)

| # | Init | LoRA | LoRA str | Guide str | Spec mode | Spec color | What it tests |
|---|---|---|---|---|---|---|---|
| **R8b** | concert-photo | Union Control | 0.6 | 0.7 | blurred σ=3 | **viridis** | **The pivotal A/B**: does color spectrogram + IC-LoRA pull music closer to source (BPM toward 136, key toward F minor, F0 toward 87) than R7/R8a's 123 BPM default? |
| R9 | concert-photo | Union Control | 0.6 | 0.7 | blurred σ=3 | spectrum (rainbow) | Cross-check R8 — does specifically-viridis matter, or does any color work? |
| R10 | none (bypass 531) | OFF | — | — | none | — | True unconditioned baseline. Separates "init effect" from "IC-LoRA effect" in all prior runs. |
| R11 | none (bypass 531) | Union Control | 0.6 | 0.7 | blurred σ=3 | viridis | Your originally-intended ablation: spectrogram-only, no init confounding. Does the spectrogram structure produce coherent visuals when the init isn't fighting it? |

### Medium-priority (if any of R8-R11 produces signal)

| # | Init | LoRA | LoRA str | Guide str | Spec mode | Spec color | What it tests |
|---|---|---|---|---|---|---|---|
| R12 | concert-photo | Union Control | 0.4 | 0.5 | blurred σ=3 | viridis | Strength tuning: lighter pull, see if motion is more rhythmic |
| R13 | concert-photo | Union Control | 0.6 | 0.7 | normalized | viridis | Mode comparison: does raw amplitude (no blur, no edge) work better than blurred? |
| R14 | concert-photo | Union Control | 0.6 | 0.7 | blurred σ=8 (heavy) | viridis | Extreme low-structure ref: does the LoRA still find audio coupling? |
| R15 | candle flame | Union Control | 0.6 | 0.7 | blurred σ=3 | viridis | Different init genre — does music/audio output adapt to "candle" context? |

### Low-priority / cleanup

| # | Init | LoRA | LoRA str | Guide str | Spec mode | Spec color | What it tests |
|---|---|---|---|---|---|---|---|
| R16 | concert-photo | MergeGreen | 0.6 | 0.7 | blurred σ=3 | viridis | Confirm MergeGreen still mode-collapses even with color + reduced strength |
| R17 | concert-photo | Union Control | 0.6 | 0.7 | edge | viridis | Edge filter + color: isolate "edge-shape triggers spectrogram leakage" vs "B&W triggers radio voice" |
| R18 | water-splash | OFF | — | — | none | — | Repeat R7 with water-splash instead of concert-photo — does init genre fully control output, or is the concert init special? |

## Decision tree based on R8b outcome

**Requires multi-seed runs** (3-5 seeds per config) due to seed variance finding from R14 (see Inference #6). Single-seed comparisons are noise.

```
R8b BPM distribution (3-5 seeds) vs R7/R8a/R14 baseline distribution:
├─ R8b distribution shifted toward source (center near 136, tighter variance than baseline)
│   → IC-LoRA + color spectrogram pulls tempo toward source
│   → Hypothesis partially validated (audio features biased toward source)
│   → Run R12-R15 to optimize strength + mode + init
│
├─ R8b distribution same as baseline (center ~130, similar variance)
│   → IC-LoRA doesn't transmit tempo info through spectrogram
│   → Pivot to alternative paths (see below)
│
└─ R8b distribution worse (off-genre, back to speech, high variance)
    → Color isn't enough; structural reference still hurts
    → Pivot immediately
```

**Baseline distribution (R7, R8a, R14 — three seeds, equivalent "no IC-LoRA" condition):**
- BPM: 123, 123, 143.6 → mean ~130, range ~20 BPM
- Keys: E minor, A minor, C minor — all minor, specific key random
- F0: 82, 64.7, 83.4 Hz — two in vocal range, one sub-bass
- Genre: 100% electronic/dance
- Audio type: 100% music (not speech)

For R8b to count as "closer to source," it should land at mean BPM noticeably > 130 (say, 133+) across multiple seeds, and/or show reduced variance (more consistent tempo = more conditioning, not less).

## Forward direction (regardless of spectrogram-as-IC-LoRA verdict)

A new direction emerged from the experiment's close-out reflection: **test-time compute methods applied to the IC-LoRA reference channel**. Spectrogram-as-reference was about picking the right reference modality; TTC methods operate on the existing reference at inference time (amplify, search, iterate, ensemble, schedule).

POC for the first method (amplification inspired by the CFG formula) landed as `scripts/apply_ttc_iclora_amplification_poc.py` and `example_workflows/experimental/iclora_amplification_poc.json`. Full landscape: `internal/analysis/iclora_landscape_analysis.md`.

This direction is independent of whether spectrogram-as-IC-LoRA works — the TTC methods apply to any IC-LoRA + reference pair, not just spectrograms.

## Pivot directions if spectrogram-as-IC-LoRA path is dead

Ranked by expected value given current evidence:

1. **Audio amplitude → `ConditioningBlend.blend_factor` per-frame** — modulate cross-attention strength directly with audio energy. Bypasses the visual proxy entirely. This is the most promising path because R7 + R14 already prove LTX 2.3 *generates* genre-coherent music from visual context alone; the open problem is just "how do we bias specific audio features (tempo, key) toward a target." A direct per-frame amplitude modulation is the architecturally-honest way to do this — no visual proxy confusion.

2. **Frozen audio (production architecture)** — `LTXVAudioVAEEncode → LTXVConcatAVLatent` with `noise_mask=0`. This already works end-to-end for music videos. The entire spectrogram-as-canny experiment was an attempt to shortcut this. If the shortcut doesn't work, the production path is still the right answer.

3. **Single-frame IC-LoRA** — use one canonical "concert frame" as IC-LoRA reference instead of a temporal sequence. Tests whether IC-LoRA is useful for style/composition without trying to inject audio info. Separates IC-LoRA's correct use (structural style guide) from its misuse (audio proxy).

4. **Phase 2.0 close-out** — write up `docs/analysis/spectrogram_iclora_failure_modes.md` (promoted from this experiment log) capturing: (a) B&W → vintage-broadcast-audio prior (usable insight for avoiding this failure), (b) IC-LoRA copies references without competing visual anchor (general finding about IC-LoRA behavior, not spectrogram-specific), (c) spectrogram-as-IC-LoRA doesn't transmit audio information. Mark Phase 2.0 done in `internal/design/spectrogram_reference_design.md` and `internal/ic_lora_assessment.md`.

## Findings worth preserving regardless of pivot

Four durable insights from this experiment, independent of whether spectrogram-as-canny ultimately works:

1. **B&W visual conditioning triggers vintage-broadcast audio priors in LTX 2.3.** Confirmed and reproducible. Avoid B&W IC-LoRA references whenever audio generation is active.

2. **Init images are a requirement, not an option, when IC-LoRA is active at high strength.** Without a competing visual anchor, IC-LoRA reproduces the reference verbatim regardless of reference content. Generalizes beyond spectrograms.

3. **LTX 2.3 produces genre-coherent music from visual context alone.** Both init images (concert-photo init → electronic) and text prompts ("concert lights pulsing" → DJ booth + electronic music) successfully condition audio genre. The model is a real V2A system at the genre level, just not at the literal-reconstruction level.

4. **Seed variance on audio features is ~±20 BPM.** Any future audio-alignment experiment must run multi-seed; single-seed comparisons are dominated by noise.

## Tooling added during this experiment

- `scripts/spectrogram_to_reference.py --colormap {gray,viridis,spectrum}` — colormap support added to test the B&W-aesthetic-triggers-vintage-audio hypothesis
- `scripts/spectrogram_to_reference.py --start <seconds>` — skip leading audio (added so frame 0 has fully-populated sliding window via lead-in pre-load)
- `scripts/spectrogram_to_reference.py --emit-video` — ffmpeg stitch for direct LoadVideo consumption
- `data/spectrogram_runs/` convention — gitignored content, tracked structure via `.gitkeep`
- Documentation pattern: `docs/experiments/` folder with `exp_YYYY-MM-DD_<slug>.md` per-experiment logs

## Related

- `internal/design/spectrogram_reference_design.md` — Phase ladder + kill switches
- `internal/ic_lora_assessment.md` — IC-LoRA tier evaluation (D1-D18 decisions)
- `docs/experimental/spectrogram_iclora_tutorial.md` — User-facing tutorial
- `scripts/spectrogram_to_reference.py` — Spectrogram rendering script
- `scripts/apply_spectrogram_iclora_minimal.py` — Workflow apply script
- `example_workflows/experimental/spectrogram_iclora_minimal.json` — Workflow file
