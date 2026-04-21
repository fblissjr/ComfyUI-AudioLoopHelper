Last updated: 2026-04-17 (post-DR1 status: decoder is now LTXVTiledVAEDecode; widget recommendations below remain valid)

> **⚠ Post-DR1 note**: Phase 1 of the blend fix and Phase DR1 of the
> decoder reliability track have both shipped since this doc was
> originally written. Decoder in all example workflows is now
> `LTXVTiledVAEDecode` (no temporal tiling, no stride invariant).
> Widget recommendations below remain correct post-DR1 (they were
> already Phase-1-aware). See `prompt_comedy4.md` for the current
> richer schedule and `docs/debugging_guide.md` for current workflow
> guidance.

Original date: 2026-04-17

# Standup Comedy — Example Prompt Schedule (v1)

Companion example to `prompt.md` / `prompt2.md` / `prompt3.md` /
`prompt4.md`, but for the **standup comedy** domain rather than music
video. Demonstrates the LLM-mediated path with a custom system prompt
(see `docs/system_prompt.md`) working around the music-centric default
`llm_system_prompt`.

## Inputs

- **Audio**: `/mnt/hub/ai/img/input/norm_trimmed.mp3` — 184s (3:04)
  standup routine. Detected: F0 129 Hz male, sections
  CHORUS/VERSE/VERSE/VERSE/CHORUS (librosa's music labels; read as
  energy proxies — loud CHORUS bookends = cold-open + closing laughter;
  medium VERSEs = dialogue bits).
- **Image**: Chicago Comedy Club — blonde male comedian in blue blazer
  with mic, orange stage lights, brick wall with neon sign, crowd of
  6-8 visible in foreground.
- **Subject string**: `a blonde male comedian in a blue blazer holding
  a microphone on a brick-wall stage with a laughing crowd in the
  foreground, Chicago Comedy Club neon sign visible behind him`
- **Tier**: `1b` (performance_live, wide stage mood bundle).
- **Montage**: off.

## Generation process

1. Ran `analyze_audio_features.py` with `--scene-diversity 1b` to get
   the audio analysis JSON.
2. Discarded the auto-embedded `llm_system_prompt` (music-specific).
3. Fed the remaining JSON + init image + creative direction to Gemini,
   using the standup system prompt from `docs/system_prompt.md`.
4. Corrected one critical issue in Gemini's output (timestamp format:
   Gemini emitted bare decimal seconds like `15.98-35.00` instead of
   M:SS; our `_fmt_ts` truncates to integer M:SS by convention) and
   one minor (harmonized the `Wide stage framing, warm stage wash.`
   mood-bundle phrase to appear in every entry, not just three).

## Final schedule (corrected)

```
node_169_prompt: Style: cinematic. In a wide establishing shot, static camera, locked off shot, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is pausing for the laugh, smiling warmly, holding the mic close to his chest. Wide stage framing, warm stage wash. A few people in the crowd are mid-laugh, settling into their seats. Ambient club sounds, a clink of glasses.

schedule:
0:00-0:15: Style: cinematic. In a wide establishing shot, static camera, locked off shot, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is pausing for the laugh, smiling warmly, holding the mic close to his chest. Wide stage framing, warm stage wash. A few people in the crowd are mid-laugh, settling into their seats. Ambient club sounds, a clink of glasses.
0:15-0:35: Style: cinematic. In a medium wide shot, static camera, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is delivering the setup, gesturing with his free hand, pacing slightly to the left. Wide stage framing, warm stage wash. The crowd is quiet, leaning in slightly. Faint chair shifting in the room. Delivery in a dry deadpan.
0:35-0:55: Style: cinematic. In a medium shot, slow dolly in, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is telling a joke, raising an eyebrow, shifting his weight. Wide stage framing, warm stage wash. One person in the foreground taking a sip from a drink. Brisk rhythmic delivery.
0:55-1:17: Style: cinematic. In a medium close-up, static camera, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is delivering the punchline, leaning into the mic with sudden energy, eyes wide. Wide stage framing, warm stage wash. The crowd silhouetted in the foreground reacting. Distant bar murmur.
1:17-1:37: Style: cinematic. In a wide shot, slight handheld sway, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is mid-bit, shaking his head, pacing back to center stage. Wide stage framing, warm stage wash. The crowd watching attentively, relaxed posture. Delivering in a low rhythmic tone. Faint hum of the venue.
1:37-1:57: Style: cinematic. In a medium shot, static camera, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is smiling wryly, pausing briefly, looking out into the audience. Wide stage framing, warm stage wash. A couple of patrons whispering to each other in the dark. Delivery leaning toward a whisper.
1:57-2:15: Style: cinematic. In a medium close-up, slow dolly in, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is telling a joke, gesturing sharply with his left hand to emphasize a point. Wide stage framing, warm stage wash. Foreground crowd members shifting their weight. Steady room presence.
2:15-2:31: Style: cinematic. In a medium shot, static camera, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is delivering the punchline, leaning back, pointing at a crowd member. Wide stage framing, warm stage wash. Someone in the foreground wiping their eye from laughing. Bright clear vocal tone.
2:31-2:53: Style: cinematic. In a medium close-up, slow jib up slightly, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is leaning into the mic, building the final premise, maintaining intense eye contact with the crowd. Wide stage framing, warm stage wash. The crowd leaning in, highly attentive. Muffled clink of a glass on a wood table. Intense, rapid delivery.
2:53+: Style: cinematic. In a wide shot, slow dolly out, camera pulling back, a blonde male comedian in a blue blazer holding a microphone on a brick-wall stage with a laughing crowd in the foreground, Chicago Comedy Club neon sign visible behind him is reacting to the crowd, waving his free hand, stepping back from the mic stand. Wide stage framing, warm stage wash. The crowd animated, shoulders shaking in the foreground silhouettes. Room tone settling quietly.
```

## Negative prompt

Paste into node 507 (the negative CLIPTextEncode) — same slot the
example workflows use.

**Baseline (standardized across all four example workflows):**

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, fourth character, twin, clone
```

**Standup-tuned (recommended for this use case):**

Add mouth-motion and identity-stability terms on top of the baseline.
The 3-minute duration + medium-energy dialogue sections are where
identity drift and mouth freezing hurt most, and the stock negative
wasn't written with sustained dialogue in mind.

```
still image with no motion, frozen mouth, locked jaw, unnatural stillness, lip sync drift, mouth not matching audio, subtitles, deformed facial features, warped face, plastic skin, identity shift, extra limbs, disfigured hands, duplicate comedian, second speaker, twin, clone, singing, musical instrument
```

Additions vs. baseline and why:

| Added term | Why |
|------------|-----|
| `frozen mouth, locked jaw, unnatural stillness` | Dialogue sections (medium RMS) are the failure mode where LTX sometimes reads the audio as "calm speech" and emits a still face. Explicit mouth-motion negatives push toward animation. |
| `lip sync drift, mouth not matching audio` | Sustained 60-70s dialogue windows are where cross-attention can desync; naming the failure mode is a cheap nudge. |
| `warped face, plastic skin, identity shift` | 3 minutes is long enough for the comedian's face to morph subtly. Kept separate from the baseline `deformed facial features` because that targets anatomy, these target identity. |
| `duplicate comedian, second speaker` | The baseline had `duplicate character, fourth character` (written for multi-subject music videos). Restate in domain terms so the model doesn't hallucinate a co-host. |
| `singing, musical instrument` | The init image anchors "comedian on stage with mic" which *could* be confused with "singer on stage with mic". Negating `singing` explicitly steers away from that interpretation, especially since our default template universe is music-centric. |

## Workflow widget values

Beyond the two prompts, set these on the example image workflow
(`example_workflows/audio-loop-music-video_image.json`). Defaults are
fine for standup; notes below explain when to deviate.

| Widget | Node | Start value | Standup note |
|--------|------|-------------|--------------|
| `window_seconds` | AudioLoopController | **19.88** | Default. Don't change unless you know why. Loop iteration length. |
| `overlap_seconds` | AudioLoopController | **2.0** | Default. Overlap between iterations. Stride = 19.88 − 2.0 = 17.88s per iteration. Bump to **3.0** *only* if you see identity twitches at iteration boundaries (rare with a static stage + sustained subject). More overlap = smoother continuity but less new content per iteration. |
| `blend_seconds` | TimestampPromptSchedule | **0.0** | **CORRECTION from v1.** Post-Phase-1 fix: values between 0 and stride_seconds (~17.88s) are auto-clamped to stride_seconds because they can't produce smooth ramps at iteration resolution. Start at 0 (hard switch) — with snapped boundaries + identical subject across entries, transitions are clean without cross-fade. Only bump to **≥ stride_seconds** (e.g. 20) if you see a visible seam at prompt boundaries, and only if the cross-fade is worth diluting shorter entries. |
| `snap_boundaries` | TimestampPromptSchedule | **True** | Leave on. Rounds schedule boundaries to the iteration grid so every iteration runs on exactly one prompt (no mixed conditioning). Turning it off re-enables the legacy spike-blend behavior and is only useful if you want sub-stride timing precision AND accept the jitter risk. |
| `start_index` (trim) | node 567 | **0** | Audio is already trimmed per filename `norm_trimmed`. If your source has applause/music intro before the routine starts, set to the seconds to skip. |
| `fps` | (LTX default) | **25** | Per LTX-2's training configs and `ltx-pipelines/README.md` — 25 fps is what the model was trained at. Don't use 24. |
| `iteration_seed` | AudioLoopController | **auto** | Let the controller advance it per iteration. |

**Why overlap=2.0 / blend=0 are the right standup starting points (Phase 1)**:

- Standup has a *static stage* + *identity-anchored comedian* + *visible
  crowd committed by the init image*. Low risk of identity drift
  between iterations → no reason to waste iteration budget on extra
  overlap.
- Prompt deltas between entries are small (framing + body beat +
  crowd state). With `snap_boundaries=True`, each iteration runs on
  ONE pure prompt, so hard switches at snapped boundaries are visually
  clean — you don't need a cross-fade to hide a mixed-conditioning
  iteration (there aren't any).
- Any `blend_seconds` in `0 < x < stride_seconds` used to produce
  single-iteration spikes that read as jitter. That mode is now
  auto-clamped to `stride_seconds` with a warning; values above
  stride_seconds produce a raised-cosine ramp across multiple
  iterations.

If you run this and see **iteration-boundary twitches** on the
comedian's face, bump `overlap_seconds` to 3.0 first (costs you ~1s
of new content per iteration in exchange for smoother identity). If
you see **visible seams at prompt-schedule boundaries** (at snapped
timestamps), try `blend_seconds = stride_seconds` (~17.88) for a
smooth cross-fade across one iteration on each side. Higher values
(up to 2-3x stride) give softer ramps but dilute the distinctness
of adjacent prompts — trade-off.

## Observations for future refinement

- **Timestamp-format drift** was the only critical LLM failure. Worth
  adding an explicit "truncate decimal seconds, never emit bare
  seconds" rule to `docs/system_prompt.md` before the next run —
  Gemini in particular tends to preserve input decimal precision unless
  told otherwise.
- **Section-label interpretation was the real test.** Gemini correctly
  read the librosa-labeled CHORUS 0:00-0:15 loud as *crowd laughter the
  comedian pauses through* (cold-open applause) and the closing CHORUS
  2:53+ as *outro applause the comedian reacts to*. This is the right
  translation of music labels → standup domain, and it's evidence the
  LLM can do the domain mapping reliably when told the labels are
  "energy proxies, not structural roles."
- **Verb pool rotation** was good: 9 unique standup verbs across 10
  entries, no repetition in adjacent entries. The pool in
  `docs/system_prompt.md` is the right size for ~10-entry runs; scale
  the pool if your routine produces more entries.
- **Crowd state variation** worked as intended (R4): rotated through
  laughter, attentive silence, sipping drinks, whispering, wiping eyes,
  shoulder shake. The crowd reads as alive without ever stealing focus
  from the comedian's singing-verb slot.
- **Subject-verb flow awkwardness** (`...behind him is pausing...`) is
  inherent to the subject string ending with a trailing clause. If
  regenerating, rephrase the subject to end with a noun (e.g.
  `...with the Chicago Comedy Club neon sign behind him on the brick
  wall`) so the performance verb attaches cleanly.
- **Mood-bundle consistency**: Gemini initially included
  `Wide stage framing, warm stage wash.` in only 3 of 10 entries. The
  music examples in `docs/analysis/llm_prompt_generation_guide.md`
  show mood bundles in **every** entry — consistent with a signature.
  Correction harmonized to all entries.

## First-class standup support (future)

If this domain becomes recurring, a `--domain {music, standup,
dialogue, lecture}` flag on `analyze_audio_features.py` would swap out
`_SECTION_MODIFIERS`, the embedded `_LLM_SYSTEM_PROMPT`, and the
section-label → prompt mapping. Standup's domain definition would
include the verb pool above + a recalibration of "loud section = laugh
peak, medium section = dialogue bit" for the label mapping. Discussed
in `internal/audio_analysis_evolution.md`.
