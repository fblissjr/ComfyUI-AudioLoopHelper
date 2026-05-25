---
name: ltx-dialogue-prompt
description: Generate an LTX 2.3 prompt that makes a character speak SPECIFIC dialogue, formatted per the official LTX 2.3 prompt guide (short quoted segments with acting directions between, accent, acoustic environment). Use when the user wants to put words in a character's mouth for the AV inversion / one-pass / keyframe workflows (voice-cloned dubbing, dialogue replacement).
---

Last updated: 2026-05-25

# LTX 2.3 Dialogue Prompt Generator

Turn "make them say X, in this scene" into a paste-ready prompt for the audio-loop
AV workflows. Output goes into `TimestampPromptScheduleBatchEncode` (the
`0:00+: ...` schedule entry).

## Usage

`/ltx-dialogue-prompt "<what you want said>" [scene context] [--accent <accent>]`

If the user gives a raw transcript, clean ASR artifacts (dupes, overlaps) first.

## Format rules (LTX 2.3 prompt guide — https://ltx.io/model/model-blog/ltx-2-3-prompt-guide)

1. **Dialogue in double quotes**, woven into narrative prose — NOT a `Dialogue:` label.
2. **Break the speech into SHORT segments** with an **acting direction between each**
   (a gesture, expression, or beat). This is the load-bearing rule — one long quoted
   block degrades; segmented + directed lands far better. The guide's own shape:
   > "...first phrase." He pauses and looks aside, then continues, "...next phrase."
   > His eyes widen. He finishes, voice cracking, "...last phrase."
3. **Specify accent / language** if it matters (e.g. `American accent`).
4. **Describe the acoustic environment + voice qualities** (e.g. `close intimate
   restaurant ambience`, `tense, animated delivery`). 2.3 rewards attention on audio.
5. Lead with a **shot + scene** clause (`In a medium shot at a dim restaurant table,`)
   so the prose has a frame to sit in.

## Output

A single schedule entry, ready to paste:

```
0:00+: In a <shot> at <place>, <subject> <action> and says "<phrase 1>." <acting
direction>, "<phrase 2>." <acting direction>, "<phrase 3>." <accent>, <delivery>,
<acoustic environment>.
```

Generic example (input: make a man at a bar say he's quitting his job):

```
0:00+: In a medium shot at a crowded bar, a tired man sets down his glass and says
"I'm done." He exhales, shaking his head, "I'm not going back there tomorrow." He
half-laughs, "thirty years, and for what?" His voice steadies, "I'm done." American
accent, weary and resigned delivery, noisy bar ambience.
```

## Which workflow this prompt feeds, and the seed caveat

- The **first ~2s (or the chosen `AudioTemporalMask` seed window)** is REAL audio from
  the source clip — it clones the voice. The prompt drives the REGENERATED rest. Pick a
  clean voice slice as the seed (set `AudioTemporalMask.invert=True` + `[start,end]` to
  grab a mid-clip window, not just the noisy opening).
- **Matching lips to NEW words requires the video to be GENERATED** (keyframe / one-pass
  variant), not frozen. On the frozen-video inversion, new words won't match the original
  lips (dub effect) — fine if you only care about the audio; use the keyframe/one-pass
  path for lip-synced output.
- Keep the prompt's quoted words = what you want HEARD; for lip-sync the generated video
  syncs to them automatically (LTX joint AV).

## Notes

- Profanity / explicit words may get sanitized or dropped by the model — keep or soften
  per taste; it's not guaranteed verbatim.
- LTX is not a TTS: it biases toward the words, nails short phrases better than long
  monologues, and won't reproduce 15s+ of exact scripted lines perfectly. Segment
  aggressively.
- Full background on the verb/cross-attention behavior: `docs/guides/prompt_creation_guide.md`.
