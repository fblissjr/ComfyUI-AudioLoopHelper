Last updated: 2026-05-23

# Audio-reactive workflows — single-shot preview + full-length loop

Experimental. Two paired workflows for **audio-driven video**: an init image
animated so its motion tracks an audio track, using LTX 2.3's native joint
audio-video cross-attention. The audio is frozen and *drives* the visuals —
e.g. a painted heart pulsing to a drum loop, a subject moving to a beat.

The loop variant passed its render gate and is promoted to the top-level
shipped surface (`example_workflows/audio_reactive_loop.json`); the single-shot
stays an experimental tuning rig. Use the single-shot to dial in the look, then
render long-form with the loop.

---

## The two workflows

| Workflow | Generator | Role |
|---|---|---|
| `example_workflows/experimental/audio_driven_single_shot.json` | `scripts/apply_audio_driven_single_shot.py` | **Preview / tuning rig.** Loop removed, one render pass (~14 s, ~1–2 min compute). Zero cross-iteration drift. Iterate on look / prompt / knobs fast. |
| `example_workflows/audio_reactive_loop.json` (promoted to top-level) | `scripts/apply_audioreactive_loop.py` | **Full-length render.** Loop intact — auto-tracks the whole track (3 / 5 / 20 min) and the prompt schedule can evolve the visual across a set's sections. |

Both are forks of `example_workflows/audio-loop-music-video_latent.json`. The
single-shot removes the TensorLoop subsystem and reads the initial-render
latent straight into the decoder; the loop variant is topology-identical to
the canonical (passes every loop audit invariant) and only presets widgets.

**Recommended flow:** tune on the single-shot (seconds per render), then carry
the winning knob values onto the loop via its `--flag`s and render the full
piece.

---

## Inputs

Both workflows take exactly two inputs:

| Node | Input | Notes |
|---|---|---|
| `#444 LoadImage` | init image | The subject. Establishes style, palette, framing. |
| `#565 LoadAudio` | audio track | The driver. Frozen; its rhythm drives the motion. |

Put both files in your ComfyUI `input/` directory first — `LoadImage` /
`LoadAudio` read from there. Output is trimmed to the audio length and the
track is muxed into the result mp4.

**Init-image tips:**
- A stylized/painterly init drifts toward photoreal across a long render
  (cross-attention is photoreal-trained). The single-shot avoids this; the
  loop manages it via `first_frame_guide_strength` (below).
- Match your prompt's medium to the init ("oil-painted …" for a painting, not
  "glistening …" which leans photoreal).

---

## The knobs

All are preset by the generators and overridable with `--flag`s (so re-running
is deterministic — see "Regenerating"). Node-level values if you tune in the UI:

| Node | Knob | Default | What it does |
|---|---|---|---|
| `#1523 LTX2AttentionTunerPatch` | `audio_to_video_scale` (widget 3) | `2.5` | How hard the audio drives video attention. `1.0` = neutral. Raise to 3–5 if the beat coupling is too weak; watch for artifacts. |
| `#508 LTX2_NAG` | `nag_scale` (widget 0) | `5` | The canonical/KJNodes default `11` is the documented distilled freeze-risk knob. `5` is safer for a motion-first render; raise toward 7–11 only if motion is too loose. |
| `#507 CLIPTextEncode` | NAG negative text | motion + frame-quality terms | Pushes away from "still / no motion / blurry". Default drops person/singer tokens (faces / hands / mic) that don't fit a non-person subject. |
| `#1269 first_frame_guide_strength` | value (widget 0) | `0.7` (loop only; = canonical) | **Loop only.** Per-iteration init re-anchor strength = the drift-vs-motion dial. `1.0` = holds the init hard but suppresses motion; lower = more motion but more cross-iter style drift. A/B for your image (≈0.5–0.8 for a painterly init). |
| `#446 LTXVPreprocess` | `img_compression` | `35` (single-shot) | Anti frozen-frame: a pristine init reads as a static photo. Single-shot only (the loop keeps the canonical `18` to preserve init/loop preprocess symmetry). |
| `#1615 TimestampPromptScheduleBatchEncode` | schedule (widget 0) | single `0:00+:` entry | One prompt held for the whole render (both workflows). Add `M:SS+:` entries to evolve the loop per section (see below). |

---

## How to run

### Single-shot (preview)

1. Put your init image and audio in ComfyUI's `input/`.
2. Open `example_workflows/experimental/audio_driven_single_shot.json`.
3. Set `#444 LoadImage` to your image, `#565 LoadAudio` to your audio.
4. (Optional) Edit `#1615` schedule[0] — keep the `0:00+:` prefix, an action
   verb that matches the audio (`pulses` / `beats` / `dances`), and `In a
   [shot], [camera]` framing.
5. Queue. ~14 s of video, trimmed to your audio.
6. Tune `audio_to_video_scale` / `nag_scale` and re-run until the look + beat
   coupling are right.

### Loop (full-length render)

1. Same inputs.
2. Open `example_workflows/audio_reactive_loop.json`.
3. **Prompt:** the default is a single `0:00+:` entry held for the whole render.
   To evolve the visual per section, replace it with one entry per section:
   ```
   0:00+: In a tight macro close-up, <subject> <verb>s to the beat, <detail>. The camera holds steady.
   1:30+: In a medium shot, <subject> <verb>s harder as the energy builds, <detail>. The camera slowly pushes in.
   3:00+: In a wide shot, <subject> at peak intensity, <detail> bursting on each beat. The camera slowly orbits.
   ```
   Use real timestamps from your track (build / drop / breakdown). `In a
   [shot]` continuation framing — **not** "Cut to" (the model reads scene-cut
   language as a discontinuation directive).
4. Carry the `audio_to_video_scale` / `nag_scale` you settled on in preview;
   set `first_frame_guide_strength` (start `0.7`).
5. Queue. The loop auto-sizes to the audio length.

---

## Long renders (3–20 min)

- **Render per track, not one monolith.** A set is usually multiple tracks;
  rendering each and concatenating is far more manageable and resumable than
  one continuous 20-minute job.
- **Render time scales with length.** Each ~20 s window is one 8-step pass
  (~1 min). A 3-minute track is ~10 windows ≈ 10–15 min; budget accordingly.
- **Use the latent → upscale path** for efficiency: render at base resolution,
  capture the assembled latent (toggle the bypassed `SaveLatent` in the loop),
  and upscale separately via the upscale workflow rather than rendering at full
  res in one pass. See `docs/guides/upscale_guide.md`.
- **Drift over many windows** is bounded by re-anchoring but breathes; in an
  installation context some breathing of a painterly piece reads as
  intentional. Tune `first_frame_guide_strength` to taste.

---

## Regenerating / tuning via the generators

The generators bake the knobs as defaults and expose every one as a flag, so a
no-arg run reproduces the committed workflow and a flag run re-tunes it. A
second run is a no-op (protects in-UI edits, especially an authored loop
schedule); pass `--force` to regenerate.

```bash
# Single-shot, harder audio drive + a custom prompt:
uv run --group dev python scripts/apply_audio_driven_single_shot.py --force \
    --audio-to-video-scale 3.5 --prompt "0:00+: <your prompt>"

# Loop, softer anchor for more motion:
uv run --group dev python scripts/apply_audioreactive_loop.py --force \
    --first-frame-guide-strength 0.5 --audio-to-video-scale 3.0
```

`--dry-run` reports planned ops; `--revert` removes the generated file
(refuses if the target isn't the expected variant).

---

## Gate for promotion out of experimental

1. A render confirms the audio visibly drives motion (beat coupling) on the
   single-shot.
2. A full-length loop render holds style + tracks sections without unacceptable
   drift.
3. `scripts/audit_workflows.py` stays clean on both (single-shot audits as a
   loopless fork; the loop variant passes all loop invariants).
4. At least one case study (an `internal/log/` entry or a `docs/experiments/`
   note).

The loop variant met 1–3 and was promoted to `example_workflows/`; the
single-shot and this writeup stay in `docs/experimental/` pending a case
study (4).
