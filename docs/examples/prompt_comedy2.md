Last updated: 2026-04-25 (audio descriptors and vocal-delivery ornaments stripped per `docs/guides/prompt_creation_guide.md`; pre-v4 conventions preserved structurally as part of the historical evolution arc.)

> **⚠ Post-DR1 note**: Phase DR1 shipped after this doc was written.
> Decoder in all example workflows is now `LTXVTiledVAEDecode`
> (spatial-only tiling, no stride-alignment concern). Widget
> recommendations below remain correct post-DR1. See
> `prompt_comedy4.md` for the current richer schedule with camera
> variety.

Original date: 2026-04-17

# Standup Comedy — Example Prompt Schedule (v2)

Second iteration on `internal/prompt_comedy1.md`, with two substantial
changes:

1. **New init image** — `Performing_standup_at_202604171215.png`
   (Chicago standup club, "LIVE STANDUP CHICAGO" neon sign). The new
   framing is much tighter on the comedian than v1's image. That tight
   framing is critical for lip sync: LTX 2.3's audio-video
   cross-attention works better when the face is large and stable in
   frame. The v1 image was too wide and zoom-outs were shrinking the
   face across iterations.

2. **Post-Phase-1 fixes** — all boundaries pre-snapped to the iteration
   grid (stride = 17.92s); subject string byte-exact across every
   entry; no mid-window camera moves (static camera throughout except
   the final OUTRO dolly-out); dolly-out reserved for the final entry
   only per R7. `node_169_prompt` now explicitly provided, byte-exact
   to the first schedule entry.

## Inputs

- **Audio**: `/mnt/hub/ai/img/input/norm_trimmed.mp3` — 184s (3:04)
  standup routine. Already intro-trimmed (`norm_trimmed` in filename).
  **Note: set node 567's `start` to `0`** for this audio — the
  workflow default is 5, which would double-trim.
- **Image**: `Performing_standup_at_202604171215.png`. Comedian with
  short spiky light-brown hair in a navy-and-gray horizontal-striped
  quarter-zip sweater, holding a microphone. Blue neon "LIVE STANDUP
  CHICAGO" sign on the brick wall left of him. A small crowd visible
  on the right in the dim. Tight framing — comedian fills the center
  of the frame.
- **Subject string (byte-exact across all entries)**: `a male comedian
  with short spiky light-brown hair wearing a navy-and-gray
  horizontal-striped quarter-zip sweater, holding a microphone at a
  brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign
  behind him on the left and a small crowd visible on the right in
  the dim`
- **Tier**: `1b` (performance_live, wide-stage mood bundle — though the
  mood-bundle phrase is dropped from this version since "wide stage"
  conflicts with the tight init image).
- **Montage**: off.

## Why v2 differs from v1 beyond the image change

v1's Gemini output had several structural issues surfaced by the actual
run and caught on re-review. All addressed here:

| Problem in v1 | Fix in v2 |
|---|---|
| Prompts included `"wide shot"` / `"wide stage framing"` that fought the tight init image, pushing the model to zoom out and shrink the face — hurting lip sync | All entries use `"medium shot"` or `"medium close-up"`. No `"wide"`. `"wide stage framing, warm stage wash"` mood-bundle boilerplate removed |
| `"slow dolly in"`, `"slight handheld sway"`, `"slow jib up"` INSIDE iteration windows caused face size / position to change frame-to-frame within a single sampler call, destabilizing mouth pixels for cross-attention | All non-final entries are `"static camera, locked off shot"`. Only the final OUTRO entry uses `"slow dolly out, camera pulling back"` per R7's fade-out exception |
| Entry 1's subject string was completely different from entries 2-10 ("spiky blonde hair, quarter-zip sweater" vs. "blue and gray striped shirt, laughing crowd in the foreground") — identity drift guaranteed at the 0:15 boundary | Subject is byte-exact identical across all 11 entries including node_169_prompt |
| No `node_169_prompt` emitted — Node 169 would have had some other prompt than the first schedule entry, re-introducing the ~20s seam | `node_169_prompt` provided and byte-exact to the 0:00-0:17 entry |
| Timestamps at `0:15, 0:35, 0:55, 1:17, 1:37, 1:57, 2:15, 2:31, 2:53` — not on the iteration grid. Runtime snap-to-grid would have collapsed the 2:15 and 2:31 entries into one (dropping the punchline-and-reaction entry) | Pre-snapped to `0:00, 0:17, 0:35, 0:53, 1:11, 1:29, 1:47, 2:05, 2:23, 2:41, 2:58+`. No collapsed entries |
| `"laughing crowd in the foreground"` didn't match the image (crowd is on the right, not foreground) | `"a small crowd visible on the right in the dim"` — matches actual image composition |
| Double-space in `"foreground,  neon sign"` — LLM templating sloppiness | Cleaned up |

## Final schedule (corrected)

```
node_169_prompt: Style: cinematic. In a medium shot, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is pausing for the laugh, smiling brightly, holding the mic close to his chest. Warm stage wash. A few crowd members on the right mid-laugh.

schedule:
0:00-0:17: Style: cinematic. In a medium shot, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is pausing for the laugh, smiling brightly, holding the mic close to his chest. Warm stage wash. A few crowd members on the right mid-laugh.
0:17-0:35: Style: cinematic. In a medium shot, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is delivering the setup, gesturing with his free hand, keeping a steady stance. The crowd on the right leaning in, quiet.
0:35-0:53: Style: cinematic. In a medium shot, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is telling a joke, raising an eyebrow, shifting his weight slightly. One person on the right taking a sip from a drink.
0:53-1:11: Style: cinematic. In a medium close-up, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is delivering the punchline, leaning into the mic with sudden energy, eyes wide. The crowd on the right reacting, shoulders moving with laughter.
1:11-1:29: Style: cinematic. In a medium shot, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is mid-bit, shaking his head, relaxed posture. The crowd on the right watching attentively.
1:29-1:47: Style: cinematic. In a medium shot, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is smiling wryly, pausing briefly, looking out into the audience. A couple of patrons on the right whispering to each other in the dark.
1:47-2:05: Style: cinematic. In a medium close-up, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is telling a joke, gesturing sharply with his left hand to emphasize a point. Crowd members on the right shifting their weight.
2:05-2:23: Style: cinematic. In a medium shot, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is delivering the punchline, leaning back slightly, pointing at a crowd member on the right. Someone on the right wiping their eye from laughing.
2:23-2:41: Style: cinematic. In a medium close-up, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is leaning into the mic, building the final premise, maintaining intense eye contact with the crowd. The crowd on the right leaning in, highly attentive.
2:41-2:58: Style: cinematic. In a medium shot, static camera, locked off shot, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is delivering the final punchline, smiling wide, mic lowered slightly. The crowd on the right fully laughing now.
2:58+: Style: cinematic. In a wide shot, slow dolly out, camera pulling back, a male comedian with short spiky light-brown hair wearing a navy-and-gray horizontal-striped quarter-zip sweater, holding a microphone at a brick-wall stage with a blue neon "LIVE STANDUP CHICAGO" sign behind him on the left and a small crowd visible on the right in the dim, is reacting to the crowd, waving his free hand, stepping back from the mic stand. The crowd on the right animated, shoulders shaking.
```

## Negative prompt (unchanged from v1)

Same standup-tuned negative as v1. See
`internal/prompt_comedy1.md#negative-prompt` for the rationale.

```
still image with no motion, frozen mouth, locked jaw, unnatural stillness, lip sync drift, mouth not matching audio, subtitles, deformed facial features, warped face, plastic skin, identity shift, extra limbs, disfigured hands, duplicate comedian, second speaker, twin, clone, singing, musical instrument
```

## Workflow widget values

Mostly unchanged from v1 — with **one important correction**: node 567
should be `start=0` for this audio file (it's pre-trimmed). The
workflow's shipped default of `5` would double-trim and miss the first
5 seconds of the routine.

| Widget | Node | Value | Standup note |
|--------|------|-------|--------------|
| `window_seconds` | AudioLoopController (688 constant) | **19.88** | Default. Don't change. |
| `overlap_seconds` | AudioLoopController | **2.0** | Default. Bump to **3.0** *only* if identity twitches at iteration boundaries remain after fixing the v1 issues. |
| `blend_seconds` | TimestampPromptSchedule | **0.0** | Default. With snap_boundaries=True + byte-exact subject, transitions are clean. Sub-stride values are auto-clamped. |
| `snap_boundaries` | TimestampPromptSchedule | **True** | Default. Leave on. Schedule above is already pre-snapped so no runtime snapping needed, but this belt-and-suspenders is safe. |
| **`start` (trim)** | **node 567 TrimAudioDuration** | **0** *(CHANGED from workflow default 5)* | Audio is already trimmed per `norm_trimmed` filename. The workflow ships with `start=5` intended for untrimmed sources. |
| `duration` (outer trim) | node 567 | **300** | Default. Keeps the full remaining audio. |
| `start` (window trim) | node 601 TrimAudioDuration | **0** | Unchanged. Initial render grabs from t=0 of the (outer-trimmed) audio. |
| `duration` (window trim) | node 601 | *linked to window_seconds (19.88)* | The widget shows `10` but is overridden by the link. Don't edit the widget; adjust window_seconds constant instead if you ever need to. |
| `fps` | (LTX default) | **25** | Per LTX-2's training configs (all set `frame_rate: 25.0`) and `ltx-pipelines/README.md` quickstart. The workflow's Node 344 `length=497` is sized to 19.88 s × 25 fps. Don't change to 24. |
| `sampler` | KSamplerSelect (node 154) | **euler** | CHANGED from `euler_ancestral` per previous session's findings — deterministic euler reduces inter-iteration drift. |

## Lip-sync-specific reasoning

The user's v1 run had poor lip sync specifically because:

1. The v1 image had the comedian smaller in frame (the "Chicago Comedy
   Club" neon-sign image with crowd dominating the bottom third).
2. Several v1 schedule entries pushed `"wide shot"` and `"wide stage
   framing"`, making the model zoom out and further shrink the face.
3. In-window camera moves (`dolly in`, `jib up`, `dolly out`)
   repositioned/rescaled the face across the 18-20s sampler window,
   forcing the model to re-establish mouth geometry frame-by-frame
   within a single generation pass.

v2 addresses all three:

1. Tighter init image (new picture) starts the face large.
2. Every entry calls for `"medium shot"` or `"medium close-up"`, never
   wide — keeps face large across the full run.
3. Every entry is `"static camera, locked off shot"` — face stays at
   the same size/position across the full window, giving LTX's
   audio-video cross-attention maximum signal for mouth pixel
   correspondence.

Plus Phase 1 of the blend fix: `snap_boundaries=True` ensures every
iteration runs on one pure prompt (no mid-iteration conditioning mix),
and the byte-exact subject string across entries eliminates the
identity-drift jump at each boundary.

## Observations for future refinement

- **If lip sync is still not clean after v2**, the remaining lever is
  `overlap_seconds` (2.0 → 3.0). More overlap = more context carryover
  between iterations = smoother identity. Cost: one more second of
  audio per iteration is re-generated, so ~1.05s less new content per
  iteration.
- **If it IS clean**, that gives us evidence for Phase 1 being
  sufficient — Phase 2 (latent overlap cross-fading) stays parked.
- **The "tier 1b" mood bundle ("wide stage framing, warm stage wash")
  is explicitly dropped from this version** because it contradicts the
  tight image. That's a generic lesson: mood bundles are advisory;
  drop them when they fight the actual image composition. Candidate
  for a `docs/reference/standup_system_prompt.md` refinement.
- **Entry 2:41-2:58 is new** (v1 collapsed after snap; v2 restores it).
  We could consider whether the raw LLM should have generated for this
  dwell in the first place — might be a signal that R9 (pre-snap
  rule) + `workflow_context.stride_seconds` guidance should also tell
  the LLM *how many entries* to target given duration / stride.
